import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model import *
from data import PatchSet, Mode
from utils import *

import shutil
from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd


class Experiment(object):
    def __init__(self, option):
        self.device = torch.device('cuda' if option.cuda else 'cpu')
        self.image_size = option.image_size

        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)
        self.best = self.train_dir / 'best.pth'
        self.checkpoint = self.train_dir / 'model.pth'

        self.logger = get_logger()
        self.logger.info('Model initialization')

        self.model = SFFusion().to(self.device)
        self.pretrained = AutoEncoder().to(self.device)
        load_pretrained(self.pretrained, 'assets/autoencoder.pth')
        self.criterion = ReconstructionLoss(self.pretrained)

        device_ids = [i for i in range(option.ngpu)]
        if option.cuda and option.ngpu > 1:
            self.model = nn.DataParallel(self.model, device_ids)
        self.optimizer = optim.Adam(self.model.parameters(), lr=option.lr)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for model.')
        self.logger.info(str(self.model))

    def train_on_epoch(self, n_epoch, data_loader):
        self.model.train()
        epoch_loss = AverageMeter()
        epoch_error = AverageMeter()

        batches = len(data_loader)
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):
            t_start = timer()
            data = [im.to(self.device) for im in data]
            inputs, target = data[:-1], data[-1]
            self.model.zero_grad()
            prediction = self.model(inputs)
            loss = self.criterion(prediction, target)
            loss.backward()
            self.optimizer.step()
            epoch_loss.update(loss.item())
            mse = F.mse_loss(prediction.detach(), target).item()
            epoch_error.update(mse)
            t_end = timer()
            self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                             f'Loss: {loss.item():.6f} - '
                             f'MSE: {mse:.6f} - '
                             f'Time: {t_end - t_start}s')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        save_checkpoint(self.model, self.optimizer, self.checkpoint)
        return epoch_loss.avg, epoch_error.avg

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        self.model.eval()
        epoch_loss = AverageMeter()
        epoch_error = AverageMeter()
        for data in data_loader:
            data = [im.to(self.device) for im in data]
            inputs, target = data[:-1], data[-1]
            prediction = self.model(inputs)
            loss = self.criterion(prediction, target)
            epoch_loss.update(loss.item())
            error = F.mse_loss(prediction, target)
            epoch_error.update(error.item())
        return epoch_loss.avg, epoch_error.avg

    def train(self, train_dir, val_dir, patch_stride, batch_size,
              num_workers=0, epochs=50, resume=True):
        last_epoch = -1
        least_error = float('inf')
        if resume and self.history.exists():
            df = pd.read_csv(self.history)
            last_epoch = int(df.iloc[-1]['epoch'])
            least_error = df['val_error'].min()
            load_checkpoint(self.checkpoint, self.model, optimizer=self.optimizer)
        start_epoch = last_epoch + 1

        # 加载数据
        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size, PATCH_SIZE, patch_stride, mode=Mode.TRAINING)
        val_set = PatchSet(val_dir, self.image_size, PATCH_SIZE, mode=Mode.VALIDATION)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

        self.logger.info('Training...')
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=20)
        for epoch in range(start_epoch, epochs + start_epoch):
            self.logger.info(f"Learning rate for the model: "
                             f"{self.optimizer.param_groups[0]['lr']}")
            train_loss, train_error = self.train_on_epoch(epoch, train_loader)
            val_loss, val_error = self.test_on_epoch(val_loader)
            csv_header = ['epoch', 'train_loss', 'train_error', 'val_loss', 'val_error']
            csv_values = [epoch, train_loss, train_error, val_loss, val_error]
            log_csv(self.history, csv_values, header=csv_header)
            scheduler.step(val_loss)

            if val_error < least_error:
                least_error = val_error
                shutil.copy(str(self.checkpoint), str(self.best))

    @torch.no_grad()
    def test(self, test_dir, patch_size, num_workers=0):
        self.model.eval()
        load_checkpoint(self.best, model=self.model)
        self.logger.info('Testing...')
        assert self.image_size[0] % patch_size[0] == 0
        assert self.image_size[1] % patch_size[1] == 0
        rows = int(self.image_size[1] / patch_size[1])
        cols = int(self.image_size[0] / patch_size[0])
        n_blocks = rows * cols
        image_dirs = iter([p for p in test_dir.iterdir() if p.is_dir()])
        test_set = PatchSet(test_dir, self.image_size, patch_size, mode=Mode.PREDICTION)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)

        pixel_scale = 10000
        patches = []
        t_start = timer()
        for inputs in test_loader:
            inputs = [im.to(self.device) for im in inputs]
            prediction = self.model(inputs)
            prediction = prediction.squeeze().cpu().numpy()
            patches.append(prediction * pixel_scale)

            # 完成一张影像以后进行拼接
            if len(patches) == n_blocks:
                result = np.empty((NUM_BANDS, *self.image_size), dtype=np.float32)
                block_count = 0
                for i in range(rows):
                    row_start = i * patch_size[1]
                    for j in range(cols):
                        col_start = j * patch_size[0]
                        result[:,
                        col_start: col_start + patch_size[0],
                        row_start: row_start + patch_size[1]
                        ] = patches[block_count]
                        block_count += 1
                patches.clear()
                # 存储预测影像结果
                result = result.astype(np.int16)
                metadata = {
                    'driver': 'GTiff',
                    'width': self.image_size[1],
                    'height': self.image_size[0],
                    'count': NUM_BANDS,
                    'dtype': np.int16
                }
                name = f'PRED_{next(image_dirs).stem}.tif'
                save_array_as_tif(result, self.test_dir / name, metadata)
                t_end = timer()
                self.logger.info(f'Time cost: {t_end - t_start}s on {name}')
                t_start = timer()
