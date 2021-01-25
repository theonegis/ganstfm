import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchgan.losses import LeastSquaresDiscriminatorLoss, LeastSquaresGeneratorLoss

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
        self.last_g = self.train_dir / 'generator.pth'
        self.last_d = self.train_dir / 'discriminator.pth'

        self.logger = get_logger()
        self.logger.info('Model initialization')

        self.generator = SFFusion().to(self.device)
        self.discriminator = MSDiscriminator().to(self.device)
        self.pretrained = AutoEncoder().to(self.device)
        load_pretrained(self.pretrained, 'assets/autoencoder.pth')

        self.criterion = ReconstructionLoss(self.pretrained)
        self.g_loss = LeastSquaresGeneratorLoss()
        self.d_loss = LeastSquaresDiscriminatorLoss()

        device_ids = [i for i in range(option.ngpu)]
        if option.cuda and option.ngpu > 1:
            self.generator = nn.DataParallel(self.generator, device_ids)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=option.lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=option.lr)

        n_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for generator.')
        n_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for discriminator.')
        self.logger.info(str(self.generator))
        self.logger.info(str(self.discriminator))

    def train_on_epoch(self, n_epoch, data_loader):
        self.generator.train()
        self.discriminator.train()
        epg_loss = AverageMeter()
        epd_loss = AverageMeter()
        epg_error = AverageMeter()

        batches = len(data_loader)
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):
            t_start = timer()
            data = [im.to(self.device) for im in data]
            inputs, target = data[:-1], data[-1]
            ############################
            # (1) Update D network
            ###########################
            self.discriminator.zero_grad()
            self.generator.zero_grad()
            prediction = self.generator(inputs)
            d_loss = (self.d_loss(self.discriminator(torch.cat((target, inputs[0]), 1)),
                                  self.discriminator(torch.cat((prediction.detach(), inputs[0]), 1))))
            d_loss.backward()
            self.d_optimizer.step()
            epd_loss.update(d_loss.item())
            ############################
            # (2) Update G network
            ###########################
            g_loss = (self.criterion(prediction, target) + 1e-3 *
                      self.g_loss(self.discriminator(torch.cat((prediction, inputs[0]), 1))))
            g_loss.backward()
            self.g_optimizer.step()
            epg_loss.update(g_loss.item())
            mse = F.mse_loss(prediction.detach(), target).item()
            epg_error.update(mse)
            t_end = timer()
            self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                             f'G-Loss: {g_loss.item():.6f} - '
                             f'D-Loss: {d_loss.item():.6f} - '
                             f'MSE: {mse:.6f} - '
                             f'Time: {t_end - t_start}s')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        save_checkpoint(self.generator, self.g_optimizer, self.last_g)
        save_checkpoint(self.discriminator, self.d_optimizer, self.last_d)
        return epg_loss.avg, epd_loss.avg, epg_error.avg

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        self.generator.eval()
        self.discriminator.eval()
        epoch_error = AverageMeter()
        for data in data_loader:
            data = [im.to(self.device) for im in data]
            inputs, target = data[:-1], data[-1]
            prediction = self.generator(inputs)
            g_loss = F.mse_loss(prediction, target)
            epoch_error.update(g_loss.item())
        return epoch_error.avg

    def train(self, train_dir, val_dir, patch_stride, batch_size,
              num_workers=0, epochs=50, resume=True):
        last_epoch = -1
        least_error = float('inf')
        if resume and self.history.exists():
            df = pd.read_csv(self.history)
            last_epoch = int(df.iloc[-1]['epoch'])
            least_error = df['val_error'].min()
            load_checkpoint(self.last_g, self.generator, optimizer=self.g_optimizer)
            load_checkpoint(self.last_d, self.discriminator, optimizer=self.d_optimizer)
        start_epoch = last_epoch + 1

        # 加载数据
        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size, PATCH_SIZE, patch_stride, mode=Mode.TRAINING)
        val_set = PatchSet(val_dir, self.image_size, PATCH_SIZE, mode=Mode.VALIDATION)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

        self.logger.info('Training...')
        for epoch in range(start_epoch, epochs + start_epoch):
            self.logger.info(f"Learning rate for Generator: "
                             f"{self.g_optimizer.param_groups[0]['lr']}")
            self.logger.info(f"Learning rate for Discriminator: "
                             f"{self.d_optimizer.param_groups[0]['lr']}")
            train_g_loss, train_d_loss, train_g_error = self.train_on_epoch(epoch, train_loader)
            val_error = self.test_on_epoch(val_loader)
            csv_header = ['epoch', 'train_g_loss', 'train_d_loss', 'train_g_error', 'val_error']
            csv_values = [epoch, train_g_loss, train_d_loss, train_g_error, val_error]
            log_csv(self.history, csv_values, header=csv_header)

            if val_error < least_error:
                least_error = val_error
                shutil.copy(str(self.last_g), str(self.best))

    @torch.no_grad()
    def test(self, test_dir, patch_size, num_workers=0):
        self.generator.eval()
        load_checkpoint(self.best, model=self.generator)
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
            prediction = self.generator(inputs)
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
