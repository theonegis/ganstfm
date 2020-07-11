import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from experiment import Experiment

import os
import faulthandler

# os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
faulthandler.enable()

"""
jsub -q qgpu -J yastfn -m gpu01 -e error.txt -o output.txt python run.py --lr 2e-4 --num_workers 28 --batch_size 28 --epochs 500 --cuda --ngpu 4 --image_size 2040 1720 --save_dir out --data_dir data

jsub -q qgpu -J yastfn -m gpu01 -e error.txt -o output.txt python run.py --lr 2e-4 --num_workers 28 --batch_size 28 --epochs 500 --cuda --ngpu 4 --image_size 2720 3200 --predict_strid 1360 1600 --save_dir out --data_dir data
"""

# 获取模型运行时必须的一些参数
parser = argparse.ArgumentParser(description='Acquire some parameters for fusion restore')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='the initial learning rate')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--num_workers', type=int, default=0, help='number of threads to load data')
parser.add_argument('--save_dir', type=Path, default=Path('.'),
                    help='the output directory')

# 获取对输入数据进行预处理时的一些参数
parser.add_argument('--data_dir', type=Path, required=True,
                    help='the training data directory')
parser.add_argument('--image_size', type=int, nargs='+', required=True,
                    help='the image size (height, width)')
parser.add_argument('--patch_stride', type=int, nargs='+', default=256,
                    help='the patch stride')
parser.add_argument('--predict_stride', type=int, nargs='+', default=None,
                    help='the patch stride for prediction')
opt = parser.parse_args()

torch.manual_seed(2020)
if not torch.cuda.is_available():
    opt.cuda = False
if opt.cuda:
    torch.cuda.manual_seed_all(2020)
    cudnn.benchmark = True
    cudnn.deterministic = True

opt.predict_stride = opt.image_size if opt.predict_stride is None else opt.predict_stride

if __name__ == '__main__':
    experiment = Experiment(opt)
    train_dir = opt.data_dir / 'train'
    val_dir = opt.data_dir / 'val'
    test_dir = val_dir
    if opt.epochs > 0:
        if opt.epochs > 0:
            experiment.train(train_dir, val_dir,
                             opt.patch_stride, opt.batch_size,
                             num_workers=opt.num_workers, epochs=opt.epochs)
    experiment.test(test_dir, opt.predict_stride, num_workers=opt.num_workers)
