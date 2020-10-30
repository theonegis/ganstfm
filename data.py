from pathlib import Path
import numpy as np
import rasterio
import math
from enum import Enum, auto, unique

import torch
from torch.utils.data import Dataset

from utils import make_tuple


@unique
class Mode(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    PREDICTION = auto()


def get_pair_path(directory: Path, mode: Mode):
    paths: list = [None] * 3
    if mode is Mode.TRAINING:
        year, coarse, fine = directory.name.split('_')
        for f in directory.glob('*.tif'):
            paths[0 if year + coarse in f.name else 2] = f
        refs = [p for p in (directory.parents[1] / 'refs').glob('*.tif')]
        paths[1] = refs[np.random.randint(0, len(refs))]
    else:
        ref_label, pred_label = directory.name.split('-')
        ref_tokens, pred_tokens = ref_label.split('_'), pred_label.split('_')
        for f in directory.glob('*.tif'):
            order = {
                pred_tokens[0] + pred_tokens[1] in f.name: 0,
                ref_tokens[0] + ref_tokens[2] in f.name: 1,
                pred_tokens[0] + pred_tokens[2] in f.name: 2
            }
            if True in order.keys():
                paths[order[True]] = f
        if mode is Mode.PREDICTION:
            del paths[2]
    return paths


def load_image_pair(directory: Path, mode: Mode):
    paths = get_pair_path(directory, mode=mode)
    images = []
    for p in paths:
        with rasterio.open(str(p)) as ds:
            im = ds.read()
            images.append(im)
    return images


class PatchSet(Dataset):
    """
    每张图片分割成小块进行加载
    Pillow中的Image是列优先，而Numpy中的ndarray是行优先
    """

    def __init__(self, image_dir, image_size, patch_size, patch_stride=None, mode=Mode.TRAINING):
        super(PatchSet, self).__init__()
        patch_size = make_tuple(patch_size)
        patch_stride = make_tuple(patch_stride) if patch_stride else patch_size
        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.mode = mode

        self.image_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]
        self.num_im_pairs = len(self.image_dirs)
        # 计算出图像进行分块以后的patches的数目
        self.n_patch_x = math.ceil((image_size[0] - patch_size[0] + 1) / patch_stride[0])
        self.n_patch_y = math.ceil((image_size[1] - patch_size[1] + 1) / patch_stride[1])
        self.num_patch = self.num_im_pairs * self.n_patch_x * self.n_patch_y

    @staticmethod
    def transform(data):
        data[data < 0] = 0
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
        out = data.mul_(0.0001)
        return out

    def map_index(self, index):
        id_n = index // (self.n_patch_x * self.n_patch_y)
        residual = index % (self.n_patch_x * self.n_patch_y)
        id_x = self.patch_stride[0] * (residual % self.n_patch_x)
        id_y = self.patch_stride[1] * (residual // self.n_patch_x)
        return id_n, id_x, id_y

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        images = load_image_pair(self.image_dirs[id_n], mode=self.mode)
        patches = [None] * len(images)

        for i in range(len(patches)):
            im = images[i][:,
                 id_x: (id_x + self.patch_size[0]),
                 id_y: (id_y + self.patch_size[1])]
            patches[i] = self.transform(im)

        del images[:]
        del images
        return patches

    def __len__(self):
        return self.num_patch
