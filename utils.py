import sys
import logging
import csv
from pathlib import Path
import torch
import torch.nn as nn
from osgeo import gdal


def make_tuple(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) and len(x) == 1:
        return x[0], x[0]
    return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_array_as_tif(array, fname, driver='GTiff', prototype=None):
    nbands = array.shape[0] if array.ndim == 3 else 1
    driver: gdal.Driver = gdal.GetDriverByName(driver)
    prototype = gdal.Open(prototype) if isinstance(prototype, str) else None
    ds: gdal.Dataset = (driver.CreateCopy(str(fname), prototype) if prototype else
                        driver.Create(str(fname), array.shape[1], array.shape[0], nbands, gdal.GDT_Int16))
    if array.ndim == 2:
        ds.GetRasterBand(1).WriteArray(array)
    else:
        for i in range(nbands):
            ds.GetRasterBand(i + 1).WriteArray(array[i])
    ds.FlushCache()
    del ds
    if prototype is not None:
        del prototype


def get_logger(logpath=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if logpath is not None:
            file_handler = logging.FileHandler(logpath)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return logger


def save_checkpoint(model, optimizer, path):
    if path.exists():
        path.unlink()
    model = model.module if isinstance(model, nn.DataParallel) else model
    state = {'state_dict': model.state_dict()}
    if optimizer:
        state = {'state_dict': model.state_dict(),
                 'optim_dict': optimizer.state_dict()}
    if isinstance(path, Path):
        torch.save(state, str(path.resolve()))
    else:
        torch.save(state, str(path.resolve()))


def load_checkpoint(checkpoint, model, optimizer=None, map_location=None):
    if not checkpoint.exists():
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    state = torch.load(checkpoint, map_location=map_location)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state['state_dict'])

    if optimizer:
        optimizer.load_state_dict(state['optim_dict'])
    return state


def log_csv(filepath, values, header=None, multirows=False):
    empty = False
    if not filepath.exists():
        filepath.touch()
        empty = True

    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        if empty and header:
            writer.writerow(header)
        if multirows:
            writer.writerows(values)
        else:
            writer.writerow(values)


def load_pretrained(model, pretrained, requires_grad=False):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained)['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False


def set_no_grad(model):
    for param in model.parameters():
        param.requires_grad = False
