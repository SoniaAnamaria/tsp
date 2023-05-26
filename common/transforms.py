import random

import torch
import torch.nn.functional as F


def crop(video, i, j, h, w):
    return video[..., i:(i + h), j:(j + w)]


def center_crop(video, output_size):
    in_h, in_w = video.shape[-2:]
    out_h, out_w = output_size
    i = round((in_h - out_h) / 2.)
    j = round((in_w - out_w) / 2.)
    return crop(video, i, j, out_h, out_w)


def horizontal_flip(video):
    return video.flip(dims=(-1,))


def resize(video, size):
    scale_factor = None
    if isinstance(size, int):
        scale_factor = float(size) / min(video.shape[-2:])
        size = None
    return F.interpolate(video, size=size, scale_factor=scale_factor, mode='bilinear', align_corners=False)


def to_float_tensor_in_zero_one(video):
    return video.permute(3, 0, 1, 2).to(torch.float32) / 255


def normalize(video, mean, std):
    shape = (-1,) + (1,) * (video.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (video - mean) / std


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(video, output_size):
        in_h, in_w = video.shape[-2:]
        out_h, out_w = output_size
        if in_h == out_h and in_w == out_w:
            return 0, 0, in_h, in_w
        i = random.randint(0, in_h - out_h)
        j = random.randint(0, in_w - out_w)
        return i, j, out_h, out_w

    def __call__(self, video):
        i, j, h, w = self.get_params(video, self.size)
        return crop(video, i, j, h, w)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        return center_crop(video, self.size)


class RandomHorizontalFlip(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, video):
        if random.random() < self.p:
            return horizontal_flip(video)
        return video


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        return resize(video, self.size)


class ToFloatTensorInZeroOne(object):
    def __call__(self, video):
        return to_float_tensor_in_zero_one(video)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        return normalize(video, self.mean, self.std)
