import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18 as _r2plus1d_18
from torchvision.models.video import r3d_18 as _r3d_18
from torchvision.models.video.resnet import VideoResNet, R2Plus1dStem, BasicBlock

__all__ = ['r2plus1d_34', 'r2plus1d_18', 'r3d_18']

R2PLUS1D_34_MODEL_URL = "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth"


def r2plus1d_34(pretrained=True, progress=False, **kwargs):
    model = VideoResNet(
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem,
        **kwargs,
    )

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            R2PLUS1D_34_MODEL_URL, progress=progress
        )
        model.load_state_dict(state_dict)

    return model


def r2plus1d_18(pretrained=True, progress=False, **kwargs):
    return _r2plus1d_18(pretrained=pretrained, progress=progress, **kwargs)


def r3d_18(pretrained=True, progress=False, **kwargs):
    return _r3d_18(pretrained=pretrained, progress=progress, **kwargs)


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (
                in_planes * 3 * 3 + 3 * out_planes
        )
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


def i3d(pretrained=True, progress=False, **kwargs):
    model = I3D(in_channels=3)
    if pretrained:
        model.load_state_dict(torch.load('rgb_imagenet.pt'))

    return model


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        np.ceil(float(t) / float(self.stride[0]))
        np.ceil(float(h) / float(self.stride[1]))
        np.ceil(float(w) / float(self.stride[2]))

        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)


        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False):
        super(Unit3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.use_batch_norm = use_batch_norm
        self.activation_fn = activation_fn
        self.use_bias = use_bias
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=self.kernel_shape,
                                stride=self.stride,
                                padding=0,
                                bias=self.use_bias)

        if self.use_batch_norm:
            self.bn = nn.BatchNorm3d(self.out_channels, eps=0.001, momentum=0.9)

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_shape[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_shape[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()

        np.ceil(float(t) / float(self.stride[0]))
        np.ceil(float(h) / float(self.stride[1]))
        np.ceil(float(w) / float(self.stride[2]))

        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        x = self.conv3d(x)
        if self.use_batch_norm:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, out_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0)
        self.b1a = Unit3D(in_channels=in_channels, out_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0)
        self.b1b = Unit3D(in_channels=out_channels[1], out_channels=out_channels[2], kernel_shape=[3, 3, 3])
        self.b2a = Unit3D(in_channels=in_channels, out_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0)
        self.b2b = Unit3D(in_channels=out_channels[3], out_channels=out_channels[4], kernel_shape=[3, 3, 3])
        self.b3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, out_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0)

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class I3D(torch.nn.Module):
    def __init__(self, num_classes=400, in_channels=3, dropout_keep_prob=0.5):
        super(I3D, self).__init__()
        self.num_classes = num_classes

        self.Conv3d_1a_7x7 = Unit3D(in_channels=in_channels, out_channels=64, kernel_shape=[7, 7, 7],
                                    stride=(2, 2, 2), padding=(3, 3, 3))
        self.MaxPool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        self.Conv3d_2b_1x1 = Unit3D(in_channels=64, out_channels=64, kernel_shape=[1, 1, 1], padding=0)
        self.Conv3d_2c_3x3 = Unit3D(in_channels=64, out_channels=192, kernel_shape=[3, 3, 3], padding=1)
        self.MaxPool3d_3a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64])
        self.MaxPool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0)
        self.Mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionModule(528, [256, 160, 320, 32, 128, 128])
        self.MaxPool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.Mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128])
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=1024, out_channels=self.num_classes, kernel_shape=[1, 1, 1], padding=0,
                             activation_fn=None, use_batch_norm=False, use_bias=True)

    def forward(self, x, features=True):
        out = self.Conv3d_1a_7x7(x)
        out = self.MaxPool3d_2a_3x3(out)
        out = self.Conv3d_2b_1x1(out)
        out = self.Conv3d_2c_3x3(out)
        out = self.MaxPool3d_3a_3x3(out)
        out = self.Mixed_3b(out)
        out = self.Mixed_3c(out)
        out = self.MaxPool3d_4a_3x3(out)
        out = self.Mixed_4b(out)
        out = self.Mixed_4c(out)
        out = self.Mixed_4d(out)
        out = self.Mixed_4e(out)
        out = self.Mixed_4f(out)
        out = self.MaxPool3d_5a_2x2(out)
        out = self.Mixed_5b(out)
        out = self.Mixed_5c(out)
        out = self.avg_pool(out)  # <- [1, 1024, 8 (for T=64) or 3 (for T=24), 1, 1]

        if features:
            out = out.squeeze(3)  # <- (B, 1024, 8 (for T=64) or 3 (for T=24), 1)
            out = out.squeeze(3)  # <- (B, 1024, 8 (for T=64) or 3 (for T=24))
            out = out.mean(2)  # <- (B, 1024)
            return out  # (B, 1024)
        else:
            out = self.dropout(out)
            out = self.logits(out)
            out = out.squeeze(3)
            out = out.squeeze(3)
            out = out.mean(2)
            return out
