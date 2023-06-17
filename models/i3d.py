import torch
import torch.nn.functional as F
from torch import nn


class MaxPool3dSamePadding(nn.MaxPool3d):
    def forward(self, x):
        x = add_padding_all_dimensions(x, self.kernel_size, self.stride)
        return super(MaxPool3dSamePadding, self).forward(x)


def compute_padding_one_dimension(kernel, stride, idx, dimension):
    modulus = dimension % stride[idx]
    if modulus == 0:
        return max(kernel[idx] - stride[idx], 0)
    else:
        return max(kernel[idx] - modulus, 0)


def add_padding_all_dimensions(x, kernel, stride):
    (batch, channel, t, h, w) = x.size()
    pad_t = compute_padding_one_dimension(kernel, stride, 0, t)
    pad_h = compute_padding_one_dimension(kernel, stride, 1, h)
    pad_w = compute_padding_one_dimension(kernel, stride, 2, w)
    pad_t_front = pad_t // 2
    pad_t_back = pad_t - pad_t_front
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    pad = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_t_front, pad_t_back)
    return F.pad(x, pad)


class Unit3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(1, 1, 1), stride=(1, 1, 1), activation_fn=F.relu,
                 use_batch_norm=True, use_bias=False):
        super(Unit3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        self.conv3d = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels,
                                kernel_size=self.kernel, stride=self.stride, padding=0, bias=self.use_bias)
        if self.use_batch_norm:
            self.bn = nn.BatchNorm3d(self.out_channels, eps=0.001, momentum=0.9)

    def forward(self, x):
        x = add_padding_all_dimensions(x, self.kernel, self.stride)
        x = self.conv3d(x)
        if self.use_batch_norm:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, out_channels=out_channels[0], kernel=[1, 1, 1])
        self.b1a = Unit3D(in_channels=in_channels, out_channels=out_channels[1], kernel=[1, 1, 1])
        self.b1b = Unit3D(in_channels=out_channels[1], out_channels=out_channels[2], kernel=[3, 3, 3])
        self.b2a = Unit3D(in_channels=in_channels, out_channels=out_channels[3], kernel=[1, 1, 1])
        self.b2b = Unit3D(in_channels=out_channels[3], out_channels=out_channels[4], kernel=[3, 3, 3])
        self.b3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.b3b = Unit3D(in_channels=in_channels, out_channels=out_channels[5], kernel=[1, 1, 1])

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
        self.Conv3d_1a_7x7 = Unit3D(in_channels=in_channels, out_channels=64, kernel=[7, 7, 7], stride=(2, 2, 2))
        self.MaxPool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.Conv3d_2b_1x1 = Unit3D(in_channels=64, out_channels=64, kernel=[1, 1, 1])
        self.Conv3d_2c_3x3 = Unit3D(in_channels=64, out_channels=192, kernel=[3, 3, 3])
        self.MaxPool3d_3a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64])
        self.MaxPool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.Mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionModule(528, [256, 160, 320, 32, 128, 128])
        self.MaxPool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.Mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128])
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=1024, out_channels=self.num_classes, kernel=[1, 1, 1], activation_fn=None,
                             use_batch_norm=False, use_bias=True)

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
        out = self.avg_pool(out)
        if features:
            out = out.squeeze(3)
            out = out.squeeze(3)
            out = out.mean(2)
            return out
        else:
            out = self.dropout(out)
            out = self.logits(out)
            out = out.squeeze(3)
            out = out.squeeze(3)
            out = out.mean(2)
            return out
