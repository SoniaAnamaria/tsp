import torch
import torch.nn as nn
from torchvision.models.video.resnet import VideoResNet, R2Plus1dStem, BasicBlock

from models.i3d import I3D
from models.x3d import X3D, modify_model

__all__ = ['r2plus1d_34', 'i3d', 'x3d']

R2PLUS1D_34_MODEL_URL = "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth"


def r2plus1d_34(pretrained=True, progress=False, **kwargs):
    model = VideoResNet(
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem,
        **kwargs,
    )

    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(R2PLUS1D_34_MODEL_URL, progress=progress)
        model.load_state_dict(state_dict)

    return model


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, stride=1, padding=1):
        mid_planes = (in_planes * out_planes * 3 * 3 * 3) // (
                in_planes * 3 * 3 + 3 * out_planes
        )
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                mid_planes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                mid_planes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )


def i3d(pretrained=True, progress=False, **kwargs):
    model = I3D(in_channels=3)
    if pretrained:
        model.load_state_dict(torch.load('/home/ubuntu/PycharmProjects/tsp/models/rgb_imagenet.pt'))

    return model


def x3d(pretrained=True, progress=False, **kwargs):
    model = X3D()
    if pretrained:
        state_dict = torch.load("../x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth")
        modify_model(state_dict)
        torch.save(state_dict, "../x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth")
        model.load_state_dict(torch.load("../x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth"))

    return model
