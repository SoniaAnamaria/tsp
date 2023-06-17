import torch.nn as nn


class SqueezeExcitation(nn.Module):

    def __init__(self, channels, reduction=0.0625):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channels_reduced = _round_width(channels, reduction)
        self.conv1 = nn.Conv3d(channels, self.channels_reduced, kernel_size=1)
        self.conv2 = nn.Conv3d(self.channels_reduced, channels, kernel_size=1)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

    def forward(self, x):
        se_input = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        return se_input * x


def _round_width(channels_in, multiplier, min_width=8):
    channels_in *= multiplier
    channels_out = max(min_width, int(channels_in + min_width / 2) // min_width * min_width)
    if channels_out < 0.9 * channels_in:
        channels_out += min_width
    return int(channels_out)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class X3DStem(nn.Module):
    def __init__(self, in_channels, out_channels, norm_eps=1e-3, norm_momentum=0.9):
        super().__init__()
        self.conv1_s = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3, 3),
                                 stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.conv1_t = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(5, 1, 1),
                                 stride=(1, 1, 1), padding=(2, 0, 0), bias=False, groups=out_channels)
        self.norm = nn.BatchNorm3d(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.conv1_t(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class X3DBlock(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, stride=1, norm_eps=1e-3, norm_momentum=0.9,
                 reduction=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.reduction = reduction
        if self.in_channels != self.out_channels or self.stride != 1:
            self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False)
            self.down_norm = nn.BatchNorm3d(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
        self.conv_1 = nn.Conv3d(in_channels=in_channels, out_channels=inter_channels, kernel_size=(1, 1, 1), bias=False)
        self.norm_1 = nn.BatchNorm3d(num_features=inter_channels, eps=norm_eps, momentum=norm_momentum)
        self.activation_1 = nn.ReLU()
        self.conv_2 = nn.Conv3d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=(3, 3, 3),
                                stride=(1, stride, stride), padding=(1, 1, 1), bias=False, groups=inter_channels)
        self.norm_2 = nn.BatchNorm3d(num_features=inter_channels, eps=norm_eps, momentum=norm_momentum)
        if self.reduction != 0.0:
            self.se = SqueezeExcitation(inter_channels, reduction)
        self.activation_2 = Swish()
        self.conv_3 = nn.Conv3d(in_channels=inter_channels, out_channels=out_channels, kernel_size=(1, 1, 1),
                                bias=False)
        self.norm_3 = nn.BatchNorm3d(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
        self.activation_3 = nn.ReLU()

    def forward(self, x):
        block_input = x
        if self.in_channels != self.out_channels or self.stride != 1:
            block_input = self.down_conv(x)
            block_input = self.down_norm(block_input)
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        if self.reduction != 0.0:
            x = self.se(x)
        x = self.activation_2(x)
        x = self.conv_3(x)
        x = self.norm_3(x)
        x = x + block_input
        x = self.activation_3(x)
        return x


class X3DStage(nn.Module):
    def __init__(self, nr_blocks, in_channels, inter_channels, out_channels, reduction=0.0625):
        super().__init__()
        blocks = []
        for idx in range(nr_blocks):
            block = X3DBlock(in_channels=in_channels if idx == 0 else out_channels,
                             inter_channels=inter_channels,
                             out_channels=out_channels,
                             stride=2 if idx == 0 else 1,
                             reduction=reduction if (idx + 1) % 2 else 0.0)
            blocks.append(block)
        self.x3dStage = nn.ModuleList(blocks)

    def forward(self, x):
        for _, block in enumerate(self.x3dStage):
            x = block(x)
        return x


class X3DHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_eps=1e-3, norm_momentum=0.9):
        super().__init__()
        self.conv5 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1), bias=False)
        self.norm = nn.BatchNorm3d(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv5(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return x


class X3D(nn.Module):
    def __init__(self, in_channels=3, in_features=24, gamma_b=2.25):
        super().__init__()
        self.stage_blocks = [3, 5, 11, 7]
        blocks = [X3DStem(in_channels, in_features)]
        for i, nr_blocks in enumerate(self.stage_blocks):
            out_channels = in_features if i == 0 else in_features * 2
            inter_channels = int(out_channels * gamma_b)
            blocks.append(X3DStage(nr_blocks, in_features, inter_channels, out_channels))
            in_features = out_channels
        blocks.append(X3DHead(in_features, int(in_features * gamma_b)))
        self.x3dBlocks = nn.ModuleList(blocks)

    def forward(self, x):
        for _, block in enumerate(self.x3dBlocks):
            x = block(x)
        return x


def modify_key(model, new_key, old_key):
    model[new_key] = model[old_key]
    model.pop(old_key)


def modify_conv(model, new_conv, old_conv):
    modify_key(model, new_conv + ".weight", old_conv + ".weight")


def modify_bn(model, new_bn, old_bn):
    modify_key(model, new_bn + ".weight", old_bn + ".weight")
    modify_key(model, new_bn + ".bias", old_bn + ".bias")
    modify_key(model, new_bn + ".running_mean", old_bn + ".running_mean")
    modify_key(model, new_bn + ".running_var", old_bn + ".running_var")
    model.pop(old_bn + ".num_batches_tracked")


def modify_block(model, new_block, old_block, with_se=False):
    modify_conv(model, new_block + ".conv_1", old_block + ".conv1.conv")
    modify_bn(model, new_block + ".norm_1", old_block + ".conv1.bn")
    modify_conv(model, new_block + ".conv_2", old_block + ".conv2.conv")
    modify_bn(model, new_block + ".norm_2", old_block + ".conv2.bn")
    if with_se:
        modify_conv(model, new_block + ".se.conv1", old_block + ".se_module.fc1")
        modify_key(model, new_block + ".se.conv1.bias", old_block + ".se_module.fc1.bias")
        modify_conv(model, new_block + ".se.conv2", old_block + ".se_module.fc2")
        modify_key(model, new_block + ".se.conv2.bias", old_block + ".se_module.fc2.bias")
    modify_conv(model, new_block + ".conv_3", old_block + ".conv3.conv")
    modify_bn(model, new_block + ".norm_3", old_block + ".conv3.bn")


def modify_model(model):
    modify_conv(model, "x3dBlocks.0.conv1_s", "backbone.conv1_s.conv")
    modify_conv(model, "x3dBlocks.0.conv1_t", "backbone.conv1_t.conv")
    modify_bn(model, "x3dBlocks.0.norm", "backbone.conv1_t.bn")

    modify_conv(model, "x3dBlocks.1.x3dStage.0.down_conv", "backbone.layer1.0.downsample.conv")
    modify_bn(model, "x3dBlocks.1.x3dStage.0.down_norm", "backbone.layer1.0.downsample.bn")
    modify_block(model, "x3dBlocks.1.x3dStage.0", "backbone.layer1.0", with_se=True)
    modify_block(model, "x3dBlocks.1.x3dStage.1", "backbone.layer1.1")
    modify_block(model, "x3dBlocks.1.x3dStage.2", "backbone.layer1.2", with_se=True)

    modify_conv(model, "x3dBlocks.2.x3dStage.0.down_conv", "backbone.layer2.0.downsample.conv")
    modify_bn(model, "x3dBlocks.2.x3dStage.0.down_norm", "backbone.layer2.0.downsample.bn")
    modify_block(model, "x3dBlocks.2.x3dStage.0", "backbone.layer2.0", with_se=True)
    modify_block(model, "x3dBlocks.2.x3dStage.1", "backbone.layer2.1")
    modify_block(model, "x3dBlocks.2.x3dStage.2", "backbone.layer2.2", with_se=True)
    modify_block(model, "x3dBlocks.2.x3dStage.3", "backbone.layer2.3")
    modify_block(model, "x3dBlocks.2.x3dStage.4", "backbone.layer2.4", with_se=True)

    modify_conv(model, "x3dBlocks.3.x3dStage.0.down_conv", "backbone.layer3.0.downsample.conv")
    modify_bn(model, "x3dBlocks.3.x3dStage.0.down_norm", "backbone.layer3.0.downsample.bn")
    modify_block(model, "x3dBlocks.3.x3dStage.0", "backbone.layer3.0", with_se=True)
    modify_block(model, "x3dBlocks.3.x3dStage.1", "backbone.layer3.1")
    modify_block(model, "x3dBlocks.3.x3dStage.2", "backbone.layer3.2", with_se=True)
    modify_block(model, "x3dBlocks.3.x3dStage.3", "backbone.layer3.3")
    modify_block(model, "x3dBlocks.3.x3dStage.4", "backbone.layer3.4", with_se=True)
    modify_block(model, "x3dBlocks.3.x3dStage.5", "backbone.layer3.5")
    modify_block(model, "x3dBlocks.3.x3dStage.6", "backbone.layer3.6", with_se=True)
    modify_block(model, "x3dBlocks.3.x3dStage.7", "backbone.layer3.7")
    modify_block(model, "x3dBlocks.3.x3dStage.8", "backbone.layer3.8", with_se=True)
    modify_block(model, "x3dBlocks.3.x3dStage.9", "backbone.layer3.9")
    modify_block(model, "x3dBlocks.3.x3dStage.10", "backbone.layer3.10", with_se=True)

    modify_conv(model, "x3dBlocks.4.x3dStage.0.down_conv", "backbone.layer4.0.downsample.conv")
    modify_bn(model, "x3dBlocks.4.x3dStage.0.down_norm", "backbone.layer4.0.downsample.bn")
    modify_block(model, "x3dBlocks.4.x3dStage.0", "backbone.layer4.0", with_se=True)
    modify_block(model, "x3dBlocks.4.x3dStage.1", "backbone.layer4.1")
    modify_block(model, "x3dBlocks.4.x3dStage.2", "backbone.layer4.2", with_se=True)
    modify_block(model, "x3dBlocks.4.x3dStage.3", "backbone.layer4.3")
    modify_block(model, "x3dBlocks.4.x3dStage.4", "backbone.layer4.4", with_se=True)
    modify_block(model, "x3dBlocks.4.x3dStage.5", "backbone.layer4.5")
    modify_block(model, "x3dBlocks.4.x3dStage.6", "backbone.layer4.6", with_se=True)

    modify_conv(model, "x3dBlocks.5.conv5", "backbone.conv5.conv")
    modify_bn(model, "x3dBlocks.5.norm", "backbone.conv5.bn")

    model.pop("cls_head.fc1.weight")
    model.pop("cls_head.fc2.weight")
    model.pop("cls_head.fc2.bias")
