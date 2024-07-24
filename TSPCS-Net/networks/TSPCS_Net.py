import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import Constants
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
nonlinearity = partial(F.relu, inplace=True)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,
            dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=True,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = norm_layer(planes)
        if downsample == True:  # !!!
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif isinstance(downsample, nn.Module):
            self.downsample = downsample
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock_2(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out

class Leftsidedownsample4(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Leftsidedownsample4, self).__init__()
        self.lsconv1 = nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False)
        self.lsnorm1 = nn.BatchNorm2d(outchannels)
        self.relu = nonlinearity
        self.lsconv2 = nn.Conv2d(outchannels, outchannels, 3, 2, 1,
                                 bias=False)
        self.lsnorm2 = nn.BatchNorm2d(outchannels)

    def forward(self, x):
        x = self.lsconv1(x)
        x = self.lsnorm1(x)
        x = self.relu(x)
        x = self.lsconv2(x)
        x = self.lsnorm2(x)
        return x

class Leftsidedownsample2(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Leftsidedownsample2, self).__init__()
        self.lsconv1 = nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False)
        self.lsnorm1 = nn.BatchNorm2d(outchannels)

    def forward(self, x):
        x = self.lsconv1(x)
        x = self.lsnorm1(x)
        return x

class LsChannelMeanMaxAttention(nn.Module):
    def __init__(self, num_channels):
        super(LsChannelMeanMaxAttention, self).__init__()
        num_channels_reduced = num_channels // 2
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nonlinearity

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        squeeze_tensor_mean = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        fc_out_1_mean = self.relu(self.fc1(squeeze_tensor_mean))
        fc_out_2_mean = self.fc2(fc_out_1_mean)

        squeeze_tensor_max = input_tensor.view(batch_size, num_channels, -1).max(dim=2)[0]
        fc_out_1_max = self.relu(self.fc1(squeeze_tensor_max))
        fc_out_2_max = self.fc2(fc_out_1_max)

        a, b = squeeze_tensor_mean.size()
        result = torch.Tensor(a, b)
        result = torch.add(fc_out_2_mean, fc_out_2_max)
        fc_out_2 = F.sigmoid(result)
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class LsSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(LsSpatialAttention, self).__init__()
        padding = 3
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_tensor = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * input_tensor

class ResNet(nn.Module):

    def __init__(
            self,
            initial_channel: int,
            block: Type[Union[BasicBlock, BasicBlock_2]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(initial_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.leftside1 = Leftsidedownsample4(64, 64)
        self.leftside2 = Leftsidedownsample2(64, 128)
        self.leftside3 = Leftsidedownsample2(128, 256)

        self.Lschannelmeanmaxattention1 = LsChannelMeanMaxAttention(64)
        self.Lsspatialattention1 = LsSpatialAttention()
        self.Lschannelmeanmaxattention2 = LsChannelMeanMaxAttention(128)
        self.Lsspatialattention2 = LsSpatialAttention()
        self.Lschannelmeanmaxattention3 = LsChannelMeanMaxAttention(256)
        self.Lsspatialattention3 = LsSpatialAttention()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=False, dilation=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=False, dilation=4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=False, dilation=8)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, BasicBlock_2]], planes: int, blocks: int, stride: int = 1,
                    dilate: bool = False, dilation: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, groups=self.groups, base_width=self.base_width,
                                dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        ls1 = self.leftside1(x0)
        ls2 = self.leftside2(ls1)
        ls3 = self.leftside3(ls2)

        ls1a = self.Lschannelmeanmaxattention1(ls1)
        ls1a = self.Lsspatialattention1(ls1a)
        ls2a = self.Lschannelmeanmaxattention2(ls2)
        ls2a = self.Lsspatialattention2(ls2a)
        ls3a = self.Lschannelmeanmaxattention3(ls3)
        ls3a = self.Lsspatialattention3(ls3a)

        x1 = self.maxpool(x0)
        e1 = self.layer1(x1)

        e1m = self.maxpool1(e1)
        e1a = torch.add(e1m, ls1a)
        e1a = self.relu(e1a)
        e2 = self.layer2(e1a)

        e2m = self.maxpool1(e2)
        e2a = torch.add(e2m, ls2a)
        e2a = self.relu(e2a)
        e3 = self.layer3(e2a)

        e3m = self.maxpool1(e3)
        e3a = torch.add(e3m, ls3a)
        e3a = self.relu(e3a)
        e4 = self.layer4(e3a)

        return e1, e2, e3, e4

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 80, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 96, kernel_size=1)
        self.branch3x3_2a = conv_block(96, 96, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(96, 96, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 112, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(112, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(96, 96, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(96, 96, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 48, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class ChannelMeanAttention(nn.Module):
    def __init__(self, num_channels):
        super(ChannelMeanAttention, self).__init__()
        num_channels_reduced = num_channels // 2
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nonlinearity

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = F.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class ChannelMeanMaxAttention(nn.Module):
    def __init__(self, num_channels):
        super(ChannelMeanMaxAttention, self).__init__()
        num_channels_reduced = num_channels // 2
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels_reduced, bias=True)
        self.relu = nonlinearity

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()  # b*2c*w*h
        squeeze_tensor_mean = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        fc_out_1_mean = self.relu(self.fc1(squeeze_tensor_mean))
        fc_out_2_mean = self.fc2(fc_out_1_mean)

        squeeze_tensor_max = input_tensor.view(batch_size, num_channels, -1).max(dim=2)[0]
        fc_out_1_max = self.relu(self.fc1(squeeze_tensor_max))
        fc_out_2_max = self.fc2(fc_out_1_max)

        a, b = fc_out_2_mean.size()
        result = torch.Tensor(a, b)
        result = torch.add(fc_out_2_mean, fc_out_2_max)
        fc_out_2 = F.sigmoid(result)
        fc_out_final = fc_out_2.view(a, b, 1, 1)
        # output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return fc_out_final

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_tensor = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * input_tensor

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="bilinear"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True,
                          recompute_scale_factor=True)
        return x

class conv_11(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(conv_11, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class MultiScaleSupervisionBlock(nn.Module):
    def __init__(self, input_channels, upsample_factor):
        super(MultiScaleSupervisionBlock, self).__init__()
        self.conv3 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv_11(64, 1)
        self.upsample = Upsample(upsample_factor)

    def forward(self, x):
        x = self.conv3(x)
        x = self.conv4(self.upsample(x))
        return x

class TSPCS_Net_(nn.Module):
    def __init__(self, num_classes=Constants.BINARY_CLASS, num_channels=3):
        super(TSPCS_Net_, self).__init__()

        filters = [64, 128, 256, 512]
        input_channels = 3
        self.encoder = ResNet(input_channels, BasicBlock, [3, 3, 3, 3])
        self.dblockinception = InceptionE(512)
        self.decoder4 = DecoderBlock(512, filters[2])
        self.channelmeanmaxattention1 = ChannelMeanMaxAttention(filters[2] * 2)
        self.spatialattention1 = SpatialAttention()

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.channelmeanmaxattention2 = ChannelMeanMaxAttention(filters[1] * 2)
        self.spatialattention2 = SpatialAttention()

        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.channelmeanmaxattention3 = ChannelMeanMaxAttention(filters[0] * 2)
        self.spatialattention3 = SpatialAttention()

        self.multiscalesupervision1 = MultiScaleSupervisionBlock(256, 16)
        self.multiscalesupervision2 = MultiScaleSupervisionBlock(128, 8)
        self.multiscalesupervision3 = MultiScaleSupervisionBlock(64, 4)

        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        e4 = self.dblockinception(e4)
        d4_high = self.decoder4(e4)
        d4_before = torch.cat([d4_high, e3], 1)
        d4 = self.channelmeanmaxattention1(d4_before)
        d4 = torch.mul(e3, d4)
        d4 = torch.add(d4, d4_high)
        d4 = self.spatialattention1(d4)
        x_out4 = self.multiscalesupervision1(d4)

        d3_high = self.decoder3(d4)
        d3_before = torch.cat([d3_high, e2], 1)
        d3 = self.channelmeanmaxattention2(d3_before)
        d3 = torch.mul(e2, d3)
        d3 = torch.add(d3, d3_high)
        d3 = self.spatialattention2(d3)
        x_out3 = self.multiscalesupervision2(d3)

        d2_high = self.decoder2(d3)
        d2_before = torch.cat([d2_high, e1], 1)
        d2 = self.channelmeanmaxattention3(d2_before)
        d2 = torch.mul(e1, d2)
        d2 = torch.add(d2, d2_high)
        d2 = self.spatialattention3(d2)
        x_out2 = self.multiscalesupervision3(d2)

        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out), F.sigmoid(x_out2), F.sigmoid(x_out3), F.sigmoid(x_out4)



