import os, sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _single, _pair, _triple
import torch.nn.functional as F


# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, affine=True),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, affine=True),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, affine=True),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, affine=True),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DilateConv(nn.Module):
    """
    d_rate: dilation rate
    H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)
    set kernel size to 3, stride to 1, padding==d_rate ==> spatial size kept
    """

    def __init__(self, d_rate, in_ch, out_ch):
        super(DilateConv, self).__init__()
        self.d_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                stride=1, padding=d_rate, dilation=d_rate)

    def forward(self, x):
        return self.d_conv(x)


class RCF_ASPP(nn.Module):
    def __init__(self):
        super(RCF_ASPP, self).__init__()
        # lr 1 2 decay 1 0
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # lr 100 200 decay 1 0
        # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        # self.conv5_1 = DilateConv(d_rate=2, in_ch=512, out_ch=512) # error ! name conv5_1.dconv.weight erro in load vgg16
        # self.conv5_2 = DilateConv(d_rate=2, in_ch=512, out_ch=512)
        # self.conv5_3 = DilateConv(d_rate=2, in_ch=512, out_ch=512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        # 添加ASPP模块
        self.aspp = ASPP(in_channels=512, atrous_rates=[6, 12, 18], out_channels=512)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        # lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(64, 21, 1, padding=0)

        self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(128, 21, 1, padding=0)

        self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(512, 21, 1, padding=0)

        self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(512, 21, 1, padding=0)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        # 应用ASPP模块
        conv5_3 = self.aspp(conv5_3)

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)
        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        weight_deconv5 = make_bilinear_weights(32, 1).cuda()

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)

        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        ### crop way suggested by liu
        # so1 = crop_caffe(0, so1, img_H, img_W)
        # so2 = crop_caffe(1, upsample2, img_H, img_W)
        # so3 = crop_caffe(2, upsample3, img_H, img_W)
        # so4 = crop_caffe(4, upsample4, img_H, img_W)
        # so5 = crop_caffe(8, upsample5, img_H, img_W)
        ## upsample way
        # so1 = F.upsample_bilinear(so1, size=(img_H,img_W))
        # so2 = F.upsample_bilinear(so2, size=(img_H,img_W))
        # so3 = F.upsample_bilinear(so3, size=(img_H,img_W))
        # so4 = F.upsample_bilinear(so4, size=(img_H,img_W))
        # so5 = F.upsample_bilinear(so5, size=(img_H,img_W))

        # 创建边框遮罩
        def create_border_mask(h, w, border_size=5, device='cuda'):
            mask = torch.ones((1, 1, h, w), device=device)
            mask[:, :, :border_size, :] = 0
            mask[:, :, -border_size:, :] = 0
            mask[:, :, :, :border_size] = 0
            mask[:, :, :, -border_size:] = 0
            return mask

        border_mask = create_border_mask(img_H, img_W, border_size=20, device=x.device)

        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)
        # 对每个输出应用遮罩
        # so1 = so1 * border_mask
        # so2 = so2 * border_mask
        # so3 = so3 * border_mask
        # so4 = so4 * border_mask
        # so5 = so5 * border_mask
        # fuse = fuse * border_mask
        # results = [so1, so2, so3, so4, so5, fuse]
        # results = [so1, so2, so3, so4, so5, so5]   # 只取so5作为训练结果
        results = [so1, so2, so3, so4, so5, fuse]
        # results = [torch.sigmoid(r) * border_mask for r in results]  # 先激活再遮罩
        results = [torch.sigmoid(r) for r in results]  # 激活

        return results


def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]


def crop_caffe(location, variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(location)
    y1 = int(location)
    return variable[:, :, y1: y1 + th, x1: x1 + tw]


# make a bilinear interpolation kernel
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(in_channels, out_channels, h, w):
    weights = np.zeros([in_channels, out_channels, h, w])
    if in_channels != out_channels:
        raise ValueError("Input Output channel!")
    if h != w:
        raise ValueError("filters need to be square!")
    filt = upsample_filt(h)
    weights[range(in_channels), range(out_channels), :, :] = filt
    return np.float32(weights)


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def upsample(input, stride, num_channels=1):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)
