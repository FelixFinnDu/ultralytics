"""my Block modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = (
    "MyConv",
    "UFNONet",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MyConv(nn.Module):
    """MyConv: 标准 3x3 卷积模块（Conv + BN + SiLU）"""
    def __init__(self, c1, c2, k=3, s=1):  
        super().__init__()
        # 3x3 卷积，自动推理 padding (k//2) 以保持尺寸，去偏置以配合BN
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=k // 2, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU()  # SiLU激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-2, -1])

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, -self.modes2:], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, -self.modes2:], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)

        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels * 2, output_channels)
        self.deconv0 = self.deconv(input_channels * 2, output_channels)

        self.output_layer = self.output(input_channels * 2, output_channels,
                                        kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)


class UFNONet(nn.Module):
    def __init__(self, modes1, modes2, width, width2):
        super(UFNONet, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.width2 = width2

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)

        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.unet4 = U_net(self.width, self.width, 3, 0)
        self.unet5 = U_net(self.width, self.width, 3, 0)

        self.fc1 = nn.Linear(self.width, width2)
        self.fc2 = nn.Linear(width2, 3)

    def forward(self, x):
        # x_ = x
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        # x1 = self.conv1(x)
        # x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        # x = x1 + x2
        # x = F.relu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        # x = x1 + x2
        # x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.unet3(x)
        x = x1 + x2 + x3
        x = F.relu(x)

        # x1 = self.conv4(x)
        # x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        # x3 = self.unet4(x)
        # x = x1 + x2 + x3
        # x = F.relu(x)

        # x1 = self.conv5(x)
        # x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        # x3 = self.unet5(x)
        # x = x1 + x2 + x3
        # x = F.relu(x)

        x = x.permute(0, 2, 3, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = x.view(batchsize, 3, size_x, size_y)
        # x = x.view(batchsize, self.width2, size_x, size_y)

        return x

    
