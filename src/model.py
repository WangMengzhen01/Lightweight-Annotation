import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from torchsummary import summary  # 导入 summary

'''-------------SE模块-----------------------------'''
# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)


class ConvBNReLU(nn.Module):  # 定义 ConvBNReLU 类
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBNRELUSE(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv_bn_relu = ConvBNReLU(in_ch, out_ch, kernel_size, dilation)
        self.se = SE_Block(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_bn_relu(x)
        x = self.se(x)
        return x


class DownConvBNRELUSE(ConvBNRELUSE):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        return super().forward(x)


class UpConvBNRELUSE(ConvBNRELUSE):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return super().forward(torch.cat([x1, x2], dim=1))


class RSUSE(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNRELUSE(in_ch, out_ch)

        encode_list = [DownConvBNRELUSE(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNRELUSE(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNRELUSE(mid_ch, mid_ch))
            decode_list.append(UpConvBNRELUSE(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNRELUSE(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4FSE(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNRELUSE(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNRELUSE(out_ch, mid_ch),
                                             ConvBNRELUSE(mid_ch, mid_ch, dilation=2),
                                             ConvBNRELUSE(mid_ch, mid_ch, dilation=4),
                                             ConvBNRELUSE(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNRELUSE(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNRELUSE(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNRELUSE(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class U2NetSE(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])

        encode_list = []
        side_list = []
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            encode_list.append(RSUSE(*c[:4]) if c[4] is False else RSU4FSE(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        decode_list = []
        for c in cfg["decode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            decode_list.append(RSUSE(*c[:4]) if c[4] is False else RSU4FSE(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        # collect encode outputs
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i!= self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        # collect decode outputs
        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m(torch.concat([x, x2], dim=1))
            decode_outputs.insert(0, x)

        # collect side outputs
        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)

        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [x] + side_outputs
        else:
            return torch.sigmoid(x)


def u2net_full_se(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 32, 64, False, False],  # En1
                   [6, 64, 32, 128, False, False],  # En2
                   [5, 128, 64, 256, False, False],  # En3
                   [4, 256, 128, 512, False, False],  # En4
                   [4, 512, 256, 512, True, False],  # En5
                   [4, 512, 256, 512, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 1024, 256, 512, True, True],  # De5
                   [4, 1024, 128, 256, False, True],  # De4
                   [5, 512, 64, 128, False, True],  # De3
                   [6, 256, 32, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return U2NetSE(cfg, out_ch)


def u2net_lite_se(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return U2NetSE(cfg, out_ch)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = u2net_full_se().to(device)
    # 打印网络结构和参数
    summary(net, (3, 224, 224))