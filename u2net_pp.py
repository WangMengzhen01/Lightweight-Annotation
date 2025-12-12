import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List

# =========================================================
# 【消融实验独立控制台】
# 修改这里来组合出 4 种实验模式：
# 1. Baseline: False, False
# 2. Only SE:   False, True
# 3. Only CBAM: True, False
# 4. Ours:      True, True (最终模型)
# =========================================================
USE_CBAM = True  # 开关 1：控制主干 RSU 模块是否使用 CBAM
USE_SE_SKIP = True  # 开关 2：控制跳跃连接是否使用 SE 模块
# =========================================================

'''------------- 1. 基础注意力模块定义 -------------'''


# SE模块
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# CBAM模块组件
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class CBAM_Block(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM_Block, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


'''------------- 2. 基础卷积单元修改 (受 USE_CBAM 控制) -------------'''


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ConvBNReLU_CBAM(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv_bn_relu = ConvBNReLU(in_ch, out_ch, kernel_size, dilation)

        # 【独立开关】只有 USE_CBAM 为 True 时，才加载 CBAM
        if USE_CBAM:
            self.att = CBAM_Block(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_bn_relu(x)

        # 【独立开关】只有 USE_CBAM 为 True 时，才计算 CBAM
        if USE_CBAM:
            x = self.att(x)
        return x


'''------------- 3. RSU 模块  -------------'''


class DownConvBNReLU_CBAM(ConvBNReLU_CBAM):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        return super().forward(x)


class UpConvBNReLU_CBAM(ConvBNReLU_CBAM):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return super().forward(torch.cat([x1, x2], dim=1))


class RSU_CBAM(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        assert height >= 2
        self.conv_in = ConvBNReLU_CBAM(in_ch, out_ch)
        encode_list = [DownConvBNReLU_CBAM(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU_CBAM(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU_CBAM(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU_CBAM(mid_ch * 2, mid_ch if i < height - 3 else out_ch))
        encode_list.append(ConvBNReLU_CBAM(mid_ch, mid_ch, dilation=2))
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


class RSU4F_CBAM(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU_CBAM(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU_CBAM(out_ch, mid_ch),
                                             ConvBNReLU_CBAM(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU_CBAM(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU_CBAM(mid_ch, mid_ch, dilation=8)])
        self.decode_modules = nn.ModuleList([ConvBNReLU_CBAM(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU_CBAM(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU_CBAM(mid_ch * 2, out_ch)])

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


'''------------- 4. U2Net 主网络 (核心修改区域) -------------'''


class U2Net_Improved(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        self.encode_num = len(cfg["encode"])

        # 1. 编码器
        encode_list = []
        side_list = []
        enc_out_channels = []
        for c in cfg["encode"]:
            encode_list.append(RSU_CBAM(*c[:4]) if c[4] is False else RSU4F_CBAM(*c[1:4]))
            enc_out_channels.append(c[3])
            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        # 2. 解码器
        decode_list = []
        for c in cfg["decode"]:
            decode_list.append(RSU_CBAM(*c[:4]) if c[4] is False else RSU4F_CBAM(*c[1:4]))
            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)

        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

        # 3. 跳跃连接 SE 模块 (受 USE_SE_SKIP 控制)
        self.skip_se_modules = nn.ModuleList()
        # 【独立开关】只有 USE_SE_SKIP 为 True 时，才加载 SE
        if USE_SE_SKIP:
            skip_se_list = []
            for ch in enc_out_channels[:-1][::-1]:
                skip_se_list.append(SE_Block(ch))
            self.skip_se_modules = nn.ModuleList(skip_se_list)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        # Encoding
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        # Decoding
        x = encode_outputs.pop()
        decode_outputs = [x]

        # 【独立开关】逻辑判断
        if USE_SE_SKIP:
            # 模式 A: 使用 SE 跳跃连接
            for m, se_skip in zip(self.decode_modules, self.skip_se_modules):
                x2 = encode_outputs.pop()
                x2 = se_skip(x2)  # 计算 SE
                x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
                x = m(torch.cat([x, x2], dim=1))
                decode_outputs.insert(0, x)
        else:
            # 模式 B: 不使用 SE (直接拼接)
            for m in self.decode_modules:
                x2 = encode_outputs.pop()
                # 无 SE 操作
                x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
                x = m(torch.cat([x, x2], dim=1))
                decode_outputs.insert(0, x)

        # Side Output
        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)

        x = self.out_conv(torch.cat(side_outputs, dim=1))

        if self.training:
            return [x] + side_outputs
        else:
            return torch.sigmoid(x)


'''------------- 5. 配置与实例化 -------------'''


def u2net_improved(out_ch: int = 1):
    cfg = {
        "encode": [[7, 3, 32, 64, False, False],
                   [6, 64, 32, 128, False, False],
                   [5, 128, 64, 256, False, False],
                   [4, 256, 128, 512, False, False],
                   [4, 512, 256, 512, True, False],
                   [4, 512, 256, 512, True, True]],
        "decode": [[4, 1024, 256, 512, True, True],
                   [4, 1024, 128, 256, False, True],
                   [5, 512, 64, 128, False, True],
                   [6, 256, 32, 64, False, True],
                   [7, 128, 16, 64, False, True]]
    }
    return U2Net_Improved(cfg, out_ch)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 50)
    print(f"当前实验配置:")
    print(f"  [1] RSU CBAM 模块:   {'【开启】' if USE_CBAM else '关闭'}")
    print(f"  [2] Skip SE  模块:   {'【开启】' if USE_SE_SKIP else '关闭'}")
    print("=" * 50)

    net = u2net_improved().to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    output = net(x)
    print("模型构建成功！输出形状:", output[0].shape)