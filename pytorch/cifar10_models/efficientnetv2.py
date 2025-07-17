import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['efficientnetv2_b0', 'efficientnetv2_b0_custom']

class MBConv(nn.Module):
    '''EfficientNetV2 MBConv block with SE and expansion'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(MBConv, self).__init__()
        self.stride = stride
        mid_planes = in_planes * expansion

        self.use_residual = stride == 1 and in_planes == out_planes

        self.expand_conv = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False) if expansion != 1 else None
        self.bn0 = nn.BatchNorm2d(mid_planes) if expansion != 1 else None

        self.dwconv = nn.Conv2d(mid_planes, mid_planes, 3, stride, 1, groups=mid_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        # SE block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_planes, mid_planes // 4, 1),
            nn.SiLU(),
            nn.Conv2d(mid_planes // 4, mid_planes, 1),
            nn.Sigmoid()
        )

        self.project_conv = nn.Conv2d(mid_planes, out_planes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        identity = x

        out = x
        if self.expand_conv:
            out = self.expand_conv(out)
            out = self.bn0(out)
            out = F.silu(out)

        out = self.dwconv(out)
        out = self.bn1(out)
        out = F.silu(out)

        # SE
        w = self.se(out)
        out = out * w

        out = self.project_conv(out)
        out = self.bn2(out)

        if self.use_residual:
            out += identity
        return out


class FusedMBConv(nn.Module):
    '''Fused MBConv: expansion and depthwise in one step'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(FusedMBConv, self).__init__()
        self.stride = stride
        mid_planes = in_planes * expansion
        self.use_residual = stride == 1 and in_planes == out_planes

        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.SiLU(),
            nn.Conv2d(mid_planes, out_planes, kernel_size=1, bias=False) if expansion != 1 else nn.Identity(),
            nn.BatchNorm2d(out_planes) if expansion != 1 else nn.Identity()
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out += x
        return out


class EfficientNetV2(nn.Module):
    def __init__(self, cfg, num_classes=101):
        super(EfficientNetV2, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)

        self.layers = self._make_layers(24)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(256, num_classes)


    def _make_layers(self, in_planes):
        layers = []
        for block_type, expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for s in strides:
                if block_type == 'F':
                    layers.append(FusedMBConv(in_planes, out_planes, expansion, s))
                else:
                    layers.append(MBConv(in_planes, out_planes, expansion, s))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def efficientnetv2_bo():
    cfg = [
        # block, exp, out, n, s
        ('F', 1, 24, 2, 1),
        ('F', 4, 48, 4, 2),
        ('F', 4, 64, 4, 2),
        ('M', 4, 128, 6, 2),
        ('M', 6, 160, 9, 1),
        ('M', 6, 256, 15, 2),
    ]
    return EfficientNetV2(cfg)

def efficientnetv2_bo_custom():
    return efficientnetv2_bo()

__all__ = ['efficientnetv2_bo', 'efficientnetv2_bo_custom']
