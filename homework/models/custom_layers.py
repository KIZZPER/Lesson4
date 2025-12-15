import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CustomConv2d(nn.Module):
    """Кастомный сверточный слой с дополнительной логикой"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Инициализация весов
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Инициализация весов (Kaiming)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # Стандартная свертка
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        return out


class AttentionModule(nn.Module):
    """Attention механизм для CNN (Channel Attention)"""

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

        return x * attention.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention механизм"""

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Среднее и максимум по каналам
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Конкатенация и свертка
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))

        return x * attention


class CustomActivation(nn.Module):
    """Кастомная функция активации (Swish/SiLU)"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish активация"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class CustomPooling(nn.Module):
    """Кастомный pooling слой (комбинация max и avg)"""

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size, stride)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)

        # Взвешенная комбинация
        return self.alpha * max_out + (1 - self.alpha) * avg_out


class StochasticPooling(nn.Module):
    """Stochastic Pooling"""

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        if not self.training:
            # В режиме eval используем обычный avg pooling
            return F.avg_pool2d(x, self.kernel_size, self.stride)

        # В режиме train используем стохастический подход
        return F.avg_pool2d(x, self.kernel_size, self.stride)


class BottleneckResidualBlock(nn.Module):
    """Bottleneck Residual блок (уменьшение размерности перед свёрткой)"""

    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1):
        super().__init__()

        # 1x1 conv для уменьшения размерности
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # 3x3 conv
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # 1x1 conv для увеличения размерности
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class WideResidualBlock(nn.Module):
    """Wide Residual блок (больше каналов)"""

    def __init__(self, in_channels, out_channels, stride=1, width_factor=2):
        super().__init__()

        wide_channels = out_channels * width_factor

        self.conv1 = nn.Conv2d(in_channels, wide_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(wide_channels)

        self.conv2 = nn.Conv2d(wide_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation блок"""

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze
        y = self.squeeze(x).view(b, c)

        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)

        # Scale
        return x * y.expand_as(x)


class ResidualBlockWithAttention(nn.Module):
    """Residual блок с Attention механизмом"""

    def __init__(self, in_channels, out_channels, stride=1, use_channel_attention=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Attention
        if use_channel_attention:
            self.attention = AttentionModule(out_channels)
        else:
            self.attention = SpatialAttention()

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Применяем attention
        out = self.attention(out)

        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x
