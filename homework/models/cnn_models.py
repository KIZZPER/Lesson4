import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Простая CNN для MNIST (2-3 conv слоя)"""

    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """Residual блок"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class CNNWithResidual(nn.Module):
    """CNN с Residual блоками для MNIST"""

    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, stride=2)
        self.res3 = ResidualBlock(64, 64)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNNForCIFAR(nn.Module):
    """CNN с Residual блоками для CIFAR-10"""

    def __init__(self, input_channels=3, num_classes=10, use_dropout=False):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 128, stride=2)
        self.res3 = ResidualBlock(128, 128)
        self.res4 = ResidualBlock(128, 256, stride=2)
        self.res5 = ResidualBlock(256, 256)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3) if use_dropout else nn.Identity()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class VariableKernelCNN(nn.Module):
    """CNN с различными размерами ядер свертки"""

    def __init__(self, kernel_size=3, input_channels=1, num_classes=10):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeepCNN(nn.Module):
    """Глубокая CNN (6+ conv слоев)"""

    def __init__(self, input_channels=1, num_classes=10, num_conv_layers=6):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1))
        self.layers.append(nn.BatchNorm2d(32))

        channels = [32, 64, 64, 128, 128, 256]
        for i in range(1, num_conv_layers):
            in_ch = channels[i - 1]
            out_ch = channels[min(i, len(channels) - 1)]

            self.layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(out_ch))

            if i % 2 == 1:
                self.layers.append(nn.MaxPool2d(2, 2))

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        final_channels = channels[min(num_conv_layers - 1, len(channels) - 1)]
        self.fc = nn.Linear(final_channels * 4 * 4, num_classes)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d)):
                if isinstance(layer, nn.Conv2d):
                    x = layer(x)
                elif isinstance(layer, nn.BatchNorm2d):
                    x = F.relu(layer(x))
                else:
                    x = layer(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x