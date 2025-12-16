import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
import os
from pathlib import Path
import logging
import json

# Добавляем корневую директорию проекта
sys.path.append(str(Path(__file__).parent.parent))

from models.custom_layers import (
    CustomConv2d,
    CustomActivation,
    Mish,
    CustomPooling,
    StochasticPooling,
    ResidualBlockWithAttention,
    BottleneckResidualBlock,
    WideResidualBlock,
    DepthwiseSeparableConv
)
from utils.training_utils import train_model, count_parameters
from utils.visualization_utils import (
    plot_training_history,
    plot_model_comparison
)
from utils.comparison_utils import create_comparison_table

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def prepare_data(batch_size=64):
    """Подготовка CIFAR-10 для более интересных экспериментов"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_root = os.path.join(Path(__file__).parent.parent, 'data')
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def save_results(results, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    json_results = {
        name: {
            'parameters': int(data['parameters']),
            'final_test_acc': float(data['final_test_acc']),
            'history': {'test_acc': [float(x) for x in data['history']['test_acc']]}
        } for name, data in results.items()
    }
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=4)


class ExperimentalNet(nn.Module):
    """
    Вспомогательная модель для тестирования компонентов
    """
    def __init__(self, layer_type='standard', activation='relu', pooling='max'):
        super().__init__()

        # Выбор сверточного слоя
        if layer_type == 'custom_conv':
            self.conv1 = CustomConv2d(3, 32)
            self.conv2 = CustomConv2d(32, 64)
        elif layer_type == 'depthwise':
            self.conv1 = DepthwiseSeparableConv(3, 32)
            self.conv2 = DepthwiseSeparableConv(32, 64)
        else:
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # Выбор функции активации
        if activation == 'custom':
            self.act = CustomActivation()
        elif activation == 'mish':
            self.act = Mish()
        else:
            self.act = nn.ReLU()

        # Выбор пулинга
        if pooling == 'custom':
            self.pool = CustomPooling()
        elif pooling == 'stochastic':
            self.pool = StochasticPooling()
        else:
            self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(64 * 8 * 8, 10)  # Для 32x32 -> pool -> pool -> 8x8

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualExperimentsNet(nn.Module):
    def __init__(self, block_type='basic'):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Выбор типа блока
        if block_type == 'bottleneck':
            # in, bottleneck, out
            self.layer = BottleneckResidualBlock(64, 32, 64)
        elif block_type == 'wide':
            self.layer = WideResidualBlock(64, 64, width_factor=2)
        elif block_type == 'attention':
            self.layer = ResidualBlockWithAttention(64, 64)
        else:  # basic
            self.layer = WideResidualBlock(64, 64, width_factor=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def experiment_3_1_custom_layers():
    """
    Задание 3.1: Реализация кастомных слоев
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3.1: Custom Layers Evaluation")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = {
        'Baseline_Standard': ExperimentalNet(),
        'Custom_Activation_Mish': ExperimentalNet(activation='mish'),
        'Custom_Pooling_Weighted': ExperimentalNet(pooling='custom'),
        'Depthwise_Separable_Conv': ExperimentalNet(layer_type='depthwise')
    }

    results = {}
    criterion = nn.CrossEntropyLoss()

    for name, model in configs.items():
        print(f"\nTesting {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device=device)

        results[name] = {
            'history': history,
            'parameters': count_parameters(model),
            'final_test_acc': history['test_acc'][-1]
        }

    print("\nCustom Layers Comparison:")
    print(create_comparison_table(results))

    plot_model_comparison(
        results,
        metric='final_test_acc',
        title="Custom Layers Performance",
        save_path="plots/custom_layers/layers_comparison.png"
    )

    save_results(results, 'results/custom_layers/layers_results.json')
    return results


def experiment_3_2_residual_blocks():
    """
    Задание 3.2: Эксперименты с Residual блоками
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3.2: Advanced Residual Blocks")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = {
        'Basic_ResBlock': ResidualExperimentsNet(block_type='basic'),
        'Bottleneck_ResBlock': ResidualExperimentsNet(block_type='bottleneck'),
        'Wide_ResBlock': ResidualExperimentsNet(block_type='wide'),
        'Attention_ResBlock': ResidualExperimentsNet(block_type='attention')
    }

    results = {}
    criterion = nn.CrossEntropyLoss()

    for name, model in configs.items():
        print(f"\nTesting {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=12, device=device)

        results[name] = {
            'history': history,
            'parameters': count_parameters(model),
            'final_test_acc': history['test_acc'][-1]
        }

        plot_training_history(history, title=name, save_path=f"plots/custom_layers/{name}_history.png")

    print("\nResidual Blocks Comparison:")
    print(create_comparison_table(results))

    plot_model_comparison(
        results,
        metric='final_test_acc',
        title="Residual Blocks Comparison",
        save_path="plots/custom_layers/blocks_comparison.png"
    )

    save_results(results, 'results/custom_layers/blocks_results.json')
    return results


def main():
    print("\n" + "#" * 80)
    print("HOMEWORK: CUSTOM LAYERS & EXPERIMENTS")
    print("#" * 80 + "\n")

    Path('plots/custom_layers').mkdir(parents=True, exist_ok=True)
    Path('results/custom_layers').mkdir(parents=True, exist_ok=True)

    experiment_3_1_custom_layers()
    experiment_3_2_residual_blocks()

    print("\nAll custom layer experiments completed.")


if __name__ == "__main__":
    main()
