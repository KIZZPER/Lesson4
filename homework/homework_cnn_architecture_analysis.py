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


# Добавляем корневую директорию проекта в путь поиска модулей
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_models import VariableKernelCNN, DeepCNN, CNNWithResidual
from utils.training_utils import train_model, count_parameters
from utils.visualization_utils import (
    plot_training_history,
    plot_model_comparison,
    visualize_feature_maps,
    plot_gradient_flow
)
from utils.comparison_utils import (
    create_comparison_table,
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def prepare_data(batch_size=64):
    """Подготовка данных MNIST для быстрого анализа архитектур"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_root = os.path.join(Path(__file__).parent.parent, 'data')

    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def save_results(results, filepath):
    """Сохранение результатов в JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            'parameters': int(data['parameters']),
            'final_test_acc': float(data['final_test_acc']),
            'history': {
                'train_loss': [float(x) for x in data['history']['train_loss']],
                'test_acc': [float(x) for x in data['history']['test_acc']]
            }
        }

    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=4)
    logging.info(f"Saved results to {filepath}")


def experiment_2_1_kernel_size_analysis():
    """
    Задание 2.1: Влияние размера ядра свертки
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2.1: Kernel Size Analysis")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    kernel_sizes = [3, 5, 7]
    results = {}
    criterion = nn.CrossEntropyLoss()

    for k in kernel_sizes:
        model_name = f'Kernel_{k}x{k}'
        print(f"\nTraining {model_name}...")

        # Создаем модель с нужным размером ядра
        model = VariableKernelCNN(kernel_size=k, input_channels=1, num_classes=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        params = count_parameters(model)
        history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device=device)

        results[model_name] = {
            'history': history,
            'parameters': params,
            'final_test_acc': history['test_acc'][-1]
        }

        # Визуализация активаций первого слоя
        visualize_feature_maps(
            model,
            next(iter(test_loader))[0][0:1].to(device),
            layer_idx=0,
            title=f"Activations (Kernel {k}x{k})",
            save_path=f"plots/architecture_analysis/activations_k{k}.png"
        )

    # Сравнение
    print("\nKernel Size Comparison:")
    print(create_comparison_table(results))

    plot_model_comparison(results, metric='final_test_acc', title="Kernel Size Impact",
                          save_path="plots/architecture_analysis/kernel_comparison.png")

    save_results(results, 'results/architecture_analysis/kernel_size_results.json')
    return results


def experiment_2_2_depth_analysis():
    """
    Задание 2.2: Влияние глубины CNN
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2.2: CNN Depth Analysis")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models_config = {
        'Shallow_CNN_2_layers': DeepCNN(num_conv_layers=2),
        'Medium_CNN_4_layers': DeepCNN(num_conv_layers=4),
        'Deep_CNN_6_layers': DeepCNN(num_conv_layers=6),
        'ResNet_Alternative': CNNWithResidual()
    }

    results = {}
    criterion = nn.CrossEntropyLoss()

    for name, model in models_config.items():
        print(f"\nTraining {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=12, device=device)

        # Анализ градиентов
        model.zero_grad()
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()

        plot_gradient_flow(
            model.named_parameters(),
            title=f"Gradient Flow: {name}",
            save_path=f"plots/architecture_analysis/gradients_{name}.png"
        )

        results[name] = {
            'history': history,
            'parameters': count_parameters(model),
            'final_test_acc': history['test_acc'][-1]
        }

        plot_training_history(history, title=name, save_path=f"plots/architecture_analysis/{name}_history.png")

    print("\nDepth Analysis Comparison:")
    print(create_comparison_table(results))

    plot_model_comparison(results, metric='final_test_acc', title="Depth Impact Analysis",
                          save_path="plots/architecture_analysis/depth_comparison.png")

    save_results(results, 'results/architecture_analysis/depth_results.json')
    return results


def main():
    print("\n" + "#" * 80)
    print("HOMEWORK: CNN ARCHITECTURE ANALYSIS")
    print("#" * 80 + "\n")

    Path('plots/architecture_analysis').mkdir(parents=True, exist_ok=True)
    Path('results/architecture_analysis').mkdir(parents=True, exist_ok=True)

    experiment_2_1_kernel_size_analysis()
    experiment_2_2_depth_analysis()

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
