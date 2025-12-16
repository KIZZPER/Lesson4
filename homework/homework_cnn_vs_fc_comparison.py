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

# Импорт модулей
from models.fc_models import SimpleFCNet, DeepFCNet, FCNetForCIFAR
from models.cnn_models import SimpleCNN, CNNWithResidual, CNNForCIFAR
from utils.training_utils import train_model, measure_inference_time, count_parameters
from utils.visualization_utils import (
    plot_training_history,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_gradient_flow,
    plot_training_time_comparison
)
from utils.comparison_utils import (
    get_predictions,
    compute_classification_metrics,
    analyze_overfitting,
    create_comparison_table
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def prepare_mnist_data(batch_size=64):
    """
    Подготовка датасета MNIST
    """
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


def prepare_cifar_data(batch_size=64):
    """
    Подготовка датасета CIFAR-10
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    data_root = os.path.join(Path(__file__).parent.parent, 'data')

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def save_results_to_json(results, filepath):
    """Сохранение результатов в JSON файл"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Конвертируем результаты в JSON формат
    json_results = {}
    for model_name, data in results.items():
        metrics_clean = {}
        if 'metrics' in data:
            metrics_clean['accuracy'] = float(data['metrics']['accuracy'])
            if 'classification_report' in data['metrics']:
                metrics_clean['report'] = data['metrics']['classification_report']

        json_results[model_name] = {
            'parameters': int(data['parameters']),
            'inference_time': float(data['inference_time']),
            'final_test_acc': float(data['final_test_acc']),
            'best_test_acc': float(data['best_test_acc']),
            'overfitting': {k: float(v) if isinstance(v, (int, float)) else v for k, v in data['overfitting'].items()},
            'history': {
                'train_acc': [float(x) for x in data['history']['train_acc']],
                'test_acc': [float(x) for x in data['history']['test_acc']],
                'train_loss': [float(x) for x in data['history']['train_loss']],
                'test_loss': [float(x) for x in data['history']['test_loss']],
                'epoch_times': [float(x) for x in data['history']['epoch_times']]
            }
        }

    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=4)

    logging.info(f"Results saved to {filepath}")


def experiment_1_1_mnist_comparison():
    """
    Задание 1.1: Сравнение моделей на MNIST
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1.1: MNIST Model Comparison (FC vs CNN vs ResNet)")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_mnist_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # Определение моделей
    models_config = {
        'FC_Net': SimpleFCNet(input_size=784, hidden_sizes=[256, 128], num_classes=10),
        'Simple_CNN': SimpleCNN(input_channels=1, num_classes=10),
        'ResNet_CNN': CNNWithResidual(input_channels=1, num_classes=10)
    }

    results = {}
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10

    for name, model in models_config.items():
        print(f"\nTraining {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 1. Подсчет параметров
        params = count_parameters(model)

        # 2. Обучение
        history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

        # 3. Инференс
        inf_time = measure_inference_time(model, test_loader, device)

        # 4. Метрики и анализ
        y_true, y_pred = get_predictions(model, test_loader, device)
        metrics = compute_classification_metrics(y_true, y_pred, class_names=[str(i) for i in range(10)])
        overfitting = analyze_overfitting(history)

        results[name] = {
            'history': history,
            'parameters': params,
            'inference_time': inf_time,
            'metrics': metrics,
            'overfitting': overfitting,
            'final_test_acc': history['test_acc'][-1],
            'best_test_acc': max(history['test_acc'])
        }

        # Визуализация истории для каждой модели
        plot_training_history(
            history,
            title=f"{name} Training History",
            save_path=f"plots/mnist_comparison/{name}_history.png"
        )

    # Общее сравнение
    print("\nComparison Table:")
    print(create_comparison_table(results))

    # Графики сравнения
    plot_model_comparison(results, metric='final_test_acc', title="MNIST: Accuracy Comparison",
                          save_path="plots/mnist_comparison/accuracy_comp.png")
    plot_training_time_comparison(results, save_path="plots/mnist_comparison/time_comp.png")

    # Сохранение результатов
    save_results_to_json(results, 'results/mnist_comparison/mnist_results.json')
    return results


def experiment_1_2_cifar_comparison():
    """
    Задание 1.2: Сравнение моделей на CIFAR-10
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1.2: CIFAR-10 Model Comparison & Regularization")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_cifar_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    models_config = {
        'Deep_FC_Net': FCNetForCIFAR(input_size=3 * 32 * 32, num_classes=10),
        'ResNet_Base': CNNForCIFAR(input_channels=3, num_classes=10, use_dropout=False),
        'ResNet_Regularized': CNNForCIFAR(input_channels=3, num_classes=10, use_dropout=True)
    }

    results = {}
    criterion = nn.CrossEntropyLoss()
    num_epochs = 20

    for name, model in models_config.items():
        print(f"\nTraining {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        params = count_parameters(model)


        history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

        inf_time = measure_inference_time(model, test_loader, device)
        y_true, y_pred = get_predictions(model, test_loader, device)
        metrics = compute_classification_metrics(y_true, y_pred, class_names=classes)
        overfitting = analyze_overfitting(history)

        results[name] = {
            'history': history,
            'parameters': params,
            'inference_time': inf_time,
            'metrics': metrics,
            'overfitting': overfitting,
            'final_test_acc': history['test_acc'][-1],
            'best_test_acc': max(history['test_acc'])
        }

        # Визуализации
        plot_training_history(history, title=f"{name} (CIFAR-10)",
                              save_path=f"plots/cifar_comparison/{name}_history.png")

        plot_confusion_matrix(y_true, y_pred, class_names=classes,
                              title=f"{name} Confusion Matrix",
                              save_path=f"plots/cifar_comparison/{name}_cm.png")

        # Анализ градиентов
        if 'CNN' in name or 'ResNet' in name:
            # Делаем один backward pass, чтобы получить градиенты
            model.zero_grad()
            dummy_input = torch.randn(1, 3, 32, 32).to(device)
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()
            plot_gradient_flow(model.named_parameters(), title=f"{name} Gradients",
                               save_path=f"plots/cifar_comparison/{name}_gradients.png")

    # Сравнение
    print("\nComparison Table:")
    print(create_comparison_table(results))

    plot_model_comparison(results, metric='final_test_acc', title="CIFAR-10: Accuracy Comparison",
                          save_path="plots/cifar_comparison/accuracy_comp.png")

    save_results_to_json(results, 'results/cifar_comparison/cifar_results.json')
    return results


def main():
    print("\n" + "#" * 80)
    print("HOMEWORK: CNN vs FC ARCHITECTURE COMPARISON")
    print("#" * 80 + "\n")

    # Создание папок
    for subdir in ['mnist_comparison', 'cifar_comparison']:
        Path(f'plots/{subdir}').mkdir(parents=True, exist_ok=True)
        Path(f'results/{subdir}').mkdir(parents=True, exist_ok=True)

    # Запуск экспериментов
    experiment_1_1_mnist_comparison()
    experiment_1_2_cifar_comparison()

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()