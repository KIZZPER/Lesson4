import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import time
from sklearn.metrics import classification_report, confusion_matrix


def compare_model_parameters(models: Dict[str, nn.Module]) -> Dict[str, int]:
    """Сравнение количества параметров в моделях"""
    param_counts = {}

    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_counts[name] = {
            'total': total_params,
            'trainable': trainable_params
        }

    return param_counts


def compare_inference_speed(models: Dict[str, nn.Module], input_size: Tuple,
                            device: torch.device, num_iterations: int = 100) -> Dict[str, float]:
    """Сравнение скорости инференса моделей"""
    inference_times = {}
    dummy_input = torch.randn(input_size).to(device)

    for name, model in models.items():
        model.eval()
        model = model.to(device)

        # Прогрев
        with torch.no_grad():
            _ = model(dummy_input)

        # Измерение
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)

        inference_times[name] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }

    return inference_times


def get_predictions(model: nn.Module, data_loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Получение предсказаний модели на датасете"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: List[str] = None) -> Dict:
    """Вычисление метрик классификации"""
    accuracy = np.mean(y_true == y_pred) * 100
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }


def analyze_overfitting(history: Dict[str, List[float]]) -> Dict[str, float]:
    """Анализ переобучения модели"""
    train_acc = history['train_acc']
    test_acc = history['test_acc']
    train_loss = history['train_loss']
    test_loss = history['test_loss']

    final_acc_gap = train_acc[-1] - test_acc[-1]
    final_loss_gap = test_loss[-1] - train_loss[-1]
    max_acc_gap = max([t - v for t, v in zip(train_acc, test_acc)])
    max_loss_gap = max([v - t for t, v in zip(test_loss, train_loss)])

    if len(test_acc) >= 5:
        last_5_trend = np.mean(np.diff(test_acc[-5:]))
    else:
        last_5_trend = 0.0

    return {
        'final_acc_gap': final_acc_gap,
        'final_loss_gap': final_loss_gap,
        'max_acc_gap': max_acc_gap,
        'max_loss_gap': max_loss_gap,
        'test_acc_trend': last_5_trend,
        'is_overfitting': final_acc_gap > 10.0
    }


def compare_model_complexity(models: Dict[str, nn.Module]) -> Dict:
    """Сравнение сложности моделей"""
    complexity = {}

    for name, model in models.items():
        num_layers = len(list(model.modules()))
        num_conv_layers = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
        num_linear_layers = len([m for m in model.modules() if isinstance(m, nn.Linear)])
        num_bn_layers = len([m for m in model.modules() if isinstance(m, nn.BatchNorm2d)])

        complexity[name] = {
            'total_layers': num_layers,
            'conv_layers': num_conv_layers,
            'linear_layers': num_linear_layers,
            'bn_layers': num_bn_layers
        }

    return complexity


def create_comparison_table(results: Dict[str, Dict]) -> str:
    """Создание таблицы сравнения моделей"""
    header = f"{'Model':<30} {'Test Acc':<12} {'Train Time':<15} {'Parameters':<15}"
    separator = "=" * 80

    lines = [separator, header, separator]

    for model_name, data in results.items():
        test_acc = data.get('test_acc', [0])[-1] if isinstance(data.get('test_acc'), list) else data.get('test_acc', 0)
        train_time = sum(data.get('epoch_times', [0]))
        params = data.get('parameters', 0)

        line = f"{model_name:<30} {test_acc:>10.2f}% {train_time:>13.2f}s {params:>13,d}"
        lines.append(line)

    lines.append(separator)

    return "\n".join(lines)
