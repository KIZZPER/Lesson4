import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from sklearn.metrics import confusion_matrix
import os


def plot_training_history(history: Dict[str, List[float]], title: str = "Training History",
                          save_path: str = None):
    """Визуализация кривых обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # График loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # График accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str] = None,
                          title: str = "Confusion Matrix",
                          save_path: str = None):
    """Визуализация confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)))
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_model_comparison(results: Dict[str, Dict], metric: str = 'test_acc',
                          title: str = "Model Comparison", save_path: str = None):
    """Сравнение различных моделей"""
    model_names = list(results.keys())
    values = [results[name][metric][-1] if isinstance(results[name][metric], list)
              else results[name][metric] for name in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def visualize_feature_maps(model: nn.Module, input_tensor: torch.Tensor,
                           layer_idx: int = 0, num_maps: int = 16,
                           title: str = "Feature Maps", save_path: str = None):
    """Визуализация feature maps сверточного слоя"""
    model.eval()

    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if layer_idx >= len(conv_layers):
        return

    hook = conv_layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(input_tensor)

    hook.remove()

    feature_maps = activations[0][0].cpu().numpy()
    num_maps = min(num_maps, feature_maps.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            ax.imshow(feature_maps[i], cmap='viridis')
            ax.set_title(f'Map {i}', fontsize=10)
        ax.axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_gradient_flow(named_parameters, title: str = "Gradient Flow", save_path: str = None):
    """Визуализация потока градиентов в сети"""
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", label="max-gradient")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b", label="mean-gradient")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90, fontsize=8)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=max(max_grads) * 1.1 if max_grads else 0.1)
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Gradient", fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_training_time_comparison(results: Dict[str, Dict], save_path: str = None):
    """Сравнение времени обучения моделей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, data in results.items():
        if 'epoch_times' in data:
            ax1.plot(data['epoch_times'], label=model_name, linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Training Time per Epoch', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    model_names = list(results.keys())
    total_times = [sum(results[name]['epoch_times']) if 'epoch_times' in results[name] else 0
                   for name in model_names]

    bars = ax2.bar(model_names, total_times, color=['skyblue', 'lightcoral', 'lightgreen'])

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}s',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Total Time (seconds)', fontsize=12)
    ax2.set_title('Total Training Time', fontsize=14)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()
