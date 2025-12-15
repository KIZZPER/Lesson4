import torch
import torch.nn as nn
import time
import logging
from typing import Tuple, Dict, List


def count_parameters(model: nn.Module) -> int:
    """Подсчет количества обучаемых параметров в модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, train_loader, criterion, optimizer, device) -> Tuple[float, float]:
    """
    Обучение модели на одной эпохе

    Returns:
        Tuple[float, float]: средняя loss и accuracy на эпохе
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device) -> Tuple[float, float]:
    """
    Оценка модели на тестовом наборе

    Returns:
        Tuple[float, float]: средняя loss и accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def train_model(model, train_loader, test_loader, criterion, optimizer,
                num_epochs, device, verbose=True) -> Dict[str, List[float]]:
    """
    Полный цикл обучения модели

    Returns:
        Dict с историей обучения (train_loss, train_acc, test_loss, test_acc, times)
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        epoch_time = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)

        if verbose:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} - "
                         f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                         f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% - "
                         f"Time: {epoch_time:.2f}s")

    return history


def measure_inference_time(model, test_loader, device, num_iterations=100) -> float:
    """
    Измерение времени инференса модели

    Returns:
        float: среднее время инференса на батч (в секундах)
    """
    model.eval()
    times = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_iterations:
                break

            inputs = inputs.to(device)

            # Прогрев GPU
            if i == 0:
                _ = model(inputs)
                continue

            start = time.time()
            _ = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)

    return sum(times) / len(times) if times else 0.0


def compute_gradient_stats(model) -> Dict[str, float]:
    """
    Вычисление статистики градиентов модели

    Returns:
        Dict со статистикой: mean, std, max, min градиентов
    """
    gradients = []

    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1))

    if not gradients:
        return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}

    all_grads = torch.cat(gradients)

    return {
        'mean': all_grads.mean().item(),
        'std': all_grads.std().item(),
        'max': all_grads.max().item(),
        'min': all_grads.min().item()
    }