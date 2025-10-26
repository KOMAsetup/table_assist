import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import segmentation_models_pytorch as smp
from model import model_masrrcnn, model_unet, model_resnet  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm


def get_model(name):
    """Возвращает выбранную модель по имени"""
    if name == "maskrcnn":
        return model_masrrcnn
    elif name == "unet":
        return model_unet
    elif name == "deeplabv3":
        return model_resnet
    else:
        raise ValueError(f"Unknown model: {name}")

def get_loss(model_name):
    """Выбирает функцию потерь по типу модели"""
    if model_name == "maskrcnn":
        return None
    else:
        return nn.BCELoss()



def train_model(model, train_loader, val_loader, device, optimizer, num_epochs=10, print_freq=10):
    """
    Обучение модели Faster R-CNN на заданное количество эпох с валидацией.

    Args:
        model: PyTorch модель (Faster R-CNN)
        train_loader: DataLoader для тренировочного датасета
        val_loader: DataLoader для валидационного датасета
        device: 'cuda' или 'cpu'
        optimizer: оптимизатор
        num_epochs: количество эпох
        print_freq: как часто печатать лосс внутри эпохи
    """
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for i, (images, targets) in enumerate(tqdm(train_loader, desc="Training")):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            if (i + 1) % print_freq == 0:
                avg_loss = running_loss / print_freq
                print(f"Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Валидация после каждой эпохи
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_val_loss += losses.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {avg_val_loss:.4f}")

