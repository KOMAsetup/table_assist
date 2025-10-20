import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import segmentation_models_pytorch as smp
from model import model_masrrcnn, model_unet, model_resnet  

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
        # Для Mask R-CNN встроенная loss функция
        return None
    else:
        # Для сегментационных моделей бинарная кросс-энтропия
        return nn.BCELoss()



def train(model_name, train_loader, val_loader=None, epochs=10, lr=1e-3, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device)
    loss_fn = get_loss(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            if model_name == "maskrcnn":
                # Mask R-CNN возвращает dict с лоссами
                targets = [{"boxes": torch.zeros(1, 4).to(device), "labels": torch.zeros(1, dtype=torch.int64).to(device), "masks": masks}]
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            else:
                preds = model(images)
                loss = loss_fn(preds, masks)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

        if val_loader:
            validate(model, val_loader, loss_fn, model_name, device)

def validate(model, val_loader, loss_fn, model_name, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            
            if model_name == "maskrcnn":
                targets = [{"boxes": torch.zeros(1, 4).to(device), "labels": torch.zeros(1, dtype=torch.int64).to(device), "masks": masks}]
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            else:
                preds = model(images)
                loss = loss_fn(preds, masks)
            
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss/len(val_loader)}")
