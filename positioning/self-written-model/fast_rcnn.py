from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets_classes import CocoDataset
from os import path
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch

path_to_data = "/home/timofey/Documents/projects/table_assist/data/new_roboflow_datapack/Restaurant_Tables.v1i.coco"

train_path = path.join(path_to_data, "train")
train_json = path.join(train_path, "_annotations.coco.json")

val_path = path.join(path_to_data, "valid")
val_json = path.join(val_path, "_annotations.coco.json")



model_fastrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model_fastrcnn.roi_heads.box_predictor.cls_score.in_features
model_fastrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

def train_fast_rcnn(model, train_loader, val_loader, optimiser, device="cuda",  num_epochs=20):
    device = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    model.to(device)
    print_freq = 4
    loss_history = []

    print("==================training===================")

    for epoch in range(num_epochs):
        curr_loss = 0.0
        model.train()

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]  

            loss_dict = model(images, targets)
            loss = sum(value for value in loss_dict.values())

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            curr_loss += loss.item()

        avg_loss = curr_loss / len(train_loader)
        if epoch % print_freq == 0:
            print(f"Epoch - {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        loss_history.append(avg_loss)


@torch.no_grad()
def eval_fast_rcnn(model, val_loader, device="cuda"):
    model.eval()
    total_loss = 0.0

    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        loss_dict = model(images, targets)
        loss = sum(value for value in loss_dict.values())
        total_loss += loss.item()


    avg_loss = total_loss / len(val_loader)
    return avg_loss


    









def simple_transform(img, target):
    img = F.to_tensor(img)  # из PIL Image в FloatTensor [C,H,W], значения 0..1
    return img, target



# Создаем датасет
dataset = CocoDataset(train_path, train_json, transforms=simple_transform)

# Создаем DataLoader
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))



