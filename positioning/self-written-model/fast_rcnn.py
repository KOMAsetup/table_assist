from torch.utils.data import DataLoader
from datasets_classes import CocoDataset
from os import path
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


if torch.cuda.is_available() : print("cuda is available") 
else : print("cuda !!! not !!! available !!!")


path_to_data = "/home/timofey/Documents/projects/table_assist/data/new_roboflow_datapack/Restaurant_Tables.v1i.coco"

train_path = path.join(path_to_data, "train")
train_json = path.join(train_path, "_annotations.coco.json")

val_path = path.join(path_to_data, "valid")
val_json = path.join(val_path, "_annotations.coco.json")


def train_fast_rcnn(model, train_loader, val_loader, optimizer, device="cuda",  num_epochs=20):
    device = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    model.to(device)
    print_freq = 1
    loss_history = []
    eval_loss_history = []

    print("==================training===================")

    for epoch in range(num_epochs):
        curr_loss = 0.0
        model.train()

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]  

            loss_dict = model(images, targets)
            loss = sum(value for value in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_loss += loss.item()

        eval_avg_loss = eval_fast_rcnn(model, val_loader, device)

        avg_loss = curr_loss / len(train_loader)
        if epoch % print_freq == 0:
            print(f"Epoch - {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f} - evaluation average loss: {eval_avg_loss}")

        loss_history.append(avg_loss)
        eval_loss_history.append(eval_avg_loss)


@torch.no_grad()
def eval_fast_rcnn(model, val_loader, device="cuda"):
    model.train() ###########################################
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




model_fastrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
num_classes = 3
in_features = model_fastrcnn.roi_heads.box_predictor.cls_score.in_features
model_fastrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)





train_dataset = CocoDataset(train_path, train_json, transforms=simple_transform)

train_loader = DataLoader(train_dataset, 
    batch_size=3, 
    shuffle=True, 
    collate_fn=lambda x: tuple(zip(*x))
)

val_dataset = CocoDataset(val_path, val_json, transforms=simple_transform)

val_loader = DataLoader(val_dataset, 
    batch_size=3, 
    shuffle=True, 
    collate_fn=lambda x: tuple(zip(*x))
)

optimizer = torch.optim.SGD(
    model_fastrcnn.parameters(),
    lr=0.005,       
    momentum=0.9,   
    weight_decay=0.0005  # L2 регуляризация
)

train_fast_rcnn(model_fastrcnn, 
    train_loader, 
    val_loader, 
    optimizer, 
    device="cuda", 
    num_epochs=20
)




