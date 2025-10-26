import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json

class CocoDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None):
        """
        images_dir: путь к папке с изображениями
        annotations_file: путь к JSON файлу с аннотациями в формате COCO
        transforms: torchvision.transforms для изображения и аннотаций
        """
        self.images_dir = images_dir
        self.transforms = transforms

        # Загружаем аннотации
        with open(annotations_file) as f:
            self.coco = json.load(f)

        # Создаем словарь id изображения -> аннотации
        self.imgs = {img['id']: img for img in self.coco['images']}
        self.annos = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annos:
                self.annos[img_id] = []
            self.annos[img_id].append(ann)

        self.ids = list(self.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Получаем аннотации для этого изображения
        annos = self.annos.get(img_id, [])

        boxes = []
        labels = []
        for anno in annos:
            # COCO bbox формат: [x, y, width, height]
            x, y, w, h = anno['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(anno['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
