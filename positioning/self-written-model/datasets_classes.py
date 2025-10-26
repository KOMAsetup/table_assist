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

        self.keep_categories = {
            19: 1,   # people -> 1
            25: 2    # table -> 2
        }

        self.ids = []
        for img_id, img_info in self.imgs.items():
            annos = self.annos.get(img_id, [])
            valid_boxes = [
                anno['bbox'] for anno in annos
                if anno['category_id'] in self.keep_categories and anno['bbox'][2] > 0 and anno['bbox'][3] > 0
            ]
            if len(valid_boxes) > 0:
                self.ids.append(img_id)



    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        annos = self.annos.get(img_id, [])

        boxes = []
        labels = []
        for anno in annos:
            cat_id = anno['category_id']
            if cat_id in self.keep_categories:  # оставляем только нужные классы
                x, y, w, h = anno['bbox']
                if w <= 0 or h <= 0:  # игнорируем некорректные боксы
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(self.keep_categories[cat_id])  # новые лейблы 1 и 2

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
