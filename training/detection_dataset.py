from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import json


class CervixDetectionDataset(Dataset):
    def __init__(self, images_dir, annotations_json, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        with open(annotations_json) as f:
            self.coco_data = json.load(f)

        # Indexa as anotações por image_id para acesso rápido
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        # Mantém a lista de imagens
        self.images = self.coco_data['images']

    def __getitem__(self, idx):
        image_info = self.images[idx]
        img_path = os.path.join(self.images_dir, image_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Recupera as anotações para essa imagem
        ann_list = self.image_id_to_annotations.get(image_info['id'], [])
        boxes = []
        labels = []

        for ann in ann_list:
            x, y, w, h = ann['bbox']
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_info['id']])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)
