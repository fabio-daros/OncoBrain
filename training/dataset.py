import os
from PIL import Image
from torch.utils.data import Dataset


class TumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    self.samples.append((os.path.join(class_path, filename), class_name))
        class_names = ["NILM", "LSIL", "HSIL", "INVALID"]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        for cls in self.class_to_idx:
            expected_dir = os.path.join(root_dir, cls)
            if not os.path.exists(expected_dir):
                raise ValueError(f"Expected class directory not found: {expected_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_to_idx[label]
        return image, label_idx
