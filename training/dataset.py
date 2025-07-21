import os
from PIL import Image
from torch.utils.data import Dataset


class TumorDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]  # Aceitar string única ou lista

        self.transform = transform
        self.samples = []
        class_names = ["NILM", "LSIL", "HSIL", "INVALID"]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        for root_dir in root_dirs:
            print(f"🔍 Carregando imagens de: {root_dir}")
            for class_name in class_names:
                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    for filename in os.listdir(class_path):
                        img_path = os.path.join(class_path, filename)
                        self.samples.append((img_path, class_name))
                else:
                    print(f"⚠️ Aviso: classe '{class_name}' não encontrada em {root_dir}")

        if not self.samples:
            raise ValueError("Nenhuma imagem encontrada nos diretórios fornecidos.")

        print(f"✅ Total de imagens carregadas: {len(self.samples)}")
        print(f"📂 Classes mapeadas: {self.class_to_idx}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_to_idx[label]
        return image, label_idx
