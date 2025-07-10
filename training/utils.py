import torch
import os


def save_model(model, output_dir, model_name="onco_vit_model.pth"):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, model_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved in: {save_path}")


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)
