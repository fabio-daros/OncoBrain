import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model.faster_rcnn_model import create_faster_rcnn_model
from training.detection_dataset import CervixDetectionDataset
from training.utils import collate_fn, save_model
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from training import config

# Configurações
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = config.num_classes
epochs = config.epochs
batch_size = config.batch_size
learning_rate = config.learning_rate
output_dir = os.path.join(config.model_output_dir, "faster_rcnn_model_checkpoints_with_logs")
os.makedirs(output_dir, exist_ok=True)

# TensorBoard
log_dir = os.path.join(output_dir, "runs")
writer = SummaryWriter(log_dir=log_dir)

# Dataset
transform = transforms.ToTensor()

dataset = CervixDetectionDataset(
    config.detection_images_dir,
    config.detection_annotations_path,
    transforms=transform
)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Modelo
model = create_faster_rcnn_model(num_classes=num_classes)
model.to(device)

# Otimizador
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

# Histórico de perda
epoch_losses = []

# Treinamento
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Log por batch no TensorBoard
        writer.add_scalar("Loss/batch", losses.item(), epoch * len(data_loader) + batch_idx)

        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {losses.item():.4f}")

    avg_epoch_loss = running_loss / len(data_loader)
    epoch_losses.append(avg_epoch_loss)
    writer.add_scalar("Loss/epoch", avg_epoch_loss, epoch)

    # Salvar checkpoint por epoch
    save_model(model, output_dir, model_name=f"faster_rcnn_epoch_{epoch}.pth")
    print(f"Epoch {epoch} completed - Average Loss: {avg_epoch_loss:.4f} - Checkpoint saved.")

# Plot final da perda por epoch
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), epoch_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Training Loss per Epoch - Faster R-CNN")
plt.grid(True)
loss_plot_path = os.path.join(output_dir, "training_loss.png")
plt.savefig(loss_plot_path)
plt.show()

print(f"Training loss plot saved at: {loss_plot_path}")

# Finaliza TensorBoard
writer.close()
