from training.dataset import TumorDataset
from training import config

dataset = TumorDataset(root_dir=config.train_data_dir)

print("Classes found:", dataset.class_to_idx)

print("\nLabels of the first 20 samples:")
for i in range(min(20, len(dataset))):
    _, label_idx = dataset[i]
    print(f"SAMPLE {i}: CLASS {label_idx}")
