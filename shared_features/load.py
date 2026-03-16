import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -----------------------------
# Dataset Class
# -----------------------------

class IconDataset(Dataset):

    def __init__(self, dataset_path, transform=None):

        self.image_paths = []
        self.labels = []
        self.transform = transform

        # read class folders
        classes = sorted(os.listdir(dataset_path))

        for label, class_name in enumerate(classes):

            class_folder = os.path.join(dataset_path, class_name)

            if not os.path.isdir(class_folder):
                continue

            # read images inside class folder
            for img in os.listdir(class_folder):

                img_path = os.path.join(class_folder, img)

                if not os.path.isfile(img_path):
                    continue

                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -----------------------------
# Image Transform
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# -----------------------------
# Dataset Path
# -----------------------------

dataset_path = "data_preprocessing/dataset_split/train"


# -----------------------------
# Create Dataset
# -----------------------------

dataset = IconDataset(dataset_path, transform=transform)


# -----------------------------
# Create DataLoader
# -----------------------------

loader = DataLoader(dataset, batch_size=32, shuffle=True)


# -----------------------------
# Test DataLoader
# -----------------------------

for images, labels in loader:

    print("Image batch shape:", images.shape)
    print("Labels:", labels)

    break