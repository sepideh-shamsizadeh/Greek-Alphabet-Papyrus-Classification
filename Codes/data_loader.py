import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data_split import train_labels, train_image_dirs, train_bboxs

class CustomDataset(Dataset):
    def __init__(self, image_dirs, bboxs, targets, transform=None):
        self.image_dirs = image_dirs
        self.bboxs = bboxs
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join('Data/HomerCompTraining', self.image_dirs[idx])
        image = Image.open(image_path).convert('RGB')

        # Get bounding boxes and labels for the current image
        bboxs = self.bboxs[idx]
        targets = self.targets[idx]

        images = []
        labels = []
        for bbox, target in zip(bboxs, targets):
            x, y, w, h = bbox
            crop = image.crop((x, y, x + w, y + h))

            if self.transform:
                crop = self.transform(crop)

            images.append(crop)
            labels.append(target)

        return images, labels

# Define any transformations (if needed)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128 (optional)
    transforms.ToTensor(),  # Convert to tensor
])

# Create dataset and dataloader
dataset = CustomDataset(train_image_dirs, train_bboxs, train_labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Example usage
for batch in dataloader:
    images, labels = batch
    for img, lbl in zip(images[0], labels[0]):
        print(img.shape, lbl)  # Tensor shape and label

print("Total batches:", len(dataloader))
