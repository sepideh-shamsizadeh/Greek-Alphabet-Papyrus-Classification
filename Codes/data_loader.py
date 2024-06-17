import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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

        # Convert lists to tensors
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        return images, labels
