import os
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageFile
import numpy as np

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_data(json_path):
    with open(json_path) as f:
        data = json.load(f)

    category_id_to_name = {item['id']: item['name'] for item in data['categories']}
    unique_ids = set(item['id'] for item in data['images'])

    return data, category_id_to_name, unique_ids

def get_image_data(image_id, data):
    category_ids = []
    image_file_name = None
    
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            category_ids.append(annotation['category_id'])
    
    for image in data['images']:
        if image['id'] == image_id:
            image_file_name = image['file_name']
            break
    
    return image_file_name, category_ids

class PapyrusDataset(Dataset):
    def __init__(self, image_ids, data, category_id_to_name, transform=None):
        self.image_ids = image_ids
        self.data = data
        self.category_id_to_name = category_id_to_name
        self.transform = transform
        self.num_classes = len(category_id_to_name)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_file_name, category_ids = get_image_data(image_id, self.data)
        
        labels = np.zeros(self.num_classes, dtype=np.float32)
        for cat_id in category_ids:
            labels[cat_id] = 1.0
        
        image_path = os.path.join('Data/HomerCompTraining', image_file_name[2:])
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(labels)

def create_dataloaders(data, category_id_to_name, unique_ids, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_ids = list(unique_ids)
    dataset = PapyrusDataset(image_ids, data, category_id_to_name, transform)

    # Calculate split sizes
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
