import os
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader

from data_split import PapyrusDataProcessor

def preprocess_image(image):
    print("Preprocessing image...")
    display_image('croped',image)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image('gray', gray)
    # Contrast and Brightness Adjustment
    alpha = 1.5  # Contrast control
    beta = 50    # Brightness control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Noise Reduction
    blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)

    # Binarization
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    display_image('binary', binary)

    # Morphological Transformations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    display_image('opening', opening)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    display_image('closing', closing)

    # Deskewing (if necessary)
    coords = np.column_stack(np.where(closing > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = closing.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(closing, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    display_image('deskewed', deskewed)

    # Normalize size
    final = cv2.resize(opening, (128, 128))
    return final

def check_images(image_dirs):
    print("Checking images...")
    valid_image_dirs = []
    for img_dir in image_dirs:
        try:
            img = Image.open(img_dir)
            img.verify()  # Verify that it is an image
            valid_image_dirs.append(img_dir)
        except (IOError, SyntaxError) as e:
            print(f"Corrupted image file: {img_dir}")
    return valid_image_dirs

def display_image(title, image):
    print(f"Displaying image: {title}")
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()
    plt.pause(0.001)  # Pause to ensure the plot is updated

class CustomDataset(Dataset):
    def __init__(self, image_dirs, bboxs, targets, transform=None):
        print("Initializing CustomDataset...")
        self.image_dirs = image_dirs
        self.bboxs = bboxs
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        print(f"Getting item {idx}...")
        image_path = os.path.join('Data/HomerCompTraining', self.image_dirs[idx])
        print(f"Loading image from {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None, None
        display_image('original', image)
        
        # Get bounding boxes and labels for the current image
        bboxs = self.bboxs[idx]
        targets = self.targets[idx]

        images = []
        labels = []
        for bbox, target in zip(bboxs, targets):
            x, y, w, h = bbox
            crop = image[y:y+h, x:x+w]  # Crop the image
            crop = preprocess_image(crop)  # Apply preprocessing to the crop

            # Display preprocessed crop
            display_image('Preprocessed Crop', crop)
            
            # Convert back to PIL Image for further transforms if needed
            crop = Image.fromarray(crop).convert('RGB')
            
            # Display RGB crop
            # display_image('RGB Crop', np.array(crop))

            if self.transform:
                crop = self.transform(crop)

            images.append(crop)
            labels.append(int(target))  # Ensure labels are integers

        return torch.stack(images), torch.tensor(labels)

def main(json_path, data_dir):
    print("Starting main...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processor = PapyrusDataProcessor(json_path, data_dir)
    train_data, validation_data, test_data = processor.split_data()
    train_bboxs, train_labels, train_image_dirs = zip(*train_data)
    validation_bboxs, validation_labels, validation_image_dirs = zip(*validation_data)
    test_bboxs, test_labels, test_image_dirs = zip(*test_data)
    train_dataset = CustomDataset(train_image_dirs, train_bboxs, train_labels, transform=transform)
    val_dataset = CustomDataset(validation_image_dirs, validation_bboxs, validation_labels, transform=transform)
    test_dataset = CustomDataset(test_image_dirs, test_bboxs, test_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Test iteration over the training data
    print("Iterating over the training data...")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}:")
        for j, img in enumerate(images):
            display_image(f'Image {j} in Batch {i}', img.permute(1, 2, 0).numpy())  # Adjust for Tensor image
        if i >= 2:  # Display a few batches only
            break

if __name__ == '__main__':
    json_path = 'Data/HomerCompTrainingReadCoco.json'
    data_dir = 'Data/HomerCompTraining'
    main(json_path, data_dir)
