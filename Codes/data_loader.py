import os
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Display the original image
    # display_image('Original', image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # display_image('Gray', gray)
    
    # Contrast and Brightness Adjustment
    alpha = 1.5  # Contrast control
    beta = 50    # Brightness control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Noise Reduction
    blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
    
    # Binarization
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological Transformations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    

    
    # Normalize size
    final = cv2.resize(opening, (128, 128))
    return final

def check_images(image_dirs):
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
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

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
        image = cv2.imread(image_path)
        
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
            # display_image('Preprocessed Crop', crop)
            
            # Convert back to PIL Image for further transforms if needed
            crop = Image.fromarray(crop).convert('RGB')
            
            # Display RGB crop
            # display_image('RGB Crop', np.array(crop))

            if self.transform:
                crop = self.transform(crop)

            images.append(crop)
            labels.append(int(target))  # Ensure labels are integers

        return torch.stack(images), torch.tensor(labels)