import os
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the JSON data
def load_json_data(json_path):
    try:
        with open(json_path) as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: The file {json_path} is not a valid JSON file.")
        raise

# Extract category mapping
def get_category_mapping(data):
    try:
        category_id_to_name = {item['id']: item['name'] for item in data['categories']}
        name_to_category_id = {v: k for k, v in category_id_to_name.items()}
        return category_id_to_name, name_to_category_id
    except KeyError as e:
        print(f"Error: Key {e} not found in the JSON data.")
        raise

def validate_bbox(bbox, image_size):
    # Ensure bbox is within image bounds
    x, y, w, h = bbox
    image_width, image_height = image_size
    x = max(0, x)
    y = max(0, y)
    w = min(w, image_width - x)
    h = min(h, image_height - y)
    return (x, y, w, h)

def crop_box(image, bbox):
    bbox = validate_bbox(bbox, image.size)
    cropped_image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
    return cropped_image

def show_image(image):
    # Convert tensor to PIL image if needed
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()

class GreekCharactersDataset(Dataset):
    def __init__(self, json_path, images_dir, transform=None):
        self.data = load_json_data(json_path)
        self.images_dir = images_dir
        self.transform = transform
        self.category_id_to_name, self.name_to_category_id = get_category_mapping(self.data)
        
        # Initialize image_id_to_annotations dictionary
        self.image_id_to_annotations = {item['id']: [] for item in self.data['images']}
        for annotation in self.data['annotations']:
            self.image_id_to_annotations[annotation['image_id']].append(annotation)
        
        self.images_info = self.data['images']
        self.img_list = []
        self.labels = []

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        image_id = image_info['id']
        image_path = os.path.join(self.images_dir, image_info['file_name'][2:])
        
        # Load image and convert to RGB
        image = Image.open(image_path).convert("RGB")
        
        # # Debug: Show the original image
        # print(f"Original image - ID: {image_id}")
        # show_image(image)
        
        annotations = self.image_id_to_annotations[image_id]

        
        for ann in annotations:
            try:
                cropped_img = crop_box(image, ann['bbox'])
                self.img_list.append(cropped_img)
                self.labels.append(ann['category_id'])
                # # Debug: Show cropped image before transformation
                # print(f"Cropped image before transformation")
                # show_image(cropped_img)
                # print(ann['category_id'])
            except Exception as e:

                print(f"Error cropping image ID {image_id} with bbox {ann['bbox']}: {e}")

        if self.transform:
            self.img_list = [self.transform(img) for img in self.img_list]

        # # Debug: Show cropped images after transformation
        # for i, img in enumerate(self.img_list):
        #     print(f"Cropped image {i+1} after transformation")
        #     show_image(img)

        return img_list, labels

# Example usage
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = GreekCharactersDataset(json_path='Data/HomerCompTrainingReadCoco.json', images_dir='Data/HomerCompTraining', transform=transform)

# Display the first image and its cropped versions
img_list, labels = dataset[0]
for img,lbl in zip(img_list, labels):
    show_image(img)
    print()
