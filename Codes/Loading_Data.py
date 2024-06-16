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

        # Load and process all images
        self._process_all_images()

    def _process_all_images(self):
        for image_info in self.images_info:
            image_id = image_info['id']
            image_path = os.path.join(self.images_dir, image_info['file_name'][2:])
            
            # Load image and convert to RGB
            image = Image.open(image_path).convert("RGB")
            
            annotations = self.image_id_to_annotations[image_id]
            for ann in annotations:
                try:
                    cropped_img = crop_box(image, ann['bbox'])
                    if self.transform:
                        cropped_img = self.transform(cropped_img)
                    self.img_list.append(cropped_img)
                    self.labels.append(ann['category_id'])
                except Exception as e:
                    print(f"Error cropping image ID {image_id} with bbox {ann['bbox']}: {e}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.img_list[idx], self.labels[idx]

# Example usage
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = GreekCharactersDataset(json_path='Data/HomerCompTrainingReadCoco.json', images_dir='Data/HomerCompTraining', transform=transform)
print(len(dataset))

# Display the first cropped image and its label
# img, label = dataset[0]
# show_image(img)
# print(label)
