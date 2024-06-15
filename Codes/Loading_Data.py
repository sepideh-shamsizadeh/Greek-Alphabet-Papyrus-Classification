import os
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile

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

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        try:
            image_info = self.data['images'][idx]
            image_id = image_info['id']
            image_path = os.path.join(self.images_dir, image_info['file_name'][2:])
            image = Image.open(image_path).convert("RGB")
            annotations = self.image_id_to_annotations[image_id]

            boxes = [ann['bbox'] for ann in annotations]
            labels = [ann['category_id'] for ann in annotations]

            if self.transform:
                image = self.transform(image)

            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }

            return image, target['labels'], target['boxes']
        except IndexError:
            print(f"Error: Index {idx} is out of range.")
            raise
        except KeyError as e:
            print(f"Error: Key {e} not found in the annotations.")
            raise
        except FileNotFoundError:
            print(f"Error: Image file at {image_path} was not found.")
            raise
