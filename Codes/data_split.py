import os
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the JSON data
with open('Data/HomerCompTrainingReadCoco.json') as f:
    data = json.load(f)


category_id_to_name = {item['id']: item['name'] for item in data['categories']}
unique_ids = set(item['id'] for item in data['images'])

def get_image_data(image_id, data):
    # Initialize lists to store the results
    bounding_boxes = []
    category_ids = []
    image_file_name = None

    # Filter annotations by the given image_id
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            bounding_boxes.append(annotation['bbox'])
            category_ids.append(annotation['category_id'])

    # Get the file name for the image
    for image in data['images']:
        if image['id'] == image_id:
            image_file_name = image['file_name']
            break


    return bounding_boxes, image_file_name, category_ids




def plot_image_with_bbox(image, bbox, labels):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    # Draw the bounding boxes
    for box, label in zip(bbox, labels):
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(box[0], box[1] - 10, label, color='white', backgroundcolor='red', fontsize=5, weight='bold')
    plt.show()


bboxs = []
labels = []
image_dirs = []
for image_id in unique_ids:
    # Example usage
    bounding_boxes, image_file_name, category_ids = get_image_data(image_id, data)

    # Get category names
    categories_name = [category_id_to_name[id] for id in category_ids]
    bboxs.append(bounding_boxes)
    image_dirs.append(image_file_name[2:])
    labels.append(categories_name)
    # # Open the image
    # image_path = os.path.join('Data/HomerCompTraining', image_file_name[2:])
    # image = Image.open(image_path).convert("RGB")
    # plot_image_with_bbox(image, bounding_boxes, categories_name)

print(len(labels), len(image_dirs), len(bboxs))


# Combine the lists into a single list of tuples
data = list(zip(bboxs, labels, image_dirs))

# First split: train and temp (test + validation)
train_data, temp_data = train_test_split(data, test_size=0.30, random_state=42)

# Second split: validation and test
validation_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)

# Separate the data back into individual lists
train_bboxs, train_labels, train_image_dirs = zip(*train_data)
test_bboxs, test_labels, test_image_dirs = zip(*test_data)
validation_bboxs, validation_labels, validation_image_dirs = zip(*validation_data)

# Convert to lists
train_bboxs = list(train_bboxs)
train_labels = list(train_labels)
train_image_dirs = list(train_image_dirs)

test_bboxs = list(test_bboxs)
test_labels = list(test_labels)
test_image_dirs = list(test_image_dirs)

validation_bboxs = list(validation_bboxs)
validation_labels = list(validation_labels)
validation_image_dirs = list(validation_image_dirs)

# # Print the split data
# print("Train Data:", train_bboxs, train_labels, train_image_dirs)
# print("Test Data:", test_bboxs, test_labels, test_image_dirs)
# print("Validation Data:", validation_bboxs, validation_labels, validation_image_dirs)