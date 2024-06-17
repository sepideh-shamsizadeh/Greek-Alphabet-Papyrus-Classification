import os
import json
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PapyrusDataProcessor:
    def __init__(self, json_path, image_dir):
        self.json_path = json_path
        self.image_dir = image_dir
        self.data = self.load_json()
        self.category_id_to_name = {item['id']: item['name'] for item in self.data['categories']}
        self.unique_ids = set(item['id'] for item in self.data['images'])
        self.bboxs = []
        self.labels = []
        self.image_dirs = []
        self.process_data()

    def load_json(self):
        with open(self.json_path) as f:
            data = json.load(f)
        return data

    def get_image_data(self, image_id):
        bounding_boxes = []
        category_ids = []
        image_file_name = None

        for annotation in self.data['annotations']:
            if annotation['image_id'] == image_id:
                bounding_boxes.append(annotation['bbox'])
                category_ids.append(annotation['category_id'])

        for image in self.data['images']:
            if image['id'] == image_id:
                image_file_name = image['file_name']
                break

        return bounding_boxes, image_file_name, category_ids

    def process_data(self):
        for image_id in self.unique_ids:
            bounding_boxes, image_file_name, category_ids = self.get_image_data(image_id)
            self.bboxs.append(bounding_boxes)
            self.image_dirs.append(image_file_name[2:])
            self.labels.append(category_ids)
        print(len(self.labels), len(self.image_dirs), len(self.bboxs))

    def split_data(self, test_size=0.15, val_size=0.15, random_state=42):
        data = list(zip(self.bboxs, self.labels, self.image_dirs))
        train_data, temp_data = train_test_split(data, test_size=(test_size + val_size), random_state=random_state)
        validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=random_state)
        return train_data, validation_data, test_data

    def plot_image_with_bbox(self, image, bbox, labels):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        for box, label in zip(bbox, labels):
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(box[0], box[1] - 10, label, color='white', backgroundcolor='red', fontsize=5, weight='bold')
        plt.show()

# Example usage
if __name__ == '__main__':
    processor = PapyrusDataProcessor('Data/HomerCompTrainingReadCoco.json', 'Data/HomerCompTraining')
    train_data, validation_data, test_data = processor.split_data()

    train_bboxs, train_labels, train_image_dirs = zip(*train_data)
    validation_bboxs, validation_labels, validation_image_dirs = zip(*validation_data)
    test_bboxs, test_labels, test_image_dirs = zip(*test_data)

    train_bboxs = list(train_bboxs)
    train_labels = list(train_labels)
    train_image_dirs = list(train_image_dirs)

    validation_bboxs = list(validation_bboxs)
    validation_labels = list(validation_labels)
    validation_image_dirs = list(validation_image_dirs)

    test_bboxs = list(test_bboxs)
    test_labels = list(test_labels)
    test_image_dirs = list(test_image_dirs)

    print('Train:', len(train_labels), 'Validation:', len(validation_labels), 'Test:', len(test_labels))
