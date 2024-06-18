import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer
from torchvision import transforms
from torch.utils.data import DataLoader


from models import get_model
from data_loader import CustomDataset
from data_split import PapyrusDataProcessor


# Define argument parser
parser = argparse.ArgumentParser(description='Train a model on Papyrus dataset.')
parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file with annotations.')
parser.add_argument('--data_dir', type=str, required=True, help='Directory with image data.')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model to train.')
parser.add_argument('--save_path', type=str, required=True, help='Directory to save the result')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
args = parser.parse_args()

NUM_CLASSES = 227


def main():
    # Apply some transormations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Split data to train, validation and test
    processor = PapyrusDataProcessor(args.json_path, args.data_dir)
    train_data, validation_data, test_data = processor.split_data()

    train_bboxs, train_labels, train_image_dirs = zip(*train_data)
    validation_bboxs, validation_labels, validation_image_dirs = zip(*validation_data)
    test_bboxs, test_labels, test_image_dirs = zip(*test_data)

    train_dataset = CustomDataset(train_image_dirs, args.data_dir, train_bboxs, train_labels, transform=transform)
    val_dataset = CustomDataset(validation_image_dirs, args.data_dir, validation_bboxs, validation_labels, transform=transform)
    test_dataset = CustomDataset(test_image_dirs, args.data_dir, test_bboxs, test_labels, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = get_model(args.model_name, NUM_CLASSES)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, device, criterion, optimizer)

    print('Start training...')
    trainer.train(train_loader, args.num_epochs, val_loader)
    trainer.test(test_loader)
    print('Finished Training and Testing')
    torch.save(model.state_dict(), os.path.join(args.save_path, f'{args.model_name}.pth'))


if __name__ == '__main__':
    main()
