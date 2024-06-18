import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from data_split import PapyrusDataProcessor
from data_loader import CustomDataset
from trainer import Trainer
from cnn_model import SimpleCNN, resnet
from transformer_model import ResNetTransformer, SimpleCNNTransformer, vit_transformer

def get_model(model_name, num_classes):
    if model_name == 'resnet_transformer':
        return ResNetTransformer(num_classes)
    elif model_name == 'simple_cnn_transformer':
        return SimpleCNNTransformer(num_classes)
    elif model_name == 'simple_cnn':
        return SimpleCNN(num_classes)
    elif model_name == 'resnet':
        return resnet(num_classes)
    elif model_name == "vit_transformer":
        return vit_transformer(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



def main(json_path, data_dir, model_name):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet typically uses 224x224 input size
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
    
    num_classes = 227
    model = get_model(model_name, num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100

    trainer = Trainer(model, device, criterion, optimizer)

    print('Start training...')
    trainer.train(train_loader, num_epochs, val_loader)
    trainer.test(test_loader)
    print('Finished Training and Testing')
    torch.save(model.state_dict(), f'Data/model_{model_name}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on Papyrus dataset.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file with annotations.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with image data.')
    parser.add_argument('--model_name', type=str, required=True, choices=['resnet_transformer', 'simple_cnn_transformer', 'simple_cnn', 'resnet', 'vit_transformer'], help='Name of the model to train.')
    args = parser.parse_args()

    main(args.json_path, args.data_dir, args.model_name)
