import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from data_split import PapyrusDataProcessor
from data_loader import CustomDataset, check_images
from trainer import Trainer
from cnn_model import SimpleCNN, resnet

def main():
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(),  # Add horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
    processor = PapyrusDataProcessor('Data/HomerCompTrainingReadCoco.json', 'Data/HomerCompTraining')
    train_data, validation_data, test_data = processor.split_data()

    train_bboxs, train_labels, train_image_dirs = zip(*train_data)
    validation_bboxs, validation_labels, validation_image_dirs = zip(*validation_data)
    test_bboxs, test_labels, test_image_dirs = zip(*test_data)


    train_dataset = CustomDataset(train_image_dirs, train_bboxs, train_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_dataset = CustomDataset(validation_image_dirs, validation_bboxs, validation_labels, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    test_dataset = CustomDataset(test_image_dirs, test_bboxs, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    num_classes = 227
    # model = SimpleCNN(num_classes)
    model = resnet(num_classes)

    
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
    torch.save(model.state_dict(), 'Data/')




if __name__ == '__main__':
    main()