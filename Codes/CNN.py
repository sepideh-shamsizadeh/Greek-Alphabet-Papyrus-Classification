import torch
import torch.nn as nn
import torch.optim as optim
from data_split import PapyrusDataProcessor
from data_loader import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import random


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(32*32*32, 1000)
        self.fc2 = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class Trainer:
    def __init__(self, model, device, criterion, optimizer):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader, num_epochs, val_loader=None):
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                for img, lbl in zip(images, labels):
                    outputs = self.model(img)
                    loss = self.criterion(outputs, lbl)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(lbl.cpu().numpy())

                    if (i+1) % 10 == 0:
                        accuracy = accuracy_score(all_labels, all_preds)
                        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}, Accuracy: {accuracy:.4f}')
                        running_loss = 0.0
                        all_preds = []
                        all_labels = []

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = accuracy_score(all_labels, all_preds)
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}')
            
            # Validation phase
            if val_loader is not None:
                print("Running validation...")
                val_loss, val_accuracy = self.validate(val_loader)
                print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    def validate(self, val_loader):
        self.model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                for img, lbl in zip(images, labels):
                    outputs = self.model(img)
                    loss = self.criterion(outputs, lbl)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(lbl.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        return val_loss, val_accuracy

    def test(self, test_loader):
        self.model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                for img, lbl in zip(images, labels):
                    outputs = self.model(img)
                    _, predicted = torch.max(outputs.data, 1)
                    test_preds.extend(predicted.cpu().numpy())
                    test_labels.extend(lbl.cpu().numpy())

        test_accuracy = accuracy_score(test_labels, test_preds)
        print(f'Test Accuracy: {test_accuracy:.4f}')
        return test_accuracy


class BinarizeTransform:
    def __call__(self, img):
        img = img.convert('L')  # Convert to grayscale
        np_img = np.array(img)
        np_img = (np_img > 128).astype(np.uint8) * 255  # Simple threshold
        return Image.fromarray(np_img)


class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


def main():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        BinarizeTransform(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.1)
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
    model = SimpleCNN(num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    trainer = Trainer(model, device, criterion, optimizer)

    print('Start training...')
    trainer.train(train_loader, num_epochs, val_loader)
    trainer.test(test_loader)
    print('Finished Training and Testing')

if __name__ == '__main__':
    main()
