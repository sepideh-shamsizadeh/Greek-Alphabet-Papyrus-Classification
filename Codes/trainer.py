import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from data_split import PapyrusDataProcessor
from data_loader import CustomDataset, check_images


class Trainer:
    def __init__(self, model, device, criterion, optimizer):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader, num_epochs, val_loader=None):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            for i, (images_batch, labels_batch) in enumerate(train_loader):
                # Flatten the batch of images and labels
                images = torch.cat([img.unsqueeze(0) for images in images_batch for img in images], dim=0).to(self.device)
                labels = torch.cat([lbl for lbl in labels_batch], dim=0).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if (i+1) % 10 == 0:
                    accuracy = accuracy_score(all_labels, all_preds)
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}, Accuracy: {accuracy:.4f}')
                    running_loss = 0.0
                    all_preds = []
                    all_labels = []

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = accuracy_score(all_labels, all_preds)
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}')
            
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
            for images_batch, labels_batch in val_loader:
                images = torch.cat([img.unsqueeze(0) for images in images_batch for img in images], dim=0).to(self.device)
                labels = torch.cat([lbl for lbl in labels_batch], dim=0).to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        return val_loss, val_accuracy

    def test(self, test_loader):
        self.model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for images_batch, labels_batch in test_loader:
                images = torch.cat([img.unsqueeze(0) for images in images_batch for img in images], dim=0).to(self.device)
                labels = torch.cat([lbl for lbl in labels_batch], dim=0).to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_accuracy = accuracy_score(test_labels, test_preds)
        print(f'Test Accuracy: {test_accuracy:.4f}')
        return test_accuracy