import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights, VGG16_Weights, EfficientNet_B0_Weights
from Preprocessing import load_data, create_dataloaders
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to get the model
def get_model(model_name, num_classes):
    if model_name == 'resnet':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg':
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Invalid model name")
    
    return model.to(device)

# Load the data
json_path = 'Data/HomerCompTrainingReadCoco.json'
data, category_id_to_name, unique_ids = load_data(json_path)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(data, category_id_to_name, unique_ids)

# Initialize the model, criterion and optimizer
num_classes = len(category_id_to_name)
model_name = 'resnet'  # Change to 'vgg' or 'efficientnet' for other models
model = get_model(model_name, num_classes)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Validation Loss: {val_loss/len(val_loader)}")

print("Training complete")

# Test the model
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')

# Prediction function
def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def predict_single_image(model, image_path, transform, category_id_to_name, threshold=0.5):
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path, transform)
    image = image.to(device)

    # Put the model in evaluation mode and disable gradient computation
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    
    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

    # Get predicted labels based on threshold
    predicted_labels = [category_id_to_name[idx] for idx, prob in enumerate(probabilities) if prob >= threshold]

    return predicted_labels, probabilities

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Sample image path for prediction
sample_image_path = 'path_to_your_sample_image.jpg'

# Load your trained model weights (if not already loaded)
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model = model.to(device)

# Predict
predicted_labels, probabilities = predict_single_image(model, sample_image_path, transform, category_id_to_name)

# Display the results
print(f"Predicted labels: {predicted_labels}")
print(f"Probabilities: {probabilities}")

# Optionally, display the image
image = Image.open(sample_image_path).convert("RGB")
plt.imshow(image)
plt.title(f"Predicted labels: {', '.join(predicted_labels)}")
plt.axis('off')
plt.show()
