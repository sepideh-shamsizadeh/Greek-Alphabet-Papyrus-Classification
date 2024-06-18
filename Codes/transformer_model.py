import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from data_split import PapyrusDataProcessor
from data_loader import CustomDataset
from timm.models.vision_transformer import Block
from torchvision.models import resnet50, ResNet50_Weights


class SimpleCNNTransformer(nn.Module):
    def __init__(self, num_classes, num_transformer_layers=6, dim=256, num_heads=8, mlp_ratio=4.0):
        super(SimpleCNNTransformer, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Compute the output dimension of the CNN
        dummy_input = torch.randn(1, 3, 224, 224)
        cnn_output = self.cnn(dummy_input)
        self.transformer_input_dim = cnn_output.numel() // cnn_output.shape[0]

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.transformer_input_dim, dim)

        self.transformer = nn.Sequential(
            *[Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(num_transformer_layers)]
        )

        self.fc2 = nn.Linear(dim, 1000)
        self.fc3 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = x.unsqueeze(1)  # Adding sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)  # Removing sequence dimension
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ResNetTransformer(nn.Module):
    def __init__(self, num_classes, num_transformer_layers=6, dim=512, num_heads=8, mlp_ratio=4.0):
        super(ResNetTransformer, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.transformer_input_dim = self.resnet.fc.in_features  # Get in_features before replacing fc
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.transformer_input_dim, dim)

        self.transformer = nn.Sequential(
            *[Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(num_transformer_layers)]
        )

        self.fc2 = nn.Linear(dim, 1000)
        self.fc3 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = x.unsqueeze(1)  # Adding sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)  # Removing sequence dimension
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def vit_transformer(num_classes):
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return model