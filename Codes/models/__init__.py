
from .cnn import SimpleCNN, resnet
from .transformers import ResNetTransformer, SimpleCNNTransformer, vit_transformer

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
