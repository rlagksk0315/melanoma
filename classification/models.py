import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, mobilenet_v2, MobileNet_V2_Weights, densenet121, DenseNet121_Weights

def get_efficientnet(num_classes=1, pretrained=True):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model

def get_mobilenet(num_classes=1, pretrained=True):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model

def get_densenet(num_classes=1, pretrained=True):
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model