import torch
from torchvision import models
from torch import nn

class CustomResNet(nn.Module):
    def __init__(self, pretrained_model, num_classes=10, feature_size=32):
        super(CustomResNet, self).__init__()
        self.features = nn.Sequential(
            # Copy all layers except the original fully connected layer
            *list(pretrained_model.children())[:-1],
            nn.Flatten()
        )
        self.intermediate = nn.Linear(pretrained_model.fc.in_features, feature_size)
        self.classifier = nn.Linear(feature_size, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.intermediate(x)
        logits = self.classifier(x)
        return x, logits  # Return both the logits and the intermediate features

# Initialize the custom model
pretrained_model = models.resnet50(pretrained=True)
custom_model = CustomResNet(pretrained_model, num_classes=10, feature_size=32)
