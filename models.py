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
        # for param in self.features.parameters():
        #     param.requires_grad = False
        self.fc1 = nn.Linear(pretrained_model.fc.in_features, 512) # 2048
        self.fc2 = nn.Linear(512, feature_size)
        self.classifier = nn.Linear(feature_size, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.classifier(x)
        return x, logits  # Return both the logits and the intermediate features

