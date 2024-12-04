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

class resnet101(nn.Module):
    def __init__(self, num_class=100, feature_size=64):
        super(resnet101, self).__init__()
        self.model = models.resnet101(pretrained=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 512) # 2048
        self.fc2 = nn.Linear(512, feature_size)
        self.classifier = nn.Linear(feature_size, num_class)


    def forward(self, x):
        batch = x.size(0)
        # Upsample
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = self.fc1(x)
        features = self.fc2(x)
        logits = self.classifier(features)
        return features, logits

class resnet50(nn.Module):
    def __init__(self, num_class=100, feature_size=64):
        super(resnet50, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 512) # 2048
        self.fc2 = nn.Linear(512, feature_size)
        self.classifier = nn.Linear(feature_size, num_class)


    def forward(self, x):
        batch = x.size(0)
        # Upsample
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = self.fc1(x)
        features = self.fc2(x)
        logits = self.classifier(features)
        return features, logits

if __name__ == '__main__':
    print("Test")
    # Test resnet 101
    model = resnet50(10, 64)
    # model = resnet101(10, 64)
    eg = torch.ones((100, 3, 32, 32))
    features, logits = model(eg)
    print(features.shape, logits.shape)
