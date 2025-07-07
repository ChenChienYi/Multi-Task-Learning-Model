import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class AttrTaskCelebAModel(nn.Module):
    def __init__(self, num_attributes):
        super(AttrTaskCelebAModel, self).__init__()

        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
          param.requires_grad = False

        self.backbone.fc = nn.Identity()

        num_features = 512
        self.attribute_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_attributes)
        )

    def forward(self, x):
        features = self.backbone(x)
        attribute_logits = self.attribute_head(features)

        return attribute_logits

class LandmarkTaskCelebAModel(nn.Module):
    def __init__(self, num_landmarks=10):
        super(LandmarkTaskCelebAModel, self).__init__()

        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
          param.requires_grad = False

        self.backbone.fc = nn.Identity()

        num_features = 512
        self.landmark_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_landmarks)
        )

    def forward(self, x):
        features = self.backbone(x)
        landmarks_pred = self.landmark_head(features)

        return landmarks_pred

class MultiTaskCelebAModel(nn.Module):
    def __init__(self, num_attributes, num_landmarks=10):
        super(MultiTaskCelebAModel, self).__init__()

        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
          param.requires_grad = False

        self.backbone.fc = nn.Identity()

        num_features = 512
        self.attribute_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_attributes)
        )

        self.landmark_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_landmarks)
        )

    def forward(self, x):
        features = self.backbone(x)
        attribute_logits = self.attribute_head(features)
        landmarks_pred = self.landmark_head(features)

        return attribute_logits, landmarks_pred