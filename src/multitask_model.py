import torch.nn as nn
from torchvision import models


class MultitaskModel(nn.Module):
    """
    ResNet[]-based model for multitask classification.
    Pretrained with the following parameters:
    Input space: "RGB"
    input range: [0, 1]
    input dimensions: [3, 224, 224]
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    """

    def __init__(self, num_categories=15, num_conditions=5):
        super(MultitaskModel, self).__init__()

        # use pretrained ResNet model for feature extraction
        model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

        num_fc_in_features = model.fc.in_features
        self.fc_categories = nn.Linear(num_fc_in_features, num_categories)
        self.fc_conditions = nn.Linear(num_fc_in_features, num_conditions)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        out_categories = self.fc_categories(x)
        out_conditions = self.fc_conditions(x)

        return out_categories, out_conditions
