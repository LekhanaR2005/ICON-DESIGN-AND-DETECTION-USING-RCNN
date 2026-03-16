import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureInterface(nn.Module):

    def __init__(self, output_size=(7,7)):
        super(FeatureInterface, self).__init__()

        # Adaptive pooling ensures fixed size output
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, feature_maps):

        # Standardize spatial size
        shared_features = self.pool(feature_maps)

        return shared_features