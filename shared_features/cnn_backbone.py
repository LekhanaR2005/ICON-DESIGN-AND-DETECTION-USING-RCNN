import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True)

        # Use only convolution layers (feature extractor)
        self.features = vgg.features

    def forward(self, x):

        feature_maps = self.features(x)

        return feature_maps