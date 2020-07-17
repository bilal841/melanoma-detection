import pretrainedmodels
import efficientnet_pytorch

import numpy as np
import pandas as pd
import torch.nn as nn

from torch.nn import functional as F


class SEResNext50_32x4d(nn.Module):
    """Class representing the Squeeze-and-Excitation network.

    See https://arxiv.org/abs/1611.05431 for details.
    """

    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"](pretrained=pretrained)
        # take care of the sigmoid before calculating the loss
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape

        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(batch_size, -1)

        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))
        return out, loss


class EfficientNet(nn.Module):
    """Class representing the Efficient Net model

    See https://github.com/lukemelas/EfficientNet-PyTorch for details.
    """

    def __init__(self):
        super(EfficientNet, self).__init__()
        self.base_model = efficientnet_pytorch.EfficientNet.from_pretrained(
            'efficientnet-b0')
        self.base_model._fc = nn.Linear(
            in_features=1280,
            out_features=1,
            bias=True
        )

    def forward(self, image, targets):
        out = self.base_model(image)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))
        return out, loss
