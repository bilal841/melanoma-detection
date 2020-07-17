import os
import torch
import pretrainedmodels
import albumentations
import efficientnet_pytorch

import numpy as np
import pandas as pd
import torch.nn as nn

from torch.nn import functional as F
from sklearn import metrics
from wtfml.data_loaders.image import ClassificationLoader
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine


class SEResNext50_32x4d(nn.Module):
    """Class representing the Squeeze-and-Excitation network.

    See https://arxiv.org/abs/1611.05431 for details.
    """

    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](
            pretrained=pretrained
        )
        self.out = nn.Linear(
            2048, 1
        )  # note that we only have to rank the images using AUC

    def forward(self, image, targets):
        batch_size, channel, height, width = image.shape
        x = self.model.features(image)
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
            "efficientnet-b0"
        )
        self.base_model._fc = nn.Linear(in_features=1280, out_features=1, bias=True)

    def forward(self, image, targets):
        out = self.base_model(image)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))
        return out, loss


def train(fold):
    """Train a specific fold of the dataset

    Args:
        fold (integer): integer representing the fold
    """
    # import the training dataset that includes folds
    training_data_path = "../data/train_resized"
    df = pd.read_csv("../data/train_with_folds.csv")

    # set the parameters
    device = "cuda"
    epochs = 1
    train_bs = 32
    val_bs = 16

    # create separate dataframes for the types of data based on their associated fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True)

    # use the mean and std of ImageNet for normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # normalize and augment the data
    train_augmentation = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )
    val_augmentation = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )

    # define the training images and their class
    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values

    # define the validation images and their class
    val_images = df_val.image_name.values.tolist()
    val_images = [os.path.join(training_data_path, i + ".jpg") for i in val_images]
    val_targets = df_val.target.values

    # define the train and validation dataset
    train_dataset = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_augmentation,
    )
    val_dataset = ClassificationLoader(
        image_paths=val_images,
        targets=val_targets,
        resize=None,
        augmentations=val_augmentation,
    )

    # define the training and validation dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_bs, shuffle=False, num_workers=4
    )

    # define the model
    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    # define the optimizer, scheduler, and incorporate early stopping
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, mode="max"
    )
    es = EarlyStopping(patience=3, mode="max")

    # train the model
    for epoch in range(epochs):
        training_loss = Engine.train(train_loader, model, optimizer, device, fp16=False)
        predictions, val_loss = Engine.evaluate(val_loader, model, device)
        predictions = np.vstack((predictions)).ravel()

        # compute the metric we focus on
        auc = metrics.roc_auc_score(val_targets, predictions)

        # take a step
        scheduler.step(auc)
        es(auc, model, model_path=f"../models/model_fold_{fold}.bin")

        print(f"finished epoch: {epoch}, obtained auc: {auc}")

        if es.early_stop:
            print("performing early stopping . . .")
            break


def predict(fold):
    # import the dataset
    test_data_path = "../data/test_resized/"
    df = pd.read_csv("../data/test.csv")

    # set the parameters
    device = "cuda"
    model_path = f"../models/model_fold_{fold}.bin"

    # use the mean and std of ImageNet
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # normalize the images
    aug = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )

    # get the test images and their associated target values
    images = df.image_name.values.tolist()
    images = [os.path.join(test_data_path, i + ".jpg") for i in images]
    targets = np.zeros(len(images))

    # define the test dataset and dataloader
    test_dataset = ClassificationLoader(
        image_paths=images, targets=targets, resize=None, augmentations=aug,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    # use the trained model for predictions
    model = SEResNext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    predictions = Engine.predict(test_loader, model, device=device)
    predictions = np.vstack((predictions)).ravel()

    return predictions


if __name__ == "__main__":
    train(0)
    predict(0)
