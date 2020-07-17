import os
import glob
import torch

import numpy as np

from tqdm import tqdm
from joblib import delayed, Parallel
from PIL import ImageFile, Image


try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_img(img_path, output_folder, resize):
    """Resizes all images in the image path to a specific size.

    Args:
        img_path (string): name of the path containing all images
        output_folder (string): name of the output folder
        resize (tuple): tuple consisting new image shape
    """
    base_name = os.path.basename(img_path)
    output_path = os.path.join(output_folder, base_name)
    img = Image.open(img_path)
    img = img.resize(
        (resize[1], resize[0]), resample=Image.BILINEAR
    )  # resize the images
    img.save(output_path)


class EarlyStopping:
    """Custom EarlyStopping classes from 
    https://github.com/abhishekkrthakur/wtfml/blob/master/wtfml/utils/early_stopping.py 
    """

    def __init__(self, patience=7, mode="max", delta=0.0001, tpu=False):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.tpu = tpu
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.tpu:
                xm.master_print(
                    "EarlyStopping counter: {} out of {}".format(
                        self.counter, self.patience
                    )
                )
            else:
                print(
                    "EarlyStopping counter: {} out of {}".format(
                        self.counter, self.patience
                    )
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.tpu:
                xm.master_print(
                    "Validation score improved ({} --> {}). Saving model!".format(
                        self.val_score, epoch_score
                    )
                )
            else:
                print(
                    "Validation score improved ({} --> {}). Saving model!".format(
                        self.val_score, epoch_score
                    )
                )
            if self.tpu:
                xm.save(model.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # resize training data
    input_folder = "../data/train/"
    output_folder = "../data/train_resized/"
    images = glob.glob(os.path.join(input_folder, "*.jpg"))
    Parallel(n_jobs=10)(
        delayed(resize_img)(i, output_folder, (128, 128)) for i in tqdm(images)
    )

    # resize test data
    input_folder = "../data/test/"
    output_folder = "../data/test_resized/"
    images = glob.glob(os.path.join(input_folder, "*.jpg"))
    Parallel(n_jobs=10)(
        delayed(resize_img)(i, output_folder, (128, 128)) for i in tqdm(images)
    )
