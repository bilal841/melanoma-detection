import os
import glob

from tqdm import tqdm
from joblib import delayed, Parallel
from PIL import ImageFile, Image

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


# resize training data
input_folder = "../data/train/"
output_folder = "../data/train_resized/"
images = glob.glob(os.path.join(input_folder, "*.jpg"))  # create a list of all images
Parallel(n_jobs=10)(
    delayed(resize_img)(i, output_folder, (128, 128)) for i in tqdm(images)
)

# resize test data
input_folder = "../data/test/"
output_folder = "../data/test_resized/"
images = glob.glob(os.path.join(input_folder, "*.jpg"))  # create a list of all images
Parallel(n_jobs=10)(
    delayed(resize_img)(i, output_folder, (128, 128)) for i in tqdm(images)
)
