import cv2
import numpy as np
from glob import glob
import tensorflow as tf
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from io import BytesIO
import argparse
from pathlib import Path
from tqdm import trange

from transform import (
    UndistortBirdeye,
    UndistortBirdeyeParameters,
    overlay_img_with_mask,
)


def prepare_file_dataset(
    folders, dataset_name, image_transformation, mask_transformation
):
    images = []
    for folder in folders:
        images.extend(glob(os.path.join(folder, 'images/*.jpg')))

    masks = [i.replace('images', 'masks').replace('jpg', 'png') for i in images]
    os.makedirs(dataset_name, exist_ok=True)

    dataset_path = Path(dataset_name)
    for subfolder in ['images', 'masks', 'images_with_masks']:
        (dataset_path / subfolder).mkdir(parents=True, exist_ok=True)

    for i in trange(len(images), desc='Processing images'):
        folder, name = os.path.split(images[i])

        img = np.array(Image.open(images[i]))
        img[:, -2:, :] = 0

        img = image_transformation(img)
        img = Image.fromarray(img)
        img.save(dataset_path / 'images' / name)

        mask = np.array(Image.open(masks[i]))
        mask[:, -2:] = 0
        mask = mask_transformation(mask)
        mask = Image.fromarray(mask)
        mask.save(dataset_path / 'masks' / name)

        Image.fromarray(overlay_img_with_mask(img, mask)).save(
            dataset_path / 'images_with_masks' / name
        )


def expand(img, target_height_ratio):
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    expanded = np.zeros(
        (int(target_height_ratio * img.shape[1]), img.shape[1], img.shape[2]),
        dtype=img.dtype,
    )
    expanded[-img.shape[0] :, :, :] = img

    return np.squeeze(expanded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Create a undistorted, birdeye view dataset from labeled images.'
    )
    parser.add_argument(
        '--undistort_birdeye_parameters',
        type=str,
        required=True,
        help='file specifying the parameters of undistort and birdeye view transformations',
    )
    parser.add_argument(
        '--resize_shape',
        type=int,
        nargs=2,
        metavar=('HEIGHT', 'WIDTH'),
        required=True,
        help='Before applying the transformations, resize the images to specified size.',
    )
    parser.add_argument(
        '--target_shape',
        type=int,
        nargs=2,
        metavar=('HEIGHT', 'WIDTH'),
        required=True,
        help='The target size of the birdeye view images.',
    )
    parser.add_argument(
        'extracted_dataset',
        type=str,
        help='location of the extracted images with labels',
    )
    parser.add_argument(
        'output', type=str, help='the resulting dataset will be written here'
    )
    args = parser.parse_args()

    with Path(args.undistort_birdeye_parameters).open('r') as fp:
        undistort_birdeye_params = UndistortBirdeyeParameters.from_json(fp)

    def resize(img):
        resized = cv2.resize(
            img, (args.resize_shape[1], args.resize_shape[0]), cv2.INTER_LINEAR
        )
        return resized

    undistort_birdeyeview = UndistortBirdeye(
        input_shape=(args.resize_shape[1], args.resize_shape[0]),
        target_shape=(args.target_shape[1], args.target_shape[0]),
        parameters=undistort_birdeye_params,
    )

    height_ratio = args.resize_shape[0] / args.resize_shape[1]
    transformation = lambda img: undistort_birdeyeview(
        resize(expand(img, height_ratio)), undistort_interpolation=cv2.INTER_LINEAR
    )

    prepare_file_dataset(
        [args.extracted_dataset],
        args.output,
        image_transformation=transformation,
        mask_transformation=transformation,
    )
