import cv2
import tensorflow as tf
from pathlib import Path
from typing import Sequence

num_parallel_calls = 8


def get_image_filenames(
    dataset_folder: Path, allowed_tags: Sequence[str]
) -> Sequence[Path]:
    filenames = dataset_folder.glob('images/*.png')
    return [f for f in filenames if any(a in f.name for a in allowed_tags)]


def load(image_filenames: Sequence[Path]) -> tf.data.Dataset:
    mask_filenames = [f.parent.parent / 'masks' / f.name for f in image_filenames]

    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(f) for f in image_filenames], [str(f) for f in mask_filenames])
    )

    dataset = dataset.map(
        lambda img_fn, mask_fn: (tf.io.read_file(img_fn), tf.io.read_file(mask_fn))
    )

    def process(img_string, mask_string):
        img = tf.io.decode_png(img_string, channels=3)
        mask = tf.io.decode_png(mask_string, channels=1)

        img = tf.cast(img, tf.float32) / 255

        return img, mask

    dataset = dataset.map(process, num_parallel_calls)
    return dataset
