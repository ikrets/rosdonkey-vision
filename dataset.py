import cv2
import tensorflow as tf
import numpy as np
from pathlib import Path
import re
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


unsupervised_feature_description = {
    'img': tf.io.FixedLenFeature([], tf.string),
    'name': tf.io.FixedLenFeature([], tf.string),
    'tag': tf.io.FixedLenFeature([], tf.string),
}


def load_unsupervised_tfrecords(tfrecord_path: Path, allowed_tags: Sequence[str]):
    tfrecords = [str(f) for f in tfrecord_path.glob('**/*.tfrecord')]
    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset.map(
        lambda proto: tf.io.parse_single_example(
            proto, unsupervised_feature_description
        ),
        num_parallel_calls,
    )

    @tf.function
    def allowed_tag_present(item):
        found = False
        for allowed_tag in allowed_tags:
            if tf.strings.regex_full_match(item['tag'], f'{allowed_tag}.*'):
                found = True

        return found

    dataset = dataset.filter(allowed_tag_present)
    num_examples = dataset.reduce(np.int64(0), lambda x, _: x + 1)
    if tf.executing_eagerly():
        num_examples = num_examples.numpy()
    else:
        sess = tf.compat.v1.keras.backend.get_session()
        num_examples = sess.run(num_examples)

    dataset = dataset.map(
        lambda item: tf.cast(tf.image.decode_png(item['img']), tf.float32) / 255,
        num_parallel_calls,
    )

    return dataset, num_examples
