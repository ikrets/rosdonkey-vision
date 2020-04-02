import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import segmentation_models as sm
import math
import json

from dataset import get_image_filenames, load
from models import unet, MeanIoUFromBinary, VisualizePredsCallback
from augment import color_jitter

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--crop_size', type=int, required=True)
parser.add_argument('--allowed_tags', type=str, nargs='+', required=True)
parser.add_argument('--num_folds', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--decoder_filters_base', type=int, required=True)
parser.add_argument('--num_stages', type=int, choices=[2, 3, 4, 5], required=True)
parser.add_argument('--alpha', choices=[0.35, 0.5, 0.75, 1], type=float)
parser.add_argument('--lr_values', type=float, nargs='+', required=True)
parser.add_argument('--lr_boundaries', type=float, nargs='+', required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--visualize_preds_period', type=int, default=100)
parser.add_argument('--augment_strength', type=float, required=True)
parser.add_argument('--bn_momentum', type=float, default=0.9)
parser.add_argument('--l2_regularization', type=float, default=0.0)
parser.add_argument('--freeze_encoder', action='store_true')
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

shuffle_buffer_size = 500
num_parallel_calls = 8


def piecewise_linear(values, boundaries):
    def schedule(epoch):
        for i in range(len(boundaries)):
            if epoch < boundaries[i]:
                return values[i]
        return values[len(boundaries)]

    return schedule


@tf.function
def augment(img, mask, crop_size, strength):
    random_crop_h = tf.random.uniform(
        [1], maxval=tf.shape(img)[0] - crop_size, dtype=tf.int32
    )[0]
    random_crop_w = tf.random.uniform(
        [1], maxval=tf.shape(img)[1] - crop_size, dtype=tf.int32
    )[0]

    img = tf.image.crop_to_bounding_box(
        img, random_crop_h, random_crop_w, crop_size, crop_size
    )
    mask = tf.image.crop_to_bounding_box(
        mask, random_crop_h, random_crop_w, crop_size, crop_size
    )

    hflip = tf.random.uniform([1], maxval=2, dtype=tf.int32)[0]
    if hflip == 1:
        img, mask = tf.image.flip_left_right(img), tf.image.flip_left_right(mask)

    rot90 = tf.random.uniform([1], maxval=4, dtype=tf.int32)[0]
    img = tf.image.rot90(img, k=rot90)
    mask = tf.image.rot90(mask, k=rot90)

    img = color_jitter(img, strength=strength, random_order=True)

    return img, mask


def train_and_eval(log_dir, train_filenames, val_filenames=None):
    tf.keras.backend.clear_session()
    log_dir.mkdir(parents=True, exist_ok=True)

    with (log_dir / 'train_filenames.txt').open('w') as fp:
        fp.write('\n'.join(str(f) for f in train_filenames))

    if val_filenames is not None:
        with (log_dir / 'val_filenames.txt').open('w') as fp:
            fp.write('\n'.join(str(f) for f in val_filenames))

    train_dataset = (
        load(train_filenames)
        .shuffle(shuffle_buffer_size)
        .map(
            lambda img, mask: augment(img, mask, args.crop_size, args.augment_strength),
            num_parallel_calls,
        )
        .repeat()
        .batch(args.batch_size)
        .prefetch(1)
    )
    train_steps = math.ceil(len(train_filenames) / args.batch_size)
    if val_filenames is not None:
        val_dataset = load(val_filenames).batch(args.batch_size).repeat().prefetch(1)
        val_steps = math.ceil(len(val_filenames) / args.batch_size)
    else:
        val_dataset = None
        val_steps = None

    schedule = piecewise_linear(boundaries=args.lr_boundaries, values=args.lr_values)
    callbacks = [
        VisualizePredsCallback(
            log_dir=str(log_dir / 'train_images'),
            data=load(train_filenames).shuffle(shuffle_buffer_size).batch(16).take(1),
            period=args.visualize_preds_period,
        ),
        tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1),
        tf.keras.callbacks.TensorBoard(str(log_dir), profile_batch=0),
    ]

    if val_filenames is not None:
        callbacks.append(
            VisualizePredsCallback(
                log_dir=str(log_dir / 'val_images'),
                data=load(val_filenames).shuffle(shuffle_buffer_size).batch(16).take(1),
                period=args.visualize_preds_period,
            )
        )

    adam = tf.keras.optimizers.Adam(0.0)
    decoder_filters = [
        args.decoder_filters_base * (2 ** i) for i in range(0, args.num_stages)
    ][::-1]
    model = unet(
        input_shape=[None, None, 3],
        decoder_filters=decoder_filters,
        alpha=args.alpha,
        bn_momentum=args.bn_momentum,
        l2_regularization=args.l2_regularization,
        freeze_encoder=args.freeze_encoder,
    )

    model.compile(
        adam, loss=sm.losses.binary_focal_loss, metrics=[MeanIoUFromBinary()],
    )
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        epochs=args.epochs,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    model.save(log_dir / 'model.hdf5', include_optimizer=False)
    if val_filenames is not None:
        with (log_dir / 'metric.json').open('w') as fp:
            json.dump(
                {'val_mean_io_u': float(history.history['val_mean_io_u'][-1])},
                fp,
                indent=4,
            )

    return history


kf = KFold(n_splits=args.num_folds)
filenames = get_image_filenames(Path(args.data), allowed_tags=args.allowed_tags)
filenames = np.array(filenames)

ious = []
for i, train_val_indices in enumerate(kf.split(filenames)):
    train_filenames = filenames[train_val_indices[0]]
    val_filenames = filenames[train_val_indices[1]]

    history = train_and_eval(
        Path(args.output_dir) / f'fold_{i}', train_filenames, val_filenames
    )
    ious.append(history.history['val_mean_io_u'][-1])

with (Path(args.output_dir) / 'metric.json').open('w') as fp:
    json.dump({'val_mean_io_u': float(np.mean(ious))}, fp)

train_and_eval(Path(args.output_dir) / 'full_train', filenames)
