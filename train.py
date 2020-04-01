import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import segmentation_models as sm
import math
import json

from dataset import get_image_filenames, load
from models import unet, binary_iou_score, VisualizePredsCallback

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--data_shape', type=int, nargs=2, required=True)
parser.add_argument('--allowed_tags', type=str, nargs='+', required=True)
parser.add_argument('--num_folds', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--decoder_filters_base', type=int, required=True)
parser.add_argument('--alpha', choices=[0.35, 0.5, 0.75, 1], type=float)
parser.add_argument('--lr_values', type=float, nargs='+', required=True)
parser.add_argument('--lr_boundaries', type=float, nargs='+', required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--visualize_preds_period', type=int, default=100)
parser.add_argument('--brightness_delta', type=float, default=0.01)
parser.add_argument('--contrast_delta', type=float, default=0.01)
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


def augment(img, mask, shape, brightness_delta, contrast_delta):
    hflip = tf.random.uniform([1], maxval=2, dtype=tf.int32)[0]

    def flip():
        return tf.image.flip_left_right(img), tf.image.flip_left_right(mask)

    def noop():
        return img, mask

    img, mask = tf.cond(tf.equal(hflip, 1), true_fn=flip, false_fn=noop)

    img = tf.image.random_brightness(img, max_delta=brightness_delta)
    img = tf.image.random_contrast(img, 1 - contrast_delta, 1 + contrast_delta)

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
            lambda img, mask: augment(
                img, mask, args.data_shape, args.brightness_delta, args.contrast_delta
            ),
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
            data=load(train_filenames).shuffle(shuffle_buffer_size).batch(8).take(1),
            period=args.visualize_preds_period,
        ),
        tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1),
        tf.keras.callbacks.TensorBoard(str(log_dir), profile_batch=0),
    ]

    if val_filenames is not None:
        callbacks.append(
            VisualizePredsCallback(
                log_dir=str(log_dir / 'val_images'),
                data=load(val_filenames).shuffle(shuffle_buffer_size).batch(8).take(1),
                period=args.visualize_preds_period,
            )
        )

    adam = tf.keras.optimizers.Adam(0.0)
    adam = tf.train.experimental.enable_mixed_precision_graph_rewrite(adam)
    decoder_filters = [args.decoder_filters_base * (2 ** i) for i in range(4, -1, -1)]
    model = unet(
        input_shape=[*args.data_shape, 3],
        decoder_filters=decoder_filters,
        alpha=args.alpha,
        bn_momentum=args.bn_momentum,
        l2_regularization=args.l2_regularization,
        freeze_encoder=args.freeze_encoder,
    )

    model.compile(adam, loss=sm.losses.binary_focal_loss, metrics=[binary_iou_score])
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
                {'iou_score': float(history.history['val_binary_iou_score'][-1])},
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
    ious.append(history.history['val_binary_iou_score'][-1])

with (Path(args.output_dir) / 'metric.json').open('w') as fp:
    json.dump({'iou_score': float(np.mean(ious))}, fp)

train_and_eval(Path(args.output_dir) / 'full_train', filenames)
