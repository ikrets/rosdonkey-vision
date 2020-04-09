import tensorflow as tf
import math
import argparse
import json
from pathlib import Path

from dataset import load_unsupervised_tfrecords
from simclr.objective import add_contrastive_loss
from simclr.data_util import random_crop_with_resize
from augment import random_color_jitter
from utils import save_run_parameters

tfkl = tf.keras.layers

num_parallel_calls = 8


def simclr_mobilenetv2(input_shape, alpha, last_layer, contrastive_head_features):
    mobilenetv2 = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, alpha=alpha, include_top=False, weights=None
    )

    out = tfkl.Flatten()(mobilenetv2.get_layer(name=last_layer).output)
    out = tfkl.Dense(contrastive_head_features, activation=None, use_bias=False)(out)
    out = tfkl.BatchNormalization()(out)
    out = tfkl.ReLU()(out)

    out = tfkl.Dense(contrastive_head_features, activation=None, use_bias=True)(out)

    return tf.keras.Model(inputs=mobilenetv2.input, outputs=out)


def augment(img, height, width, jitter_p, jitter_strength):
    img = tf.image.random_crop(img, size=[height, width, 3])
    img = tf.image.random_flip_left_right(img)
    img = random_color_jitter(img, height, width, jitter_p, jitter_strength)

    return img


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--crop_size', type=int, default=48)
parser.add_argument('--ntxent_temperature', type=float, default=0.5)
parser.add_argument('--jitter_prob', type=float, default=0.5)
parser.add_argument('--jitter_strength', type=float, default=0.5)
parser.add_argument('--shuffle_buffer_size', type=int, default=10000)
parser.add_argument('--allowed_tags', type=str, nargs='+', default=('home', '60fps'))
parser.add_argument('--mobilenet_alpha', type=float, default=0.35)
parser.add_argument('--mobilenet_last_layer', type=str, default='block_11_expand_relu')
parser.add_argument('--contrastive_head_features', type=int, default=128)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

save_run_parameters(Path(args.output_dir), dict(vars(args)))


def contrastive_loss(_, y_pred):
    return add_contrastive_loss(y_pred)[0]


def contrastive_accuracy(_, y_pred):
    _, logits, labels = add_contrastive_loss(y_pred)
    return tf.keras.metrics.categorical_accuracy(labels, logits)


@tf.function
def form_simclr_batch(img_batch, label_batch):
    n = tf.shape(img_batch)[0]

    augmented = tf.TensorArray(tf.float32, size=n * 2)
    for i in range(tf.shape(img_batch)[0]):
        augmented = augmented.write(
            index=i,
            value=augment(
                img_batch[i],
                args.crop_size,
                args.crop_size,
                jitter_p=args.jitter_prob,
                jitter_strength=args.jitter_strength,
            ),
        )
        augmented = augmented.write(
            index=n + i,
            value=augment(
                img_batch[i],
                args.crop_size,
                args.crop_size,
                jitter_p=args.jitter_prob,
                jitter_strength=args.jitter_strength,
            ),
        )

    simclr_image_batch = augmented.stack()
    simclr_label_batch = tf.tile(label_batch, [2])

    return simclr_image_batch, simclr_label_batch


data, num_examples = load_unsupervised_tfrecords(
    Path(args.data), allowed_tags=args.allowed_tags
)
data = (
    data.cache()
    .shuffle(args.shuffle_buffer_size)
    .map(lambda X: (X, 0), num_parallel_calls)
    .batch(args.batch_size, drop_remainder=True)
    .map(form_simclr_batch, num_parallel_calls)
    .repeat()
    .prefetch(2)
)


model = simclr_mobilenetv2(
    [args.crop_size, args.crop_size, 3],
    alpha=args.mobilenet_alpha,
    last_layer=args.mobilenet_last_layer,
    contrastive_head_features=args.contrastive_head_features,
)
steps_per_epoch = num_examples // args.batch_size
schedule = tf.keras.experimental.CosineDecay(
    args.learning_rate, decay_steps=args.epochs * steps_per_epoch
)
adam = tf.keras.optimizers.Adam(schedule)
adam = tf.train.experimental.enable_mixed_precision_graph_rewrite(adam)
model.compile(adam, loss=contrastive_loss, metrics=[contrastive_accuracy])

history = model.fit(data, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
with (Path(args.output_dir) / 'metric.json').open('w') as fp:
    h = history.history
    json.dump(
        {
            'train_contrastive_loss': h['loss'][-1].astype(float),
            'train_contrastive_accuracy': h['contrastive_accuracy'][-1].astype(float),
        },
        fp,
        indent=4,
    )
model.save(Path(args.output_dir) / 'model.hdf5', include_optimizer=False)
