import tensorflow as tf
import numpy as np
from PIL import Image
import segmentation_models as sm
from pathlib import Path
import math

tfkl = tf.keras.layers

sm.set_framework('tf.keras')
old_upsampling = tf.keras.layers.UpSampling2D


def upsampling_bilinear(*args, **kwargs):
    kwargs['interpolation'] = 'bilinear'
    return old_upsampling(*args, **kwargs)


tf.keras.layers.UpSampling2D = upsampling_bilinear

num_parallel_calls = 8


def unet(
    input_shape, decoder_filters, alpha, bn_momentum, l2_regularization, freeze_encoder,
):
    tfk_kwargs = {
        'backend': tf.keras.backend,
        'layers': tf.keras.layers,
        'models': tf.keras.models,
        'utils': tf.keras.utils,
    }
    for k, v in tfk_kwargs.items():
        setattr(sm.models.unet, k, v)

    backbone = sm.Backbones.get_backbone(
        'mobilenetv2',
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        alpha=alpha,
        **tfk_kwargs,
    )
    if freeze_encoder:
        for layer in backbone.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    encoder_features = sm.Backbones.get_feature_layers('mobilenetv2', n=4)
    num_stages = len(decoder_filters)
    if num_stages < 5:
        backbone = tf.keras.Model(
            inputs=backbone.input,
            outputs=backbone.get_layer(name=encoder_features[4 - num_stages]).output,
        )

    model = sm.models.unet.build_unet(
        backbone=backbone,
        decoder_block=sm.models.unet.DecoderUpsamplingX2Block,
        skip_connection_layers=encoder_features[5 - num_stages :],
        decoder_filters=decoder_filters,
        classes=1,
        activation='sigmoid',
        n_upsample_blocks=num_stages,
        use_batchnorm=True,
    )

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = bn_momentum

    # techically this is not needed because set_regularization carries this out
    # keeping it for verbosity
    updated_model = tf.keras.models.model_from_json(model.to_json())
    updated_model.set_weights(model.get_weights())

    updated_model = sm.utils.set_regularization(
        updated_model, kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
    )

    return updated_model


def visualize_preds(images, labels, preds):
    preds = np.transpose(preds > 0.5, [1, 0, 2, 3])
    preds = np.reshape(preds, [preds.shape[0], -1, 1])

    labels = np.transpose(labels, [1, 0, 2, 3])
    labels = np.reshape(labels, [labels.shape[0], -1, 1])

    img_pred = np.concatenate(
        [labels * (1 - preds), preds * labels, preds * (1 - labels)], axis=-1
    )
    img = np.transpose(images, [1, 0, 2, 3])
    img = np.reshape(img, [img.shape[0], -1, 3])

    result_vis = np.concatenate([img, img_pred], axis=0)
    result_vis = (result_vis * 255).astype(np.uint8)

    return result_vis


class VisualizePredsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, data, period, **kwargs):
        super(VisualizePredsCallback, self).__init__(**kwargs)
        self.period = period

        self.data = data
        self.writer = tf.summary.FileWriter(log_dir)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs):
        super(VisualizePredsCallback, self).on_epoch_end(epoch, logs)
        if not epoch or epoch % self.period:
            return

        sess = tf.keras.backend.get_session()
        item = tf.compat.v1.data.make_one_shot_iterator(self.data).get_next()
        images = []
        labels = []
        try:
            while True:
                item_result = sess.run(item)
                images.append(item_result[0])
                # store binary labels
                labels.append(item_result[1])
        except tf.errors.OutOfRangeError:
            pass

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = self.model.predict(images)

        result_vis = visualize_preds(images=images, labels=labels, preds=preds)
        Image.fromarray(result_vis).save(self.log_dir / f'epoch_{epoch}.png')


class MeanIoUFromBinary(tf.keras.metrics.MeanIoU):
    def __init__(self, **kwargs):
        super(MeanIoUFromBinary, self).__init__(
            num_classes=2, name='mean_io_u', **kwargs
        )

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_pred = tf.cast(y_pred > 0.5, tf.int32)
        return super(MeanIoUFromBinary, self).update_state(
            y_true, y_pred, *args, **kwargs
        )
