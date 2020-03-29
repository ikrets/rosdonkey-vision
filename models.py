import tensorflow as tf
import numpy as np
import cv2
import segmentation_models as sm
from pathlib import Path
import math


sm.set_framework('tf.keras')
old_upsampling = tf.keras.layers.UpSampling2D


def upsampling_bilinear(*args, **kwargs):
    kwargs['interpolation'] = 'bilinear'
    return old_upsampling(*args, **kwargs)


tf.keras.layers.UpSampling2D = upsampling_bilinear

num_parallel_calls = 8


def unet(input_shape, decoder_filters, alpha, bn_momentum, l2_regularization):
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

    encoder_features = sm.Backbones.get_feature_layers('mobilenetv2', n=4)
    model = sm.models.unet.build_unet(
        backbone=backbone,
        decoder_block=sm.models.unet.DecoderUpsamplingX2Block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=1,
        activation='sigmoid',
        n_upsample_blocks=len(decoder_filters),
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

        preds = self.model.predict(self.data)
        preds = np.transpose(preds, [1, 0, 2, 3])
        preds = np.reshape(preds, [preds.shape[0], -1])
        cv2.imwrite(
            str(self.log_dir / f'epoch_{epoch}.png'), (preds * 255).astype(np.uint8)
        )


def binary_iou_score(pred, true):
    pred = tf.concat([1 - pred, pred], axis=-1)
    true = tf.concat([1 - true, true], axis=-1)

    return sm.metrics.iou_score(pred, true)
