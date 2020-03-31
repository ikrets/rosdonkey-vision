import argparse
from pathlib import Path
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import segmentation_models as sm
from edgetpu.basic.basic_engine import BasicEngine
from time import time
import subprocess

from dataset import get_image_filenames, load


class SegmentationEngine(BasicEngine):
    def __init__(self, model_path):
        BasicEngine.__init__(self, model_path)

    def segment(self, img):
        input_tensor_shape = self.get_input_tensor_shape()
        _, height, width, _ = input_tensor_shape

        input_tensor = np.asarray(img).flatten()
        latency, result = self.run_inference(input_tensor)
        result = result.reshape((height, width, -1))

        return latency, result


def convert_evaluate_fold(fold_dir):
    train_filenames = pd.read_csv(
        fold_dir / 'train_filenames.txt', squeeze=True, header=None
    )
    if (fold_dir / 'val_filenames.txt').exists():
        val_filenames = pd.read_csv(
            fold_dir / 'val_filenames.txt', squeeze=True, header=None
        )
    else:
        val_filenames = None

    train_dataset = load(map(Path, train_filenames)).batch(1)
    sess = tf.keras.backend.get_session()
    item = tf.compat.v1.make_one_shot_iterator(train_dataset).get_next()
    train_images = (sess.run(item)[0] for _ in range(len(train_filenames)))

    converter = tf.lite.TFLiteConverter.from_keras_model_file(fold_dir / 'model.hdf5')
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = train_images
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    tflite_quant_model = converter.convert()

    with (fold_dir / 'converted_model.tflite').open('wb') as fp:
        fp.write(tflite_quant_model)
    subprocess.run(
        executable='edgetpu_compiler',
        args=['converted_model.tflite'],
        shell=True,
        pwd=fold_dir,
    )

    if val_filenames:
        segmentation_engine = SegmentationEngine(
            fold_dir / 'converted_model_edgetpu.tflite'
        )
        val_dataset = load(map(Path, val_filenames)).batch(1)
        item = tf.compat.v1.make_one_shot_iterator(val_dataset).get_next()
        latencies = []
        ious = []
        for _ in range(len(val_filenames)):
            image, label = sess.run(item)
            latency, pred = segmentation_engine.segment(image)
            latencies.append(latency)
            ious.append(sm.metrics.iou_score(label, pred))

        return np.mean(latencies), np.mean(ious)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    folds = Path(args.model_dir).glob('fold_*')
    for fold in folds:
        latency, iou = convert_evaluate_fold(fold)
        with (Path(args.output) / 'evaluation' / fold.name).open('w') as fp:
            json.dump({'latency': float(latency), 'iou': float(iou)}, fp)

    convert_evaluate_fold(Path(args.model_dir) / 'full_train')
