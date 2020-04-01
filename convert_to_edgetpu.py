import argparse
from pathlib import Path
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from edgetpu.basic.basic_engine import BasicEngine
from time import time
import subprocess

from dataset import get_image_filenames, load
from models import binary_iou_score


class SegmentationEngine(BasicEngine):
    def __init__(self, model_path):
        BasicEngine.__init__(self, str(model_path))

    def segment(self, img):
        input_tensor_shape = self.get_input_tensor_shape()
        _, height, width, _ = input_tensor_shape

        input_tensor = np.asarray(img).flatten()
        latency, result = self.run_inference(input_tensor)
        result = result.reshape((height, width, -1))

        return latency, result


def convert_evaluate_fold(fold_dir, output_dir):
    train_filenames = pd.read_csv(
        fold_dir / 'train_filenames.txt', squeeze=True, header=None
    )
    if (fold_dir / 'val_filenames.txt').exists():
        val_filenames = pd.read_csv(
            fold_dir / 'val_filenames.txt', squeeze=True, header=None
        )
    else:
        val_filenames = None

    train_dataset = load([Path(f) for f in train_filenames]).batch(1)
    sess = tf.keras.backend.get_session()
    item = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()
    images = [sess.run(item)[0] for _ in range(len(train_filenames))]

    def train_images():
        for image in images:
            yield [image]

    converter = tf.lite.TFLiteConverter.from_keras_model_file(
        Path(fold_dir) / 'model.hdf5'
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = train_images
    tflite_quant_model = converter.convert()

    (output_dir / fold_dir.name).mkdir(parents=True, exist_ok=True)

    with (output_dir / fold_dir.name / 'converted_model.tflite').open('wb') as fp:
        fp.write(tflite_quant_model)
        fp.flush()

    subprocess.run(
        ['edgetpu_compiler', 'converted_model.tflite'], cwd=output_dir / fold_dir.name,
    )

    if val_filenames is not None:
        sess = tf.keras.backend.get_session()
        segmentation_engine = SegmentationEngine(
            output_dir / fold_dir.name / 'converted_model_edgetpu.tflite'
        )
        val_dataset = load([Path(f) for f in val_filenames]).batch(1)
        item = tf.compat.v1.data.make_one_shot_iterator(val_dataset).get_next()

        latencies = []
        labels = []
        preds = []
        for _ in range(len(val_filenames)):
            image, label = sess.run(item)
            image = (image * 255).astype(np.uint8)
            latency, pred = segmentation_engine.segment(image)
            labels.append(label[0])
            # important to copy, as pred will be overwritten on every inference!
            preds.append(pred.copy())
            latencies.append(latency)

        iou = sess.run(
            binary_iou_score(
                np.stack(labels, axis=0).astype(np.float32), np.stack(preds, axis=0)
            )
        )

        return np.mean(latencies), iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    folds = Path(args.model_dir).glob('fold_*')
    fold_ious = []
    for fold in folds:
        latency, iou = convert_evaluate_fold(fold, Path(args.output_dir))
        with (Path(args.output_dir) / fold.name / 'converted_metric.json').open(
            'w'
        ) as fp:
            json.dump(
                {'latency': float(latency), 'iou_score': float(iou)}, fp, indent=4,
            )

        fold_ious.append(iou)

    with (Path(args.output_dir) / 'converted_metric.json').open('w') as fp:
        json.dump({'iou_score': float(np.mean(fold_ious))}, fp, indent=4)

    convert_evaluate_fold(Path(args.model_dir) / 'full_train', Path(args.output_dir))
