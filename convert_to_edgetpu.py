import argparse
from pathlib import Path
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from edgetpu.basic.basic_engine import BasicEngine
from time import time
import subprocess
from PIL import Image

from dataset import get_image_filenames, load
from models import MeanIoUFromBinary, visualize_preds


class SegmentationEngine(BasicEngine):
    def __init__(self, model_path, output_shape):
        BasicEngine.__init__(self, str(model_path))
        self.output_shape = output_shape

    def segment(self, img):
        input_tensor_shape = self.get_input_tensor_shape()
        _, height, width, _ = input_tensor_shape

        input_tensor = np.asarray(img).flatten()
        latency, result = self.run_inference(input_tensor)
        result = result.reshape(self.output_shape)

        return latency, result


def convert_evaluate_fold(fold_dir, output_dir, input_shape):
    train_filenames = pd.read_csv(
        fold_dir / 'train_filenames.txt', squeeze=True, header=None
    )
    if (fold_dir / 'val_filenames.txt').exists():
        val_filenames = pd.read_csv(
            fold_dir / 'val_filenames.txt', squeeze=True, header=None
        )
    else:
        val_filenames = None

    train_dataset = load([Path(f) for f in train_filenames]).shuffle(200).batch(1)

    def train_images():
        for image, _ in train_dataset:
            yield [image]

    model = tf.keras.models.load_model(Path(fold_dir) / 'model.hdf5')
    output_shape = model.predict(np.random.rand(1, *input_shape, 3)).shape[1:]
    converter = tf.lite.TFLiteConverter.from_keras_model_file(
        Path(fold_dir) / 'model.hdf5', input_shapes={'input_1': [1, *input_shape, 3]}
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
        segmentation_engine = SegmentationEngine(
            output_dir / fold_dir.name / 'converted_model_edgetpu.tflite', output_shape
        )
        val_dataset = load([Path(f) for f in val_filenames]).batch(1)

        latencies = []
        images = []
        labels = []
        preds = []

        mean_iou = MeanIoUFromBinary()

        for image, label in val_dataset:
            images.append(image.numpy())
            latency, pred = segmentation_engine.segment(
                (image.numpy() * 255).astype(np.uint8)
            )
            labels.append(label.numpy())
            # important to copy, as pred will be overwritten on every inference!
            preds.append(pred.copy())
            latencies.append(latency)

            mean_iou.update_state(label, pred.copy()[tf.newaxis, ...])

        result_vis = visualize_preds(
            images=np.concatenate(images, axis=0),
            labels=np.concatenate(labels, axis=0),
            preds=np.stack(preds, axis=0),
        )
        Image.fromarray(result_vis).save(output_dir / fold_dir.name / 'val_preds.png')

        return np.mean(latencies), mean_iou.result().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shape', type=int, nargs=2, required=True)
    parser.add_argument('model_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    tf.enable_eager_execution()

    folds = Path(args.model_dir).glob('fold_*')
    fold_ious = []
    fold_latencies = []
    for fold in folds:
        latency, iou = convert_evaluate_fold(
            fold, Path(args.output_dir), args.input_shape
        )
        with (Path(args.output_dir) / fold.name / 'converted_metric.json').open(
            'w'
        ) as fp:
            json.dump(
                {'latency': float(latency), 'val_mean_io_u': float(iou)}, fp, indent=4,
            )

        fold_ious.append(iou)
        fold_latencies.append(latency)

    with (Path(args.output_dir) / 'converted_metric.json').open('w') as fp:
        json.dump(
            {
                'val_mean_io_u': float(np.mean(fold_ious)),
                'latency': float(np.mean(fold_latencies)),
            },
            fp,
            indent=4,
        )

    convert_evaluate_fold(
        Path(args.model_dir) / 'full_train', Path(args.output_dir), args.input_shape
    )
