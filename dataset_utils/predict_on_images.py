import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

from dataset_utils.transform import UndistortBirdeye, UndistortBirdeyeParameters
from convert_to_edgetpu import SegmentationEngine
from models import visualize_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_shape', type=int, nargs=2, required=True)
    parser.add_argument('--output_shape', type=int, nargs=2, required=True)
    parser.add_argument('--transform_configuration', type=str, required=True)
    parser.add_argument('--tflite_model', type=str, required=True)
    parser.add_argument('image_dir', type=str)
    parser.add_argument('output_dir', type=str)

    args = parser.parse_args()

    with open(args.transform_configuration, 'r') as fp:
        parameters = UndistortBirdeyeParameters.from_json(fp)

    transform = UndistortBirdeye(args.data_shape, args.output_shape, parameters)
    engine = SegmentationEngine(Path(args.tflite_model), args.output_shape)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image in tqdm(list(Path(args.image_dir).glob('*_image.jpg'))):
        transformed_image = transform(np.array(Image.open(image)))
        _, prediction = engine.segment(transformed_image)

        output_name = output_path / f'{image.stem}.png'
        visualization = visualize_preds(
            images=transformed_image[np.newaxis, ...] / 255,
            labels=None,
            preds=prediction[np.newaxis, ...],
        )
        Image.fromarray(visualization).save(output_name)
