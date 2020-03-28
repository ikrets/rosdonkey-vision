import numpy as np
import requests
from PIL import Image
import os
from skimage.draw import polygon
import cv2
from io import BytesIO
import argparse
from tqdm import tqdm
import json
from pathlib import Path
import subprocess

from transform import overlay_img_with_mask

parser = argparse.ArgumentParser(
    description='Extract labeled images from the semantic segmentation editor.'
)
parser.add_argument(
    '--crop_size',
    type=int,
    nargs=2,
    required=True,
    metavar=('HEIGHT', 'WIDTH'),
    help='crop size, measuring from the bottom left corner of the image',
)
parser.add_argument(
    '--image_size',
    type=int,
    nargs=2,
    required=True,
    metavar=('HEIGHT', 'WIDTH'),
    help='images with different aspect ratio other than specified will not be extracted',
)
parser.add_argument(
    'images_folder',
    type=str,
    help='same as images-folder in the semantic segmentation editor',
)
parser.add_argument('output_folder', type=str, help='folder to write results into')

args = parser.parse_args()

base_dir = args.images_folder
output_dir = args.output_folder

crop = (
    args.image_size[1] - args.crop_size[1],
    args.image_size[0] - args.crop_size[0],
    args.image_size[1],
    args.image_size[0],
)

examples = []
class_appearances = [0, 0]

for n in ['images', 'masks', 'images_with_masks']:
    os.makedirs(os.path.join(output_dir, n), exist_ok=True)

with (Path(output_dir) / 'parameters.json').open('w') as fp:
    parameters = dict(vars(args))
    parameters['commit'] = (
        subprocess.run('git rev-parse HEAD', shell=True, stdout=subprocess.PIPE)
        .stdout.decode('ascii')
        .strip()
    )
    json.dump(parameters, fp, indent=4)

for item in tqdm(
    requests.get('http://localhost:3000/api/listing').json(),
    desc='Processing labeled images',
):
    item['folder'] = item['folder'][1:]

    filename = os.path.join(base_dir, item['folder'], item['file'])
    if not os.path.exists(filename):
        tqdm.write(f'Missing {filename}')
        continue

    img = Image.open(filename)
    original_img_size = img.size
    if img.size[0] / args.image_size[1] != img.size[1] / args.image_size[0]:
        tqdm.write(
            f'{item["file"]} has different aspect ratio: {img.size[1]}x{img.size[0]}'
        )
        continue

    adjusted_crop = tuple(int(c * img.size[0] / args.image_size[1]) for c in crop)

    img = img.crop(adjusted_crop)
    image_bytes = BytesIO()
    img.save(image_bytes, format='PNG')
    output_filename = '{}_{}'.format(item['folder'].replace('/', '-'), item['file'])

    labels = requests.get('http://localhost:3000/api/json' + item['url']).json()
    mask = np.zeros(original_img_size[::-1], dtype=np.uint8)

    class1 = 0
    for label in labels['objects']:
        xs = []
        ys = []

        for point in label['polygon']:
            xs.append(point['x'])
            ys.append(point['y'])

        p = polygon(xs, ys)
        p_inside_img = (
            (p[0] >= 0)
            & (p[0] < original_img_size[0])
            & (p[1] >= 0)
            & (p[1] < original_img_size[1])
        )
        mask[p[1][p_inside_img], p[0][p_inside_img]] = 1

    mask = mask[
        adjusted_crop[1] : adjusted_crop[3], adjusted_crop[0] : adjusted_crop[2]
    ]
    _, mask_bytes = cv2.imencode('.png', mask)

    img.save(os.path.join(output_dir, 'images', output_filename))

    cv2.imwrite(
        os.path.join(output_dir, 'images_with_masks', output_filename),
        overlay_img_with_mask(img, mask),
    )
    cv2.imwrite(
        os.path.join(
            output_dir, 'masks', os.path.splitext(output_filename)[0] + '.png'
        ),
        mask,
    )
