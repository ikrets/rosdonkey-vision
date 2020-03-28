import unittest
import numpy as np
from pathlib import Path

from dataset_utils.transform import UndistortBirdeye, UndistortBirdeyeParameters

def expand(img, target_height_ratio):
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    expanded = np.zeros(
        (int(target_height_ratio * img.shape[1]), img.shape[1], img.shape[2]),
        dtype=img.dtype,
    )
    expanded[-img.shape[0] :, :, :] = img

    return np.squeeze(expanded)

class TransformTestCase(unittest.TestCase):
    def test_repeated_transform(self):
        with Path('data/undistort_birdeye_params.json').open('r') as fp:
            parameters = UndistortBirdeyeParameters.from_json(fp)

        transform = UndistortBirdeye(
            input_shape=(320, 240), target_shape=(96, 64), parameters=parameters
        )

        first_img = np.random.randint(255, size=(240, 320, 3)).astype(np.uint8)
        first_transformed = transform(expand(first_img, target_height_ratio=240 / 320))

        second_img = np.random.randint(2, size=(240, 320)).astype(np.uint8)
        second_transformed = transform(
            expand(second_img, target_height_ratio=240 / 320)
        )

        self.assertFalse(np.array_equal(first_transformed, second_transformed))


if __name__ == '__main__':
    unittest.main()
