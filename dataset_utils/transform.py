import numpy as np
import cv2
import json
import dataclasses
from typing import Tuple, TextIO


@dataclasses.dataclass
class UndistortBirdeyeParameters:
    target_img_size: Tuple[int, int]
    K: np.array
    D: np.array
    birdeye_src: np.array
    birdeye_dst: np.array

    @staticmethod
    def from_json(fp: TextIO) -> 'UndistortBirdeyeParameters':
        params = json.load(fp)
        for k, v in params.items():
            params[k] = np.array(v)

        return UndistortBirdeyeParameters(**params)


class UndistortBirdeye:
    def __init__(
        self,
        input_shape: Tuple[int, int],
        target_shape: Tuple[int, int],
        parameters: UndistortBirdeyeParameters,
    ) -> None:
        src = (
            parameters.birdeye_src
            * float(input_shape[0])
            / parameters.target_img_size[0]
        )
        dst = parameters.birdeye_dst * float(target_shape[1]) / 96
        desired_size = target_shape
        dst[:, 0] += float(desired_size[1]) / 2

        K = parameters.K * float(input_shape[0]) / parameters.target_img_size[0]
        K[2, 2] = 1.0

        nK = K.copy()
        nK[0, 0] /= 2
        nK[1, 1] /= 2

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K.astype(np.float32),
            parameters.D.astype(np.float32),
            np.eye(3),
            nK,
            tuple(input_shape),
            cv2.CV_16SC2,
        )

        perspective_transform = cv2.getPerspectiveTransform(
            src.astype(np.float32), dst.astype(np.float32)
        )
        iTM = cv2.invert(perspective_transform)[1]
        self.trans_map_x = np.empty(dtype=np.float32, shape=desired_size)
        self.trans_map_y = np.empty(dtype=np.float32, shape=desired_size)

        for y in range(self.trans_map_x.shape[0]):
            fy = float(y)
            for x in range(self.trans_map_y.shape[1]):
                fx = float(x)
                w = iTM[2, 0] * fx + iTM[2, 1] * fy + iTM[2, 2]
                w = 1.0 / w if w != 0.0 else 0.0

                self.trans_map_x[y, x] = (
                    iTM[0, 0] * fx + iTM[0, 1] * fy + iTM[0, 2]
                ) * w
                self.trans_map_y[y, x] = (
                    iTM[1, 0] * fx + iTM[1, 1] * fy + iTM[1, 2]
                ) * w

        self.trans_map_x, self.trans_map_y = cv2.convertMaps(
            self.trans_map_x, self.trans_map_y, cv2.CV_16SC2
        )

        # without this distinction the call will not convert [height, width] images
        # after [height, width, 3] properly
        self.undistorted_img_3ch = np.zeros(
            (input_shape[1], input_shape[0], 3), dtype=np.uint8
        )
        self.undistorted_img_1ch = np.zeros(
            (input_shape[1], input_shape[0]), dtype=np.uint8
        )

    def __call__(self, img, dst=None, undistort_interpolation=cv2.INTER_NEAREST):
        if not (len(img.shape) == 3 and img.shape[-1] == 3) and not len(img.shape) == 2:
            raise ValueError('Image must either be [H, W, 3] or [H, W]!')
        if img.dtype != np.uint8:
            raise ValueError('Image must be uint8')

        undistorted_img = (
            self.undistorted_img_3ch
            if len(img.shape) == 3
            else self.undistorted_img_1ch
        )

        img = cv2.remap(
            img,
            self.map1,
            self.map2,
            interpolation=undistort_interpolation,
            dst=undistorted_img,
        )
        if dst is None:
            return cv2.remap(
                undistorted_img,
                self.trans_map_x,
                self.trans_map_y,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            cv2.remap(
                undistorted_img,
                self.trans_map_x,
                self.trans_map_y,
                dst=dst,
                interpolation=cv2.INTER_LINEAR,
            )

        return img


def overlay_img_with_mask(img: np.array, mask: np.array, alpha: float = 0.5):
    overlayed = np.array(img)
    mask = np.asarray(mask)

    overlayed[mask == 1] = (
        alpha * np.array([0, 255, 0]) + (1 - alpha) * overlayed[mask == 1]
    ).round()

    return overlayed
