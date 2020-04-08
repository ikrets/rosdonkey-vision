import tensorflow as tf
from simclr.data_util import (
    random_apply,
    color_jitter,
    to_grayscale,
    random_crop_with_resize,
)


# only reason this exists is simclr.data_util.random_color_jitter depends on FLAGS
# and i'm using argparse
def random_color_jitter(image, height, width, p, strength):
    def transform(image):
        image = random_apply(
            lambda X: color_jitter(X, strength=strength), p=0.8, x=image
        )
        return random_apply(to_grayscale, p=0.2, x=image)

    return random_apply(transform, p=p, x=image)
