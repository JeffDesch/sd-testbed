from os import path
import numpy as np
from PIL import Image


def test_interops():
    image_name_first = "geometry1.png"
    image_name_last = "geometry2.png"
    output_base_name = "geometry_interp"
    test_steps = 30
    path_first_input = path.join("images", "input", image_name_first)
    path_last_input = path.join("images", "input", image_name_last)
    path_output = path.join("images", "output")

    start_img = load_image(path_first_input)
    end_img = load_image(path_last_input)

    for idx, output_image in zip(range(test_steps), pixel_interp(start_img, end_img, test_steps)):
        save_image(output_image, path_output, (output_base_name + str(idx) + ".png"))


def load_image(filepath: str) -> np.array:
    return np.array(Image.open(filepath))


def save_image(img: np.ndarray, filepath: str, filename: str):
    pil_img = Image.fromarray(img)
    save_path = path.join(filepath, filename)
    pil_img.save(save_path)


def pixel_interp(start_img: np.array, end_img: np.array, steps: int = 30):
    """ Generates steps-1 in-between frames transitioning from one image to the other """

    for frame in range(steps):
        coeff = frame / steps
        yield (start_img * (1-coeff) + end_img * coeff).astype(start_img.dtype)

