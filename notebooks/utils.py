import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage


S2_BANDS = {
    "B1": { "resolution": 60, "description": "Coastal aerosol"},
    "B2": { "resolution": 10, "description": "Blue"},
    "B3": { "resolution": 10, "description": "Green"},
    "B4": { "resolution": 10, "description": "Red"},
    "B5": { "resolution": 20, "description": "Red Edge 1"},
    "B6": { "resolution": 20, "description": "Red Edge 2"},
    "B7": { "resolution": 20, "description": "Red Edge 3"},
    "B8": { "resolution": 10, "description": "Near-Infrared"},
    "B8A": { "resolution": 20, "description": "Near-Infrared narrow"},
    "B9": { "resolution": 60, "description": "Water vapor"},
    "B10": { "resolution": 60, "description": "Shortwave-Infrared cirrus"},
    "B11": { "resolution": 20, "description": "Shortwave Infrared 1"},
    "B12": { "resolution": 20, "description": "Shortwave-Infrared 2"},
}


def composition(band1, band2, band3):
    """
    Stack arrays representing three optical bands of an image
    and rescale intensities to the image's min and max reflectance
    values, ready for visualization.
    """
    im_comp = np.dstack([band1 / band1.max(), band2 / band2.max(), band3 / band3.max()])

    for i in range(3):
        v_min, v_max = np.percentile(im_comp[:, :, i], (1, 98))
        # Adjust levels to images' min and max reflectance values
        im_comp[:, :, i] = skimage.exposure.rescale_intensity(
            im_comp[:, :, i], in_range=(v_min, v_max)
        )
    return im_comp


def image_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Create a grayscale copy of an image. The band dimension
    is preserved so that color can still be used when
    visualizing the image.
    """
    gray = np.zeros(image.shape)
    gray_band = skimage.color.rgb2gray(image)

    for i in range(image.shape[-1]):
        gray[:, :, i] = gray_band

    return gray


def mask_pixels_rgb(
    image: np.ndarray,
    px_mask: pd.DataFrame,
    colour: str,
) -> None:
    assert image.shape[-1] == 3, "Image must have 3 bands"

    band_vals = {
        "blue": (0, 0, 1),
        "red": (1, 0, 0),
        "yellow": (1, 1, 0),
        "green": (0, 1, 0),
    }.get(colour, (0, 0, 1))

    for i in range(3):
        image[px_mask.x, px_mask.y, i] = band_vals[i]


def add_points_to_image(
    image: np.ndarray, points: pd.DataFrame, colour: str
) -> None:
    mask_pixels_rgb(image, points, colour)
    mask_pixels_rgb(image, points.assign(x=points.x + 1), colour)
    mask_pixels_rgb(image, points.assign(x=points.x - 1), colour)
    mask_pixels_rgb(image, points.assign(y=points.y + 1), colour)
    mask_pixels_rgb(image, points.assign(y=points.y - 1), colour)


def imshow(predictions, figsize=(20, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(predictions)


def plot_classes_legend(ax) -> None:
    # Plot the legend
    ax.axis("off")
    ax.imshow(
        np.vstack(
            (
                np.zeros((10, 100)),
                np.ones((10, 100)),
                2 * np.ones((10, 100)),
                3 * np.ones((10, 100)),
            )
        )
    )
    ax.text(40, 6, "Water", fontsize=15, color="white")
    ax.text(40, 16, "Artificial", fontsize=15, color="black")
    ax.text(40, 26, "Low vegetation", fontsize=15, color="black")
    ax.text(40, 36, "Tree cover", fontsize=15, color="black")