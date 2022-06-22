
import numpy as np
from PIL import Image, ImageOps

from fastreid.utils.file_io import PathManager

def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.
    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"
    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            image = image[:, :, ::-1]
        if format == "L":
            image = np.expand_dims(image, -1)
        image = Image.fromarray(image)
        return image
