"""Compute image descriptors."""

import cv2
import numpy as np
from PIL import Image
from skimage import color
from skimage.util import shape
from sklearn import preprocessing

from . import utils

__all__ = [
    "compute_image_descriptor",
    "compute_image_descriptor_from_filepath",
]


def compute_image_descriptor(img_rgb, kernels, response_bins_per_axis, num_color_bins):
    """
    Compute a GIST descriptor for an RGB image array.

    See the `background <https://bit.ly/2KlCICO>`_ example notebook for more details.

    Parameters
    ----------
    img_rgb : array-like
        The image in RGB format, i.e., in a 3-D array.
    kernels : list-like
        List of kernel 2-D arrays that correspond to the filter bank.
    response_bins_per_axis : int
        Number of spatial bins per axis into which the responses to the filter bank will
        be aggregated. For example, a value of 2 will aggregate the responses into the
        four quadrants of the image (i.e., 2x2, 2 bins in each axis of the image).
    num_color_bins : int
        Number of color bins per axis of the L*a*b color space with which the joint
        color histogram will be computed.

    Returns
    -------
    img_descr : array-like
        Vector representing GIST descriptor of `img_rgb`.
    """
    # gist descriptor
    num_blocks = response_bins_per_axis**2
    gist_descr = np.zeros(len(kernels) * num_blocks)
    img_gray = color.rgb2gray(img_rgb)
    block_shape = tuple(size // response_bins_per_axis for size in img_gray.shape)
    divides_evenly = True
    for size in img_gray.shape:
        if size % response_bins_per_axis != 0:
            divides_evenly = False
            break

    if not divides_evenly:
        # TODO: warn?
        target_height, target_width = (
            size * response_bins_per_axis for size in block_shape
        )
        img_gray = np.array(
            Image.fromarray(img_gray).resize((target_width, target_height))
        )

    for i, kernel in enumerate(kernels):
        filter_response = cv2.filter2D(img_gray, ddepth=-1, kernel=kernel)
        response_bins = shape.view_as_blocks(filter_response, block_shape)
        bin_sum = response_bins.sum(axis=(2, 3)).flatten()
        gist_descr[i * num_blocks : (i + 1) * num_blocks] = bin_sum

    # color descriptor
    img_lab = color.rgb2lab(img_rgb)
    img_lab_dn = img_lab.reshape(img_lab.shape[0] * img_lab.shape[1], img_lab.shape[2])
    H, _ = np.histogramdd(img_lab_dn, bins=num_color_bins)
    color_descr = H.flatten()

    # normalize the gist and color descriptors to the l1 norm and concatenate them
    img_descr = np.concatenate(
        [
            preprocessing.normalize(row.reshape(1, -1), norm="l1").flatten()
            for row in [gist_descr, color_descr]
        ]
    )

    return img_descr


def compute_image_descriptor_from_filepath(
    img_filepath, kernels, response_bins_per_axis, num_color_bins
):
    """
    Compute a GIST descriptor for RGB image file.

    See the `background <https://bit.ly/2KlCICO>`_ example notebook for more details.

    Parameters
    ----------
    img_filepath : str, file object or pathlib.Path object
        Path to a file, URI, file object opened in binary ('rb') mode, or a Path object
        representing the image for which a GIST descriptor will be computed. The value
        will be passed to `rasterio.open`.
    kernels : list-like
        List of kernel 2-D arrays that correspond to the filter bank.
    response_bins_per_axis : int
        Number of spatial bins per axis into which the responses to the filter bank will
        be aggregated. For example, a value of 2 will aggregate the responses into the
        four quadrants of the image (i.e., 2x2, 2 bins in each axis of the image).
    num_color_bins : int
        Number of color bins per axis of the L*a*b color space with which the joint
        color histogram will be computed.

    Returns
    -------
    img_descr : array-like
        Vector representing GIST descriptor of `img_rgb`.
    """
    img_rgb = utils.img_rgb_from_filepath(img_filepath)
    return compute_image_descriptor(
        img_rgb, kernels, response_bins_per_axis, num_color_bins
    )
