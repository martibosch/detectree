import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color
from skimage.util import shape
from sklearn import preprocessing

from . import utils

__all__ = [
    'compute_image_descriptor', 'compute_image_descriptor_from_filepath'
]


def compute_image_descriptor(img_rgb, kernels, response_bins_per_axis,
                             num_blocks, num_color_bins):
    # gist descriptor
    gist_descr = np.zeros(len(kernels) * num_blocks)
    img_gray = color.rgb2gray(img_rgb)
    block_shape = tuple(size // response_bins_per_axis
                        for size in img_gray.shape)
    divides_evenly = True
    for size in img_gray.shape:
        if size % response_bins_per_axis != 0:
            divides_evenly = False
            break

    if not divides_evenly:
        # TODO: warn?
        target_height, target_width = (size * response_bins_per_axis
                                       for size in block_shape)
        img_gray = np.array(
            Image.fromarray(img_gray).resize((target_width, target_height)))

    for i, kernel in enumerate(kernels):
        filter_response = ndi.convolve(img_gray, kernel)
        response_bins = shape.view_as_blocks(filter_response, block_shape)
        bin_sum = response_bins.sum(axis=(2, 3)).flatten()
        gist_descr[i * num_blocks:(i + 1) * num_blocks] = bin_sum

    # color descriptor
    img_lab = color.rgb2lab(img_rgb)
    img_lab_dn = img_lab.reshape(img_lab.shape[0] * img_lab.shape[1],
                                 img_lab.shape[2])
    H, _ = np.histogramdd(img_lab_dn, bins=num_color_bins)
    color_descr = H.flatten()

    # normalize the gist and color descriptors to the l1 norm and concatenate
    # them
    img_descr = np.concatenate([
        preprocessing.normalize(row.reshape(1, -1), norm='l1').flatten()
        for row in [gist_descr, color_descr]
    ])

    return img_descr


def compute_image_descriptor_from_filepath(img_filepath, kernels,
                                           response_bins_per_axis, num_blocks,
                                           num_color_bins):
    img_rgb = utils.img_rgb_from_filepath(img_filepath)
    return compute_image_descriptor(img_rgb, kernels, response_bins_per_axis,
                                    num_blocks, num_color_bins)
