import numpy as np
from scipy import ndimage as ndi
from skimage import color
from skimage.util import shape
from sklearn import preprocessing

from . import utils

__all__ = ['ImageDescriptor']


class ImageDescriptor(object):
    def __init__(self, img_filepath):
        """
 
        Parameters
        ----------
        img_filepath : 
        """
        super(ImageDescriptor, self).__init__()

        self.img_rgb = utils.img_rgb_from_filepath(img_filepath)

    def get_gist_descriptor(self, kernels, response_bins_per_axis, num_blocks):
        img_gray = color.rgb2gray(self.img_rgb)

        gist_row = np.zeros(len(kernels) * num_blocks)
        block_shape = tuple(size // response_bins_per_axis
                            for size in img_gray.shape)
        for i, kernel in enumerate(kernels):
            filter_response = ndi.convolve(img_gray, kernel)
            response_bins = shape.view_as_blocks(filter_response, block_shape)
            bin_sum = response_bins.sum(axis=(2, 3)).flatten()
            gist_row[i * num_blocks:(i + 1) * num_blocks] = bin_sum

        return gist_row

    def get_color_descriptor(self, num_color_bins):
        """     
        Parameters
        ----------
        num_color_bins : int

        Returns
        -------
        color_row : 
        """

        img_lab = color.rgb2lab(self.img_rgb)
        img_lab_dn = img_lab.reshape(img_lab.shape[0] * img_lab.shape[1],
                                     img_lab.shape[2])
        H, _ = np.histogramdd(img_lab_dn, bins=num_color_bins)
        return H.flatten()

    def get_image_descriptor(self, kernels, response_bins_per_axis, num_blocks,
                             num_color_bins):
        rows = [
            self.get_gist_descriptor(kernels, response_bins_per_axis,
                                     num_blocks),
            self.get_color_descriptor(num_color_bins)
        ]
        feature_row = np.concatenate([
            preprocessing.normalize(row.reshape(1, -1), norm='l1').flatten()
            for row in rows
        ])
        return feature_row
