import glob
from os import path

import numpy as np

from . import filters, image_descriptor, settings

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__all__ = ['TrainingSelector']


class TrainingSelector(object):
    def __init__(self, tile_filepaths=None, tile_dir=None,
                 tile_filename_pattern='*.tif'):
        super(TrainingSelector, self).__init__()

        if tile_filepaths is None:
            tile_filepaths = glob.glob(
                path.join(tile_dir, tile_filename_pattern))

        self.tile_filepaths = tile_filepaths

        # TODO: boolean arg for equal tile size (and pass `block_shape` to
        # `get_gist_descriptor`)?

    def get_descr_feature_matrix(self, gabor_frequencies=None,
                                 gabor_num_orientations=None,
                                 response_bins_per_axis=4, num_color_bins=8):
        """     
        Parameters
        ----------
        gabor_frequencies : list-like

        gabor_num_orientations : int or list-like

        response_bins_per_axis : int

        num_color_bins : int

        Returns
        -------
        descr_feature_matrix
        """

        # preprocess keyword arguments
        if gabor_frequencies is None:
            gabor_frequencies = settings.GIST_GABOR_DEFAULT_FREQUENCIES

        if gabor_num_orientations is None:
            gabor_num_orientations = settings.GIST_GABOR_DEFAULT_ORIENTATIONS
        elif not isinstance(gabor_num_orientations, (list, tuple)):
            gabor_num_orientations = tuple(gabor_num_orientations
                                           for _ in gabor_frequencies)

        kernels = filters.get_gabor_filter_bank(
            frequencies=gabor_frequencies,
            num_orientations=gabor_num_orientations)

        num_blocks = response_bins_per_axis**2

        feature_row_list = []
        iterator = self.tile_filepaths
        if tqdm is not None:
            iterator = tqdm(iterator)

        for tile_filepath in iterator:
            feature_row_list.append(
                image_descriptor.ImageDescriptor(
                    tile_filepath).get_image_descriptor(
                        kernels, response_bins_per_axis, num_blocks,
                        num_color_bins))

        descr_feature_matrix = np.vstack(feature_row_list)

        return descr_feature_matrix
