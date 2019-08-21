import glob
from os import path

import dask
import numpy as np
import rasterio as rio
from dask import diagnostics
from scipy import ndimage as ndi
from skimage import color, measure, morphology
from skimage.filters import rank

from . import filters, settings

__all__ = ['TilePixelFeatures', 'pixel_features_from_tiling', 'build_features']

# to convert to illumination-invariant color space
# https://www.cs.harvard.edu/~sjg/papers/cspace.pdf
B = np.array([[0.9465229, 0.2946927, -0.1313419],
              [-0.1179179, 0.9929960, 0.007371554],
              [0.09230461, -0.04645794, 0.9946464]])

A = np.array([[27.07439, -22.80783, -1.806681],
              [-5.646736, -7.722125, 12.86503],
              [-4.163133, -4.579428, -4.576049]])

# TODO: only one `NUM_CHANNELS` constant?
NUM_RGB_CHANNELS = 3
NUM_LAB_CHANNELS = 3
NUM_XYZ_CHANNELS = 3
NUM_ILL_CHANNELS = 3


class TilePixelFeatures(object):
    def __init__(self, img_rgb):
        """
        Parameters
        ----------
        img_rgb : ndarray
        """
        super(TilePixelFeatures, self).__init__()

        # the third component `_` is actually the number of channels in RGB,
        # which is already defined in the constant `NUM_RGB_CHANNELS`
        self.img_rgb = img_rgb
        self.num_rows, self.num_cols, _ = img_rgb.shape
        self.img_lab = color.rgb2lab(img_rgb)
        self.img_lab_l = self.img_lab[:, :, 0]  # ACHTUNG: this is a view

    def compute_color_features(self, color_features):
        img_lab_vec = self.img_lab.reshape(self.num_rows * self.num_cols,
                                           NUM_LAB_CHANNELS)
        img_xyz_vec = color.rgb2xyz(self.img_rgb).reshape(
            self.num_rows * self.num_cols, NUM_XYZ_CHANNELS)
        img_ill_vec = np.dot(A, np.log(np.dot(B, img_xyz_vec.transpose()) +
                                       1)).transpose()
        # assemble the color features
        # color_features = np.zeros((self.num_rows * self.num_cols,
        #                            NUM_LAB_CHANNELS + NUM_ILL_CHANNELS))
        color_features[:, :NUM_LAB_CHANNELS] = img_lab_vec
        color_features[:, NUM_LAB_CHANNELS:NUM_LAB_CHANNELS +
                       NUM_ILL_CHANNELS] = img_ill_vec
        return color_features

    def compute_texture_features(self, texture_features, sigmas,
                                 num_orientations):
        # texture_features = np.zeros(
        #     (self.num_rows * self.num_cols, num_orientations * len(sigmas)))

        for i, sigma in enumerate(sigmas):
            base_kernel_arr = filters.get_texture_kernel(sigma)
            for j, orientation in enumerate(range(num_orientations)):
                # theta = orientation / num_orientations * np.pi
                theta = orientation * 180 / num_orientations
                oriented_kernel_arr = ndi.interpolation.rotate(
                    base_kernel_arr, theta)
                img_filtered = ndi.convolve(self.img_lab_l,
                                            oriented_kernel_arr)
                img_filtered_vec = img_filtered.flatten()
                texture_features[:, i * num_orientations +
                                 j] = img_filtered_vec

        return texture_features

    def compute_entropy_features(self, entropy_features, base_neighborhood,
                                 scales):
        # entropy_features = np.zeros(
        #     (self.num_rows * self.num_cols, num_neighborhoods))

        entropy_features[:, 0] = rank.entropy(self.img_lab_l.astype(np.uint16),
                                              base_neighborhood).flatten()

        for i, factor in enumerate(scales[1:], start=1):
            img = ndi.zoom(
                measure.block_reduce(self.img_lab_l,
                                     (factor, factor)).astype(np.uint16),
                factor)
            entropy_features[:, i] = rank.entropy(img,
                                                  base_neighborhood).flatten()

        return entropy_features

    def compute_entropy_features_(self, entropy_features, neighborhoods):
        # entropy_features = np.zeros(
        #     (self.num_rows * self.num_cols, len(neighborhoods)))

        for i, neighborhood in enumerate(neighborhoods):
            entropy_features[:, i] = rank.entropy(
                self.img_lab_l.astype(np.uint16), neighborhood).flatten()

        return entropy_features


def pixel_features_from_tiling(tile_filepaths=None, tile_dir=None,
                               tile_filename_pattern=None, sigmas=None,
                               num_orientations=None, neighborhoods=None,
                               min_neighborhood_range=None,
                               num_neighborhoods=None):

    if tile_filepaths is None:
        if tile_filename_pattern is None:
            tile_filename_pattern = settings.TILE_DEFAULT_FILENAME_PATTERN
        tile_filepaths = glob.glob(path.join(tile_dir, tile_filename_pattern))

    # preprocess technical keyword arguments
    # texture features
    if sigmas is None:
        sigmas = settings.GAUSS_DEFAULT_SIGMAS
    if num_orientations is None:
        num_orientations = settings.GAUSS_DEFAULT_NUM_ORIENTATIONS
    # entropy features
    if neighborhoods is None:
        if min_neighborhood_range is None:
            min_neighborhood_range = \
                settings.ENTROPY_DEFAULT_MIN_NEIGHBORHOOD_RANGE
        if num_neighborhoods is None:
            num_neighborhoods = settings.ENTROPY_DEFAULT_NUM_NEIGHBORHOODS
        scales = np.geomspace(1, 2**(num_neighborhoods - 1),
                              num_neighborhoods).astype(int)
        neighborhood = morphology.square(2 * min_neighborhood_range + 1)
    else:
        # TODO: compute entropy features with `neighborhoods` kwarg
        num_neighborhoods = len(neighborhoods)

    # tile_df = pd.DataFrame({'tile_filepath': tile_filepaths})
    # def get_num_pixels(tile_filepath):
    #     with rio.open(tile_filepath) as src:
    #         return src.shape[0] * src.shape[1]
    # tile_df['num_pixels'] = tile_df['tile_filepath'].apply(get_num_pixels)
    # num_tiling_pixels = tile_df['num_pixels'].sum()

    num_color_features = NUM_LAB_CHANNELS + NUM_ILL_CHANNELS
    num_texture_features = num_orientations * len(sigmas)
    num_entropy_features = num_neighborhoods
    num_tile_features = num_color_features + num_texture_features + \
        num_entropy_features
    # self.X = np.zeros((num_tiling_pixels, num_tile_features))

    color_slice = slice(num_color_features)
    texture_end = num_color_features + num_texture_features
    texture_slice = slice(num_color_features, texture_end)
    entropy_slice = slice(texture_end, texture_end + num_entropy_features)

    def get_tile_features(tile_filepath):
        with rio.open(tile_filepath) as src:
            num_pixels = src.shape[0] * src.shape[1]
            tpf = TilePixelFeatures(np.rollaxis(src.read()[:3], 0, 3))

        X_tile = np.zeros((num_pixels, num_tile_features), dtype=np.float32)
        tpf.compute_color_features(X_tile[:, color_slice])
        tpf.compute_texture_features(X_tile[:, texture_slice], sigmas,
                                     num_orientations)
        tpf.compute_entropy_features(X_tile[:, entropy_slice], neighborhood,
                                     scales)
        # TODO: compute entropy features with `neighborhoods` kwarg
        return X_tile

    values = [
        dask.delayed(get_tile_features)(tile_filepath)
        for tile_filepath in tile_filepaths
    ]

    with diagnostics.ProgressBar():
        X_tiles = dask.compute(*values)

    return np.vstack(X_tiles)


def build_features(split_df, method=None, output_filepath=None,
                   output_dir=None):
    # def dump_train_test_feature_arrays(df, output_train_filepath,
    #                                    output_test_filepath):
    #     X_train = pixel_features_from_tiling(
    #         df[df['train']]['tile_filepath'])
    #     np.save(output_train_filepath, X_train, allow_pickle=False)

    #     X_test = pixel_features_from_tiling(
    #         df[~df['train']]['tile_filepath'])
    #     np.save(output_test_filepath, X_test, allow_pickle=False)
    def dump_train_feature_arrays(df, output_filepath):
        np.save(output_filepath,
                pixel_features_from_tiling(df[df['train']]['tile_filepath']),
                allow_pickle=False)

    if method is None:
        if 'tile_cluster' in split_df:
            method = 'II'
        else:
            method = 'I'

    if method == 'I':
        dump_train_feature_arrays(split_df, output_filepath)
    else:
        for cluster_label, cluster_df in split_df.groupby('tile_cluster'):
            dump_train_feature_arrays(
                cluster_df, path.join(output_dir, f"X{cluster_label}.npy"))
