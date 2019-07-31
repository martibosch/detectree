import glob
from os import path

import dask
import numpy as np
from dask import diagnostics
from scipy import ndimage as ndi
from skimage import color, morphology, transform
from skimage.filters import rank

from . import filters, settings, utils

__all__ = ['PixelFeaturesBuilder']

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


class PixelFeaturesBuilder(object):
    def __init__(self, sigmas=None, num_orientations=None, neighborhood=None,
                 min_neighborhood_range=None, num_neighborhoods=None):
        # preprocess technical keyword arguments
        # texture features
        if sigmas is None:
            sigmas = settings.GAUSS_DEFAULT_SIGMAS
        self.sigmas = sigmas

        if num_orientations is None:
            num_orientations = settings.GAUSS_DEFAULT_NUM_ORIENTATIONS
        self.num_orientations = num_orientations

        # entropy features
        # TODO: compute entropy features with `neighborhoods` kwarg
        # if neighborhoods is None:
        #     if min_neighborhood_range is None:
        #         min_neighborhood_range = \
        #             settings.ENTROPY_DEFAULT_MIN_NEIGHBORHOOD_RANGE
        #     if num_neighborhoods is None:
        #         num_neighborhoods = \
        #             settings.ENTROPY_DEFAULT_NUM_NEIGHBORHOODS
        #     scales = np.geomspace(1, 2**(num_neighborhoods - 1),
        #                           num_neighborhoods).astype(int)
        #     neighborhood = morphology.square(2 * min_neighborhood_range + 1)
        # else:
        #     num_neighborhoods = len(neighborhoods)
        if neighborhood is None:
            if min_neighborhood_range is None:
                min_neighborhood_range = \
                    settings.ENTROPY_DEFAULT_MIN_NEIGHBORHOOD_RANGE
            if num_neighborhoods is None:
                num_neighborhoods = settings.ENTROPY_DEFAULT_NUM_NEIGHBORHOODS
            neighborhood = morphology.square(2 * min_neighborhood_range + 1)
        self.neighborhood = neighborhood
        # TODO: scales NOT used when `neighborhoods` kwarg is provided (see
        # the commented code above)
        self.scales = np.geomspace(1, 2**(num_neighborhoods - 1),
                                   num_neighborhoods).astype(int)

        self.num_color_features = NUM_LAB_CHANNELS + NUM_ILL_CHANNELS
        self.num_texture_features = num_orientations * len(sigmas)
        self.num_entropy_features = num_neighborhoods
        self.num_pixel_features = self.num_color_features + \
            self.num_texture_features + self.num_entropy_features
        # self.X = np.zeros((num_tiling_pixels, num_img_features))

    def build_features_from_arr(self, img_rgb):
        # the third component `_` is actually the number of channels in RGB,
        # which is already defined in the constant `NUM_RGB_CHANNELS`
        num_rows, num_cols, _ = img_rgb.shape
        num_pixels = num_rows * num_cols
        img_lab = color.rgb2lab(img_rgb)
        img_lab_l = img_lab[:, :, 0]  # ACHTUNG: this is a view

        X = np.zeros((num_pixels, self.num_pixel_features), dtype=np.float32)

        # color features
        # tpf.compute_color_features(X_img[:, self.color_slice])
        img_lab_vec = img_lab.reshape(num_rows * num_cols, NUM_LAB_CHANNELS)
        img_xyz_vec = color.rgb2xyz(img_rgb).reshape(num_rows * num_cols,
                                                     NUM_XYZ_CHANNELS)
        img_ill_vec = np.dot(A, np.log(np.dot(B, img_xyz_vec.transpose()) +
                                       1)).transpose()
        X[:, :NUM_LAB_CHANNELS] = img_lab_vec
        X[:, NUM_LAB_CHANNELS:NUM_LAB_CHANNELS +
          NUM_ILL_CHANNELS] = img_ill_vec

        # texture features
        # tpf.compute_texture_features(X_img[:, self.texture_slice],
        #                              self.sigmas, self.num_orientations)
        for i, sigma in enumerate(self.sigmas):
            base_kernel_arr = filters.get_texture_kernel(sigma)
            for j, orientation in enumerate(range(self.num_orientations)):
                # theta = orientation / num_orientations * np.pi
                theta = orientation * 180 / self.num_orientations
                oriented_kernel_arr = ndi.interpolation.rotate(
                    base_kernel_arr, theta)
                img_filtered = ndi.convolve(img_lab_l, oriented_kernel_arr)
                img_filtered_vec = img_filtered.flatten()
                X[:, self.num_color_features + i * self.num_orientations +
                  j] = img_filtered_vec

        # entropy features
        # tpf.compute_entropy_features(X_img[:, self.entropy_slice],
        #                              self.neighborhood, self.scales)
        entropy_start = self.num_color_features + self.num_texture_features
        X[:, entropy_start] = rank.entropy(img_lab_l.astype(np.uint16),
                                           self.neighborhood).flatten()

        for i, factor in enumerate(self.scales[1:], start=1):
            img = transform.resize(
                transform.downscale_local_mean(img_lab_l, (factor, factor)),
                img_lab_l.shape).astype(np.uint16)
            X[:, entropy_start + i] = rank.entropy(
                img, self.neighborhood).flatten()

        return X

    def build_features_from_filepath(self, img_filepath):
        img_rgb = utils.img_rgb_from_filepath(img_filepath)
        return self.build_features_from_arr(img_rgb)

    def build_features(self, split_df=None, img_filepaths=None, img_dir=None,
                       img_filename_pattern=None, method=None,
                       img_cluster=None):
        """
        Build the pixel features for a list of images

        Parameters
        -------
        split_df : pd.DataFrame
            Data frame
        img_filepaths : list of image file paths, optional
            List of images to be transformed into features. Alternatively, the
            same information can be provided by means of the `img_dir` and
            `img_filename_pattern` keyword arguments. Ignored if providing
            `split_df`
        img_dir : str representing path to a directory, optional
            Path to the directory where the images whose filename matches
            `img_filename_pattern` are to be located. Ignored if `split_df` or
            `img_filepaths` is provided.
        img_filename_pattern : str representing a file-name pattern, optional
            Filename pattern to be matched in order to obtain the list of
            images. If no value is provided, the default value set in
            `settings.IMG_DEFAULT_FILENAME_PATTERN` will be taken. Ignored if
            `split_df` or `img_filepaths` is provided.
        method : {'cluster-I', 'cluster-II'}, optional
            Method used in the train/test split
        img_cluster : int, optional
            The label of the cluster of images. Only used if `method` is
            'cluster-II'
        Returns
        -------
        X : np.ndarray
            Array with the pixel features
        """
        # TODO: accept `neighborhoods` kwarg
        if split_df is not None:
            if method is None:
                if 'img_cluster' in split_df:
                    method = 'cluster-II'
                else:
                    method = 'cluster-I'

            if method == 'cluster-I':
                # dump_train_feature_arrays(split_df, output_filepath)
                img_filepaths = split_df[split_df['train']]['img_filepath']
            else:
                if img_cluster is None:
                    raise ValueError(
                        "If `method` is 'cluster-II', `img_cluster` must be "
                        "provided")
                img_filepaths = utils.get_img_filepaths(
                    split_df, img_cluster, True)

        else:
            if img_filepaths is None:
                if img_filename_pattern is None:
                    img_filename_pattern = \
                        settings.IMG_DEFAULT_FILENAME_PATTERN
                if img_dir is None:
                    raise ValueError(
                        "Either `split_df`, `img_filepaths` or `img_dir` must "
                        "be provided")

                img_filepaths = glob.glob(
                    path.join(img_dir, img_filename_pattern))

        values = [
            dask.delayed(self.build_features_from_filepath)(img_filepath)
            for img_filepath in img_filepaths
        ]

        with diagnostics.ProgressBar():
            X = dask.compute(*values)

        return np.vstack(X)
