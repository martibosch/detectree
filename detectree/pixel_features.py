import numpy as np
from scipy import ndimage as ndi
from skimage import color, measure, morphology
from skimage.filters import rank

from . import filters

__all__ = ['ImagePixelFeatures']

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


class ImagePixelFeatures(object):
    def __init__(self, img_rgb):
        """
        Parameters
        ----------
        img_rgb : ndarray
        """
        super(ImagePixelFeatures, self).__init__()

        # the third component `_` is actually the number of channels in RGB,
        # which is already defined in the constant `NUM_RGB_CHANNELS`
        self.img_rgb = img_rgb
        self.num_rows, self.num_cols, _ = img_rgb.shape
        self.img_lab = color.rgb2lab(img_rgb)
        self.img_lab_l = self.img_lab[:, :, 0]  # ACHTUNG: this is a view

    def get_color_features(self):
        img_lab_vec = self.img_lab.reshape(self.num_rows * self.num_cols,
                                           NUM_LAB_CHANNELS)
        img_xyz_vec = color.rgb2xyz(self.img_rgb).reshape(
            self.num_rows * self.num_cols, NUM_XYZ_CHANNELS)
        img_ill_vec = np.dot(A, np.log(np.dot(B, img_xyz_vec.transpose()) +
                                       1)).transpose()
        # assemble the color features
        color_features = np.zeros((self.num_rows * self.num_cols,
                                   NUM_LAB_CHANNELS + NUM_ILL_CHANNELS))
        color_features[:, :NUM_LAB_CHANNELS] = img_lab_vec
        color_features[:, NUM_LAB_CHANNELS:NUM_LAB_CHANNELS +
                       NUM_ILL_CHANNELS] = img_ill_vec
        return color_features

    def get_texture_features(self, sigmas, num_orientations):
        num_sigmas = len(sigmas)
        texture_features = np.zeros(
            (self.num_rows * self.num_cols, num_orientations * num_sigmas))

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

    def get_entropy_features(self, neighborhoods=None, min_neighbor_range=2,
                             num_neighborhoods=3):
        """
        Parameters
        ----------
        neighborhood : , optional


        Returns
        -------
        out : 
        """

        if neighborhoods is None:
            scales = np.geomspace(1, 2**(num_neighborhoods - 1),
                                  num_neighborhoods)
            neighborhood = morphology.square(2 * min_neighbor_range + 1)
            entropy_features = np.zeros(
                (self.num_rows * self.num_cols, num_neighborhoods))

            entropy_features[:, 0] = rank.entropy(
                self.img_lab_l.astype(np.uint16), neighborhood).flatten()

            for i, factor in enumerate(scales[1:], start=1):
                img = ndi.zoom(
                    measure.block_reduce(self.img_lab_l,
                                         (factor, factor)).astype(np.uint16),
                    factor)
                entropy_features[:, i] = rank.entropy(img,
                                                      neighborhood).flatten()
        else:
            entropy_features = np.zeros(
                (self.num_rows * self.num_cols, len(neighborhoods)))

            for i, neighborhood in enumerate(neighborhoods):
                entropy_features[:, i] = rank.entropy(
                    self.img_lab_l.astype(np.uint16), neighborhood).flatten()

        return entropy_features
