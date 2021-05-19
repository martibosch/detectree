"""Utilities to produce filters."""

import numpy as np
from scipy.ndimage.filters import _gaussian_kernel1d
from skimage.filters import gabor_kernel

__all__ = ["get_texture_kernel", "get_gabor_filter_bank"]


def _get_gaussian_kernel1d(sigma, *, order=0, truncate=4.0):
    """Based on scipy.ndimage.filters.gaussian_filter1d."""
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # # Since we are calling correlate, not convolve, revert the kernel
    # weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    weights = _gaussian_kernel1d(sigma, order, lw)
    return weights


def get_texture_kernel(sigma):
    """
    Get a texture kernel based on Yang et al. (2009).

    Parameters
    ----------
    sigma : numeric
        Scale parameter to build a texture kernel, based on a Gaussian on the
        X dimension and a second-derivative Gaussian in the Y dimension

    Returns
    -------
    texture_kernel : array-like
    """
    g0_kernel_arr = _get_gaussian_kernel1d(sigma, order=0)
    g2_kernel_arr = _get_gaussian_kernel1d(sigma, order=2)

    return np.dot(g2_kernel_arr.reshape(1, -1).T, g0_kernel_arr.reshape(1, -1))


def get_gabor_filter_bank(frequencies, num_orientations):
    """
    Get a Gabor filter bank with different frequencies and orientations.

    Parameters
    ----------
    frequencies : list-like
        Set of frequencies used to build the Gabor filter bank.
    num_orientations : int or list-like
        Number of orientations used to build the Gabor filter bank. If an
        integer is provided, the corresponding number of orientations will be
        used for each scale (determined by `gabor_frequencies`). If a tuple is
        provided, each element will determine the number of orientations that
        must be used at its matching scale (determined by `gabor_frequencies`)
        - thus the tuple must match the length of `frequencies`.

    Returns
    -------
    kernels : list-like
        List of kernel 2-D arrays that correspond to the filter bank
    """
    kernels = []

    for frequency, _num_orientations in zip(frequencies, num_orientations):
        for orientation_i in range(_num_orientations):
            theta = orientation_i / _num_orientations * np.pi
            kernel = np.real(gabor_kernel(frequency, theta=theta))
            kernels.append(kernel)

    return kernels
