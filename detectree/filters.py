import numpy as np
from scipy.ndimage.filters import _gaussian_kernel1d
from skimage.filters import gabor_kernel

__all__ = ['get_texture_kernel', 'get_gabor_filter_bank']


def _get_gaussian_kernel1d(sigma, order=0, truncate=4.0):
    """
    Based on scipy.ndimage.filters.gaussian_filter1d
    """
    truncate = 4
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # # Since we are calling correlate, not convolve, revert the kernel
    # weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    weights = _gaussian_kernel1d(sigma, order, lw)
    return weights


def get_texture_kernel(sigma):
    """
    Parameters
    ----------
    sigma : numeric

    Returns
    -------
    texture_kernel :
    """
    g0_kernel_arr = _get_gaussian_kernel1d(sigma, 0)
    g2_kernel_arr = _get_gaussian_kernel1d(sigma, 2)

    return np.dot(g2_kernel_arr.reshape(1, -1).T, g0_kernel_arr.reshape(1, -1))


def get_gabor_filter_bank(frequencies, num_orientations):
    """
    Parameters
    ----------
    frequencies : list-like

    num_orientations : int or list-like


    Returns
    -------
    kernels :
    """
    kernels = []

    for frequency, _num_orientations in zip(frequencies, num_orientations):
        for orientation_i in range(_num_orientations):
            theta = orientation_i / _num_orientations * np.pi
            kernel = np.real(gabor_kernel(frequency, theta=theta))
            kernels.append(kernel)

    return kernels
