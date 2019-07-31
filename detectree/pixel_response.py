import glob
from os import path

import numpy as np
import rasterio as rio

from . import settings, utils

__all__ = ['PixelResponseBuilder']


class PixelResponseBuilder(object):
    # It is really not necessary to use a class for this, but we do so for the
    # sake of API consistency with the `pixel_features` module
    def __init__(self, tree_val=None, nontree_val=None):
        if tree_val is None:
            tree_val = settings.RESPONSE_DEFAULT_TREE_VAL
        self.tree_val = tree_val

        if nontree_val is None:
            nontree_val = settings.RESPONSE_DEFAULT_NONTREE_VAL
        self.nontree_val = nontree_val

    def build_response_from_arr(self, img_binary):
        response_arr = img_binary.copy()
        response_arr[response_arr == self.tree_val] = 1
        response_arr[response_arr == self.nontree_val] = 0

        return response_arr.flatten()

    def build_response_from_filepath(self, img_filepath):
        with rio.open(img_filepath) as src:
            img_binary = src.read(1)

        return self.build_response_from_arr(img_binary)

    def build_response(self, split_df=None, response_img_dir=None,
                       response_img_filepaths=None, img_filename_pattern=None,
                       method=None, img_cluster=None):
        """
        TODO

        Parameters
        -------
        split_df : pd.DataFrame
            Data frame
        response_img_dir : str representing path to a directory, optional
            Path to the directory where the response images are located.
            Required if providing `split_df`. Otherwise `response_img_dir`
            might either be ignored if providing `response_img_filepaths`, or
            be used as the directory where the images whose filename matches
            `img_filename_pattern` are to be located.
        response_img_filepaths : list of image file paths, optional
            List of images to be transformed into the response. Alternatively,
            the same information can be provided by means of the `img_dir` and
            `img_filename_pattern` keyword arguments. Ignored if providing
            `split_df`
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
        y : np.ndarray
            Array with the pixel responses
        """
        if split_df is not None:
            if response_img_dir is None:
                raise ValueError(
                    "If `split_df` is provided, `response_img_dir` must also "
                    "be provided")
            if method is None:
                if 'img_cluster' in split_df:
                    method = 'cluster-II'
                else:
                    method = 'cluster-I'

            if method == 'cluster-I':
                img_filepaths = split_df[split_df['train']]['img_filepath']
            else:
                if img_cluster is None:
                    raise ValueError(
                        "If `method` is 'cluster-II', `img_cluster` must be "
                        "provided")
                img_filepaths = utils.get_img_filepaths(
                    split_df, img_cluster, True)

            response_img_filepaths = img_filepaths.apply(
                lambda filepath: path.join(response_img_dir,
                                           path.basename(filepath)))
        else:
            if response_img_filepaths is None:
                if img_filename_pattern is None:
                    img_filename_pattern = \
                        settings.IMG_DEFAULT_FILENAME_PATTERN
                if response_img_dir is None:
                    raise ValueError(
                        "Either `split_df`, `response_img_filepaths` or "
                        "`response_img_dir` must be provided")

                response_img_filepaths = glob.glob(
                    path.join(response_img_dir, img_filename_pattern))
            # TODO: `response_img_filepaths`

        # no need for dask here
        values = []
        for response_img_filepath in response_img_filepaths:
            values.append(
                self.build_response_from_filepath(response_img_filepath))

        return np.vstack(values).flatten()
