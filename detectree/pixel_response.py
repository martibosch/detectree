import glob
from os import path

import numpy as np
import rasterio as rio

from . import settings, utils

__all__ = ['pixel_response_from_tiling', 'build_response']


def pixel_response_from_tiling(tile_filepaths=None, tile_dir=None,
                               tile_filename_pattern=None, tree_val=None,
                               nontree_val=None):
    if tile_filepaths is None:
        if tile_filename_pattern is None:
            tile_filename_pattern = settings.TILE_DEFAULT_FILENAME_PATTERN
        tile_filepaths = glob.glob(path.join(tile_dir, tile_filename_pattern))

    if tree_val is None:
        tree_val = settings.RESPONSE_DEFAULT_TREE_VAL
    if nontree_val is None:
        nontree_val = settings.RESPONSE_DEFAULT_NONTREE_VAL

    def get_tile_response(tile_filepath):
        with rio.open(tile_filepath) as src:
            arr = src.read(1)

        tile_arr = arr.copy()
        tile_arr[arr == tree_val] = 1
        tile_arr[arr == nontree_val] = 0

        return tile_arr.flatten()

    values = []
    for tile_filepath in tile_filepaths:
        values.append(get_tile_response(tile_filepath))

    return np.vstack(values).flatten()


def build_response(split_df, response_train_tiles_dir, method=None,
                   output_filepath=None, output_dir=None):
    def dump_train_feature_arrays(df, output_filepath):
        y_train = pixel_response_from_tiling(
            df[df['train']]['tile_filepath'].apply(
                utils.get_response_tile_filepath,
                args=(response_train_tiles_dir, )))
        np.save(output_filepath, y_train, allow_pickle=False)

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
                cluster_df, path.join(output_dir, f"y{cluster_label}.npy"))
