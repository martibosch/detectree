import itertools
from os import path

import numpy as np
import rasterio as rio
from rasterio import windows

from . import settings

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__all__ = ['split_into_tiles', 'img_rgb_from_filepath']

# See https://bit.ly/2KkELpI


def _get_window_transform(ds, width, height):
    num_rows, num_cols = ds.meta['height'], ds.meta['width']
    offsets = itertools.product(range(0, num_cols, width),
                                range(0, num_rows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=num_cols,
                                height=num_rows)
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=width,
                                height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


def split_into_tiles(input_filepath, output_dir, tile_width=None,
                     tile_height=None, output_filename=None,
                     only_full_tiles=False, custom_meta=None):

    if tile_width is None:
        tile_width = settings.TILE_DEFAULT_WIDTH
    if tile_height is None:
        tile_height = settings.TILE_DEFAULT_HEIGHT
    if output_filename is None:
        output_filename = settings.TILE_DEFAULT_OUTPUT_FILENAME

    tile_count = 0
    with rio.open(input_filepath) as src:
        meta = src.meta.copy()
        if custom_meta is not None:
            meta.update(custom_meta)

        iterator = _get_window_transform(src, tile_width, tile_height)
        if tqdm is not None:
            iterator = tqdm(iterator)
        for window, transform in iterator:
            if not only_full_tiles or (window.width == tile_width
                                       and window.height == tile_height):
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                with rio.open(
                        path.join(
                            output_dir,
                            output_filename.format(int(window.col_off),
                                                   int(window.row_off))), 'w',
                        **meta) as dst:
                    dst.write(src.read(window=window))
                tile_count += 1

    return tile_count


def img_rgb_from_filepath(input_filepath):
    with rio.open(input_filepath) as src:
        arr = src.read()

    return np.rollaxis(arr[:3], 0, 3)


## non-image utils
def get_response_tile_filepath(tile_filepath, response_train_tiles_dir):
    return path.join(response_train_tiles_dir, path.basename(tile_filepath))
