import itertools
from os import path

import numpy as np
import rasterio as rio
from rasterio import windows

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


def split_into_tiles(input_filepath, output_dir, tile_width=512,
                     tile_height=512, output_filename='tile_{}-{}.tif',
                     custom_meta=None):
    with rio.open(input_filepath) as src:
        meta = src.meta.copy()
        if custom_meta is not None:
            meta.update(custom_meta)

        iterator = _get_window_transform(src, tile_width, tile_height)
        if tqdm is not None:
            iterator = tqdm(iterator)
        for window, transform in iterator:
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            with rio.open(
                    path.join(
                        output_dir,
                        output_filename.format(int(window.col_off),
                                               int(window.row_off))), 'w',
                    **meta) as dst:
                dst.write(src.read(window=window))


def img_rgb_from_filepath(input_filepath):
    with rio.open(input_filepath) as src:
        arr = src.read()

    return np.rollaxis(arr[:3], 0, 3)
