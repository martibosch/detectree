"""detectree general utility functions."""

import datetime as dt
import itertools
import logging as lg
import os
import sys
import unicodedata
from os import path

import numpy as np
import rasterio as rio
from rasterio import windows
from tqdm import tqdm

from . import settings

__all__ = [
    "split_into_tiles",
    "img_rgb_from_filepath",
    "get_img_filename_ser",
    "log",
    "get_logger",
]


# See https://bit.ly/2KkELpI
def split_into_tiles(
    input_filepath,
    output_dir,
    *,
    tile_width=None,
    tile_height=None,
    output_filename=None,
    only_full_tiles=False,
    keep_empty_tiles=False,
    custom_meta=None,
):
    """
    Split the image of `input_filepath` into tiles.

    Parameters
    ----------
    input_filepath : str, file object or pathlib.Path object
        Path to a file, URI, file object opened in binary ('rb') mode, or a Path object
        representing the image to be classified. The value will be passed to
        `rasterio.open`
    output_dir : str or pathlib.Path object
        Path to the directory where the predicted images are to be dumped.
    tile_width : int, optional
        Tile width in pixels. If no value is provided, the value set in
        `settings.TILE_WIDTH` is used.
    tile_height : int, optional
        Tile height in pixels. If no value is provided, the value set in
        `settings.TILE_HEIGHT` is used.
    output_filename : str, optional
        Template to be string-formatted in order to name the output tiles. If no value
        is provided, the value set in `settings.TILE_OUTPUT_FILENAME` is used.
    only_full_tiles : bool, optional (default False)
        Whether only full tiles (of size `tile_width`x`tile_height`) should be dumped.
    keep_empty_tiles : bool, optional (default False)
        Whether tiles containing only no-data pixels should be dumped.
    custom_meta : dict, optional
        Custom meta data for the output tiles.

    Returns
    -------
    output_filepaths : list
        List of the file paths of the dumped tiles.
    """
    if tile_width is None:
        tile_width = settings.TILE_WIDTH
    if tile_height is None:
        tile_height = settings.TILE_HEIGHT
    if output_filename is None:
        output_filename = settings.TILE_OUTPUT_FILENAME

    output_filepaths = []
    with rio.open(input_filepath) as src:
        meta = src.meta.copy()

        if custom_meta is not None:
            meta.update(custom_meta)

        def _get_window_transform(width, height):
            num_rows, num_cols = src.meta["height"], src.meta["width"]
            offsets = itertools.product(
                range(0, num_cols, width), range(0, num_rows, height)
            )
            big_window = windows.Window(
                col_off=0, row_off=0, width=num_cols, height=num_rows
            )
            for col_off, row_off in offsets:
                window = windows.Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=width,
                    height=height,
                ).intersection(big_window)
                transform = windows.transform(window, src.transform)
                yield window, transform

        iterator = _get_window_transform(tile_width, tile_height)
        if tqdm is not None:
            iterator = tqdm(iterator)

        # tests whether a given tile should be dumped or not. Since there are two
        # possible tests that depend on the arguments provided by the user, we will use
        # a list of tests and then check whether any test must be applied. This
        # mechanism avoids having to check whether tests must be applied at each
        # iteration (see the if/else at the end of this function).
        tests = []
        if only_full_tiles:

            def test_full_tile(window):
                return window.width == tile_width and window.height == tile_height

            tests.append(test_full_tile)

        if not keep_empty_tiles:

            def test_empty_tile(window):
                return np.any(src.dataset_mask(window=window))

            tests.append(test_empty_tile)

        def inner_loop(window, transform):
            meta["transform"] = transform
            meta["width"], meta["height"] = window.width, window.height
            output_filepath = path.join(
                output_dir,
                output_filename.format(int(window.col_off), int(window.row_off)),
            )
            with rio.open(output_filepath, "w", **meta) as dst:
                dst.write(src.read(window=window))
            log(f"Dumped tile to {output_filepath}")
            output_filepaths.append(output_filepath)

        if tests:
            for window, transform in iterator:
                if all(test(window) for test in tests):
                    inner_loop(window, transform)
        else:
            for window, transform in iterator:
                inner_loop(window, transform)

    return output_filepaths


def img_rgb_from_filepath(img_filepath):
    """
    Read an RGB image file into a 3-D array.

    See the `background <https://bit.ly/2KlCICO>`_ example notebook for more details.

    Parameters
    ----------
    img_filepath : str, file object or pathlib.Path object
        Path to a file, URI, file object opened in binary ('rb') mode, or a Path object
        representing the image for which a GIST descriptor will be computed. The value
        will be passed to `rasterio.open`.

    Returns
    -------
    img_rgb : numpy.ndarray
        3-D array with the RGB image.
    """
    with rio.open(img_filepath) as src:
        arr = src.read()

    return np.rollaxis(arr[:3], 0, 3)


# non-image utils
def get_img_filename_ser(split_df, img_cluster, train):
    """
    Get image filenames from a train/test split data frame.

    Parameters
    ----------
    split_df : pandas DataFrame
        Data frame with the train/test split.
    img_cluster : int
        The label of the cluster of tiles.
    train : bool
        Whether the list of training (True) or testing (False) tiles must be
        returned.

    Returns
    -------
    img_filenames : pandas Series
        List of image file names.
    """
    if train:
        train_cond = split_df["train"]
    else:
        train_cond = ~split_df["train"]
    try:
        return split_df[train_cond & (split_df["img_cluster"] == img_cluster)][
            "img_filename"
        ]
    except KeyError:
        raise ValueError(
            "If `method` is 'cluster-II', `split_df` must have a "
            "'img_cluster' column"
        )


# logging (from https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py)
def log(message, *, level=None, name=None, filename=None):
    """
    Write a message to the log file and/or print to the the console.

    Parameters
    ----------
    message : string
        the content of the message to log.
    level : int
        one of the logger.level constants.
    name : string
        name of the logger.
    filename : string
        name of the log file.
    """
    if level is None:
        level = settings.log_level
    if name is None:
        name = settings.log_name
    if filename is None:
        filename = settings.log_filename

    # if logging to file is turned on
    if settings.log_file:
        # get the current logger (or create a new one, if none), then log message at
        # requested level
        logger = get_logger(level=level, name=name, filename=filename)
        if level == lg.DEBUG:
            logger.debug(message)
        elif level == lg.INFO:
            logger.info(message)
        elif level == lg.WARNING:
            logger.warning(message)
        elif level == lg.ERROR:
            logger.error(message)

    # if logging to console is turned on, convert message to ascii and print to the
    # console
    if settings.log_console:
        # capture current stdout, then switch it to the console, print the message, then
        # switch back to what had been the stdout. this prevents logging to notebook -
        # instead, it goes to console
        standard_out = sys.stdout
        sys.stdout = sys.__stdout__

        # convert message to ascii for console display so it doesn't break windows
        # terminals
        message = (
            unicodedata.normalize("NFKD", str(message))
            .encode("ascii", errors="replace")
            .decode()
        )
        print(message)
        sys.stdout = standard_out


def get_logger(*, level=None, name=None, filename=None):
    """
    Create a logger or return the current one if already instantiated.

    Parameters
    ----------
    level : int
        one of the logger.level constants.
    name : string
        name of the logger.
    filename : string
        name of the log file.

    Returns
    -------
    logger.logger
    """
    if level is None:
        level = settings.log_level
    if name is None:
        name = settings.log_name
    if filename is None:
        filename = settings.log_filename

    logger = lg.getLogger(name)

    # if a logger with this name is not already set up
    if not getattr(logger, "handler_set", None):
        # get today's date and construct a log filename
        todays_date = dt.datetime.today().strftime("%Y_%m_%d")
        log_filename = path.join(
            settings.logs_folder, "{}_{}.log".format(filename, todays_date)
        )

        # if the logs folder does not already exist, create it
        if not path.exists(settings.logs_folder):
            os.makedirs(settings.logs_folder)

        # create file handler and log formatter and set them up
        handler = lg.FileHandler(log_filename, encoding="utf-8")
        formatter = lg.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.handler_set = True

    return logger
