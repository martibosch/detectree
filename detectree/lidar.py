"""Utilities to get canopy information from LiDAR data."""

import laspy
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
from rasterio import enums, features

from . import settings

__all__ = ["rasterize_lidar", "LidarToCanopy"]


def rasterize_lidar(lidar_filepath, lidar_tree_values, dst_shape, dst_transform):
    """Rasterize a LiDAR file.

    Transforms a LiDAR file into a raster aligned to `ref_img_filepath`, where each
    pixel of the target raster represents the number of LiDAR points of the classes set
    in `lidar_tree_values` that occur in the pixel's geographic extent.

    Parameters
    ----------
    lidar_filepath : str, file object or pathlib.Path object
        Path to a file, URI, file object opened in binary ('rb') mode, or a Path object
        representing the LiDAR file from which a tree canopy mask will be computed. The
        value will be passed to `laspy.file.File`.
    lidar_tree_values : int or list-like
        LiDAR point classes that correspond to trees.
    dst_shape : tuple
        Shape of the output raster.
    dst_transform : Affine
        Affine transformation of the output raster.

    Returns
    -------
    lidar_arr : numpy ndarray
        Array with the rasterized lidar.
    """
    las = laspy.read(lidar_filepath)
    c = np.array(las.classification)
    x = np.array(las.x)
    y = np.array(las.y)

    cond = np.isin(c, lidar_tree_values)
    lidar_df = pd.DataFrame({"class_val": c[cond], "x": x[cond], "y": y[cond]})
    try:
        # note that rasterize automatically sets the dst dtype
        return features.rasterize(
            shapes=[
                (geom, 1)
                for geom in shapely.points(
                    *[lidar_df[coord].astype("float64").values for coord in ["x", "y"]]
                )
            ],
            out_shape=dst_shape,
            transform=dst_transform,
            merge_alg=enums.MergeAlg("ADD"),
        )
    except ValueError:
        # there are no LiDAR points of the target classes (`lidar_tree_values`). Return
        # array of zeros of uint8 dtype
        return np.zeros(dst_shape, np.uint8)


class LidarToCanopy:
    """Extract raster canopy masks from LiDAR data."""

    def __init__(
        self,
        *,
        tree_threshold=None,
        output_dtype=None,
        output_tree_val=None,
        output_nodata=None,
    ):
        """
        Extract raster canopy masks from LiDAR data.

        Parameters
        ----------
        tree_threshold : numeric, optional
            Threshold of lidar points classified as tree by pixel at which
            point the pixel is considered a tree. As a rule of thumb, the value can be
            set to result of dividing the point density of the lidar (e.g., pts/m^2) by
            the pixel area (e.g., m^2). If no value is provided, the value set in
            `settings.LIDAR_TREE_THRESHOLD` is used.
        output_dtype : str or numpy dtype, optional
            The desired data type of the output raster canopy masks. If no value is
            provided, the value set in `settings.LIDAR_OUTPUT_DTYPE` is used.
        output_tree_val : int, optional
            The value that designates tree pixels in the output raster canopy masks. If
            no value is provided, the value set in `settings.LIDAR_OUTPUT_TREE_VAL` is
            used.
        output_nodata : int, optional
            The value that designates non-tree pixels in the output raster canopy masks.
            If no value is provided, the value set in `settings.LIDAR_OUTPUT_NODATA` is
            used.
        """
        if tree_threshold is None:
            tree_threshold = settings.LIDAR_TREE_THRESHOLD
        if output_dtype is None:
            output_dtype = settings.LIDAR_OUTPUT_DTYPE
        if output_tree_val is None:
            output_tree_val = settings.LIDAR_OUTPUT_TREE_VAL
        if output_nodata is None:
            output_nodata = settings.LIDAR_OUTPUT_NODATA

        self.tree_threshold = tree_threshold
        self.output_dtype = output_dtype
        self.output_tree_val = output_tree_val
        self.output_nodata = output_nodata

    def to_canopy_mask(
        self,
        lidar_filepath,
        lidar_tree_values,
        ref_img_filepath,
        *,
        output_filepath=None,
        postprocess_func=None,
        postprocess_func_args=None,
        postprocess_func_kwargs=None,
    ):
        """
        Transform a LiDAR file into a canopy mask.

        Parameters
        ----------
        lidar_filepath : str, file object or pathlib.Path object
            Path to a file, URI, file object opened in binary ('rb') mode, or a Path
            object representing the LiDAR file from which a tree canopy mask will be
            computed. The value will be passed to `laspy.file.File`.
        lidar_tree_values : int or list-like
            LiDAR point classes that correspond to trees.
        ref_img_filepath : str, file object or pathlib.Path object
            Reference raster image to which the LiDAR data will be rasterized.
        output_filepath : str, file object or pathlib.Path object, optional
            Path to a file, URI, file object opened in binary ('rb') mode, or a Path
            object representing where the predicted image is to be dumped. The value
            will be passed to `rasterio.open` in 'write' mode.
        postprocess_func : function
            Post-processing function which takes as input the rasterized lidar as a
            boolean ndarray and returns a the post-processed lidar also as a boolean
            ndarray.
        postprocess_func_args : list-like, optional
            Arguments to be passed to `postprocess_func`.
        postprocess_func_kwargs : dict, optional
            Keyword arguments to be passed to `postprocess_func`.

        Returns
        -------
        canopy_arr : numpy ndarray
            Array with the canopy mask (only tree/non-tree values).
        """
        # canopy_arr = ndi.binary_dilation(
        #     ndi.binary_opening(arr >= self.tree_val,
        #                        iterations=self.num_opening_iterations),
        #     iterations=self.num_dilation_iterations).astype(
        #         self.output_dtype) * self.output_tree_val
        with rio.open(ref_img_filepath) as src:
            meta = src.meta.copy()
        lidar_arr = rasterize_lidar(
            lidar_filepath,
            lidar_tree_values,
            (meta["height"], meta["width"]),
            meta["transform"],
        )
        canopy_arr = lidar_arr >= self.tree_threshold
        if postprocess_func is not None:
            canopy_arr = postprocess_func(
                canopy_arr, *postprocess_func_args, **postprocess_func_kwargs
            )
        canopy_arr = np.where(
            canopy_arr, self.output_tree_val, self.output_nodata
        ).astype(self.output_dtype)

        if output_filepath is not None:
            meta.update(dtype=self.output_dtype, count=1, nodata=self.output_nodata)
            with rio.open(output_filepath, "w", **meta) as dst:
                dst.write(canopy_arr, 1)

        return canopy_arr
