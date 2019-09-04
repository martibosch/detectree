import glob
from os import path

import numpy as np
import pandas as pd
from sklearn import cluster, decomposition, metrics

from . import filters, image_descriptor

try:
    import dask
    from dask import diagnostics
except ImportError:
    dask = None

__all__ = ['TrainingSelector']


class TrainingSelector(object):
    def __init__(self, tile_filepaths=None, tile_dir=None,
                 tile_filename_pattern='*.tif',
                 gabor_frequencies=(.1, .25, .4),
                 gabor_num_orientations=(4, 8, 8), response_bins_per_axis=4,
                 num_color_bins=8):
        """
        Parameters
        ----------
        gabor_frequencies : tuple

        gabor_num_orientations : int or tuple

        response_bins_per_axis : int

        num_color_bins : int

        Returns
        -------
        descr_feature_matrix
        """

        super(TrainingSelector, self).__init__()

        if tile_filepaths is None:
            tile_filepaths = glob.glob(
                path.join(tile_dir, tile_filename_pattern))

        self.tile_filepaths = tile_filepaths

        # TODO: boolean arg for equal tile size (and pass `block_shape` to
        # `get_gist_descriptor`)?
        self.gabor_frequencies = gabor_frequencies

        if isinstance(gabor_num_orientations, tuple):
            self.gabor_num_orientations = gabor_num_orientations
        else:
            # `gabor_num_orientations` is an int
            self.gabor_num_orientations = tuple(gabor_num_orientations
                                                for _ in gabor_frequencies)

        self.response_bins_per_axis = response_bins_per_axis
        self.num_color_bins = num_color_bins

    @staticmethod
    def _get_image_descr(tile_filepath, kernels, response_bins_per_axis,
                         num_blocks, num_color_bins):
        return image_descriptor.ImageDescriptor(
            tile_filepath).get_image_descriptor(kernels,
                                                response_bins_per_axis,
                                                num_blocks, num_color_bins)

    @property
    def descr_feature_matrix(self):
        try:
            return self._descr_feature_matrix
        except AttributeError:
            kernels = filters.get_gabor_filter_bank(
                frequencies=self.gabor_frequencies,
                num_orientations=self.gabor_num_orientations)

            num_blocks = self.response_bins_per_axis**2

            if dask is not None:
                values = [
                    dask.delayed(TrainingSelector._get_image_descr)(
                        tile_filepath, kernels, self.response_bins_per_axis,
                        num_blocks, self.num_color_bins)
                    for tile_filepath in self.tile_filepaths
                ]

                with diagnostics.ProgressBar():
                    feature_rows = dask.compute(*values)
            else:
                feature_rows = [
                    TrainingSelector._get_image_descr(
                        tile_filepath, kernels, self.response_bins_per_axis,
                        num_blocks, self.num_color_bins)
                    for tile_filepath in self.tile_filepaths
                ]

            self._descr_feature_matrix = np.vstack(feature_rows)

            # TODO: cache as instance attribute (or even use property with and
            # pass this method's arguments to init), and then let people
            # interactively choose the number of PCA components until they're
            # happy with the represented variance? I vote yes.
            # TODO: cache this (via persistence): if `tile_filepaths` and the
            # technical parameters coincide, load from a file instead of
            # recomputing it
            # TODO: return copy?
            return self._descr_feature_matrix

    def train_test_split(self, method='II', num_components=12,
                         num_tile_clusters=4, train_prop=.01,
                         return_evr=False):
        X = self.descr_feature_matrix
        pca = decomposition.PCA(n_components=num_components).fit(X)

        X_pca = pca.transform(X)
        X_cols = range(num_components)
        df = pd.concat((pd.Series(self.tile_filepaths, name='tile_filepath'),
                        pd.DataFrame(X_pca, columns=X_cols)), axis=1)

        if method == 'I':
            km = cluster.KMeans(n_clusters=int(np.ceil(train_prop *
                                                       len(df)))).fit(X_pca)
            closest, _ = metrics.pairwise_distances_argmin_min(
                km.cluster_centers_, df[X_cols])
            train_idx = df.iloc[closest].index

            df['train'] = [True if i in train_idx else False for i in df.index]
        else:

            def cluster_train_test_split(tile_cluster_ser):
                X_cluster_df = df.loc[tile_cluster_ser.index, X_cols]
                # use `ceil` to avoid zeros, which might completely ignore a
                # significant tile cluster
                num_train = int(np.ceil(train_prop * len(X_cluster_df)))
                cluster_km = cluster.KMeans(
                    n_clusters=num_train).fit(X_cluster_df)
                closest, _ = metrics.pairwise_distances_argmin_min(
                    cluster_km.cluster_centers_, X_cluster_df)
                train_idx = X_cluster_df.iloc[closest].index
                return [
                    True if i in train_idx else False
                    for i in X_cluster_df.index
                ]

            df['tile_cluster'] = cluster.KMeans(
                n_clusters=num_tile_clusters).fit_predict(X_pca)
            df['train'] = df.groupby('tile_cluster')['tile_cluster'].transform(
                cluster_train_test_split)

        split_df = df.drop(X_cols, axis=1)

        if return_evr:
            return split_df, pca.explained_variance_ratio_.sum()
        else:
            return split_df
