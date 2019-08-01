import glob
import os
import shutil
from os import path

import numpy as np
import pandas as pd
from sklearn import cluster, decomposition, metrics

from . import filters, image_descriptor, settings

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__all__ = ['TrainingSelector']


class TrainingSelector(object):
    def __init__(self, tile_filepaths=None, tile_dir=None,
                 tile_filename_pattern='*.tif', gabor_frequencies=None,
                 gabor_num_orientations=None, response_bins_per_axis=4,
                 num_color_bins=8):
        """     
        Parameters
        ----------
        gabor_frequencies : list-like

        gabor_num_orientations : int or list-like

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

        # preprocess technical keyword arguments
        if gabor_frequencies is None:
            self.gabor_frequencies = settings.GIST_GABOR_DEFAULT_FREQUENCIES

        if gabor_num_orientations is None:
            self.gabor_num_orientations = settings.GIST_GABOR_DEFAULT_ORIENTATIONS
        elif not isinstance(gabor_num_orientations, (list, tuple)):
            self.gabor_num_orientations = tuple(gabor_num_orientations
                                                for _ in gabor_frequencies)

        self.response_bins_per_axis = response_bins_per_axis
        self.num_color_bins = num_color_bins

    @property
    def descr_feature_matrix(self):
        try:
            return self._descr_feature_matrix
        except AttributeError:
            kernels = filters.get_gabor_filter_bank(
                frequencies=self.gabor_frequencies,
                num_orientations=self.gabor_num_orientations)

            num_blocks = self.response_bins_per_axis**2

            feature_row_list = []
            iterator = self.tile_filepaths
            if tqdm is not None:
                iterator = tqdm(iterator)

            for tile_filepath in iterator:
                feature_row_list.append(
                    image_descriptor.ImageDescriptor(
                        tile_filepath).get_image_descriptor(
                            kernels, self.response_bins_per_axis, num_blocks,
                            self.num_color_bins))

            self._descr_feature_matrix = np.vstack(feature_row_list)

            # TODO: cache as instance attribute (or even use property with and
            # pass this method's arguments to init), and then let people
            # interactively choose the number of PCA components until they're
            # happy with the represented variance? I vote yes.
            # TODO: cache this (via persistence): if `tile_filepaths` and the
            # technical parameters coincide, load from a file instead of
            # recomputing it
            # TODO: return copy?
            return self._descr_feature_matrix

    def train_test_split(self, base_output_dir, num_components=12,
                         num_tile_clusters=4, train_prop=.01):
        X = self.descr_feature_matrix
        pca = decomposition.PCA(n_components=num_components).fit(X)
        # TODO: how to log this so that it is outputted in a Jupyter notebook,
        # Python interpreter and CLI (click) interface?
        print(pca.explained_variance_ratio_.sum())

        X_pca = pca.transform(X)
        X_cols = range(num_components)
        df = pd.concat((pd.Series(self.tile_filepaths, name='tile_filepath'),
                        pd.DataFrame(X_pca, columns=X_cols)), axis=1)
        df['tile_cluster'] = cluster.KMeans(
            n_clusters=num_tile_clusters).fit_predict(X_pca)

        for cluster_label, cluster_df in df.groupby('tile_cluster'):
            X_cluster = cluster_df[X_cols]
            # use `ceil` to avoid zeros, which might completely ignore a
            # significant tile cluster
            num_train = int(np.ceil(train_prop * len(X_cluster)))
            cluster_km = cluster.KMeans(n_clusters=num_train).fit(X_cluster)
            closest, _ = metrics.pairwise_distances_argmin_min(
                cluster_km.cluster_centers_, X_cluster)
            train_idx = cluster_df.iloc[closest].index

            cluster_dir = path.join(base_output_dir, str(cluster_label))
            os.mkdir(cluster_dir)

            # TODO: refactor the two loops below into two data frame `apply`
            # calls to a function that copies the files
            # TODO: `train` dir name from settings?
            cluster_train_dir = path.join(cluster_dir, 'train')
            os.mkdir(cluster_train_dir)
            for tile_filepath in cluster_df.loc[train_idx, 'tile_filepath']:
                shutil.copy(
                    tile_filepath,
                    path.join(cluster_train_dir, path.basename(tile_filepath)))

            # TODO: `test` dir name from settings?
            cluster_test_dir = path.join(cluster_dir, 'test')
            os.mkdir(cluster_test_dir)
            for tile_filepath in cluster_df[~cluster_df.index.
                                            isin(train_idx)]['tile_filepath']:
                shutil.copy(
                    tile_filepath,
                    path.join(cluster_test_dir, path.basename(tile_filepath)))
