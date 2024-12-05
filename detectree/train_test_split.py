"""Split the set of images into training and testing sets."""

import glob
from os import path

import dask
import numpy as np
import pandas as pd
from dask import diagnostics
from sklearn import cluster, decomposition, metrics

from . import filters, image_descriptor, settings

__all__ = ["TrainingSelector"]


class TrainingSelector:
    """Select the images/tiles to be used to train the classifier(s)."""

    def __init__(
        self,
        *,
        img_filepaths=None,
        img_dir=None,
        img_filename_pattern=None,
        gabor_frequencies=None,
        gabor_num_orientations=None,
        response_bins_per_axis=None,
        num_color_bins=None,
    ):
        """
        Initialize the training selector.

        The arguments provided to the initialization method will determine how the image
        descriptors are computed. See the `background <https://bit.ly/2KlCICO>`_ example
        notebook for more details.

        Parameters
        ----------
        img_filepaths : list-like, optional
            List of paths to the input tiles whose features will be used to train the
            classifier.
        img_dir : str representing path to a directory, optional
            Path to the directory where the images whose filename matches
            `img_filename_pattern` are to be located. Ignored if `img_filepaths` is
            provided.
        img_filename_pattern : str representing a file-name pattern, optional
            Filename pattern to be matched in order to obtain the list of images. If no
            value is provided, the value set in `settings.IMG_FILENAME_PATTERN` is used.
            Ignored if `img_filepaths` is provided.
        gabor_frequencies : tuple, optional
            Set of frequencies used to build the Gabor filter bank. If no value is
            provided, the value set in `settings.GIST_GABOR_FREQUENCIES` is used.
        gabor_num_orientations : int or tuple, optional
            Number of orientations used to build the Gabor filter bank. If an integer is
            provided, the corresponding number of orientations will be used for each
            scale (determined by `gabor_frequencies`). If a tuple is provided, each
            element will determine the number of orientations that must be used at its
            matching scale (determined by `gabor_frequencies`) - thus the tuple must
            match the length of `gabor_frequencies`. If no value is provided, the value
            set in `settings.GIST_GABOR_NUM_ORIENTATIONS` is used.
        response_bins_per_axis : int, optional
            Number of spatial bins per axis into which the responses to the Gabor filter
            bank will be aggregated. For example, a value of 2 will aggregate the
            responses into the four quadrants of the image (i.e., 2x2, 2 bins in each
            axis of the image). If no value is provided, the value set in
            `settings.GIST_RESPONSE_BINS_PER_AXIS` is used.
        num_color_bins : int, optional
            Number of bins in each dimension used to compute a joint color histogram in
            the L*a*b color space. If no value is provided, the value set in
            `settings.GIST_NUM_COLOR_BINS` is used.
        """
        super().__init__()

        # get `None` keyword-arguments from settings
        if img_filename_pattern is None:
            img_filename_pattern = settings.IMG_FILENAME_PATTERN
        if gabor_frequencies is None:
            gabor_frequencies = settings.GIST_GABOR_FREQUENCIES
        if gabor_num_orientations is None:
            gabor_num_orientations = settings.GIST_GABOR_NUM_ORIENTATIONS
        if response_bins_per_axis is None:
            response_bins_per_axis = settings.GIST_RESPONSE_BINS_PER_AXIS
        if num_color_bins is None:
            num_color_bins = settings.GIST_NUM_COLOR_BINS

        # now proceed
        if img_filepaths is None:
            img_filepaths = glob.glob(path.join(img_dir, img_filename_pattern))

        self.img_filepaths = img_filepaths

        # TODO: boolean arg for equal tile size (and pass `block_shape` to
        # `get_gist_descriptor`)?
        self.gabor_frequencies = gabor_frequencies

        if isinstance(gabor_num_orientations, tuple):
            self.gabor_num_orientations = gabor_num_orientations
        else:
            # `gabor_num_orientations` is an int
            self.gabor_num_orientations = tuple(
                gabor_num_orientations for _ in gabor_frequencies
            )

        self.response_bins_per_axis = response_bins_per_axis
        self.num_color_bins = num_color_bins

    @property
    def descr_feature_matrix(self):
        """Compute matrix of descriptors (feature rows)."""
        try:
            return self._descr_feature_matrix
        except AttributeError:
            kernels = filters.get_gabor_filter_bank(
                frequencies=self.gabor_frequencies,
                num_orientations=self.gabor_num_orientations,
            )

            # num_blocks = self.response_bins_per_axis**2

            # feature_rows = [
            #      TrainingSelector._get_image_descr(
            #          img_filepath, kernels, self.response_bins_per_axis,
            #          num_blocks, self.num_color_bins)
            #      for img_filepath in self.img_filepaths
            #  ]
            values = [
                dask.delayed(image_descriptor.compute_image_descriptor_from_filepath)(
                    img_filepath,
                    kernels,
                    self.response_bins_per_axis,
                    self.num_color_bins,
                )
                for img_filepath in self.img_filepaths
            ]

            with diagnostics.ProgressBar():
                feature_rows = dask.compute(*values)

            self._descr_feature_matrix = np.vstack(feature_rows)

            # TODO: cache as instance attribute (or even use property with and pass this
            # method's arguments to init), and then let people interactively choose the
            # number of PCA components until they're happy with the represented
            # variance? I vote yes.
            # TODO: cache this (via persistence): if `img_filepaths` and the technical
            # parameters coincide, load from a file instead of recomputing it
            # TODO: return copy?
            return self._descr_feature_matrix

    def train_test_split(
        self,
        *,
        method="cluster-II",
        n_components=12,
        num_img_clusters=4,
        train_prop=0.01,
        return_evr=False,
        pca_kwargs=None,
        kmeans_kwargs=None,
    ):
        """
        Select the image/tiles to be used for training.

        See the `background <https://bit.ly/2KlCICO>`_ example notebook
        for more details.

        Parameters
        ----------
        method : {'cluster-I', 'cluster-II'}, optional (default 'cluster-II')
            Method used in the train/test split.
        n_components : int, default 12
            Number of principal components into which the image descriptors should be
            represented when applying the *k*-means clustering.
        num_img_clusters : int, optional (default 4)
            Number of first-level image clusters of the 'cluster-II' `method`.  Ignored
            if `method` is 'cluster-I'.
        train_prop : float, optional
            Overall proportion of images/tiles that must be selected for training.
        return_evr : bool, optional (default False)
            Whether the explained variance ratio of the principal component
            analysis should be returned
        pca_kwargs : dict, optional
            Keyword arguments to be passed to the `sklearn.decomposition.PCA` class
            constructor (except for `n_components`).
        kmeans_kwargs : dict, optional
            Keyword arguments to be passed to the `sklearn.cluster.KMeans` class
            constructor (except for `n_clusters`).

        Returns
        -------
        split_df : pandas.DataFrame
            The train/test split data frame.
        evr : numeric, optional
            Expected variance ratio of the principal component analysis.
        """
        X = self.descr_feature_matrix
        if pca_kwargs is None:
            _pca_kwargs = {}
        else:
            _pca_kwargs = pca_kwargs.copy()
            # if `n_components` is provided in `pca_kwargs`, it will be ignored
            _ = _pca_kwargs.pop("n_components", None)
        pca = decomposition.PCA(n_components=n_components, **_pca_kwargs).fit(X)

        X_pca = pca.transform(X)
        X_cols = range(n_components)
        df = pd.concat(
            (
                pd.Series(self.img_filepaths, name="img_filename").apply(path.basename),
                pd.DataFrame(X_pca, columns=X_cols),
            ),
            axis=1,
        )

        if kmeans_kwargs is None:
            _kmeans_kwargs = {}
        else:
            _kmeans_kwargs = kmeans_kwargs.copy()
            # if `n_clusters` is provided in `kmeans_kwargs`, it will be ignored
            _ = _kmeans_kwargs.pop("n_clusters", None)
        if method == "cluster-I":
            km = cluster.KMeans(
                n_clusters=int(np.ceil(train_prop * len(df))), **_kmeans_kwargs
            ).fit(X_pca)
            closest, _ = metrics.pairwise_distances_argmin_min(
                km.cluster_centers_, df[X_cols]
            )
            train_idx = df.iloc[closest].index

            df["train"] = [True if i in train_idx else False for i in df.index]
        else:

            def cluster_train_test_split(img_cluster_ser):
                X_cluster_df = df.loc[img_cluster_ser.index, X_cols]
                # use `ceil` to avoid zeros, which might completely ignore a significant
                # image cluster
                num_train = int(np.ceil(train_prop * len(X_cluster_df)))
                cluster_km = cluster.KMeans(n_clusters=num_train, **_kmeans_kwargs).fit(
                    X_cluster_df
                )
                closest, _ = metrics.pairwise_distances_argmin_min(
                    cluster_km.cluster_centers_, X_cluster_df
                )
                train_idx = X_cluster_df.iloc[closest].index
                return [True if i in train_idx else False for i in X_cluster_df.index]

            df["img_cluster"] = cluster.KMeans(
                n_clusters=num_img_clusters, **_kmeans_kwargs
            ).fit_predict(X_pca)
            df["train"] = df.groupby("img_cluster")["img_cluster"].transform(
                cluster_train_test_split
            )

        split_df = df.drop(X_cols, axis=1)

        if return_evr:
            return split_df, pca.explained_variance_ratio_.sum()
        else:
            return split_df
