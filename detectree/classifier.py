import glob
from os import path

import dask
import maxflow as mf
import numpy as np
import rasterio as rio
from dask import diagnostics
from sklearn import ensemble

from . import pixel_features, pixel_response, settings, utils

__all__ = ['ClassifierTrainer', 'Classifier']

MOORE_NEIGHBORHOOD_ARR = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])


class ClassifierTrainer(object):
    def __init__(self, num_estimators=None, pixel_features_builder_kws={},
                 pixel_response_builder_kws={}, adaboost_kws={}):
        """
        Class to train a binary tree/non-tree classifier(s) of the image
        features. See the `background <https://bit.ly/2KlCICO>`_ example
        notebook for more details.

        Parameters
        ----------
        num_estimators : int, optional
            The maximum number of estimators at which boosting is terminated.
            Directly passed to the `n_estimators` keyword argument of
            `sklearn.ensemble.AdaBoostClassifier`. If no value is provided,
            the default value set in `settings.CLF_DEFAULT_NUM_ESTIMATORS`
            will be taken.
        pixel_features_builder_kws : dict, optional
            Keyword arguments that will be passed to
            `detectree.PixelFeaturesBuilder`, which customize how the pixel
            features are built.
        pixel_response_builder_kws : dict, optional
            Keyword arguments that will be passed to
            `detectree.PixelResponseBuilder`, which customize how the pixel
            tree/non-tree responses are built.
        adaboost_kws : dict, optional
            Keyword arguments that will be passed to
            `sklearn.ensemble.AdaBoostClassifier`.
        """

        super(ClassifierTrainer, self).__init__()

        if num_estimators is None:
            num_estimators = settings.CLF_DEFAULT_NUM_ESTIMATORS
        self.num_estimators = num_estimators

        self.pixel_features_builder_kws = pixel_features_builder_kws
        self.pixel_response_builder_kws = pixel_response_builder_kws
        self.adaboost_kws = adaboost_kws

    def train_classifier(self, split_df=None, response_img_dir=None,
                         img_filepaths=None, response_img_filepaths=None,
                         img_dir=None, img_filename_pattern=None, method=None,
                         img_cluster=None):
        """
        Train a classifier. See the `background <https://bit.ly/2KlCICO>`_
        example notebook for more details.

        Parameters
        ----------
        split_df : pandas DataFrame, optional
            Data frame with the train/test split
        response_img_dir : str representing path to a directory, optional
            Path to the directory where the response tiles are located.
            Required if providing `split_df`. Otherwise `response_img_dir`
            might either be ignored if providing `response_img_filepaths`, or
            be used as the directory where the images whose filename matches
            `img_filename_pattern` are to be located.
        img_filepaths : list-like, optional
            List of paths to the input tiles whose features will be used to
            train the classifier. Ignored if `split_df` is provided.
        response_img_filepaths : list-like, optional
            List of paths to the binary response tiles that will be used to
            train the classifier. Ignored if `split_df` is provided.
        img_dir : str representing path to a directory, optional
            Path to the directory where the images whose filename matches
            `img_filename_pattern` are to be located. Ignored if `split_df` or
            `img_filepaths` is provided.
        img_filename_pattern : str representing a file-name pattern, optional
            Filename pattern to be matched in order to obtain the list of
            images. If no value is provided, the default value set in
            `settings.IMG_DEFAULT_FILENAME_PATTERN` will be taken. Ignored if
            `split_df` or `img_filepaths` is provided.
        method : {'cluster-I', 'cluster-II'}, optional
            Method used in the train/test split
        img_cluster : int, optional
            The label of the cluster of tiles. Only used if `method` is
            'cluster-II'

        Returns
        -------
        clf : scikit-learn AdaBoostClassifier
            The trained classifier
        """
        if split_df is None and response_img_filepaths is None:
            # this is the only case that needs argument tweaking:
            # otherwise, if we pass `img_filepaths`/`img_dir` to
            # `build_features` and `response_img_dir` to `build_response`, the
            # latter would build a response with all the image files in
            # `response_img_dir`. Instead, we need to build the response only
            # for the files speficied in `img_filepaths`/`img_dir`
            if img_filepaths is None:
                # TODO: this is copied from `build_features` - ideally, we
                # should DRY it
                if img_filename_pattern is None:
                    img_filename_pattern = \
                        settings.IMG_DEFAULT_FILENAME_PATTERN
                if img_dir is None:
                    raise ValueError(
                        "Either `split_df`, `img_filepaths` or `img_dir` must "
                        "be provided")
                img_filepaths = glob.glob(
                    path.join(img_dir, img_filename_pattern))

            response_img_filepaths = [
                path.join(response_img_dir, path.basename(img_filepath))
                for img_filepath in img_filepaths
            ]

        X = pixel_features.PixelFeaturesBuilder(
            **self.pixel_features_builder_kws).build_features(
                split_df=split_df, img_filepaths=img_filepaths,
                img_dir=img_dir, img_filename_pattern=img_filename_pattern,
                method=method, img_cluster=img_cluster)

        y = pixel_response.PixelResponseBuilder(
            **self.pixel_response_builder_kws).build_response(
                split_df=split_df, response_img_dir=response_img_dir,
                response_img_filepaths=response_img_filepaths,
                img_filename_pattern=img_filename_pattern, method=method,
                img_cluster=img_cluster)

        clf = ensemble.AdaBoostClassifier(n_estimators=self.num_estimators,
                                          **self.adaboost_kws)
        clf.fit(X, y)

        return clf

    def train_classifiers(self, split_df, response_img_dir):
        """
        Train a classifier for each first-level cluster in `split_df`. See the
        `background <https://bit.ly/2KlCICO>`_ example notebook for more
        details.

        Parameters
        ----------
        split_df : pandas DataFrame
            Data frame with the train/test split, which must have an
            `img_cluster` column with the first-level cluster labels.
        response_img_dir : str representing path to a directory
            Path to the directory where the response tiles are located.

        Returns
        -------
        clf_dict : dictionary
            Dictionary mapping a scikit-learn AdaBoostClassifier to each
            first-level cluster label
        """
        if 'img_cluster' not in split_df:
            raise ValueError(
                "`split_df` must have an 'img_cluster' column ('cluster-II'). "
                "For 'cluster-I', use `train_classifier`.")

        clfs_lazy = {}
        for img_cluster, _ in split_df.groupby('img_cluster'):
            clfs_lazy[img_cluster] = dask.delayed(self.train_classifier)(
                split_df=split_df, response_img_dir=response_img_dir,
                method='cluster-II', img_cluster=img_cluster)

        with diagnostics.ProgressBar():
            clfs_dict = dask.compute(clfs_lazy)[0]

        return clfs_dict


class Classifier(object):
    def __init__(self, tree_val=None, nontree_val=None, refine=None,
                 refine_beta=None, refine_int_rescale=None,
                 **pixel_features_builder_kws):
        """
        Class use the trained classifier(s) to classify tree/non-tree pixels.
        See the `background <https://bit.ly/2KlCICO>`_ example notebook for
        more details.

        Parameters
        ----------
        tree_val : int, optional
            Label used to denote tree pixels in the predicted images. If no
            value is provided, the default value set in
            `settings.CLF_DEFAULT_TREE_VAL` will be taken.
        nontree_val : int, optional
            Label used to denote non-tree pixels in the predicted images. If
            no value is provided, the default value set in
            `settings.CLF_DEFAULT_NONTREE_VAL` will be taken.
        refine : bool, optional
            Whether the pixel-level classification should be refined by
            optimizing the consistence between neighboring pixels. If no value
            is provided, the default value set in `settings.CLF_DEFAULT_REFINE`
            will be taken.
        refine_beta : int, optional
            Parameter of the refinement procedure that controls the
            smoothness of the labelling. Larger values lead to smoother shapes.
            If no value is provided, the default value set in
            `settings.CLF_DEFAULT_REFINE_BETA` will be taken.
        refine_int_rescale : int, optional
            Parameter of the refinement procedure that controls the precision
            of the transformation of float to integer edge weights, required
            for the employed graph cuts algorithm. Larger values lead to
            greater precision. If no value is provided, the default value set
            in `settings.CLF_DEFAULT_REFINE_INT_RESCALE` will be taken.
        pixel_features_builder_kws : dict, optional
            Keyword arguments that will be passed to
            `detectree.PixelFeaturesBuilder`, which customize how the pixel
            features are built.
        """
        super(Classifier, self).__init__()

        if tree_val is None:
            tree_val = settings.CLF_DEFAULT_TREE_VAL
        if nontree_val is None:
            nontree_val = settings.CLF_DEFAULT_NONTREE_VAL
        if refine is None:
            refine = settings.CLF_DEFAULT_REFINE
        if refine_beta is None:
            refine_beta = settings.CLF_DEFAULT_REFINE_BETA
        if refine_int_rescale is None:
            refine_int_rescale = settings.CLF_DEFAULT_REFINE_INT_RESCALE

        self.tree_val = tree_val
        self.nontree_val = nontree_val
        self.refine = refine
        self.refine_beta = refine_beta
        self.refine_int_rescale = refine_int_rescale

        self.pixel_features_builder_kws = pixel_features_builder_kws

    def classify_img(self, img_filepath, clf, output_filepath=None):
        """
        Classify the image in `img_filepath` with the classifier `clf`, and
        optionally dump it to `output_filepath`.

        Parameters
        ----------
        img_filepath : str, file object or pathlib.Path object
            Path to a file, URI, file object opened in binary ('rb') mode, or
            a Path object representing the image to be classified. The value
            will be passed to `rasterio.open`
        clf : scikit-learn AdaBoostClassifier
            Trained classifier
        output_filepath : str, file object or pathlib.Path object, optional
            Path to a file, URI, file object opened in binary ('rb') mode, or
            a Path object representing where the predicted image is to be
            dumped. The value will be passed to `rasterio.open` in 'write' mode

        Returns
        -------
        y_pred : np.ndarray
            Array with the pixel responses
        """

        src = rio.open(img_filepath)
        img_shape = src.shape

        X = pixel_features.PixelFeaturesBuilder(
            **self.pixel_features_builder_kws).build_features_from_filepath(
                img_filepath)

        if not self.refine:
            y_pred = clf.predict(X).reshape(img_shape)
        else:
            p_nontree, p_tree = np.hsplit(clf.predict_proba(X), 2)
            g = mf.Graph[int]()
            node_ids = g.add_grid_nodes(img_shape)
            P_nontree = p_nontree.reshape(img_shape)
            P_tree = p_tree.reshape(img_shape)

            # The AdaBoost probabilities are floats between 0 and 1, and the
            # graph cuts algorithm requires an integer representation.
            # Therefore, we will multiply the probabilities by an arbitrary
            # large number and then transform the result to integers. For
            # instance, we could use a `refine_int_rescale` of `100` so that
            # the probabilities are rescaled into integers between 0 and 100
            # like percentages). The larger `refine_int_rescale`, the greater
            # the precision.
            # ACHTUNG: the data term when the pixel is a tree is
            # `log(1 - P_tree)`, i.e., `log(P_nontree)`, so the two lines
            # below are correct
            D_tree = (self.refine_int_rescale * np.log(P_nontree)).astype(int)
            D_nontree = (self.refine_int_rescale * np.log(P_tree)).astype(int)
            # TODO: option to choose Moore/Von Neumann neighborhood?
            g.add_grid_edges(node_ids, self.refine_beta,
                             structure=MOORE_NEIGHBORHOOD_ARR)
            g.add_grid_tedges(node_ids, D_tree, D_nontree)
            g.maxflow()
            # y_pred = g.get_grid_segments(node_ids)
            # transform boolean `g.get_grid_segments(node_ids)` to an array of
            # `self.tree_val` and `self.nontree_val`
            y_pred = np.full(img_shape, self.nontree_val)
            y_pred[g.get_grid_segments(node_ids)] = self.tree_val

        # TODO: make the profile of output rasters more customizable (e.g., via
        # the `settings` module)
        # output_filepath = path.join(output_dir,
        #                             f"tile_{tile_start}-{tile_end}.tif")
        if output_filepath is not None:
            with rio.open(output_filepath, 'w', driver='GTiff',
                          width=y_pred.shape[1], height=y_pred.shape[0],
                          count=1, dtype=np.uint8, nodata=self.nontree_val,
                          crs=src.crs, transform=src.transform) as dst:
                dst.write(y_pred.astype(np.uint8), 1)

        src.close()
        return y_pred

    def _classify_imgs(self, img_filepaths, clf, output_dir):
        pred_imgs_lazy = []
        pred_img_filepaths = []
        for img_filepath in img_filepaths:
            # filename, ext = path.splitext(path.basename(img_filepath))
            # pred_img_filepath = path.join(
            #     output_dir, f"{filename}-pred{ext}")
            pred_img_filepath = path.join(output_dir,
                                          path.basename(img_filepath))
            pred_imgs_lazy.append(
                dask.delayed(self.classify_img)(img_filepath, clf,
                                                pred_img_filepath))
            pred_img_filepaths.append(pred_img_filepath)

        with diagnostics.ProgressBar():
            dask.compute(*pred_imgs_lazy)

        return pred_img_filepaths

    def classify_imgs(
            self,
            split_df,
            output_dir,
            clf=None,
            clf_dict=None,
            method=None,
            img_cluster=None,
    ):
        """
        Classify thes image in `img_filepaths` with the classifier(s) `clf` or
        `clf_dict` (depending on the train/test split method) dump them to
        `output_dir`. See the `background <https://bit.ly/2KlCICO>`_ example
        notebook for more details.


        Parameters
        -------
        split_df : pandas DataFrame, optional
            Data frame with the train/test split.
        output_dir : str or pathlib.Path object
            Path to the directory where the predicted images are to be dumped.
        clf : scikit-learn AdaBoostClassifier
            Trained classifier.
        clf_dict : dictionary
            Dictionary mapping a trained scikit-learn AdaBoostClassifier to
            each first-level cluster label.
        method : {'cluster-I', 'cluster-II'}, optional
            Method used in the train/test split.
        img_cluster : int, optional
            The label of the cluster of tiles. Only used if `method` is
            'cluster-II'.

        Returns
        -------
        pred_imgs : list or dict
            File paths of the dumped tiles.
        """

        if method is None:
            if 'img_cluster' in split_df:
                method = 'cluster-II'
            else:
                method = 'cluster-I'

        if method == 'cluster-I':
            if clf is None:
                raise ValueError(
                    "If using 'cluster-I' method, `clf` must be provided")
            return self._classify_imgs(
                split_df[~split_df['train']]['img_filepath'], clf, output_dir)
        else:
            if img_cluster is not None:
                if clf is None:
                    if clf_dict is not None:
                        clf = clf_dict[img_cluster]
                    else:
                        raise ValueError(
                            "Either `clf` or `clf_dict` must be provided")

                return self._classify_imgs(
                    utils.get_img_filepaths(split_df, img_cluster, False), clf,
                    output_dir)

            if clf_dict is None:
                raise ValueError(
                    "If using 'cluster-II' method and not providing "
                    "`img_cluster`, `clf_dict` must be provided")
            pred_imgs = {}
            for img_cluster, img_cluster_df in split_df.groupby('img_cluster'):
                pred_imgs[img_cluster] = self._classify_imgs(
                    utils.get_img_filepaths(split_df, img_cluster, False),
                    clf_dict[img_cluster], output_dir)

            return pred_imgs
