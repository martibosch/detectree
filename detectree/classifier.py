"""Binary tree/non-tree classifier(s)."""

import glob
from os import path

import dask
import huggingface_hub as hf_hub
import maxflow as mf
import numpy as np
import rasterio as rio
import skops
from dask import diagnostics
from skops import io

from . import pixel_features, pixel_response, settings, utils

__all__ = ["ClassifierTrainer", "Classifier"]

MOORE_NEIGHBORHOOD_ARR = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])


class ClassifierTrainer:
    """Train binary tree/non-tree classifier(s) of the pixel features."""

    def __init__(
        self,
        *,
        sigmas=None,
        num_orientations=None,
        neighborhood=None,
        min_neighborhood_range=None,
        num_neighborhoods=None,
        tree_val=None,
        nontree_val=None,
        classifier_class=None,
        **classifier_kwargs,
    ):
        """
        Initialize the classifier.

        See the `background <https://bit.ly/2KlCICO>`_ example notebook for details.

        Parameters
        ----------
        sigmas : list-like, optional
            The list of scale parameters (sigmas) to build the Gaussian filter bank that
            will be used to compute the pixel-level features. The provided argument will
            be passed to the initialization method of the `PixelFeaturesBuilder` class.
            If no value is provided, the value set in `settings.GAUSS_SIGMAS` will be
            taken.
        num_orientations : int, optional
            The number of equally-distributed orientations to build the Gaussian filter
            bank that will be used to compute the pixel-level features. The provided
            argument will be passed to the initialization method of the
            `PixelFeaturesBuilder` class. If no value is provided, the value set in
            `settings.GAUSS_NUM_ORIENTATIONS` is used.
        neighborhood : array-like, optional
            The base neighborhood structure that will be used to compute the entropy
            features. Theprovided argument will be passed to the initialization method
            of the `PixelFeaturesBuilder` class. If no value is provided, a square with
            a side size of `2 * min_neighborhood_range + 1` is used.
        min_neighborhood_range : int, optional
            The range (i.e., the square radius) of the smallest neighborhood window that
            will be used to compute the entropy features. The provided argument will be
            passed to the initialization method of the `PixelFeaturesBuilder` class. If
            no value is provided, the value set in
            `settings.ENTROPY_MIN_NEIGHBORHOOD_RANGE` is used.
        num_neighborhoods : int, optional
            The number of neighborhood windows (whose size follows a geometric
            progression starting at `min_neighborhood_range`) that will be used to
            compute the entropy features. The provided argument will be passed to the
            initialization method of the `PixelFeaturesBuilder` class. If no value is
            provided, the value set in `settings.ENTROPY_NUM_NEIGHBORHOODS` is used.
        tree_val : int, optional
            The value that designates tree pixels in the response images. The provided
            argument will be passed to the initialization method of the
            `PixelResponseBuilder` class. If no value is provided, the value set in
            `settings.RESPONSE_TREE_VAL` is used.
        nontree_val : int, optional
            The value that designates non-tree pixels in the response images. The
            provided argument will be passed to the initialization method of the
            `PixelResponseBuilder` class. If no value is provided, the value set in
            `settings.RESPONSE_NONTREE_VAL` is used.
        classifier_class : class, optional
            The class of the classifier to be trained. It can be any scikit-learn
            compatible estimator that implements the `fit`, `predict` and
            `predict_proba` methods and that can be saved to and loaded from memory
            using skops. If no value is provided, the value set in `settings.CLF_CLASS`
            is used.
        classifier_kwargs : key-value pairings, optional
            Keyword arguments that will be passed to the initialization of
            `classifier_class`. If no value is provided, the value set in
            `settings.CLF_KWARGS` is used.
        """
        self.pixel_features_builder_kwargs = dict(
            sigmas=sigmas,
            num_orientations=num_orientations,
            neighborhood=neighborhood,
            min_neighborhood_range=min_neighborhood_range,
            num_neighborhoods=num_neighborhoods,
        )
        self.pixel_response_builder_kwargs = dict(
            tree_val=tree_val, nontree_val=nontree_val
        )
        if classifier_class is None:
            classifier_class = settings.CLF_CLASS
        self.classifier_class = classifier_class
        if classifier_kwargs == {}:
            classifier_kwargs = settings.CLF_KWARGS
        self.classifier_kwargs = classifier_kwargs

    def train_classifier(
        self,
        *,
        split_df=None,
        img_dir=None,
        response_img_dir=None,
        img_filepaths=None,
        response_img_filepaths=None,
        img_filename_pattern=None,
        method=None,
        img_cluster=None,
    ):
        """
        Train a classifier.

        See the `background <https://bit.ly/2KlCICO>`_ example notebook for more
        details.

        Parameters
        ----------
        split_df : pandas DataFrame, optional
            Data frame with the train/test split.
        img_dir : str representing path to a directory, optional
            Path to the directory where the images from `split_df` or whose filename
            matches `img_filename_pattern` are located. Required if `split_df` is
            provided. Ignored if `img_filepaths` is provided.
        response_img_dir : str representing path to a directory, optional
            Path to the directory where the response tiles are located. Required if
            providing `split_df`. Otherwise `response_img_dir` might either be ignored
            if providing `response_img_filepaths`, or be used as the directory where the
            images whose filename matches `img_filename_pattern` are to be located.
        img_filepaths : list-like, optional
            List of paths to the input tiles whose features will be used to train the
            classifier. Ignored if `split_df` is provided.
        response_img_filepaths : list-like, optional
            List of paths to the binary response tiles that will be used to train the
            classifier. Ignored if `split_df` is provided.
        img_filename_pattern : str representing a file-name pattern, optional
            Filename pattern to be matched in order to obtain the list of images. If no
            value is provided, the value set in `settings.IMG_FILENAME_PATTERN` is used.
            Ignored if `split_df` or `img_filepaths` is provided.
        method : {'cluster-I', 'cluster-II'}, optional
            Method used in the train/test split.
        img_cluster : int, optional
            The label of the cluster of tiles. Only used if `method` is 'cluster-II'.

        Returns
        -------
        clf : scikit-learn-like classifier
            The trained classifier.
        """
        if split_df is None and response_img_filepaths is None:
            # this is the only case that needs argument tweaking: otherwise, if we pass
            # `img_filepaths`/`img_dir` to `build_features` and `response_img_dir` to
            # `build_response`, the latter would build a response with all the image
            # files in `response_img_dir`. Instead, we need to build the response only
            # for the files specified in `img_filepaths`/`img_dir`
            if img_filepaths is None:
                # TODO: this is copied from `build_features` - ideally, we should DRY it
                if img_filename_pattern is None:
                    img_filename_pattern = settings.IMG_FILENAME_PATTERN
                if img_dir is None:
                    raise ValueError(
                        "Either `split_df`, `img_filepaths` or `img_dir` must "
                        "be provided"
                    )
                img_filepaths = glob.glob(path.join(img_dir, img_filename_pattern))

            response_img_filepaths = [
                path.join(response_img_dir, path.basename(img_filepath))
                for img_filepath in img_filepaths
            ]

        X = pixel_features.PixelFeaturesBuilder(
            **self.pixel_features_builder_kwargs
        ).build_features(
            split_df=split_df,
            img_filepaths=img_filepaths,
            img_dir=img_dir,
            img_filename_pattern=img_filename_pattern,
            method=method,
            img_cluster=img_cluster,
        )

        y = pixel_response.PixelResponseBuilder(
            **self.pixel_response_builder_kwargs
        ).build_response(
            split_df=split_df,
            response_img_dir=response_img_dir,
            response_img_filepaths=response_img_filepaths,
            img_filename_pattern=img_filename_pattern,
            method=method,
            img_cluster=img_cluster,
        )

        clf = self.classifier_class(**self.classifier_kwargs)
        clf.fit(X, y)

        return clf

    def train_classifiers(self, split_df, img_dir, response_img_dir):
        """
        Train a classifier for each first-level cluster in `split_df`.

        See the `background <https://bit.ly/2KlCICO>`_ example notebook for more
        details.

        Parameters
        ----------
        split_df : pandas DataFrame
            Data frame with the train/test split, which must have an `img_cluster`.
            column with the first-level cluster labels.
        img_dir : str representing path to a directory
            Path to the directory where the images from `split_df` or whose filename
            matches `img_filename_pattern` are located. Required if `split_df` is
            provided. Ignored if `img_filepaths` is provided.
        response_img_dir : str representing path to a directory
            Path to the directory where the response tiles are located.

        Returns
        -------
        clf_dict : dictionary
            Dictionary mapping a scikit-learn-like classifier to each first-level
            cluster label.
        """
        if "img_cluster" not in split_df:
            raise ValueError(
                "`split_df` must have an 'img_cluster' column ('cluster-II'). "
                "For 'cluster-I', use `train_classifier`."
            )

        clfs_lazy = {}
        for img_cluster, _ in split_df.groupby("img_cluster"):
            clfs_lazy[img_cluster] = dask.delayed(self.train_classifier)(
                split_df=split_df,
                img_dir=img_dir,
                response_img_dir=response_img_dir,
                method="cluster-II",
                img_cluster=img_cluster,
            )

        with diagnostics.ProgressBar():
            clfs_dict = dask.compute(clfs_lazy)[0]

        return clfs_dict


class Classifier:
    """Use trained classifier(s) to predict tree pixels."""

    def __init__(
        self,
        *,
        clf=None,
        clf_dict=None,
        tree_val=None,
        nontree_val=None,
        refine=None,
        refine_beta=None,
        refine_int_rescale=None,
        **pixel_features_builder_kwargs,
    ):
        """
        Initialize the classifier instance.

        See the `background <https://bit.ly/2KlCICO>`_ example notebook for more
        details.

        Parameters
        ----------
        clf : scikit-learn-like classifier, optional
            Trained classifier. If no value is provided, the latest detectree
            pre-trained classifier is used. Ignored if `clf_dict` is provided.
        clf_dict : dictionary, optional
            Dictionary mapping a trained scikit-learn-like classifier to each
            first-level cluster label.
        tree_val : int, optional
            Label used to denote tree pixels in the predicted images. If no value is
            provided, the value set in `settings.CLF_TREE_VAL` is used.
        nontree_val : int, optional
            Label used to denote non-tree pixels in the predicted images. If no value is
            provided, the value set in `settings.CLF_NONTREE_VAL` is used.
        refine : bool, optional
            Whether the pixel-level classification should be refined by optimizing the
            consistence between neighboring pixels. If no value is provided, the value
            set in `settings.CLF_REFINE` is used.
        refine_beta : int, optional
            Parameter of the refinement procedure that controls the smoothness of the
            labelling. Larger values lead to smoother shapes.  If no value is provided,
            the value set in `settings.CLF_REFINE_BETA` is used.
        refine_int_rescale : int, optional
            Parameter of the refinement procedure that controls the precision of the
            transformation of float to integer edge weights, required for the employed
            graph cuts algorithm. Larger values lead to greater precision. If no value
            is provided, the value set in `settings.CLF_REFINE_INT_RESCALE` is used.
        pixel_features_builder_kwargs : dict, optional
            Keyword arguments that will be passed to `detectree.PixelFeaturesBuilder`,
            which customize how the pixel features are built.
        """
        super().__init__()

        if clf_dict is not None:
            self.clf_dict = clf_dict
        elif clf is not None:
            self.clf = clf
        else:
            self.clf = io.load(
                hf_hub.hf_hub_download(
                    repo_id=settings.HF_HUB_REPO_ID,
                    filename=settings.HF_HUB_FILENAME,
                    library_name="skops",
                    library_version=skops.__version__,
                ),
                trusted=settings.SKOPS_TRUSTED,
            )

        if tree_val is None:
            tree_val = settings.CLF_TREE_VAL
        if nontree_val is None:
            nontree_val = settings.CLF_NONTREE_VAL
        if refine is None:
            refine = settings.CLF_REFINE
        if refine_beta is None:
            refine_beta = settings.CLF_REFINE_BETA
        if refine_int_rescale is None:
            refine_int_rescale = settings.CLF_REFINE_INT_RESCALE

        self.tree_val = tree_val
        self.nontree_val = nontree_val
        self.refine = refine
        self.refine_beta = refine_beta
        self.refine_int_rescale = refine_int_rescale

        self.pixel_features_builder_kwargs = pixel_features_builder_kwargs

    def _predict_img(self, img_filepath, clf, *, output_filepath=None):
        # ACHTUNG: Note that we do not use keyword-only arguments in this method because
        # `output_filepath` works as the only "optional" argument
        src = rio.open(img_filepath)
        img_shape = src.shape

        X = pixel_features.PixelFeaturesBuilder(
            **self.pixel_features_builder_kwargs
        ).build_features_from_filepath(img_filepath)

        if not self.refine:
            y_pred = clf.predict(X).reshape(img_shape)
        else:
            p_nontree, p_tree = np.hsplit(clf.predict_proba(X), 2)
            g = mf.Graph[int]()
            node_ids = g.add_grid_nodes(img_shape)
            P_nontree = p_nontree.reshape(img_shape)
            P_tree = p_tree.reshape(img_shape)

            # The classifier probabilities are floats between 0 and 1, and the graph
            # cuts algorithm requires an integer representation. Therefore, we multiply
            # the probabilities by an arbitrary large number and then transform the
            # result to integers. For instance, we could use a `refine_int_rescale` of
            # `100` so that the probabilities are rescaled into integers between 0 and
            # 100 like percentages). The larger `refine_int_rescale`, the greater the
            # precision.
            # ACHTUNG: the data term when the pixel is a tree is `log(1 - P_tree)`,
            # i.e., `log(P_nontree)`, so the two lines below are correct
            D_tree = (self.refine_int_rescale * np.log(P_nontree)).astype(int)
            D_nontree = (self.refine_int_rescale * np.log(P_tree)).astype(int)
            # TODO: option to choose Moore/Von Neumann neighborhood?
            g.add_grid_edges(
                node_ids, self.refine_beta, structure=MOORE_NEIGHBORHOOD_ARR
            )
            g.add_grid_tedges(node_ids, D_tree, D_nontree)
            g.maxflow()
            # y_pred = g.get_grid_segments(node_ids)
            # transform boolean `g.get_grid_segments(node_ids)` to an array of
            # `self.tree_val` and `self.nontree_val`
            y_pred = np.full(img_shape, self.nontree_val)
            y_pred[g.get_grid_segments(node_ids)] = self.tree_val

        # TODO: make the profile of output rasters more customizable (e.g., via the
        # `settings` module)
        # output_filepath = path.join(output_dir,
        #                             f"tile_{tile_start}-{tile_end}.tif")
        if output_filepath is not None:
            with rio.open(
                output_filepath,
                "w",
                driver="GTiff",
                width=y_pred.shape[1],
                height=y_pred.shape[0],
                count=1,
                dtype=np.uint8,
                nodata=self.nontree_val,
                crs=src.crs,
                transform=src.transform,
            ) as dst:
                dst.write(y_pred.astype(np.uint8), 1)

        src.close()
        return y_pred

    def _predict_imgs(self, img_filepaths, clf, output_dir):
        pred_imgs_lazy = []
        pred_img_filepaths = []
        for img_filepath in img_filepaths:
            # filename, ext = path.splitext(path.basename(img_filepath))
            # pred_img_filepath = path.join(
            #     output_dir, f"{filename}-pred{ext}")
            pred_img_filepath = path.join(output_dir, path.basename(img_filepath))
            pred_imgs_lazy.append(
                dask.delayed(self._predict_img)(
                    img_filepath, clf, output_filepath=pred_img_filepath
                )
            )
            pred_img_filepaths.append(pred_img_filepath)

        with diagnostics.ProgressBar():
            dask.compute(*pred_imgs_lazy)

        return pred_img_filepaths

    def predict_img(self, img_filepath, *, img_cluster=None, output_filepath=None):
        """
        Use a trained classifier to predict tree pixels in an image.

        Optionally dump the predicted tree/non-tree image to `output_filepath`.

        Parameters
        ----------
        img_filepath : str, file object or pathlib.Path object
            Path to a file, URI, file object opened in binary ('rb') mode, or a Path
            object representing the image to be classified. The value will be passed to
            `rasterio.open`.
        img_cluster : int, optional
            The label of the cluster of tiles. Only used if the `Classifier` instance
            was initialized with `clf_dict` (i.e., "cluster-II" method).
        output_filepath : str, file object or pathlib.Path object, optional
            Path to a file, URI, file object opened in binary ('rb') mode, or a Path
            object representing where the predicted image is to be dumped. The value
            will be passed to `rasterio.open` in 'write' mode.

        Returns
        -------
        y_pred : numpy ndarray
            Array with the pixel responses.
        """
        clf = getattr(self, "clf", None)
        if clf is None:
            if img_cluster is not None:
                try:
                    clf = self.clf_dict[img_cluster]
                except KeyError:
                    raise ValueError(
                        f"Classifier for cluster {img_cluster} not found in"
                        " `self.clf_dict`."
                    )
            else:
                raise ValueError(
                    "A valid `img_cluster` must be provided for classifiers"
                    " instantiated with `clf_dict`."
                )
        return self._predict_img(img_filepath, clf, output_filepath=output_filepath)

    def predict_imgs(self, split_df, img_dir, output_dir):
        """
        Use trained classifier(s) to predict tree pixels in multiple images.

        See the `background <https://bit.ly/2KlCICO>`_ example notebook for more
        details.

        Parameters
        ----------
        split_df : pandas DataFrame, optional
            Data frame with the train/test split.
        img_dir : str representing path to a directory
            Path to the directory where the images from `split_df` or whose filename
            matches `img_filename_pattern` are located. Required if `split_df` is
            provided. Ignored if `img_filepaths` is provided.
        output_dir : str or pathlib.Path object
            Path to the directory where the predicted images are to be dumped.

        Returns
        -------
        pred_imgs : list or dict
            File paths of the dumped tiles.
        """
        if hasattr(self, "clf"):
            return self._predict_imgs(
                split_df[~split_df["train"]]["img_filename"].apply(
                    lambda img_filename: path.join(img_dir, img_filename)
                ),
                self.clf,
                output_dir,
            )
        else:
            # `self.clf_dict` is not `None`
            pred_imgs = {}
            for img_cluster, _ in split_df.groupby("img_cluster"):
                try:
                    clf = self.clf_dict[img_cluster]
                except KeyError:
                    raise ValueError(
                        f"Classifier for cluster {img_cluster} not found in"
                        " `self.clf_dict`."
                    )
                pred_imgs[img_cluster] = self._predict_imgs(
                    utils.get_img_filename_ser(split_df, img_cluster, False).apply(
                        lambda img_filename: path.join(img_dir, img_filename)
                    ),
                    clf,
                    output_dir,
                )

            return pred_imgs
