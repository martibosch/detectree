"""Binary tree/non-tree classifier(s)."""

from os import path

import dask
import huggingface_hub as hf_hub
import numpy as np
import rasterio as rio
import skops
from dask import diagnostics
from skops import io

from detectree import evaluate, pixel_features, pixel_response, settings, utils

__all__ = ["PixelDatasetTransformer", "ClassifierTrainer", "Classifier"]


class PixelDatasetTransformer:
    """Build pixel features and responses for training."""

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
        tree_val, nontree_val : int, optional
            The values that designate tree and non-tree pixels respectively in the
            response images. The provided arguments will be passed to the initialization
            method of the `PixelResponseBuilder` class. If no values are provided, the
            values set in `settings.TREE_VAL` and `settings.NON_TREE_VAL` are
            respectively used.
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
        if not classifier_kwargs:
            classifier_kwargs = settings.CLF_KWARGS
        self.classifier_kwargs = classifier_kwargs
        self.pixel_features_builder = pixel_features.PixelFeaturesBuilder(
            **self.pixel_features_builder_kwargs
        )
        self.pixel_response_builder = pixel_response.PixelResponseBuilder(
            **self.pixel_response_builder_kwargs
        )

    def fit(self, X=None, y=None, **kwargs):  # noqa: ARG002
        """Fit method for sklearn compatibility."""
        return self

    def transform(
        self,
        *,
        split_df=None,
        img_dir=None,
        response_img_dir=None,
        img_filepaths=None,
        response_img_filepaths=None,
        img_filename_pattern=None,
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

        Returns
        -------
        X : numpy ndarray
            Array with the pixel features.
        y : numpy ndarray
            Array with the pixel responses.
        """
        if split_df is None and response_img_filepaths is None:
            # this is the only case that needs argument tweaking: otherwise, if we pass
            # `img_filepaths`/`img_dir` to `build_features` and `response_img_dir` to
            # `build_response`, the latter would build a response with all the image
            # files in `response_img_dir`. Instead, we need to build the response only
            # for the files specified in `img_filepaths`/`img_dir`
            if img_filepaths is None:
                if img_dir is None:
                    raise ValueError(
                        "Either `split_df`, `img_filepaths` or `img_dir` must be"
                        " provided"
                    )
                img_filepaths = utils.get_img_filepaths(
                    img_dir, img_filename_pattern=img_filename_pattern
                )

            if response_img_dir is None:
                raise ValueError(
                    "Either `split_df`, `response_img_filepaths` or "
                    "`response_img_dir` must be provided"
                )
            response_img_filepaths = [
                path.join(response_img_dir, path.basename(img_filepath))
                for img_filepath in img_filepaths
            ]

        X = self.pixel_features_builder.build_features(
            split_df=split_df,
            img_filepaths=img_filepaths,
            img_dir=img_dir,
            img_filename_pattern=img_filename_pattern,
        )

        y = self.pixel_response_builder.build_response(
            split_df=split_df,
            response_img_dir=response_img_dir,
            response_img_filepaths=response_img_filepaths,
            img_filename_pattern=img_filename_pattern,
        )

        return X, y

    def fit_transform(self, *args, **kwargs):
        """Fit and transform method for sklearn compatibility."""
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)


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
        tree_val, nontree_val : int, optional
            The values that designate tree and non-tree pixels respectively in the
            response images. The provided arguments will be passed to the initialization
            method of the `PixelResponseBuilder` class. If no values are provided, the
            values set in `settings.TREE_VAL` and `settings.NON_TREE_VAL` are
            respectively used.
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
        if not classifier_kwargs:
            classifier_kwargs = settings.CLF_KWARGS
        self.classifier_kwargs = classifier_kwargs
        self.pixel_training_transformer = PixelDatasetTransformer(
            **self.pixel_features_builder_kwargs,
            **self.pixel_response_builder_kwargs,
        )

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
        if split_df is not None and method is None:
            if "img_cluster" in split_df:
                method = "cluster-II"
            else:
                method = "cluster-I"

        if split_df is not None and method == "cluster-I" and "img_cluster" in split_df:
            split_df = split_df.drop(columns=["img_cluster"])
        elif split_df is not None and method == "cluster-II":
            if img_cluster is None:
                raise ValueError(
                    "If `method` is 'cluster-II', `img_cluster` must be provided"
                )
            if img_dir is None:
                raise ValueError(
                    "If `split_df` is provided, `img_dir` must also be provided"
                )
            img_filename_ser = utils.get_img_filename_ser(split_df, img_cluster, True)
            img_filepaths = img_filename_ser.apply(
                lambda img_filename: path.join(img_dir, img_filename)
            )
            split_df = None

        X, y = self.pixel_training_transformer.fit_transform(
            split_df=split_df,
            img_dir=img_dir,
            response_img_dir=response_img_dir,
            img_filepaths=img_filepaths,
            response_img_filepaths=response_img_filepaths,
            img_filename_pattern=img_filename_pattern,
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
        hf_hub_repo_id=None,
        hf_hub_clf_filename=None,
        hf_hub_download_kwargs=None,
        skops_trusted=None,
        tree_val=None,
        nontree_val=None,
        refine_method=None,
        refine_kwargs=None,
        return_proba=None,
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
        hf_hub_repo_id, hf_hub_clf_filename : str, optional
            HuggingFace Hub repository id (string with the user or organization and
            repository name separated by a `/`) and file name of the skops classifier
            respectively. If no value is provided, the values set in
            `settings.HF_HUB_REPO_ID` and `settings.HF_HUB_CLF_FILENAME` Ignored if
            `clf` or `clf_dict` are provided.
        hf_hub_download_kwargs : dict, optional
            Additional keyword arguments (besides "repo_id", "filename", "library_name"
            and "library_version") to pass to `huggingface_hub.hf_hub_download`.
        skops_trusted : list, optional
            List of trusted object types to load the classifier from HuggingFace Hub,
            passed to `skops.io.load`. If no value is provided, the value from
            `settings.SKOPS_TRUSTED` is used. Ignored if `clf` or `clf_dict` are
            provided.
        tree_val, nontree_val : int, optional
            The values that designate tree and non-tree pixels respectively in the
            response images. If no values are provided, the values set in
            `settings.TREE_VAL` and `settings.NON_TREE_VAL` are respectively used.
        refine_method : callable or bool, optional
            Method to refine the pixel-level classification, e.g., to optimize the
            consistence between neighboring pixels. If `False` is provided, no
            refinement is performed. If `None` is provided and `return_proba` is `None`
            or `False`, the value from `settings.CLF_REFINE` is used.
        refine_method_kwargs : dict, optional
            Keyword arguments that will be passed to the `refine_method`. If no value
            is provided, the value set in `settings.CLF_REFINE_KWARGS` is used. Ignored
            if no refinement is performed (i.e., `refine_method` is `False` or
            `refine_method` is `None` and `return_proba` is `True`).
        return_proba : bool, optional
            If `True`, the classifier will return the probabilities of each pixel
            belonging to the tree class. If `False`, the classifier will return the
            predicted class labels. Ignored if a valid `refine_method` is provided.
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
            if hf_hub_repo_id is None:
                hf_hub_repo_id = settings.HF_HUB_REPO_ID
            if hf_hub_clf_filename is None:
                hf_hub_clf_filename = settings.HF_HUB_CLF_FILENAME
            if hf_hub_download_kwargs is None:
                _hf_hub_download_kwargs = {}
            else:
                _hf_hub_download_kwargs = hf_hub_download_kwargs.copy()
                for key in [
                    "repo_id",
                    "filename",
                    "library_name",
                    "library_version",
                ]:
                    _ = _hf_hub_download_kwargs.pop(key, None)
            if skops_trusted is None:
                skops_trusted = settings.SKOPS_TRUSTED
            self.clf = io.load(
                hf_hub.hf_hub_download(
                    repo_id=hf_hub_repo_id,
                    filename=hf_hub_clf_filename,
                    library_name="skops",
                    library_version=skops.__version__,
                    **_hf_hub_download_kwargs,
                ),
                trusted=skops_trusted,
            )

        if tree_val is None:
            tree_val = settings.TREE_VAL
        if nontree_val is None:
            nontree_val = settings.NONTREE_VAL
        if refine_method is None and not return_proba:
            refine_method = settings.CLF_REFINE_METHOD
        if refine_method:
            if refine_kwargs is None:
                refine_kwargs = settings.CLF_REFINE_KWARGS
            self.refine_method = refine_method
            self.refine_kwargs = refine_kwargs
            self._predict_X = self._predict_X_refine
            self.tree_val = tree_val
            self.nontree_val = nontree_val
            self.dst_nodata = nontree_val
        else:
            if return_proba is None:
                # we will only get here if `refine_method` is `False`
                return_proba = settings.CLF_RETURN_PROBA
            if return_proba:
                # there is no refine method, return proba
                self._predict_X = self._predict_X_proba
                # TODO: how to manage this better?
                self.dst_nodata = -1
            else:
                # there is no refine method, return labels
                self._predict_X = self._predict_X_labels
                self.dst_nodata = nontree_val

        self.tree_val = tree_val
        self.nontree_val = nontree_val

        self.pixel_features_builder_kwargs = pixel_features_builder_kwargs
        self.pixel_training_transformer = PixelDatasetTransformer(
            tree_val=tree_val,
            nontree_val=nontree_val,
            **self.pixel_features_builder_kwargs,
        )
        self.pixel_features_builder = (
            self.pixel_training_transformer.pixel_features_builder
        )

    def _predict_X_refine(self, X, clf, img_shape):
        # TODO: properly manage the order classes in `clf`, i.e., are we sure that
        # "tree" is always the second class? If so, we could probably fully omit
        # `tree_val` and `nontree_val` and get them from `clf.classes_`
        # p_nontree_img, p_tree_img = np.hsplit(clf.pred_proba(X), 2)
        p_tree_img = clf.predict_proba(X)[:, 1].reshape(img_shape)
        return self.refine_method(
            p_tree_img, self.tree_val, self.nontree_val, **self.refine_kwargs
        ).astype(np.uint8)

    def _predict_X_proba(self, X, clf, img_shape):
        return clf.predict_proba(X)[:, 1].reshape(img_shape)

    def _predict_X_labels(self, X, clf, img_shape):
        return clf.predict(X).reshape(img_shape).astype(np.uint8)

    def _predict_img(self, img_filepath, clf, *, output_filepath=None):
        # ACHTUNG: Note that we do not use keyword-only arguments in this method because
        # `output_filepath` works as the only "optional" argument
        src = rio.open(img_filepath)
        img_shape = src.shape

        X = self.pixel_features_builder.build_features_from_filepath(img_filepath)

        y_pred = self._predict_X(X, clf, img_shape)

        # TODO: make the profile of output rasters more customizable (e.g., via the
        # `settings` module)
        # output_filepath = path.join(output_dir,
        #                             f"tile_{tile_start}-{tile_end}.tif")
        if output_filepath is not None:
            with rio.open(
                output_filepath,
                "w",
                # driver="GTiff",
                width=y_pred.shape[1],
                height=y_pred.shape[0],
                count=1,
                dtype=y_pred.dtype,
                nodata=self.dst_nodata,
                crs=src.crs,
                transform=src.transform,
            ) as dst:
                dst.write(y_pred, 1)

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
                except KeyError as exc:
                    raise ValueError(
                        f"Classifier for cluster {img_cluster} not found in"
                        " `self.clf_dict`."
                    ) from exc
            else:
                raise ValueError(
                    "A valid `img_cluster` must be provided for classifiers"
                    " instantiated with `clf_dict`."
                )
        return self._predict_img(img_filepath, clf, output_filepath=output_filepath)

    def predict_imgs(
        self,
        output_dir,
        *,
        split_df=None,
        img_dir=None,
        img_filepaths=None,
        img_filename_pattern=None,
    ):
        """
        Use trained classifier(s) to predict tree pixels in multiple images.

        See the `background <https://bit.ly/2KlCICO>`_ example notebook for more
        details.

        Parameters
        ----------
        output_dir : str or pathlib.Path object
            Path to the directory where the predicted images are to be dumped.
        split_df : pandas DataFrame, optional
            Data frame with the train/test split.
        img_dir : str representing path to a directory, optional
            Path to the directory where the images from `split_df` are located. Required
            if `split_df` is provided. Ignored if `img_filepaths` is provided.
        img_filepaths : list-like, optional
            List of paths to the tiles that will be used for validation. Ignored if
            `split_df` is provided.
        img_filename_pattern : str representing a file-name pattern, optional
            Filename pattern to be matched in order to obtain the list of images. If no
            value is provided, the value set in `settings.IMG_FILENAME_PATTERN` is used.
            Ignored if `split_df` or `img_filepaths` is provided.

        Returns
        -------
        pred_imgs : list
            File paths of the dumped tiles.
        """
        if hasattr(self, "clf"):
            # predicting with a single classifier
            if split_df is None:
                if img_filepaths is None:
                    if img_dir is None:
                        raise ValueError(
                            "Either `split_df`, `img_filepaths` or `img_dir` must be"
                            " provided."
                        )
                    img_filepaths = utils.get_img_filepaths(
                        img_dir,
                        img_filename_pattern=img_filename_pattern,
                    )
            else:
                img_filepaths = split_df["img_filename"].apply(
                    lambda img_filename: path.join(img_dir, img_filename)
                )
            pred_imgs = self._predict_imgs(
                img_filepaths,
                self.clf,
                output_dir,
            )

        else:
            # `self.clf_dict` is not `None`
            # predicting with multiple classifiers
            pred_imgs = []
            for img_cluster, _ in split_df.groupby("img_cluster"):
                try:
                    clf = self.clf_dict[img_cluster]
                except KeyError as exc:
                    raise ValueError(
                        f"Classifier for cluster {img_cluster} not found in"
                        " `self.clf_dict`."
                    ) from exc
                pred_imgs += self._predict_imgs(
                    utils.get_img_filename_ser(split_df, img_cluster, False).apply(
                        lambda img_filename: path.join(img_dir, img_filename)
                    ),
                    clf,
                    output_dir,
                )

        return pred_imgs

    def compute_eval_metrics(
        self,
        metrics=None,
        metrics_kwargs=None,
        refine_method=None,
        refine_kwargs=None,
        split_df=None,
        img_dir=None,
        response_img_dir=None,
        img_filepaths=None,
        response_img_filepaths=None,
        img_filename_pattern=None,
    ):
        """
        Compute evaluation metrics for validation images.

        Parameters
        ----------
        metrics : str, func or list of str or func
            The metrics to compute, must be either a string with a function of the
            `sklearn.metrics`, a function that takes a `y_true` and `y_pred` positional
            arguments with the true and predicted labels respectively or a list-like of
            any of the two options. If no value is provided, the values set in
            `settings.EVAL_METRICS` are used.
        metrics_kwargs : dict or list of dict
            Additional keyword arguments to pass to each of the metric functions.
        refine_method : callable or bool, optional
            Method to refine the pixel-level classification. If `False` is provided, no
            refinement is performed. If any non-None value is provided, it overrides the
            `refine_method` argument provided at instantiation time. If `None` is
            provided, the value from `self.refine_method` is used if set, otherwise no
            refinement is performed.
        refine_kwargs : dict, optional
            Keyword arguments that will be passed to `refine_method`. If any non-None
            value is provided, it overrides the `refine_kwargs` argument provided at
            instantiation time. If `None` is provided, the value from
            `self.refine_kwargs` is used if set. Ignored if no refinement is performed.
        split_df : pandas DataFrame, optional
            Data frame with the validation images.
        img_dir : str representing path to a directory, optional
            Path to the directory where the images from `split_df` are located. Required
            if `split_df` is provided. Ignored if `img_filepaths` is provided.
        response_img_dir : str representing path to a directory, optional
            Path to the directory where the response tiles are located. Ignored if
            providing `response_img_filepaths`. Only images with a matching response (by
            basename) are evaluated.
        img_filepaths : list-like, optional
            List of paths to the tiles that will be used for validation. Ignored if
            `split_df` is provided.
        response_img_filepaths : list-like, optional
            List of paths to the binary response tiles that will be used for evaluation.
            Ignored if `split_df` is provided. Only images with a matching response (by
            basename) are evaluated.
        img_filename_pattern : str representing a file-name pattern, optional
            Filename pattern to be matched in order to obtain the list of images. If no
            value is provided, the value set in `settings.IMG_FILENAME_PATTERN` is used.
            Ignored if `split_df` or `img_filepaths` is provided.

        Returns
        -------
        metric_dict : numeric, dict
            Values of the metrics computed for the validation images. If only one metric
            is provided, a single value is returned. If multiple metrics are provided, a
            dict with a key for each metric is returned. The metric values can be of
            different types depending on the metric function used, e.g.,
            `precision_score` returns a single float value, `precision_recall_curve`
            returns a tuple of arrays, and `confusion_matrix` returns a two-dimensional
            array.
        """
        if refine_method is None:
            refine_method = getattr(self, "refine_method", None)
        if refine_kwargs is None:
            refine_kwargs = getattr(self, "refine_kwargs", None)
        return evaluate.compute_eval_metrics(
            metrics=metrics,
            metrics_kwargs=metrics_kwargs,
            clf=getattr(self, "clf", None),
            clf_dict=getattr(self, "clf_dict", None),
            refine_method=refine_method,
            refine_kwargs=refine_kwargs,
            split_df=split_df,
            img_dir=img_dir,
            response_img_dir=response_img_dir,
            img_filepaths=img_filepaths,
            response_img_filepaths=response_img_filepaths,
            img_filename_pattern=img_filename_pattern,
        )

    def eval_refine_params(
        self,
        refine_method=None,
        refine_params_list=None,
        metrics=None,
        metrics_kwargs=None,
        tree_val=None,
        nontree_val=None,
        split_df=None,
        img_dir=None,
        img_filepaths=None,
        img_filename_pattern=None,
        response_img_dir=None,
    ):
        """
        Evaluate a refinement procedure for different parameters.

        Parameters
        ----------
        refine_method : callable, optional
            Refinement method that takes a probability image as the first positional
            argument followed by tree and non-tree values, e.g.,
            `refine_method(p_tree_img, tree_val, nontree_val, **kwargs)`. If no value is
            provided, the value from `self.refine_method` is used if set, otherwise the
            value from `settings.CLF_REFINE_METHOD` is used.
        refine_params_list : list of dict, optional

            Parameters to evaluate for the refinement method, as a list of keyword
            arguments. The metrics will be computed for each item of this list. If no
            value is provided, the value from `settings.EVAL_REFINE_PARAMS` is used.
        metrics : str, func or list of str or func
            The metrics to compute, must be either a string with a function of the
            `sklearn.metrics`, a function that takes a `y_true` and `y_pred`
            positional arguments with the true and predicted labels respectively
            or a list-like of any of the two options. If no value is provided,
            the values set in `settings.EVAL_METRICS` are used.
        metrics_kwargs : dict or list of dict
            Additional keyword arguments to pass to each of the metric functions.
        tree_val, nontree_val : int, optional
            The values that designate tree and non-tree pixels respectively in
            the response images. If no values are provided, the values from this
            instance are used.
        split_df : pandas DataFrame, optional
            Data frame with the validation images.
        img_dir : str representing path to a directory, optional
            Path to the directory where the images from `split_df` are located.
            Required if `split_df` is provided. Ignored if `img_filepaths` is
            provided.
        img_filepaths : list-like, optional
            List of paths to the tiles that will be used for validation. Ignored if
            `split_df` is provided.
        img_filename_pattern : str representing a file-name pattern, optional
            Filename pattern to be matched in order to obtain the list of images. If no
            value is provided, the value set in `settings.IMG_FILENAME_PATTERN` is used.
            Ignored if `split_df` or `img_filepaths` is provided.
        response_img_dir : str representing path to a directory, optional
            Path to the directory where the response tiles are located.

        Returns
        -------
        results : pandas DataFrame
            A DataFrame with the computed values for each metric (row) and each
            refinement keyword argument set (column, stringified).
        """
        if refine_method is None:
            refine_method = getattr(self, "refine_method", None)
        if tree_val is None:
            tree_val = self.tree_val
        if nontree_val is None:
            nontree_val = self.nontree_val
        return evaluate.eval_refine_params(
            refine_method=refine_method,
            refine_params_list=refine_params_list,
            metrics=metrics,
            metrics_kwargs=metrics_kwargs,
            clf=getattr(self, "clf", None),
            clf_dict=getattr(self, "clf_dict", None),
            tree_val=tree_val,
            nontree_val=nontree_val,
            split_df=split_df,
            img_dir=img_dir,
            img_filepaths=img_filepaths,
            img_filename_pattern=img_filename_pattern,
            response_img_dir=response_img_dir,
        )
