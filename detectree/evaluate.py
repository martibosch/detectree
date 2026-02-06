"""Evaluation."""

import tempfile
from os import path

import numpy as np
import pandas as pd
import rasterio as rio
from sklearn import metrics as sklearn_metrics

from detectree import classifier, settings


def _get_true_pred_arr(
    pred_img_filepaths, *, response_img_filepaths=None, response_img_dir=None
):
    if response_img_filepaths is None:
        if response_img_dir is None:
            raise ValueError(
                "Either `response_img_filepaths` or `response_img_dir` must be "
                "provided."
            )
        response_img_filepaths = [
            path.join(response_img_dir, path.basename(pred_img_filepath))
            for pred_img_filepath in pred_img_filepaths
        ]

    true_arrs = []
    pred_arrs = []
    for pred_img_filepath, response_img_filepath in zip(
        pred_img_filepaths, response_img_filepaths
    ):
        with rio.open(response_img_filepath) as src:
            true_arr = src.read(1).flatten()
        with rio.open(pred_img_filepath) as src:
            pred_arr = src.read(1).flatten()
        # results.append((true_arr, pred_arr))
        true_arrs.append(true_arr)
        pred_arrs.append(pred_arr)

    return np.vstack([np.concatenate(arrs, axis=0) for arrs in [true_arrs, pred_arrs]])


def get_true_pred_arr(
    *,
    pred_img_filepaths=None,
    clf=None,
    clf_dict=None,
    hf_hub_repo_id=None,
    hf_hub_clf_filename=None,
    hf_hub_download_kwargs=None,
    skops_trusted=None,
    refine_method=None,
    refine_kwargs=None,
    split_df=None,
    img_dir=None,
    response_img_dir=None,
    img_filepaths=None,
    response_img_filepaths=None,
    img_filename_pattern=None,
    **classifier_kwargs,
):
    """
    Get true and predicted values for the validation images.

    Parameters
    ----------
    pred_img_filepaths : list-like, optional
        List of paths to precomputed predicted images. If provided, classification is
        skipped and predictions are read directly from these files - therefore all
        arguments except `response_img_dir` or `response_img_filepaths` are ignored.
    clf : scikit-learn-like classifier, optional
        Trained classifier. If no value is provided, the classifier is loaded from
        HuggingFace Hub using the values provided in `hf_hub_repo_id` and
        `hf_hub_clf_filename`.
    clf_dict : dictionary, optional
        Dictionary mapping a trained scikit-learn-like classifier to each first-level
        cluster label.
    hf_hub_repo_id, hf_hub_clf_filename : str, optional
        HuggingFace Hub repository id (string with the user or organization and
        repository name separated by a `/`) and file name of the skops classifier
        respectively. If no value is provided, the values set in
        `settings.HF_HUB_REPO_ID` and `settings.HF_HUB_CLF_FILENAME` Ignored if `clf` or
        `clf_dict` are provided.
    hf_hub_download_kwargs : dict, optional
        Additional keyword arguments (besides "repo_id", "filename", "library_name"  and
        "library_version") to pass to `huggingface_hub.hf_hub_download`.
    skops_trusted : list, optional
        List of trusted object types to load the classifier from HuggingFace Hub, passed
        to `skops.io.load`. If no value is provided, the value from
        `settings.SKOPS_TRUSTED` is used. Ignored if `clf` or `clf_dict` are provided.
    refine_method : callable or bool, optional
        Method to refine the pixel-level classification. If `False` is provided, no
        refinement is performed. If `None` is provided, the default behavior of
        `detectree.classifier.Classifier` is used.
    refine_kwargs : dict, optional
        Keyword arguments that will be passed to `refine_method`. Ignored if no
        refinement is performed.
    split_df : pandas DataFrame, optional
        Data frame with the validation images.
    img_dir : str representing path to a directory, optional
        Path to the directory where the images from `split_df` are located. Required if
        `split_df` is provided. Ignored if `img_filepaths` is provided.
    response_img_dir : str representing path to a directory, optional
        Path to the directory where the response tiles are located. Required if
        providing `split_df`. Otherwise `response_img_dir` might either be ignored if
        providing `response_img_filepaths`, or be used as the directory where the images
        whose filename matches `img_filename_pattern` are to be located.
    img_filepaths : list-like, optional
        List of paths to the tiles that will be used for validation. Ignored if
        `split_df` is provided.
    response_img_filepaths : list-like, optional
        List of paths to the binary response tiles that will be used for evaluation.
        Ignored if `split_df` is provided.
    img_filename_pattern : str representing a file-name pattern, optional
        Filename pattern to be matched in order to obtain the list of images. If no
        value is provided, the value set in `settings.IMG_FILENAME_PATTERN` is used.
        Ignored if `split_df` or `img_filepaths` is provided.
    classifier_kwargs : dict, optional
        Additional keyword arguments to pass to the initialization of
        `detectree.classifier.Classifier` class.

    Returns
    -------
    true_pred : numpy.ndarray
        Array with two rows respectively containing the true and predicted values for
        the provided images.
    """
    if pred_img_filepaths is not None:
        # no need for inference

        # return here because the other return will need the tmp_dir context manager
        # anyway so they cannot be merged
        return _get_true_pred_arr(
            pred_img_filepaths,
            response_img_filepaths=response_img_filepaths,
            response_img_dir=response_img_dir,
        )

    # inference is needed
    _classifier_kwargs = classifier_kwargs.copy()
    if refine_method is not None:
        _classifier_kwargs["refine_method"] = refine_method
    if refine_kwargs is not None:
        _classifier_kwargs["refine_kwargs"] = refine_kwargs

    c = classifier.Classifier(
        clf=clf,
        clf_dict=clf_dict,
        hf_hub_repo_id=hf_hub_repo_id,
        hf_hub_clf_filename=hf_hub_clf_filename,
        hf_hub_download_kwargs=hf_hub_download_kwargs,
        skops_trusted=skops_trusted,
        **_classifier_kwargs,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        pred_img_filepaths = c.predict_imgs(
            tmp_dir,
            split_df=split_df,
            img_dir=img_dir,
            img_filepaths=img_filepaths,
            img_filename_pattern=img_filename_pattern,
        )
        # TODO: maybe DRY with the same return on top of the function, but we need the
        # tmp_dir context manager here
        return _get_true_pred_arr(
            pred_img_filepaths,
            response_img_filepaths=response_img_filepaths,
            response_img_dir=response_img_dir,
        )


def compute_eval_metrics(
    *,
    pred_img_filepaths=None,
    metrics=None,
    metrics_kwargs=None,
    clf=None,
    clf_dict=None,
    hf_hub_repo_id=None,
    hf_hub_clf_filename=None,
    hf_hub_download_kwargs=None,
    skops_trusted=None,
    refine_method=None,
    refine_kwargs=None,
    split_df=None,
    img_dir=None,
    response_img_dir=None,
    img_filepaths=None,
    response_img_filepaths=None,
    img_filename_pattern=None,
    **classifier_kwargs,
):
    """
    Compute evaluation metrics for the validation images.

    Parameters
    ----------
    pred_img_filepaths : list-like, optional
        List of paths to precomputed predicted images. If provided, classification is
        skipped and metrics are computed directly from these files. Requires
        `response_img_dir` or `response_img_filepaths`.
    metrics : str, func or list of str or func
        The metrics to compute, must be either a string with a function of the
        `sklearn.metrics`, a function that takes a `y_true` and `y_pred` positional
        arguments with the true and predicted labels respectively or a list-like of any
        of the two options. If no value is provided, the values set in
        `settings.EVAL_METRICS` are used.
    metrics_kwargs : dict or list of dict
        Additional keyword arguments to pass to each of the metric functions.
    clf : scikit-learn-like classifier, optional
        Trained classifier. If no value is provided, the classifier is loaded from
        HuggingFace Hub using the values provided in `hf_hub_repo_id` and
        `hf_hub_clf_filename`.
    clf_dict : dictionary, optional
        Dictionary mapping a trained scikit-learn-like classifier to each first-level
        cluster label.
    hf_hub_repo_id, hf_hub_clf_filename : str, optional
        HuggingFace Hub repository id (string with the user or organization and
        repository name separated by a `/`) and file name of the skops classifier
        respectively. If no value is provided, the values set in
        `settings.HF_HUB_REPO_ID` and `settings.HF_HUB_CLF_FILENAME` Ignored if `clf` or
        `clf_dict` are provided.
    hf_hub_download_kwargs : dict, optional
        Additional keyword arguments (besides "repo_id", "filename", "library_name"  and
        "library_version") to pass to `huggingface_hub.hf_hub_download`.
    skops_trusted : list, optional
        List of trusted object types to load the classifier from HuggingFace Hub, passed
        to `skops.io.load`. If no value is provided, the value from
        `settings.SKOPS_TRUSTED` is used. Ignored if `clf` or `clf_dict` are provided.
    refine_method : callable or bool, optional
        Method to refine the pixel-level classification. If `False` is provided, no
        refinement is performed. If `None` is provided, the default behavior of
        `detectree.classifier.Classifier` is used.
    refine_kwargs : dict, optional
        Keyword arguments that will be passed to `refine_method`. Ignored if no
        refinement is performed.
    split_df : pandas DataFrame, optional
        Data frame with the validation images.
    img_dir : str representing path to a directory, optional
        Path to the directory where the images from `split_df` are located. Required if
        `split_df` is provided. Ignored if `img_filepaths` is provided.
    response_img_dir : str representing path to a directory, optional
        Path to the directory where the response tiles are located. Ignored if providing
        `response_img_filepaths`.
    img_filepaths : list-like, optional
        List of paths to the tiles that will be used for validation. Ignored if
        `split_df` is provided.
    response_img_filepaths : list-like, optional
        List of paths to the binary response tiles that will be used for evaluation.
        Ignored if `split_df` is provided.
    img_filename_pattern : str representing a file-name pattern, optional
        Filename pattern to be matched in order to obtain the list of images. If no
        value is provided, the value set in `settings.IMG_FILENAME_PATTERN` is used.
        Ignored if `split_df` or `img_filepaths` is provided.
    classifier_kwargs : dict, optional
        Additional keyword arguments to pass to the initialization of
        `detectree.classifier.Classifier` class.

    Returns
    -------
    metric_values : numeric, tuple, ndarray or list
        Values of the metrics computed for the validation images. If only one metric is
        provided, a single value is returned. If multiple metrics are provided, a list
        of values is returned, one for each metric. The returned values can be of
        different types, depending on the metric function used, e.g., `precision_score`
        returns a single float value, `precision_recall_curve` returns a tuple of
        arrays, and `confusion_matrix` returns a two-dimensional array.
    """
    if metrics is None:
        metrics = settings.EVAL_METRICS
    if metrics_kwargs is None:
        metrics_kwargs = [{}] * len(metrics)

    true_pred_arr = get_true_pred_arr(
        pred_img_filepaths=pred_img_filepaths,
        clf=clf,
        clf_dict=clf_dict,
        hf_hub_repo_id=hf_hub_repo_id,
        hf_hub_clf_filename=hf_hub_clf_filename,
        hf_hub_download_kwargs=hf_hub_download_kwargs,
        skops_trusted=skops_trusted,
        refine_method=refine_method,
        refine_kwargs=refine_kwargs,
        split_df=split_df,
        img_dir=img_dir,
        img_filepaths=img_filepaths,
        response_img_filepaths=response_img_filepaths,
        img_filename_pattern=img_filename_pattern,
        response_img_dir=response_img_dir,
        **classifier_kwargs,
    )

    metric_values = []
    for metric, kwargs in zip(metrics, metrics_kwargs):
        if isinstance(metric, str):
            metric_func = getattr(sklearn_metrics, metric)
        elif callable(metric):
            metric_func = metric
        else:
            raise TypeError(
                "Metrics must be either a string with a function of the "
                "`sklearn.metrics` module, a function that takes `y_true` and `y_pred` "
                "positional arguments or a list-like of any of the two options."
            )

        metric_values.append(metric_func(true_pred_arr[0], true_pred_arr[1], **kwargs))

    if len(metric_values) == 1:
        return metric_values[0]
    return metric_values


def eval_refine_params(
    *,
    refine_method=None,
    refine_params_list=None,
    metrics=None,
    metrics_kwargs=None,
    clf=None,
    clf_dict=None,
    hf_hub_repo_id=None,
    hf_hub_clf_filename=None,
    hf_hub_download_kwargs=None,
    skops_trusted=None,
    tree_val=None,
    nontree_val=None,
    split_df=None,
    img_dir=None,
    img_filepaths=None,
    img_filename_pattern=None,
    response_img_dir=None,
    **classifier_kwargs,
):
    """
    Evaluate refinement procedures for different parameters.

    Parameters
    ----------
    refine_method : callable, optional
        Refinement method that takes a probability image as the first positional
        argument followed by tree and non-tree values, e.g.,
        `refine_method(p_tree_img, tree_val, nontree_val, **kwargs)`. If no value is
        provided, the value from `settings.CLF_REFINE_METHOD` is used.
    refine_params_list : list of dict, optional

        Parameters to evaluate for the refinement method, as a list of keyword
        arguments. The metrics will be computed for each item of this list. If no value
        is provided, the value from `settings.EVAL_REFINE_PARAMS` is used.
    metrics : str, func or list of str or func
        The metrics to compute, must be either a string with a function of the
        `sklearn.metrics`, a function that takes a `y_true` and `y_pred` positional
        arguments with the true and predicted labels respectively or a list-like of any
        of the two options. If no value is provided, the values set in
        `settings.EVAL_METRICS` are used.
    metrics_kwargs : dict or list of dict
        Additional keyword arguments to pass to each of the metric functions.
    clf : scikit-learn-like classifier, optional
        Trained classifier. If no value is provided, the classifier is loaded from
        HuggingFace Hub using the values provided in `hf_hub_repo_id` and
        `hf_hub_clf_filename`.
    clf_dict : dictionary, optional
        Dictionary mapping a trained scikit-learn-like classifier to each first-level
        cluster label.
    hf_hub_repo_id, hf_hub_clf_filename : str, optional
        HuggingFace Hub repository id (string with the user or organization and
        repository name separated by a `/`) and file name of the skops classifier
        respectively. If no value is provided, the values set in
        `settings.HF_HUB_REPO_ID` and `settings.HF_HUB_CLF_FILENAME` Ignored if `clf` or
        `clf_dict` are provided.
    hf_hub_download_kwargs : dict, optional
        Additional keyword arguments (besides "repo_id", "filename", "library_name"  and
        "library_version") to pass to `huggingface_hub.hf_hub_download`.
    skops_trusted : list, optional
        List of trusted object types to load the classifier from HuggingFace Hub, passed
        to `skops.io.load`. If no value is provided, the value from
        `settings.SKOPS_TRUSTED` is used. Ignored if `clf` or `clf_dict` are provided.
    tree_val, nontree_val : int, optional
        The values that designate tree and non-tree pixels respectively in the response
        images. If no values are provided, the values set in `settings.TREE_VAL` and
        `settings.NON_TREE_VAL` are respectively used.
    split_df : pandas DataFrame, optional
        Data frame with the validation images.
    img_dir : str representing path to a directory, optional
        Path to the directory where the images from `split_df` are located. Required if
        `split_df` is provided. Ignored if `img_filepaths` is provided.
    img_filepaths : list-like, optional
        List of paths to the tiles that will be used for validation. Ignored if
        `split_df` is provided.
    img_filename_pattern : str representing a file-name pattern, optional
        Filename pattern to be matched in order to obtain the list of images. If no
        value is provided, the value set in `settings.IMG_FILENAME_PATTERN` is used.
        Ignored if `split_df` or `img_filepaths` is provided.
    response_img_dir : str representing path to a directory, optional
        Path to the directory where the response tiles are located. Ignored if providing
        `response_img_filepaths`.
    classifier_kwargs : dict, optional
        Additional keyword arguments to pass to the initialization of
        `detectree.classifier.Classifier` class.

    Returns
    -------
    results : pandas DataFrame
        A DataFrame with the computed values for each metric (row) and each refinement
        keyword argument set (column, stringified).
    """
    if refine_method is None:
        refine_method = settings.CLF_REFINE_METHOD
    if refine_params_list is None:
        refine_params_list = settings.EVAL_REFINE_PARAMS
    # refine_params_list = list(refine_params_list)

    if metrics is None:
        metrics = settings.EVAL_METRICS
    if metrics_kwargs is None:
        metrics_kwargs = [{}] * len(metrics)

    if tree_val is None:
        tree_val = settings.TREE_VAL
    if nontree_val is None:
        nontree_val = settings.NONTREE_VAL

    _classifier_kwargs = classifier_kwargs.copy()
    for key in ("refine_method", "refine_kwargs", "return_proba"):
        _classifier_kwargs.pop(key, None)

    c = classifier.Classifier(
        clf=clf,
        clf_dict=clf_dict,
        hf_hub_repo_id=hf_hub_repo_id,
        hf_hub_clf_filename=hf_hub_clf_filename,
        hf_hub_download_kwargs=hf_hub_download_kwargs,
        skops_trusted=skops_trusted,
        tree_val=tree_val,
        nontree_val=nontree_val,
        return_proba=True,
        **_classifier_kwargs,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        pred_img_filepaths = c.predict_imgs(
            tmp_dir,
            split_df=split_df,
            img_dir=img_dir,
            img_filepaths=img_filepaths,
            img_filename_pattern=img_filename_pattern,
        )
        true_arrs = []
        pred_refined_by_kwargs = [[] for _ in refine_params_list]
        for pred_img_filepath in pred_img_filepaths:
            with rio.open(
                path.join(response_img_dir, path.basename(pred_img_filepath))
            ) as src:
                true_arrs.append(src.read(1).flatten())
            with rio.open(pred_img_filepath) as src:
                pred_arr = src.read(1)
            for idx, refine_kwargs in enumerate(refine_params_list):
                pred_refined_by_kwargs[idx].append(
                    refine_method(
                        pred_arr, tree_val, nontree_val, **refine_kwargs
                    ).flatten()
                )

    true_arr = np.concatenate(true_arrs)
    pred_refined_arrs = [
        np.concatenate(arrs, axis=0) for arrs in pred_refined_by_kwargs
    ]

    metric_values = []
    for metric, kwargs in zip(metrics, metrics_kwargs):
        metric_results = []
        if isinstance(metric, str):
            metric_func = getattr(sklearn_metrics, metric)
        elif callable(metric):
            metric_func = metric
        else:
            raise TypeError(
                "Metrics must be either a string with a function of the "
                "`sklearn.metrics` module, a function that takes `y_true` and `y_pred` "
                "positional arguments or a list-like of any of the two options."
            )

        for refined_arr in pred_refined_arrs:
            metric_results.append(metric_func(true_arr, refined_arr, **kwargs))
        metric_values.append(metric_results)

    return pd.DataFrame(
        metric_values,
        index=[metric.__name__ if callable(metric) else metric for metric in metrics],
        columns=[str(kwargs) for kwargs in refine_params_list],
    )
