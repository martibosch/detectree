"""detectree CLI."""

import logging
from importlib import metadata
from os import path

import click
import pandas as pd
from skops import io

import detectree as dtr
from detectree import settings


# utils for the CLI
class _OptionEatAll(click.Option):
    # Option that can take an unlimided number of arguments Copied from Stephen Rauch's
    # answer in stack overflow.  https://bit.ly/2kstLhe
    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop("save_other_options", True)
        nargs = kwargs.pop("nargs", -1)
        assert nargs == -1, "nargs, if set, must be -1 not {}".format(nargs)
        super().__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super().add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


def _dict_from_kwargs(kwargs):
    # Multiple key:value pair arguments in click, see https://bit.ly/32BaES3
    if kwargs is not None:
        kwargs = eval(kwargs)
    else:
        kwargs = {}

    return kwargs


def _init_classifier_trainer(
    sigmas,
    num_orientations,
    min_neighborhood_range,
    num_neighborhoods,
    tree_val,
    nontree_val,
    classifier_kwargs,
):
    # pixel_features_builder_kws = _dict_from_kws(pixel_features_builder_kws)
    # pixel_response_builder_kws = _dict_from_kws(pixel_response_builder_kws)
    # note that since unlike the predict CLI functions that pass kwargs to multiple
    # methods, the classifier CLI functions only have the classifier kwargs so we could
    # use the "forwarding unknown options" via the click context class instead of
    # passing a string with an encoded dict as we do for the predict functions and here
    # see https://github.com/fastapi/typer/issues/163
    classifier_kwargs = _dict_from_kwargs(classifier_kwargs)

    return dtr.ClassifierTrainer(
        sigmas=sigmas,
        num_orientations=num_orientations,
        min_neighborhood_range=min_neighborhood_range,
        num_neighborhoods=num_neighborhoods,
        tree_val=tree_val,
        nontree_val=nontree_val,
        **classifier_kwargs,
    )


def _dump_clf(clf, output_filepath, logger):
    # joblib.dump(clf, output_filepath)
    io.dump(clf, output_filepath)
    logger.info("Dumped trained classifier to %s", output_filepath)


# CLI
@click.group()
@click.version_option(version=metadata.version("detectree"), message="%(version)s")
@click.pass_context
def cli(ctx):
    """Detectree CLI."""
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    ctx.ensure_object(dict)
    ctx.obj["LOGGER"] = logger


@cli.command()
@click.pass_context
@click.option("--img-filepaths", cls=_OptionEatAll)
@click.option("--img-dir", type=click.Path(exists=True))
@click.option("--img-filename-pattern")
@click.option("--gabor-frequencies", cls=_OptionEatAll)
@click.option("--gabor-num-orientations", cls=_OptionEatAll)
@click.option("--response-bins-per-axis", type=int)
@click.option("--num-color-bins", type=int)
@click.option("--method")
@click.option("--num-components", type=int)
@click.option("--num-img-clusters", type=int)
@click.option("--train-prop", type=float)
@click.option("--output-filepath", type=click.Path())
def train_test_split(
    ctx,
    img_filepaths,
    img_dir,
    img_filename_pattern,
    gabor_frequencies,
    gabor_num_orientations,
    response_bins_per_axis,
    num_color_bins,
    method,
    num_components,
    num_img_clusters,
    train_prop,
    output_filepath,
):
    """Split the set of images into training and testing sets."""
    logger = ctx.obj["LOGGER"]

    ts = dtr.TrainingSelector(
        img_filepaths=img_filepaths,
        img_dir=img_dir,
        img_filename_pattern=img_filename_pattern,
        gabor_frequencies=gabor_frequencies,
        gabor_num_orientations=gabor_num_orientations,
        response_bins_per_axis=response_bins_per_axis,
        num_color_bins=num_color_bins,
    )
    num_imgs = len(ts.img_filepaths)
    if img_dir is not None:
        logger.info("Loaded %d images from %s", num_imgs, img_dir)
    else:
        logger.info("Loaded %d images", num_imgs)

    tts_kwargs = {}
    if method is not None:
        tts_kwargs["method"] = method
    if num_components is not None:
        tts_kwargs["num_components"] = num_components
    if num_img_clusters is not None:
        tts_kwargs["num_img_clusters"] = num_img_clusters
    if train_prop is not None:
        tts_kwargs["train_prop"] = train_prop
    df, evr = ts.train_test_split(return_evr=True, **tts_kwargs)
    logger.info("Variance ratio explained by PCA: %f", evr)

    if output_filepath is None:
        output_filepath = "split.csv"

    df.to_csv(output_filepath)
    logger.info("Dumped train/test split to %s", output_filepath)


@cli.command()
@click.pass_context
@click.option("--split-filepath", type=click.Path(exists=True))
@click.option("--img-dir", type=click.Path(exists=True))
@click.option("--response-img-dir", type=click.Path(exists=True))
@click.option("--img-filepaths", cls=_OptionEatAll)
@click.option("--response-img-filepaths", cls=_OptionEatAll)
@click.option("--img-filename-pattern")
@click.option("--method")
@click.option("--img-cluster", type=int)
@click.option("--sigmas", cls=_OptionEatAll)
@click.option("--num-orientations", type=int)
@click.option("--min-neighborhood-range", type=int)
@click.option("--num-neighborhoods", type=int)
@click.option("--tree-val", type=int)
@click.option("--nontree-val", type=int)
@click.option("--classifier-kwargs", type=str)
@click.option("--output-filepath", type=click.Path())
def train_classifier(
    ctx,
    split_filepath,
    img_dir,
    response_img_dir,
    img_filepaths,
    response_img_filepaths,
    img_filename_pattern,
    method,
    img_cluster,
    sigmas,
    num_orientations,
    min_neighborhood_range,
    num_neighborhoods,
    tree_val,
    nontree_val,
    classifier_kwargs,
    output_filepath,
):
    """Train a tree/non-tree pixel classifier."""
    logger = ctx.obj["LOGGER"]

    logger.info("Training classifier")
    if split_filepath is not None:
        split_df = pd.read_csv(split_filepath)
    else:
        split_df = None

    ct = _init_classifier_trainer(
        sigmas=sigmas,
        num_orientations=num_orientations,
        min_neighborhood_range=min_neighborhood_range,
        num_neighborhoods=num_neighborhoods,
        tree_val=tree_val,
        nontree_val=nontree_val,
        classifier_kwargs=classifier_kwargs,
    )
    clf = ct.train_classifier(
        split_df=split_df,
        img_dir=img_dir,
        response_img_dir=response_img_dir,
        img_filepaths=img_filepaths,
        response_img_filepaths=response_img_filepaths,
        img_filename_pattern=img_filename_pattern,
        method=method,
        img_cluster=img_cluster,
    )

    if output_filepath is None:
        output_filepath = "clf.skops"

    _dump_clf(clf, output_filepath, logger)


@cli.command()
@click.pass_context
@click.argument("split_filepath", type=click.Path(exists=True))
@click.argument("img_dir", type=click.Path(exists=True))
@click.argument("response_img_dir", type=click.Path(exists=True))
@click.option("--sigmas", cls=_OptionEatAll)
@click.option("--num-orientations", type=int)
@click.option("--min-neighborhood-range", type=int)
@click.option("--num-neighborhoods", type=int)
@click.option("--tree-val", type=int)
@click.option("--nontree-val", type=int)
@click.option("--classifier-kwargs", type=str)
@click.option("--output-dir", type=click.Path(exists=True))
def train_classifiers(
    ctx,
    split_filepath,
    img_dir,
    response_img_dir,
    sigmas,
    num_orientations,
    min_neighborhood_range,
    num_neighborhoods,
    tree_val,
    nontree_val,
    classifier_kwargs,
    output_dir,
):
    """Train tree/non-tree pixel classifier(s) for a given train/test split."""
    logger = ctx.obj["LOGGER"]

    logger.info("Training classifiers")
    if split_filepath is not None:
        split_df = pd.read_csv(split_filepath)
    else:
        split_df = None

    ct = _init_classifier_trainer(
        sigmas=sigmas,
        num_orientations=num_orientations,
        min_neighborhood_range=min_neighborhood_range,
        num_neighborhoods=num_neighborhoods,
        tree_val=tree_val,
        nontree_val=nontree_val,
        classifier_kwargs=classifier_kwargs,
    )
    clfs_dict = ct.train_classifiers(split_df, img_dir, response_img_dir)

    if output_dir is None:
        output_dir = ""

    for img_cluster in clfs_dict:
        _dump_clf(
            clfs_dict[img_cluster],
            path.join(output_dir, f"{img_cluster}.skops"),
            logger,
        )


@cli.command()
@click.pass_context
@click.argument("img_filepath", type=click.Path(exists=True))
@click.option("--clf-filepath", type=click.Path(exists=True))
@click.option("--hf-hub-repo-id", type=str)
@click.option("--hf-hub-clf-filename", type=str)
@click.option("--hf-hub-download-kwargs", type=str)
# @click.option("--skops-trusted", cls=_OptionEatAll)
@click.option("--tree-val", type=int)
@click.option("--nontree-val", type=int)
@click.option("--refine/--no-refine", default=True)
@click.option("--refine-kwargs", type=str)
@click.option("--pixel-features-builder-kwargs", type=str)
@click.option("--output-filepath", type=click.Path())
def predict_img(
    ctx,
    img_filepath,
    clf_filepath,
    hf_hub_repo_id,
    hf_hub_clf_filename,
    hf_hub_download_kwargs,
    # skops_trusted,
    tree_val,
    nontree_val,
    refine,
    refine_kwargs,
    pixel_features_builder_kwargs,
    output_filepath,
):
    """Predict tree pixels in an image."""
    logger = ctx.obj["LOGGER"]

    if clf_filepath:
        # if not skops_trusted:
        #     skops_trusted = settings.SKOPS_TRUSTED
        clf = io.load(clf_filepath, trusted=settings.SKOPS_TRUSTED)
        _clf_msg = clf_filepath

    else:
        clf = None
        # we are repeating the logic from `Classifier.__init__` here just so we can log
        # which classifier is being used - TODO: DRY
        if hf_hub_repo_id is None:
            hf_hub_repo_id = settings.HF_HUB_REPO_ID
        _clf_msg = hf_hub_repo_id
    logger.info("Classifying %s with classifier of %s", img_filepath, _clf_msg)

    hf_hub_download_kwargs = _dict_from_kwargs(hf_hub_download_kwargs)
    pixel_features_builder_kwargs = _dict_from_kwargs(pixel_features_builder_kwargs)
    refine_kwargs = _dict_from_kwargs(refine_kwargs)
    if not refine_kwargs:
        refine_kwargs = None
    if refine:
        refine_method = None
    else:
        refine_method = False
        refine_kwargs = None
    c = dtr.Classifier(
        clf=clf,
        hf_hub_repo_id=hf_hub_repo_id,
        hf_hub_clf_filename=hf_hub_clf_filename,
        hf_hub_download_kwargs=hf_hub_download_kwargs,
        # skops_trusted=skops_trusted,
        tree_val=tree_val,
        nontree_val=nontree_val,
        refine_method=refine_method,
        refine_kwargs=refine_kwargs,
        **pixel_features_builder_kwargs,
    )

    if output_filepath is None:
        filename, ext = path.splitext(path.basename(img_filepath))
        output_filepath = f"{filename}-pred{ext}"

    c.predict_img(
        img_filepath,
        output_filepath=output_filepath,
    )
    logger.info("Dumped predicted image to %s", output_filepath)


@cli.command()
@click.pass_context
@click.argument("output_dir", type=click.Path(exists=True))
@click.option("--clf-filepath", type=click.Path(exists=True))
@click.option("--clf-dir", type=click.Path(exists=True))
@click.option("--hf-hub-repo-id", type=str)
@click.option("--hf-hub-clf-filename", type=str)
@click.option("--hf-hub-download-kwargs", type=str)
# @click.option("--skops-trusted", cls=_OptionEatAll)
@click.option("--split-filepath", type=click.Path(exists=True))
@click.option("--tree-val", type=int)
@click.option("--nontree-val", type=int)
@click.option("--refine/--no-refine", default=True)
@click.option("--refine-kwargs", type=str)
@click.option("--img-dir", type=click.Path(exists=True))
# @click.option("--img-filepaths", type=_OptionEatAll)
@click.option("--img-filename-pattern", type=str)
@click.option("--pixel-features-builder-kwargs", type=str)
def predict_imgs(
    ctx,
    output_dir,
    clf_filepath,
    clf_dir,
    hf_hub_repo_id,
    hf_hub_clf_filename,
    hf_hub_download_kwargs,
    # skops_trusted,
    tree_val,
    nontree_val,
    refine,
    refine_kwargs,
    split_filepath,
    img_dir,
    # img_filepaths,
    img_filename_pattern,
    pixel_features_builder_kwargs,
):
    """Predict tree pixels in multiple images."""
    logger = ctx.obj["LOGGER"]

    # init them as None and change them if needed
    clf = None
    clf_dict = None
    split_df = None
    if split_filepath is not None:
        split_df = pd.read_csv(split_filepath)
    if clf_filepath is not None:
        # if not skops_trusted:
        #     skops_trusted = settings.SKOPS_TRUSTED
        clf = io.load(clf_filepath, settings.SKOPS_TRUSTED)
        _clf_msg = clf_filepath
    elif clf_dir is not None:
        clf_dict = {}
        if split_df is None:
            raise ValueError(
                "If `clf_dir` is provided, `split_filepath` must also be provided."
            )
        for img_cluster in split_df["img_cluster"].unique():
            clf_dict[img_cluster] = io.load(
                path.join(clf_dir, f"{img_cluster}.skops"),
                settings.SKOPS_TRUSTED,
            )
        _clf_msg = clf_dir
    else:
        # at this point, this means that neither `clf_filepath` nor `clf_dir` have been
        # provided, so we use the pre-trained classifier
        # we are repeating the logic from `Classifier.__init__` here just so we can log
        # which classifier is being used - TODO: DRY
        if hf_hub_repo_id is None:
            hf_hub_repo_id = settings.HF_HUB_REPO_ID
        _clf_msg = hf_hub_repo_id
    logger.info(
        "Predicting with classifier(s) from %s",
        _clf_msg,
    )

    hf_hub_download_kwargs = _dict_from_kwargs(hf_hub_download_kwargs)
    pixel_features_builder_kwargs = _dict_from_kwargs(pixel_features_builder_kwargs)
    refine_kwargs = _dict_from_kwargs(refine_kwargs)
    if not refine_kwargs:
        refine_kwargs = None
    if refine:
        refine_method = None
    else:
        refine_method = False
        refine_kwargs = None
    c = dtr.Classifier(
        clf=clf,
        clf_dict=clf_dict,
        hf_hub_repo_id=hf_hub_repo_id,
        hf_hub_clf_filename=hf_hub_clf_filename,
        hf_hub_download_kwargs=hf_hub_download_kwargs,
        # skops_trusted=skops_trusted,
        tree_val=tree_val,
        nontree_val=nontree_val,
        refine_method=refine_method,
        refine_kwargs=refine_kwargs,
        **pixel_features_builder_kwargs,
    )

    pred_imgs = c.predict_imgs(
        output_dir,
        split_df=split_df,
        img_dir=img_dir,
        # img_filepaths=img_filepaths,
        img_filename_pattern=img_filename_pattern,
    )
    logger.info("Dumped %d predicted images to %s", len(pred_imgs), output_dir)
