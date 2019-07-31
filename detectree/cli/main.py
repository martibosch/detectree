import logging
from os import path

import click
import joblib
import pandas as pd

import detectree as dtr


# utils for the CLI
class OptionEatAll(click.Option):
    # Option that can take an unlimided number of arguments
    # Copied from Stephen Rauch's answer in stack overflow.
    # https://bit.ly/2kstLhe
    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop('save_other_options', True)
        nargs = kwargs.pop('nargs', -1)
        assert nargs == -1, 'nargs, if set, must be -1 not {}'.format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
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

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(
                name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


def _dict_from_kws(kws):
    # Multiple key:value pair arguments in click, see https://bit.ly/32BaES3
    if kws is not None:
        kws = dict(kw.split(':') for kw in kws)
    else:
        kws = {}

    return kws


def _init_classifier_trainer(pixel_features_builder_kws,
                             pixel_response_builder_kws, adaboost_kws,
                             num_estimators):
    pixel_features_builder_kws = _dict_from_kws(pixel_features_builder_kws)
    pixel_response_builder_kws = _dict_from_kws(pixel_response_builder_kws)
    adaboost_kws = _dict_from_kws(adaboost_kws)

    return dtr.ClassifierTrainer(
        num_estimators=num_estimators,
        pixel_features_builder_kws=pixel_features_builder_kws,
        pixel_response_builder_kws=pixel_response_builder_kws,
        adaboost_kws=adaboost_kws)


def _dump_clf(clf, output_filepath, logger):
    joblib.dump(clf, output_filepath)
    logger.info("Dumped trained classifier to %s", output_filepath)


# CLI
@click.group()
@click.version_option(version=dtr.__version__, message="%(version)s")
@click.pass_context
def cli(ctx):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    ctx.ensure_object(dict)
    ctx.obj['LOGGER'] = logger


@cli.command()
@click.pass_context
@click.option('--img-filepaths', cls=OptionEatAll)
@click.option('--img-dir', type=click.Path(exists=True))
@click.option('--img-filename-pattern')
@click.option('--gabor-frequencies', cls=OptionEatAll)
@click.option('--gabor-num-orientations', cls=OptionEatAll)
@click.option('--response-bins-per-axis', type=int)
@click.option('--num-color-bins', type=int)
@click.option('--method')
@click.option('--num-components', type=int)
@click.option('--num-img-clusters', type=int)
@click.option('--train-prop', type=float)
@click.option('--output-filepath', type=click.Path())
def train_test_split(ctx, img_filepaths, img_dir, img_filename_pattern,
                     gabor_frequencies, gabor_num_orientations,
                     response_bins_per_axis, num_color_bins, method,
                     num_components, num_img_clusters, train_prop,
                     output_filepath):
    logger = ctx.obj['LOGGER']

    ts = dtr.TrainingSelector(img_filepaths=img_filepaths, img_dir=img_dir,
                              img_filename_pattern=img_filename_pattern,
                              gabor_frequencies=gabor_frequencies,
                              gabor_num_orientations=gabor_num_orientations,
                              response_bins_per_axis=response_bins_per_axis,
                              num_color_bins=num_color_bins)
    num_imgs = len(ts.img_filepaths)
    if img_dir is not None:
        logger.info("Loaded %d images from %s", num_imgs, img_dir)
    else:
        logger.info("Loaded %d images", num_imgs)

    tts_kws = {}
    if method is not None:
        tts_kws['method'] = method
    if num_components is not None:
        tts_kws['num_components'] = num_components
    if num_img_clusters is not None:
        tts_kws['num_img_clusters'] = num_img_clusters
    if train_prop is not None:
        tts_kws['train_prop'] = train_prop
    df, evr = ts.train_test_split(return_evr=True, **tts_kws)
    logger.info("Variance ratio explained by PCA: %f", evr)

    if output_filepath is None:
        output_filepath = 'split.csv'

    df.to_csv(output_filepath)
    logger.info("Dumped train/test split to %s", output_filepath)


@cli.command()
@click.pass_context
@click.option('--split-filepath', type=click.Path(exists=True))
@click.option('--response-img-dir', type=click.Path(exists=True))
@click.option('--img-filepaths', cls=OptionEatAll)
@click.option('--response-img-filepaths', cls=OptionEatAll)
@click.option('--img-dir', type=click.Path(exists=True))
@click.option('--img-filename-pattern')
@click.option('--method')
@click.option('--img-cluster', type=int)
@click.option('--num-estimators', type=int)
@click.option('--pixel-features-builder-kws', cls=OptionEatAll)
@click.option('--pixel-response-builder-kws', cls=OptionEatAll)
@click.option('--adaboost-kws', cls=OptionEatAll)
@click.option('--output-filepath', type=click.Path())
def train_classifier(ctx, split_filepath, response_img_dir, img_filepaths,
                     response_img_filepaths, img_dir, img_filename_pattern,
                     method, img_cluster, num_estimators,
                     pixel_features_builder_kws, pixel_response_builder_kws,
                     adaboost_kws, output_filepath):
    logger = ctx.obj['LOGGER']

    logger.info("Training classifier")
    if split_filepath is not None:
        split_df = pd.read_csv(split_filepath)
    else:
        split_df = None

    ct = _init_classifier_trainer(pixel_features_builder_kws,
                                  pixel_response_builder_kws, adaboost_kws,
                                  num_estimators)

    clf = ct.train_classifier(split_df=split_df,
                              response_img_dir=response_img_dir,
                              img_filepaths=img_filepaths,
                              response_img_filepaths=response_img_filepaths,
                              img_dir=img_dir,
                              img_filename_pattern=img_filename_pattern,
                              method=method, img_cluster=img_cluster)

    if output_filepath is None:
        output_filepath = 'clf.joblib'

    _dump_clf(clf, output_filepath, logger)


@cli.command()
@click.pass_context
@click.argument('split_filepath', type=click.Path(exists=True))
@click.argument('response_img_dir', type=click.Path(exists=True))
@click.option('--num-estimators', type=int)
@click.option('--pixel-features-builder-kws', cls=OptionEatAll)
@click.option('--pixel-response-builder-kws', cls=OptionEatAll)
@click.option('--adaboost-kws', cls=OptionEatAll)
@click.option('--output-dir', type=click.Path(exists=True))
def train_classifiers(ctx, split_filepath, response_img_dir, num_estimators,
                      pixel_features_builder_kws, pixel_response_builder_kws,
                      adaboost_kws, output_dir):
    logger = ctx.obj['LOGGER']

    logger.info("Training classifiers")
    if split_filepath is not None:
        split_df = pd.read_csv(split_filepath)
    else:
        split_df = None

    ct = _init_classifier_trainer(pixel_features_builder_kws,
                                  pixel_response_builder_kws, adaboost_kws,
                                  num_estimators)

    clfs_dict = ct.train_classifiers(split_df, response_img_dir)

    if output_dir is None:
        output_dir = ''

    for img_cluster in clfs_dict:
        _dump_clf(clfs_dict[img_cluster],
                  path.join(output_dir, f"{img_cluster}.joblib"), logger)


@cli.command()
@click.pass_context
@click.argument('img_filepath', type=click.Path(exists=True))
@click.argument('clf_filepath', type=click.Path(exists=True))
@click.option('--tree-val', type=int)
@click.option('--nontree-val', type=int)
@click.option('--refine', is_flag=True)
@click.option('--refine-beta', type=int)
@click.option('--refine-int-rescale', type=int)
@click.option('--pixel-features-builder-kws', cls=OptionEatAll)
@click.option('--output-filepath', type=click.Path())
def classify_img(ctx, img_filepath, clf_filepath, tree_val, nontree_val,
                 refine, refine_beta, refine_int_rescale,
                 pixel_features_builder_kws, output_filepath):
    logger = ctx.obj['LOGGER']

    logger.info("Classifying %s with classifier of %s", img_filepath,
                clf_filepath)

    pixel_features_builder_kws = _dict_from_kws(pixel_features_builder_kws)
    c = dtr.Classifier(tree_val=tree_val, nontree_val=nontree_val,
                       refine=refine, refine_beta=refine_beta,
                       refine_int_rescale=refine_int_rescale,
                       **pixel_features_builder_kws)

    if output_filepath is None:
        filename, ext = path.splitext(path.basename(img_filepath))
        output_filepath = f"{filename}-pred{ext}"

    c.classify_img(img_filepath, joblib.load(clf_filepath), output_filepath)
    logger.info("Dumped predicted image to %s", output_filepath)


@cli.command()
@click.pass_context
@click.argument('split_filepath', type=click.Path(exists=True))
@click.option('--clf-filepath', type=click.Path(exists=True))
@click.option('--clf-dir', type=click.Path(exists=True))
@click.option('--method')
@click.option('--img-cluster', type=int)
@click.option('--tree-val', type=int)
@click.option('--nontree-val', type=int)
@click.option('--refine', is_flag=True)
@click.option('--refine-beta', type=int)
@click.option('--refine-int-rescale', type=int)
@click.option('--pixel-features-builder-kws', cls=OptionEatAll)
@click.option('--output-dir', type=click.Path(exists=True))
def classify_imgs(ctx, split_filepath, clf_filepath, clf_dir, method,
                  img_cluster, tree_val, nontree_val, refine, refine_beta,
                  refine_int_rescale, pixel_features_builder_kws, output_dir):
    logger = ctx.obj['LOGGER']

    split_df = pd.read_csv(split_filepath)

    if clf_filepath is not None:
        clf_dict = None
        clf = joblib.load(clf_filepath)
        logger.info("Classifying images from %s with classifier of %s",
                    split_filepath, clf_filepath)

    if clf_dir is not None:
        clf = None
        clf_dict = {}
        for img_cluster in split_df['img_cluster'].unique():
            clf_dict[img_cluster] = joblib.load(
                path.join(clf_dir, f"{img_cluster}.joblib"))

    pixel_features_builder_kws = _dict_from_kws(pixel_features_builder_kws)

    c = dtr.Classifier(tree_val=tree_val, nontree_val=nontree_val,
                       refine=refine, refine_beta=refine_beta,
                       refine_int_rescale=refine_int_rescale,
                       **pixel_features_builder_kws)

    if output_dir is None:
        output_dir = ''

    pred_imgs = c.classify_imgs(split_df, output_dir, clf=clf,
                                clf_dict=clf_dict, method=method,
                                img_cluster=img_cluster)
    logger.info("Dumped %d predicted images to %s", len(pred_imgs), output_dir)
