import logging
import os

import click
import pandas as pd

import detectree as dtr

try:
    from dask import distributed
except ImportError:
    distributed = None


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


@click.group()
@click.pass_context
def cli(ctx):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    ctx.ensure_object(dict)
    ctx.obj['LOGGER'] = logger


@cli.command()
@click.pass_context
@click.option('--tile-filepaths', required=False, cls=OptionEatAll)
@click.option('--tile-dir', type=click.Path(exists=True), required=False)
@click.option('--tile-filename-pattern', required=False)
@click.option('--gabor-frequencies', required=False, cls=OptionEatAll)
@click.option('--gabor-num-orientations', required=False, cls=OptionEatAll)
@click.option('--response-bins-per-axis', type=int, required=False)
@click.option('--num-color-bins', type=int, required=False)
@click.option('--method', required=False)
@click.option('--num-components', type=int, required=False)
@click.option('--num-tile-clusters', type=int, required=False)
@click.option('--train-prop', type=float, required=False)
@click.option('--output-filepath', type=click.Path(), required=True)
@click.option('--num-workers', type=int, default=4, required=False)
@click.option('--threads-per-worker', type=float, default=4, required=False)
def train_test_split(ctx, tile_filepaths, tile_dir, tile_filename_pattern,
                     gabor_frequencies, gabor_num_orientations,
                     response_bins_per_axis, num_color_bins, method,
                     num_components, num_tile_clusters, train_prop,
                     output_filepath, num_workers, threads_per_worker):
    logger = ctx.obj['LOGGER']

    if distributed:
        client = distributed.Client(n_workers=num_workers,
                                    threads_per_worker=threads_per_worker)
        logger.info("Started dask client: %s", client)

    # there must be at least `tile_filepaths` or `tile_dir`
    if tile_filepaths is None and tile_dir is None:
        raise click.UsageError(
            "An option out of `--tile-filepaths` or `--tile-dir` must be "
            "provided")
    if tile_filepaths:
        # ensure that each `tile_filepath` exist and throw a nice error
        # message otherwise - this is already managed by passing
        # `click.Path(exists=True)` to the `@click.option` decorator,
        # nevertheless, we cannot do it here because we are using
        # `OptionEatAll`
        for tile_filepath in tile_filepaths:
            try:
                os.stat(tile_filepath)
            except OSError:
                raise click.UsageError('Tile "%s" does not exist.' %
                                       tile_filepath)

    # process the rest of train/test split keyword arguments
    ts_kws = {}
    if tile_filename_pattern is not None:
        ts_kws['tile_filename_pattern'] = tile_filename_pattern
    if gabor_frequencies is not None:
        ts_kws['gabor_frequencies'] = gabor_frequencies
    if gabor_num_orientations is not None:
        ts_kws['gabor_num_orientations'] = gabor_num_orientations
    if response_bins_per_axis is not None:
        ts_kws['response_bins_per_axis'] = response_bins_per_axis
    if num_color_bins is not None:
        ts_kws['num_color_bins'] = num_color_bins

    ts = dtr.TrainingSelector(tile_filepaths=tile_filepaths, tile_dir=tile_dir,
                              **ts_kws)
    logger.info("Loaded %d tiles from %s", len(ts.tile_filepaths), tile_dir)

    tts_kws = {}
    if method is not None:
        tts_kws['method'] = method
    if num_components is not None:
        tts_kws['num_components'] = num_components
    if num_tile_clusters is not None:
        tts_kws['num_tile_clusters'] = num_tile_clusters
    if train_prop is not None:
        tts_kws['train_prop'] = train_prop
    df, evr = ts.train_test_split(return_evr=True, **tts_kws)
    logger.info("Variance ratio explained by PCA: %f", evr)

    if distributed:
        client.close()
        logger.info("Closed dask client")

    df.to_csv(output_filepath)
    logger.info("Dumped train/test split to %s", output_filepath)


@cli.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def out_train_tiles(input_filepath):
    df = pd.read_csv(input_filepath)
    tile_filepaths = df[df['train']]['tile_filepath']
    print(' '.join(str(tile_filepath) for tile_filepath in tile_filepaths))
