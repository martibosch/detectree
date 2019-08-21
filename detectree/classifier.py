from os import path

import maxflow as mf
import numpy as np
import rasterio as rio
from sklearn import ensemble

from . import pixel_features, pixel_response, settings

__all__ = ['train_classifier', 'classify']

MOORE_NEIGHBORHOOD_ARR = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])

# TODO:
# class Classifier(object):
#     pass


def train_classifier(tile_filepaths=None, response_tile_filepaths=None,
                     tile_dir=None, tile_filename_pattern=None,
                     response_tile_dir=None,
                     response_tile_filename_pattern=None, X=None, y=None,
                     num_estimators=None, **adaboost_kws):

    if X is None:
        X = pixel_features.pixel_features_from_tiling(
            tile_filepaths=tile_filepaths, tile_dir=tile_dir,
            tile_filename_pattern=tile_filename_pattern)
    if y is None:
        y = pixel_response.pixel_response_from_tiling(
            tile_filepaths=response_tile_filepaths, tile_dir=response_tile_dir,
            tile_filename_pattern=response_tile_filename_pattern)

    if num_estimators is None:
        num_estimators = settings.ADABOOST_DEFAULT_NUM_ESTIMATORS

    clf = ensemble.AdaBoostClassifier(n_estimators=num_estimators,
                                      **adaboost_kws)
    clf.fit(X, y)
    # TODO: log clf.score(X, y)?
    return clf


# def train_classifiers_from_split_df(split_df, response_train_tiles_dir,
#                                     method=None, output_dir=None,
#                                     num_estimators=None, **adaboost_kws):
#     def dump_classifier(df, output_filepath):
#         tile_filepaths = df[df['train']]['tile_filepath']
#         response_tile_filepaths = tile_filepaths.apply(
#             utils.get_response_tile_filepath,
#             args=(response_train_tiles_dir, ))
#         clf = train_classifier(tile_filepaths=tile_filepaths,
#                                response_tile_filepaths=response_tile_filepaths,
#                                num_estimators=num_estimators, **adaboost_kws)
#         jl.dump(clf, output_filepath)

#     if method is None:
#         if 'tile_cluster' in split_df:
#             method = 'II'
#         else:
#             method = 'I'

#     if num_estimators is None:
#         num_estimators = settings.ADABOOST_DEFAULT_NUM_ESTIMATORS

#     if method == 'I':
#         dump_classifier(split_df, path.join(output_dir, "clf.joblib"))
#     else:
#         for cluster_label, cluster_df in split_df.groupby('tile_cluster'):
#             dump_classifier(
#                 cluster_df, path.join(output_dir,
#                                       f"clf{cluster_label}.joblib"))


def classify(clf, tile_filepaths=None, tile_dir=None,
             tile_filename_pattern=None, X=None, tile_shape=None, refine=True,
             refine_beta=None, refine_scale=None, output_dir=None):

    if X is None:
        X = pixel_features.pixel_features_from_tiling(
            tile_filepaths=tile_filepaths, tile_dir=tile_dir,
            tile_filename_pattern=tile_filename_pattern)

    if not refine:
        return clf.predict(X)
    else:
        if refine_beta is None:
            refine_beta = settings.REFINE_DEFAULT_BETA
        if refine_scale is None:
            refine_scale = settings.REFINE_DEFAULT_SCALE

        def _classify_refine(X_tile):
            p_nontree, p_tree = np.hsplit(clf.predict_proba(X_tile), 2)
            g = mf.Graph[int]()
            node_ids = g.add_grid_nodes(tile_shape)
            P_nontree = p_nontree.reshape(tile_shape)
            P_tree = p_tree.reshape(tile_shape)

            # The AdaBoost probabilities are floats between 0 and 1, and the
            # graph cuts algorithm requires an integer representation.
            # Therefore, we will multiply the probabilities by an arbitrary
            # large number and then transform the result to integers. For
            # instance, we could use a `refine_scale` of `100` so that the
            # probabilities are rescaled into integers between 0 and 100 (like
            # percentages). The larger `refine_scale`, the larger the
            # precision.
            # ACHTUNG: the data term when the pixel is a tree is
            # `log(1 - P_tree)`, i.e., `log(P_nontree)`, so the two lines
            # below are correct
            D_tree = (refine_scale * np.log(P_nontree)).astype(int)
            D_nontree = (refine_scale * np.log(P_tree)).astype(int)
            # TODO: option to choose Moore/Von Neumann neighborhood?
            g.add_grid_edges(node_ids, refine_beta,
                             structure=MOORE_NEIGHBORHOOD_ARR)
            g.add_grid_tedges(node_ids, D_tree, D_nontree)
            g.maxflow()
            return g.get_grid_segments(node_ids)

        z_tiles = []
        if tile_filepaths is None:
            num_tile_pixels = tile_shape[0] * tile_shape[1]
            for tile_start in range(0, len(X), num_tile_pixels):
                tile_end = tile_start + num_tile_pixels
                z_tile = _classify_refine(X[tile_start:tile_end])
                if output_dir is None:
                    z_tiles.append(z_tile)
                else:
                    # TODO: make the profile of output rasters more
                    # customizable (e.g., via the `settings` module)
                    output_filepath = path.join(
                        output_dir, f"tile_{tile_start}-{tile_end}.tif")
                    with rio.open(output_filepath, 'w', driver='GTiff',
                                  width=z_tile.shape[1],
                                  height=z_tile.shape[0], count=1,
                                  dtype='uint8', nodata=255) as dst:
                        dst.write(z_tile.astype(np.uint8) * 255, 1)
                    z_tiles.append(output_filepath)
        else:
            tile_start = 0
            for tile_filepath in tile_filepaths:
                with rio.open(tile_filepath) as src:
                    tile_shape = src.shape
                    num_tile_pixels = tile_shape[0] * tile_shape[1]
                    z_tile = _classify_refine(X[tile_start:tile_start +
                                                num_tile_pixels])
                    tile_start += num_tile_pixels
                    if output_dir is None:
                        z_tiles.append(z_tile)
                    else:
                        # TODO: make the profile of output rasters more
                        # customizable (e.g., via the `settings` module)
                        meta = src.meta.copy()
                        meta.update({
                            'count': 1,
                            'dtype': 'uint8',
                            'nodata': 255
                        })
                        output_filepath = path.join(
                            output_dir, path.basename(tile_filepath))
                        with rio.open(output_filepath, 'w', **meta) as dst:
                            dst.write(z_tile.astype(np.uint8) * 255, 1)
                        z_tiles.append(output_filepath)

        return z_tiles
