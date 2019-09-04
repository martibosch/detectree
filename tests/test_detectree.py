import glob
import unittest
from os import path

import pandas as pd

import detectree as dtr


class TestImports(unittest.TestCase):
    def test_base_imports(self):
        import glob
        import itertools
        from os import path

        import dask
        import maxflow as mf
        import numpy as np
        import pandas as pd
        import rasterio
        from dask import diagnostics
        from rasterio import windows
        from scipy import ndimage as ndi
        from scipy.ndimage.filters import _gaussian_kernel1d
        from skimage import color, measure, morphology
        from skimage.filters import gabor_kernel, rank
        from skimage.util import shape
        from sklearn import cluster, decomposition, ensemble, metrics, \
            preprocessing


class TestTrainTestSplit(unittest.TestCase):
    def setUp(self):
        self.tile_dir = 'tests/data'
        self.tile_filepaths = glob.glob(path.join(self.tile_dir, '*.tif'))

    def test_init(self):
        # if providing `tile_filepaths`, `tile_dir` and `tile_filename_pattern`
        # are ignored
        ts = dtr.TrainingSelector(tile_filepaths=self.tile_filepaths)
        self.assertEqual(
            ts.tile_filepaths,
            dtr.TrainingSelector(tile_filepaths=self.tile_filepaths,
                                 tile_dir='foo').tile_filepaths)
        self.assertEqual(
            ts.tile_filepaths,
            dtr.TrainingSelector(tile_filepaths=self.tile_filepaths,
                                 tile_filename_pattern='foo').tile_filepaths)
        self.assertEqual(
            ts.tile_filepaths,
            dtr.TrainingSelector(tile_filepaths=self.tile_filepaths,
                                 tile_dir='foo',
                                 tile_filename_pattern='foo').tile_filepaths)

        # if not providing `tile_filepaths`, inexistent `tile_dir` or non-tif
        # `tile_filename_pattern` will result in an empty `tile_filepaths`
        # attribute
        self.assertEqual(
            len(dtr.TrainingSelector(tile_dir='foo').tile_filepaths), 0)
        self.assertEqual(
            len(
                dtr.TrainingSelector(
                    tile_dir=self.tile_dir,
                    tile_filename_pattern='foo').tile_filepaths), 0)
        # otherwise, there should be at least one tile
        self.assertGreater(
            len(dtr.TrainingSelector(tile_dir=self.tile_dir).tile_filepaths),
            0)

        # even when providing an integer in the `gabor_num_orientations`
        # argument, the respective attribute will be a tuple after `__init__`
        # is executed
        self.assertIsInstance(
            dtr.TrainingSelector(
                self.tile_filepaths,
                gabor_num_orientations=8).gabor_num_orientations, tuple)

        def test_train_test_split(self):
            ts = dtr.TrainingSelector(tile_filepaths=self.tile_filepaths)

            X = ts.descr_feature_matrix
            self.assertEqual(len(X), len(self.tile_filepaths))

            # test `method` argument
            split_df = ts.train_test_split()
            self.assertIn('train', split_df)
            self.assertIn('tile_cluster', split_df)
            split_df = ts.train_test_split(method='I')
            self.assertIn('train', split_df)
            self.assertNotIn('tile_cluster', split_df)

            # test `return_evr` argument (expected variance ratio)
            split_df, evr = ts.train_test_split(return_evr=True)
            self.assertIsInstance(split_df, pd.DataFrame)
            self.assertIsInstance(evr, float)
