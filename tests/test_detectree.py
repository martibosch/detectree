import glob
import os
import shutil
import unittest
from os import path

import numpy as np
import pandas as pd
import rasterio as rio
from click import testing
from sklearn import ensemble

import detectree as dtr
from detectree import (filters, image_descriptor, pixel_features,
                       pixel_response, utils)
from detectree.cli import main


class TestImports(unittest.TestCase):
    def test_base_imports(self):
        import glob
        import itertools
        from os import path

        import dask
        import maxflow as mf
        import numpy as np
        import pandas as pd
        import rasterio as rio
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
        self.img_dir = 'tests/data/img'
        self.img_filepaths = glob.glob(path.join(self.img_dir, '*.tif'))

    def test_init(self):
        # if providing `img_filepaths`, `img_dir` and `img_filename_pattern`
        # are ignored
        ts = dtr.TrainingSelector(img_filepaths=self.img_filepaths)
        self.assertEqual(
            ts.img_filepaths,
            dtr.TrainingSelector(img_filepaths=self.img_filepaths,
                                 img_dir='foo').img_filepaths)
        self.assertEqual(
            ts.img_filepaths,
            dtr.TrainingSelector(img_filepaths=self.img_filepaths,
                                 img_filename_pattern='foo').img_filepaths)
        self.assertEqual(
            ts.img_filepaths,
            dtr.TrainingSelector(img_filepaths=self.img_filepaths,
                                 img_dir='foo',
                                 img_filename_pattern='foo').img_filepaths)

        # if not providing `img_filepaths`, inexistent `img_dir` or non-tif
        # `img_filename_pattern` will result in an empty `img_filepaths`
        # attribute
        self.assertEqual(
            len(dtr.TrainingSelector(img_dir='foo').img_filepaths), 0)
        self.assertEqual(
            len(
                dtr.TrainingSelector(
                    img_dir=self.img_dir,
                    img_filename_pattern='foo').img_filepaths), 0)
        # otherwise, there should be at least one img
        self.assertGreater(
            len(dtr.TrainingSelector(img_dir=self.img_dir).img_filepaths), 0)

        # even when providing an integer in the `gabor_num_orientations`
        # argument, the respective attribute will be a tuple after `__init__`
        # is executed
        self.assertIsInstance(
            dtr.TrainingSelector(
                self.img_filepaths,
                gabor_num_orientations=8).gabor_num_orientations, tuple)

    def test_train_test_split(self):
        ts = dtr.TrainingSelector(img_filepaths=self.img_filepaths)

        X = ts.descr_feature_matrix
        self.assertEqual(len(X), len(self.img_filepaths))

        # test `method` argument
        split_df = ts.train_test_split()
        self.assertIn('train', split_df)
        self.assertIn('img_cluster', split_df)
        split_df = ts.train_test_split(method='cluster-I')
        self.assertIn('train', split_df)
        self.assertNotIn('img_cluster', split_df)

        # test `return_evr` argument (expected variance ratio)
        split_df, evr = ts.train_test_split(return_evr=True)
        self.assertIsInstance(split_df, pd.DataFrame)
        self.assertIsInstance(evr, float)


class TestImageDescriptor(unittest.TestCase):
    def setUp(self):
        self.img_dir = 'tests/data/img'
        self.img_filepath = glob.glob(path.join(self.img_dir, '*.tif'))[0]

    def test_image_descriptor(self):
        gabor_frequencies = (.1, .25, .4)
        gabor_num_orientations = (4, 8, 8)
        kernels = filters.get_gabor_filter_bank(
            frequencies=gabor_frequencies,
            num_orientations=gabor_num_orientations)
        response_bins_per_axis = 4
        num_blocks = response_bins_per_axis**2
        num_color_bins = 8
        img_descr = image_descriptor.compute_image_descriptor_from_filepath(
            self.img_filepath, kernels, response_bins_per_axis, num_blocks,
            num_color_bins)
        self.assertEqual(len(img_descr),
                         len(kernels) * num_blocks + num_color_bins**3)

        # TODO: more technical test, e.g., passing an all-zero filter bank
        # should return an all-zero gist descriptor

        # TODO: more technical test, e.g., ensure that a negative number of
        # bins raises some NumPy error

        # TODO: more technical test, e.g., ensure that all values in
        # `img_descr_row` are within the unit norm


class TestPixelFeatures(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'tests/data'
        self.img_dir = path.join(self.data_dir, 'img')
        with rio.open(path.join(self.img_dir, '1091-322_00.tif')) as src:
            self.pixels_per_img = src.shape[0] * src.shape[1]
        self.split_i_df = pd.read_csv(
            path.join(self.data_dir, 'split_cluster-I.csv'), index_col=0)
        self.split_ii_df = pd.read_csv(
            path.join(self.data_dir, 'split_cluster-II.csv'), index_col=0)
        self.pfb = pixel_features.PixelFeaturesBuilder()
        # TODO: test arguments of `PixelFeaturesBuilder`

    def test_build_features(self):
        img_cluster = 0
        num_pixel_features = self.pfb.num_pixel_features

        shape_i = (len(self.split_i_df[self.split_i_df['train']]) *
                   self.pixels_per_img, num_pixel_features)
        shape_ii = (len(self.split_ii_df[self.split_ii_df['train'] & (
            self.split_ii_df['img_cluster'] == img_cluster)]) *
                    self.pixels_per_img, num_pixel_features)

        # test providing `method` implicitly (and `split_df`)
        self.assertEqual(
            self.pfb.build_features(self.split_i_df).shape, shape_i)
        self.assertEqual(
            self.pfb.build_features(self.split_ii_df,
                                    img_cluster=img_cluster).shape, shape_ii)

        # test providing `method` explicitly (and `split_df`)
        self.assertEqual(
            self.pfb.build_features(self.split_i_df, method='cluster-I').shape,
            shape_i)
        self.assertEqual(
            self.pfb.build_features(self.split_ii_df, method='cluster-II',
                                    img_cluster=img_cluster).shape, shape_ii)

        # test that `method='cluster-I'` will ignore the 'img_cluster' column
        # of the split data frame
        self.assertEqual(
            self.pfb.build_features(self.split_ii_df,
                                    method='cluster-I').shape,
            (len(self.split_ii_df[self.split_ii_df['train']]) *
             self.pixels_per_img, num_pixel_features))

        # test that `method='cluster-II'` and non-None `img_cluster` raises a
        # ValueError
        self.assertRaises(ValueError, self.pfb.build_features, self.split_i_df,
                          method='cluster-II')

        # test that `method='cluster-II'` raises a `ValueError` if `split_df`
        # does not have a `img_cluster` column (when using the method
        # 'cluster-I')
        self.assertRaises(ValueError, self.pfb.build_features, self.split_i_df,
                          method='cluster-II')

        # test providing `img_filepaths`
        img_filepaths = self.split_i_df[
            self.split_i_df['train']]['img_filepath']

        # the shape of the feature matrix below is the same as `shape_i`
        self.assertEqual(
            self.pfb.build_features(img_filepaths=img_filepaths).shape,
            (len(img_filepaths) * self.pixels_per_img, num_pixel_features))

        # test providing `img_dir`. In this case all the images (not only the
        # ones selected for training) will be transformed into feature vectors
        self.assertEqual(
            self.pfb.build_features(img_dir=self.img_dir).shape,
            (len(self.split_i_df) * self.pixels_per_img, num_pixel_features))

        # test that if none of `split_df`, `img_filepaths` or `img_dir` are
        # provided, a `ValueError` is raised
        self.assertRaises(ValueError, self.pfb.build_features)


class TestPixelResponse(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'tests/data'
        self.img_dir = path.join(self.data_dir, 'img')
        example_img_filename = '1091-322_00.tif'
        with rio.open(path.join(self.img_dir, example_img_filename)) as src:
            self.pixels_per_img = src.shape[0] * src.shape[1]
        self.split_i_df = pd.read_csv(
            path.join(self.data_dir, 'split_cluster-I.csv'), index_col=0)
        self.split_ii_df = pd.read_csv(
            path.join(self.data_dir, 'split_cluster-II.csv'), index_col=0)
        self.response_img_dir = path.join(self.data_dir, 'response_img')
        self.prb = pixel_response.PixelResponseBuilder()
        # TODO: test arguments of `PixelResponseBuilder`
        self.bad_response_img_filepath = path.join(self.data_dir,
                                                   'bad_response_img',
                                                   example_img_filename)

    def test_build_response(self):
        img_cluster = 0

        response_i = self.prb.build_response(self.split_i_df,
                                             self.response_img_dir)
        response_ii = self.prb.build_response(self.split_ii_df,
                                              self.response_img_dir,
                                              img_cluster=img_cluster)

        # test that all responses are ones and zeros
        for unique_response in (np.unique(response_i), np.unique(response_ii)):
            self.assertTrue(np.all(unique_response == np.arange(2)))
        # test shapes
        shape_i = (len(self.split_i_df[self.split_i_df['train']]) *
                   self.pixels_per_img, )
        shape_ii = (len(self.split_ii_df[self.split_ii_df['train'] & (
            self.split_ii_df['img_cluster'] == img_cluster)]) *
                    self.pixels_per_img, )

        # test for `response_i` and `response_ii`, which have been obtained by
        # providing `method` implicitly (and `split_df`)
        self.assertEqual(response_i.shape, shape_i)
        self.assertEqual(response_ii.shape, shape_ii)

        # test providing `method` implicitly (and `split_df`)
        self.assertEqual(
            self.prb.build_response(self.split_i_df, self.response_img_dir,
                                    method='cluster-I').shape, shape_i)
        self.assertEqual(
            self.prb.build_response(self.split_ii_df, self.response_img_dir,
                                    method='cluster-II',
                                    img_cluster=img_cluster).shape, shape_ii)

        # test that `method='cluster-I'` will ignore the 'img_cluster' column
        # of the split data frame
        self.assertEqual(
            self.prb.build_response(self.split_ii_df, self.response_img_dir,
                                    method='cluster-I').shape,
            (len(self.split_ii_df[self.split_ii_df['train']]) *
             self.pixels_per_img, ))

        # test that when providing `split_df`, `response_img_dir` is required
        self.assertRaises(ValueError, self.prb.build_response, self.split_i_df,
                          method='cluster-II')

        # test that `method='cluster-II'` and non-None `img_cluster` raises a
        # ValueError
        self.assertRaises(ValueError, self.prb.build_response, self.split_i_df,
                          self.response_img_dir, method='cluster-II')

        # test that `method='cluster-II'` raises a `ValueError` if `split_df`
        # does not have a `img_cluster` column (when using the method
        # 'cluster-I')
        self.assertRaises(ValueError, self.prb.build_response, self.split_i_df,
                          self.response_img_dir, method='cluster-II')

        # test providing `img_filepaths`
        img_filepaths = self.split_i_df[self.split_i_df['train']][
            'img_filepath'].apply(lambda filepath: path.join(
                self.response_img_dir, path.basename(filepath)))

        # the shape of the feature matrix below is the same as `shape_i`
        self.assertEqual(
            self.prb.build_response(
                response_img_filepaths=img_filepaths).shape,
            (len(img_filepaths) * self.pixels_per_img, ))

        # test that if none of `split_df`, `img_filepaths` or `img_dir` are
        # provided, a `ValueError` is raised
        self.assertRaises(ValueError, self.prb.build_response)

        # test that providing a response whose pixel values are not
        # exclusively the `tree_val` and `nontree_val` attributes of the
        # `PixelResponseBuilder` instance raises a `ValueError`
        self.assertRaises(ValueError, self.prb.build_response_from_filepath,
                          self.bad_response_img_filepath)


class TestTrainClassifier(unittest.TestCase):
    def setUp(self):
        self.img_cluster = 0
        self.data_dir = 'tests/data'
        self.img_dir = path.join(self.data_dir, 'img')
        self.split_i_df = pd.read_csv(
            path.join(self.data_dir, 'split_cluster-I.csv'), index_col=0)
        self.split_ii_df = pd.read_csv(
            path.join(self.data_dir, 'split_cluster-II.csv'), index_col=0)
        self.response_img_dir = path.join(self.data_dir, 'response_img')
        self.tmp_train_dir = path.join(self.data_dir, 'tmp_train')
        os.mkdir(self.tmp_train_dir)
        # this file must exist in `response_img`
        self.train_filename = '1091-322_00.tif'
        shutil.copyfile(path.join(self.img_dir, self.train_filename),
                        path.join(self.tmp_train_dir, self.train_filename))
        # to store temporary outputs
        self.tmp_output_dir = path.join(self.data_dir, 'tmp_output')
        os.mkdir(self.tmp_output_dir)

        # TODO: test init arguments of `ClassifierTrainer` other than
        # `num_estimators`
        num_estimators = 2  # to speed-up the tests
        self.ct = dtr.ClassifierTrainer(num_estimators=num_estimators)
        # cache this first trained classifier to reuse it below
        self.clf = self.ct.train_classifier(self.split_i_df,
                                            self.response_img_dir)
        # cache the classifier dict to reuse it below
        self.clf_dict = self.ct.train_classifiers(self.split_ii_df,
                                                  self.response_img_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_train_dir)
        shutil.rmtree(self.tmp_output_dir)

    def test_classifier_trainer(self):

        # test that all the combinations of arguments of the `train_classifier`
        # method return an instance of `sklearn.ensemble.AdaBoostClassifier`
        # option 1a: `split_df` and `response_img_dir` with implicit method
        # (note that we are using `self.clf` obtained in `setUp`)
        self.assertIsInstance(self.clf, ensemble.AdaBoostClassifier)
        self.assertIsInstance(
            self.ct.train_classifier(self.split_ii_df, self.response_img_dir,
                                     img_cluster=self.img_cluster),
            ensemble.AdaBoostClassifier)
        # option 1b: `split_df` and `response_img_dir` with explicit method
        self.assertIsInstance(
            self.ct.train_classifier(self.split_i_df, self.response_img_dir,
                                     method='cluster-I'),
            ensemble.AdaBoostClassifier)
        self.assertIsInstance(
            self.ct.train_classifier(self.split_ii_df, self.response_img_dir,
                                     method='cluster-II',
                                     img_cluster=self.img_cluster),
            ensemble.AdaBoostClassifier)
        # option 2: `img_filepaths` and `response_img_dir`
        img_filepaths = self.split_i_df[
            self.split_i_df['train']]['img_filepath']
        self.assertIsInstance(
            self.ct.train_classifier(img_filepaths=img_filepaths,
                                     response_img_dir=self.response_img_dir),
            ensemble.AdaBoostClassifier)
        # option 3: `img_filepaths` and `response_img_filepaths`
        response_img_filepaths = img_filepaths.apply(
            lambda filepath: path.join(self.response_img_dir,
                                       path.basename(filepath)))
        self.assertIsInstance(
            self.ct.train_classifier(
                img_filepaths=img_filepaths,
                response_img_filepaths=response_img_filepaths),
            ensemble.AdaBoostClassifier)
        # from here below, we use `self.tmp_train_dir`, which is a directory
        # with only one image, namely `self.train_filename`, so that the
        # training does not take long
        img_dir = self.tmp_train_dir
        # here we could use `img_dir` or `self.img_dir`
        img_filepaths = [path.join(self.img_dir, self.train_filename)]
        response_img_filepaths = [
            path.join(self.response_img_dir, self.train_filename)
        ]
        # option 4: `img_dir` and `response_img_dir`
        self.assertIsInstance(
            self.ct.train_classifier(img_dir=img_dir,
                                     response_img_dir=self.response_img_dir),
            ensemble.AdaBoostClassifier)
        # option 5: `img_dir` and `response_img_filepaths`
        self.assertIsInstance(
            self.ct.train_classifier(
                img_dir=img_dir,
                response_img_filepaths=response_img_filepaths),
            ensemble.AdaBoostClassifier)
        # option 6: `img_filepaths` and `response_img_dir`
        self.assertIsInstance(
            self.ct.train_classifier(img_filepaths=img_filepaths,
                                     response_img_dir=self.response_img_dir),
            ensemble.AdaBoostClassifier)
        # option 7: `img_filepaths` and `response_img_filepaths`
        self.assertIsInstance(
            self.ct.train_classifier(
                img_filepaths=img_filepaths,
                response_img_filepaths=response_img_filepaths),
            ensemble.AdaBoostClassifier)

        # test that either `split_df`, `img_filepaths` or `img_dir` must be
        # provided
        self.assertRaises(ValueError, self.ct.train_classifier)

        # test that `train_classifiers` raises a `ValueError` if `split_df`
        # doesn't have a 'img_cluster' column
        self.assertRaises(ValueError, self.ct.train_classifiers,
                          self.split_i_df, self.response_img_dir)
        # test that `train_classifiers` returns a dict otherwise
        # (note that we are using `self.clf_dict` obtained in `setUp`)
        self.assertIsInstance(self.clf_dict, dict)

    def _test_imgs_exist_and_rm(self, pred_imgs):
        for pred_img in pred_imgs:
            self.assertTrue(os.path.exists(pred_img))
            # remove it so that the output dir is clean in the tests below
            os.remove(pred_img)

    def test_classifier(self):
        # TODO: test init arguments of `Classifier`
        c = dtr.Classifier()

        img_filepath = self.split_i_df.iloc[0]['img_filepath']
        # test that `classify_img` returns a ndarray
        self.assertIsInstance(c.classify_img(img_filepath, self.clf),
                              np.ndarray)
        # test that `classify_img` with `output_filepath` returns a ndarray
        # and dumps it
        output_filepath = path.join(self.tmp_output_dir, 'foo.tif')
        y_pred = c.classify_img(img_filepath, self.clf, output_filepath)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertTrue(os.path.exists(output_filepath))
        # remove it so that the output dir is clean in the tests below
        os.remove(output_filepath)

        # test that `classify_imgs` with implicit `cluster-I` method returns a
        # list and that the images have been dumped
        pred_imgs = c.classify_imgs(self.split_i_df, self.tmp_output_dir,
                                    self.clf)
        self.assertIsInstance(pred_imgs, list)
        self._test_imgs_exist_and_rm(pred_imgs)

        # test that `classify_imgs` with implicit `cluster-II` method, `clf`
        # and `img_label` returns a list and that the images have been dumped
        pred_imgs = c.classify_imgs(self.split_ii_df, self.tmp_output_dir,
                                    self.clf, img_cluster=self.img_cluster)
        self.assertIsInstance(pred_imgs, list)
        self._test_imgs_exist_and_rm(pred_imgs)
        # test that this works equally when providing `clf_dict`
        pred_imgs = c.classify_imgs(self.split_ii_df, self.tmp_output_dir,
                                    clf_dict=self.clf_dict,
                                    img_cluster=self.img_cluster)
        self.assertIsInstance(pred_imgs, list)
        self._test_imgs_exist_and_rm(pred_imgs)

        # test that `classify_imgs` with implicit `cluster-II` method and
        # `clf_dict` returns a dict and that the images have been dumped
        pred_imgs = c.classify_imgs(self.split_ii_df, self.tmp_output_dir,
                                    clf_dict=self.clf_dict)
        self.assertIsInstance(pred_imgs, dict)
        for img_cluster in pred_imgs:
            self._test_imgs_exist_and_rm(pred_imgs[img_cluster])

        # test that `clf=None` with 'cluster-I' raises a `ValueError`
        self.assertRaises(ValueError, c.classify_imgs, self.split_i_df,
                          self.tmp_output_dir)

        # test that `clf=None` and `clf_dict=None` with 'cluster-II' raises a
        # `ValueError`
        self.assertRaises(ValueError, c.classify_imgs, self.split_ii_df,
                          self.tmp_output_dir)
        # test that `clf_dict=None` with 'cluster-II' and `img_cluster=None`
        # raises a `ValueError`, even when providing a non-None `clf`
        self.assertRaises(ValueError, c.classify_imgs, self.split_ii_df,
                          self.tmp_output_dir, clf=c)

        # TODO: test with explicit `method` keyword argument

        # test that `Classifier` with `refine=False` also returns an ndarray
        c = dtr.Classifier(refine=False)
        img_filepath = self.split_i_df.iloc[0]['img_filepath']
        # test that `classify_img` returns a ndarray
        self.assertIsInstance(c.classify_img(img_filepath, self.clf),
                              np.ndarray)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'tests/data'
        self.img_dir = path.join(self.data_dir, 'img')
        self.img_filepath = path.join(self.data_dir, 'big_img/1091-322_00.tif')
        self.tmp_tiles_dir = path.join(self.data_dir, 'tiles')
        os.mkdir(self.tmp_tiles_dir)
        self.split_i_df = pd.read_csv(
            path.join(self.data_dir, 'split_cluster-I.csv'), index_col=0)

    def tearDown(self):
        shutil.rmtree(self.tmp_tiles_dir)

    def test_split_into_tiles(self):
        tiles = dtr.split_into_tiles(self.img_filepath, self.tmp_tiles_dir)
        full_tiles = dtr.split_into_tiles(self.img_filepath,
                                          self.tmp_tiles_dir,
                                          only_full_tiles=True)
        maybe_empty_tiles = dtr.split_into_tiles(self.img_filepath,
                                                 self.tmp_tiles_dir,
                                                 keep_empty_tiles=True)
        self.assertTrue(len(full_tiles) <= len(tiles))
        self.assertTrue(len(tiles) <= len(maybe_empty_tiles))

    def test_get_img_filepaths(self):
        self.assertRaises(ValueError, utils.get_img_filepaths, self.split_i_df,
                          0, True)

    def test_logging(self):
        # Taken from OSMnx
        # https://github.com/gboeing/osmnx/blob/master/tests/test_osmnx.py
        import logging as lg
        utils.log('test a fake debug', level=lg.DEBUG)
        utils.log('test a fake info', level=lg.INFO)
        utils.log('test a fake warning', level=lg.WARNING)
        utils.log('test a fake error', level=lg.ERROR)


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'tests/data'
        self.img_dir = path.join(self.data_dir, 'img')
        self.models_dir = path.join(self.data_dir, 'models')
        self.response_img_dir = path.join(self.data_dir, 'response_img')

        self.split_ii_filepath = path.join(self.data_dir,
                                           'split_cluster-II.csv')

        self.tmp_dir = path.join(self.data_dir, 'tmp')
        os.mkdir(self.tmp_dir)

        self.num_estimators = 2  # to speed-up the tests

        self.runner = testing.CliRunner()

        # TODO: test more possibilities of `args` in `invoke`
        # TODO: test `_dict_from_kws`

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_version(self):
        # Taken from rasterio
        # https://github.com/mapbox/rasterio/blob/master/tests/test_cli_main.py
        result = self.runner.invoke(main.cli, ['--version'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(dtr.__version__, result.output)

    def test_train_test_split(self):
        result = self.runner.invoke(main.cli, [
            'train-test-split', '--img-dir', self.img_dir, '--output-filepath',
            path.join(self.tmp_dir, 'split.csv')
        ])
        self.assertEqual(result.exit_code, 0)

    def test_train_classifier(self):
        result = self.runner.invoke(main.cli, [
            'train-classifier', '--split-filepath',
            path.join(self.data_dir, 'split_cluster-I.csv'),
            '--response-img-dir', self.response_img_dir, '--num-estimators',
            self.num_estimators, '--output-filepath',
            path.join(self.tmp_dir, 'clf.joblib')
        ])
        self.assertEqual(result.exit_code, 0)

    def test_train_classifiers(self):
        result = self.runner.invoke(main.cli, [
            'train-classifiers', self.split_ii_filepath, self.response_img_dir,
            '--num-estimators', self.num_estimators, '--output-dir',
            self.tmp_dir
        ])
        self.assertEqual(result.exit_code, 0)

    def test_classify_img(self):
        result = self.runner.invoke(main.cli, [
            'classify-img',
            glob.glob(path.join(self.img_dir, '*.tif'))[0],
            path.join(self.models_dir, 'clf.joblib'), '--output-filepath',
            path.join(self.tmp_dir, 'foo.tif')
        ])
        self.assertEqual(result.exit_code, 0)

    def test_classify_imgs(self):
        result = self.runner.invoke(main.cli, [
            'classify-imgs', self.split_ii_filepath, '--clf-dir',
            self.models_dir, '--output-dir', self.tmp_dir
        ])
        self.assertEqual(result.exit_code, 0)
