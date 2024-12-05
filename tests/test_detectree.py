import glob
import os
import shutil
import unittest
from os import path

import numpy as np
import pandas as pd
import rasterio as rio
from click import testing
from scipy import ndimage as ndi

import detectree as dtr
from detectree import (
    filters,
    image_descriptor,
    pixel_features,
    pixel_response,
    settings,
    utils,
)
from detectree.cli import main


def _create_tmp_dir(tmp_dir):
    if path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)


class TestImports(unittest.TestCase):
    def test_base_imports(self):
        pass


class TestTrainTestSplit(unittest.TestCase):
    def setUp(self):
        self.img_dir = "tests/data/img"
        self.img_filepaths = glob.glob(path.join(self.img_dir, "*.tif"))

    def test_init(self):
        # if providing `img_filepaths`, `img_dir` and `img_filename_pattern` are ignored
        ts = dtr.TrainingSelector(img_filepaths=self.img_filepaths)
        self.assertEqual(
            ts.img_filepaths,
            dtr.TrainingSelector(
                img_filepaths=self.img_filepaths, img_dir="foo"
            ).img_filepaths,
        )
        self.assertEqual(
            ts.img_filepaths,
            dtr.TrainingSelector(
                img_filepaths=self.img_filepaths, img_filename_pattern="foo"
            ).img_filepaths,
        )
        self.assertEqual(
            ts.img_filepaths,
            dtr.TrainingSelector(
                img_filepaths=self.img_filepaths,
                img_dir="foo",
                img_filename_pattern="foo",
            ).img_filepaths,
        )

        # if not providing `img_filepaths`, inexistent `img_dir` or non-tif
        # `img_filename_pattern` will result in an empty `img_filepaths` attribute
        self.assertEqual(len(dtr.TrainingSelector(img_dir="foo").img_filepaths), 0)
        self.assertEqual(
            len(
                dtr.TrainingSelector(
                    img_dir=self.img_dir, img_filename_pattern="foo"
                ).img_filepaths
            ),
            0,
        )
        # otherwise, there should be at least one img
        self.assertGreater(
            len(dtr.TrainingSelector(img_dir=self.img_dir).img_filepaths), 0
        )

        # even when providing an integer in the `gabor_num_orientations` argument, the
        # respective attribute will be a tuple after `__init__` is executed
        self.assertIsInstance(
            dtr.TrainingSelector(
                img_filepaths=self.img_filepaths, gabor_num_orientations=8
            ).gabor_num_orientations,
            tuple,
        )

    def test_train_test_split(self):
        ts = dtr.TrainingSelector(img_filepaths=self.img_filepaths)

        X = ts.descr_feature_matrix
        self.assertEqual(len(X), len(self.img_filepaths))

        # test `method` argument
        split_df = ts.train_test_split()
        self.assertIn("train", split_df)
        self.assertIn("img_cluster", split_df)
        split_df = ts.train_test_split(method="cluster-I")
        self.assertIn("train", split_df)
        self.assertNotIn("img_cluster", split_df)

        # test `return_evr` argument (expected variance ratio)
        split_df, evr = ts.train_test_split(return_evr=True)
        self.assertIsInstance(split_df, pd.DataFrame)
        self.assertIsInstance(evr, float)

        # test pca n_components and kwargs
        # evr is greater or equal with more components (given the same seed)
        random_state = 42
        self.assertGreaterEqual(
            *(
                ts.train_test_split(
                    return_evr=True,
                    pca_kwargs={
                        "n_components": n_components,
                        "random_state": random_state,
                    },
                )[1]
                for n_components in (4, 2)
            )
        )
        # test kwargs for kmeans too
        # the result should be the same with the same seed
        tts_kwargs = dict(
            pca_kwargs={"random_state": random_state},
            kmeans_kwargs={"random_state": random_state},
        )
        self.assertTrue(
            ts.train_test_split(**tts_kwargs).equals(ts.train_test_split(**tts_kwargs))
        )


class TestImageDescriptor(unittest.TestCase):
    def setUp(self):
        self.img_dir = "tests/data/img"
        self.img_filepath = glob.glob(path.join(self.img_dir, "*.tif"))[0]

    def test_image_descriptor(self):
        gabor_frequencies = (0.1, 0.25, 0.4)
        gabor_num_orientations = (4, 8, 8)
        kernels = filters.get_gabor_filter_bank(
            frequencies=gabor_frequencies,
            num_orientations=gabor_num_orientations,
        )
        response_bins_per_axis = 4
        # num_blocks = response_bins_per_axis**2
        num_color_bins = 8
        img_descr = image_descriptor.compute_image_descriptor_from_filepath(
            self.img_filepath, kernels, response_bins_per_axis, num_color_bins
        )
        self.assertEqual(
            len(img_descr),
            len(kernels) * response_bins_per_axis**2 + num_color_bins**3,
        )

        # TODO: more technical test, e.g., passing an all-zero filter bank should return
        # an all-zero gist descriptor

        # TODO: more technical test, e.g., ensure that a negative number of bins raises
        # some numpy error

        # TODO: more technical test, e.g., ensure that all values in `img_descr_row` are
        # within the unit norm


class TestPixelFeatures(unittest.TestCase):
    def setUp(self):
        self.data_dir = "tests/data"
        self.img_dir = path.join(self.data_dir, "img")
        with rio.open(path.join(self.img_dir, "1091-322_00.tif")) as src:
            self.pixels_per_img = src.shape[0] * src.shape[1]
        self.split_i_df = pd.read_csv(
            path.join(self.data_dir, "split_cluster-I.csv"), index_col=0
        )
        self.split_ii_df = pd.read_csv(
            path.join(self.data_dir, "split_cluster-II.csv"), index_col=0
        )
        self.pfb = pixel_features.PixelFeaturesBuilder()
        # TODO: test arguments of `PixelFeaturesBuilder`

    def test_build_features(self):
        img_cluster = 0
        num_pixel_features = self.pfb.num_pixel_features

        shape_i = (
            len(self.split_i_df[self.split_i_df["train"]]) * self.pixels_per_img,
            num_pixel_features,
        )
        shape_ii = (
            len(
                self.split_ii_df[
                    self.split_ii_df["train"]
                    & (self.split_ii_df["img_cluster"] == img_cluster)
                ]
            )
            * self.pixels_per_img,
            num_pixel_features,
        )

        # test that when providing `split_df` we also need to provide `img_dir`
        self.assertRaises(
            ValueError,
            self.pfb.build_features,
            split_df=self.split_i_df,
        )
        # test providing `method` implicitly (and `split_df`)
        self.assertEqual(
            self.pfb.build_features(
                split_df=self.split_i_df, img_dir=self.img_dir
            ).shape,
            shape_i,
        )
        self.assertEqual(
            self.pfb.build_features(
                split_df=self.split_ii_df, img_dir=self.img_dir, img_cluster=img_cluster
            ).shape,
            shape_ii,
        )

        # test providing `method` explicitly (and `split_df`)
        self.assertEqual(
            self.pfb.build_features(
                split_df=self.split_i_df, img_dir=self.img_dir, method="cluster-I"
            ).shape,
            shape_i,
        )
        self.assertEqual(
            self.pfb.build_features(
                split_df=self.split_ii_df,
                img_dir=self.img_dir,
                method="cluster-II",
                img_cluster=img_cluster,
            ).shape,
            shape_ii,
        )

        # test that `method='cluster-I'` will ignore the 'img_cluster' column of the
        # split data frame
        self.assertEqual(
            self.pfb.build_features(
                split_df=self.split_ii_df, img_dir=self.img_dir, method="cluster-I"
            ).shape,
            (
                len(self.split_ii_df[self.split_ii_df["train"]]) * self.pixels_per_img,
                num_pixel_features,
            ),
        )

        # test that `method='cluster-II'` and non-None `img_cluster` raises a ValueError
        self.assertRaises(
            ValueError,
            self.pfb.build_features,
            split_df=self.split_i_df,
            img_dir=self.img_dir,
            method="cluster-II",
        )

        # test that `method='cluster-II'` raises a `ValueError` if `split_df` does not
        # have a `img_cluster` column (when using the method 'cluster-I')
        self.assertRaises(
            ValueError,
            self.pfb.build_features,
            split_df=self.split_i_df,
            img_dir=self.img_dir,
            method="cluster-II",
        )

        # test providing `img_filepaths`
        img_filepath_ser = self.split_i_df[self.split_i_df["train"]][
            "img_filename"
        ].apply(lambda img_filename: path.join(self.img_dir, img_filename))

        # the shape of the feature matrix below is the same as `shape_i`
        self.assertEqual(
            self.pfb.build_features(img_filepaths=img_filepath_ser).shape,
            (len(img_filepath_ser) * self.pixels_per_img, num_pixel_features),
        )

        # test providing `img_dir`. In this case all the images (not only the ones
        # selected for training) will be transformed into feature vectors
        self.assertEqual(
            self.pfb.build_features(img_dir=self.img_dir).shape,
            (len(self.split_i_df) * self.pixels_per_img, num_pixel_features),
        )

        # test that if none of `split_df`, `img_filepaths` or `img_dir` are provided, a
        # `ValueError` is raised
        self.assertRaises(ValueError, self.pfb.build_features)


class TestPixelResponse(unittest.TestCase):
    def setUp(self):
        self.data_dir = "tests/data"
        self.img_dir = path.join(self.data_dir, "img")
        example_img_filename = "1091-322_00.tif"
        with rio.open(path.join(self.img_dir, example_img_filename)) as src:
            self.pixels_per_img = src.shape[0] * src.shape[1]
        self.split_i_df = pd.read_csv(
            path.join(self.data_dir, "split_cluster-I.csv"), index_col=0
        )
        self.split_ii_df = pd.read_csv(
            path.join(self.data_dir, "split_cluster-II.csv"), index_col=0
        )
        self.response_img_dir = path.join(self.data_dir, "response_img")
        self.prb = pixel_response.PixelResponseBuilder()
        # TODO: test arguments of `PixelResponseBuilder`
        self.bad_response_img_filepath = path.join(
            self.data_dir, "bad_response_img", example_img_filename
        )

    def test_build_response(self):
        img_cluster = 0

        response_i = self.prb.build_response(
            split_df=self.split_i_df, response_img_dir=self.response_img_dir
        )
        response_ii = self.prb.build_response(
            split_df=self.split_ii_df,
            response_img_dir=self.response_img_dir,
            img_cluster=img_cluster,
        )

        # test that all responses are ones and zeros
        for unique_response in (np.unique(response_i), np.unique(response_ii)):
            self.assertTrue(np.all(unique_response == np.arange(2)))
        # test shapes
        shape_i = (
            len(self.split_i_df[self.split_i_df["train"]]) * self.pixels_per_img,
        )
        shape_ii = (
            len(
                self.split_ii_df[
                    self.split_ii_df["train"]
                    & (self.split_ii_df["img_cluster"] == img_cluster)
                ]
            )
            * self.pixels_per_img,
        )

        # test for `response_i` and `response_ii`, which have been obtained by providing
        # `method` implicitly (and `split_df`)
        self.assertEqual(response_i.shape, shape_i)
        self.assertEqual(response_ii.shape, shape_ii)

        # test providing `method` implicitly (and `split_df`)
        self.assertEqual(
            self.prb.build_response(
                split_df=self.split_i_df,
                response_img_dir=self.response_img_dir,
                method="cluster-I",
            ).shape,
            shape_i,
        )
        self.assertEqual(
            self.prb.build_response(
                split_df=self.split_ii_df,
                response_img_dir=self.response_img_dir,
                method="cluster-II",
                img_cluster=img_cluster,
            ).shape,
            shape_ii,
        )

        # test that `method='cluster-I'` will ignore the 'img_cluster' column of the
        # split data frame
        self.assertEqual(
            self.prb.build_response(
                split_df=self.split_ii_df,
                response_img_dir=self.response_img_dir,
                method="cluster-I",
            ).shape,
            (len(self.split_ii_df[self.split_ii_df["train"]]) * self.pixels_per_img,),
        )

        # test that when providing `split_df`, `response_img_dir` is required
        self.assertRaises(
            ValueError,
            self.prb.build_response,
            split_df=self.split_i_df,
            method="cluster-II",
        )

        # test that `method='cluster-II'` and non-None `img_cluster` raises a ValueError
        self.assertRaises(
            ValueError,
            self.prb.build_response,
            split_df=self.split_i_df,
            response_img_dir=self.response_img_dir,
            method="cluster-II",
        )

        # test that `method='cluster-II'` raises a `ValueError` if `split_df` does not
        # have a `img_cluster` column (when using the method 'cluster-I')
        self.assertRaises(
            ValueError,
            self.prb.build_response,
            split_df=self.split_i_df,
            response_img_dir=self.response_img_dir,
            method="cluster-II",
        )

        # test providing `img_filepaths`
        img_filepaths = self.split_i_df[self.split_i_df["train"]]["img_filename"].apply(
            lambda filepath: path.join(self.response_img_dir, path.basename(filepath))
        )

        # the shape of the feature matrix below is the same as `shape_i`
        self.assertEqual(
            self.prb.build_response(response_img_filepaths=img_filepaths).shape,
            (len(img_filepaths) * self.pixels_per_img,),
        )

        # test that if none of `split_df`, `img_filepaths` or `img_dir` are provided, a
        # `ValueError` is raised
        self.assertRaises(ValueError, self.prb.build_response)

        # test that providing a response whose pixel values are not exclusively the
        # `tree_val` and `nontree_val` attributes of the `PixelResponseBuilder` instance
        # raises a `ValueError`
        self.assertRaises(
            ValueError,
            self.prb.build_response_from_filepath,
            self.bad_response_img_filepath,
        )


class TestTrainClassifier(unittest.TestCase):
    def setUp(self):
        self.img_cluster = 0
        self.data_dir = "tests/data"
        self.img_dir = path.join(self.data_dir, "img")
        self.split_i_df = pd.read_csv(
            path.join(self.data_dir, "split_cluster-I.csv"), index_col=0
        )
        self.split_ii_df = pd.read_csv(
            path.join(self.data_dir, "split_cluster-II.csv"), index_col=0
        )
        self.response_img_dir = path.join(self.data_dir, "response_img")
        self.tmp_train_dir = path.join(self.data_dir, "tmp_train")
        _create_tmp_dir(self.tmp_train_dir)
        # this file must exist in `response_img`
        self.train_filename = "1091-322_00.tif"
        shutil.copyfile(
            path.join(self.img_dir, self.train_filename),
            path.join(self.tmp_train_dir, self.train_filename),
        )
        # to store temporary outputs
        self.tmp_output_dir = path.join(self.data_dir, "tmp_output")
        _create_tmp_dir(self.tmp_output_dir)

        # TODO: test init arguments of `ClassifierTrainer` other than `n_estimators`
        # ACHTUNG: note that `n_estimators` is processed as `classifiers_kwargs`
        n_estimators = 2  # to speed-up the tests
        self.ct = dtr.ClassifierTrainer(n_estimators=n_estimators)
        # cache this first trained classifier to reuse it below
        self.clf = self.ct.train_classifier(
            split_df=self.split_i_df,
            img_dir=self.img_dir,
            response_img_dir=self.response_img_dir,
        )
        # cache the classifier dict to reuse it below
        self.clf_dict = self.ct.train_classifiers(
            self.split_ii_df, self.img_dir, self.response_img_dir
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_train_dir)
        shutil.rmtree(self.tmp_output_dir)

    def test_classifier_trainer(self):
        # test that all the combinations of arguments of the `train_classifier` method
        # return an instance of `sklearn.settings.CLF_DEFAULT_CLASS` option 1a:
        # `split_df` and `response_img_dir` with implicit method (note that we are using
        # `self.clf` obtained in `setUp`)
        self.assertIsInstance(self.clf, settings.CLF_CLASS)
        self.assertIsInstance(
            self.ct.train_classifier(
                split_df=self.split_ii_df,
                img_dir=self.img_dir,
                response_img_dir=self.response_img_dir,
                img_cluster=self.img_cluster,
            ),
            settings.CLF_CLASS,
        )
        # option 1b: `split_df` and `response_img_dir` with explicit method
        self.assertIsInstance(
            self.ct.train_classifier(
                split_df=self.split_i_df,
                img_dir=self.img_dir,
                response_img_dir=self.response_img_dir,
                method="cluster-I",
            ),
            settings.CLF_CLASS,
        )
        self.assertIsInstance(
            self.ct.train_classifier(
                split_df=self.split_ii_df,
                img_dir=self.img_dir,
                response_img_dir=self.response_img_dir,
                method="cluster-II",
                img_cluster=self.img_cluster,
            ),
            settings.CLF_CLASS,
        )
        # option 2: `img_filepaths` and `response_img_dir`
        img_filename_ser = self.split_i_df[self.split_i_df["train"]]["img_filename"]
        img_filepaths = img_filename_ser.apply(
            lambda img_filename: path.join(self.img_dir, img_filename)
        )
        self.assertIsInstance(
            self.ct.train_classifier(
                img_filepaths=img_filepaths,
                response_img_dir=self.response_img_dir,
            ),
            settings.CLF_CLASS,
        )
        # option 3: `img_filepaths` and `response_img_filepaths`
        response_img_filepaths = img_filename_ser.apply(
            lambda img_filename: path.join(self.response_img_dir, img_filename)
        )
        self.assertIsInstance(
            self.ct.train_classifier(
                img_filepaths=img_filepaths,
                response_img_filepaths=response_img_filepaths,
            ),
            settings.CLF_CLASS,
        )
        # from here below, we use `self.tmp_train_dir`, which is a directory with only
        # one image, namely `self.train_filename`, so that the training does not take
        # long
        img_dir = self.tmp_train_dir
        # here we could use `img_dir` or `self.img_dir`
        img_filepaths = [path.join(self.img_dir, self.train_filename)]
        response_img_filepaths = [path.join(self.response_img_dir, self.train_filename)]
        # option 4: `img_dir` and `response_img_dir`
        self.assertIsInstance(
            self.ct.train_classifier(
                img_dir=img_dir, response_img_dir=self.response_img_dir
            ),
            settings.CLF_CLASS,
        )
        # option 5: `img_dir` and `response_img_filepaths`
        self.assertIsInstance(
            self.ct.train_classifier(
                img_dir=img_dir, response_img_filepaths=response_img_filepaths
            ),
            settings.CLF_CLASS,
        )
        # option 6: `img_filepaths` and `response_img_dir`
        self.assertIsInstance(
            self.ct.train_classifier(
                img_filepaths=img_filepaths,
                response_img_dir=self.response_img_dir,
            ),
            settings.CLF_CLASS,
        )
        # option 7: `img_filepaths` and `response_img_filepaths`
        self.assertIsInstance(
            self.ct.train_classifier(
                img_filepaths=img_filepaths,
                response_img_filepaths=response_img_filepaths,
            ),
            settings.CLF_CLASS,
        )

        # test that either `split_df`, `img_filepaths` or `img_dir` must be provided
        self.assertRaises(ValueError, self.ct.train_classifier)

        # test that `train_classifiers` raises a `ValueError` if `split_df` doesn't have
        # a 'img_cluster' column
        self.assertRaises(
            ValueError,
            self.ct.train_classifiers,
            split_df=self.split_i_df,
            img_dir=self.img_dir,
            response_img_dir=self.response_img_dir,
        )
        # test that `train_classifiers` returns a dict otherwise (note that we are using
        # `self.clf_dict` obtained in `setUp`)
        self.assertIsInstance(self.clf_dict, dict)

    def _test_imgs_exist_and_rm(self, pred_imgs):
        for pred_img in pred_imgs:
            self.assertTrue(os.path.exists(pred_img))
            # remove it so that the output dir is clean in the tests below
            os.remove(pred_img)

    def _test_predict_img(self, c, img_filepath, *, img_cluster=None):
        # test that `predict_img` returns a ndarray
        self.assertIsInstance(
            c.predict_img(img_filepath, img_cluster=img_cluster), np.ndarray
        )
        # test that `predict_img` with `output_filepath` returns a ndarray and dumps it
        output_filepath = path.join(self.tmp_output_dir, "foo.tif")
        y_pred = c.predict_img(
            img_filepath, img_cluster=img_cluster, output_filepath=output_filepath
        )
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertTrue(os.path.exists(output_filepath))
        # remove it so that the output dir is clean in the tests below
        os.remove(output_filepath)

    def test_classifier(self):
        # define this here to reuse it below
        img_filepath = path.join(self.img_dir, self.split_i_df.iloc[0]["img_filename"])
        # test init classifier
        # TODO: test init arguments of `Classifier`
        # test that for the pre-trained classifier (no init `clf`/`clf_dict` arg) and
        # for the classifier initialized with the `clf` arg, the `clf` attribute is set
        # but not `clf_dict`
        for c in [dtr.Classifier(), dtr.Classifier(clf=self.clf)]:
            self.assertTrue(hasattr(c, "clf"))
            self.assertFalse(hasattr(c, "clf_dict"))
        # test that when initializing `clf_dict`, the `clf_dict` attribute is set but
        # not `clf`
        c = dtr.Classifier(clf_dict=self.clf_dict)
        self.assertFalse(hasattr(c, "clf"))
        self.assertTrue(hasattr(c, "clf_dict"))

        # test image segmentation separately for each method
        # "cluster-I"
        for c in [
            dtr.Classifier(),
            dtr.Classifier(clf=self.clf),
        ]:
            self._test_predict_img(c, img_filepath)

            # test that `classify_imgs` returns a list and that the images have been
            # dumped. This works regardless of whether a "img_cluster" column is present
            # in the split data frame - since it is ignored for "cluster-I"
            for split_df in [self.split_i_df, self.split_ii_df]:
                pred_imgs = c.predict_imgs(split_df, self.img_dir, self.tmp_output_dir)
                self.assertIsInstance(pred_imgs, list)
                self._test_imgs_exist_and_rm(pred_imgs)

        # test that `Classifier` with `refine=False` also returns an ndarray
        for c in [
            dtr.Classifier(refine=False),
            dtr.Classifier(clf=self.clf, refine=False),
        ]:
            # test that `classify_img` returns a ndarray
            self.assertIsInstance(c.predict_img(img_filepath), np.ndarray)

        # "cluster-II"
        c = dtr.Classifier(clf_dict=self.clf_dict)
        # `predict_img` should raise a `ValueError`:
        #   - if the `img_cluster` argument is not provided
        self.assertRaises(ValueError, c.predict_img, img_filepath)
        #   - if the provided `img_cluster` is not a key of `clf_dict`
        self.assertRaises(ValueError, c.predict_img, img_filepath, img_cluster=-999)
        # otherwise, it should work
        img_cluster = list(self.clf_dict.keys())[0]
        self._test_predict_img(c, img_filepath, img_cluster=img_cluster)
        # `predict_imgs` should raise a `ValueError` if `split_df` doesn't have an
        # "img_cluster" column
        self.assertRaises(
            KeyError, c.predict_imgs, self.split_i_df, self.img_dir, self.tmp_output_dir
        )
        # `classify_imgs` should raise a `KeyError` if `split_df` doesn't have a
        # "img_cluster" column
        self.assertRaises(
            KeyError, c.predict_imgs, self.split_i_df, self.img_dir, self.tmp_output_dir
        )
        # otherwise, it should work
        pred_imgs = c.predict_imgs(self.split_ii_df, self.img_dir, self.tmp_output_dir)
        self.assertIsInstance(pred_imgs, dict)
        for _img_cluster in pred_imgs:
            self._test_imgs_exist_and_rm(pred_imgs[_img_cluster])

        # test the `refine` argument
        for c in [
            dtr.Classifier(clf_dict=self.clf_dict, refine=refine)
            for refine in [True, False]
        ]:
            # test that `classify_img` returns a ndarray
            self.assertIsInstance(
                c.predict_img(img_filepath, img_cluster=img_cluster), np.ndarray
            )


class TestLidarToCanopy(unittest.TestCase):
    def setUp(self):
        self.lidar_data_dir = "tests/data/lidar"
        self.lidar_filepath = path.join(self.lidar_data_dir, "lidar.laz")
        self.ref_img_filepath = path.join(self.lidar_data_dir, "ref-img.tif")
        self.lidar_tree_values = [4, 5]
        self.tmp_dir = path.join(self.lidar_data_dir, "tmp")
        _create_tmp_dir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_lidar_to_canopy(self):
        ltc = dtr.LidarToCanopy()

        # test that `to_canopy_mask` returns a ndarray
        self.assertIsInstance(
            ltc.to_canopy_mask(
                self.lidar_filepath,
                self.lidar_tree_values,
                self.ref_img_filepath,
            ),
            np.ndarray,
        )

        # test that we can pass a `postprocess_func` with args/kwargs to
        # `to_canopy_mask`
        y_pred = ltc.to_canopy_mask(
            self.lidar_filepath,
            self.lidar_tree_values,
            self.ref_img_filepath,
            postprocess_func=ndi.binary_dilation,
            postprocess_func_args=[ndi.generate_binary_structure(2, 2)],
            postprocess_func_kwargs={"border_value": 0},
        )
        # test that `to_canopy_mask` with `output_filepath` returns a ndarray and dumps
        # it
        output_filepath = path.join(self.tmp_dir, "foo.tif")
        y_pred = ltc.to_canopy_mask(
            self.lidar_filepath,
            self.lidar_tree_values,
            self.ref_img_filepath,
            output_filepath=output_filepath,
        )
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertTrue(os.path.exists(output_filepath))


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.data_dir = "tests/data"
        self.img_dir = path.join(self.data_dir, "img")
        self.img_filepath = path.join(self.data_dir, "big_img/1091-322_00.tif")
        self.tmp_tiles_dir = path.join(self.data_dir, "tiles")
        _create_tmp_dir(self.tmp_tiles_dir)
        self.split_i_df = pd.read_csv(
            path.join(self.data_dir, "split_cluster-I.csv"), index_col=0
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_tiles_dir)

    def test_split_into_tiles(self):
        tiles = dtr.split_into_tiles(self.img_filepath, self.tmp_tiles_dir)
        full_tiles = dtr.split_into_tiles(
            self.img_filepath, self.tmp_tiles_dir, only_full_tiles=True
        )
        maybe_empty_tiles = dtr.split_into_tiles(
            self.img_filepath, self.tmp_tiles_dir, keep_empty_tiles=True
        )
        self.assertTrue(len(full_tiles) <= len(tiles))
        self.assertTrue(len(tiles) <= len(maybe_empty_tiles))

    def test_get_img_filename_ser(self):
        self.assertRaises(
            ValueError, utils.get_img_filename_ser, self.split_i_df, 0, True
        )

    def test_logging(self):
        # Taken from osmnx
        # https://github.com/gboeing/osmnx/blob/master/tests/test_osmnx.py
        import logging as lg

        utils.log("test a fake debug", level=lg.DEBUG)
        utils.log("test a fake info", level=lg.INFO)
        utils.log("test a fake warning", level=lg.WARNING)
        utils.log("test a fake error", level=lg.ERROR)


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.data_dir = "tests/data"
        self.img_dir = path.join(self.data_dir, "img")
        self.models_dir = path.join(self.data_dir, "models")
        self.model_filepath = path.join(self.models_dir, "clf.skops")
        self.response_img_dir = path.join(self.data_dir, "response_img")

        self.split_ii_filepath = path.join(self.data_dir, "split_cluster-II.csv")

        self.tmp_dir = path.join(self.data_dir, "tmp")
        _create_tmp_dir(self.tmp_dir)

        self.runner = testing.CliRunner()

        # TODO: test more possibilities of `args` in `invoke`
        # TODO: test `_dict_from_kws`

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_version(self):
        # Taken from rasterio
        # https://github.com/mapbox/rasterio/blob/master/tests/test_cli_main.py
        result = self.runner.invoke(main.cli, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(dtr.__version__, result.output)

    def test_train_test_split(self):
        result = self.runner.invoke(
            main.cli,
            [
                "train-test-split",
                "--img-dir",
                self.img_dir,
                "--output-filepath",
                path.join(self.tmp_dir, "split.csv"),
            ],
        )
        self.assertEqual(result.exit_code, 0)

    def test_train_classifier(self):
        result = self.runner.invoke(
            main.cli,
            [
                "train-classifier",
                "--split-filepath",
                path.join(self.data_dir, "split_cluster-I.csv"),
                "--img-dir",
                self.img_dir,
                "--response-img-dir",
                self.response_img_dir,
                "--output-filepath",
                path.join(self.tmp_dir, "clf.skops"),
            ],
        )
        self.assertEqual(result.exit_code, 0)

    def test_train_classifiers(self):
        result = self.runner.invoke(
            main.cli,
            [
                "train-classifiers",
                self.split_ii_filepath,
                self.img_dir,
                self.response_img_dir,
                "--output-dir",
                self.tmp_dir,
            ],
        )
        self.assertEqual(result.exit_code, 0)

    def test_predict_img(self):
        base_args = [
            "predict-img",
            glob.glob(path.join(self.img_dir, "*.tif"))[0],
            "--output-filepath",
            path.join(self.tmp_dir, "foo.tif"),
        ]
        for extra_args in [[], ["--clf-filepath", self.model_filepath]]:
            result = self.runner.invoke(main.cli, base_args + extra_args)
            self.assertEqual(result.exit_code, 0)

    def test_predict_imgs(self):
        base_args = ["predict-imgs", self.split_ii_filepath, self.img_dir]
        final_args = [
            "--output-dir",
            self.tmp_dir,
        ]
        for args in [
            [],
            ["--clf-filepath", self.model_filepath],
            ["--clf-dir", self.models_dir],
        ]:
            result = self.runner.invoke(main.cli, base_args + args + final_args)
            self.assertEqual(result.exit_code, 0)
