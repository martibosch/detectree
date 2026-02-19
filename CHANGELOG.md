# Change log

## [v0.9.1] - 2026-02-19

### :bug: Bug Fixes

- [`51e4c19`](https://github.com/martibosch/detectree/commit/51e4c19155245777a36a93d14d63851b819c1b14) - update license refs in pyproject.toml *(commit by [@martibosch](https://github.com/martibosch))*

## [v0.9.0] - 2026-02-18

### :boom: BREAKING CHANGES

- due to [`f44847b`](https://github.com/martibosch/detectree/commit/f44847bf59a4a34d85b290187195f9dbe3e42fd1) - predict all split_df images (training too) in predict_imgs *(commit by [@martibosch](https://github.com/martibosch))*:

  predict all split_df images (training too) in predict_imgs

### :sparkles: New Features

- [`36cbeac`](https://github.com/martibosch/detectree/commit/36cbeacb906fdfdb893efb2984cccaea02dc0cae) - separate refine module, eval module with eval refine params *(commit by [@martibosch](https://github.com/martibosch))*
- [`f9230b6`](https://github.com/martibosch/detectree/commit/f9230b695ef38fcd912e589f36bbd1caf0ab7bf4) - CLI keyword arguments using str-encoded dicts *(commit by [@martibosch](https://github.com/martibosch))*
- [`a2cec04`](https://github.com/martibosch/detectree/commit/a2cec04f63678ad511d1b9ecf9df0d5107f48437) - separate PixelDatasetTransformer class *(commit by [@martibosch](https://github.com/martibosch))*
- [`71dcff3`](https://github.com/martibosch/detectree/commit/71dcff3d89efab78e27d75d5432965d0f2793d96) - random seed *(commit by [@martibosch](https://github.com/martibosch))*
- [`ce10eb9`](https://github.com/martibosch/detectree/commit/ce10eb94f8e9ffc4bbe7558904bc745c1aa57a8c) - suppress lgbm feature name warning *(commit by [@martibosch](https://github.com/martibosch))*
- [`f44847b`](https://github.com/martibosch/detectree/commit/f44847bf59a4a34d85b290187195f9dbe3e42fd1) - predict all split_df images (training too) in predict_imgs *(commit by [@martibosch](https://github.com/martibosch))*

## [v0.8.1] - 2025-07-27

### :bug: Bug Fixes

- [`33c8b55`](https://github.com/martibosch/detectree/commit/33c8b555c8c670a1e7a0ab9d47aca76294d83c76) - min python to 3.10 in pyproject.toml *(commit by [@martibosch](https://github.com/martibosch))*

## [v0.8.0] - 2025-07-25

### :sparkles: New Features

- [`0c1a0d9`](https://github.com/martibosch/detectree/commit/0c1a0d912f7f137101ee400610b6b92fe9c5eebf) - changed `TREE_VAL` to 1 for direct sklearn metrics compat *(commit by [@martibosch](https://github.com/martibosch))*
- [`bd72cf8`](https://github.com/martibosch/detectree/commit/bd72cf814286dc78f80df4b097a8a3171797a81e) - accept img_dir/img_filepaths in Classifier.predict_imgs *(commit by [@martibosch](https://github.com/martibosch))*
- [`9ba3a2f`](https://github.com/martibosch/detectree/commit/9ba3a2f6381e958ddbce2fc059d2ec49597b414f) - do NOT force geotiff driver in output *(commit by [@martibosch](https://github.com/martibosch))*

### :bug: Bug Fixes

- [`0594040`](https://github.com/martibosch/detectree/commit/059404079f016ceef0bab267e3dd9ef168255afb) - flake8 fixes *(commit by [@martibosch](https://github.com/martibosch))*

### :recycle: Refactors

- [`578722a`](https://github.com/martibosch/detectree/commit/578722a4be706e3a604a7a8f4afcaa01fb589480) - unify TREE_VAL/NONTREE_VAL settings and merge docstrings *(commit by [@martibosch](https://github.com/martibosch))*
- [`713f864`](https://github.com/martibosch/detectree/commit/713f864259dde59244374d67cc3acffc54d18f42) - DRY get_img_filepaths in utils *(commit by [@martibosch](https://github.com/martibosch))*

## [v0.7.0] - 2025-07-24

### :sparkles: New Features

- [`d02c405`](https://github.com/martibosch/detectree/commit/d02c4057b8325a0b6379fc10735a7a137d82a535) - python 3.13 support *(commit by [@martibosch](https://github.com/martibosch))*
- [`0fa88d5`](https://github.com/martibosch/detectree/commit/0fa88d5b358ff80e702db679b05f180ccf013848) - run any skops model from hf hub *(commit by [@martibosch](https://github.com/martibosch))*

### :bug: Bug Fixes

- [`e76a043`](https://github.com/martibosch/detectree/commit/e76a043f59c682e520c99950b815d29793a848c1) - add collections.OrderedDict to skops trusted object types *(commit by [@martibosch](https://github.com/martibosch))*

### :recycle: Refactors

- [`92c065c`](https://github.com/martibosch/detectree/commit/92c065c3738ed8d0a6a6ca36dfbd06c515c3fcf6) - change square->footprint_rectangle in skimage.morphology *(commit by [@martibosch](https://github.com/martibosch))*
- [`6e8f250`](https://github.com/martibosch/detectree/commit/6e8f25054666dfbccf263af73b3b38900443c664) - use "columns" instead of 0 in pandas axis arg *(commit by [@martibosch](https://github.com/martibosch))*
- [`bf2399d`](https://github.com/martibosch/detectree/commit/bf2399da97c456b8b1efdc3f968bb9ce72cb02eb) - use detectree instead of . in relative imports *(commit by [@martibosch](https://github.com/martibosch))*

## [v0.6.0] - 2025-01-02

### :boom: BREAKING CHANGES

- due to [`24fe2d4`](https://github.com/martibosch/detectree/commit/24fe2d4b6c0ae51757d6a2f26b1aa113ebece84e) - split_df with file names instead of paths, required img_dir *(commit by [@martibosch](https://github.com/martibosch))*:

  split_df with file names instead of paths, required img_dir

### :sparkles: New Features

- [`24fe2d4`](https://github.com/martibosch/detectree/commit/24fe2d4b6c0ae51757d6a2f26b1aa113ebece84e) - split_df with file names instead of paths, required img_dir *(commit by [@martibosch](https://github.com/martibosch))*

### :white_check_mark: Tests

- [`9657e62`](https://github.com/martibosch/detectree/commit/9657e626b76d2509367d1b31368fcaaed1aa502b) - conda install lightgbm to avoid macos issues; use tox-gh *(commit by [@martibosch](https://github.com/martibosch))*
- [`0941763`](https://github.com/martibosch/detectree/commit/0941763c40ae5064fd503f97d20836d24046d4bd) - revert to tox.ini (see github.com/tox-dev/tox/issues/3457) *(commit by [@martibosch](https://github.com/martibosch))*

## [v0.5.0] - 2024-03-28

### :sparkles: New Features

- [`6bdbdc3`](https://github.com/martibosch/detectree/commit/6bdbdc3c348922488ec5cea06cf271d41028fbc8) - rasterize_lidar with shape/transform args and return zeros *(commit by [@martibosch](https://github.com/martibosch))*
- [`c619239`](https://github.com/martibosch/detectree/commit/c619239c3e7c3fd3bb01cd6c49763e887491d0f6) - adaboost->generic classifier (default lgb), joblib->skops *(commit by [@martibosch](https://github.com/martibosch))*
- [`853d9ad`](https://github.com/martibosch/detectree/commit/853d9ad2f22886fbb20fd334de7ab84f390e4c6e) - fix to enforce all keyword-only args *(commit by [@martibosch](https://github.com/martibosch))*
- [`0729a3d`](https://github.com/martibosch/detectree/commit/0729a3dcbd4be964311b8c29ff5d116a2385fc9f) - accept pca/kmeans kwargs in train/test split *(commit by [@martibosch](https://github.com/martibosch))*
- [`a3b1362`](https://github.com/martibosch/detectree/commit/a3b1362c237097c2dd572b72a86c8b8706680fe4) - update to new ndi namespace for rotate *(commit by [@martibosch](https://github.com/martibosch))*
- [`4f50349`](https://github.com/martibosch/detectree/commit/4f50349a6647abb802d61e63c2416260c8b2c631) - pre-trained model from huggingface hub *(commit by [@martibosch](https://github.com/martibosch))*

### :bug: Bug Fixes

- [`bbb7c19`](https://github.com/martibosch/detectree/commit/bbb7c197bbdf13897f49ddeb07089a470571b59d) - l2c meta height/width instead of shape, lazrs lidar backend *(commit by [@martibosch](https://github.com/martibosch))*

### :recycle: Refactors

- [`694966a`](https://github.com/martibosch/detectree/commit/694966ad7541de2a2d9708774255c8c1a05ccf08) - using opencv for faster convolution *(commit by [@martibosch](https://github.com/martibosch))*
- [`5c00c95`](https://github.com/martibosch/detectree/commit/5c00c954134a9ce2484bb8fa6f09d0da2f059c14) - dropped `DEFAULT` from settings, consistent docs *(commit by [@martibosch](https://github.com/martibosch))*
- [`0c9da36`](https://github.com/martibosch/detectree/commit/0c9da361f89452c7d34c6a3763fb5063cf9f469b) - kws->kwargs *(commit by [@martibosch](https://github.com/martibosch))*
- [`73ab0c4`](https://github.com/martibosch/detectree/commit/73ab0c4492f097755ecadb002b179553176a6368) - change `classify_img`->`predict_img` *(commit by [@martibosch](https://github.com/martibosch))*

### :white_check_mark: Tests

- [`9696124`](https://github.com/martibosch/detectree/commit/96961244679d1ebca5e5414d8678576f7e73bb03) - \_create_tmp_dir to rm test tmp dirs if existing *(commit by [@martibosch](https://github.com/martibosch))*

## 0.4.2 (24/10/2022)

- moved `_gaussain_kernel1d` from scipy to detectree codebase

## 0.4.1 (05/07/2021)

- added postprocess func args and kwargs to `LidarToCanopy`

## 0.4.0 (03/07/2021)

- added pre-commit
- updated to laspy 2.0.0 (with optional laszip)
- updated bumpversion to double quotes (Python black)
- updated docs build
- tests and release to pypi and github with github actions
- added github issue templates, pull request template and updated contributing docs
- using pydocstyle and black
- added lidar to canopy module
- using keyword-only arguments

## 0.3.1 (11/03/2020)

- drop `num_blocks` argument of `compute_image_descriptor` and `compute_image_descriptor_from_filepath`

## 0.3.0 (02/03/2020)

- set default post-classification refinement parameter `refine_beta` to 50 (instead of 100)
- keyword arguments to `PixelFeaturesBuilder` and `PixelResponseBuilder` can be explicitly provided to the initialization of `ClassifierTrainer`, and are documented there
- raise a `ValueError` when a provided response is not a binary tree/non-tree image

## 0.2.0 (11/12/2019)

- correction (typo) `keep_emtpy_tiles` -> `keep_empty_tiles` in `split_into_tiles`

## 0.1.0 (14/11/2019)

- initial release

[v0.5.0]: https://github.com/martibosch/detectree/compare/v0.4.2...v0.5.0
[v0.6.0]: https://github.com/martibosch/detectree/compare/v0.5.1...v0.6.0
[v0.7.0]: https://github.com/martibosch/detectree/compare/v0.6.0...v0.7.0
[v0.8.0]: https://github.com/martibosch/detectree/compare/v0.7.0...v0.8.0
[v0.8.1]: https://github.com/martibosch/detectree/compare/v0.8.0...v0.8.1
[v0.9.0]: https://github.com/martibosch/detectree/compare/v0.8.1...v0.9.0
[v0.9.1]: https://github.com/martibosch/detectree/compare/v0.9.0...v0.9.1
