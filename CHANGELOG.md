# Change log

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
- [`9696124`](https://github.com/martibosch/detectree/commit/96961244679d1ebca5e5414d8678576f7e73bb03) - _create_tmp_dir to rm test tmp dirs if existing *(commit by [@martibosch](https://github.com/martibosch))*


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