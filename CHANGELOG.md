# Change log

## 0.4.0 (03/07/2021)

* added pre-commit
* updated to laspy 2.0.0 (with optional laszip)
* updated bumpversion to double quotes (Python black)
* updated docs build
* tests and release to pypi and github with github actions
* added github issue templates, pull request template and updated contributing docs
* using pydocstyle and black
* added lidar to canopy module
* using keyword-only arguments

## 0.3.1 (11/03/2020)

* drop `num_blocks` argument of `compute_image_descriptor` and `compute_image_descriptor_from_filepath`

## 0.3.0 (02/03/2020)

* set default post-classification refinement parameter `refine_beta` to 50 (instead of 100)
* keyword arguments to `PixelFeaturesBuilder` and `PixelResponseBuilder` can be explicitly provided to the initialization of `ClassifierTrainer`, and are documented there
* raise a `ValueError` when a provided response is not a binary tree/non-tree image

## 0.2.0 (11/12/2019)

* correction (typo) `keep_emtpy_tiles` -> `keep_empty_tiles` in `split_into_tiles`

## 0.1.0 (14/11/2019)

* initial release
