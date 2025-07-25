===============
Advanced Topics
===============

Most use cases of DetecTree only make use of the `TrainingSelector`, `ClassifierTrainer` and `Classifier` classes and their respective methods. Nevertheless,
See the `"background" example notebook <https://github.com/martibosch/detectree-example/blob/master/notebooks/background.ipynb>`_ and the article of Yang et al. :cite:`yang2009tree` for more information.

----------------
Train/test split
----------------

In order to enhance the robustness of the classifier, it is important that the subset of pixels selected as training samples are representative of the whole dataset. Given the large variety of scenes that can be found in such a dataset of urban aerial imagery (e.g., lakes, buildings, parks, forests...), a random selection of training tiles might not be representative of such variety and therefore lead to a classifier with low overall accuracy.

To overcome such problem, Yang et al. :cite:`yang2009tree` proposed a procedure of selecting training samples that intends to find the set of tiles that is most representative of the dataset. The scene structure of an image can be represented by a Gist descriptor :cite:`oliva2001modeling`, a low dimensional vector encoding which captures the high-level semantics of real-world aerial images. Following the approach of Yang et al. :cite:`yang2009tree`, the image descriptor is computed by:

* convolving it with Gabor filters on 3 frequencies and 4, 8 and orientations respectively, which accounts for 320 components
* computing a 8x8x8 joint color histogram in the Lab color space, which accounts for 512 components the two components are normalized to unit L-1 norm separately and then concatenated to form a 832-component image descriptor.

Nevertheless, the way in which such image descriptor is computer can be customized by means of the arguments of `TrainingSelector.__init__`. Such arguments will then be forwarded to the following function in order to compute the GIST descriptor of the input image:

.. autofunction:: detectree.image_descriptor.compute_image_descriptor_from_filepath

The GIST descriptor might also be directly computed from an array with the RGB representation of the image:

.. autofunction:: detectree.image_descriptor.compute_image_descriptor

On the other hand, in order to obtain a Gabor filter bank (e.g., for the `kernels` argument), the following function can be used:

.. autofunction:: detectree.filters.get_gabor_filter_bank

--------------------
Pixel classification
--------------------

In order to perform a binary pixel-level classification of tree/non-tree pixels, each pixel is transformed into a feature vector. In DetecTree, the way in which feature vectors are computed can be customized by means of the arguments of  `Classifier.__init__`.  With the default argument values, which follow the methods of Yang et al. :cite:`yang2009tree`, each pixel is transformed into a 27-feature vector where 6, 18 and 3 features capture characteristics of color, texture and entropy respectively. Such arguments are forwarded to the following class:

.. autoclass:: detectree.pixel_features.PixelFeaturesBuilder
   :members:  __init__, build_features

The texture features are obtained by convolving the images with a filter bank, which is obtained by means of the following function:

.. autofunction:: detectree.filters.get_texture_kernel

The arguments of  `Classifier.__init__` also serve to customize how the pixel response (i.e., tree/non-tree labels of each pixel) is computed, by forwarding them to the following class:

.. autoclass:: detectree.pixel_response.PixelResponseBuilder
   :members:  __init__, build_response

Optionally, the predicted pixel labels can be refined with the graph-cut max-flow procedure from Boykov and Komogrov :cite:`boykov2004experimental` (used by default when refinement is enabled in `Classifier`), implemented as:

.. autofunction:: detectree.refine.maxflow_refine

----------
Evaluation
----------

You can evaluate the pixel classification using the `compute_eval_metrics` function, whose `metrics` argument accepts either metric names from `sklearn.metrics` (strings) or callables that accept `y_true` and `y_pred`:

.. autofunction:: detectree.evaluate.compute_eval_metrics

Additionally, you can use the `metrics_kwargs` argument to provide per-metric options (as a list of keyword arguments passed to each matching item of `metrics`). The function will return the values of the metrics computed for the validation images.

It is also possible to compare the performance of refinement parameters with the `eval_refine_params` function, which given a `refine_method` (by default the graph-cut max-flow procedure of `detectree.maxflow_refine` as defined in the `settings.CLF_REFINE_METHOD`) will compute the metrics for a list of parameter dicts (`refine_params_list`):

.. autofunction:: detectree.evaluate.eval_refine_params

The function returns a DataFrame with metrics as rows and parameter sets as columns.
