===============
Advanced Topics
===============

Most use cases of DetecTree only make use of the `TrainingSelector`, `ClassifierTrainer` and `Classifier` classes and their respective methods. Nevertheless, 
See the `"background" example notebook <https://github.com/martibosch/detectree-example/blob/master/notebooks/background.ipynb>`_ and the article of Yang et al. :cite:`yang2009tree` for more information.

----------------
Train/test split
----------------

In order to enhance the robustness of the classifier, it is important that the subset of pixels selected as training samples are representative of the whole dataset. Given the large variety of scenes that can be found in such a datset of urban aerial imagery (e.g., lakes, buildings, parks, forests...), a random selection of training tiles might not be representative of such variety and therefore lead to a classifier with low overall accuracy.

To overcome such problem, Yang et al. :cite:`yang2009tree` proposed a procedure of selecting training samples that intends to find the set of tiles that is most representative of the dataset. The scene structure of an image can be represented by a Gist descriptor :cite:`oliva2001modeling`, a low dimensional vector encoding which captures the high-level semantics of real-world aerial images. Following the approach of Yang et al. :cite:`yang2009tree`, the image descriptor is computed by:

* convolving it with Gabor filters on 3 frequencies and 4, 8 and orientations respectively, which accounts for 320 components
* computing a 8x8x8 joint color histogram in the Lab color space, which accounts for 512 components the two components are normalized to unit L-1 norm separatedly and then concatenated to form a 832-component image descriptor.

Nevertheless, the way in which such image descriptor is computer can be customized by means of the arguments of `TrainingSelector.__init__`. Such arguments will then be forwarded to the following function in order to compute the GIST descriptor of the input image:

.. autofunction:: detectree.image_descriptor.compute_image_descriptor_from_filepath

The GIST descriptor might also be directly computed from an array with the RGB representation of the image:
                  
.. autofunction:: detectree.image_descriptor.compute_image_descriptor                  

On the other hand, in order to obtain a Gabor filter bank (e.g., for the `kernels` argument), the following function can be used:

.. autofunction:: detectree.filters.get_gabor_filter_bank

--------------------
Pixel classification
--------------------

In order to perform a binary pixel-level classification of tree/non-tree pixels, each pixel is transformed into a feature vector. In DetecTree, the way in which feature vectors are computed can be customized by means of the arguments of  `Classifier.__init__`.  With the default argument values, which follow the methods of Yang et al. [1], each pixel is transformed into a 27-feature vector where 6, 18 and 3 features capture characteristics of color, texture and entropy respectively. Such arguments are forwarded to the following class:

.. autoclass:: detectree.pixel_features.PixelFeaturesBuilder
   :members:  __init__, build_features

The texture features are obtained by convolving the images with a filter bank, which is obtained by means of the following function:
      
.. autofunction:: detectree.filters.get_texture_kernel

The arguments of  `Classifier.__init__` also serve to customize how the pixel response (i.e., tree/non-tree labels of each pixel) is computed, by forwarding them to the following class:

.. autoclass:: detectree.pixel_response.PixelResponseBuilder
   :members:  __init__, build_response
