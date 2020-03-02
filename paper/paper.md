---
title: 'DetecTree: Tree detection from aerial imagery in Python'
tags:
  - Python
  - image classification 
  - GIS
authors:
 - name: Martí Bosch
   orcid: 0000-0001-8735-9144
affiliations:
 - name: Urban and Regional Planning Community, École Polytechnique Fédérale de Lausanne, Switzerland
date: 
bibliography: paper.bib
---

# Summary

Urban tree canopy datasets are important to a wide variety of social and environmental studies in cities. Surveys to collect such information manually are costly and hard to maintain, which has motivated a growing interest in automated approaches to tree detection in recent years. To that end, LIDAR point clouds are a very appropriate data source, especially given its capability to represent spatial features in three dimensions. However, collecting LIDAR data requires expensive equipment and raw datasets are rarely made openly available. On the other hand, aerial imagery is another major source of data that is able to capture a wide range of objects on Earth, including trees. Although aerial imagery depicts the spatial features in only two dimensions, its main advantage with respect to LIDAR is its greater availability.

The aim of DetecTree is therefore to provide an open source library that performs a binary classification of tree/non-tree pixels from aerial imagery. To that end, it follows the supervised learning approach of Yang et al. [@yang2009tree], which requires an RGB aerial imagery dataset as only input, and consists of the following steps:

* **Step 0**: split of the dataset into image tiles. Since aerial imagery datasets often already come as a mosaic of image tiles, this step might not be necessary. In any case, DetecTree provides a `split_into_tiles` function that can be used to divide a large image into a mosaic of tiles of a specified dimension.
* **Step 1**: selection of the tiles to be used for training a classifier. As a supervised learning task, the ground-truth maps must be provided for some subset of the dataset. Since this part is likely to involve manual, it is crucial that the training set has as least tiles as possible. At the same time, to enhance the classifier's ability to detect trees in the diverse scenes of the dataset, the training set should contain as many of the diverse geographic features as possible. Thus, in order to optimize the representativity of the training set, the training tiles are selected according to their GIST descriptor [@oliva2001modeling], i.e., a vector describing the key semantics fo the tile's scene. More precisely, *k*-means clustering is applied to the GIST descriptors of all the tiles, with the number of clusters *k* set to the number of tiles of the training set (by default, one percent of the tiles is used). Then, for each cluster, the tile whose GIST descriptor is closest to the cluster's centroid is added to the training set. In DetecTree, this is done by the `train_test_split` method of the `TrainingSelector` class.
* **Step 2**: provision of the ground truth tree/non-tree masks for the training tiles. For each tile of the training set, the ground-truth tree/non-tree masks must be provided to get the pixel-level responses that will be used to train the classifier. To that end, an image editing software such as GIMP or Adobe Photoshop might be used. Alternatively, if LIDAR data for the training tiles is available, it might also be exploied to create the ground truth masks.
* **Step 3**: train a binary pixel-level classifier. For each pixel of the training tiles, a vector of 27 features is computed, where 6, 18 and 3 features capture characteristics of color, texture and entropy respectively. A binary AdaBoost classifier [@freund1995desicion] is then trained by mapping the feature vector of each pixel to its class in the ground truth masks (i.e., tree or non-tree).
* **Step 4**: tree detection in the testing tiles. Given a trained classifier, the `classify_img` and `classify_imgs` methods of the `Classifier` class might respectively be used to classify the tree pixels of a single image tile or of multiple image tiles at scale. For each image tile, the pixel-level classification is refined by means of a graph cuts algorithm [@boykov2004experimental] to avoid sparse pixels classified as trees by enforcing consistency between adjacent tree pixels. An example of an image tile, its pre-refinement pixel-level classification and the final refined result is displayed below:

![Example of an image tile (left), its pre-refinement pixel-level classification (center) and the final refined result (right).](figure.png)

The code of DetecTree is organized following an object-oriented approach, and relies on NumPy [@van2011numpy] to represent most data structures and perform operations upon them in a vectorized manner. The Scikit-learn library [@pedregosa2011scikit] is used to implement the AdaBoost pixel-level classifier as well as to perform the *k*-means clustering to select the training tiles. The computation of pixel-level features and GIST descriptors make use of various features provided by the Scikit-image [@van2014scikit] and SciPy [@virtanen2020scipy] libraries. On the other hand, the classification refinement employs the graph cuts algorithm implementation provided by the library [PyMaxFlow](https://github.com/pmneila/PyMaxflow). Finally, when possible, DetecTree uses the Dask library [@rocklin2015dask] to perform various computations in parallel.


The target audience of DetecTree is researchers and practitioners in GIS that are interested in two-dimensional aspects of trees, such as their proportional abundance and spatial distribution throughout a region of study. The approach is of special relevance when LIDAR data is not available or it is too costly in monetary or computational terms. The features of DetecTree are implemented in a manner that enhances the flexibility of the library so that the user can integrate it into complex computational workflows, and also provide custom arguments for the technical aspects. Furthermore, the functionalities of DetecTree might be used through its Python API as well as through its command-line interface (CLI), which is implemented by means of the Click Python package.


# Availability

The source code of DetecTree is fully available at [a GitHub repository](https://github.com/martibosch/detectree). A dedicated Python package has been created and is hosted at the [Python Package Index (PyPI)](https://pypi.org/project/detectree/). The documentation site is hosted at [Read the Docs](https://detectree.readthedocs.io/), and an example repository with Jupyter notebooks of an example application to an openly-available orthophoto of Zurich is provided at a [dedicated GitHub repository](https://github.com/martibosch/detectree-example), which can be executed interactively online by means of the Binder web service [@jupyter2018binder].

Unit tests are run within the [Travis CI](https://travis-ci.org/martibosch/detectree) platform every time that new commits are pushed to the GitHub repository. Additionally, test coverage [is reported on Coveralls](https://coveralls.io/github/martibosch/detectree?branch=master).


# Acknowledgments

This research has been supported by the École Polytechnique Fédérale de Lausanne.


# References
