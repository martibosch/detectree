[![PyPI version fury.io](https://badge.fury.io/py/detectree.svg)](https://pypi.python.org/pypi/detectree/)
[![Documentation Status](https://readthedocs.org/projects/detectree/badge/?version=latest)](https://detectree.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/martibosch/detectree.svg?branch=master)](https://travis-ci.org/martibosch/detectree)
[![Coverage Status](https://coveralls.io/repos/github/martibosch/detectree/badge.svg?branch=master)](https://coveralls.io/github/martibosch/detectree?branch=master)
[![GitHub license](https://img.shields.io/github/license/martibosch/detectree.svg)](https://github.com/martibosch/detectree/blob/master/LICENSE)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02172/status.svg)](https://doi.org/10.21105/joss.02172)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3908338.svg)](https://doi.org/10.5281/zenodo.3908338)

# DetecTree

## Overview

DetecTree is a Pythonic library to classify tree/non-tree pixels from aerial imagery, following the methods of Yang et al. [1]. The target audience is researchers and practitioners in GIS that are interested in two-dimensional aspects of trees, such as their proportional abundance and spatial distribution throughout a region of study. These measurements can be used to assess important aspects of urban planning such as the provision of urban ecosystem services. The approach is of special relevance when LIDAR data is not available or it is too costly in monetary or computational terms.

```python
import detectree as dtr
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio import plot

# select the training tiles from the tiled aerial imagery dataset
ts = dtr.TrainingSelector(img_dir='data/tiles')
split_df = ts.train_test_split(method='cluster-I')

# train a tree/non-tree pixel classfier
clf = dtr.ClassifierTrainer().train_classifier(
    split_df=split_df, response_img_dir='data/response_tiles')
    
# use the trained classifier to predict the tree/non-tree pixels
test_filepath = split_df[~split_df['train'].sample(1).iloc[0]['img_filepath']
y_pred = dtr.Classifier().classify_img(test_filepath, clf)

# side-by-side plot of the tile and the predicted tree/non-tree pixels
figwidth, figheight = plt.rcParams['figure.figsize']
fig, axes = plt.subplots(1, 2, figsize=(2 * figwidth, figheight))

with rio.open(img_filepath) as src:
    plot.show(src.read(), ax=axes[0])
axes[1].imshow(y_pred)
```

![Example](figures/example.png)

A full example application of DetecTree to predict a tree canopy map for the Aussersihl district in Zurich [is available as a Jupyter notebook](https://github.com/martibosch/detectree-example/blob/master/notebooks/aussersihl-canopy.ipynb). See also [the API reference documentation](https://detectree.readthedocs.io/en/latest/?badge=latest) and the [example repository](https://github.com/martibosch/detectree-example) for more information on the background and some example notebooks.

## Citation

Bosch M. 2020. “DetecTree: Tree detection from aerial imagery in Python”. *Journal of Open Source Software, 5(50), 2172.* [doi.org/10.21105/joss.02172](https://doi.org/10.21105/joss.02172)

Note that DetecTree is based on the methods of Yang et al. [1], therefore it seems fair to reference their work too. An example citation in an academic paper might read as follows:

> The classification of tree pixels has been performed with the Python library DetecTree (Bosch, 2020), which is based on the approach of Yang et al. (2009).

## Installation

To install use pip:

    $ pip install detectree

## See also

* [lausanne-tree-canopy](https://github.com/martibosch/lausanne-tree-canopy): example computational workflow to get the tree canopy of Lausanne with DetecTree
* [A video of a talk about DetecTree](https://www.youtube.com/watch?v=USwF2KyxVjY) in the [Applied Machine Learning Days of EPFL (2020)](https://appliedmldays.org/) and [its respective slides](https://martibosch.github.io/detectree-amld-2020)

## Acknowledgments

* With the support of the École Polytechnique Fédérale de Lausanne (EPFL)


## References

1. Yang, L., Wu, X., Praun, E., & Ma, X. (2009). Tree detection from aerial imagery. In Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (pp. 131-137). ACM.
