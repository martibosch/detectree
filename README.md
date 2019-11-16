[![PyPI version fury.io](https://badge.fury.io/py/detectree.svg)](https://pypi.python.org/pypi/detectree/)
[![Documentation Status](https://readthedocs.org/projects/detectree/badge/?version=latest)](https://detectree.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/martibosch/detectree.svg?branch=master)](https://travis-ci.org/martibosch/detectree)
[![Coverage Status](https://coveralls.io/repos/github/martibosch/detectree/badge.svg?branch=master)](https://coveralls.io/github/martibosch/detectree?branch=master)
[![GitHub license](https://img.shields.io/github/license/martibosch/detectree.svg)](https://github.com/martibosch/detectree/blob/master/LICENSE)

# DetecTree

## Overview

DetecTree is a Pythonic library to classify tree/non-tree pixels from aerial imagery, following the methods of Yang et al. [1].

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

See [the API documentation](https://detectree.readthedocs.io/en/latest/?badge=latest) and the [example repository](https://github.com/martibosch/detectree-example) to get started.

## Installation

To install use pip:

    $ pip install detectree


## Acknowledgments

* With the support of the École Polytechnique Fédérale de Lausanne (EPFL)


## References

1. Yang, L., Wu, X., Praun, E., & Ma, X. (2009). Tree detection from aerial imagery. In Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (pp. 131-137). ACM.
