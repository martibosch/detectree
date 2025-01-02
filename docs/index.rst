DetecTree documentation
=======================

DetecTree is a Pythonic library to classify tree/non-tree pixels from aerial imagery, following the methods of Yang et al. :cite:`yang2009tree`.

.. toctree::
   :maxdepth: 1
   :caption: Reference Guide:

   train_test_split
   pixel_classification
   advanced_topics

   utils

.. toctree::
   :maxdepth: 1
   :caption: Command-line interface (CLI):

   cli

.. toctree::
   :maxdepth: 1
   :caption: Development:

   changelog
   contributing

This documentation is intended as an API reference. A full example application of DetecTree to predict a tree canopy map for the Aussersihl district in Zurich `is available as a Jupyter notebook <https://github.com/martibosch/detectree-example/blob/master/notebooks/aussersihl-canopy.ipynb>`_. See the `detectree-example <https://github.com/martibosch/detectree-example>`_ repository for more information on the background and some example notebooks.

Usage
-----

To install use pip:

.. code-block:: bash

    $ pip install detectree

References
----------

.. bibliography::
    :all:
    :style: plain
