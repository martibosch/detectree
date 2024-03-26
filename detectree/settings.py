"""detectree general settings."""

import logging as lg

import lightgbm as lgb
import numpy as np

# train/test split
GIST_GABOR_FREQUENCIES = (0.1, 0.25, 0.4)
GIST_GABOR_NUM_ORIENTATIONS = (4, 8, 8)
GIST_RESPONSE_BINS_PER_AXIS = 4
GIST_NUM_COLOR_BINS = 8

# pixel features
GAUSS_SIGMAS = [1, np.sqrt(2), 2]
GAUSS_NUM_ORIENTATIONS = 6
ENTROPY_MIN_NEIGHBORHOOD_RANGE = 2
ENTROPY_NUM_NEIGHBORHOODS = 3

# build response
RESPONSE_TREE_VAL = 255
RESPONSE_NONTREE_VAL = 0

# classifier
CLF_CLASS = lgb.LGBMClassifier
CLF_KWARGS = {"n_estimators": 200}
CLF_TREE_VAL = 255
CLF_NONTREE_VAL = 0
CLF_REFINE = True
CLF_REFINE_BETA = 50
CLF_REFINE_INT_RESCALE = 10000
SKOPS_TRUSTED = [
    "collections.defaultdict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]
HF_HUB_REPO_ID = "martibosch/detectree"
HF_HUB_FILENAME = "clf.skops"

# LIDAR
LIDAR_TREE_THRESHOLD = 15
LIDAR_OUTPUT_DTYPE = "uint8"
LIDAR_OUTPUT_TREE_VAL = 255
LIDAR_OUTPUT_NODATA = 0

# utils
TILE_WIDTH = 512
TILE_HEIGHT = 512
TILE_OUTPUT_FILENAME = "tile_{}-{}.tif"
IMG_FILENAME_PATTERN = "*.tif"

# logging (from https://github.com/gboeing/osmnx/blob/master/osmnx/settings.py)
logs_folder = "logs"
# write log to file and/or to console
log_file = False
log_console = True
log_level = lg.INFO
log_name = "detectree"
log_filename = "detectree"
