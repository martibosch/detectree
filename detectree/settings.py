import logging as lg

import numpy as np

# train/test split
GIST_DEFAULT_GABOR_FREQUENCIES = (.1, .25, .4)
GIST_DEFAULT_GABOR_NUM_ORIENTATIONS = (4, 8, 8)
GIST_DEFAULT_RESPONSE_BINS_PER_AXIS = 4
GIST_DEFAULT_NUM_COLOR_BINS = 8

# pixel features
GAUSS_DEFAULT_SIGMAS = [1, np.sqrt(2), 2]
GAUSS_DEFAULT_NUM_ORIENTATIONS = 6
ENTROPY_DEFAULT_MIN_NEIGHBORHOOD_RANGE = 2
ENTROPY_DEFAULT_NUM_NEIGHBORHOODS = 3

# build response
RESPONSE_DEFAULT_TREE_VAL = 255
RESPONSE_DEFAULT_NONTREE_VAL = 0

# classifier
CLF_DEFAULT_NUM_ESTIMATORS = 200
CLF_DEFAULT_TREE_VAL = 255
CLF_DEFAULT_NONTREE_VAL = 0
CLF_DEFAULT_REFINE = True
CLF_DEFAULT_REFINE_BETA = 100
CLF_DEFAULT_REFINE_INT_RESCALE = 10000

# utils
TILE_DEFAULT_WIDTH = 512
TILE_DEFAULT_HEIGHT = 512
TILE_DEFAULT_OUTPUT_FILENAME = 'tile_{}-{}.tif'
IMG_DEFAULT_FILENAME_PATTERN = '*.tif'

# logging (from https://github.com/gboeing/osmnx/blob/master/osmnx/settings.py)
logs_folder = 'logs'
# write log to file and/or to console
log_file = False
log_console = True
log_level = lg.INFO
log_name = 'detectree'
log_filename = 'detectree'
