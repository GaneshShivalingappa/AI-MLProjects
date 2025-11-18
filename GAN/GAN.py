import os
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf
import glob
import imageio
import numpy as np
from matplotlib import pyplot as plt
import time
from IPython import display
from tqdm import tqdm

layers = tf.keras.layers
