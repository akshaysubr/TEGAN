import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from data_processing import *
from defaultFlags import *
from model.model import TEResNet
from model.ops import print_configuration_op


FLAGS = defaultFlags()

# Print the configuration of the model
print_configuration_op(FLAGS)

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

if FLAGS.mode == 'train':
    
