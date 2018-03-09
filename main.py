import numpy as np
import os
import h5py
import tensorflow as tf

from lib.defaultFlags import defaultFlags
from lib.model import TEResNet, TEGAN
from lib.ops import print_configuration_op
from lib.readTFRecord import getTFRecordFilenamesIn 

FLAGS = defaultFlags()

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

# Check the summary_dir is given
if FLAGS.summary_dir is None:
    raise ValueError('The summary directory is needed')

# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

filenames_HR = getTFRecordFilenamesIn(FLAGS.input_dir_HR)
print("Using files: ", filenames_HR)
if len(filenames_HR) % FLAGS.batch_size != 0:
    raise RuntimeError("Tread with CAUTION! Training dataset size is not a multiple of batch_size")

net = TEGAN(filenames_HR, FLAGS)

with tf.Session() as sess:
    net.initialize(sess)
    print("Finished initializing :D")

    if FLAGS.mode == 'train':
        net.optimize(sess)

    elif FLAGS.mode == 'test':
        HR, LR, HR_out, loss = net.evaluate(sess)

        print("Saving test data to {}".format(os.path.join(FLAGS.output_dir, 'test_data.h5')))
        h5f = h5py.File(os.path.join(FLAGS.output_dir, 'test_data.h5'), 'w')
        h5f.create_dataset('HR', data=HR)
        h5f.create_dataset('LR', data=LR)
        h5f.create_dataset('output', data=HR_out)
        h5f.close()

        print("Test loss: ", loss)
