import numpy as np
import os
import h5py
import tensorflow as tf

from lib.defaultFlags import defaultFlags
from lib.TEResNet import TEResNet
from lib.TEGAN import TEGAN
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

filenames_train = getTFRecordFilenamesIn(FLAGS.train_dir)
if len(filenames_train) % FLAGS.batch_size != 0:
    raise RuntimeError("Tread with CAUTION! Training dataset size is not a multiple of batch_size")

filenames_dev = getTFRecordFilenamesIn(FLAGS.dev_dir)
if len(filenames_dev) % FLAGS.batch_size != 0:
    raise RuntimeError("Tread with CAUTION! Training dataset size is not a multiple of batch_size")

if FLAGS.task == 'TEResNet':
    net = TEResNet(filenames_train, filenames_dev, FLAGS)
elif FLAGS.task == 'TEGAN':
    net = TEGAN(filenames_train, filenames_dev, FLAGS)
else:
    raise ValueError('Need to specify FLAGS.task to be TEResNet or TEGAN')

with tf.Session() as sess:
    print("--------------- {} mode -----------------".format(FLAGS.mode))
    net.initialize(sess)    
    print("Finished initializing :D")

    if FLAGS.mode == 'train':
        net.optimize(sess)

    elif FLAGS.mode == 'test':
        filenames_test = getTFRecordFilenamesIn(FLAGS.test_dir)
        output = net.evaluate(sess, [ filenames_test[0] ])
        losses = net.evaluate_losses(sess, filenames_test)

        filename_out = os.path.basename(filenames_test[0]).replace('.tfrecord','.h5')
        print("Saving test data to {}".format(os.path.join(FLAGS.output_dir, filename_out)))
        h5f = h5py.File(os.path.join(FLAGS.output_dir, filename_out), 'w')
        h5f.create_dataset('HR', data=output[0][0])
        h5f.create_dataset('LR', data=output[0][1])
        h5f.create_dataset('output', data=output[0][2])
        h5f.close()

        for loss, val in losses.items():
            print("{0:26s} = {1:0.6f}".format(loss,np.mean(val)))
    print("--------------- ******** -----------------")
