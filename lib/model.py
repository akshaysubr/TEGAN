import os
import tensorflow as tf

from lib.readTFRecord import parseTFRecordExample 
import lib.ops as ops

# Definition of the generator
def generator(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise  ValueError('No FLAGS is provided for generator')

    # The Bx residual blocks
    def residual_block(inputs, output_channels, stride, scope):
        with tf.variable_scope(scope):
            net = ops.conv3d(inputs, 3, output_channels, stride, use_bias=False, scope='conv_1')
            net = ops.batchnorm(net, FLAGS.is_training)
            net = ops.prelu_tf(net)
            net = ops.conv3d(net, 3, output_channels, stride, use_bias=False, scope='conv_2')
            net = ops.batchnorm(net, FLAGS.is_training)
            net = net + inputs
        return net

    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = ops.conv3d(gen_inputs, 9, 64, 1, scope='conv')
            net = ops.prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, FLAGS.num_resblock+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = ops.conv3d(net, 3, 64, 1, use_bias=False, scope='conv')
            net = ops.batchnorm(net, FLAGS.is_training)

        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            net = ops.conv3d(net, 3, 256, 1, scope='conv')
            net = ops.pixelShuffler(net, scale=2)
            net = ops.prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = ops.conv3d(net, 3, 256, 1, scope='conv')
            net = ops.pixelShuffler(net, scale=2)
            net = ops.prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = ops.conv3d(net, 9, gen_output_channels, 1, scope='conv')

    return net


class TEResNet(object):

    def __init__(self, filenames_HR, FLAGS):

        self.filenames_HR = filenames_HR

        self.dataset = tf.data.TFRecordDataset(self.filenames_HR)
        self.dataset = self.dataset.map(parseTFRecordExample)
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        if FLAGS.mode == 'train':
            self.dataset = self.dataset.batch(FLAGS.batch_size)
            self.dataset = self.dataset.repeat(FLAGS.max_epoch)
        else:
            self.dataset = self.dataset.batch(len(filenames_HR))
            self.dataset = self.dataset.repeat(1)

        self.iterator = self.dataset.make_one_shot_iterator()
    
        next_batch_HR = self.iterator.get_next()
        next_batch_LR = ops.filter3d(next_batch_HR)

        # TODO: Fix batch_size not being fator of total dataset size
        next_batch_LR.set_shape([FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, FLAGS.input_size, 4])

        self.FLAGS = FLAGS

        # Build the generator part
        with tf.variable_scope('generator'):
            self.output_channels = next_batch_HR.get_shape().as_list()[-1]
            self.gen_output = generator(next_batch_LR, self.output_channels, reuse=False, FLAGS=FLAGS)
            # self.gen_output.set_shape([FLAGS.batch_size, FLAGS.input_size * 4, FLAGS.input_size * 4, FLAGS.input_size * 4, 4])


        # Calculating the generator loss
        with tf.variable_scope('generator_loss'):

            # Content loss
            with tf.variable_scope('content_loss'):
                # Compute the euclidean distance between the two features
                self.content_loss = tf.reduce_mean( tf.square(self.gen_output - next_batch_HR) )

            self.gen_loss = self.content_loss

        # Define the learning rate and global step
        with tf.variable_scope('get_learning_rate_and_global_step'):

            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                                            staircase=FLAGS.stair)
            # self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

        with tf.variable_scope('generator_train'):

            # Need to wait discriminator to perform train step
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                gen_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=FLAGS.beta)
                self.gen_train = gen_optimizer.minimize( self.gen_loss, self.global_step )

        exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
        self.update_loss = exp_averager.apply([self.content_loss])

        # Define data saver
        self.saver = tf.train.Saver(max_to_keep=10)
        self.weights_initializer = tf.train.Saver(gen_tvars)


    def initialize(self, session):

        if self.FLAGS.checkpoint is None:
            session.run( tf.global_variables_initializer() )
        else:
            print("Restoring weights from {}".format(self.FLAGS.checkpoint))
            self.weights_initializer.restore(session, self.FLAGS.checkpoint)


    def optimize(self, session):

        if self.FLAGS.mode != 'train':
            raise RuntimeError("Cannot optimize if not in train mode!!!")

        results = session.run( (self.gen_loss, self.global_step, self.gen_train) )

        # Save after every save_freq iterations
        if (results[1] % 10) == 0:
            print("Saving weights to {}".format(os.path.join(self.FLAGS.output_dir, 'model')))
            self.saver.save(session, os.path.join(self.FLAGS.output_dir, 'model'), global_step=results[1])

        return results


    def evaluate(self, session):
        return session.run( (self.gen_output, self.gen_loss) )


