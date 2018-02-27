# Definition of the generator
def generator(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise  ValueError('No FLAGS is provided for generator')

    # The Bx residual blocks
    def residual_block(inputs, output_channels, stride, scope):
        with tf.variable_scope(scope):
            net = conv3d(inputs, 3, output_channels, stride, use_bias=False, scope='conv_1')
            net = batchnorm(net, FLAGS.is_training)
            net = prelu_tf(net)
            net = conv3d(net, 3, output_channels, stride, use_bias=False, scope='conv_2')
            net = batchnorm(net, FLAGS.is_training)
            net = net + inputs
        return net

    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv3d(gen_inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, FLAGS.num_resblock+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv3d(net, 3, 64, 1, use_bias=False, scope='conv')
            net = batchnorm(net, FLAGS.is_training)

        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            net = conv3d(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = conv3d(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = conv3d(net, 9, gen_output_channels, 1, scope='conv')

    return net


class TEResNet(object):

    def __init__(self, inputs, targets, FLAGS):

        # Build the generator part
        with tf.variable_scope('generator'):
            self.output_channels = targets.get_shape().as_list()[-1]
            self.gen_output = generator(inputs, self.output_channels, reuse=False, FLAGS=FLAGS)
            # self.gen_output.set_shape([FLAGS.batch_size, FLAGS.input_size * 4, FLAGS.input_size * 4, FLAGS.input_size * 4, 4])


        # Calculating the generator loss
        with tf.variable_scope('generator_loss'):

            # Content loss
            with tf.variable_scope('content_loss'):
                # Compute the euclidean distance between the two features
                self.content_loss = tf.reduce_mean( tf.square(self.gen_output - targets) )

            self.gen_loss = self.content_loss

        # Define the learning rate and global step
        with tf.variable_scope('get_learning_rate_and_global_step'):

            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                                       staircase=FLAGS.stair)
            self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

        with tf.variable_scope('generator_train'):

            # Need to wait discriminator to perform train step
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                gen_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=FLAGS.beta)
                self.gen_train = gen_optimizer.minimize( self.gen_loss )

        exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
        self.update_loss = exp_averager.apply([self.content_loss])


    def optimize(self, session):

        results = session.run( (self.update_loss, self.incr_global_step, self.gen_train) )




