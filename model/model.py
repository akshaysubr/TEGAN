# Definition of the generator
def generator(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise  ValueError('No FLAGS is provided for generator')

    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv3d(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
            net = batchnorm(net, FLAGS.is_training)
            net = prelu_tf(net)
            net = conv3d(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
            net = batchnorm(net, FLAGS.is_training)
            net = net + inputs
        return net
