import numpy as np
import tensorflow as tf

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
            if (FLAGS.GAN_type == 'GAN'):
                net = ops.batchnorm(net, FLAGS.is_training)
            net = ops.prelu_tf(net)
            net = ops.conv3d(net, 3, output_channels, stride, use_bias=False, scope='conv_2')
            if (FLAGS.GAN_type == 'GAN'):
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
            if (FLAGS.GAN_type == 'GAN'):
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



def discriminator(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for discriminator')


    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = ops.conv3d(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            if (FLAGS.GAN_type == 'GAN'):
                net = ops.batchnorm(net, FLAGS.is_training)
            net = ops.lrelu(net, 0.2)

        return net

    # with tf.device('/gpu:0'):
    with tf.variable_scope('discriminator_unit'):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = ops.conv3d(dis_inputs, 3, 64, 1, scope='conv')
            net = ops.lrelu(net, 0.2) # (m, 64, 64, 64, 64)

        # The discriminator block part
        # block 1
        net = discriminator_block(net, 64, 3, 2, 'disblock_1') # (m, 32, 32, 32, 64)

        # block 2
        net = discriminator_block(net, 128, 3, 1, 'disblock_2') # (m, 32, 32, 32, 128)

        # block 3
        net = discriminator_block(net, 128, 3, 2, 'disblock_3') # (m, 16, 16, 16, 128)

        # block 4
        net = discriminator_block(net, 256, 3, 1, 'disblock_4') # (m, 16, 16, 16, 256)

        # block 5
        net = discriminator_block(net, 256, 3, 2, 'disblock_5') # (m, 8, 8, 8, 256)

        # block 6
        net = discriminator_block(net, 512, 3, 1, 'disblock_6') # (m, 8, 8, 8, 512)

        # block_7
        net = discriminator_block(net, 512, 3, 2, 'disblock_7') # (m, 4, 4, 4, 512)

        # block_8
        net = discriminator_block(net, 1024, 3, 1, 'disblock_8') # (m, 4, 4, 4, 1024)

        # block_9
        net = discriminator_block(net, 1024, 3, 2, 'disblock_9') # (m, 2, 2, 2, 1024)

        # The dense layer 1
        with tf.variable_scope('dense_layer_1'):
            net = tf.contrib.layers.flatten(net)
            net = ops.denselayer(net, 1024)
            net = ops.lrelu(net, 0.2)

        # The dense layer 2
        with tf.variable_scope('dense_layer_2'):
            net = ops.denselayer(net, 1)
            net = tf.nn.sigmoid(net)

    return net
