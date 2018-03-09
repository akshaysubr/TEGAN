import numpy as np
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



def discriminator(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for discriminator')


    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = ops.conv3d(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
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
    
        self.next_batch_HR = self.iterator.get_next()
        self.next_batch_LR = ops.filter3d(self.next_batch_HR)

        # TODO: Fix batch_size not being fator of total dataset size
        self.next_batch_LR.set_shape([FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, FLAGS.input_size, 4])

        self.FLAGS = FLAGS

        # Build the generator part
        with tf.variable_scope('generator'):
            self.output_channels = self.next_batch_HR.get_shape().as_list()[-1]
            self.gen_output = generator(self.next_batch_LR, self.output_channels, reuse=False, FLAGS=FLAGS)
            # self.gen_output.set_shape([FLAGS.batch_size, FLAGS.input_size * 4, FLAGS.input_size * 4, FLAGS.input_size * 4, 4])


        # Calculating the generator loss
        with tf.variable_scope('generator_loss'):

            # Content loss
            with tf.variable_scope('content_loss'):
                # Compute the euclidean distance between the two features
                self.content_loss = tf.reduce_mean( tf.square(self.gen_output - self.next_batch_HR) )

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
        return session.run( ( self.next_batch_HR, self.next_batch_LR, self.gen_output, self.gen_loss) )


class TEGAN(object):

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
    
        self.next_batch_HR = self.iterator.get_next()
        self.next_batch_LR = ops.filter3d(self.next_batch_HR)

        # TODO: Fix batch_size not being fator of total dataset size
        self.next_batch_LR.set_shape([FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, FLAGS.input_size, 4])
        self.next_batch_HR.set_shape([FLAGS.batch_size, FLAGS.input_size * 4, FLAGS.input_size * 4, FLAGS.input_size * 4, 4])

        self.FLAGS = FLAGS

        # Build the generator part
        with tf.variable_scope('generator'):
            self.output_channels = self.next_batch_HR.get_shape().as_list()[-1]
            self.gen_output = generator(self.next_batch_LR, self.output_channels, reuse=False, FLAGS=FLAGS)
            # self.gen_output.set_shape([FLAGS.batch_size, FLAGS.input_size * 4, FLAGS.input_size * 4, FLAGS.input_size * 4, 4])

        # Build the fake discriminator
        with tf.name_scope('fake_discriminator'):
            with tf.variable_scope('discriminator', reuse=False):
                self.discrim_fake_output = discriminator(self.gen_output, FLAGS=FLAGS)

        # Build the real discriminator
        with tf.name_scope('real_discriminator'):
            with tf.variable_scope('discriminator', reuse=True):
                self.discrim_real_output = discriminator(self.next_batch_HR, FLAGS=FLAGS)


        # Calculating the generator loss
        with tf.variable_scope('generator_loss'):

            # Content loss
            with tf.variable_scope('content_loss'):
                # Compute the euclidean distance between the two features
                self.content_loss = tf.reduce_mean( tf.square(self.gen_output - self.next_batch_HR) )

            with tf.variable_scope('adversarial_loss'):
                self.adversarial_loss = tf.reduce_mean(-tf.log(self.discrim_fake_output + FLAGS.EPS))

            self.gen_loss = (1 - FLAGS.adversarial_ratio) * self.content_loss + (FLAGS.adversarial_ratio) * self.adversarial_loss

        tf.summary.scalar('Generator loss', self.gen_loss)
        tf.summary.scalar('Adversarial loss', self.adversarial_loss)
        tf.summary.scalar('Content loss', self.content_loss)
        
        # Calculating the discriminator loss
        with tf.variable_scope('discriminator_loss'):
            discrim_fake_loss = tf.log(1 - self.discrim_fake_output + FLAGS.EPS)
            discrim_real_loss = tf.log(self.discrim_real_output + FLAGS.EPS)

            self.discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))

        tf.summary.scalar('Discriminator loss', self.discrim_loss)

        with tf.variable_scope('get_learning_rate_and_global_step'):

            self.global_step = tf.train.create_global_step()
            self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                                            staircase=FLAGS.stair)
            # self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

        with tf.variable_scope('dicriminator_train'):

            discrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            discrim_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=FLAGS.beta)
            self.discrim_train = discrim_optimizer.minimize( self.discrim_loss, self.global_step )

        with tf.variable_scope('generator_train'):

            # Need to wait discriminator to perform train step
            with tf.control_dependencies( [self.discrim_train] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                gen_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=FLAGS.beta)
                self.gen_train = gen_optimizer.minimize( self.gen_loss )

        exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
        self.update_loss = exp_averager.apply([self.discrim_loss, self.content_loss, self.adversarial_loss])

        # Define data saver
        self.saver = tf.train.Saver(max_to_keep=10)
        self.weights_initializer = tf.train.Saver(discrim_tvars + gen_tvars)
        self.weights_initializer_g = tf.train.Saver(gen_tvars)

        # Summary
        # tf.summary.image("u-velocity", tf.convert_to_tensor( 1.0*np.random.randint(0,255,(5,64,64,3)) )  )
        # tf.summary.image("v-velocity", tf.convert_to_tensor( 1.0*np.random.randint(0,255,(1,64,64,3)) )  )
        # tf.summary.image("pressure"  , tf.convert_to_tensor( 1.0*np.random.randint(0,255,(1,64,64,3)) )  )
        # self.merged_summary = tf.summary.merge([self.dloss_summary,self.gloss_summary,summary_image])
        self.merged_summary = tf.summary.merge_all()

    def initialize(self, session):

        session.run( tf.global_variables_initializer() )

        if self.FLAGS.checkpoint is not None:
            if self.FLAGS.pre_trained_generator:
                print("Restoring generator weights from {}".format(self.FLAGS.checkpoint))
                self.weights_initializer_g.restore(session, self.FLAGS.checkpoint)
            elif self.FLAGS.pre_trained_model:
                print("Restoring generator weights from {}".format(self.FLAGS.checkpoint))
                self.weights_initializer.restore(session, self.FLAGS.checkpoint)
            else:
                print("Restoring weights from {}".format(self.FLAGS.checkpoint))
                self.saver.restore(session, self.FLAGS.checkpoint)
    
        self.summary_writer = tf.summary.FileWriter( self.FLAGS.summary_dir, session.graph )


    def optimize(self, session):

        if self.FLAGS.mode != 'train':
            raise RuntimeError("Cannot optimize if not in train mode!!!")

        with open('log.dat', 'a') as f:

            for i in range(self.FLAGS.max_iter):
                try:
                    if ( (i+1) % self.FLAGS.gen_freq) == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                        d_loss, g_loss, train, step, summary = session.run( (self.discrim_loss, self.gen_loss, self.gen_train, self.global_step, self.merged_summary) ,
                                                                            options=run_options,
                                                                            run_metadata=run_metadata)

                        f.write('%06d %26.16e %26.16e\n' %(step, d_loss, g_loss))
                        print("Iteration {}: discriminator loss = {}, generator loss = {}".format(i, d_loss, g_loss))

                        self.summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                        self.summary_writer.add_summary(summary, step)
                    else:
                        d_loss, train, step = session.run( (self.discrim_loss, self.discrim_train, self.global_step) )
                        print("Iteration {}: discriminator loss = {}".format(i, d_loss))

                except tf.errors.OutOfRangeError:
                    # training terminated
                    print("Finished training!")
                    break


                # Save after every save_freq iterations
                if (step % 10) == 0:
                    print("Saving weights to {}".format(os.path.join(self.FLAGS.output_dir, 'TEGAN')))
                    self.saver.save(session, os.path.join(self.FLAGS.output_dir, 'TEGAN'), global_step=step)

        return results


    def evaluate(self, session):
        return session.run( ( self.next_batch_HR, self.next_batch_LR, self.gen_output, self.gen_loss ) )

