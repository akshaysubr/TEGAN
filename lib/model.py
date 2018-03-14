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

class TEResNet(object):

    def __init__(self, filenames_train, filenames_dev, FLAGS):

        self.filenames_train = filenames_train
        self.filenames_dev   = filenames_dev

        self.dataset_train = tf.data.TFRecordDataset(self.filenames_train)
        self.dataset_train = self.dataset_train.map(parseTFRecordExample)
        self.dataset_train = self.dataset_train.shuffle(buffer_size=10000)
        self.dataset_train = self.dataset_train.batch(FLAGS.batch_size)
        if FLAGS.mode == 'train':
            self.dataset_train = self.dataset_train.repeat(FLAGS.max_epoch)
        else:
            self.dataset_train = self.dataset_train.repeat(1)

        self.iterator_train = self.dataset_train.make_one_shot_iterator()
        # self.iterator_train_handle = session.run(self.iterator_train.string_handle())
    
        self.dataset_dev = tf.data.TFRecordDataset(self.filenames_dev)
        self.dataset_dev = self.dataset_dev.map(parseTFRecordExample)
        self.dataset_dev = self.dataset_dev.shuffle(buffer_size=10000)
        self.dataset_dev = self.dataset_dev.batch(FLAGS.batch_size)
        if FLAGS.mode == 'train':
            self.dataset_dev = self.dataset_dev.repeat()
        else:
            self.dataset_dev = self.dataset_dev.repeat(1)

        self.iterator_dev = self.dataset_dev.make_one_shot_iterator()
        # self.iterator_dev_handle = session.run(self.iterator_dev.string_handle())
   
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.iterator_train.output_types)

        # TODO: Fix batch_size not being fator of total dataset size
        self.next_batch_HR = self.iterator.get_next()
        self.next_batch_HR.set_shape([FLAGS.batch_size, FLAGS.input_size * 4, FLAGS.input_size * 4, FLAGS.input_size * 4, 4])

        self.next_batch_LR = ops.filter3d(self.next_batch_HR)
        self.next_batch_LR.set_shape([FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, FLAGS.input_size, 4])

        self.FLAGS = FLAGS

        # Build the generator part
        with tf.variable_scope('generator'):
            self.output_channels = self.next_batch_HR.get_shape().as_list()[-1]
            self.gen_output = generator(self.next_batch_LR, self.output_channels, reuse=False, FLAGS=FLAGS)
            # self.gen_output.set_shape([FLAGS.batch_size, FLAGS.input_size * 4, FLAGS.input_size * 4, FLAGS.input_size * 4, 4])

        # Summary
        tf.summary.image("High resolution", self.next_batch_HR[0:1,:,:,0,0:1]  )
        tf.summary.image("Low resolution", self.next_batch_LR[0:1,:,:,0,0:1]  )
        tf.summary.image("Generated", self.gen_output[0:1,:,:,0,0:1]  )
        tf.summary.image("Concat", tf.concat( [self.next_batch_HR[0:1,:,:,0,0:1], self.gen_output[0:1,:,:,0,0:1]], axis=2 ))
        
        # Calculating the generator loss
        with tf.variable_scope('generator_loss'):

            dx = 2.*np.pi/(4.*self.FLAGS.input_size) 

            vel_grad = ops.get_velocity_grad(self.gen_output, dx, dx, dx) 
            vel_grad_HR = ops.get_velocity_grad(self.next_batch_HR, dx, dx, dx) 
            strain_rate_2_HR = tf.reduce_mean( tf.reduce_mean( tf.reduce_mean( \
                               ops.get_strain_rate_mag2(vel_grad_HR), axis=1, keep_dims=True), \
                               axis=2, keep_dims=True), axis=3, keep_dims=True)

            self.continuity_res = ops.get_continuity_residual(vel_grad) 
            self.pressure_res = ops.get_pressure_residual(self.gen_output, vel_grad, dx, dx, dx) 

            tke_gen = ops.get_TKE(self.gen_output) 
            tke_hr  = ops.get_TKE(self.next_batch_HR) 
            tke_hr_mean2 = tf.reduce_mean( tf.reduce_mean( tf.reduce_mean( \
                           tf.square(tke_hr), axis=1, keep_dims=True), axis=2, keep_dims=True), axis=3, keep_dims=True )
            self.tke_loss = tf.reduce_mean( tf.square(tke_gen-tke_hr) / tke_hr_mean2 )

            vorticity_gen = ops.get_vorticity(vel_grad) 
            vorticity_hr  = ops.get_vorticity(vel_grad_HR) 
            # self.vorticity_loss = tf.reduce_mean(tf.square(vorticity_gen-vorticity_hr)) 
 
            ens_gen = ops.get_enstrophy(vorticity_gen) 
            ens_hr  = ops.get_enstrophy(vorticity_hr) 
            ens_hr_mean2 = tf.reduce_mean( tf.reduce_mean( tf.reduce_mean( \
                           tf.square(ens_hr), axis=1, keep_dims=True), axis=2, keep_dims=True), axis=3, keep_dims=True )
            self.ens_loss = tf.reduce_mean( tf.square(ens_gen-ens_hr) / ens_hr_mean2 )

            # Compute the euclidean distance between the two features
            mse_hr_mean2 = tf.reduce_mean( tf.reduce_mean( tf.reduce_mean( \
                           tf.square(self.next_batch_HR), axis=1, keep_dims=True), axis=2, keep_dims=True), axis=3, keep_dims=True )
            self.mse_loss = tf.reduce_mean( tf.square(self.gen_output - self.next_batch_HR) / mse_hr_mean2 )

            # Content loss
            with tf.variable_scope('content_loss'):
                # Content loss => mse + enstrophy
                self.content_loss = (1 - self.FLAGS.lambda_ens) * self.mse_loss + self.FLAGS.lambda_ens * self.ens_loss

            # Physics loss 
            with tf.variable_scope('physics_loss'): 
                self.continuity_loss = tf.reduce_mean( tf.square(self.continuity_res) / strain_rate_2_HR )
                self.pressure_loss = tf.reduce_mean( tf.square(self.pressure_res) / strain_rate_2_HR**2 )

                self.physics_loss = (1 - self.FLAGS.lambda_con) * self.pressure_loss + self.FLAGS.lambda_con * self.continuity_loss


            self.gen_loss = (1 - self.FLAGS.lambda_phy) * self.content_loss + self.FLAGS.lambda_phy * self.physics_loss

        tf.summary.scalar('Generator loss', self.gen_loss) 
        tf.summary.scalar('Content loss', self.content_loss) 
        tf.summary.scalar('Physics loss', self.physics_loss) 
        tf.summary.scalar('MSE error', tf.sqrt( self.mse_loss) ) 
        tf.summary.scalar('Continuity error', tf.sqrt( self.continuity_loss) )
        tf.summary.scalar('Pressure error', tf.sqrt(self.pressure_loss) )
        tf.summary.scalar('TKE error', tf.sqrt(self.tke_loss) )
        # tf.summary.scalar('Vorticity loss', self.vorticity_loss) 
        tf.summary.scalar('Enstrophy error', tf.sqrt(self.ens_loss) )

        tf.summary.image('Z - Continuity residual',self.continuity_res[0:1,:,:,0,0:1])
        tf.summary.image('Z - Pressure residual',self.pressure_res[0:1,:,:,0,0:1])

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

        self.merged_summary = tf.summary.merge_all()

    def initialize(self, session):

        if self.FLAGS.checkpoint is None:
            session.run( tf.global_variables_initializer() )
        else:
            print("Restoring weights from {}".format(self.FLAGS.checkpoint))
            self.weights_initializer.restore(session, self.FLAGS.checkpoint)

        self.summary_writer_train = tf.summary.FileWriter( os.path.join(self.FLAGS.summary_dir, 'train'), session.graph )
        self.summary_writer_dev   = tf.summary.FileWriter( os.path.join(self.FLAGS.summary_dir, 'dev'),   session.graph )

        self.iterator_train_handle = session.run(self.iterator_train.string_handle())
        self.iterator_dev_handle = session.run(self.iterator_dev.string_handle())

    def optimize(self, session):

        if self.FLAGS.mode != 'train':
            raise RuntimeError("Cannot optimize if not in train mode!!!")

        for i in range(self.FLAGS.max_iter):
            try:
                if ( (i+1) % self.FLAGS.summary_freq) == 0:
                    g_loss, train, step, summary  = session.run( (self.gen_loss, self.gen_train, self.global_step, self.merged_summary),
                                                                 feed_dict={self.handle: self.iterator_train_handle} )
                    self.summary_writer_train.add_summary(summary, step)

                    if ( (i+1) % (self.FLAGS.summary_freq*self.FLAGS.dev_freq) )  == 0:
                        g_loss_dev, summary = session.run( (self.gen_loss, self.merged_summary) ,
                                                           feed_dict={self.handle: self.iterator_dev_handle} )
                        self.summary_writer_dev.add_summary(summary, step)

                    with open(self.FLAGS.log_file, 'a') as f:
                        f.write('%06d %26.16e\n' %(step, g_loss))
                        f.flush()
                    print("Iteration {}: generator loss = {}".format(i, g_loss))
                else:
                    g_loss, train, step = session.run( (self.gen_loss, self.gen_train, self.global_step),
                                                       feed_dict={self.handle: self.iterator_train_handle} )
                    print("Iteration {}: generator loss = {}".format(i, g_loss))

            except tf.errors.OutOfRangeError:
                # training terminated
                print("Finished training!")
                break

            # Save after every save_freq iterations
            if (step % self.FLAGS.save_freq) == 0:
                print("Saving weights to {}".format(os.path.join(self.FLAGS.output_dir, 'model')))
                self.saver.save(session, os.path.join(self.FLAGS.output_dir, 'model'), global_step=step)
            
        return 


    def evaluate(self, session):
        return session.run( ( self.next_batch_HR, self.next_batch_LR, self.gen_output, self.gen_loss),
                            feed_dict={self.handle: self.iterator_train_handle} )


class TEGAN(object):

    def __init__(self, filenames_train, filenames_dev, FLAGS):

        self.filenames_train = filenames_train
        self.filenames_dev   = filenames_dev

        self.dataset_train = tf.data.TFRecordDataset(self.filenames_train)
        self.dataset_train = self.dataset_train.map(parseTFRecordExample)
        self.dataset_train = self.dataset_train.shuffle(buffer_size=10000)
        self.dataset_train = self.dataset_train.batch(FLAGS.batch_size)
        if FLAGS.mode == 'train':
            self.dataset_train = self.dataset_train.repeat(FLAGS.max_epoch)
        else:
            self.dataset_train = self.dataset_train.repeat(1)

        self.iterator_train = self.dataset_train.make_one_shot_iterator()
        # self.iterator_train_handle = session.run(self.iterator_train.string_handle())
    
        self.dataset_dev = tf.data.TFRecordDataset(self.filenames_dev)
        self.dataset_dev = self.dataset_dev.map(parseTFRecordExample)
        self.dataset_dev = self.dataset_dev.shuffle(buffer_size=10000)
        self.dataset_dev = self.dataset_dev.batch(FLAGS.batch_size)
        if FLAGS.mode == 'train':
            self.dataset_dev = self.dataset_dev.repeat()
        else:
            self.dataset_dev = self.dataset_dev.repeat(1)

        self.iterator_dev = self.dataset_dev.make_one_shot_iterator()
        # self.iterator_dev_handle = session.run(self.iterator_dev.string_handle())
   
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.iterator_train.output_types)

        # TODO: Fix batch_size not being fator of total dataset size
        self.next_batch_HR = self.iterator.get_next()
        self.next_batch_HR.set_shape([FLAGS.batch_size, FLAGS.input_size * 4, FLAGS.input_size * 4, FLAGS.input_size * 4, 4])

        self.next_batch_LR = ops.filter3d(self.next_batch_HR)
        self.next_batch_LR.set_shape([FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, FLAGS.input_size, 4])

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

        # Summary
        tf.summary.image("High resolution", self.next_batch_HR[0:1,:,:,0,0:1]  )
        tf.summary.image("Low resolution", self.next_batch_LR[0:1,:,:,0,0:1]  )
        tf.summary.image("Generated", self.gen_output[0:1,:,:,0,0:1]  )
        tf.summary.image("Concat", tf.concat( [self.next_batch_HR[0:1,:,:,0,0:1], self.gen_output[0:1,:,:,0,0:1]], axis=2 ))
        
        # Calculating the generator loss
        with tf.variable_scope('generator_loss'):

            dx = 2.*np.pi/(4.*self.FLAGS.input_size) 

            vel_grad = ops.get_velocity_grad(self.gen_output, dx, dx, dx) 
            vel_grad_HR = ops.get_velocity_grad(self.next_batch_HR, dx, dx, dx) 
            strain_rate_2_HR = tf.reduce_mean( tf.reduce_mean( tf.reduce_mean( \
                               ops.get_strain_rate_mag2(vel_grad_HR), axis=1, keep_dims=True), \
                               axis=2, keep_dims=True), axis=3, keep_dims=True)

            self.continuity_res = ops.get_continuity_residual(vel_grad) 
            self.pressure_res = ops.get_pressure_residual(self.gen_output, vel_grad, dx, dx, dx) 

            tke_gen = ops.get_TKE(self.gen_output) 
            tke_hr  = ops.get_TKE(self.next_batch_HR) 
            tke_hr_mean2 = tf.reduce_mean( tf.reduce_mean( tf.reduce_mean( \
                           tf.square(tke_hr), axis=1, keep_dims=True), axis=2, keep_dims=True), axis=3, keep_dims=True )
            self.tke_loss = tf.reduce_mean( tf.square(tke_gen-tke_hr) / tke_hr_mean2 )

            vorticity_gen = ops.get_vorticity(vel_grad) 
            vorticity_hr  = ops.get_vorticity(vel_grad_HR) 
            # self.vorticity_loss = tf.reduce_mean(tf.square(vorticity_gen-vorticity_hr)) 
 
            ens_gen = ops.get_enstrophy(vorticity_gen) 
            ens_hr  = ops.get_enstrophy(vorticity_hr) 
            ens_hr_mean2 = tf.reduce_mean( tf.reduce_mean( tf.reduce_mean( \
                           tf.square(ens_hr), axis=1, keep_dims=True), axis=2, keep_dims=True), axis=3, keep_dims=True )
            self.ens_loss = tf.reduce_mean( tf.square(ens_gen-ens_hr) / ens_hr_mean2 )

            # Compute the euclidean distance between the two features
            mse_hr_mean2 = tf.reduce_mean( tf.reduce_mean( tf.reduce_mean( \
                           tf.square(self.next_batch_HR), axis=1, keep_dims=True), axis=2, keep_dims=True), axis=3, keep_dims=True )
            self.mse_loss = tf.reduce_mean( tf.square(self.gen_output - self.next_batch_HR) / mse_hr_mean2 )

            # Content loss
            with tf.variable_scope('content_loss'):
                # Content loss => mse + enstrophy
                self.content_loss = (1 - self.FLAGS.lambda_ens) * self.mse_loss + self.FLAGS.lambda_ens * self.ens_loss

            # Physics loss 
            with tf.variable_scope('physics_loss'): 
                self.continuity_loss = tf.reduce_mean( tf.square(self.continuity_res) / strain_rate_2_HR )
                self.pressure_loss = tf.reduce_mean( tf.square(self.pressure_res) / strain_rate_2_HR**2 )

                self.physics_loss = (1 - self.FLAGS.lambda_con) * self.pressure_loss + self.FLAGS.lambda_con * self.continuity_loss

            with tf.variable_scope('adversarial_loss'):
                if (FLAGS.GAN_type == 'GAN'):
                    self.adversarial_loss = tf.reduce_mean(-tf.log(self.discrim_fake_output + FLAGS.EPS))
                
                if (FLAGS.GAN_type == 'WGAN_GP'):
                    self.adversarial_loss = tf.reduce_mean(-self.discrim_fake_output)

            self.gen_loss = (1 - self.FLAGS.lambda_phy) * self.content_loss + self.FLAGS.lambda_phy * self.physics_loss
            self.gen_loss = (1 - self.FLAGS.adversarial_ratio) * self.gen_loss + (self.FLAGS.adversarial_ratio) * self.adversarial_loss

        tf.summary.scalar('Generator loss', self.gen_loss) 
        tf.summary.scalar('Adversarial loss', self.adversarial_loss) 
        tf.summary.scalar('Content loss', self.content_loss) 
        tf.summary.scalar('Physics loss', self.physics_loss) 
        tf.summary.scalar('MSE error', tf.sqrt( self.mse_loss) ) 
        tf.summary.scalar('Continuity error', tf.sqrt( self.continuity_loss) )
        tf.summary.scalar('Pressure error', tf.sqrt(self.pressure_loss) )
        tf.summary.scalar('TKE error', tf.sqrt(self.tke_loss) )
        # tf.summary.scalar('Vorticity loss', self.vorticity_loss) 
        tf.summary.scalar('Enstrophy error', tf.sqrt(self.ens_loss) )

        tf.summary.image('Z - Continuity residual',self.continuity_res[0:1,:,:,0,0:1])
        tf.summary.image('Z - Pressure residual',self.pressure_res[0:1,:,:,0,0:1])
        
        # Create a new instance of the discriminator for gradient penalty
        if (FLAGS.GAN_type == 'WGAN_GP'):
            eps_WGAN = tf.random_uniform(shape=[FLAGS.batch_size, 1, 1, 1, 1], minval = 0., maxval = 1.)
            inpt_hat = eps_WGAN * self.next_batch_HR + (1 - eps_WGAN) * self.gen_output

            # Build the interpolatd discriminator for WGAN-GP
            with tf.name_scope('hat_discriminator'):
                with tf.variable_scope('discriminator', reuse=True):
                    discrim_hat_output = discriminator(inpt_hat, FLAGS=FLAGS)
                
        # Calculating the discriminator loss
        with tf.variable_scope('discriminator_loss'):
            if (FLAGS.GAN_type == 'GAN'):
                discrim_fake_loss = tf.log(1 - self.discrim_fake_output + FLAGS.EPS)
                discrim_real_loss = tf.log(self.discrim_real_output + FLAGS.EPS)

                self.discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))

            if (FLAGS.GAN_type == 'WGAN_GP'):
                self.discrim_loss = tf.reduce_mean(self.discrim_fake_output -self.discrim_real_output)
                
                grad_dicrim_inpt_hat = tf.gradients(discrim_hat_output, [inpt_hat])[0]

                # L2-Norm across channels
                gradnorm_discrim_inpt_hat = tf.sqrt(tf.reduce_sum(tf.square(grad_dicrim_inpt_hat), reduction_indices=[-1]))
                gradient_penalty = tf.reduce_mean((gradnorm_discrim_inpt_hat - 1.)**2)

                self.discrim_loss += FLAGS.lambda_WGAN * gradient_penalty

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
        tf.summary.scalar("Discriminator fake output", self.discrim_fake_output[0,0])
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
    
        self.summary_writer_train = tf.summary.FileWriter( os.path.join(self.FLAGS.summary_dir, 'train'), session.graph )
        self.summary_writer_dev   = tf.summary.FileWriter( os.path.join(self.FLAGS.summary_dir, 'dev'),   session.graph )

        self.iterator_train_handle = session.run(self.iterator_train.string_handle())
        self.iterator_dev_handle = session.run(self.iterator_dev.string_handle())

    def optimize(self, session):

        if self.FLAGS.mode != 'train':
            raise RuntimeError("Cannot optimize if not in train mode!!!")

        for i in range(self.FLAGS.max_iter):

            get_summary = False
            if ( (i+1) % self.FLAGS.summary_freq):
                get_summary = True
            
            try:
                if (i >= self.FLAGS.gen_start) and ( (i+1) % self.FLAGS.gen_freq == 0 ):
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    if get_summary:
                        d_loss, g_loss, train, step, summary = session.run( (self.discrim_loss, self.gen_loss, self.gen_train, self.global_step, self.merged_summary) ,
                                                                            feed_dict={self.handle: self.iterator_train_handle},
                                                                            options=run_options,
                                                                            run_metadata=run_metadata)
                        self.summary_writer_train.add_run_metadata(run_metadata, 'step%06d' % step)
                        self.summary_writer_train.add_summary(summary, step)
                    else:
                        d_loss, g_loss, train, step = session.run( (self.discrim_loss, self.gen_loss, self.gen_train, self.global_step),
                                                                   feed_dict={self.handle: self.iterator_train_handle},
                                                                   options=run_options,
                                                                   run_metadata=run_metadata)


                    with open(self.FLAGS.log_file, 'a') as f:
                        f.write('%06d %26.16e %26.16e\n' %(step, d_loss, g_loss))
                        f.flush()
                    print("Iteration {}: discriminator loss = {}, generator loss = {}".format(i, d_loss, g_loss))
                else:
                    if get_summary:
                        d_loss, train, step, summary = session.run( (self.discrim_loss, self.discrim_train, self.global_step, self.merged_summary),
                                                                    feed_dict={self.handle: self.iterator_train_handle})
                        self.summary_writer_train.add_summary(summary, step)
                    else:
                        d_loss, train, step = session.run( (self.discrim_loss, self.discrim_train, self.global_step), feed_dict={self.handle: self.iterator_train_handle})
                    print("Iteration {}: discriminator loss = {}".format(i, d_loss))

                if ( (i+1) % (self.FLAGS.dev_freq) )  == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    d_loss_dev, g_loss_dev, summary = session.run( (self.discrim_loss, self.gen_loss, self.merged_summary) ,
                                                                        feed_dict={self.handle: self.iterator_dev_handle},
                                                                        options=run_options,
                                                                        run_metadata=run_metadata)

                    self.summary_writer_dev.add_run_metadata(run_metadata, 'step%06d' % step)
                    self.summary_writer_dev.add_summary(summary, step)


            except tf.errors.OutOfRangeError:
                # training terminated
                print("Finished training!")
                break


            # Save after every save_freq iterations
            if (step % self.FLAGS.save_freq) == 0:
                print("Saving weights to {}".format(os.path.join(self.FLAGS.output_dir, 'TEGAN')))
                self.saver.save(session, os.path.join(self.FLAGS.output_dir, 'TEGAN'), global_step=step)

        return


    def evaluate(self, session):
        return session.run( ( self.next_batch_HR, self.next_batch_LR, self.gen_output, self.gen_loss ), feed_dict={self.handle: self.iterator_train_handle})

