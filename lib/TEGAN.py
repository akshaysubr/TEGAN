import numpy as np
import os
import tensorflow as tf

import lib.model as model
import lib.ops as ops
from lib.readTFRecord import parseTFRecordExample 

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
            self.gen_output = model.generator(self.next_batch_LR, self.output_channels, reuse=False, FLAGS=FLAGS)
            # self.gen_output.set_shape([FLAGS.batch_size, FLAGS.input_size * 4, FLAGS.input_size * 4, FLAGS.input_size * 4, 4])

        # Build the fake discriminator
        with tf.name_scope('fake_discriminator'):
            with tf.variable_scope('discriminator', reuse=False):
                self.discrim_fake_output = model.discriminator(self.gen_output, FLAGS=FLAGS)

        # Build the real discriminator
        with tf.name_scope('real_discriminator'):
            with tf.variable_scope('discriminator', reuse=True):
                self.discrim_real_output = model.discriminator(self.next_batch_HR, FLAGS=FLAGS)

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
                    discrim_hat_output = model.discriminator(inpt_hat, FLAGS=FLAGS)
                
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
            self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

        with tf.variable_scope('dicriminator_train'):

            discrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            discrim_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=FLAGS.beta)
            discrim_grads_and_vars = discrim_optimizer.compute_gradients(self.discrim_loss, discrim_tvars)
            self.discrim_train = discrim_optimizer.apply_gradients(discrim_grads_and_vars)
            # self.discrim_train = discrim_optimizer.minimize( self.discrim_loss, self.global_step )

        with tf.variable_scope('generator_train'):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

            # Need to wait discriminator to perform train step
            with tf.control_dependencies( [self.discrim_train] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                gen_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=FLAGS.beta)
                gen_grads_and_vars = gen_optimizer.compute_gradients(self.gen_loss, gen_tvars)
                self.gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)
                # self.gen_train = gen_optimizer.minimize( self.gen_loss )

        exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
        self.update_loss = exp_averager.apply([self.discrim_loss, self.content_loss, self.adversarial_loss])

        # Define data saver
        self.saver = tf.train.Saver(max_to_keep=10)
        self.weights_initializer = tf.train.Saver(discrim_tvars + gen_tvars)
        self.weights_initializer_g = tf.train.Saver(gen_tvars)

        # Summary
        tf.summary.scalar("Discriminator fake output", self.discrim_fake_output[0,0])
        tf.summary.scalar("Learning rate", self.learning_rate)
        self.merged_summary = tf.summary.merge_all()

    def initialize(self, session):

        session.run( tf.global_variables_initializer() )

        if self.FLAGS.checkpoint is not None:
            if self.FLAGS.pre_trained_generator:
                print("Restoring generator weights from {}".format(self.FLAGS.checkpoint))
                self.weights_initializer_g.restore(session, self.FLAGS.checkpoint)
            elif self.FLAGS.pre_trained_model:
                print("Restoring model weights from {}".format(self.FLAGS.checkpoint))
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

                    if get_summary:
                        d_loss, g_loss, train, step, summary = session.run( (self.discrim_loss, self.gen_loss, self.gen_train, self.incr_global_step, self.merged_summary) ,
                                                                            feed_dict={self.handle: self.iterator_train_handle})
                        self.summary_writer_train.add_summary(summary, step)
                    else:
                        d_loss, g_loss, train, step = session.run( (self.discrim_loss, self.gen_loss, self.gen_train, self.incr_global_step),
                                                                   feed_dict={self.handle: self.iterator_train_handle})


                    with open(self.FLAGS.log_file, 'a') as f:
                        f.write('%06d %26.16e %26.16e\n' %(step, d_loss, g_loss))
                        f.flush()
                    print("Iteration {}: discriminator loss = {}, generator loss = {}".format(i, d_loss, g_loss))
                else:
                    if get_summary:
                        d_loss, g_loss, a_loss, train, step, summary = session.run( (self.discrim_loss, self.content_loss, self.adversarial_loss, self.discrim_train, self.incr_global_step, self.merged_summary),
                                                                    feed_dict={self.handle: self.iterator_train_handle})
                        self.summary_writer_train.add_summary(summary, step)
                    else:
                        d_loss, g_loss, a_loss, train, step = session.run( (self.discrim_loss, self.content_loss, self.adversarial_loss, self.discrim_train, self.incr_global_step), feed_dict={self.handle: self.iterator_train_handle})
                    print("Iteration {}: discriminator loss = {}, content loss = {}, adversarial loss = {}    Only training the discriminator".format(i, d_loss, g_loss, a_loss))

                if ( (i+1) % (self.FLAGS.dev_freq) )  == 0:
                    d_loss_dev, g_loss_dev, summary = session.run( (self.discrim_loss, self.gen_loss, self.merged_summary) ,
                                                                        feed_dict={self.handle: self.iterator_dev_handle} )
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


    def evaluate(self, session, filenames_test):
        
        dataset_test = tf.data.TFRecordDataset(filenames_test)
        dataset_test = dataset_test.map(parseTFRecordExample)
        dataset_test = dataset_test.repeat(1)
        dataset_test = dataset_test.batch(1)

        iterator_test = dataset_test.make_one_shot_iterator()
        iterator_test_handle = session.run(iterator_test.string_handle())

        output = []

        for i in range(len(filenames_test)):
            output.append(session.run( ( self.next_batch_HR, self.next_batch_LR, self.gen_output ), feed_dict={self.handle: iterator_test_handle}))

        return output

    def evaluate_losses(self, session, filenames_test):

        dataset_test = tf.data.TFRecordDataset(filenames_test)
        dataset_test = dataset_test.map(parseTFRecordExample)
        dataset_test = dataset_test.repeat(1)
        dataset_test = dataset_test.batch(self.FLAGS.batch_size)

        iterator_test = dataset_test.make_one_shot_iterator()
        iterator_test_handle = session.run(iterator_test.string_handle())

        output = {'gen_loss':[],'adversarial_loss':[], \
                  'content_loss':[], 'mse_loss':[], 'enstrophy_loss':[], \
                  'physics_loss':[], 'continuity_loss':[], 'pressure_loss':[], \
                  'discriminator_loss':[], 'discriminator_fake_output':[]}

        for i in range(len(filenames_test)):
            losses = session.run( ( self.gen_loss, self.adversarial_loss, \
                                    self.content_loss, self.mse_loss, self.ens_loss, \
                                    self.physics_loss, self.continuity_loss, self.pressure_loss, \
                                    self.discrim_loss, self.discrim_fake_output  ), feed_dict={self.handle: iterator_test_handle})
            output['gen_loss'].append(losses[0]);
            output['adversarial_loss'].append(losses[1]);
            output['content_loss'].append(losses[2]);
            output['mse_loss'].append(losses[3]);
            output['enstrophy_loss'].append(losses[4]);
            output['physics_loss'].append(losses[5]);
            output['continuity_loss'].append(losses[6]);
            output['pressure_loss'].append(losses[7]);
            output['discriminator_loss'].append(losses[8]);
            output['discriminator_fake_output'].append(losses[9]);

        return output

        
