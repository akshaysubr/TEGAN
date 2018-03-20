import tensorflow as tf

def defaultFlags():
    Flags = tf.app.flags

    # The system parameter
    Flags.DEFINE_string('output_dir', '', 'The output directory of the checkpoint')
    Flags.DEFINE_string('summary_dir', '', 'The dirctory to output the summary')
    Flags.DEFINE_string('log_file', 'log.dat', 'File to write the logs to')
    Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
    Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
    Flags.DEFINE_boolean('pre_trained_generator', False, 'If set True, the weight will be loaded but the global_step will still '
                     'be 0. If set False, you are going to continue the training. That is, '
                     'the global_step will be initiallized from the checkpoint, too')
    Flags.DEFINE_boolean('pre_trained_model', False, 'If set True, the weight will be loaded but the global_step will still '
                     'be 0. If set False, you are going to continue the training. That is, '
                     'the global_step will be initiallized from the checkpoint, too')
    Flags.DEFINE_boolean('is_training', True, 'Training => True, Testing => False')
    Flags.DEFINE_string('task', 'TEGAN', 'The task: TEGAN, TEResNet')

    # The data preparing operation
    Flags.DEFINE_integer('batch_size', 5, 'Batch size of the input batch')
    Flags.DEFINE_integer('input_size', 16, 'Size of the input tensor')
    Flags.DEFINE_string('train_dir', '', 'The directory of the high resolution training data')
    Flags.DEFINE_string('dev_dir', '', 'The directory of the high resolution dev set')
    Flags.DEFINE_string('test_dir', '', 'The directory of the high resolution test set')
    Flags.DEFINE_integer('name_queue_capacity', 2048, 'The capacity of the filename queue (suggest large to ensure'
                         'enough random shuffle.')
    Flags.DEFINE_integer('image_queue_capacity', 2048, 'The capacity of the image queue (suggest large to ensure'
                     'enough random shuffle')
    Flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue (More threads can speedup the training process.')

    # Generator configuration
    Flags.DEFINE_integer('num_resblock', 4, 'How many residual blocks are there in the generator')

    # The training parameters
    Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
    Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
    Flags.DEFINE_float('decay_rate', 1.0, 'The decay rate of each decay step')
    Flags.DEFINE_boolean('stair', True, 'Whether perform staircase decay. True => decay in discrete interval.')
    Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
    Flags.DEFINE_integer('max_epoch', 100, 'The max epoch for the training')
    Flags.DEFINE_integer('max_iter', 1000, 'The max iteration of the training')
    Flags.DEFINE_integer('gen_start', 0, 'Iterations after which to start generator training')
    Flags.DEFINE_integer('gen_freq', 1, 'The diplay frequency of the training process')
    Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
    Flags.DEFINE_integer('summary_freq', 1, 'The frequency of writing summary')
    Flags.DEFINE_integer('save_freq', 10, 'The frequency of saving checkpoints')
    Flags.DEFINE_integer('dev_freq', 1, 'The frequency of saving dev summary')
    Flags.DEFINE_float('EPS', 1.e-12, 'Threshold for loss computations inside the log')
    Flags.DEFINE_float('adversarial_ratio', 1.e-3, 'Weighting factor for the adversarial loss')
    Flags.DEFINE_float('lambda_ens', 0.2, 'Weighting factor for enstrophy in content loss')
    Flags.DEFINE_float('lambda_con', 0.5, 'Weighting factor for continuity in physics loss')
    Flags.DEFINE_float('lambda_phy', 0.125, 'Weighting factor for physics loss in generator loss')
    Flags.DEFINE_string('GAN_type', 'GAN', 'GAN_type: GAN or WGAN_GP')
    Flags.DEFINE_float('lambda_WGAN', 10., 'Weightage for gradient penalty in WGAN-GP')
    
    FLAGS = Flags.FLAGS
    
    return FLAGS
