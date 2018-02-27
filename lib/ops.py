import tensorflow as tf
import tensorflow.contrib.slim as slim

def periodic_padding(inpt, pad):
    L = inpt[:,:pad[0][0],:,:,:]
    R = inpt[:,-pad[0][1]:,:,:,:]
    inpt_pad = tf.concat([R, inpt, L], axis=1)
    
    L = inpt_pad[:,:,:pad[1][0],:,:]
    R = inpt_pad[:,:,-pad[1][1]:,:,:]
    inpt_pad = tf.concat([R, inpt_pad, L], axis=2)
    
    L = inpt_pad[:,:,:,:pad[2][0],:]
    R = inpt_pad[:,:,:,-pad[2][1]:,:]
    inpt_pad = tf.concat([R, inpt_pad, L], axis=3)
    
    return inpt_pad

def conv3d_withPeriodicPadding(inpt, filtr, strides, name=None):
    ### Does not work for large strides ###
    inpt_shape = inpt.get_shape().as_list()
    filtr_shape = filtr.get_shape().as_list()
    pad = []
    
    for i_dim in range(3):
        padL = int(0.5*((inpt_shape[i_dim+1]-1)*strides[i_dim+1]
                        + filtr_shape[i_dim] - inpt_shape[i_dim+1]))
        padR = padL
        pad_idim = (padL,padR)
        pad.append(pad_idim)      
            
    inpt_pad = periodic_padding(inpt, pad)
    output = tf.nn.conv3d(inpt_pad, filtr, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    
    return output

def conv3d(inpt, f, output_channels, s, use_bias=False, scope='conv', name=None):
    inpt_shape = inpt.get_shape().as_list()
    with tf.variable_scope(scope):
        filtr = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                shape=[f,f,f,inpt_shape[-1],output_channels],name='filtr')
        
    strides = [1,s,s,s,1]
    output = conv3d_withPeriodicPadding(inpt,filtr,strides,name)
    
    if use_bias:
        with tf.variable_scope(scope):
            bias = tf.get_variable(intializer=tf.zeros_initializer(
                [1,1,1,1,output_channels],dtype=tf.float32),name='bias')
            output = output + bias;
    
    return output

def filter3d(inpt, scope='filter', name=None):
    inpt_shape = inpt.get_shape().as_list()
    with tf.variable_scope(scope):
        filter1D = tf.constant([0.04997364, 0.13638498, 0.20002636, 0.22723004, 0.20002636, 0.13638498, 0.04997364], dtype=tf.float32)
        filter1Dx = tf.reshape(filter1D, shape=(-1,1,1))
        filter1Dy = tf.reshape(filter1D, shape=(1,-1,1))
        filter1Dz = tf.reshape(filter1D, shape=(1,1,-1))
        filter3D = filter1Dx * filter1Dy * filter1Dz # Tensor product 3D filter using broadcasting

        filter3D = tf.expand_dims(filter3D, axis=3)
        zero = tf.constant( np.zeros((7,7,7,1), dtype=np.float32) )
        filter3Du = tf.concat( [filter3D, zero, zero, zero], axis=3 )
        filter3Dv = tf.concat( [zero, filter3D, zero, zero], axis=3 )
        filter3Dw = tf.concat( [zero, zero, filter3D, zero], axis=3 )
        filter3Dp = tf.concat( [zero, zero, zero, filter3D], axis=3 )
        filter3D = tf.stack( [filter3Du, filter3Dv, filter3Dw, filter3Dp], axis=4, name='filter' )

    strides = [1,4,4,4,1]
    inpt_pad = periodic_padding( inpt, ((3,3),(3,3),(3,3)) )
    output = tf.nn.conv3d(inpt_pad, filter3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    
    return output

def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha',inputs.get_shape()[-1],
                                 initializer=tf.zeros_initializer(),dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg


def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs,decay=0.9,epsilon=0.001,
                           updates_collections=tf.GraphKeys.UPDATE_OPS,
                           scale=False,fused=True,is_training=is_training)


def print_configuration_op(FLAGS):
    print('[Configurations]:')
    a = FLAGS.mode
    #pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        if type(value) == float:
            print('\t%s: %f'%(name, value))
        elif type(value) == int:
            print('\t%s: %d'%(name, value))
        elif type(value) == str:
            print('\t%s: %s'%(name, value))
        elif type(value) == bool:
            print('\t%s: %s'%(name, value))
        else:
            print('\t%s: %s' % (name, value))

    print('End of configuration')

                                   
def phaseShift(inputs, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 4, 2, 5, 3, 6])

    return tf.reshape(X, shape_2)


# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    d = size[1]
    h = size[2]
    w = size[3]
    c = size[4]

    # Get the target channel size
    channel_target = c // (scale * scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, d, h, w, scale, scale, scale]
    shape_2 = [batch_size, d * scale, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=4)
    output = tf.concat([phaseShift(x, shape_1, shape_2) for x in input_split], axis=4)

    return output


if __name__ == "__main__":
    import numpy as np
    
    # Check if derivatives are correct with periodic padding
    X = tf.placeholder(tf.float32, shape=(1, 5, 5, 5, 1))
    pad = [(1,1),(1,1),(1,1)]
    X_pad = periodic_padding(X, pad)
    
    x_sum = tf.reduce_sum(X_pad)
    dX = tf.gradients(x_sum, X)

    with tf.Session() as sess:
        grad, = sess.run(dX, feed_dict={X: np.ones((1, 5, 5, 5, 1), dtype=np.float32)})

    print(grad[0,:,:,:,0])

    # Checking the padding in conv3d
    filtr = tf.constant(np.ones((3,3,3,1,1), dtype =np.float32)) 
    output = conv3d_withPeriodicPadding(X, filtr, [1,1,1,1,1])
    with tf.Session() as sess:
        XConv = sess.run(output, feed_dict={X: np.ones((1, 5, 5, 5, 1), dtype=np.float32)})
       
    print(XConv.shape)
    print(XConv[0,:,:,:,0])

    Xphase = tf.placeholder(tf.float32, shape=(1, 4, 4, 4, 4))
    shape_1 = [1, 4, 4, 4, 2, 2, 1]
    shape_2 = [1, 8, 8, 4, 1]
    X_ups = phaseShift(Xphase, shape_1, shape_2)

    with tf.Session() as sess:
        xphase = np.zeros((1,4,4,4,4))
        for i in range(4):
            xphase[:,:,:,:,i] = i
        xups = sess.run( X_ups, feed_dict={ Xphase: xphase } )

    print(xups[0,:,:,0,0])
    print(xups.shape)


    HR = tf.placeholder(tf.float32, shape=(2,16,16,16,4))
    LR = filter3d(HR)

    with tf.Session() as sess:
        hr = np.zeros( (2,16,16,16,4), dtype=np.float32 )
        for i in range(4):
            hr[:,:,:,:,i] = i

        lr = sess.run(LR, feed_dict={HR: hr} )
        print(lr[0,:,:,:,3])
        print(lr.shape)
