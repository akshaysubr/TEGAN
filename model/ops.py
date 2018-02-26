import tensorflow as tf
import tensorFlor.contrib.slim as slim

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

def conv3d(inpt, f, output_channels, s, use_bias=False, scope, name=None):
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
