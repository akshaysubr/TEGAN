import tensorflow as tf

def periodic_padding(inpt, pad):

    L = inpt[:,:pad,:,:,:]
    R = inpt[:,-pad:,:,:,:]
    inpt_pad = tf.concat([R, inpt, L], axis=1)

    L = inpt_pad[:,:,:pad,:,:]
    R = inpt_pad[:,:,-pad:,:,:]
    inpt_pad = tf.concat([R, inpt_pad, L], axis=2)

    L = inpt_pad[:,:,:,:pad,:]
    R = inpt_pad[:,:,:,-pad:,:]
    inpt_pad = tf.concat([R, inpt_pad, L], axis=3)

    return inpt_pad


if __name__ == "__main__":
    import numpy as np
    
    # Check if derivatives are correct with periodic padding
    X = tf.placeholder(tf.float32, shape=(1, 3, 3, 3, 1))
    X_pad = periodic_padding(X, 1)
    x_sum = tf.reduce_sum(X_pad)
    dX = tf.gradients(x_sum, X)

    with tf.Session() as sess:
        grad, = sess.run(dX, feed_dict={X: np.ones((1, 3, 3, 3, 1), dtype=np.float32)})

    print(grad[0,:,:,:,0])
