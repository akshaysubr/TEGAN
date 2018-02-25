import tensorflow as tf

def periodic_padding(inpt, pad):

    L = inpt[:,:2,:,:,:]
    R = inpt[:,-2:,:,:,:]
    inpt_pad = tf.concat([R, inpt, L], axis=1)

    L = inpt_pad[:,:,:2,:,:]
    R = inpt_pad[:,:,-2:,:,:]
    inpt_pad = tf.concat([R, inpt_pad, L], axis=2)

    L = inpt_pad[:,:,:,:2,:]
    R = inpt_pad[:,:,:,-2:,:]
    inpt_pad = tf.concat([R, inpt_pad, L], axis=3)

    return inpt_pad

