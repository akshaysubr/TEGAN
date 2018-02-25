import glob
import tensorflow as tf

def parseTFRecordExample(example_proto):
    features = {'u': tf.VarLenFeature(tf.float32),
                'v': tf.VarLenFeature(tf.float32),
                'w': tf.VarLenFeature(tf.float32),
                'p': tf.VarLenFeature(tf.float32),
                'shape': tf.FixedLenFeature((3), tf.int64)}
    
    parsed_features = tf.parse_single_example(example_proto, features)
   
    shape = tf.stack(parsed_features['shape'])

    u = tf.sparse_tensor_to_dense(parsed_features['u'])
    u = tf.reshape(u,shape)

    v = tf.sparse_tensor_to_dense(parsed_features['v'])
    v = tf.reshape(v,shape)    
    
    w = tf.sparse_tensor_to_dense(parsed_features['w'])
    w = tf.reshape(w,shape)
    
    p = tf.sparse_tensor_to_dense(parsed_features['p'])
    p = tf.reshape(p,shape)
    
    _, u_var = tf.nn.moments(u, axes=[0,1,2])
    _, v_var = tf.nn.moments(v, axes=[0,1,2])
    _, w_var = tf.nn.moments(w, axes=[0,1,2])

    u_rms = tf.sqrt( u_var + v_var + w_var )

    u = u / u_rms
    v = v / u_rms
    w = w / u_rms
    p = p / (u_rms * u_rms)

    element = tf.stack( [u, v, w, p], axis=3 )

    return element


def getTFRecordFilenamesIn(directory):
    return glob.glob(directory + '/*.tfrecord')


if __name__ == "__main__":
    import numpy as np
    from hitDataTools import read_data

    directory = '/farmshare/user_data/akshays/TEGAN/Data'
    filenames = [directory + '/Run01_001000.tfrecord']*2
    print(filenames)

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parseTFRecordExample)
    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()

    L = next_element[:2,:,:,:]
    R = next_element[-2:,:,:,:]
    next_element_pad = tf.concat([R, next_element, L], axis=0)

    L = next_element_pad[:,:2,:,:]
    R = next_element_pad[:,-2:,:,:]
    next_element_pad = tf.concat([R, next_element_pad, L], axis=1)

    L = next_element_pad[:,:,:2,:]
    R = next_element_pad[:,:,-2:,:]
    next_element_pad = tf.concat([R, next_element_pad, L], axis=2)

    with tf.Session() as sess:
        ne, nep = sess.run([next_element, next_element_pad])
        print("next_element:     ", ne.shape)
        print("next_element_pad: ", nep.shape)

    u, v, w, p = read_data( (directory + '/Run01_', 1000) )
    u_rms = np.sqrt( (u*u + v*v + w*w).mean() )
    u = u / u_rms
    v = v / u_rms
    w = w / u_rms
    p = p / (u_rms * u_rms)

    element = np.stack([u, v, w, p], axis=3)
    element = np.pad(element, [(2,2), (2,2), (2,2), (0,0)], mode='wrap')

    print("Error in u = {}".format( np.absolute(u - ne[:,:,:,0]).max() ))
    print("Error in v = {}".format( np.absolute(v - ne[:,:,:,1]).max() ))
    print("Error in w = {}".format( np.absolute(w - ne[:,:,:,2]).max() ))
    print("Error in p = {}".format( np.absolute(p - ne[:,:,:,3]).max() ))

    print("Error in padding = {}".format( np.absolute(element - nep).max() ))


