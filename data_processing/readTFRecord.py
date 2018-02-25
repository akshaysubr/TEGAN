def _parse_function(example_proto):
    features = {'u': tf.VarLenFeature(tf.float32),
                'v': tf.VarLenFeature(tf.float32),
                'w': tf.VarLenFeature(tf.float32),
                'p': tf.VarLenFeature(tf.float32),
                'shape': tf.FixedLenFeature((3), tf.int64)}
    
    parsed_features = tf.parse_single_example(example_proto, features)
    
    u = tf.sparse_tensor_to_dense(parsed_features['u'])
    u = tf.reshape(u,tf.stack(parsed_features['shape']))

    v = tf.sparse_tensor_to_dense(parsed_features['v'])
    v = tf.reshape(v,tf.stack(parsed_features['shape']))    
    
    w = tf.sparse_tensor_to_dense(parsed_features['w'])
    w = tf.reshape(w,tf.stack(parsed_features['shape']))
    
    p = tf.sparse_tensor_to_dense(parsed_features['p'])
    p = tf.reshape(p,tf.stack(parsed_features['shape']))
    
    return u, v, w, p

filenames = [tfrecords_filename]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
iterator = dataset.make_one_shot_iterator()