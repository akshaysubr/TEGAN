import numpy as np
import tensorflow as tf

from hitDataTools import read_data

def _floatFeature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value));

def _int64Feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value));

# write a single example to a single TFRecord file 
def writeTFRecord(u,v,w,p,shape,filename):
    features = tf.train.Features(feature={'u':_floatFeature(u),
                                          'v':_floatFeature(v),
                                          'w':_floatFeature(w),
                                          'p':_floatFeature(p),
                                          'shape':_int64Feature(shape)})

    example = tf.train.Example(features=features)
    
    writer = tf.python_io.TFRecordWriter(filename)
    writer.write(example.SerializeToString())
    writer.close()
    
    return

def convertBinaryToTFRecord(input_filename):
    u,v,w,p = read_data(input_filename)
    shape = list(u.shape)
    
    u = u.reshape([-1])
    v = v.reshape([-1])
    w = w.reshape([-1])
    p = p.reshape([-1])
    
    output_filename = input_filename[0] + "{:06d}".format(input_filename[1]) + '.tfrecord'

    writeTFRecord(u,v,w,p,shape,output_filename)
    
    return
    
def binaryToTFRecord(input_filename_list, verbose=True):
    for i, input_filename in enumerate(input_filename_list):
        if verbose:
            print( "Processing file {} of {}".format(i+1, len(input_filename_list)) )
        convertBinaryToTFRecord(input_filename)
        
    return
