import numpy as np
import os
import scipy.misc
import tensorflow as tf

def get_tags_from_event(filename):
    sess = tf.InteractiveSession()
    i = 0;
    tags = [];
    with sess.as_default():
        for event in tf.train.summary_iterator(filename):
            if (i==0):
                printed = 0;
                for val in event.summary.value:
                    print(val.tag)
                    tags.append[val.tag]
                    printed = 1
                if (printed):
                    i = 1
            else:
                break;
    return tags

def read_images_data_from_event(filename, tag):
    
    image_str = tf.placeholder(tf.string)
    image_tf = tf.image.decode_image(image_str)

    image_list = [];
    
    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for event in tf.train.summary_iterator(filename):
            for val in event.summary.value:
                if val.tag == tag:
                    im = image_tf.eval({image_str: val.image.encoded_image_string})
                    image_list.append(im)
                    count += 1

        return image_list
                    
def save_images_from_event(filename, tag, output_dir='./'):
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    image_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for event in tf.train.summary_iterator(filename):
            for val in event.summary.value:
                if val.tag == tag:
                    im = image_tf.eval({image_str: val.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{}_{:05d}.png'.format(output_dir, tag, count))
                    print("Saving '{}'".format(output_fn))
                    scipy.misc.imsave(output_fn, im)
                    count += 1
    return

def read_summary_value(filename, tag='MSE error'):

    value = []
    
    sess = tf.InteractiveSession()
    with sess.as_default():
        for event in tf.train.summary_iterator(filename):
            for val in event.summary.value:
                if val.tag == tag:
                    value.append(val.simple_value)

    return value

def read_summaryall_value(filename):

    tags = get_tags_from_event(filename)
    summary = {}
    for tag in tags:
        summary[tag] = []        
    
    sess = tf.InteractiveSession()
    with sess.as_default():
        for event in tf.train.summary_iterator(filename):
            for val in event.summary.value:
                summary[tag].append(val.simple_value)

    return summary

def get_scaled_image_data(filename, ind_start=0, ind_step=10000, ind_end=-1):
    tag_hr = 'High_resolution/image/0'
    tag_lr = 'Low_resolution/image/0'
    tag_gr = 'Generated/image/0'
    tag_ct = 'Concat/image/0'
    
    im_size = 64;

    im_lr_read = read_images_data_from_event(filename, tag_lr)
    im_ct_read = read_images_data_from_event(filename, tag_ct)    

    im_lr = [im[:,:,0] for im in im_lr_read[ind_start:ind_step:ind_end]]
    im_hr = [im[:,0:im_size,0] for im in im_ct_read[ind_start:ind_step:ind_end]]
    im_gr = [im[:,im_size:,0] for im in im_ct_read[ind_start:ind_step:ind_end]]

    hr_max = [np.max(im) for im in im_hr]
    hr_min = [np.min(im) for im in im_hr]

    im_hr_scaled = [(2.*(im-hr_min[i])/(hr_max[i]-hr_min[i]) - 1.) for i,im in enumerate(im_hr)]
    im_gr_scaled = [(2.*(im-hr_min[i])/(hr_max[i]-hr_min[i]) - 1.) for i,im in enumerate(im_gr)]
    im_lr_scaled = [(2.*(im-hr_min[i])/(hr_max[i]-hr_min[i]) - 1.) for i,im in enumerate(im_lr)]

    return im_lr_scaled, im_gr_scaled, im_hr_scaled
