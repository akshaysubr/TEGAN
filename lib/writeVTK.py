from hitDataTools import read_data
import numpy as np
from pyevtk.hl import gridToVTK
from readTFRecord import *
import tensorflow as tf

directory = '../Data'
filenames = [directory + '/Run01_591337.tfrecord']
print(filenames)

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(parseTFRecordExample)
batched_dataset = dataset.batch(4)
iterator = batched_dataset.make_one_shot_iterator()

next_element = iterator.get_next()

with tf.Session() as sess:
    ne, = sess.run([next_element])
    print("next_element:     ", ne.shape)

u = np.copy(ne[0,:,:,:,0], order='C')
v = np.copy(ne[0,:,:,:,1], order='C')
w = np.copy(ne[0,:,:,:,2], order='C')
p = np.copy(ne[0,:,:,:,3], order='C')

# Dimensions 
nx, ny, nz = 64, 64, 64
lx, ly, lz = 2.0*np.pi, 2.0*np.pi, 2.0*np.pi 
dx, dy, dz = lx/nx, ly/ny, lz/nz 

# Coordinates 

X = np.arange(0, lx, dx, dtype='float64') 
Y = np.arange(0, ly, dy, dtype='float64') 
Z = np.arange(0, lz, dz, dtype='float64') 

x = np.zeros((nx, ny, nz)) 
y = np.zeros((nx, ny, nz)) 
z = np.zeros((nx, ny, nz)) 

for k in range(nz): 
    for j in range(ny):
        for i in range(nx): 
            x[i,j,k] = X[i] 
            y[i,j,k] = Y[j] 
            z[i,j,k] = Z[k] 

# Variables 
gridToVTK("./vtk_output", x, y, z, cellData = {"u" : u, "v" : v, "w" : w, "p" : p})
