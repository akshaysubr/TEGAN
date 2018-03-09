import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys

varindices = {'u': 0, 'v': 1, 'w': 2, 'p': 3}

def plot(var, ax, extent=(0, 2.*np.pi,0, 2.*np.pi), vmin=None, vmax=None, cmap=None):
    # if vmin == None:
    #     vmin = var.min()
    # if vmax == None:
    #     vmax = var.max()

    if cmap == None:
        cmap = plt.get_cmap('viridis')

    im = ax.imshow( var, extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', interpolation='none', aspect='equal' )
    ax.set_xlim((extent[0],extent[1]))
    ax.set_ylim((extent[2],extent[3]))
    return im

def convert_to_rgb(a, vmin, vmax,  cmap=plt.cm.viridis):
    rgb = cmap( (a - vmin)/(vmax - vmin) )
    return rgb[:,:,:-1]


if len(sys.argv) < 5:
    print("Usage: ")
    print("    python {} <HDF5 output file> <variable> <plane> <index>".format(sys.argv[0]))
    sys.exit()

filename = sys.argv[1]
var = varindices[ sys.argv[2] ]
plane = int(sys.argv[3])
index = int(sys.argv[4])

h5f = h5py.File(filename, 'r+')
HR  = h5f['HR'].value
LR  = h5f['LR'].value
out = h5f['output'].value

batch_size, nx, ny, nz, _ = HR.shape
x = np.linspace(0,2.*np.pi,num=nx+1)[:-1].reshape((nx,1,1)).repeat(ny, axis=1).repeat(nz, axis=2)
y = np.linspace(0,2.*np.pi,num=ny+1)[:-1].reshape((1,ny,1)).repeat(nx, axis=0).repeat(nz, axis=2)
z = np.linspace(0,2.*np.pi,num=nz+1)[:-1].reshape((1,1,nz)).repeat(nx, axis=0).repeat(ny, axis=1)

batch = np.random.randint(batch_size, size=1)[0]

if plane == 0:
    var_HR  =  HR[batch, index, :, :, var]
    var_LR  =  LR[batch, index, :, :, var]
    var_out = out[batch, index, :, :, var]
elif plane == 1:
    var_HR  =  HR[batch, :, index, :, var]
    var_LR  =  LR[batch, :, index, :, var]
    var_out = out[batch, :, index, :, var]
elif plane == 2:
    var_HR  =  HR[batch,  :, :,index, var]
    var_LR  =  LR[batch,  :, :,index, var]
    var_out = out[batch,  :, :,index, var]
else:
    raise ValueError('Plane has to be 0, 1 or 2. Given {}'.format(plane))

vmin = var_HR.min()
vmax = var_HR.max()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(11,4.3))

im1 = plot(var_HR,  ax1, vmin=vmin, vmax=vmax)
ax1.set_title('High resolution')

im2 = plot(var_LR,  ax2, vmin=vmin, vmax=vmax)
ax2.set_title('Low resolution')

im3 = plot(var_out, ax3, vmin=vmin, vmax=vmax)
ax3.set_title('Generator output')

fig.tight_layout()
plt.show()

