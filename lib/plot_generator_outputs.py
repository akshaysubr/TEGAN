import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.interpolate as si
from scipy import ndimage
import sys

varindices = {'u': 0, 'v': 1, 'w': 2, 'p': 3}

def bicubic_interpolation(LR, HR_shape):

    LR_padded = np.zeros((LR.shape[0]+1,LR.shape[1]+1,LR.shape[2]+1))
    LR_padded[:-1,:-1,:-1] = LR[:,:,:]
    LR_padded[-1,:-1,:-1] = LR[0,:,:]
    LR_padded[:-1,-1,:-1] = LR[:,0,:]
    LR_padded[:-1,:-1,-1] = LR[:,:,0]
    LR_padded[:-1,-1,-1] = LR[:,0,0]
    LR_padded[-1,:-1,-1] = LR[0,:,0]
    LR_padded[-1,-1,:-1] = LR[0,0,:]
    LR_padded[-1,-1,-1] = LR[0,0,0]
    
    x_HR = np.linspace(0, LR.shape[0], num=HR_shape[0]+1)[:-1]
    y_HR = np.linspace(0, LR.shape[1], num=HR_shape[1]+1)[:-1]
    z_HR = np.linspace(0, LR.shape[2], num=HR_shape[2]+1)[:-1]
    
    xx, yy, zz = np.meshgrid(x_HR, y_HR, z_HR, indexing='ij')
    
    xx = xx.reshape((-1))
    yy = yy.reshape((-1))
    zz = zz.reshape((-1))
    out_BC = ndimage.map_coordinates(LR_padded, [xx, yy, zz], order=3, mode='wrap').reshape(HR_shape)
    
    return out_BC

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


def make_comparison_plots(LR, HR, out, output_label='Generated output'):

    vmin = HR.min()
    vmax = HR.max()

    LR_padded = np.zeros((LR.shape[0]+1,LR.shape[1]+1))
    LR_padded[:-1,:-1] = LR[:,:]
    LR_padded[-1,:-1] = LR[0,:]
    LR_padded[:-1,-1] = LR[:,0]
    LR_padded[-1,-1] = LR[0,0]

    x_HR = np.linspace(0, LR.shape[0], num=HR.shape[0]+1)[:-1]
    y_HR = np.linspace(0, LR.shape[1], num=HR.shape[1]+1)[:-1]
    print(x_HR)

    yy, xx = np.meshgrid(y_HR, x_HR)
    xx = xx.reshape((-1))
    yy = yy.reshape((-1))
    out_BC = ndimage.map_coordinates(LR_padded, [xx, yy], order=3, mode='wrap').reshape(HR.shape)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(17,5))

    im1 = plot(LR,  ax1, vmin=vmin, vmax=vmax)
    ax1.set_title('Low resolution')

    im2 = plot(out_BC,  ax2, vmin=vmin, vmax=vmax)
    ax2.set_title('Bicubic')

    im3 = plot(out,  ax3, vmin=vmin, vmax=vmax)
    ax3.set_title(output_label)

    im4 = plot(HR, ax4, vmin=vmin, vmax=vmax)
    ax4.set_title('High resolution')

    fig.tight_layout()
    
    return fig, (ax1, ax2, ax3, ax4)



def convert_to_rgb(a, vmin, vmax,  cmap=plt.cm.viridis):
    rgb = cmap( (a - vmin)/(vmax - vmin) )
    return rgb[:,:,:-1]


if __name__ == '__main__':

    if len(sys.argv) < 6:
        print("Usage: ")
        print("    python {} <HDF5 output file> <variable> <plane> <index> <output filename>".format(sys.argv[0]))
        sys.exit()

    filename = sys.argv[1]
    var = varindices[ sys.argv[2] ]
    plane = int(sys.argv[3])
    index = int(sys.argv[4])
    output_filename = sys.argv[5]

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

    fig, (ax1, ax2, ax3, ax4) = make_comparison_plots(var_LR, var_HR, var_out)
    fig.savefig(output_filename)
    print("Saved plot to {}".format(output_filename))

