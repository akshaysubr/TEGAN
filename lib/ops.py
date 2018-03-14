import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def periodic_padding(inpt, pad):
    L = inpt[:,:pad[0][0],:,:,:]
    if pad[0][1] > 0:
        R = inpt[:,-pad[0][1]:,:,:,:]
    else:
        R = inpt[:,:0,:,:,:]
    inpt_pad = tf.concat([R, inpt, L], axis=1)
    
    L = inpt_pad[:,:,:pad[1][0],:,:]
    if pad[1][1] > 0:
        R = inpt_pad[:,:,-pad[1][1]:,:,:]
    else:
        R = inpt_pad[:,:,:0,:,:]
    inpt_pad = tf.concat([R, inpt_pad, L], axis=2)
    
    L = inpt_pad[:,:,:,:pad[2][0],:]
    if pad[2][1] > 0:
        R = inpt_pad[:,:,:,-pad[2][1]:,:]
    else:
        R = inpt_pad[:,:,:,:0,:]
    inpt_pad = tf.concat([R, inpt_pad, L], axis=3)
    
    return inpt_pad


def conv3d_withPeriodicPadding(inpt, filtr, strides, name=None):
    ### Does not work for large strides ###
    inpt_shape = inpt.get_shape().as_list()
    filtr_shape = filtr.get_shape().as_list()
    pad = []
    
    for i_dim in range(3):
        # Compute pad assuming output_size = input_size / stride and odd filter sizes
        padL = int( 0.5*(filtr_shape[i_dim] - 1) )
        padR = padL
        pad_idim = (padL,padR)
        pad.append(pad_idim)      
            
    inpt_pad = periodic_padding(inpt, pad)
    output = tf.nn.conv3d(inpt_pad, filtr, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    
    return output


def conv3d(inpt, f, output_channels, s, use_bias=False, scope='conv', name=None):
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


def filter3d(inpt, scope='filter', name=None):
    inpt_shape = inpt.get_shape().as_list()
    with tf.variable_scope(scope):
        filter1D = tf.constant([0.04997364, 0.13638498, 0.20002636, 0.22723004, 0.20002636, 0.13638498, 0.04997364], dtype=tf.float32)
        filter1Dx = tf.reshape(filter1D, shape=(-1,1,1))
        filter1Dy = tf.reshape(filter1D, shape=(1,-1,1))
        filter1Dz = tf.reshape(filter1D, shape=(1,1,-1))
        filter3D = filter1Dx * filter1Dy * filter1Dz # Tensor product 3D filter using broadcasting

        filter3D = tf.expand_dims(filter3D, axis=3)
        zero = tf.constant( np.zeros((7,7,7,1), dtype=np.float32) )
        filter3Du = tf.concat( [filter3D, zero, zero, zero], axis=3 )
        filter3Dv = tf.concat( [zero, filter3D, zero, zero], axis=3 )
        filter3Dw = tf.concat( [zero, zero, filter3D, zero], axis=3 )
        filter3Dp = tf.concat( [zero, zero, zero, filter3D], axis=3 )
        filter3D = tf.stack( [filter3Du, filter3Dv, filter3Dw, filter3Dp], axis=4, name='filter' )

    strides = [1,4,4,4,1]
    inpt_pad = periodic_padding( inpt, ((3,3),(3,3),(3,3)) )
    output = tf.nn.conv3d(inpt_pad, filter3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    
    return output


def ddx(inpt, channel, dx, scope='ddx', name=None):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    with tf.variable_scope(scope):
        ddx1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
        ddx3D = tf.reshape(ddx1D, shape=(-1,1,1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding( var, ((3,3),(0,0),(0,0)) )
    output = tf.nn.conv3d(var_pad, ddx3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dx, output)
    
    return output


def ddy(inpt, channel, dy, scope='ddy', name=None):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    with tf.variable_scope(scope):
        ddy1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
        ddy3D = tf.reshape(ddy1D, shape=(1,-1,1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding( var, ((0,0),(3,3),(0,0)) )
    output = tf.nn.conv3d(var_pad, ddy3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dy, output)
    
    return output


def ddz(inpt, channel, dz, scope='ddz', name=None):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    with tf.variable_scope(scope):
        ddz1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
        ddz3D = tf.reshape(ddz1D, shape=(1,1,-1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding( var, ((0,0),(0,0),(3,3)) )
    output = tf.nn.conv3d(var_pad, ddz3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dz, output)
    
    return output


def d2dx2(inpt, channel, dx, scope='d2dx2', name=None):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    with tf.variable_scope(scope):
        ddx1D = tf.constant([1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.], dtype=tf.float32)
        ddx3D = tf.reshape(ddx1D, shape=(-1,1,1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding( var, ((3,3),(0,0),(0,0)) )
    output = tf.nn.conv3d(var_pad, ddx3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dx**2, output)
    
    return output


def d2dy2(inpt, channel, dy, scope='d2dy2', name=None):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    with tf.variable_scope(scope):
        ddy1D = tf.constant([1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.], dtype=tf.float32)
        ddy3D = tf.reshape(ddy1D, shape=(1,-1,1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding( var, ((0,0),(3,3),(0,0)) )
    output = tf.nn.conv3d(var_pad, ddy3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dy**2, output)
    
    return output


def d2dz2(inpt, channel, dz, scope='d2dz2', name=None):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    with tf.variable_scope(scope):
        ddz1D = tf.constant([1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.], dtype=tf.float32)
        ddz3D = tf.reshape(ddz1D, shape=(1,1,-1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding( var, ((0,0),(0,0),(3,3)) )
    output = tf.nn.conv3d(var_pad, ddz3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dz**2, output)
    
    return output


def get_TKE(inpt, name='TKE'):
    with tf.name_scope(name):
        TKE = tf.square( inpt[:,:,:,0] )
        TKE = TKE + tf.square( inpt[:,:,:,1] )
        TKE = TKE + tf.square( inpt[:,:,:,2] )
        TKE = 0.5*TKE
        TKE = tf.expand_dims(TKE, axis=4)

    return TKE


def get_velocity_grad(inpt, dx, dy, dz, scope='vel_grad', name=None):
    with tf.variable_scope(scope):
        dudx = ddx(inpt, 0, dx, scope='dudx')
        dudy = ddy(inpt, 0, dy, scope='dudy')
        dudz = ddz(inpt, 0, dz, scope='dudz')

        dvdx = ddx(inpt, 1, dx, scope='dvdx')
        dvdy = ddy(inpt, 1, dy, scope='dvdy')
        dvdz = ddz(inpt, 1, dz, scope='dvdz')

        dwdx = ddx(inpt, 2, dx, scope='dwdx')
        dwdy = ddy(inpt, 2, dy, scope='dwdy')
        dwdz = ddz(inpt, 2, dz, scope='dwdz')

    return dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz


def get_strain_rate_mag2(vel_grad, scope='strain_rate_mag', name=None):
    dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz = vel_grad

    strain_rate_mag2 = dudx**2 + dvdy**2 + dwdz**2 \
                     + 2*( (0.5*(dudy + dvdx))**2 + (0.5*(dudz + dwdx))**2 + (0.5*(dvdz + dwdy))**2 )

    return strain_rate_mag2


def get_vorticity(vel_grad, scope='vorticity', name=None):
    dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz = vel_grad
    vort_x = dwdy - dvdz
    vort_y = dudz - dwdx
    vort_z = dvdx - dudy
    return vort_x, vort_y, vort_z

def get_enstrophy(vorticity, name='enstrophy'):
    omega_x, omega_y, omega_z = vorticity

    with tf.name_scope(name):
        Omega = omega_x**2 + omega_y**2 + omega_z**2

    return Omega

def get_continuity_residual(vel_grad, name='continuity'):

    dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz = vel_grad
    with tf.name_scope(name):
        res = dudx + dvdy + dwdz

    return res


def get_pressure_residual(inpt, vel_grad, dx, dy, dz, scope='pressure'):

    dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz = vel_grad

    with tf.variable_scope(scope):
        d2pdx2 =d2dx2(inpt, 3, dx)
        d2pdy2 =d2dy2(inpt, 3, dy)
        d2pdz2 =d2dz2(inpt, 3, dz)

        res = (d2pdx2 + d2pdy2 + d2pdz2)
        res = res + dudx*dudx + dvdy*dvdy + dwdz*dwdz \
               + 2*(dudy*dvdx + dudz*dwdx + dvdz*dwdy)

    return res


def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha',inputs.get_shape()[-1],
                                 initializer=tf.zeros_initializer(),dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg


def lrelu(inputs, alpha):
    return tf.keras.layers.LeakyReLU(alpha=alpha).call(inputs)


def denselayer(inputs, output_size):
    output = tf.layers.dense(inputs, output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return output


def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs,decay=0.9,epsilon=0.001,
                           updates_collections=tf.GraphKeys.UPDATE_OPS,
                           scale=False,fused=True,is_training=is_training)


def print_configuration_op(FLAGS):
    print('[Configurations]:')
    a = FLAGS.mode
    #pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        if type(value) == float:
            print('\t%s: %f'%(name, value))
        elif type(value) == int:
            print('\t%s: %d'%(name, value))
        elif type(value) == str:
            print('\t%s: %s'%(name, value))
        elif type(value) == bool:
            print('\t%s: %s'%(name, value))
        else:
            print('\t%s: %s' % (name, value))

    print('End of configuration')

                                   
def phaseShift(inputs, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 4, 2, 5, 3, 6])

    return tf.reshape(X, shape_2)


# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
    # size = tf.shape(inputs)
    size = inputs.get_shape().as_list()
    batch_size = size[0]
    d = size[1]
    h = size[2]
    w = size[3]
    c = size[4]

    # Get the target channel size
    channel_target = c // (scale * scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, d, h, w, scale, scale, scale]
    shape_2 = [batch_size, d * scale, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=4)
    output = tf.concat([phaseShift(x, shape_1, shape_2) for x in input_split], axis=4)

    return output


def convert_to_rgba(a, vmin, vmax,  cmap=plt.cm.viridis):
    rgba = cmap( (a - vmin)/(vmax - vmin) )
    return rgba


def get_slice_images(HR, LR, out, n_images=1):

    batch_size, nx, ny, nz, nvars = HR.shape

    grid_size = [nx, ny, nz]
    images = []

    for i in range(n_images):
        batch = np.random.randint(batch_size, size=1)[0]
        var   = np.random.randint(nvars-1, size=1)[0]
        plane = np.random.randint(3, size=1)[0]
        index = np.random.randint(grid_size[plane]//4, size=1)[0]
        print("batch = {}, var = {}, plane = {}, index = {}".format(batch, var, plane, index))

        if plane == 0:
            var_HR  =  HR[batch, index*4, :, :, var]
            var_LR  =  LR[batch, index,   :, :, var]
            var_out = out[batch, index*4, :, :, var]
        elif plane == 1:
            var_HR  =  HR[batch, :, index*4, :, var]
            var_LR  =  LR[batch, :, index,   :, var]
            var_out = out[batch, :, index*4, :, var]
        elif plane == 2:
            var_HR  =  HR[batch,  :, :,index*4, var]
            var_LR  =  LR[batch,  :, :,index,   var]
            var_out = out[batch,  :, :,index*4, var]
        else:
            raise ValueError('Plane has to be 0, 1 or 2. Given {}'.format(plane))

        vmin = var_HR.min() - 1.e-10
        vmax = var_HR.max() + 1.e-10

        # Repeat values to make it 64^3
        var_LR = np.repeat(var_LR, 4, axis=0)
        var_LR = np.repeat(var_LR, 4, axis=1)

        im_HR  = convert_to_rgba(var_HR,  vmin, vmax)
        im_LR  = convert_to_rgba(var_LR,  vmin, vmax)
        im_out = convert_to_rgba(var_out, vmin, vmax)

        im = np.concatenate((im_HR, im_LR, im_out), axis=1)

        images.append( im )

    return np.stack(images, axis=0)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
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

    Xphase = tf.placeholder(tf.float32, shape=(1, 4, 4, 4, 4))
    shape_1 = [1, 4, 4, 4, 2, 2, 1]
    shape_2 = [1, 8, 8, 4, 1]
    X_ups = phaseShift(Xphase, shape_1, shape_2)

    with tf.Session() as sess:
        xphase = np.zeros((1,4,4,4,4))
        for i in range(4):
            xphase[:,:,:,:,i] = i
        xups = sess.run( X_ups, feed_dict={ Xphase: xphase } )

    print(xups[0,:,:,0,0])
    print(xups.shape)


    HR = tf.placeholder(tf.float32, shape=(2,16,16,16,4))
    LR = filter3d(HR)

    with tf.Session() as sess:
        hr = np.zeros( (2,16,16,16,4), dtype=np.float32 )
        for i in range(4):
            hr[:,:,:,:,i] = i

        lr = sess.run(LR, feed_dict={HR: hr} )
        print(lr[0,:,:,:,3])
        print(lr.shape)


    var = tf.placeholder(tf.float32, shape=(2,16,16,16,4))
    dudx =  ddx(var, 0, 2.*np.pi/16.)
    dudy =  ddy(var, 0, 2.*np.pi/16.)
    dudz =  ddz(var, 0, 2.*np.pi/16.)
    d2udx2 =  d2dx2(var, 0, 2.*np.pi/16.)
    d2udy2 =  d2dy2(var, 0, 2.*np.pi/16.)
    d2udz2 =  d2dz2(var, 0, 2.*np.pi/16.)

    with tf.Session() as sess:
        x = np.linspace(0,2.*np.pi,num=16+1)[:-1].reshape((16,1,1)).repeat(16, axis=1).repeat(16, axis=2)
        y = np.linspace(0,2.*np.pi,num=16+1)[:-1].reshape((1,16,1)).repeat(16, axis=0).repeat(16, axis=2)
        z = np.linspace(0,2.*np.pi,num=16+1)[:-1].reshape((1,1,16)).repeat(16, axis=0).repeat(16, axis=1)

        v = np.zeros( (2,16,16,16,4) )
        v[0,:,:,:,0] = np.sin(x)
        v[1,:,:,:,0] = np.cos(x)

        dv = np.zeros( (2,16,16,16,1) )
        dv[0,:,:,:,0] = np.cos(x)
        dv[1,:,:,:,0] = -np.sin(x)

        dv2 = np.zeros( (2,16,16,16,1) )
        dv2[0,:,:,:,0] = -np.sin(x)
        dv2[1,:,:,:,0] = -np.cos(x)

        du, du2 = sess.run( [dudx, d2udx2], feed_dict={var: v})
        print(du.shape)
        print("Max X derivative error = {}".format( np.absolute(du-dv).max()))
        print("Max X 2nd derivative error = {}".format( np.absolute(du2-dv2).max()))

        v[0,:,:,:,0] = np.sin(y)
        v[1,:,:,:,0] = np.cos(y)

        dv[0,:,:,:,0] = np.cos(y)
        dv[1,:,:,:,0] = -np.sin(y)

        dv2[0,:,:,:,0] = -np.sin(y)
        dv2[1,:,:,:,0] = -np.cos(y)

        du, du2 = sess.run( [dudy, d2udy2], feed_dict={var: v})
        print(du.shape)
        print("Max Y derivative error = {}".format( np.absolute(du-dv).max()))
        print("Max Y 2nd derivative error = {}".format( np.absolute(du2-dv2).max()))

        v[0,:,:,:,0] = np.sin(z)
        v[1,:,:,:,0] = np.cos(z)

        dv[0,:,:,:,0] = np.cos(z)
        dv[1,:,:,:,0] = -np.sin(z)

        dv2[0,:,:,:,0] = -np.sin(z)
        dv2[1,:,:,:,0] = -np.cos(z)

        du, du2 = sess.run( [dudz, d2udz2], feed_dict={var: v})
        print(du.shape)
        print("Max Z derivative error = {}".format( np.absolute(du-dv).max()))
        print("Max Z 2nd derivative error = {}".format( np.absolute(du2-dv2).max()))


        for i in range(4):
            v[0,:,:,:,i] = np.sin(3*z)
            v[1,:,:,:,i] = np.cos(3*z)

        images = get_slice_images(v, v[:,::4,::4,::4,:], v, n_images=1)

        print(images.shape)
