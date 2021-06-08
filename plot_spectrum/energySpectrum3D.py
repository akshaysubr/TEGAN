import numpy
import pyfftw

class energySpectrum3D(object):
    """
    Class to get the energy spectrum (not density weighted) from the physical space velocities
    """

    def __init__(self, nx, ny, nz):
        """
        Constructor for the energySpectrum class
        """

        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        u    = pyfftw.n_byte_align_empty((nx, ny, nz), 64, dtype='float64', order='C')
        self.uhat = pyfftw.n_byte_align_empty((nx, ny, nz//2+1), 64, dtype='complex128', order='C')
        self.vhat = pyfftw.n_byte_align_empty((nx, ny, nz//2+1), 64, dtype='complex128', order='C')
        self.what = pyfftw.n_byte_align_empty((nx, ny, nz//2+1), 64, dtype='complex128', order='C')

        # Create the FFTW objects for forward and backward transforms
        self.fft  = pyfftw.FFTW(u,self.uhat,axes=(0,1,2),direction='FFTW_FORWARD',  flags=('FFTW_MEASURE', ), threads=24, planning_timelimit=None)
        self.ifft = pyfftw.FFTW(self.uhat,u,axes=(0,1,2),direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', ), threads=24, planning_timelimit=None)

        # Create the wavenumbers in physical space (assuming domain length is 2 \pi)
        self.kx = numpy.hstack((numpy.arange(nx//2),-numpy.arange(1,nx//2+1)[::-1])).reshape((nx,1,1)).repeat(ny,axis=1).repeat(nz//2+1,axis=2)
        self.ky = numpy.hstack((numpy.arange(ny//2),-numpy.arange(1,ny//2+1)[::-1])).reshape((1,ny,1)).repeat(nx,axis=0).repeat(nz//2+1,axis=2)
        self.kz = numpy.arange(nz//2+1, dtype=float).reshape((1,1,nz//2+1)).repeat(nz,axis=0).repeat(nz,axis=1)

        self.kabs = numpy.sqrt(self.kx**2 + self.ky**2 + self.kz**2)

    def getEnergySpectrum3D(self, u, v, w, nbins=None, log_binning=False):
        """
        Method to get the 3D isotropic energy spectrum for the given velocities
        """
        shape = (self.nx, self.ny, self.nz)
        if not isinstance(u, numpy.ndarray) or u.shape != shape:
            raise ValueError("Array u is not of the right shape!")
        if not isinstance(v, numpy.ndarray) or v.shape != shape:
            raise ValueError("Array v is not of the right shape!")
        if not isinstance(w, numpy.ndarray) or w.shape != shape:
            raise ValueError("Array w is not of the right shape!")

        # Get 3D Fourier transform of the velocities
        self.fft(input_array=u, output_array=self.uhat)
        self.fft(input_array=v, output_array=self.vhat)
        self.fft(input_array=w, output_array=self.what)

        # Scale 3D Fourier transforms to make them physical
        self.uhat = self.uhat / (self.nx*self.ny*self.nz)
        self.vhat = self.vhat / (self.ny*self.ny*self.nz)
        self.what = self.what / (self.nz*self.ny*self.nz)
        
        #u_rms_hat = ( (numpy.absolute(self.uhat)**2).sum() + (numpy.absolute(self.uhat[:,:,1:-1])**2).sum() )
        #print "Parseval's theorem: Energy in uhat %g == %g Energy in u" %( u_rms_hat, (u*u).mean() )

        ek = numpy.absolute(self.uhat*self.uhat + self.vhat*self.vhat + self.what*self.what)
        # ek[:,:,1:-1] = 2.*ek[:,:,1:-1]

        k_bins_max = numpy.sqrt(self.nx*self.nx + self.ny*self.ny + self.nz*self.nz)/2
        nbins_ = int(k_bins_max)/2
        if nbins == None:
            k_bins_max = nbins_ * 2
        else:
            nbins_ = nbins

        # Define bins in wavenumber space for binning data
        if log_binning:
            k_bins = numpy.logspace(0, numpy.log10(k_bins_max), num=nbins_)
            k_bins = numpy.hstack(([0.], k_bins))
        else:
            k_bins = numpy.linspace(0, k_bins_max, num=int(nbins_)+1)
        
        # Get the energy spectrum
        Ek, k_edges = numpy.histogram(self.kabs, bins=k_bins, density=False, weights=ek)
        k = 0.5*(k_edges[:-1] + k_edges[1:])

        #dk = k_edges[1:] - k_edges[:-1]
        #print "u_rms^2 from spectrum = ", (Ek*dk).sum()
        #print "u_rms^2 from data = ", (u*u+v*v+w*w).mean()

        return k, Ek

    def getScalarSpectrum3D(self, u, nbins=None, log_binning=False):
        """
        Method to get the 3D isotropic energy spectrum for the given scalar
        """
        shape = (self.nx, self.ny, self.nz)
        if not isinstance(u, numpy.ndarray) or u.shape != shape:
            raise ValueError("Array u is not of the right shape!")

        # Get 3D Fourier transform of the scalar
        self.fft(input_array=u, output_array=self.uhat)

        # Scale 3D Fourier transforms to make them physical
        self.uhat = self.uhat / (self.nx*self.ny*self.nz)
        
        ek = numpy.absolute(self.uhat*self.uhat)

        k_bins_max = numpy.sqrt(self.nx*self.nx + self.ny*self.ny + self.nz*self.nz)/2.0
        nbins_ = int(k_bins_max)/2
        if nbins == None:
            k_bins_max = nbins_ * 2
        else:
            nbins_ = nbins


        # Define bins in wavenumber space for binning data
        if log_binning:
            k_bins = numpy.logspace(0, numpy.log10(k_bins_max), num=nbins_)
            k_bins = numpy.hstack(([0.], k_bins))
        else:
            k_bins = numpy.linspace(0, k_bins_max, num=int(nbins_)+1)
        
        # Get the energy spectrum
        Ek, k_edges = numpy.histogram(self.kabs, bins=k_bins, density=False, weights=ek)
        k = 0.5*(k_edges[:-1] + k_edges[1:])

        return k, Ek
