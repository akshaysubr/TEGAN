import numpy
import glob

def read_array_from_binary_file(filename, shape, dtype=numpy.float64, order='F'):
    array = numpy.fromfile(filename, dtype=dtype)
    if array.shape[0] != shape[0]*shape[1]*shape[2]:
        raise ValueError("Incorrect grid size %d x %d x %d" %shape)
    array = array.reshape(shape, order=order)
    return array


def get_grid_size(prefix, t_index):
    info_file = prefix + ('info_t%06d.out' %t_index)
    with open(info_file) as f:
        f.readline()
        nx = int( float( f.readline().strip() ) )
        ny = int( float( f.readline().strip() ) )
        nz = int( float( f.readline().strip() ) )

    return nx, ny, nz


def read_data( file_info ):
    prefix, t_index = file_info
    nx, ny, nz = get_grid_size(prefix, t_index)

    u_file = prefix + ('uVel_t%06d.out' %t_index)
    v_file = prefix + ('vVel_t%06d.out' %t_index)
    w_file = prefix + ('wVel_t%06d.out' %t_index)
    p_file = prefix + ('prss_t%06d.out' %t_index)

    u = read_array_from_binary_file(u_file, (nx, ny, nz))
    v = read_array_from_binary_file(v_file, (nx, ny, nz))
    w = read_array_from_binary_file(w_file, (nx, ny, nz))
    p = read_array_from_binary_file(p_file, (nx, ny, nz))

    return u, v, w, p


def get_all_datafile_info_in(directory):
    u_files = glob.glob(directory + '/*uVel_t*.out')

    files_info = []
    
    for u_file in u_files:
        prefix_end = u_file.find('uVel')
        ext_start = u_file.find('.out')
        prefix = u_file[:prefix_end]
        index = int(u_file[prefix_end+6:ext_start])
        files_info.append( (prefix, index) )
        
    return files_info
    