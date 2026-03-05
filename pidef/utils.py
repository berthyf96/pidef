# Adapted from bhnerf (https://github.com/aviadlevis/bhnerf)
# Original authors: Aviad Levis et al.

import h5py
import io
import math
from PIL import Image

import ehtim as eh
import ehtim.const_def as ehc
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray as xr

from pidef import constants
import pidef.emission

mse = lambda true, est: float(np.mean((true - est)**2))

psnr = lambda true, est: float(10.0 * np.log10(np.max(true)**2 / mse(true, est)))

normalize = lambda vector: vector / np.sqrt(np.dot(vector, vector))

def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)


def linspace_xr(num, start=-0.5, stop=0.5, endpoint=True, units='unitless'):
    """
    Return a DataArray with coordinates spaced over a specified interval in N-dimensions.

    Parameters
    ----------
    num: int or tuple
        Number of grid points in 1D (x) or 2D (x, y) or 3D (x, y, z). 
    start: float
        starting grid point (included in the grid)
    stop: float
        ending grid point (optionally included in the grid)
    endpoint: bool
        Optionally include the stop points in the grid.
    units: str, default='unitless'
        Store the units of the underlying grid.

    Returns
    -------
    grid: xr.DataArray
        A DataArray with coordinates linearly spaced over the desired interval
    """
    dimensions = ['x', 'y', 'z']
    num = np.atleast_1d(num)
    coords = {}
    for i, n in enumerate(num):
        coord = np.linspace(start, stop, n, endpoint=endpoint)
        coords[dimensions[i]] = coord
    grid = xr.Dataset(coords=coords)
    for dim in grid.dims:
        grid[dim].attrs.update(units=units)
    return grid

def gaussian_xr(resolution, center, std,  fov=(1.0, 'unitless'), std_clip=np.inf):
    """
    Generate a Gaussian image as xarray.DataArray.

    Parameters
    ----------
    resolution: int or nd-array,
            Number of (x,y,z)-axis grid points.
    center: int or nd-array,
        Center of the gaussian in coordinates ('x', 'y', 'z')
    std: (stdx, stdy, stdz), or float,
        Gaussian standard deviation in x,y,z directions. If scalar specified isotropic std is used.
    fov: (float, str), default=(1.0, 'unitless')
        Field of view and units. Default is unitless 1.0.
    std_clip: float, default=np.inf
        Clip after this number of standard deviations

    Returns
    -------
    emission: xr.DataArray,
        A DataArray with Gaussian emission.
    """
    if np.isscalar(std): std = (std, std, std)
    if len(resolution) != len(center): raise AttributeError('resolution and center should have same length {} != {}'.format(
        len(resolution), len(center)))
    grid = linspace_xr(resolution, start=-fov[0]/2.0, stop=fov[0]/2.0, units=fov[1])
    if 'x' in grid.dims and 'y' in grid.dims and 'z' in grid.dims:
        data = np.exp(-0.5*( ((grid.x - center[0])/std[0])**2 + ((grid.y - center[1])/std[1])**2 + ((grid.z - center[2])/std[2])**2 ))
        dims = ['x', 'y', 'z']
    elif 'x' in grid.dims and 'y' in grid.dims:
        data = np.exp(-0.5*( ((grid.y - center[1])/std[1])**2 + ((grid.x - center[0])/std[0])**2 ))
        dims = ['y', 'x']
    else:
        raise AttributeError

    threshold = np.exp(-0.5 * std_clip ** 2)
    emission = xr.DataArray(
        name='emission',
        data=data.where(data > threshold).fillna(0.0),
        coords=grid.coords,
        dims=dims,
        attrs={
            'fov': fov,
            'std': std,
            'center': center,
            'std_clip': std_clip
        })
    return emission

def rotation_matrix(axis, angle, use_jax=False):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis
    
    Parameters
    ----------
    axis: list or np.array, dim=3
        Axis of rotation
    angle: float or numpy array of floats,
        Angle of rotation in radians
    use_jax: bool, default=False
        Compuatations using jax.
        
    Returns
    -------
    rotation_matrix: np.array(shape=(3,3,...)),
        A rotation matrix. If angle is a numpy array additional dimensions are stacked at the end.
        
    References
    ----------
    [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    [2] https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    _np = jnp if use_jax else np
    
    axis = _np.array(axis)
    axis = axis / _np.sqrt(_np.dot(axis, axis))
    
    a = _np.cos(angle / 2.0)
    b, c, d = _np.stack([-ax * _np.sin(angle / 2.0) for ax in axis])
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return _np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def spherical_coords_to_rotation_axis(theta, phi):
    """
    Transform the spherical coordinates into a rotation axis and angle
    
    Parameters
    ----------
    theta: float,
        zenith angle (rad)
    phi: float,
        azimuth angle (rad)
        
    Returns
    -------
    rot_axis: 3-vector,
        Rotation axis.
    rot_angle: float, 
        Rotation angle about the rot_axis.
    """
    z_axis = np.array([0, 0, 1])
    r_vector = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    rot_axis_prime = np.cross(r_vector, z_axis)
    rot_matrix = rotation_matrix(rot_axis_prime,  np.pi/2)
    rot_axis = np.matmul(rot_matrix, r_vector)
    rot_angle = phi
    return rot_axis, rot_angle


def world_to_image_coords(coords, fov, npix, use_jax=False):
    _np = jnp if use_jax else np
    image_coords = []
    for i in range(coords.shape[-1]):
        image_coords.append((coords[...,i] + fov[i]/2.0) / fov[i] * (npix[i] - 1))
    image_coords = _np.stack(image_coords, axis=-1)
    return image_coords


def intensity_to_nchw(intensity, cmap='viridis', gamma=0.5):
    """
    Utility function to converent a grayscale image to NCHW image (for tensorboard logging).
       N: number of images in the batch
       C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)
       H: height of the image
       W: width of the image

    Parameters
    ----------
    intensity: array,
         Grayscale intensity image.
    cmap : str, default='viridis'
        A registered colormap name used to map scalar data to colors.
    gamma: float, default=0.5
        Gamma correction term
        
    Returns
    -------
    nchw_images: array, 
        Array of images.
    """
    cm = plt.get_cmap(cmap)
    norm_images = ( (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity)) )**gamma
    nchw_images = np.moveaxis(cm(norm_images)[...,:3], (0, 1, 2, 3), (3, 2, 0, 1))
    return nchw_images


def anti_aliasing_filter(image_plane, window):
    """
    Anti-aliasing flitering / blurring
    
    Parameters
    ----------
    image_plane: np.array,
        2D image or 3D movie (frames are in the first index)
    window: np.array
        2D image used for anti-aliasing filtering
    
    Returns
    -------
    image_plane: np.array,
        2D image or 3D movie (frames are in the first index)
    """
    fourier = jnp.fft.fft2(jnp.fft.ifftshift(image_plane, axes=(-2, -1))) * jnp.fft.fft2(jnp.fft.ifftshift(window))
    image_plane = jnp.fft.ifftshift(jnp.fft.ifft2(fourier), axes=(-2, -1)).real
    return image_plane


def expand_dims(x, ndim, axis=0, use_jax=False):
    _np = jnp if use_jax else np
    for i in range(ndim-_np.array(x).ndim):
        x = _np.expand_dims(x, axis=min(axis, _np.array(x).ndim))
    return x


def expand_3d(movie, fov_z, H_r=0.05, std=0.2, std_clip=3, nz=64):
    """
    Expand a 2D movie into 3D (movie) with some H/r
    
    Parameters
    ----------
    movie: np.array,
        2D movie (frames are in the first index)
    fov_z: float, 
        field of view (M) in z axis
    H_r: float, default=0.05
        tangent of expansion with increasing radius (Height/radius)
    std: float, default=0.2
        If H_r is zero then a constant Z width is used.
    std_clip: float, default=3,
        Clip values after this amount of stds in Z axis
    nz: int, default=64,
        The grid resulution in Z
    
    Returns
    -------
    emission: np.array,
        3D movie with emission values
    """
    emission = movie.expand_dims(z=np.linspace(-fov_z/2, fov_z/2, nz), axis=-1).transpose('t', 'x', 'y', 'z')
    H = H_r * np.sqrt(emission.x**2 + emission.y**2)
    if H_r == 0: H = std
    gaussian = np.exp(-0.5*(emission.z)**2 / H**2).transpose(..., 'z')
    gaussian.where(gaussian > np.exp(-0.5 * std_clip ** 2)).fillna(0.0)
    emission = emission * gaussian
    return emission


def next_power_of_two(x):
    """
    Find the next greatest power of two

    Parameters
    ----------
    x: int,
        Input integer

    Returns
    -------
    y: int
       Next greatest power of two
    """
    y = 2 ** (math.ceil(math.log(x, 2)))
    return y


def fft_transform(movies, fft_pad_factor=2):
    """
    Fast Fourier transform of one or several movies.
    Fourier is done per each time slice on image dimensions

    Parameters
    ----------
    fft_pad_factor: float, default=2
        A padding factor for increased fft resolution.

    Returns
    -------
    fft: array
        An array with the transformed signal.
    """
    ny, nx = movies.shape[-2:]
    npad = next_power_of_two(fft_pad_factor * np.max((nx, ny)))
    padvalx1 = padvalx2 = int(np.floor((npad - nx) / 2.0))
    padvaly1 = padvaly2 = int(np.floor((npad - ny) / 2.0))
    padvalx2 += 1 if nx % 2 else 0
    padvaly2 += 1 if ny % 2 else 0
    pad_width = [(0,0)] * (movies.ndim - 2) + [(padvaly1, padvaly2), (padvalx1, padvalx2)]
    padded_movies = np.pad(movies, pad_width, constant_values=0.0)

    # Compute visibilities (Fourier transform) of the entire block
    fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded_movies)))
    return fft


def fig_to_image(fig, dpi=120):
  """Convert a matplotlib figure to a PIL Image and return it."""
  buf = io.BytesIO()
  fig.savefig(buf, dpi=dpi)
  buf.seek(0)
  img = Image.open(buf)
  return img


def load_im_hdf5(filename):
    """Read in an image from an hdf5 file.
       Args:
            filename (str): path to input hdf5 file
       Returns:
            (Image): loaded image object
    """

    # Load information from hdf5 file

    hfp = h5py.File(filename,'r')
    dsource = hfp['header']['dsource'][()]          # distance to source in cm
    jyscale = hfp['header']['scale'][()]            # convert cgs intensity -> Jy flux density
    rf = hfp['header']['freqcgs'][()]               # in cgs
    tunit = hfp['header']['units']['T_unit'][()]    # in seconds
    lunit = hfp['header']['units']['L_unit'][()]    # in cm
    DX = hfp['header']['camera']['dx'][()]          # in GM/c^2
    nx = hfp['header']['camera']['nx'][()]          # width in pixels
    time = hfp['header']['t'][()] * tunit / 3600.       # time in hours
    if 'pol' in hfp:
        poldat = np.copy(hfp['pol'])[:, :, :4]            # NX,NY,{I,Q,U,V}
    else: # unpolarized data only
        unpoldat = np.copy(hfp['unpol'])                # NX,NY
        poldat = np.zeros(list(unpoldat.shape)+[4])
        poldat[:,:,0] = unpoldat
    hfp.close()

    # Correct image orientation
    # unpoldat = np.flip(unpoldat.transpose((1, 0)), axis=0)
    poldat = np.flip(poldat.transpose((1, 0, 2)), axis=0)

    # Make a guess at the source based on distance and optionally fall back on mass
    src = ehc.SOURCE_DEFAULT
    if dsource > 4.e25 and dsource < 6.2e25:
        src = "M87"
    elif dsource > 2.45e22 and dsource < 2.6e22:
        src = "SgrA"

    # Fill in information according to the source
    ra = ehc.RA_DEFAULT
    dec = ehc.DEC_DEFAULT
    if src == "SgrA":
        ra = 17.76112247
        dec = -28.992189444
    elif src == "M87":
        ra = 187.70593075
        dec = 12.391123306

    # Process image to set proper dimensions
    fovmuas = DX / dsource * lunit * 2.06265e11
    psize_x = ehc.RADPERUAS * fovmuas / nx

    Iim = poldat[:, :, 0] * jyscale
    Qim = poldat[:, :, 1] * jyscale
    Uim = poldat[:, :, 2] * jyscale
    Vim = poldat[:, :, 3] * jyscale

    outim = eh.image.Image(Iim, psize_x, ra, dec, rf=rf, source=src,
                              polrep='stokes', pol_prim='I', time=time)
    outim.add_qu(Qim, Uim)
    outim.add_v(Vim)

    return outim


def emission_dynamics(emission_0, geos, Omega, t_frames, t_injection,
                      J=1.0, t_start_obs=None, slow_light=True,
                      rot_axis=[0, 0, 1], M=constants.sgra_mass):
    """
    Compute the image-plane dynamics (movie) for a given initial emission and geodesics.
    
    Parameters
    ----------
    emission_0: np.array
        3D array with emission values
    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories) ending at the image plane.
    Omega: xr.DataArray
        A dataarray specifying the keplerian velocity field
    t_frames: array, 
        Array of time for each image frame with astropy.units
    t_injection: float, 
        Time of hotspot injection in M units.
    J: np.array(shape=(3,...)), default=None,
        Stokes polarization factors on the geodesic grid. None means no magnetic fields (non-polarized emission).
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    slow_light: bool, default=True
        Modeling the time it takes for the propogation of light.
    rot_axis: array, default=[0, 0, 1]
        Currently only equitorial plane rotation is supported
    M: astropy.Quantity, default=constants.sgra_mass,
        Mass of the black hole used to convert frame times to space-time times in units of M
        
    Returns
    -------
    images: np.array
        A movie array with image-plane frames. Polarization components are given along axis=1.
    """
    t_geos = geos.t if slow_light else 0.0
    warped_coords = pidef.emission.velocity_warp_coords(
        coords=[geos.x, geos.y, geos.z],
        Omega=Omega, 
        t_frames=t_frames, 
        t_start_obs=np.atleast_1d(t_frames)[0] if t_start_obs is None else t_start_obs, 
        t_geos=t_geos, 
        t_injection=t_injection, 
        rot_axis=rot_axis, 
        M=M  
    )

    if emission_0.ndim == 3:
        emission = pidef.emission.interpolate_coords(emission_0, warped_coords)
    
    # If emission_0 is already a movie
    elif emission_0.ndim == 4:
        emission = []
        for t in range(emission_0.shape[0]):
            emission.append(pidef.emission.interpolate_coords(emission_0[t], warped_coords))
        emission = np.array(emission)
        
    # Use magnetic fields for polarized synchrotron radiation
    if not np.isscalar(J):
        J = expand_dims(J, emission.ndim+1, 0)
        emission = J * expand_dims(emission, emission.ndim+1, 1)
        emission = np.squeeze(emission)

    return emission


def safe_arccos(inp, eps=1e-6):
    return jnp.arccos(jnp.clip(inp, -1. + eps, 1. - eps))


def safe_arctan2(x1, x2, eps=1e-6):
    return jnp.arctan2(x1 + eps, x2 + eps)


def cartesian_to_spherical(cartesian_coords):
    """Convert from Cartesian coords to spherical coords.
    
    Args:
        cartesian_coords: (nx, ny, nz, 3).
    
    Returns:
        Spherical coords array of shape (nx, ny, nz, 3).
    """
    x = cartesian_coords[..., 0]
    y = cartesian_coords[..., 1]
    z = cartesian_coords[..., 2]
    r2 = x**2 + y**2 + z**2
 
    safe_r2 = jnp.where(r2 > 0, r2, 0)
    r = jnp.where(r2 > 0, jnp.sqrt(safe_r2), 1e-6)
    theta = safe_arccos(z / r)
    phi = jnp.sign(y) * safe_arccos(x / jnp.sqrt(x**2 + y**2))
    # phi = safe_arctan2(y, x)
    # phi = jnp.mod(phi, 2 * jnp.pi)
    spherical_coords = jnp.stack((r, theta, phi), axis=-1)
    return spherical_coords


def spherical_to_cartesian(spherical_coords, safe=True):
    """Convert from spherical coords to Cartesian coords.
    
    Args:
        spherical_coords: (nx, ny, nz, 3).
    
    Returns:
        Cartesian coords array of shape (nx, ny, nz, 3).
    """
    r = spherical_coords[..., 0]
    if safe:
        r = jnp.clip(r, 0, None)
    theta = spherical_coords[..., 1]
    phi = spherical_coords[..., 2]
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    cartesian_coords = jnp.stack((x, y, z), axis=-1)
    return cartesian_coords


def get_grid_coords(rmax, resolution):
    x = np.linspace(-rmax, rmax, resolution)
    y = np.linspace(-rmax, rmax, resolution)
    z = np.linspace(-rmax, rmax, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    return np.stack((X, Y, Z), axis=-1)



def gaussian_blur(x, sigma=1.5, kernel_size=7):
    """
    Apply a 3D Gaussian blur along (H, W, D) to a tensor of shape (N, H, W, D).

    Args:
        x: jnp.ndarray, shape (N, H, W, D)
        sigma: standard deviation of the Gaussian
        kernel_size: spatial kernel size (odd integer recommended)

    Returns:
        jnp.ndarray of the same shape as x, blurred along spatial dims.
    """
    assert x.ndim == 4, "Expected shape (N, H, W, D)"
    N, H, W, D = x.shape

    # 1D Gaussian kernel
    coords = jnp.arange(kernel_size) - (kernel_size - 1) / 2
    k1 = jnp.exp(-0.5 * (coords / sigma) ** 2)
    k1 /= jnp.sum(k1)

    # 3D separable kernel
    k3 = k1[:, None, None] * k1[None, :, None] * k1[None, None, :]
    k3 = k3[..., None, None]  # [kh, kw, kd, in_ch, out_ch]

    # Add dummy channel dimension so we can use depthwise conv
    x_in = x[..., None]       # (N, H, W, D, 1)
    k3 = jnp.tile(k3, (1, 1, 1, 1, 1))  # in_ch=out_ch=1

    # 3D convolution
    x_blur = jax.lax.conv_general_dilated(
        x_in,
        k3,
        window_strides=(1, 1, 1),
        padding="SAME",
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
        feature_group_count=1,
    )

    return jnp.squeeze(x_blur, axis=-1)

