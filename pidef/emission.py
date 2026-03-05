# Adapted from bhnerf (https://github.com/aviadlevis/bhnerf)
# Original authors: Aviad Levis et al.

from astropy import units
import time

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from pidef import constants as consts
from pidef import kgeo
from pidef import utils


def generate_hotspot_xr(resolution, rot_axis, rot_angle, orbit_radius, std, fov, std_clip=np.inf, normalize=True):
    """
    Generate an emission hotspot as a Gaussian xarray.DataArray.

    Parameters
    ----------
    resolution: int or nd-array,
        Number of (x,y,z)-axis grid points.
    rot_axis: 3d array/list/tuple,
        The orbit rotation axis along which
    rot_angle: float, 
        The angle along the (2d) circular orbit. 
    orbit_radius: float, 
        Radius of the orbit.
    std: (stdx, stdy, stdz), or float,
        Gaussian standard deviation in x,y,z directions. If scalar specified isotropic std is used.
    r_isco: float,
        Radius of the inner most stable circular orbit (ISCO). 
    fov: (float, str), default=(1.0, 'unitless')
        Field of view and units. Default is unitless 1.0.
    std_clip: float, default=np.inf
        Clip after this number of standard deviations
    normalize: bool, default=True,
        If True, normalize the maximum flux to 1.0. 
        
    Returns
    -------
    emission: xr.DataArray,
        A DataArray with Gaussian emission.
    """
    center_2d = orbit_radius * np.array([np.cos(rot_angle), np.sin(rot_angle)])
    if len(resolution) == 2:
        center = center_2d
    else:
        rot_axis = np.array(rot_axis) 
        rot_axis = rot_axis / np.sqrt(np.sum(rot_axis**2))
        z_axis = np.array([0, 0, 1])
        rot_axis_prime = np.cross(z_axis, rot_axis)
        if np.sqrt(np.sum(rot_axis_prime**2)) < 1e-5: rot_axis_prime = z_axis 
        rot_angle_prime = np.arccos(np.dot(rot_axis, z_axis))
        rot_matrix = utils.rotation_matrix(rot_axis_prime, rot_angle_prime)
        center = np.matmul(rot_matrix, np.append(center_2d, 0.0))
        
    emission = utils.gaussian_xr(resolution, center, std, fov=fov, std_clip=std_clip)
    if normalize: emission /= emission.integrate(['x','y','z'])
    emission.attrs.update(
        rot_axis=rot_axis
    )
    return emission


def generate_tube_xr(resolution, rot_axis, phi_start, phi_end, orbit_radius, std, fov, std_clip=np.inf, normalize=True):
    """
    Generate an emission tube with a Gaussian profile as an xarray.DataArray.

    Parameters
    ----------
    resolution: int or nd-array,
        Number of (x,y,z)-axis grid points.
    rot_axis: 3d array/list/tuple,
        The orbit rotation axis along which
    phi_start: float, 
        The start angle for the tube.
    phi_end: float, 
        The end angle for the tube.
    orbit_radius: float, 
        Radius of the orbit.
    std: (stdx, stdy, stdz), or float,
        Gaussian standard deviation in x,y,z directions. If scalar specified isotropic std is used.
    r_isco: float,
        Radius of the inner most stable circular orbit (ISCO). 
    fov: (float, str), default=(1.0, 'unitless')
        Field of view and units. Default is unitless 1.0.
    std_clip: float, default=np.inf
        Clip after this number of standard deviations
    normalize: bool, default=True,
        If True, normalize the maximum flux to 1.0. 
        
    Returns
    -------
    emission: xr.DataArray,
        A DataArray with emission.
    """
    rot_axis = np.array(rot_axis) 
    rot_axis = rot_axis / np.sqrt(np.sum(rot_axis**2))
    z_axis = np.array([0, 0, 1])
    rot_axis_prime = np.cross(z_axis, rot_axis)
    if np.sqrt(np.sum(rot_axis_prime**2)) < 1e-5: rot_axis_prime = z_axis 
    rot_angle_prime = np.arccos(np.dot(rot_axis, z_axis))
    rot_matrix = utils.rotation_matrix(rot_axis_prime, rot_angle_prime)
        
    emission = 0
    angles = np.arange(phi_start, phi_end, 0.015)
    for phi in angles:
        center_2d = orbit_radius * np.array([np.cos(phi), np.sin(phi)]) 
        center = np.matmul(rot_matrix, np.append(center_2d, 0.0))
        emission += utils.gaussian_xr(resolution, center, std, fov=fov, std_clip=std_clip)
    if normalize: emission /= emission.integrate(['x','y','z'])
    emission.attrs.update(
        rot_axis=rot_axis,
        phi_start=phi_start, 
        phi_end=phi_end
    )
    return emission


def random_emission(fov_M, min_radius, max_radius, min_std=0.25, max_std=0.75,
                    rot_axis=[0., 0., 1.], resolution=(64, 64, 64),
                    num_hotspots=None, random_state=np.random.RandomState()):
    """Random emission generator."""
    # Number of emissions
    if num_hotspots is None:
        num_hotspots = random_state.choice(np.arange(1, 4), p=[0.6, 0.3, 0.1])

    emission = None
    for _ in range(num_hotspots):
        # Shared parameters
        orbit_radius = random_state.uniform(min_radius, max_radius)
        std = random_state.uniform(min_std, max_std)

        # Flip coin for Gaussian blob or tube.
        coin = random_state.randint(2)
        if coin == 0:
            # Gaussian blob
            rot_angle = random_state.uniform(0, 2 * np.pi)
            single_emission = generate_hotspot_xr(
                resolution=resolution, 
                rot_axis=rot_axis, 
                rot_angle=rot_angle,
                orbit_radius=orbit_radius,
                std=std,
                fov=(fov_M, 'GM/c^2')
            )
        elif coin == 1:
            # Tube
            phi_length = random_state.uniform(0, np.pi / 5)
            phi_start = random_state.uniform(0, 2 * np.pi - phi_length)
            phi_end = phi_start + phi_length
            single_emission = generate_tube_xr(
                resolution=resolution,
                rot_axis=rot_axis,
                phi_start=phi_start,
                phi_end=phi_end,
                orbit_radius=orbit_radius,
                std=std,
                fov=(fov_M, 'GM/c^2')
            )
        if emission is None:
            emission = single_emission
        else:
            emission += single_emission
    return emission


def random_emissions(t_frames, fov_M, min_radius, max_radius, min_std=0.25, max_std=0.75,
                     rot_axis=[0., 0., 1.], num_emissions=10, random_state=np.random.RandomState()):
    t_injection_list = np.linspace(t_frames[0].value, t_frames[-1].value, num_emissions, endpoint=False) * units.hr

    num_hotspots = 3 if num_emissions == 1 else None
    first_emission = random_emission(fov_M, min_radius, max_radius, min_std, max_std, rot_axis, num_hotspots=num_hotspots, random_state=random_state)

    emission_list = [first_emission]
    for _ in range(num_emissions - 1):
        new_emission = random_emission(fov_M, min_radius, max_radius, min_std, max_std, rot_axis, random_state=random_state)
        emission_list.append(new_emission)

    return emission_list, t_injection_list


def random_emissions_grmhdlike(t_frames, fov_M, min_radius, max_radius,
                               rot_axis=[0., 0., 1.], resolution=(64, 64, 64), random_state=np.random.RandomState()):
    x_domain = np.linspace(-max_radius, max_radius, 100)
    y_domain = np.linspace(-max_radius, max_radius, 100)
    x_grid, y_grid = np.meshgrid(x_domain, y_domain)

    r_grid = np.sqrt(x_grid**2 + y_grid**2)
    phi_grid = np.arctan2(y_grid, x_grid)

    # Probability of hotspot appearing at each radius:
    p_r = np.where(r_grid < min_radius, 0, 1 / np.power(r_grid, 2))

    # Probability of hotspot appearing at each angle:
    p_phi = 0.03

    r_domain = np.linspace(min_radius, max_radius, 100)
    phi_domain = np.linspace(0, 2 * np.pi, 100)
    r_grid, phi_grid = np.meshgrid(r_domain, phi_domain)

    t_injection_list = t_frames.copy()
    emission_list = []
    for _ in t_frames:
        # Sample whether or not a hotspot appears at each location.
        hotspot_at_r = random_state.binomial(1, p_r, r_grid.shape)
        hotspot_at_phi = random_state.binomial(1, p_phi, phi_grid.shape)
        hotspot_at_location = hotspot_at_r * hotspot_at_phi

        hotspot_rs = r_grid[hotspot_at_location == 1]
        hotspot_phis = phi_grid[hotspot_at_location == 1]

        # Generate all hotspots.
        # Start with an empty emission field.
        emission = utils.gaussian_xr(resolution, (0, 0, 0), std=0, fov=(fov_M, 'GM/c^2'))
        for r, phi in zip(hotspot_rs, hotspot_phis):
            std = 1 / (1 + np.exp(-1 * (r - 6)))
            std = np.clip(std, 0.2, 0.7)
            single_emission = generate_hotspot_xr(
                resolution=resolution, 
                rot_axis=rot_axis, 
                rot_angle=phi,
                orbit_radius=r,
                std=std,
                fov=(fov_M, 'GM/c^2')
            )
            emission += single_emission
        emission_list += [emission]

    return emission_list, t_injection_list


def interpolate_coords(emission, orig_coords, new_coords):
    """Interpolate 3D emission field along the new coordinates.

    Assumes that the emission field is defined on a regular grid. 
    
    Args:
        emission: 3D array with emission values, of shape (nx, ny, nz).
        orig_coords: original coordinates, of shape (nx, ny, nz, 3).
        new_coords: coordinates to interpolate to, of shape (nx, ny, nz, 3).
    """
    fov = [jnp.max(orig_coords[:, :, :, i]) - jnp.min(orig_coords[:, :, :, i]) for i in range(3)]
    npix = orig_coords.shape[:3]
    image_coords = utils.world_to_image_coords(new_coords, fov=fov, npix=npix, use_jax=True)
    image_coords = jnp.moveaxis(image_coords, -1, 0)
    return jax.scipy.ndimage.map_coordinates(emission, image_coords, order=1, cval=0.)


def interp_coords(emission, orig_points, new_coords):
    """Interpolate 3D emission field along the new coordinates."""
    interp = jax.scipy.interpolate.RegularGridInterpolator(
        orig_points, emission, method='linear', fill_value=0)
    return interp(new_coords)


def velocity_warp_coords(coords, Omega, t_frames, t_start_obs, t_geos, t_injection, rot_axis=[0,0,1], M=consts.sgra_mass, t_units=None, use_jax=False):
    """
    Generate an coordinate transoform for the velocity warp.
    
    Parameters
    ----------
    coords: list of np arrays
        A list of arrays with grid coordinates
    Omega: array, 
        Angular velocity array sampled along the coords points.
    t_frames: array, 
        Array of time for each image frame with astropy.units
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    t_geos: array, 
        Time along each geodesic (ray). This is used to account for slow light (light travels at finite velocity).
    t_injection: float, 
        Time of hotspot injection in M units.
    rot_axis: array, default=[0, 0, 1]
        Currently only equitorial plane rotation is supported
    M: astropy.Quantity, default=constants.sgra_mass,
        Mass of the black hole used to convert frame times to space-time times in units of M
    t_units: astropy.units, default=None,
        Time units. If None units are taken from t_frames.
    use_jax: bool, default=False,
        Using jax enables GPU accelerated computing.
        
    Returns
    -------
    warped_coords: array,
        An array with the new coordinates for the warp transformation.
    """
    _np = jnp if use_jax else np
    coords = _np.array(coords)
    Omega = _np.array(Omega)
    
    if isinstance(t_start_obs, units.Quantity):
        t_units = t_start_obs.unit
        t_start_obs = t_start_obs.value
    
    GM_c3 = 1.0  
    if t_units is not None:
        GM_c3 = consts.GM_c3(M).to(t_units).value

    if isinstance(t_frames, units.Quantity):
        t_frames = t_frames.to(t_units).value
    t_frames = _np.array(t_frames)

    if (_np.isscalar(Omega) or Omega.ndim == 0):
        Omega = utils.expand_dims(Omega, coords.ndim-1, axis=-1, use_jax=use_jax)

    # Extend the dimensions of `t_frames` and `coords' for an array of times 
    if not (t_frames.ndim == 0):
        coords = utils.expand_dims(coords, coords.ndim + t_frames.ndim, 1, use_jax)
        t_frames = utils.expand_dims(t_frames, t_frames.ndim + Omega.ndim, -1, use_jax)

    # Convert time units to grid units
    t_geos = (t_frames - t_start_obs)/GM_c3 + _np.array(t_geos)
    t_M = t_geos - t_injection
    
    # Insert nans for angles before the injection time
    theta_rot = _np.array(t_M * Omega)
    theta_rot = _np.where(t_M < 0.0, _np.full_like(theta_rot, fill_value=np.nan), theta_rot)

    inv_rot_matrix = utils.rotation_matrix(rot_axis, -theta_rot, use_jax=use_jax)
        
    warped_coords = _np.sum(inv_rot_matrix * coords, axis=1)
    warped_coords = _np.moveaxis(warped_coords, 0, -1)
    return warped_coords


def kgeo_velocity_warp_coords(init_coords, t_frames_M, t0_M, dt0_M, spin,
                              fac_subkep, beta_phi, beta_r, t_injection_M=0.,
                              M=1, retrograde=False):
    """Apply one update of velocity warp.

    If g_tt, g_phph, and g_tph are None, then a hard-coded 
    
    Args:
        init_coords: initial Cartesian coordinates, an array of shape (nx, ny, nz, 3).
        t_frames_M: list of times in units of M to evaluate.
        t0_M: initial time in units of M, a float.
        dt0_M: initial time step in units of M, a float.
        spin: spin of the black hole, a float.
        fac_subkep: factor for sub-Keplerian velocity, a float.
        beta_phi: factor for azimuthal velocity, a float.
        beta_r: factor for radial velocity, a float.
        t_injection_M: time of injection in units of M, a float.
    Returns: warped Cartesian coordinates, of shape (nx, ny, nz, 3).
    """
    def f(t, y, _):
        r = y[:, :, :, 0]
        theta = y[:, :, :, 1]
        # spherical_velocities = kgeo.spherical_velocities_kep(
        #     r, theta, spin, M, retrograde, infall)
        spherical_velocities = jax.lax.cond(
            t > t_injection_M,
            lambda rad: kgeo.spherical_velocities(
                rad, spin, theta=np.pi / 2, M=M,
                fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r,
                retrograde=retrograde),
            lambda _: jnp.zeros(r.shape + (3,)),
            r
        )
        return -spherical_velocities
    
    init_spherical_coords = utils.cartesian_to_spherical(init_coords)
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(f),
        diffrax.Dopri5(),
        t0=t0_M,
        t1=t_frames_M[-1],
        dt0=dt0_M,
        y0=init_spherical_coords,
        saveat=diffrax.SaveAt(ts=t_frames_M)
    )

    warped_spherical_coords = solution.ys

    # Convert back to Cartesian coordinates.
    warped_coords = jax.vmap(utils.spherical_to_cartesian)(warped_spherical_coords)
    # warped_coords = jnp.nan_to_num(warped_coords)
    return warped_coords


def sigmoidal_ramp(t_frames, t_injection, rate=100):
    return 1 / (1 + jnp.exp(-rate * (t_frames - t_injection)))


def propagate_emissions(emission_list, t_injection_list, geos, t_frames, coords,
                        fac_subkep, beta_phi, beta_r,
                        dt0=0.001 * units.hr, M=consts.sgra_mass, retrograde=False):
    GM_c3 = consts.GM_c3(M).to(t_frames.unit)

    emission_0 = emission_list[0]
    emission_points = (emission_0.coords['x'].data, emission_0.coords['y'].data, emission_0.coords['z'].data)

    warped_coords = kgeo_velocity_warp_coords(  # (nt, nx, ny, nz, 3)
        coords,
        t_frames_M=t_frames / GM_c3,
        t0_M=t_frames[0] / GM_c3,
        dt0_M=dt0 / GM_c3.value,
        spin=float(geos.spin),
        fac_subkep=fac_subkep,
        beta_phi=beta_phi,
        beta_r=beta_r,
        M=float(geos.M),  # TODO: should this be M or geos.M?
        retrograde=retrograde)

    n_emissions = len(emission_list)
    nt = len(t_frames)
    nx, ny, nz = warped_coords.shape[-4:-1]

    true_emission_per_t_list = np.zeros(
        (n_emissions, nt, nx, ny, nz), dtype=emission_0.dtype)
    for i, (t_injection, true_emission) in enumerate(zip(t_injection_list, emission_list)):
        # Offset `warped_coords` by `t_injection` so that the emission is injected at the correct time.
        num_still_frames = len(t_frames[t_frames < t_injection])
        num_moving_frames = len(t_frames) - num_still_frames
        warped_coords_offset = np.concatenate((
            np.tile(coords[None, ...], (num_still_frames, 1, 1, 1, 1)),
            warped_coords[:num_moving_frames]
        ))
        true_emission_per_t = interp_coords(
            true_emission.data, emission_points, warped_coords_offset)
        if t_injection > t_frames[0]:
            ramp = sigmoidal_ramp(t_frames, t_injection)
            true_emission_per_t = utils.batch_mul(true_emission_per_t, ramp)
        true_emission_per_t_list[i] = true_emission_per_t
    return np.sum(true_emission_per_t_list, axis=0)


def image_plane_dynamics(emission_list, t_injection_list, geos, t_frames,
                         fac_subkep, beta_phi, beta_r,
                         dt0=0.01 * units.hr, J=1.0, slow_light=False,
                         doppler=True, M=consts.sgra_mass, retrograde=False,
                         verbose=True):
    """
    Compute the image-plane dynamics (movie) for a given initial emission and geodesics.

    TODO: handle `slow_light=True`, in which case `t_injection` might not be 0.
    TODO: handle non equitorial plane rotation (i.e., `rot_axis` not [0, 0, 1]).

    Parameters
    ----------
    emission_list: list of np.array
        List of 3D arrays with emission values
    t_injection_list: list of float
        List of times of emission injection in time units.
    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories) ending at the image plane.
    t_frames: array, 
        Array of time for each image frame with astropy.units
    J: np.array(shape=(3,...)), default=None,
        Stokes polarization factors on the geodesic grid. None means no magnetic fields (non-polarized emission).
    doppler: bool, default=True
        Modeling doppler boosting.
    slow_light: bool, default=True
        Modeling the time it takes for the propogation of light.
    M: astropy.Quantity, default=constants.sgra_mass,
        Mass of the black hole used to convert frame times to space-time times in units of M
    infall: bool, default=True
        Whether to include infall inside the ISCO.

    Returns
    -------
    images: np.array
        A movie array with image-plane frames. Polarization components are given along axis=1.
    """
    if verbose:
        start_time = time.perf_counter()
        print('Simulating image-plane dynamics...', end=' ')

    if slow_light:
        raise NotImplementedError('slow_light=True not implemented')
    t_geos = geos.t if slow_light else 0.0

    geos_coords = jnp.stack((geos.x.data, geos.y.data, geos.z.data), axis=-1)

    emission = propagate_emissions(
        emission_list,
        t_injection_list,
        geos,
        t_frames,
        coords=geos_coords,
        fac_subkep=fac_subkep,
        beta_phi=beta_phi,
        beta_r=beta_r,
        dt0=dt0,
        M=M,
        retrograde=retrograde)
  
    g = 1.0
    if doppler:
        ut, ur, uth, uph = kgeo.u_general(
            geos.r.data, float(geos.spin), geos.theta.data, float(geos.M), fac_subkep, beta_phi, beta_r, retrograde)
        ut = ut.reshape(geos.r.shape)
        ur = ur.reshape(geos.r.shape)
        uth = uth.reshape(geos.r.shape)
        uph = uph.reshape(geos.r.shape)
        ut = xr.DataArray(ut, dims=['beta', 'alpha', 'geo'])
        ur = xr.DataArray(ur, dims=['beta', 'alpha', 'geo'])
        uth = xr.DataArray(uth, dims=['beta', 'alpha', 'geo'])
        uph = xr.DataArray(uph, dims=['beta', 'alpha', 'geo'])
        umu = xr.concat([ut, ur, uth, uph], dim='mu', coords='minimal')
        g = kgeo.doppler_factor(geos, umu)
        
    # Use magnetic fields for polarized synchrotron radiation
    if not np.isscalar(J):
        J = utils.expand_dims(J, emission.ndim+1, 0)
        emission = J * utils.expand_dims(emission, emission.ndim+1, 1)
        emission = np.squeeze(emission)
    images = kgeo.radiative_transfer(
        emission, np.array(g), np.array(geos.dtau), np.array(geos.Sigma))

    if verbose:
        print('done! {:.2f} seconds'.format(time.perf_counter() - start_time))

    return images


def propagate_flatspace_emission(emission_0, Omega_3D, t_frames,
                                 t_start_obs=None, rot_axis=[0,0,1], M=consts.sgra_mass):
    """
    Compute the 3D movie for a given initial emission in flat-space.
    
    Parameters
    ----------
    emission_0: np.array
        3D array with emission values
    Omega_3D: xr.DataArray
        A dataarray specifying the keplerian velocity field in flat-space 3D coords (NOT GEODESICS)
    t_frames: array, 
        Array of time for each image frame with astropy.units
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    rot_axis: array, default=[0, 0, 1]
        Currently only equitorial plane rotation is supported
    M: astropy.Quantity, default=constants.sgra_mass,
        Mass of the black hole used to convert frame times to space-time times in units of M
        
    Returns
    -------
    images: np.array
        A movie array with image-plane frames. Polarization components are given along axis=1.
    """
    x, y, z = np.meshgrid(emission_0.x, emission_0.y, emission_0.z, indexing='ij')
    warped_coords = velocity_warp_coords(
        coords=[x, y, z],
        Omega=Omega_3D,
        t_frames=t_frames,
        t_start_obs=np.atleast_1d(t_frames)[0] if t_start_obs is None else t_start_obs,
        t_geos=0,
        t_injection=0,
        rot_axis=rot_axis,
        M=M,
    )
    emission_coords = np.stack((x, y, z), axis=-1)
    emission_t = interpolate_coords(emission_0.data, emission_coords, warped_coords)
    return emission_t


def fill_unsupervised_emission(emission, coords, rmin=0, rmax=np.Inf, z_width=2.0, fill_value=0.0, use_jax=False):
    """
    Fill emission that is not within the supervision region
    
    Parameters
    ----------
    emission: np.array
        3D array with emission values
    coords: list of np.arrays
        Spatial coordinate arrays each shaped like emission
    rmin: float, default=0
        Zero values at radii < rmin
    rmax: float, default=np.inf
        Zero values at radii > rmax
    z_width: float, default=2,
        Maximum width of the disk (M units) 
    fill_value: float, default=0.0
        Fill value is default to zero 
    use_jax: bool, default=False,
        Using jax enables GPU accelerated computing.
        
    Returns
    -------
    emission: np.array
        3D array with emission values filled in
    """
    _np = jnp if use_jax else np
    r_sq = _np.sum(_np.array([_np.squeeze(x)**2 for x in coords]), axis=0)
    emission = _np.where(r_sq < rmin**2, _np.full_like(emission, fill_value=fill_value), emission)
    emission = _np.where(r_sq > rmax**2, _np.full_like(emission, fill_value=fill_value), emission)
    emission = _np.where(_np.abs(coords[2]) > z_width, _np.full_like(emission, fill_value=fill_value), emission)
    return emission


def fill_unsupervised(data, coords, rmin=0, rmax=np.Inf, z_width=2.0, fill_value=0.0, use_jax=False):
    """
    Fill emission that is not within the supervision region
    
    Parameters
    ----------
    data: np.array
        4D array with emission or velocity values
    coords: ndarray
        Spatial coordinate arrays each shaped like data, of shape (nx, ny, nz, 3).
    rmin: float, default=0
        Zero values at radii < rmin
    rmax: float, default=np.inf
        Zero values at radii > rmax
    z_width: float, default=2,
        Maximum width of the disk (M units) 
    fill_value: float, default=0.0
        Fill value is default to zero 
    use_jax: bool, default=False,
        Using jax enables GPU accelerated computing.
        
    Returns
    -------
    data: np.array
        4D array with unsupervised values filled in
    """
    _np = jnp if use_jax else np
    r2 = coords[..., 0]**2 + coords[..., 1]**2 + coords[..., 2]**2
    r = _np.sqrt(r2)
    z = coords[..., 2]

    # We assume that `data` has shape (nx, ny, nz, nc),
    # so we need to expan the dimensions of `coords` to `(nx, ny, nz, 1)`.
    r = _np.expand_dims(r, axis=-1)
    z = _np.expand_dims(z, axis=-1)

    data = _np.where(r > rmin, data, fill_value)
    data = _np.where(r < rmax, data, fill_value)
    data = _np.where(_np.abs(z) < z_width, data, fill_value)

    return data
