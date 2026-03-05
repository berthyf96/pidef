# Adapted from bhnerf (https://github.com/aviadlevis/bhnerf)
# Original authors: Aviad Levis et al.

from kgeo import *
import jax.numpy as jnp
import xarray as xr
import numpy as np
from pidef import constants as consts
from pidef import utils


def image_plane_geos(spin, inclination, alpha_range, beta_range, ngeo=100, 
                     num_alpha=64, num_beta=64, distance=1000.0, E=1.0, M=1.0, 
                     randomize_subpixel_rays=False, random_state=None, verbose=False):
    """
    Compute Kerr geodesics for the entire image plane 
    
    Parameters
    ----------
    spin: float
        normalized spin value in range [0,1]
    inclination: float, 
        inclination angle in [rad] in range [0, pi/2]
    alpha_range: tuple,
        Vertical extent (M)
    beta_range: tuple,
        Horizontal extent (M)
    ngeo: int, default=100
        Number of points along a ray
    num_alpha: int, default=64,
        Number of pixels in the vertical direction
    num_beta: int, default=64,
        Number of pixels in the horizontal direction   
    distance: float, default=1000.0
        Distance to observer
    E: float, default=1.0, 
        Photon energy at infinity 
    M: float, default=1.0, 
        Black hole mass, taken as 1 (G=c=M=1)
    randomize_subpixel_rays: bool, default=False,
        Radomize ray position within the image plane over the pixel size.
        If set to False, "center" position is taken.
        
    Returns
    -------
    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories) ending at the image plane.
        
    Notes
    -----
    Overleaf notes: https://www.overleaf.com/project/60ff0ece5aa4f90d07f2a417
    units are in GM/c^2
    """
    alpha_1d = np.linspace(*alpha_range, num_alpha)
    beta_1d = np.linspace(*beta_range, num_beta)

    if randomize_subpixel_rays:
        psize_alpha = (alpha_range[1]-alpha_range[0]) / (num_alpha-1)
        psize_beta = (beta_range[1]-beta_range[0]) / (num_beta-1)
        if random_state:
            alpha_1d += (random_state.random(num_alpha)-0.5) * psize_alpha
            beta_1d += (random_state.random(num_beta)-0.5) * psize_beta
        else:
            alpha_1d += (np.random.random(num_alpha)-0.5) * psize_alpha
            beta_1d += (np.random.random(num_beta)-0.5) * psize_beta

    alpha, beta = np.meshgrid(alpha_1d, beta_1d, indexing='ij')
    image_coords = [alpha.ravel(), beta.ravel()]
    
    observer_coords = [0, distance, inclination, 0]
    geos = raytrace_ana(spin, observer_coords, image_coords, ngeo, plotdata=False, verbose=verbose)
    geos = geos.get_dataset(num_alpha, num_beta, E, M)
    return geos


def wave_vector(geos): 
    """
    Compute the wave (photon momentum)  `k`
    
    Parameters
    ----------
    geos: xr.Dataset, 
        A dataset with the geodesic information and auxilary variables 

    Returns
    -------
    k_mu: np.array(shape=(...,4))
         First dimensions match geos dimensions.
         Last dimension is the spherical coordinate 4-vector: [k_t, k_r, k_th, k_ph]
    """
    # Plus-minus sign set according to angular (theta) and radial turning points 
    pm_r = np.sign(np.gradient(geos.r, axis=-1) / np.gradient(geos.affine, axis=-1))
    pm_th = np.sign(np.gradient(geos.theta, axis=-1) / np.gradient(geos.affine, axis=-1))

    # Global frame wave vector
    k_t  = -geos.E
    k_r  = geos.E * np.sqrt(geos.R.clip(min=0)) * pm_r / geos.Delta
    k_th = geos.E * np.sqrt(geos.Theta.clip(min=0)) * pm_th
    k_ph = geos.E * geos.lam
    k_mu = xr.concat([k_t, k_r, k_th, k_ph], dim='mu').transpose(...,'mu')
    return k_mu


def spacetime_metric(geos):
    """
    Spacetime metric g_{munu} (Boyer-Linquist coordinates)
    
    Parameters
    ----------
    geos: xr.Dataset, 
        A dataset with the geodesic information and auxiliary variables 
    
    Returns
    -------
    g_munu: xr.Dataset
        A dataset with the spacetime metric (non-zero components) 
        
    Notes
    -----
    g_munu is symmetric (g_munu = g_numu) 
    """
    g_munu = xr.Dataset({
        'tt': -(1 - 2*geos.M*geos.r / geos.Sigma), 
        'rr': geos.Sigma / geos.Delta,
        'thth': geos.Sigma,
        'phph': geos.Xi*np.sin(geos.theta)**2 / geos.Sigma, 
        'tph': -2*geos.M*geos.spin*geos.r*np.sin(geos.theta)**2 / geos.Sigma
    })
    return g_munu


def Sigma(r, theta, a):
    return r**2 + a**2 * jnp.cos(theta)**2


def Delta(r, a, M):
    return r**2 + a**2 - 2 * M * r


def Xi(r, theta, a, M):
    return (r**2 + a**2)**2 - a**2 * Delta(r, a, M) * jnp.sin(theta)**2


# Spacetime metric:
def g_tt(r, theta, a, M, eps=1e-8):
    Sig = Sigma(r, theta, a)
    safe_Sig = jnp.clip(Sig, eps, None)
    return -(1 - 2 * M * r / safe_Sig)


def g_rr(r, theta, a, M, eps=1e-8):
    Del = Delta(r, a, M)
    safe_Delta = jnp.where(jnp.abs(Del) > eps, Del, eps)
    return Sigma(r, theta, a) / safe_Delta


def g_thth(r, theta, a):
    return Sigma(r, theta, a)


def g_phph(r, theta, a, M, eps=1e-8):
    Sig = Sigma(r, theta, a)
    safe_Sig = jnp.clip(Sig, eps, None)
    return Xi(r, theta, a, M) * jnp.sin(theta)**2 / safe_Sig


def g_tph(r, theta, a, M, eps=1e-8):
    Sig = Sigma(r, theta, a)
    safe_Sig = jnp.clip(Sig, eps, None)
    return -2 * M * a * r * jnp.sin(theta)**2 / safe_Sig


# Inverse spacetime metric:
def gtt(r, theta, a, M, eps=1e-8):
    denom = Delta(r, a, M) * Sigma(r, theta, a)
    safe_denom = jnp.where(jnp.abs(denom) > eps, denom, eps)
    return jnp.where(
        jnp.abs(denom) > eps,
        -Xi(r, theta, a, M) / safe_denom, 
        -Xi(r, theta, a, M) / eps
    )


def grr(r, theta, a, M, eps=1e-8):
    Sig = Sigma(r, theta, a)
    safe_Sig = jnp.clip(Sig, eps, None)
    return Delta(r, a, M) / safe_Sig


def gthth(r, theta, a, eps=1e-8):
    Sig = Sigma(r, theta, a)
    safe_Sig = jnp.clip(Sig, eps, None)
    return 1 / safe_Sig


def gphph(r, theta, a, M, eps=1e-8):
    D = Delta(r, a, M)
    S = Sigma(r, theta, a)
    denom = D * S * jnp.sin(theta)**2
    safe_denom = jnp.where(jnp.abs(denom) > eps, denom, eps)
    return jnp.where(
        jnp.abs(denom) > eps,
        (D - a**2 * jnp.sin(theta)**2) / safe_denom,
        (D - a**2 * jnp.sin(theta)**2) / eps
    )


def gtph(r, theta, a, M, eps=1e-8):
    denom = Delta(r, a, M) * Sigma(r, theta, a)
    safe_denom = jnp.where(jnp.abs(denom) > eps, denom, eps)
    return jnp.where(
        jnp.abs(denom) > eps,
        -2 * M * a * r / safe_denom,
        -2 * M * a * r / eps
    )


def lapse(r, theta, a, M):
    """Lapse $\alpha$ for converting from normal observer velocity.
    
    We use the "double where" trick (https://docs.kidger.site/equinox/api/debug/)
    to avoid NaNs for small r.
    """
    gmunu_tt = gtt(r, theta, a, M)
    safe_gtt = jnp.where(gmunu_tt < 0, gmunu_tt, -1)
    alpha = jnp.where(gmunu_tt < 0, jnp.sqrt(-1 / safe_gtt), 0)
    return alpha


def shift(r, theta, a, M):
    """Shift 3-vector $\beta^i$ for converting from normal observer velocity."""
    # NOTE: `betar` and `betath` should both divide by `gtt`, but we directly
    # set them to 0 to avoid division by zero.
    betar = jnp.zeros_like(r)
    betath = jnp.zeros_like(r)
    betaph = -gtph(r, theta, a, M) / gtt(r, theta, a, M)
    betai = jnp.stack((betar, betath, betaph), axis=-1)
    return betai


def lorentz(utildei, r, theta, a, M):
    """Lorentz factor $\gamma$ for converting from normal observer velocity $\tilde{u}^i$.
    
    NOTE: this is where things would blow up if they were going to.
    """
    utilder = utildei[:, :, :, 0]
    utildeth = utildei[:, :, :, 1]
    utildeph = utildei[:, :, :, 2]
    g_munu_rr = g_rr(r, theta, a, M)
    g_munu_thth = g_thth(r, theta, a)
    g_munu_phph = g_phph(r, theta, a, M)
    g_ij_times_utildeiutildej = g_munu_rr * utilder**2 + g_munu_thth * utildeth**2 + g_munu_phph * utildeph**2
    sqrt_arg = 1 + g_ij_times_utildeiutildej
    safe_sqrt_arg = jnp.where(sqrt_arg > 0, sqrt_arg, 1e-12)
    gamma = jnp.where(sqrt_arg > 0, jnp.sqrt(safe_sqrt_arg), 1e-6)
    return gamma


def normal_to_3_velocity(utildei, r, theta, a, M):
    """Convert normal observer 3-velocity to Boyer-Lindquist 3-velocity.
    
    Args:
        utildei: normal observer 3-velocity, of shape (nalpha, nbeta, ngeo, 3).
        r: radius, of shape (nalpha, nbeta, ngeo).
        theta: polar angle, of shape (nalpha, nbeta, ngeo).
        a: spin, float.
        M: mass, float.
    
    Returns: Boyer-Lindquist 3-velocity $u^i$, of shape (nalpha, nbeta, ngeo, 3).
    """
    gamma = lorentz(utildei, r, theta, a, M)
    alpha = lapse(r, theta, a, M)
    betai = shift(r, theta, a, M)  # (nalpha, nbeta, ngeo, 3)
    vi = jnp.expand_dims(alpha / gamma, axis=-1) * utildei - betai
    return vi


def normal_to_4_velocity(utildei, r, theta, a, M, max_val=1000):
    """Convert normal observer 3-velocity to Boyer-Lindquist 4-velocity.
    
    Args:
        utildei: normal observer 3-velocity, of shape (nalpha, nbeta, ngeo, 3).
        r: radius, of shape (nalpha, nbeta, ngeo).
        theta: polar angle, of shape (nalpha, nbeta, ngeo).
        a: spin, float.
        M: mass, float.
        max_val: maximum value allowed for $u^mu$, since values tend to blow up
            at large r.
    
    Returns: Boyer-Lindquist 4-velocity $u^mu$, of shape (nalpha, nbeta, ngeo, 4).
    """
    gamma = lorentz(utildei, r, theta, a, M)
    alpha = lapse(r, theta, a, M)
    betai = shift(r, theta, a, M)
    ut = gamma / alpha
    ui = utildei - (gamma / alpha)[..., None] * betai
    umu = jnp.concatenate((ut[..., None], ui), axis=-1)
    umu = jnp.clip(umu, -jnp.inf, max_val)
    return umu


def bl_4_velocity_to_normal(umu, r, theta, a, M):
    # NOTE: `betar` and `betath` should both divide by `gtt`, but we directly
    # set them to 0 to avoid division by zero.
    betar = jnp.zeros_like(r)
    betath = jnp.zeros_like(r)
    betaph = -gtph(r, theta, a, M) / gtt(r, theta, a, M)
    utilder = umu[..., 1] + umu[..., 0] * betar
    utildeth = umu[..., 2] + umu[..., 0] * betath
    utildeph = umu[..., 3] + umu[..., 0] * betaph 
    return jnp.stack((utilder, utildeth, utildeph), axis=-1)


def bl_3_velocity_to_bl_4_velocity(vi, r, theta, a, M):
    """Convert Boyer-Lindquist 3-velocity $v^i$ to 4-velocity $u^\mu$."""
    # NOTE: This assumes that g_tr and g_tph cross-terms are zero.
    # vr = vi[..., 0]
    # vth = vi[..., 1]
    # vph = vi[..., 2]
    # den = (
    #     g_tt(r, theta, a, M) + 2 * g_tph(r, theta, a, M) * vph
    #     + g_rr(r, theta, a, M) * vr**2 + g_thth(r, theta, a) * vth**2 + g_phph(r, theta, a, M) * vph**2)
    # ut = jnp.sqrt(-1 / den)
    # ur = vr * ut
    # uth = vth * ut
    # uph = vph * ut
    u0 = ut(vi, r, theta, a, M)
    u1 = vi[..., 0] * u0
    u2 = vi[..., 1] * u0
    u3 = vi[..., 2] * u0
    return jnp.stack((u0, u1, u2, u3), axis=-1)


def spacetime_inv_metric(geos):
    """
    Inverse spacetime metric g^{munu} (Boyer-Linquist coordinates)
    
    Parameters
    ----------
    geos: xr.Dataset, 
        A dataset with the geodesic information and auxilary variables 
    
    Returns
    -------
    gmunu: xr.Dataset
        A dataset with the *inverse* spacetime metric (non-zero components) 
        
    Notes
    -----
    gmunu is symetric (gmunu = gnumu) 
    """
    gmunu = xr.Dataset({
        'tt': -geos.Xi / (geos.Delta * geos.Sigma),
        'rr': geos.Delta / geos.Sigma,
        'thth': 1 / geos.Sigma,
        'phph': (geos.Delta - geos.spin**2 * np.sin(geos.theta)**2) / 
                 (geos.Delta * geos.Sigma * np.sin(geos.theta)**2), 
        'tph': -2*geos.M*geos.spin*geos.r / (geos.Delta * geos.Sigma)
    })
    return gmunu


def raise_or_lower_indices(g, u):
    """
    Change contravarient to covarient vectors and vice-versa
    Lower indices: u_mu = g_munu * u^nu
    Raise indices: u^mu = g^munu * u_mu
    
    Parameters
    ----------
    g: xr.Dataset, 
        A dataset with non-zero spacetime metric components
    u: xr.DataArray, 
        A 4-vector dataarray with mu coordinates.
        
    Returns
    -------
    u_prime: xr.DataArray, 
        Raised or lowered indices depending on the metric g.
    """
    u_prime = xr.concat([
        g.tt * u.sel(mu=0) + g.tph * u.sel(mu=3),
        g.rr * u.sel(mu=1),
        g.thth * u.sel(mu=2),
        g.phph * u.sel(mu=3) + g.tph * u.sel(mu=0)
    ], dim='mu')
    return u_prime


def azimuthal_velocity_vector(geos, Omega):
    """
    Compute azimuthal velocity umu 4-vector on points sampled along geodesics
    
    Parameters
    ----------
    geos: xr.Dataset
        Dataset with Kerr geodesics (see: `kerr_geodesics` for more details)
    Omega: array, 
        An array with angular velocity specified along the geodesics coordinates
    
    Returns
    -------
    umu: xr.DataArray
        array with contravarient velocity 4 vector (index up) 
    """
    g_munu = spacetime_metric(geos)
    
    # 4-velocity vector
    # NOTE: computing `ut` in this way will throw a `RuntimeWarning`` for
    # taking the square root of a negative number.
    # This is expected for small radius values, as technically
    # this formula only holds for radius > r_isco.

    ut = 1 / np.sqrt(-(g_munu.tt + 2*Omega*g_munu.tph + g_munu.phph*Omega**2))
    ur = xr.DataArray(0)
    uth = xr.DataArray(0)
    uph = ut * Omega
    umu = xr.concat([ut, ur, uth, uph], dim='mu', coords='minimal')

    # TODO: there's a discrepancy between the formula above and the one below.
    # The one below is copied from `kgeo.u_kep`. The one above is in the
    # original bhnerf.
    # The one below assumes a simple Keplerian model without infall.
    # The one above is more general (but still doesn't account for infall).
    # M = float(geos.M)
    # s = 1  # +1 for prograde, -1 for retrograde
    # spin = np.abs(float(geos.spin))

    # ut =((geos.r**1.5 + s * spin * np.sqrt(M)) /
    #      np.sqrt(geos.r**3 - 3 * M * geos.r**2 + 2 * s * spin * geos.r**1.5 * np.sqrt(M)))
    # ur = xr.DataArray(0)
    # uth = xr.DataArray(0)
    # uph = ut * Omega
    # umu = xr.concat([ut, ur, uth, uph], dim='mu', coords='minimal')
    return umu


def radiative_transfer(emission, g, dtau, Sigma, use_jax=False):
    """
    Integrate emission over rays to get sensor pixel values.

    Parameters
    ----------
    emission: array,
        An array with emission values.
    J: np.array(shape=(3,...)),
        Stokes vector scaling factors including parallel transport (I, Q, U)
    g: array, 
        doppler boosting factor, 
    dtau: array, 
        mino time differential
    Sigma: array, 
    use_jax: bool, default=False,
        Using jax enables GPU accelerated computing.
        
    Returns
    -------
    stokes: array, 
        Image plane array with stokes vector values.
    """
    g = utils.expand_dims(g, emission.ndim, use_jax=use_jax)
    dtau = utils.expand_dims(dtau, emission.ndim, use_jax=use_jax)
    Sigma = utils.expand_dims(Sigma, emission.ndim, use_jax=use_jax)
    stokes = (g**2 * emission * dtau * Sigma).sum(axis=-1)
    return stokes


def ut(vi, r, theta, a, M):
    """Compute $u^t$ from the contravariant 3-velocity vector, $v^i$.
    
    Args:
        vi: 3-velocity vector, of shape (nx, ny, nz, 3).
    """
    vr, vth, vphi = vi[..., 0], vi[..., 1], vi[..., 2]
    # Spacetime metric:
    g_munu_rr = g_rr(r, theta, a, M)
    g_munu_tt = g_tt(r, theta, a, M)
    g_munu_phph = g_phph(r, theta, a, M)
    g_munu_thth = g_thth(r, theta, a)
    g_munu_tph = g_tph(r, theta, a, M)

    g_ti_times_vi = g_munu_tph * vphi
    g_ij_times_vivj = g_munu_rr * vr**2 + g_munu_thth * vth**2 + g_munu_phph * vphi**2
    
    sqrt_arg = -1 / (g_munu_tt + 2 * g_ti_times_vi + g_ij_times_vivj)
    safe_sqrt_arg = jnp.where(sqrt_arg > 0, sqrt_arg, 1e-12)
    return jnp.where(sqrt_arg > 0, jnp.sqrt(safe_sqrt_arg), 1e-6)


def u_kep(r, theta, a, M=1, retrograde=False, include_infall=True, safe=False):
    """Cunningham velocity for material on Keplerian orbits and infalling inside ISCO."""
    s = -1 if retrograde else 1
    ri = consts.isco_pro(a)  # ISCO prograde radius
    spin = jnp.abs(a)
    asign = jnp.sign(a)

    # Spacetime metric:
    g_munu_tt = g_tt(r, theta, a, M)
    g_munu_phph = g_phph(r, theta, a, M)
    g_munu_tph = g_tph(r, theta, a, M)

    # ------------
    # outside isco
    # ------------
    Omega = asign * s * jnp.sqrt(M) / (r**1.5 + s * spin * jnp.sqrt(M))

    # The code below is from `kgeo.u_kep`.
    # u0_outside_isco = ((r**1.5 + s * spin * jnp.sqrt(M)) /
    #                    jnp.sqrt(r**3 - 3 * M * r**2 + 2 * s * spin * r**1.5 * jnp.sqrt(M)))
    # The code below is from `azimuthal_velocity_vector`.
    sqrt_arg = -(g_munu_tt + 2 * Omega * g_munu_tph + g_munu_phph * Omega**2)
    if safe:
      safe_sqrt_arg = jnp.where(sqrt_arg > 0, sqrt_arg, 1e-3)
      u0_outside_isco = jnp.where(sqrt_arg > 0, 1 / jnp.sqrt(safe_sqrt_arg), 1e3)
    else:
      u0_outside_isco = 1 / jnp.sqrt(sqrt_arg)
    u3_outside_isco = Omega * u0_outside_isco

    # ------------
    # inside isco
    # ------------

    # preliminaries
    Delta = (r**2 - 2*r + a**2)

    # isco conserved quantities
    ell_i = s*asign*(ri**2  + a**2 - s*2*spin*jnp.sqrt(ri))/(ri**1.5 - 2*jnp.sqrt(ri) + s*spin)       
    gam_i = jnp.sqrt(1 - 2./(3.*ri)) # nice expression only for isco, prograde or retrograde

    # contravarient vel
    H = (2*r - a*ell_i) / Delta

    u0_inside_isco = gam_i * (1 + (2/r) * (1 + H))
    u3_inside_isco = gam_i * (ell_i + a * H)/(r**2)

    if include_infall:
        u0 = jnp.where(r > ri, u0_outside_isco, u0_inside_isco)
        safe_r = jnp.where(r < ri, r, ri)
        u1 = jnp.where(r > ri, 0, -jnp.sqrt(2./(3*ri)) * (ri / safe_r - 1)**1.5)
        u2 = jnp.zeros_like(r)
        u3 = jnp.where(r > ri, u3_outside_isco, u3_inside_isco)
    else:
        u0 = u0_outside_isco
        u1 = jnp.zeros_like(r)
        u2 = jnp.zeros_like(r)
        u3 = u3_outside_isco

    return (u0, u1, u2, u3)


def u_subkep(r, a, theta=np.pi / 2, M=1, fac_subkep=1, retrograde=False):
    """(sub) keplerian velocty and infalling inside isco"""
    s = -1 if retrograde else 1
    ri = consts.isco_pro(a) # ISCO prograde radius
    spin = jnp.abs(a)
    asign = jnp.sign(a)

    # ------------
    # outside isco
    # ------------
    safe_r_out = jnp.where(r >= ri, r, ri)

    # preliminaries       
    # Delta_out = (safe_r_out**2 - 2*safe_r_out + a**2)
    # Xi_out = (safe_r_out**2 + a**2)**2 - Delta_out*a**2
    Delta_out = Delta(safe_r_out, a, M)
    Xi_out = Xi(safe_r_out, theta, a, M)
    
    # conserved quantities
    ell = asign*s * (safe_r_out**2 + spin**2 - s*2*spin*jnp.sqrt(safe_r_out))/(r**1.5 - 2*jnp.sqrt(safe_r_out) + s*spin)
    ell *= fac_subkep
    sqrt_arg = Delta_out/(Xi_out/safe_r_out**2 - 4*a*ell/safe_r_out - (1-2/safe_r_out)*ell**2)
    safe_sqrt_arg = jnp.where(sqrt_arg > 0, sqrt_arg, 0)
    gam = jnp.where(sqrt_arg > 0, jnp.sqrt(safe_sqrt_arg), 0)

    # contravarient vel
    H = (2*r - a*ell)/Delta_out
    chi = 1 / (1 + (2/safe_r_out)*(1+H))
    Omega = (chi/safe_r_out**2)*(ell + a*H)
    
    u0_outside_isco = gam/chi
    u3_outside_isco = (gam/chi)*Omega

    # ------------
    # inside isco
    # ------------
    safe_r_in = jnp.where(r < ri, r, ri)

    # preliminaries
    Delta_in = Delta(safe_r_in, a, M)
    # Add eps for numerical stability where Delta_in is close to 0.
    Delta_in = Delta_in + 1e-6
    Xi_in = Xi(safe_r_in, theta, a, M)

    # isco conserved quantities
    Delta_isco = (ri**2 - 2*ri + a**2)
    Xi_isco = (ri**2 + a**2)**2 - Delta_isco*a**2
            
    ell_i = asign*s * (ri**2  + spin**2 - s*2*spin*jnp.sqrt(ri))/(ri**1.5 - 2*jnp.sqrt(ri) + s*spin)
    ell_i *= fac_subkep  
    gam_i = jnp.sqrt(Delta_isco/(Xi_isco/ri**2 - 4*a*ell_i/ri - (1-2/ri)*ell_i**2))

    # contravarient vel
    H = (2*safe_r_in - a*ell_i)/(Delta_in)  # add eps for numerical stability
    chi = 1 / (1 + (2/safe_r_in)*(1+H))
    Omega = (chi/safe_r_in**2)*(ell_i + a*H)
    sqrt_arg = Xi_in/safe_r_in**2 - 4*a*ell_i/safe_r_in - (1-2/safe_r_in)*ell_i**2 - Delta_in/gam_i**2
    safe_sqrt_arg = jnp.where(sqrt_arg > 0, sqrt_arg, 0)
    nu = (safe_r_in/Delta_in) * jnp.where(sqrt_arg > 0, jnp.sqrt(safe_sqrt_arg), 0)

    u0_inside_isco = gam_i/chi
    u1_inside_isco = -gam_i*(Delta_in/safe_r_in**2)*nu
    u3_inside_isco = (gam_i/chi)*Omega

    u0 = jnp.where(r >= ri, u0_outside_isco, u0_inside_isco)
    u1 = jnp.where(r >= ri, 0, u1_inside_isco)
    u2 = jnp.zeros_like(r)
    u3 = jnp.where(r >= ri, u3_outside_isco, u3_inside_isco)

    return (u0, u1, u2, u3)


def u_infall(r, a, theta=np.pi / 2, M=1):
    """ velocity for geodesic equatorial infall from infinity"""
    D = Delta(r, a, M)
    X = Xi(r, theta, a, M)

    u0 = X / (r**2 * D)
    u1 = -jnp.sqrt(2*r*(r**2 + a**2)) / (r**2)
    u2 = jnp.zeros_like(r)
    u3 = 2*a/(r*D)
    
    return (u0, u1, u2, u3)


def u_general(r, a, theta=np.pi / 2, M=1., fac_subkep=1, beta_phi=1, beta_r=1, retrograde=False):
    """general velocity model from AART paper, keplerian by default""" 
    Delta = r**2 + a**2 - 2*r
    (u0_infall, u1_infall, _, u3_infall) = u_infall(r, a, theta, M)
    Omega_infall = u3_infall/u0_infall # 2ar/Xi
    (u0_subkep, u1_subkep, _, u3_subkep) = u_subkep(r, a, theta, M, retrograde=retrograde,fac_subkep=fac_subkep)
    Omega_subkep = u3_subkep/u0_subkep

    u1 = u1_subkep + (1-beta_r)*(u1_infall - u1_subkep)
    Omega = Omega_subkep + (1-beta_phi)*(Omega_infall - Omega_subkep)

    sqrt_arg = 1 + (r**2) * (u1**2) / Delta
    safe_sqrt_arg = jnp.where(sqrt_arg > 0, sqrt_arg, 1e-3)
    u0 = jnp.where(sqrt_arg > 0, jnp.sqrt(safe_sqrt_arg), 1e3)

    sqrt_arg = 1 - (r**2 + a**2)*Omega**2 - (2/r)*(1 - a*Omega)**2
    safe_sqrt_arg = jnp.where(sqrt_arg > 0, sqrt_arg, 1)
    u0 /= jnp.where(sqrt_arg > 0, jnp.sqrt(safe_sqrt_arg), 1)

    u3 = u0 * Omega

    # Set all velocities where `u0` is undefined to 0.
    u1 = jnp.where(sqrt_arg > 0, u1, 0)
    u2 = jnp.zeros_like(r)
    u3 = jnp.where(sqrt_arg > 0, u3, 0)

    return (u0, u1, u2, u3)


def spherical_velocities_kep(r, theta, a, M=1,
                             retrograde=False, include_infall=True, safe=False):
    ut, ur, uth, uph = u_kep(
        r, theta, a, M,
        retrograde=retrograde, include_infall=include_infall, safe=safe)
    dr_dt = (ur /  ut).reshape(r.shape)
    dth_dt = (uth / ut).reshape(r.shape)
    dph_dt = (uph / ut).reshape(r.shape)
    dr_dt = jnp.nan_to_num(dr_dt)
    dth_dt = jnp.nan_to_num(dth_dt)
    dph_dt = jnp.nan_to_num(dph_dt)
    spherical_velocities = jnp.stack((dr_dt, dth_dt, dph_dt), axis=-1)
    return spherical_velocities


def spherical_velocities(r, a, theta=np.pi / 2, M=1., fac_subkep=1., beta_phi=1., beta_r=1., retrograde=False):
    ut, ur, uth, uph = u_general(
        r, a, theta, M, fac_subkep, beta_phi, beta_r, retrograde)
    dr_dt = (ur /  ut).reshape(r.shape)
    dth_dt = (uth / ut).reshape(r.shape)
    dph_dt = (uph / ut).reshape(r.shape)
    return jnp.stack((dr_dt, dth_dt, dph_dt), axis=-1)


def doppler_factor(geos, umu, fillna=0.0):
    """
    Compute Doppler factor as dot product of wave 4-vectors with the velocity 4-vector
    
    Parameters
    ----------
    geos: xr.Dataset
        Dataset with Kerr geodesics (see: `kerr_geodesics` for more details)
    umu: xr.DataArray
        array with contravarient velocity 4 vector (index up)
    fillna: float or False or None, 
        If float fill nans with float else if False leave nans
        
    Returns
    -------
    g: xr.DataArray,
        Doppler boosting factor sampled along the geodesics
    """
    k_mu = wave_vector(geos)
    g = geos.E / -(k_mu * umu).sum('mu', skipna=False)
    if not ((isinstance(fillna, bool) and fillna == False) or fillna is None):
        g = g.fillna(fillna)
    return g


def get_doppler_factor_kep(geos, retrograde, include_infall):
    """Get Doppler boosting factor $g$ for Keplerian orbits."""
    ut, ur, uth, uph = u_kep(
        geos.r.data, geos.theta.data, float(geos.spin), float(geos.M), retrograde, include_infall)
    ut = ut.reshape(geos.r.shape)
    ur = ur.reshape(geos.r.shape)
    uth = uth.reshape(geos.r.shape)
    uph = uph.reshape(geos.r.shape)
    ut = xr.DataArray(ut, dims=['beta', 'alpha', 'geo'])
    ur = xr.DataArray(ur, dims=['beta', 'alpha', 'geo'])
    uth = xr.DataArray(uth, dims=['beta', 'alpha', 'geo'])
    uph = xr.DataArray(uph, dims=['beta', 'alpha', 'geo'])
    umu = xr.concat([ut, ur, uth, uph], dim='mu', coords='minimal')
    return jnp.array(doppler_factor(geos, umu))


def get_doppler_factor(geos, fac_subkep, beta_phi, beta_r, retrograde=False):
    """Get Doppler boosting factor $g$ for Keplerian orbits."""
    ut, ur, uth, uph = u_general(
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
    return jnp.array(doppler_factor(geos, umu))


def get_doppler_factor_from_umu(umu, k_mu, E=1., eps=1e-8):
    """Get Doppler boosting factor $g$ from 4-velocity $u^\mu$.
    
    Args:
        umu: 4-velocity, of shape (beta, alpha, geo, 4).
        k_mu: Photon wavevector vector $k_\mu$, of shape (beta, alpha, geo, 4).
        E: Photon energy at the observer at infinity.
    """
    denom = jnp.sum(k_mu * umu, axis=-1)  # (beta, alpha, geo)
    # Protect against division by zero or near-zero.
    safe_denom = jnp.where(jnp.abs(denom) > eps, denom, eps)
    g = jnp.where(jnp.abs(denom) > eps, E / -safe_denom, 1 / eps)
    return g