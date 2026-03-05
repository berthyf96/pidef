# Adapted from bhnerf (https://github.com/aviadlevis/bhnerf)
# Original authors: Aviad Levis et al.

from astropy.constants import G, c, M_sun
from astropy import units
import ehtim.const_def as ehc
import jax.numpy as jnp

# Inner most stable circular orbit (ISCO) parameters: 
# https://en.wikipedia.org/wiki/Innermost_stable_circular_orbit
z1 = lambda a: 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
z2 = lambda a: jnp.sqrt(3 * a**2 + z1(a)**2)
isco_pro = lambda a: (3 + z2(a) - jnp.sqrt((3 - z1(a)) * (3 + z1(a) + 2*z2(a))))
isco_retro = lambda a: (3 + z2(a) + jnp.sqrt((3 - z1(a)) * (3 + z1(a) + 2*z2(a))))

# Black hole quantities
GM_c3 = lambda M: G * M / c**3
GM_c2 = lambda M: G * M / c**2

# SgrA constants / fields
sgra_mass = 4.154*10**6 * M_sun
sgra_distance = 26673 * units.lightyear

# Converting from dimensionless M to physical units
RADPERUAS = ehc.RADPERUAS * (units.rad / units.uas)
rad_per_M = lambda M, D: (GM_c2(M) / D).to(1) * units.rad