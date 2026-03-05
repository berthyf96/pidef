# Adapted from bhnerf (https://github.com/aviadlevis/bhnerf)
# Original authors: Aviad Levis et al.

import functools
from typing import Any, Callable

import diffrax
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import pidef.emission
from pidef import kgeo
from pidef import utils


safe_sin = lambda x: jnp.sin(x % (100 * jnp.pi))

@flax.struct.dataclass
class State:
  """Training state."""
  params: Any
  model_state: Any
  opt_state: Any


@flax.struct.dataclass
class EmissionVelocityState:
    """Training state for both emission predictor and velocity predictor."""
    step: int
    emission: State
    velocity: State
    data_weight: float
    pinn_weight: float
    velo_weight: float
    rng: jax.Array


class MLP(nn.Module):
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
    batch_norm: bool = True
  
    @nn.compact
    def __call__(self, x, train=True):
        """A simple Multi-Layer Preceptron (MLP) network

        Parameters
        ----------
        x: jnp.ndarray(float32), 
            [batch_size * n_samples, feature], points.
        net_depth: int, 
            the depth of the first part of MLP.
        net_width: int, 
            the width of the first part of MLP.
        activation: function, 
            the activation function used in the MLP.
        out_channel: 
            int, the number of alpha_channels.
        do_skip: boolean, 
            whether or not to use a skip connection

        Returns
        -------
        out: jnp.ndarray(float32), 
            [batch_size * n_samples, out_channel].
        """
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.he_uniform())

        if self.do_skip:
            skip_layer = self.net_depth // 2

        inputs = x
        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.activation(x)
            if self.do_skip:
                if i % skip_layer == 0 and i > 0:
                    x = jnp.concatenate([x, inputs], axis=-1)
            if self.batch_norm:
                x = nn.BatchNorm(
                    use_running_average=not train, epsilon=1e-2, momentum=0.9)(x)
        out = dense_layer(self.out_channel)(x)

        return out


def integrated_posenc(x, x_cov, max_deg, min_deg=0):
    """
    Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].

    Parameters
    ----------
    x: jnp.ndarray, variables to be encoded. Should
      be in [-pi, pi]. 
    x_cov: jnp.ndarray, covariance matrices for `x`.
    max_deg: int, the max degree of the encoding.
    min_deg: int, the min degree of the encoding. default=0.

    Returns
    -------
    encoded: jnp.ndarray, encoded variables.
    """
    if jnp.isscalar(x_cov):
        x_cov = jnp.full_like(x, x_cov)
    scales = 2**jnp.arange(min_deg, max_deg)
    shape = list(x.shape[:-1]) + [-1]
    y = jnp.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = jnp.reshape(x_cov[..., None, :] * scales[:, None]**2, shape)

    return expected_sin(
      jnp.concatenate([y, y + 0.5 * jnp.pi], axis=-1),
      jnp.concatenate([y_var] * 2, axis=-1))


def expected_sin(x, x_var):
    # When the variance is wide, shrink sin towards zero.
    y = jnp.exp(-0.5 * x_var) * jnp.sin(x)
    return y


def posenc(x, deg):
    """
    Concatenate `x` with a positional encoding of `x` with degree `deg`.
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Parameters
    ----------
    x: jnp.ndarray, 
        variables to be encoded. Note that x should be in [-pi, pi].
    deg: int, 
        the degree of the encoding.

    Returns
    -------
    encoded: jnp.ndarray, 
        encoded variables.
    """
    if deg == 0:
        return x
    scales = jnp.array([2**i for i in range(deg)])
    xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = safe_sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.concatenate([x] + [four_feat], axis=-1)


class EmissionPredictor(nn.Module):
    """
    Full function to predict emission at a time step.
    
    Parameters
    ----------
    scale: float, default=1.0       
        scale of the domain; scales the NN inputs
    t_scale: float, default=1.0
        scale of the time domain; scales the NN time input
    t_bias: float, default=0.0
        bias to add to the NN time input
    rmin: float, default=0.0        
        minimum radius for recovery
    rmax: float, default=np.inf     
        maximum radius for recovery
    z_width: float, default=np.inf  
        maximum width of the disk (M units)
    posenc_deg: int, default=3
    posenc_var: float, default=2e-5 
        Corresponds to variance of uniform distribution variance with voxel width of ~1/64.
    net_depth: int, default=4
    net_width: int, default=128
    activation: Callable[..., Any], default=nn.relu
    out_channel: int default=1
    do_skip: bool, default=True
    """
    scale: float = 1.0
    t_scale: float = 1.0
    t_bias: float = 0.0
    rmin: float = 0.0
    rmax: float = np.inf
    z_width: float = np.inf 
    posenc_deg: int = 3
    time_posenc_deg: int = 3
    posenc_var: float = 2e-5
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
    batch_norm: bool = True
    spherical: bool = False
    
    @nn.compact
    def __call__(self, t_frames, coords, train=True):
        """
        Sample emission on given coordinates at specified times assuming a velocity model
        
        Parameters
        ----------
        t_frames: array of time for each image frame, of shape (nt,).
        coords: 3D coordinates, of shape (num_alpha, num_beta, ngeo, 3).

        Returns
        -------
        emission: array with the emission points, of shape (num_alpha, num_beta, ngeo).
        """
        emission_MLP = MLP(
            self.net_depth, self.net_width, self.activation, self.out_channel,
            self.do_skip, self.batch_norm)
        def predict_emission(t_frames, coords):
            input_t = jnp.atleast_1d(t_frames)
            if self.spherical:
                input_coords = utils.cartesian_to_spherical(coords)
            else:
                input_coords = coords
            input_coords = jnp.repeat(input_coords[None, ...], len(input_t), 0)  # (nt, nx, ny, nz, 3)
            
            # TODO: normalize `t` to [0, 1].
            for i in range(1, 4):
                input_t = jnp.repeat(input_t[..., None], input_coords.shape[i], axis=i)
            input_t = jnp.expand_dims(input_t, axis=-1)  # (nt, nx, ny, nz, 1)

            input_coords = input_coords / self.scale
            input_t = (input_t + self.t_bias) / self.t_scale

            # The following way of preparing the network inputs applies the
            # positional encoding to each dimension first and then concatenates.
            input_coords_posenc_x = posenc(input_coords[:, :, :, :, 0:1], self.posenc_deg)
            input_coords_posenc_y = posenc(input_coords[:, :, :, :, 1:2], self.posenc_deg)
            input_coords_posenc_z = posenc(input_coords[:, :, :, :, 2:3], self.posenc_deg)
            input_coords_posenc = jnp.concatenate(
                (input_coords_posenc_x, input_coords_posenc_y, input_coords_posenc_z),
                axis=-1)
            input_t_posenc = posenc(input_t, self.time_posenc_deg)
            net_input = jnp.concatenate((input_coords_posenc, input_t_posenc), axis=-1)
    
            net_output = emission_MLP(net_input, train=train)
            emission = nn.sigmoid(net_output - 10.0)
            emission = pidef.emission.fill_unsupervised(
                emission, coords, self.rmin, self.rmax, self.z_width, use_jax=True)
            return emission[..., 0]

        emission = predict_emission(t_frames, coords)
        return emission


class VelocityPredictor(nn.Module):
    """
    Full function to predict velocity at a time step.
    
    Parameters
    ----------
    scale: float, default=1.0       
        scale of the domain; scales the NN inputs
    t_scale: float, default=1.0
        scale of the time domain; scales the NN time input
    t_bias: float, default=0.0
        bias to add to the NN time input
    posenc_deg: int, default=3
    posenc_var: float, default=2e-5 
        Corresponds to variance of uniform distribution variance with voxel width of ~1/64.
    net_depth: int, default=4
    net_width: int, default=128
    activation: Callable[..., Any], default=nn.relu
    out_channel: int default=1
    do_skip: bool, default=True
    """
    coordinate_type: str
    rmin: float = 0.0
    rmax: float = np.inf
    z_width: float = np.inf
    in_scale: float = 1.0
    out_r_scale: float = 1.0
    out_phi_scale: float = 1.0
    posenc_deg: int = 3
    posenc_var: float = 2e-5
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 3
    do_skip: bool = True
    batch_norm: bool = True
    residual: bool = False
    fill_unsupervised: bool = False

    @nn.compact
    def __call__(self, coords, reference_velocity, train=True):
        """
        Sample emission on given coordinates at specified times assuming a velocity model
        
        Parameters
        ----------
        coords: 3D coordinates, of shape (nalpha, nbeta, ngeo, 3).

        Returns
        -------
        velocity: jnp.ndarray of shape (nalpha, nbeta, ngeo, 3).
        """
        emission_MLP = MLP(
            self.net_depth, self.net_width, self.activation, self.out_channel,
            self.do_skip, self.batch_norm)

        def predict_velocity(coords):
            if self.coordinate_type in ['spherical', 'r', 'r+theta']:
                spherical_coords = utils.cartesian_to_spherical(coords)
                r = spherical_coords[..., 0]
                theta = spherical_coords[..., 1]
                phi = spherical_coords[..., 2]

                # Normalize to [-1, 1].
                r = 2 * r / self.in_scale - 1
                theta = 2 * theta / np.pi - 1
                phi = 2 * (phi + np.pi) / (2 * np.pi) - 1

                if self.coordinate_type == 'spherical':
                    input_coords = jnp.stack((r, theta, phi), axis=-1)
                elif self.coordinate_type == 'r':
                    input_coords = jnp.expand_dims(r, axis=-1)
                elif self.coordinate_type == 'r+theta':
                    input_coords = jnp.stack((r, theta), axis=-1)
            elif self.coordinate_type == 'cartesian':
                input_coords = coords / self.in_scale
            else:
                raise ValueError(f'Unknown coordinate type: {self.coordinate_type}')

            input_coords = jnp.expand_dims(input_coords, 0)  # (1, nx, ny, nz, 3)

            net_input = posenc(input_coords, self.posenc_deg)

            net_output = emission_MLP(net_input, train=train)

            net_output_r = net_output[..., 0][..., None]
            net_output_theta = net_output[..., 1][..., None]
            net_output_phi = net_output[..., 2][..., None]

            # Apply output activations.
            net_output_r = nn.tanh(net_output_r) * self.out_r_scale
            net_output_theta = (net_output_theta) * 0
            net_output_phi = nn.sigmoid(net_output_phi) * self.out_phi_scale

            net_output = jnp.concatenate((net_output_r, net_output_theta, net_output_phi), axis=-1)

            if self.residual:
                velocity = net_output + reference_velocity
            else:
                velocity = net_output
            velocity = velocity[0]   # (1, nx, ny, nz, 3) --> (nx, ny, nz, 3)
            if self.fill_unsupervised:
                velocity = pidef.emission.fill_unsupervised(
                    velocity, coords, self.rmin, self.rmax, self.z_width,
                    fill_value=reference_velocity, use_jax=True)
            return velocity

        velocity = predict_velocity(coords)

        return velocity


def init_variables(predictor, coords, has_t_input, seed=1):
    if has_t_input:
        variables = predictor.init(jax.random.PRNGKey(seed), 0, coords)
    else:
        variables = predictor.init(jax.random.PRNGKey(seed), coords, jnp.zeros_like(coords))
    init_model_state, init_params = flax.core.pop(variables, 'params')
    return init_params, init_model_state


def init_state_and_optimizer(emission_predictor, velocity_predictor, geos,
                             num_iters, lr_init, lr_final, lr_decay_schedule,
                             lr_decay_steps, lr_decay_begin,
                             grad_clip=None,
                             data_weight=1., pinn_weight=1., velo_weight=1.,
                             seed=1):
    """Initialize EmissionVelocityState."""
    init_coords = jnp.stack([geos.x.data, geos.y.data, geos.z.data], axis=-1)

    state_list, tx_list = [], []
    for i, predictor in enumerate([emission_predictor, velocity_predictor]):
        # Initialize params.
        has_t_input = i == 0
        params, model_state = init_variables(
            predictor, init_coords, has_t_input, seed)

        # Create optimizer.
        lr_init_val = lr_init[i] if len(np.atleast_1d(lr_init)) > 1 else lr_init
        lr_final_val = lr_final[i] if len(np.atleast_1d(lr_final)) > 1 else lr_final
        grad_clip_val = grad_clip[i] if len(np.atleast_1d(grad_clip)) > 1 else grad_clip
        if lr_decay_schedule == 'cosine':
            lr = optax.schedules.cosine_decay_schedule(
                lr_init_val, lr_decay_steps, alpha=lr_final_val / lr_init_val)
        elif lr_decay_schedule == 'linear':
            lr = optax.polynomial_schedule(
                lr_init_val, lr_final_val, power=1,
                transition_steps=lr_decay_steps, transition_begin=lr_decay_begin)
        else:
            raise ValueError(f'Unknown lr_decay_schedule: {lr_decay_schedule}')
        if grad_clip_val is not None:
            tx = optax.chain(
                optax.clip(grad_clip_val),
                optax.adam(learning_rate=lr))
        else:
            tx = optax.adam(learning_rate=lr)
        opt_state = tx.init(params)

        state = State(
            params=params,
            model_state=model_state,
            opt_state=opt_state)
        
        state_list.append(state)
        tx_list.append(tx)
    
    state = EmissionVelocityState(
        step=0,
        emission=state_list[0],
        velocity=state_list[1],
        data_weight=data_weight,
        pinn_weight=pinn_weight,
        velo_weight=velo_weight,
        rng=jax.random.PRNGKey(seed))

    return state, tx_list


def get_learned_velocity_warp_coords_fn(solver, stepsize_controller, adjoint, fac_subkep, beta_phi, beta_r,
                                        dt0_M=None, normal_observer=True, a=0, M=1, retrograde=False):

    def learned_velocity_warp_coords(velocity_params, velocity_model_state, velocity_predictor_fn,
                                     init_coords, t_frames_M, t0_M, train=False):
        """Apply velocity warp to coordinates using learned velocity model.
        
        Args:
            velocity_params: params of velocity predictor.
            velocity_model_state: model state of velocity predictor.
            velocity_predictor_fn: apply function of velocity predictor.
            init_coords: Cartesian coordinates of initial state,
                an array of shape (nx, ny, nz, 3).
            t_frames_M: list of times in units of M at which to evaluate velocity warp.
            t0_M: starting time in units of M.
            dt0_M: initial time step in units of M for Diffrax solver.
            train: whether to update batch statistics.
                TODO: return `new_model_state` if `train` is True.
        
        Returns: warped Cartesian coordinates, of shape (nx, ny, nz, 3).
        """
        def f(t, y, _):
            cartesian_coords = utils.spherical_to_cartesian(y)
            # r = y[..., 0]
            # # r = jnp.where(r > 0, r, 1e-3)
            # theta = y[..., 1]
            r = jnp.clip(y[..., 0], 1e-6, None)
            theta = y[..., 1]
            if normal_observer:
                # Compute reference velocity (umu -> utildei).
                ut, ur, uth, uph = kgeo.u_general(
                    r, a, theta, M,
                    fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r,
                    retrograde=retrograde)
                reference_umu = jnp.stack((ut, ur, uth, uph), axis=-1)
                reference_velocity = kgeo.bl_4_velocity_to_normal(
                    reference_umu, r, theta, a, M)
            else:
                reference_velocity = kgeo.spherical_velocities(
                    r, a, theta, M,
                    fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r,
                    retrograde=retrograde)
            if normal_observer:
                utilde, _ = velocity_prediction(
                    velocity_params, velocity_model_state, velocity_predictor_fn,
                    cartesian_coords, reference_velocity, train=train)
                spherical_velocities = kgeo.normal_to_3_velocity(utilde, r, theta, a, M)
            else:
                spherical_velocities, _ = velocity_prediction(
                    velocity_params, velocity_model_state, velocity_predictor_fn,
                    cartesian_coords, reference_velocity, train=train)

            return -spherical_velocities

        init_spherical_coords = utils.cartesian_to_spherical(init_coords)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(f),
            solver,
            stepsize_controller=stepsize_controller,
            t0=t0_M,
            t1=t_frames_M[-1],
            dt0=dt0_M,
            y0=init_spherical_coords,
            saveat=diffrax.SaveAt(ts=t_frames_M),
            adjoint=adjoint
        )

        warped_spherical_coords = solution.ys

        # Convert back to Cartesian coordinates.
        warped_coords = jax.vmap(utils.spherical_to_cartesian)(warped_spherical_coords)
        warped_coords = jnp.nan_to_num(warped_coords)
        return warped_coords
    
    return learned_velocity_warp_coords


def emissions_prediction(params, model_state, predictor_fn, t_frames, coords, train=False):
    variables = {'params': params, **model_state}
    if train:
        emission, new_model_state = predictor_fn(
            variables, t_frames, coords, train=train, mutable=['batch_stats'])
    else:
        emission, _ = predictor_fn(
            variables, t_frames, coords, train=train, mutable=['batch_stats'])
        new_model_state = model_state
    return emission, new_model_state


def image_plane_prediction(params, model_state, predictor_fn, t_frames, coords,
                           g, dtau, Sigma, J=1., train=False):
    variables = {'params': params, **model_state}
    if train:
        emission, new_model_state = predictor_fn(
            variables, t_frames, coords, train=train, mutable=['batch_stats'])
    else:
        emission, _ = predictor_fn(
            variables, t_frames, coords, train=train, mutable=['batch_stats'])
        new_model_state = model_state
    if not jnp.isscalar(J):
        J = utils.expand_dims(J, emission.ndim + 1, 0, use_jax=True)
        emission = J * utils.expand_dims(emission, emission.ndim+1, 1, use_jax=True)
        emission = jnp.squeeze(emission)
    images = kgeo.radiative_transfer(emission, g, dtau, Sigma, use_jax=True)
    return images, new_model_state


def emissions_and_image_plane_prediction(params, model_state, predictor_fn,
                                         t_frames, coords, J, g, dtau, Sigma, train=False):
    emission, new_model_state = emissions_prediction(
        params, model_state, predictor_fn, t_frames, coords, train=train)
    if not jnp.isscalar(J):
        J = utils.expand_dims(J, emission.ndim+1, 0, use_jax=True)
        J_emission = J * utils.expand_dims(emission, emission.ndim+1, 1, use_jax=True)
        J_emission = jnp.squeeze(J_emission)
    else:
        J_emission = emission
    images = kgeo.radiative_transfer(J_emission, g, dtau, Sigma, use_jax=True)
    return (emission, images), new_model_state


def velocity_prediction(params, model_state, predictor_fn, coords, reference_velocity, train=False):
    variables = {'params': params, **model_state}
    if train:
        velocity, new_model_state = predictor_fn(
            variables, coords, reference_velocity, train=train, mutable=['batch_stats'])
    else:
        velocity, _ = predictor_fn(
            variables, coords, reference_velocity, train=train, mutable=['batch_stats'])
        new_model_state = model_state
    return velocity, new_model_state


def evaluate_velocity(params, model_state, predictor_fn, rmin, rmax, z_width,
                      a, M, retrograde, fac_subkep, beta_phi, beta_r, normal_observer, coords=None, resolution=64):
  """Get estimated and target velocity fields on the 3D grid with radius rmax."""
  if coords is None:
    # Define cube of Cartesian coordinates.
    coords = utils.get_grid_coords(rmax, resolution)
  spherical_coords = utils.cartesian_to_spherical(coords)
  r = spherical_coords[..., 0]
  theta = spherical_coords[..., 1]

  if normal_observer:
    # Compute reference velocity (umu -> utildei).
    ut, ur, uth, uph = kgeo.u_general(
        r, a, theta, M,
        retrograde=retrograde, fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r)
    reference_umu = jnp.stack((ut, ur, uth, uph), axis=-1)
    reference_utildei = kgeo.bl_4_velocity_to_normal(
        reference_umu, r, theta, a, M)
    utildei, _ = velocity_prediction(
        params, model_state, predictor_fn,
        coords, reference_utildei, train=False)
    vi = kgeo.normal_to_3_velocity(utildei, r, theta, a, M)
  else:
    reference_vi = kgeo.spherical_velocities(
        r, theta, a, M, retrograde=retrograde,
        fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r)
    vi, _ = velocity_prediction(
        params, model_state, predictor_fn,
        coords, reference_vi, train=False)

  # Get target velocity.
  true_vi = kgeo.spherical_velocities(
      r, a, theta, M, retrograde=retrograde,
      fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r)

  # Zero-fill unsupservised region.
  vi = pidef.emission.fill_unsupervised(vi, coords, rmin, rmax, z_width, use_jax=True)
  true_vi = pidef.emission.fill_unsupervised(true_vi, coords, rmin, rmax, z_width, use_jax=True)

  return vi, true_vi


def sample_3d_grid(apply_fn, params, model_state, t_frame, coords, resolution=64, chunk=-1, train=False):
    variables = {'params': params, **model_state}
    
    # Get the a grid values sampled from the neural network.
    resolution = coords.shape[1]
    chunk = resolution if chunk < 0 else chunk

    emission = []
    for c in range(resolution//chunk):
        coords_chunk = coords[:, c * chunk : (c + 1) * chunk, :, :]
        emission.append(apply_fn(variables, t_frame, coords_chunk, train=train))
    emission = jnp.concatenate(emission, axis=0)
    return jnp.squeeze(emission)