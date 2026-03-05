from astropy import units
import jax
import jax.numpy as jnp
import numpy as np
import optax

import pidef.constants as consts
from pidef import emission
from pidef import kgeo
from pidef import network
from pidef import utils


def get_loss_fn(geos,
                learned_velocity_warp_coords_fn,
                emission_coords,
                delta_t,
                learned_doppler,
                fixed_g,
                retrograde,
                fac_subkep,
                beta_phi,
                beta_r,
                rmin,
                rmax,
                eht,
                image_weight,
                vis_weight,
                amp_weight,
                cp_weight,
                logca_weight,
                flux_weight,
                use_target_velocity=False,
                doppler=True,
                J=1.,
                t_units=units.hr,
                normal_observer=True,
                velo_loss_grid='emission',
                l1_pinn_loss=False,
                blur_for_pinn_loss=False):
  """Returns a loss function for training the model.

  * velo_loss: evaluated on entire geodesic (or emission) grid.
  * data_loss: evaluated on geodesic grid. Emission is zeroed outside of the supervised region.
  * pinn_loss: evaluated on emission grid. Emission is zeroed outside of the supervised region.

  Args:
    geos: xarray.Dataset containing data for geodesics.
    learned_velocity_warp_coords_fn: Function for applying learned velocity warp to coordinates.
    emission_coords: Coordinates of emissivity grid used for PINN loss, of shape (nx, ny, nz, 3).
    delta_t: Time interval for PINN loss.
    learned_doppler: Whether to use learned velocity in Doppler boosting factor.
    fixed_g: Fixed Doppler boosting factor if not using learned velocity.
    retrograde: Whether orbit is in retrograde.
    fac_subkep: Sub-Keplerianity factor for target velocity.
    beta_phi: beta_phi parameter for target velocity.
    beta_r: beta_r parameter for target velocity.
    r_min: Min. radius for filling unsupervised regions of learned Doppler factor.
    r_max: Max. radius for filling unsupervised regions of learned Doppler factor.
    eht: EHT forward model object with methods for computing data loss.
    use_target_velocity: Whether to use target velocity for dynamics loss.
    doppler: Whether to use Doppler boosting factor in raytracing.
    J: Stokes vector scaling factors including parallel transport (I, Q, U).
      J=1.0 gives non-polarized emission.
    t_units: Units for time.
    normal_observer: Whether to use normal observer coordinates.
    velo_loss_grid: Which coordinate grid to use for velocity loss
      ('geos' or 'emission').
    l1_pinn_loss: Whether to use L1 loss for PINN loss.
    blur_for_pinn_loss: Whether to apply Gaussian blur to emissivities before computing PINN loss. 

  Returns: Loss function.
  """
  GM_c3 = consts.GM_c3(consts.sgra_mass).to(t_units)
  delta_t_M = delta_t / GM_c3.value
  a = float(geos.spin)
  M = float(geos.M)
  k_mu_const = jnp.asarray(kgeo.wave_vector(geos).data, dtype=jnp.float32)
  E_const = jnp.asarray(geos.E.data, dtype=jnp.float32)

  if use_target_velocity:
    predicted_warped_coords_with_target_velo = emission.kgeo_velocity_warp_coords(
      emission_coords,
      t_frames_M=jnp.array([delta_t_M]),
      t0_M=0,
      dt0_M=0.001 * t_units / GM_c3.value,
      spin=float(geos.spin),
      fac_subkep=fac_subkep,
      beta_phi=beta_phi,
      beta_r=beta_r,
      M=float(geos.M),  # TODO: should this be M or geos.M?
      retrograde=retrograde)[0]

  # Get coordinates for evaluating geodesics/raytracing.
  geos_coords = jnp.stack([geos.x.data, geos.y.data, geos.z.data], axis=-1)

  if velo_loss_grid == 'geos':
    velo_coords = geos_coords
    velo_r = geos.r.data
    velo_theta = geos.theta.data
  elif velo_loss_grid == 'emission':
    velo_coords = emission_coords
    emission_spherical_coords = utils.cartesian_to_spherical(emission_coords)
    velo_r = emission_spherical_coords[:, :, :, 0]
    velo_theta = emission_spherical_coords[:, :, :, 1]
  # Get target velocity.
  target_velocity = kgeo.spherical_velocities(
      velo_r, a, velo_theta, M,
      fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r, retrograde=retrograde)
  # target_velocity = emission.fill_unsupervised(
  #     target_velocity, velo_coords, rmin, rmax, velocity_z_width, use_jax=True)

  # Get reference velocity for evaluating velocity loss. 
  if normal_observer:
    # Compute reference velocity (umu -> utildei).
    ut, ur, uth, uph = kgeo.u_general(
        velo_r, a, velo_theta, M,
        fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r, retrograde=retrograde)
    reference_umu = jnp.stack((ut, ur, uth, uph), axis=-1)
    velo_reference_velocity = kgeo.bl_4_velocity_to_normal(
        reference_umu, velo_r, velo_theta, float(geos.spin), float(geos.M))
  else:
    velo_reference_velocity = kgeo.u_general(
        velo_r, a, velo_theta, M,
        fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r, retrograde=retrograde)
  
  # Get reference velocity on geodesic grid.
  if normal_observer:
    # Compute reference velocity (umu -> utildei).
    ut, ur, uth, uph = kgeo.u_general(
        geos.r.data, a, geos.theta.data, M,
        fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r, retrograde=retrograde)
    reference_umu = jnp.stack((ut, ur, uth, uph), axis=-1)
    geos_reference_velocity = kgeo.bl_4_velocity_to_normal(
        reference_umu, geos.r.data, geos.theta.data, float(geos.spin), float(geos.M))
  else:
    geos_reference_velocity = kgeo.spherical_velocities(
        geos.r.data, a, geos.theta.data, M,
        fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r, retrograde=retrograde)

  def loss_fn(emission_params,
              velocity_params,
              model_state_list,
              predictor_fn_list,
              data_t,
              pinn_t,
              y_image,
              y_vis_expanded,
              y_sigmavis_expanded,
              y_amp,
              y_sigmaamp,
              y_cp,
              y_sigmacp,
              y_logca,
              y_sigmalogca,
              y_flux,
              A_vis_expanded,
              A_cp,
              A_logca,
              data_weight,
              pinn_weight,
              velo_weight):
    emission_model_state = model_state_list[0]
    velocity_model_state = model_state_list[1]
    emission_predictor_fn = predictor_fn_list[0]
    velocity_predictor_fn = predictor_fn_list[1]

    # Velocity loss.
    if normal_observer:
      utilde, new_velocity_model_state = network.velocity_prediction(
          velocity_params, velocity_model_state, velocity_predictor_fn,
          velo_coords, velo_reference_velocity, train=True)
      vi = kgeo.normal_to_3_velocity(utilde, velo_r, velo_theta, a, M)
      umu = kgeo.normal_to_4_velocity(utilde, velo_r, velo_theta, a, M)
    else:
      vi, new_velocity_model_state = network.velocity_prediction(
          velocity_params, velocity_model_state, velocity_predictor_fn,
          velo_coords, velo_reference_velocity, train=True)
      umu = kgeo.bl_3_velocity_to_bl_4_velocity(vi, velo_r, velo_theta, a, M)
    # vi = emission.fill_unsupervised(
        # vi, velo_coords, rmin, rmax, velocity_z_width, use_jax=True)
    velo_loss = jnp.sum(jnp.square(vi - target_velocity))

    # Data fit loss.
    if learned_doppler and velo_loss_grid == 'emission':
      # Need to get velocity on geodesic grid for data loss.
      if normal_observer:
        utilde, new_velocity_model_state = network.velocity_prediction(
            velocity_params, velocity_model_state, velocity_predictor_fn,
            geos_coords, geos_reference_velocity, train=True)
        vi = kgeo.normal_to_3_velocity(utilde, geos.r.data, geos.theta.data, a, M)
        umu = kgeo.normal_to_4_velocity(utilde, geos.r.data, geos.theta.data, a, M)
      else:
        vi, new_velocity_model_state = network.velocity_prediction(
            velocity_params, velocity_model_state, velocity_predictor_fn,
            geos_coords, geos_reference_velocity, train=True)
        umu = kgeo.bl_3_velocity_to_bl_4_velocity(vi, geos.r.data, geos.theta.data, a, M)
    if not doppler:
      g = 1.
    elif learned_doppler:
      g = kgeo.get_doppler_factor_from_umu(umu, k_mu=k_mu_const, E=E_const)
      g = emission.fill_unsupervised(
        jnp.expand_dims(g, axis=-1), geos_coords,
        rmin, rmax, z_width=np.inf, fill_value=1., use_jax=True)[..., 0]
    else:
      g = fixed_g
    (_, images), new_emission_model_state = network.emissions_and_image_plane_prediction(
        emission_params, emission_model_state, emission_predictor_fn,
        data_t, geos_coords, J, g, geos.dtau.data, geos.Sigma.data, train=True)

    image_loss = jnp.mean(jnp.sum(jnp.square(images - y_image), axis=(1, 2)))

    if vis_weight > 0:
      vis_loss = jax.vmap(eht.chi2_vis)(images, y_vis_expanded, y_sigmavis_expanded, A_vis_expanded)
      vis_loss = jnp.mean(vis_loss)
    else:
      vis_loss = 0.

    if amp_weight > 0:
      amp_loss = jax.vmap(eht.chi2_amp)(images, y_amp, y_sigmaamp, A_vis_expanded)
      amp_loss = jnp.mean(amp_loss)
    else:
      amp_loss = 0.
    
    if cp_weight > 0:
      cp_loss = jax.vmap(eht.chi2_cphase)(images, y_cp, y_sigmacp, A_cp)
      cp_loss = jnp.mean(cp_loss)
    else:
      cp_loss = 0.

    if logca_weight > 0:
      logca_loss = jax.vmap(eht.chi2_logcamp)(images, y_logca, y_sigmalogca, A_logca)
      logca_loss = jnp.mean(logca_loss)
    else:
      logca_loss = 0.

    flux_loss = jnp.mean(jax.vmap(eht.chi2_flux)(images, y_flux))

    data_loss = (image_weight * image_loss
                 + vis_weight * vis_loss
                 + amp_weight * amp_loss
                 + cp_weight * cp_loss
                 + logca_weight * logca_loss
                 + flux_weight * flux_loss)

    # PINN loss (i.e., emission prediction should agree with velocity prediction).
    def _predicted_next_emissions_fn(curr_emissions):
      if use_target_velocity:
        predicted_warped_coords = predicted_warped_coords_with_target_velo
      else:
        predicted_warped_coords = learned_velocity_warp_coords_fn(
            velocity_params, velocity_model_state, velocity_predictor_fn,
            emission_coords,
            jnp.array([delta_t_M]),
            t0_M=0,
            train=False)[0]  # train=False to avoid computing batch statistics
      return jax.vmap(emission.interpolate_coords, in_axes=(0, None, None))(
          curr_emissions, emission_coords, predicted_warped_coords)
  
    emissions, _ = network.emissions_prediction(
        emission_params, emission_model_state, emission_predictor_fn,
        t_frames=pinn_t, coords=emission_coords, train=True)
    next_emissions, _ = network.emissions_prediction(
        emission_params, emission_model_state, emission_predictor_fn,
        t_frames=pinn_t + delta_t, coords=emission_coords, train=True)
    # If `pinn_weight` is 0, then `predicted_next_emissions` is set to `next_emissions`.
    predicted_next_emissions = jax.lax.cond(
      pinn_weight > 0, _predicted_next_emissions_fn, lambda _: next_emissions, emissions)
    
    if blur_for_pinn_loss:
      blurred_predicted_next_emissions = utils.gaussian_blur(predicted_next_emissions)
      blurred_next_emissions = utils.gaussian_blur(next_emissions)
    else:
      blurred_predicted_next_emissions = predicted_next_emissions
      blurred_next_emissions = next_emissions
    if l1_pinn_loss:
      pinn_loss = jnp.mean(jnp.sum(jnp.abs(blurred_predicted_next_emissions - blurred_next_emissions), axis=(1, 2, 3)))
    else:
      pinn_loss = jnp.mean(jnp.sum(jnp.square(blurred_predicted_next_emissions - blurred_next_emissions), axis=(1, 2, 3)))

    total_loss = data_weight * data_loss + velo_weight * velo_loss + pinn_weight * pinn_loss

    return total_loss, (new_emission_model_state, new_velocity_model_state,
                        images, emissions, next_emissions, predicted_next_emissions,
                        pinn_loss, velo_loss, data_loss,
                        image_loss, vis_loss, amp_loss, cp_loss, logca_loss, flux_loss)
  return loss_fn


def get_step_fn(predictor_fn_list,
                optimizer_list,
                geos,
                learned_velocity_warp_coords_fn,
                emission_coords,
                delta_t,
                learned_doppler,
                fixed_g,
                doppler,
                normal_observer,
                retrograde,
                fac_subkep,
                beta_phi,
                beta_r,
                rmin,
                rmax,
                eht,
                image_weight,
                vis_weight,
                amp_weight,
                cp_weight,
                logca_weight,
                flux_weight,
                use_target_velocity=False,
                J=1.,
                t_units=units.hr,
                velo_loss_grid='geos',
                l1_pinn_loss=False,
                blur_for_pinn_loss=False,
                will_pmap=True):
  loss_fn = get_loss_fn(
    geos=geos,
    learned_velocity_warp_coords_fn=learned_velocity_warp_coords_fn,
    emission_coords=emission_coords,
    delta_t=delta_t,
    learned_doppler=learned_doppler,
    fixed_g=fixed_g,
    retrograde=retrograde,
    fac_subkep=fac_subkep,
    beta_phi=beta_phi,
    beta_r=beta_r,
    rmin=rmin,
    rmax=rmax,
    eht=eht,
    image_weight=image_weight,
    vis_weight=vis_weight,
    amp_weight=amp_weight,
    cp_weight=cp_weight,
    logca_weight=logca_weight,
    flux_weight=flux_weight,
    doppler=doppler,
    use_target_velocity=use_target_velocity,
    J=J,
    t_units=t_units,
    normal_observer=normal_observer,
    velo_loss_grid=velo_loss_grid,
    l1_pinn_loss=l1_pinn_loss,
    blur_for_pinn_loss=blur_for_pinn_loss)

  def step_fn(state, data_t, pinn_t, 
              y_image, y_vis_expanded, y_sigmavis_expanded,
              y_amp, y_sigmaamp, y_cp, y_sigmacp, y_logca, y_sigmalogca, y_flux,
              A_vis_expanded, A_cp, A_logca):
    """Optimization step.

    Args:
      state: Current state.
      data_target: Image-plane at sampled `data_t` times for data loss, of shape (batch, alpha, beta).
      data_t: Sampled times for data loss, of shape (batch,).
      pinn_t: Sampled times for PINN loss, of shape (batch,).

    Returns: Optimization step function.
    """
    emission_params = state.emission.params
    emission_model_state = state.emission.model_state
    emission_opt_state = state.emission.opt_state

    velocity_params = state.velocity.params
    velocity_model_state = state.velocity.model_state
    velocity_opt_state = state.velocity.opt_state

    (loss, (new_emission_model_state, new_velocity_model_state,
            images, emissions, next_emissions, predicted_next_emissions,
            pinn_loss, velo_loss, data_loss, image_loss, vis_loss,
            amp_loss, cp_loss, logca_loss, flux_loss)), (emission_grads, velocity_grads) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
        emission_params, velocity_params,
        [emission_model_state, velocity_model_state],
        predictor_fn_list,
        data_t, pinn_t,
        y_image, y_vis_expanded, y_sigmavis_expanded,
        y_amp, y_sigmaamp, y_cp, y_sigmacp, y_logca, y_sigmalogca, y_flux,
        A_vis_expanded, A_cp, A_logca,
        state.data_weight, state.pinn_weight, state.velo_weight)

    if will_pmap:
      emission_grads = jax.lax.pmean(emission_grads, axis_name='batch')
      velocity_grads = jax.lax.pmean(velocity_grads, axis_name='batch')

    # Apply updates.
    emission_updates, new_emission_opt_state = optimizer_list[0].update(
      emission_grads, emission_opt_state, emission_params)
    new_emission_params = optax.apply_updates(emission_params, emission_updates)

    velocity_updates, new_velocity_opt_state = optimizer_list[1].update(
      velocity_grads, velocity_opt_state, velocity_params)
    new_velocity_params = optax.apply_updates(velocity_params, velocity_updates)

    step = state.step + 1
    new_emission_state = state.emission.replace(
        opt_state=new_emission_opt_state,
        params=new_emission_params,
        model_state=new_emission_model_state)
    new_velocity_state = state.velocity.replace(
        opt_state=new_velocity_opt_state,
        params=new_velocity_params,
        model_state=new_velocity_model_state)
    state = state.replace(
        step=step,
        emission=new_emission_state,
        velocity=new_velocity_state)

    return loss, pinn_loss, velo_loss, data_loss, image_loss, vis_loss, amp_loss, cp_loss, logca_loss, flux_loss,state, images, emissions, next_emissions, predicted_next_emissions

  return step_fn

