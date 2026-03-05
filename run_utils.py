import os

from astropy import units
import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from pidef import constants as consts
from pidef import emission
from pidef import kgeo
from pidef import network
from pidef import utils
from pidef import visualization


def is_coordinator():
  return jax.process_index() == 0


def get_simdir(config, basedir):
  """Returns the directory for the simulation. This will contain subdirectories for each experiment."""
  exp_descr = (
    f'simfac={config.sim.fac_subkep:g}'
    f'_simbeta={config.sim.beta:g}'
  )
  if config.sim.type == 'simple':
    exp_descr += (
      f'_n={config.sim.num_emissions:g}'
      f'_r={config.sim.min_orbit_radius:g}-{config.sim.max_orbit_radius:g}'
      f'_s={config.sim.min_std:g}-{config.sim.max_std:g}'
    )

  exp_descr += (
    f'_i{config.sim.inclination_deg:g}'
    f'_tadv={config.sim.tadv:g}'
    f'_seed{config.seed:g}'
  )

  return os.path.join(basedir, exp_descr)


def get_measdir(config, basedir):
  """Returns the directory for the measurement setting (image or EHT array)."""
  simdir = get_simdir(config, basedir)
  # Add descriptor for measurement setting (image or EHT array).
  vis_weight = config.opt.vis_weight
  cp_weight = config.opt.cp_weight
  logca_weight = config.opt.logca_weight
  flux_weight = config.opt.flux_weight
  if vis_weight !=0 or cp_weight != 0 or logca_weight != 0 or flux_weight != 0:
    array_name = os.path.splitext(os.path.basename(config.sim.array))[0]
    measdir = os.path.join(simdir, array_name)
  else:
    measdir = os.path.join(simdir, 'image')
  
  if config.sim.flux_multiplier != 1.:
    measdir = f'{measdir}_fluxmult={config.sim.flux_multiplier:g}'

  return measdir


def get_workdir(config, basedir):
  """Returns the directory for the experiment.
  
  Each workdir will have the format {basedir}/{simdescr}/{measdescr}/{hparamdescr},
  where simdescr describes the simulation parameters, measdescr describes the
  measurement setting (image or EHT array), and hparamdescr describes the
  hyperparameters used for this experiment run.
  """
  measdir = get_measdir(config, basedir)

  image_weight = config.opt.image_weight
  vis_weight = config.opt.vis_weight
  amp_weight = config.opt.amp_weight
  cp_weight = config.opt.cp_weight
  logca_weight = config.opt.logca_weight
  flux_weight = config.opt.flux_weight

  def _spin_str():
    if config.opt.spin == config.sim.spin:
      return ''
    return f'_spin={config.opt.spin:g}'
  def _learning_rate_str(emission_lr_init, emission_lr_final, velocity_lr_init, velocity_lr_final):
    def _individual_learning_rate_str(lr_init, lr_final):
      if lr_init == lr_final:
        return f'{lr_init:.0e}'
      return f'{lr_init:.0e}-{lr_final:.0e}'
    emission_learning_rate_str = _individual_learning_rate_str(emission_lr_init, emission_lr_final)
    velocity_learning_rate_str = _individual_learning_rate_str(velocity_lr_init, velocity_lr_final)
    if emission_learning_rate_str == velocity_learning_rate_str:
      return f'lr={emission_learning_rate_str}'
    return f'elr={emission_learning_rate_str}_vlr={velocity_learning_rate_str}'
  def _velo_weight_str():
    if not config.opt.anneal_velo_weight:
      return f'velo={config.opt.velo_weight:g}'
    return f'velo={config.opt.warmup_velo_steps:g}-{config.opt.init_velo_weight:g}-{config.opt.velo_weight:g}-{config.opt.velo_weight_decay_rate:g}'
  def _pinn_weight_str():
    if not config.opt.anneal_pinn_weight:
      pinn_weight_str = f'pinn={config.opt.pinn_weight:g}'
    else:
      pinn_weight_str = f'pinn={config.opt.pinn_weight:g}-{config.opt.pinn_weight_pivot_steps:g}-{config.opt.pinn_weight_anneal_rate:g}'
    if config.opt.l1_pinn_loss:
      pinn_weight_str = 'l1' + pinn_weight_str
    if config.opt.blur_for_pinn_loss:
      pinn_weight_str = 'blur' + pinn_weight_str
    return pinn_weight_str
  def _data_weight_str():
    # Construct string only including non-zero weights.
    data_weight_str = ''
    for weight, name in [(image_weight, 'image'), (vis_weight, 'vis'), (amp_weight, 'amp'), (cp_weight, 'cp'), (logca_weight, 'logca'), (flux_weight, 'flux')]:
      if weight != 0:
        data_weight_str += f'_{name}={weight:g}'
    return data_weight_str

  # Construct hyperparameter descriptor.
  hparam_descr = (
    f'optfac={config.opt.fac_subkep:g}'
    f'_optbeta={config.opt.beta:g}'
    f'{_spin_str()}'
    f'_{_learning_rate_str(config.opt.emission_lr_init, config.opt.emission_lr_final, config.opt.velocity_lr_init, config.opt.velocity_lr_final)}'
    f'_bs={config.opt.batch_size}'
    f'_ezw={config.net.emission_z_width:g}'
    f'_coord={config.net.coordinate_type}'
    f'_dt={config.opt.dt_hr}'
    f'_{_pinn_weight_str()}'
    f'_{_velo_weight_str()}'
    f'{_data_weight_str()}'
  )
  if not config.opt.learned_doppler:
    hparam_descr += f'_fixedg'
  workdir = os.path.join(measdir, hparam_descr)
  return workdir


def get_geodesics(config, spin=None):
  """Returns geodesics data."""
  fov_M = config.sim.fov_M
  geos = kgeo.image_plane_geos(
    spin if spin is not None else config.sim.spin,
    np.deg2rad(config.sim.inclination_deg),
    num_alpha=config.sim.num_alpha,
    num_beta=config.sim.num_beta,
    alpha_range=[-fov_M / 2, fov_M / 2],
    beta_range=[-fov_M / 2, fov_M / 2],
    M=config.sim.M)
  return geos


def get_simulation_data(config, t_frames, geos, random_state):
  """Returns simulated true emissivity field and image-plane measurements."""
  fov_M = config.sim.fov_M

  # Sample true emissions.
  if config.sim.type == 'simple':
    emission_list, t_injection_list = emission.random_emissions(
      t_frames,
      fov_M,
      min_radius=config.sim.min_orbit_radius,
      max_radius=config.sim.max_orbit_radius,
      min_std=config.sim.min_std,
      max_std=config.sim.max_std,
      num_emissions=config.sim.num_emissions,
      rot_axis=[0., 0., 1.],
      random_state=random_state)
  elif config.sim.type == 'grmhd':
    emission_list, t_injection_list = emission.random_emissions_grmhdlike(
      t_frames,
      fov_M,
      min_radius=float(geos.r.min()),
      max_radius=config.sim.max_orbit_radius,
      rot_axis=[0., 0., 1.],
      random_state=random_state)
  
  # Simulate image-plane measurements.
  image_plane = emission.image_plane_dynamics(
    emission_list,
    t_injection_list,
    geos,
    t_frames,
    fac_subkep=config.sim.fac_subkep,
    beta_phi=config.sim.beta,
    beta_r=config.sim.beta,
    dt0=0.001 * units.hr,
    J=1.0,
    slow_light=config.sim.slow_light,
    doppler=config.sim.doppler,
    M=consts.sgra_mass,
    retrograde=config.sim.retrograde)

  return emission_list, t_injection_list, image_plane


def get_emission_coords(emission_xarray):
  """Returns the coordinates of the emission grid."""
  return np.stack(
    np.meshgrid(
      emission_xarray.coords['x'],
      emission_xarray.coords['y'],
      emission_xarray.coords['z'],
      indexing='ij'),
    axis=-1)


def is_eht_setting(config):
  vis_weight = config.opt.vis_weight
  cp_weight = config.opt.cp_weight
  logca_weight = config.opt.logca_weight
  flux_weight = config.opt.flux_weight
  return (vis_weight > 0 or cp_weight > 0 or logca_weight > 0 or flux_weight > 0)


def get_predictors(config, rmin, rmax):
  """Returns the emission and velocity predictors."""
  emission_predictor = network.EmissionPredictor(
    scale=rmax,
    rmin=rmin,
    rmax=rmax,
    z_width=config.net.emission_z_width,
    net_width=config.net.emission_width,
    posenc_deg=config.net.emission_posenc_deg,
    time_posenc_deg=config.net.time_posenc_deg,
    batch_norm=config.net.batch_norm)
  velocity_predictor = network.VelocityPredictor(
    coordinate_type=config.net.coordinate_type,
    rmin=rmin,
    rmax=rmax,
    z_width=config.net.velocity_z_width,
    in_scale=rmax,
    out_r_scale=config.net.velocity_out_r_scale,
    out_phi_scale=config.net.velocity_out_phi_scale,
    posenc_deg=config.net.velocity_posenc_deg,
    batch_norm=config.net.batch_norm,
    residual=config.net.velocity_residual,
    fill_unsupervised=config.net.velocity_fill_unsupervised)
  return emission_predictor, velocity_predictor


def get_init_state_and_optimizers(config, emission_predictor, velocity_predictor, geos):
  state, optimizers = network.init_state_and_optimizer(
    emission_predictor, velocity_predictor,
    geos,
    num_iters=config.opt.niter,
    lr_init=[config.opt.emission_lr_init, config.opt.velocity_lr_init],
    lr_final=[config.opt.emission_lr_final, config.opt.velocity_lr_final],
    lr_decay_schedule=config.opt.lr_decay_schedule,
    lr_decay_begin=config.opt.lr_decay_begin,
    lr_decay_steps=config.opt.lr_decay_steps,
    grad_clip=config.opt.grad_clip,
    pinn_weight=config.opt.pinn_weight,
    velo_weight=config.opt.velo_weight,
    data_weight=config.opt.data_weight,
    seed=config.seed)
  return state, optimizers


def get_solver(solver, scan_stages=True):
  """Return `diffrax.AbstractSolver` instance."""
  if solver == 'Euler':
    return diffrax.Euler()
  try:
    return getattr(diffrax, solver)(scan_stages=scan_stages)
  except:
    return getattr(diffrax, solver)()


def get_stepsize_controller(stepsize_controller, rtol, atol, adjoint_rms_seminorm=False):
  """Return `diffrax.AbstractStepSizeController` instance."""
  if stepsize_controller == 'ConstantStepSize':
    return diffrax.ConstantStepSize()
  elif stepsize_controller == 'PIDController':
    if adjoint_rms_seminorm:
      return diffrax.PIDController(
        norm=diffrax.adjoint_rms_seminorm, rtol=rtol, atol=atol)
    else:
      return diffrax.PIDController(rtol=rtol, atol=atol)
  else:
    raise ValueError(f'Unsupported stepsize controller: {stepsize_controller}')
  

def get_adjoint_solver(adjoint_method, adjoint_solver,
                       adjoint_stepsize_controller,
                       adjoint_rtol, adjoint_atol, adjoint_rms_seminorm):
  """Return `diffrax.AbstractSolver` for the adjoint."""
  if adjoint_method == 'RecursiveCheckpointAdjoint':
    adjoint = diffrax.RecursiveCheckpointAdjoint()
  elif adjoint_method == 'BacksolveAdjoint':
    adjoint_solver = get_solver(adjoint_solver, scan_stages=True)
    adjoint_stepsize_controller = get_stepsize_controller(
      adjoint_stepsize_controller,
      adjoint_rtol,
      adjoint_atol,
      adjoint_rms_seminorm)
    adjoint = diffrax.BacksolveAdjoint(
      solver=adjoint_solver,
      stepsize_controller=adjoint_stepsize_controller)
  else:
    raise ValueError(f'Unsupported adjoint method: {adjoint_method}')
  return adjoint


def get_eval_fn(emission_per_t, emission_coords, image_plane, t_frames,
                vis, fov_M,
                predictor_fn_list,
                geos,
                learned_velocity_warp_coords_fn,
                delta_t,
                learned_doppler, fixed_g, retrograde,
                fac_subkep, beta_phi, beta_r,
                sim_fac_subkep, sim_beta_phi, sim_beta_r,
                rmin, rmax,
                doppler, normal_observer, t_units=units.hr, fps=6):
  emission_predictor = predictor_fn_list[0]
  velocity_predictor = predictor_fn_list[1]
  geos_coords = jnp.stack([geos.x.data, geos.y.data, geos.z.data], axis=-1)
  GM_c3 = consts.GM_c3(consts.sgra_mass).to(t_units)
  delta_t_M = delta_t / GM_c3.value
  a = float(geos.spin)
  M = float(geos.M)

  # Normalize times to start from 0.
  t_frames_normalized = t_frames - t_frames[0]

  target_velocity = kgeo.spherical_velocities(
    geos.r.data, a, geos.theta.data, M,
    fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r, retrograde=retrograde)

  # Velocity loss (i.e., velocity prediction should agree with assumed velocity model).
  if normal_observer:
    # Compute reference velocity (umu -> utildei).
    ut, ur, uth, uph = kgeo.u_general(
        geos.r.data, a, geos.theta.data, M,
        fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r,
        retrograde=retrograde)
    reference_umu = jnp.stack((ut, ur, uth, uph), axis=-1)
    geos_reference_velocity = kgeo.bl_4_velocity_to_normal(
        reference_umu, geos.r.data, geos.theta.data, a, M)
  else:
    geos_reference_velocity = kgeo.spherical_velocitie(
        geos.r.data, a, geos.theta.data, M,
        fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r,
        retrograde=retrograde)
  
  # Get reference velocity for evaluating velocity loss. 
  emission_coords_r = utils.cartesian_to_spherical(emission_coords)[:, :, :, 0]
  emission_coords_theta = utils.cartesian_to_spherical(emission_coords)[:, :, :, 1]
  if normal_observer:
    # Compute reference velocity (umu -> utildei).
    ut, ur, uth, uph = kgeo.u_general(
        emission_coords_r, a, emission_coords_theta, M,
        fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r, retrograde=retrograde)
    reference_umu = jnp.stack((ut, ur, uth, uph), axis=-1)
    velo_reference_velocity = kgeo.bl_4_velocity_to_normal(
        reference_umu, emission_coords_r, emission_coords_theta, a, M)
  else:
    velo_reference_velocity = kgeo.u_general(
        emission_coords_r, a, emission_coords_theta, M,
        fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r, retrograde=retrograde)

  emission_views = []
  for emission_t in emission_per_t:
    # Render current emissivity.
    emission_for_visualizer = emission.interpolate_coords(
      emission_t, emission_coords, np.moveaxis(vis.coords, 0, -1))
    emission_view = vis.render(
      emission_for_visualizer / emission_per_t.max(), facewidth=fov_M, linewidth=0.1,
      bh_radius=2., jit=True).clip(max=1)
    emission_views.append(emission_view)

  emission_coords_r = utils.cartesian_to_spherical(emission_coords)[:, :, :, 0]
  emission_coords_theta = utils.cartesian_to_spherical(emission_coords)[:, :, :, 1]

  def _eval_true_velo_mse(state, est_emission_per_t, thresh=0.001):
    # Evaluate velocity loss on region we care about.
    est_emission_integrated = np.sum(est_emission_per_t, axis=0)

    # Get estimated velocity.
    if normal_observer:
      utildei, _ = network.velocity_prediction(
        state.velocity.params, state.velocity.model_state, velocity_predictor.apply,
        emission_coords, reference_velocity=velo_reference_velocity, train=False)
      vi = kgeo.normal_to_3_velocity(utildei, emission_coords_r, emission_coords_theta, a, M)
    else:
      vi, _ = network.velocity_prediction(
          state.velocity.params, state.velocity.model_state, velocity_predictor.apply,
          emission_coords, reference_velocity=velo_reference_velocity, train=False)
    # Get true velocity.
    true_vi = kgeo.spherical_velocities(
      emission_coords_r, a, emission_coords_theta, M, retrograde=retrograde,
      fac_subkep=sim_fac_subkep, beta_phi=sim_beta_phi, beta_r=sim_beta_r)

    eval_vi = vi[est_emission_integrated > thresh]
    eval_true_vi = true_vi[est_emission_integrated > thresh]
    return np.mean(np.square(eval_vi - eval_true_vi))

  def eval_fn(state, movie=False):
    # Evaluate velocity loss.
    if normal_observer:
      utilde, _ = network.velocity_prediction(
          state.velocity.params, state.velocity.model_state, velocity_predictor.apply,
          geos_coords, geos_reference_velocity, train=False)
      vi = kgeo.normal_to_3_velocity(utilde, geos.r.data, geos.theta.data, a, M)
      umu = kgeo.normal_to_4_velocity(utilde, geos.r.data, geos.theta.data, a, M)
    else:
      vi, _ = network.velocity_prediction(
          state.velocity.params, state.velocity.model_state, velocity_predictor.apply,
          geos_coords, geos_reference_velocity, train=False)
      umu = kgeo.bl_3_velocity_to_bl_4_velocity(vi, geos.r.data, geos.theta.data, a, M)
    velo_loss = np.sum(np.square(vi - target_velocity))

    # Get (possibly learned) Doppler boosting factor.
    if not doppler:
      g = 1.
    elif learned_doppler:
      g = kgeo.get_doppler_factor_from_umu(umu, k_mu=kgeo.wave_vector(geos).data, E=geos.E.data)
      g = emission.fill_unsupervised(
        jnp.expand_dims(g, axis=-1), geos_coords,
        rmin, rmax, z_width=np.inf, fill_value=1., use_jax=True)[..., 0]
    else:
      g = fixed_g
    
    # Reconstructed image plane
    est_image_plane = []
    for t_frame in t_frames_normalized:
      images, _ = network.image_plane_prediction(
        state.emission.params, state.emission.model_state, emission_predictor.apply,
        t_frame, geos_coords, 
        g, geos.dtau.data, geos.Sigma.data, J=1., train=False)
      est_image_plane.append(images[0])
    est_image_plane = np.array(est_image_plane)

    # Estimated 3D emissions
    est_emission_per_t = []
    for t_frame in t_frames_normalized:
      est_emission = network.sample_3d_grid(
        emission_predictor.apply, state.emission.params, state.emission.model_state,
        t_frame, coords=emission_coords, train=False)
      est_emission_per_t.append(est_emission)
    est_emission_per_t = np.array(est_emission_per_t)

    image_plane_absdiff = np.abs(est_image_plane - image_plane)
    emission_absdiff = np.abs(est_emission_per_t - emission_per_t)

    # Evaluate PINN loss on all sampled times.
    next_emission_per_t = []
    for t_frame in t_frames_normalized:
      next_emission = network.sample_3d_grid(
        emission_predictor.apply, state.emission.params, state.emission.model_state,
        t_frame.value + delta_t, coords=emission_coords, train=False)
      next_emission_per_t.append(next_emission)
    next_emission_per_t = np.array(next_emission_per_t)

    predicted_warped_coords = learned_velocity_warp_coords_fn(
      state.velocity.params, state.velocity.model_state, velocity_predictor.apply,
      emission_coords,
      jnp.array([delta_t_M]),
      t0_M=0,
      train=False)[0]  # train=False to avoid computing batch statistics
    predicted_next_emission_per_t = jax.vmap(emission.interpolate_coords, in_axes=(0, None, None))(
      est_emission_per_t, emission_coords, predicted_warped_coords)

    # Evaluate velocity MSE on region we care about.
    true_velo_mse = _eval_true_velo_mse(state, est_emission_per_t)

    if movie:
      # Render views of estimated emissions.
      est_emission_views = []
      for est_emission_t in est_emission_per_t:
        est_emission_for_visualizer = emission.interpolate_coords(
          est_emission_t, emission_coords, np.moveaxis(vis.coords, 0, -1))
        est_emission_view = vis.render(
          est_emission_for_visualizer / emission_per_t.max(), facewidth=fov_M, linewidth=0.1,
          bh_radius=2., jit=True).clip(max=1)
        est_emission_views.append(est_emission_view)

      # Render views of abs. diff. in emissivity.
      emission_diff_views = []
      for emission_diff_t in emission_absdiff:
        emission_diff_for_visualizer = emission.interpolate_coords(
          emission_diff_t, emission_coords, np.moveaxis(vis.coords, 0, -1))
        emission_diff_view = vis.render(
            emission_diff_for_visualizer / emission_absdiff.max(), facewidth=fov_M, linewidth=0.1,
            bh_radius=2., jit=True, cmap='magma').clip(max=1)
        emission_diff_views.append(emission_diff_view)

      ani = make_eval_movie(
        emission_views, est_emission_views, emission_diff_views,
        emission_absdiff, image_plane, est_image_plane, image_plane_absdiff,
        t_frames, max_emissivity=emission_per_t.max(), fps=fps)
    else:
      ani = None

    return est_image_plane, est_emission_per_t, next_emission_per_t, predicted_next_emission_per_t, velo_loss, true_velo_mse, ani

  return eval_fn


def plot_progress(step, loss_dict, true_image, est_image, image_t,
                  true_curr_emission_view, curr_emission_view,
                  next_emission_view, predicted_next_emission_view,
                  pinn_t, pinn_dt,
                  est_velocity, true_velocity, target_velocity,
                  pinn_weight, velo_weight, data_weight,
                  image_weight, vis_weight, cp_weight, logca_weight, flux_weight):
  plt.close('all')
  fig, axs = plt.subplots(6, 4, figsize=(20, 27))
  fig.suptitle(f'Step {step}', y=0.92, fontsize=21)

  # Plot loss curves.
  ax = axs[0, 0]
  ax.plot(loss_dict['total'])
  ax.set_title('Total Loss')
  ax = axs[0, 1]
  ax.plot(loss_dict['pinn'])
  ax.set_title(f'PINN Loss (w={pinn_weight:g})')
  ax = axs[0, 2]
  ax.plot(loss_dict['velo'])
  ax.set_title(f'Velocity Loss (w={velo_weight:g})')
  ax = axs[0, 3]
  ax.plot(loss_dict['data'])
  ax.set_title(f'Data Loss (w={data_weight:g})')
  
  # Plot chi2 curves.
  ax = axs[1, 0]
  ax.plot(loss_dict['vis'])
  ax.set_title(r'Vis. $\chi^2$' + f' (w={vis_weight:g})')
  ax = axs[1, 1]
  ax.plot(loss_dict['cp'])
  ax.set_title(r'CP $\chi^2$' + f' (w={cp_weight:g})')
  ax = axs[1, 2]
  ax.plot(loss_dict['logca'])
  ax.set_title(r'LogCA $\chi^2$' + f' (w={logca_weight:g})')
  ax = axs[1, 3]
  ax.plot(loss_dict['flux'])
  ax.set_title(r'Flux $\chi^2$' + f' (w={flux_weight:g})')

  # Plot image plane.
  ax = axs[2, 0]
  ax.plot(loss_dict['image'])
  ax.set_title(r'Image Data Loss' + f' (w={image_weight:g})')
  ax = axs[2, 1]
  p = ax.imshow(true_image, cmap='hot')
  fig.colorbar(p, ax=ax)
  ax.axis('off')
  ax.set_title(f'Observed Image ({image_t:.2f})')
  ax = axs[2, 2]
  p = ax.imshow(est_image, cmap='hot')
  fig.colorbar(p, ax=ax)
  ax.axis('off')
  ax.set_title(f'Reconstructed Image ({image_t:.2f})')

  axs[2, 2].axis('off')
  axs[2, 3].axis('off')

  # Plot 3D emission at sampled time.
  ax = axs[3, 0]
  ax.imshow(true_curr_emission_view)
  ax.axis('off')
  ax.set_title(f'$t=${pinn_t:.2f}\ntrue')
  ax = axs[3, 1]
  ax.imshow(curr_emission_view)
  ax.axis('off')
  ax.set_title(f'$t=${pinn_t:.2f}\ne prediction')

  # Compare next emission prediction to velocity-predicted next emission.
  ax = axs[3, 2]
  ax.imshow(next_emission_view)
  ax.axis('off')
  ax.set_title(f'$t=${pinn_t + pinn_dt:.2f}\ne prediction')
  ax = axs[3, 3]
  ax.imshow(predicted_next_emission_view)
  ax.axis('off')
  ax.set_title(f'$t=${pinn_t + pinn_dt:.2f}\nv prediction')

  # Compare velocity fields.
  est_vr = est_velocity[:, :, :, 0]
  true_vr = true_velocity[:, :, :, 0]
  target_vr = target_velocity[:, :, :, 0]
  est_vphi = est_velocity[:, :, :, 2]
  true_vphi = true_velocity[:, :, :, 2]
  target_vphi = target_velocity[:, :, :, 2]

  rmin = min(est_vr.min(), true_vr.min(), target_vr.min())
  rmax = max(est_vr.max(), true_vr.max(), target_vr.max())
  phimin = min(est_vphi.min(), true_vphi.min(), target_vphi.min())
  phimax = max(est_vphi.max(), true_vphi.max(), target_vphi.max())

  # Compare estimated velocity field to true velocity field.
  ax = axs[4, 0]
  p = ax.imshow(visualization.make_grid(true_vr, axis=2), clim=(rmin, rmax))
  fig.colorbar(p, ax=ax)
  ax.axis('off')
  ax.set_title('True $v_r$')
  ax = axs[4, 1]
  p = ax.imshow(visualization.make_grid(est_vr, axis=2), clim=(rmin, rmax))
  fig.colorbar(p, ax=ax)
  ax.axis('off')
  ax.set_title('Estimated $v_r$')
  ax = axs[4, 2]
  p = ax.imshow(visualization.make_grid(target_vr, axis=2), clim=(rmin, rmax))
  fig.colorbar(p, ax=ax)
  ax.axis('off')
  ax.set_title('Prior $v_r$')
  ax = axs[4, 3]
  ax.axis('off')

  # Compare estimated velocity field to target velocity field.
  ax = axs[5, 0]
  p = ax.imshow(visualization.make_grid(true_vphi, axis=2), clim=(phimin, phimax))
  fig.colorbar(p, ax=ax)
  ax.axis('off')
  ax.set_title('True $v_\phi$')
  ax = axs[5, 1]
  p = ax.imshow(visualization.make_grid(est_vphi, axis=2), clim=(phimin, phimax))
  fig.colorbar(p, ax=ax)
  ax.axis('off')
  ax.set_title('Estimated $v_\phi$')
  ax = axs[5, 2]
  p = ax.imshow(visualization.make_grid(target_vphi, axis=2), clim=(phimin, phimax))
  fig.colorbar(p, ax=ax)
  ax.axis('off')
  ax.set_title('Prior $v_\phi$')
  ax = axs[5, 3]
  ax.axis('off')

  return fig


def make_eval_movie(emission_views, est_emission_views, emission_diff_views,
                    emission_absdiff, image_plane, est_image_plane,
                    image_plane_absdiff, t_frames, max_emissivity, fps=6):
  fig, axs = plt.subplots(2, 3, figsize=(12, 6))
  for ax in axs.ravel():
    ax.axis('off')

  emiss_clim = (0, max_emissivity)
  image_clim = (0, image_plane.max())

  emiss_true = axs[0, 0].imshow(emission_views[0])
  emiss_est = axs[0, 1].imshow(est_emission_views[0])
  emiss_absdiff = axs[0, 2].imshow(emission_diff_views[0])
  image_true = axs[1, 0].imshow(image_plane[0], clim=image_clim, cmap='hot')
  image_est = axs[1, 1].imshow(est_image_plane[0], clim=image_clim, cmap='hot')
  image_absdiff = axs[1, 2].imshow(image_plane_absdiff[0], clim=(0, image_plane_absdiff.max()), cmap='magma')

  axs[0, 0].set_title(f'True\n$t=${t_frames[0]:.2f}', fontsize=21)
  axs[0, 1].set_title(f'Estimated\n$t=${t_frames[0]:.2f}', fontsize=21)
  axs[0, 2].set_title(f'|Estimated - True|\n$t=${t_frames[0]:.2f}', fontsize=21)

  for ax, im in zip(axs[0, :2], (emiss_true, emiss_est)):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    norm = Normalize(vmin=emiss_clim[0], vmax=emiss_clim[1])
    sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)

  ax = axs[0, 2]
  im = emiss_absdiff
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  norm = Normalize(vmin=0, vmax=emission_absdiff.max())
  sm = plt.cm.ScalarMappable(cmap='magma', norm=norm)
  sm.set_array([])
  fig.colorbar(sm, cax=cax)

  for ax, im in zip(axs[1], (image_true, image_est, image_absdiff)):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

  def update_img(n):
    emiss_true.set_data(emission_views[n])
    emiss_est.set_data(est_emission_views[n])
    emiss_absdiff.set_data(emission_diff_views[n])
    image_true.set_data(image_plane[n])
    image_est.set_data(est_image_plane[n])
    image_absdiff.set_data(image_plane_absdiff[n])
    axs[0, 0].set_title(f'True\n$t=${t_frames[n]:.2f}', fontsize=21)
    axs[0, 1].set_title(f'Estimated\n$t=${t_frames[n]:.2f}', fontsize=21)
    axs[0, 2].set_title(f'|True - Estimated|\n$t=${t_frames[n]:.2f}', fontsize=21)
    return

  plt.close(fig)
  return animation.FuncAnimation(fig, update_img, len(t_frames), interval=1e3 / fps)


def get_velocity_eval_mask(emission_per_t, thresh=0.01):
  """Get mask for evaluating velocity only on regions with non-negligible emissivity."""
  thresh = 0.01
  velocity_mask = np.zeros(emission_per_t[0].shape)
  for est_emission in emission_per_t:
    velocity_mask += est_emission > thresh
  velocity_mask = velocity_mask > 0
  return velocity_mask


def weight_annealing_fn(step, decay_rate, warmup_steps=0, initial_weight=1., final_weight=1e-8):
  """Expoential weight-annealing function."""
  return np.where(
    step < warmup_steps,
    initial_weight,
    final_weight + (initial_weight - final_weight) * np.exp(-decay_rate * (step - warmup_steps)))


def sigmoidal_weight_annealing_fn(step, rate, pivot_steps, final_weight=1.):
  """Sigmoidal weight-annealing function."""
  return final_weight / (1 + np.exp(-(step - pivot_steps) * rate))


def linear_weight_annealing_fn(step, init_weight, final_weight, warmup_steps, final_steps):
  """Linear weight-annealing function."""
  return np.where(
    step < warmup_steps,
    init_weight,
    np.where(
      step > final_steps,
      final_weight,
      init_weight + (final_weight - init_weight) * (step - warmup_steps) / (final_steps - warmup_steps)
    )
  )