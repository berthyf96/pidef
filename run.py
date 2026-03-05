"""Fit a PIDEF model of 3D dynamic emission and velocity to simulated EHT or image-plane measurements."""
import logging
import os
from pathlib import Path
import pickle
import time

from absl import app
from absl import flags
from astropy import units
import ehtim as eh
import flax
from flax.training import orbax_utils
import jax
from ml_collections.config_flags import config_flags
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm

from pidef import constants as consts
from pidef import emission
from pidef import kgeo
from pidef import losses
from pidef import network
from pidef import visualization
from pidef.eht import EHT

import run_utils

_CONFIG = config_flags.DEFINE_config_file('config', None, 'Experiment config.')
_WORKDIR = flags.DEFINE_string('workdir', 'runs/', 'Working directory.')

GM_c3 = consts.GM_c3(consts.sgra_mass).to(units.hr)


def train(emission_predictor, velocity_predictor,
          emission_list, t_injection_list, image_plane, t_frames,
          true_emission_per_t, eht):
  train_start_time = time.perf_counter()

  config = _CONFIG.value
  is_eht_setting = run_utils.is_eht_setting(config)

  sim_geos = run_utils.get_geodesics(config, spin=config.sim.spin)
  opt_geos = run_utils.get_geodesics(config, spin=config.opt.spin)

  emission_coords = run_utils.get_emission_coords(emission_list[0])
  rmin = float(sim_geos.r.min())  # minimum recovery radius
  rmax = 0.6 * config.sim.fov_M  # maximum recovery radius
  nt = len(t_frames)
  tstart = t_frames[0]

  workdir = run_utils.get_workdir(config, _WORKDIR.value)
  simdir = run_utils.get_simdir(config, _WORKDIR.value)
  measdir = run_utils.get_measdir(config, _WORKDIR.value)
  ckpt_dir = os.path.join(workdir, 'checkpoints')
  progress_dir = os.path.join(workdir, 'progress')
  movie_dir = os.path.join(workdir, 'movies')

  # Get EHT measurements.
  (y_vis_expanded, y_sigmavis_expanded, y_amp, y_sigmaamp, y_cp, y_sigmacp,
   y_logca, y_sigmalogca, y_flux) = eht.measure(
    image_plane,
    ampcal=not config.sim.station_noise,
    phasecal=not config.sim.station_noise,
    dcal=not config.sim.station_noise)

  # Create visualizer.
  vis = visualization.VolumeVisualizer(128, 128, 128)
  vis.set_view(
    cam_r=37.,
    domain_r=config.sim.fov_M / 2,
    azimuth=np.deg2rad(0),
    zenith=np.deg2rad(60),
    up=np.array([0., 0., 1.]))

  # Save video of image plane in the case of image-plane measurements.
  fps = max(int(nt / 10), 1)
  if not is_eht_setting:
    image_plane_ani = visualization.image_movie(
      image_plane, t_frames, fps=fps, cbar=True)
    image_plane_ani.save(os.path.join(measdir, 'image.mp4'))

  # Save video of blurred image plane in the case of EHT measurements.
  if is_eht_setting:
    im = eh.image.make_empty(
      eht.imsize, eht.fov, eht.ra, eht.dec, eht.rf, eht.source, pulse=eht.pulse, mjd=eht.mjd)
    blurred_image_plane = []
    for image in image_plane:
      im.ivec = image.reshape(-1)
      im = im.blur_circ(eht.ref_obs_list[0].res())
      blurred_image_plane.append(im.ivec.reshape(eht.imsize, eht.imsize))
    blurred_image_plane_ani = visualization.image_movie(
      blurred_image_plane, t_frames, fps=fps, cbar=True)
    blurred_image_plane_ani.save(os.path.join(measdir, 'blurred_image.mp4'))
    np.save(os.path.join(measdir, 'blurred_image_plane.npy'), blurred_image_plane)

  # Save video of true 3D emissivity.
  emission_views = []
  for emission_t in tqdm(true_emission_per_t, desc='Rendering true emissivity'):
    # Render current emissivity.
    emission_for_visualizer = emission.interpolate_coords(
      emission_t, emission_coords, np.moveaxis(vis.coords, 0, -1))
    emission_view = vis.render(
      emission_for_visualizer / true_emission_per_t.max(), facewidth=config.sim.fov_M, linewidth=0.1,
      bh_radius=2., jit=True).clip(max=1)
    emission_views.append(emission_view)
  np.save(os.path.join(simdir, 'emission_views.npy'), emission_views)
  emission_ani = visualization.image_movie(emission_views, t_frames, fps=fps)
  emission_ani.save(os.path.join(simdir, 'emissivity.mp4'))

  # Create checkpoint manager.
  checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
  ckpt_mgr = ocp.CheckpointManager(
    Path(ckpt_dir).resolve(),
    checkpointer,
    options=ocp.CheckpointManagerOptions(max_to_keep=config.opt.nckpt))

  # Initialize params.
  state, optimizers = run_utils.get_init_state_and_optimizers(
    config, emission_predictor, velocity_predictor, opt_geos)

  # Learned velocity coordinate warping function.
  logging.info('ODE solver: %s', config.ode.solver)
  logging.info('ODE step-size controller: %s', config.ode.stepsize_controller)
  logging.info('ODE adjoint method: %s', config.ode.adjoint_method)
  solver = run_utils.get_solver(config.ode.solver)
  stepsize_controller = run_utils.get_stepsize_controller(
    config.ode.stepsize_controller,
    config.ode.rtol,
    config.ode.atol)
  adjoint = run_utils.get_adjoint_solver(
    config.ode.adjoint_method,
    config.ode.adjoint_solver,
    config.ode.adjoint_stepsize_controller,
    config.ode.adjoint_rtol,
    config.ode.adjoint_atol,
    config.ode.adjoint_rms_seminorm)
  dt0_M = config.ode.dt0_hr / GM_c3.value if config.ode.dt0_hr > 0 else None
  learned_velocity_warp_coords = network.get_learned_velocity_warp_coords_fn(
    solver,
    stepsize_controller,
    adjoint,
    fac_subkep=config.opt.fac_subkep,
    beta_phi=config.opt.beta,
    beta_r=config.opt.beta,
    dt0_M=dt0_M,
    normal_observer=config.opt.normal_observer,
    retrograde=config.sim.retrograde,
    a=float(opt_geos.spin),
    M=float(opt_geos.M))
  
  # Get fixed Doppler boosting factor in case `learned_doppler` is `False`.
  # We use the true simulation parameters in this case.
  fixed_g = kgeo.get_doppler_factor(
    sim_geos,
    fac_subkep=config.sim.fac_subkep,
    beta_phi=config.sim.beta,
    beta_r=config.sim.beta,
    retrograde=config.sim.retrograde)

  # Get step function.
  step_fn = losses.get_step_fn(
    [emission_predictor.apply, velocity_predictor.apply],
    optimizers,
    opt_geos,
    learned_velocity_warp_coords,
    emission_coords,
    config.opt.dt_hr,
    config.opt.learned_doppler,
    fixed_g=fixed_g,
    normal_observer=config.opt.normal_observer,
    retrograde=config.sim.retrograde,
    fac_subkep=config.opt.fac_subkep,
    beta_phi=config.opt.beta,
    beta_r=config.opt.beta,
    rmin=rmin,
    rmax=rmax,
    eht=eht,
    image_weight=config.opt.image_weight,
    vis_weight=config.opt.vis_weight,
    amp_weight=config.opt.amp_weight,
    cp_weight=config.opt.cp_weight,
    logca_weight=config.opt.logca_weight,
    flux_weight=config.opt.flux_weight,
    doppler=config.sim.doppler,
    use_target_velocity=config.opt.use_target_velocity,
    J=1.,
    t_units=t_frames.unit,
    l1_pinn_loss=config.opt.l1_pinn_loss,
    blur_for_pinn_loss=config.opt.blur_for_pinn_loss,
    velo_loss_grid=config.opt.velo_loss_grid)

  # Set up objects for monitoring progress.
  loss_dict = {
    'total': [],
    'pinn': [],
    'velo': [],
    'data': [],
    'image': [],
    'vis': [],
    'amp': [],
    'cp': [],
    'logca': [],
    'flux': [],
  }
  val_dict = {
    'steps': [],
    'mse_image_plane': [],
    'mse_emission': [],
    'pinn_loss': [],
    'velo_loss': [],
    'true_velo_mse': [],
    'chisq': [],
  }
  eval_fn = run_utils.get_eval_fn(
    true_emission_per_t,
    emission_coords,
    image_plane,
    t_frames,
    vis,
    config.sim.fov_M,
    [emission_predictor, velocity_predictor],
    opt_geos,
    learned_velocity_warp_coords,
    delta_t=config.opt.dt_hr,
    learned_doppler=config.opt.learned_doppler,
    fixed_g=fixed_g,
    retrograde=config.sim.retrograde,
    fac_subkep=config.opt.fac_subkep,
    beta_phi=config.opt.beta,
    beta_r=config.opt.beta,
    sim_fac_subkep=config.sim.fac_subkep,
    sim_beta_phi=config.sim.beta,
    sim_beta_r=config.sim.beta,
    rmin=rmin,
    rmax=rmax,
    doppler=config.sim.doppler,
    normal_observer=config.opt.normal_observer)

  # Load from checkpoint.
  latest_step = ckpt_mgr.latest_step()
  if latest_step is not None:
    restore_args = orbax_utils.restore_args_from_target(state, mesh=None)
    state = ckpt_mgr.restore(
      latest_step, items=state, restore_kwargs={'restore_args': restore_args})
    with open(os.path.join(progress_dir, 'loss_dict.pkl'), 'rb') as fp:
      loss_dict = pickle.load(fp) 
    with open(os.path.join(progress_dir, 'val_dict.pkl'), 'rb') as fp:
      val_dict = pickle.load(fp)

  # Prepare for parallelization.
  pstep_fn = jax.pmap(step_fn, axis_name='batch')
  pstate = flax.jax_utils.replicate(state)
  pbatch_shape = (jax.local_device_count(), config.opt.batch_size // jax.local_device_count())

  def _get_pbatch(dataset, batch_indices):
    batch = dataset[batch_indices]
    return batch.reshape(*pbatch_shape, *dataset.shape[1:])

  # Train.
  init_step = flax.jax_utils.unreplicate(pstate.step)
  rng = flax.jax_utils.unreplicate(pstate.rng)
  step = init_step
  while step < config.opt.niter:
    start_time = time.perf_counter()

    # Sample measurements.
    rng, step_rng = jax.random.split(rng)
    batch_indices = jax.random.choice(
      step_rng, np.arange(nt), (config.opt.batch_size,), replace=False)

    pbatch_image = _get_pbatch(image_plane, batch_indices)
    pbatch_A_vis = _get_pbatch(eht.A_vis_expanded, batch_indices)
    pbatch_A_cp = _get_pbatch(eht.A_cp, batch_indices)
    pbatch_A_logca = _get_pbatch(eht.A_logca, batch_indices)
    pbatch_y_vis = _get_pbatch(y_vis_expanded, batch_indices)
    pbatch_y_sigmavis = _get_pbatch(y_sigmavis_expanded, batch_indices)
    pbatch_y_amp = _get_pbatch(y_amp, batch_indices)
    pbatch_y_sigmaamp = _get_pbatch(y_sigmaamp, batch_indices)
    pbatch_y_cp = _get_pbatch(y_cp, batch_indices)
    pbatch_y_sigmacp = _get_pbatch(y_sigmacp, batch_indices)
    pbatch_y_logca = _get_pbatch(y_logca, batch_indices)
    pbatch_y_sigmalogca = _get_pbatch(y_sigmalogca, batch_indices)
    pbatch_y_flux = _get_pbatch(y_flux, batch_indices)

    batch_data_t = t_frames[batch_indices]
    # Normalize times to start from 0.
    batch_data_t = batch_data_t - tstart
    pbatch_data_t = batch_data_t.reshape(pbatch_shape)

    # Sample continuous time values for evaluating PINN loss.
    rng, step_rng = jax.random.split(rng)
    batch_pinn_t = jax.random.uniform(
      step_rng, (config.opt.batch_size,), minval=t_frames[0], maxval=t_frames[-1])
    # Normalize times to start from 0.
    batch_pinn_t = batch_pinn_t - tstart
    pbatch_pinn_t = batch_pinn_t.reshape(pbatch_shape)

    # Anneal velocity weight.
    if config.opt.anneal_velo_weight:
      velo_weight = run_utils.weight_annealing_fn(
        step,
        warmup_steps=config.opt.warmup_velo_steps,
        initial_weight=config.opt.init_velo_weight,
        final_weight=config.opt.velo_weight,
        decay_rate=config.opt.velo_weight_decay_rate)
      pstate = pstate.replace(
        velo_weight=flax.jax_utils.replicate(velo_weight))

    (ploss, pinn_ploss, velo_ploss, data_ploss, image_ploss, vis_ploss, amp_ploss, cp_ploss, logca_ploss, flux_ploss,
      pstate, pimages, pemissions, next_pemissions, predicted_next_pemissions) = pstep_fn(
        pstate, pbatch_data_t, pbatch_pinn_t, pbatch_image, pbatch_y_vis, pbatch_y_sigmavis, pbatch_y_amp, pbatch_y_sigmaamp,
        pbatch_y_cp, pbatch_y_sigmacp, pbatch_y_logca, pbatch_y_sigmalogca, pbatch_y_flux,
        pbatch_A_vis, pbatch_A_cp, pbatch_A_logca)

    loss = flax.jax_utils.unreplicate(ploss).item()
    pinn_loss = flax.jax_utils.unreplicate(pinn_ploss).item()
    velo_loss = flax.jax_utils.unreplicate(velo_ploss).item()
    data_loss = flax.jax_utils.unreplicate(data_ploss).item()
    image_loss = flax.jax_utils.unreplicate(image_ploss).item()
    vis_loss = flax.jax_utils.unreplicate(vis_ploss).item()
    amp_loss = flax.jax_utils.unreplicate(amp_ploss).item()
    cp_loss = flax.jax_utils.unreplicate(cp_ploss).item()
    logca_loss = flax.jax_utils.unreplicate(logca_ploss).item()
    flux_loss = flax.jax_utils.unreplicate(flux_ploss).item()
    loss_dict['total'].append(loss)
    loss_dict['pinn'].append(pinn_loss)
    loss_dict['velo'].append(velo_loss)
    loss_dict['data'].append(data_loss)
    loss_dict['image'].append(image_loss)
    loss_dict['vis'].append(vis_loss)
    loss_dict['amp'].append(amp_loss)
    loss_dict['cp'].append(cp_loss)
    loss_dict['logca'].append(logca_loss)
    loss_dict['flux'].append(flux_loss)

    # Check for NaNs in loss.
    if np.any(np.isnan(data_loss)) or np.any(np.isnan(pinn_loss)) or np.any(np.isnan(velo_loss)):
      raise RuntimeError(f'NaN loss detected at step {step} (data: {data_loss:g}, pinn: {pinn_loss:g}, velo: {velo_loss:g})')
    
    # Check for NaNs in new parameters.
    new_velo_state_has_nan_or_inf = jax.tree_util.tree_reduce(
      lambda a, b: a | np.any(np.isnan(b)) | np.any(np.isinf(b)), pstate.velocity.params, initializer=False)
    if new_velo_state_has_nan_or_inf:
      raise RuntimeError(f'NaN/inf params detected in new velocity state at step {step}')

    step_time = time.perf_counter() - start_time
    step += 1

    if step == 1 or (step % config.opt.log_freq == 0):
      logging.info('[step %d] time: %.3f s, total loss: %.1f, data loss: %.1f, pinn loss: %.1e, velo loss: %.1f',
                   step, step_time, loss, data_loss, pinn_loss, velo_loss)
    
    if np.any(np.isnan(ploss)):
      raise RuntimeError(f'NaN loss detected at step {step}')

    if step % config.opt.val_freq == 0:
      # Evaluate MSE of image-plane measurements and 3D emissions, and
      # potentially evaluate movie.
      state = flax.jax_utils.unreplicate(pstate)

      eval_movie = step % (config.opt.movie_freq) == 0
      (est_image_plane, est_emission_per_t, next_emission_per_t,
       predicted_next_emission_per_t, velo_loss, true_velo_mse, ani) = eval_fn(
         state, movie=eval_movie)
      mse_image_plane = np.mean(np.square(est_image_plane - image_plane))
      mse_emission = np.mean(np.square(est_emission_per_t - true_emission_per_t))
      avg_pinn_loss = np.mean(np.sum(np.square(predicted_next_emission_per_t - next_emission_per_t), axis=(1, 2, 3)))
      val_dict['steps'].append(step)
      val_dict['mse_image_plane'].append(mse_image_plane)
      val_dict['mse_emission'].append(mse_emission)
      val_dict['pinn_loss'].append(avg_pinn_loss)
      val_dict['velo_loss'].append(velo_loss)
      val_dict['true_velo_mse'].append(true_velo_mse)
      if eval_movie:
        ani.save(os.path.join(movie_dir, f'movie_{step}.mp4'))
      
      # Evaluate chi-squared.
      mov = eh.movie.Movie(
        est_image_plane, eht.times, eht.psize, eht.ra, eht.dec, rf=eht.rf, polrep='stokes',
        pol_prim=None, pulse=eh.PULSE_DEFAULT, source=eht.source, mjd=eht.mjd)
      chisq = eht.obs.chisq(mov)
      val_dict['chisq'].append(chisq)
      logging.info('[step %d] chisq: %.2f', step, chisq)


    if step % config.opt.progress_freq == 0:
      start_time = time.perf_counter()
      state = flax.jax_utils.unreplicate(pstate)
      idx = 0
      # Sample estimated velocity.
      est_velocity, target_velocity = network.evaluate_velocity(
        state.velocity.params, state.velocity.model_state, velocity_predictor.apply,
        rmin, rmax, z_width=3., a=config.opt.spin, M=float(opt_geos.M),
        retrograde=config.sim.retrograde, fac_subkep=config.opt.fac_subkep,
        beta_phi=config.opt.beta, beta_r=config.opt.beta,
        normal_observer=config.opt.normal_observer)
      _, true_velocity = network.evaluate_velocity(
        state.velocity.params, state.velocity.model_state, velocity_predictor.apply,
        rmin, rmax, z_width=3., a=config.sim.spin, M=float(sim_geos.M),
        retrograde=config.sim.retrograde, fac_subkep=config.sim.fac_subkep,
        beta_phi=config.sim.beta, beta_r=config.sim.beta,
        normal_observer=config.opt.normal_observer)

      # Get true emission at sampled time.
      true_curr_emission = emission.propagate_emissions(
        emission_list,
        t_injection_list,
        sim_geos,
        t_frames=np.array([tstart.value, tstart.value + batch_pinn_t[idx]]) * units.hr,
        coords=np.moveaxis(vis.coords, 0, -1),
        fac_subkep=config.sim.fac_subkep,
        beta_phi=config.sim.beta,
        beta_r=config.sim.beta,
        dt0=0.001 * units.hr,
        M=consts.sgra_mass,
        retrograde=config.sim.retrograde)[1]

      # Get current emission, next emission, and velocity-predicted next emission.
      curr_emission = flax.jax_utils.unreplicate(pemissions)[idx]
      next_emission = flax.jax_utils.unreplicate(next_pemissions)[idx]
      predicted_next_emission = flax.jax_utils.unreplicate(predicted_next_pemissions)[idx]
      # NOTE: This code assumes that PINN loss is evaluated on `emission_coords`.
      curr_emission = emission.interpolate_coords(curr_emission, emission_coords, np.moveaxis(vis.coords, 0, -1))
      next_emission = emission.interpolate_coords(next_emission, emission_coords, np.moveaxis(vis.coords, 0, -1))
      predicted_next_emission = emission.interpolate_coords(predicted_next_emission, emission_coords, np.moveaxis(vis.coords, 0, -1))
      # Render emission views.
      e_list = np.array([true_curr_emission, curr_emission, next_emission, predicted_next_emission])
      emission_view_list = [vis.render(
          data / e_list.max(), facewidth=config.sim.fov_M, linewidth=0.1,
          bh_radius=2., jit=True).clip(max=1) for data in e_list]

      # Evaluate progress on:
      # - observed vs. recon. image plane at sampled time
      # - true vs. recon. 3D emission at sampled time
      # - emission-predicted vs. velocity-predicted 3D emission at next time
      # - estimated velocity field
      true_image = image_plane[batch_indices[idx]]
      est_image = flax.jax_utils.unreplicate(pimages)[idx]
      fig = run_utils.plot_progress(
        step,
        loss_dict,
        true_image,
        est_image,
        image_t=batch_data_t[idx] + tstart,
        true_curr_emission_view=emission_view_list[0],
        curr_emission_view=emission_view_list[1],
        next_emission_view=emission_view_list[2],
        predicted_next_emission_view=emission_view_list[3],
        pinn_t=batch_pinn_t[idx] + tstart,
        pinn_dt=config.opt.dt_hr,
        est_velocity=est_velocity,
        true_velocity=true_velocity,
        target_velocity=target_velocity,
        pinn_weight=state.pinn_weight,
        velo_weight=state.velo_weight,
        data_weight=state.data_weight,
        image_weight=config.opt.image_weight,
        vis_weight=config.opt.vis_weight,
        cp_weight=config.opt.cp_weight,
        logca_weight=config.opt.logca_weight,
        flux_weight=config.opt.flux_weight)
      fig.savefig(os.path.join(progress_dir, f'progress_{step:06d}.png'))

      progress_time = time.perf_counter() - start_time
      logging.info('[step %d] time to plot progress: %.3f s', step, progress_time)

    # Save checkpoint.
    if step % config.opt.ckpt_freq == 0:
      state = flax.jax_utils.unreplicate(pstate)
      save_args = orbax_utils.save_args_from_target(state)
      ckpt_mgr.save(step, state, save_kwargs={'save_args': save_args})
      # Save loss curves.
      with open(os.path.join(progress_dir, 'loss_dict.pkl'), 'wb') as fp:
        pickle.dump(loss_dict, fp)
      with open(os.path.join(progress_dir, 'val_dict.pkl'), 'wb') as fp:
        pickle.dump(val_dict, fp)

  ckpt_mgr.wait_until_finished()

  # Record training time.
  train_end_time = time.perf_counter()
  total_train_time_sec = train_end_time - train_start_time

  workdir = run_utils.get_workdir(_CONFIG.value, _WORKDIR.value)
  timing_path = os.path.join(workdir, 'training_time.txt')

  with open(timing_path, 'w') as f:
    f.write(f'Total training time (seconds): {total_train_time_sec:.2f}\n')
    f.write(f'Total training time (hours): {total_train_time_sec / 3600:.2f}\n')

  logging.info(
    'Total training time: %.2f seconds (%.2f hours)',
    total_train_time_sec,
    total_train_time_sec / 3600
  )


def main(_):
  if jax.process_count() != 1:
    raise RuntimeError('Multi-host training is not yet supported.')

  config = _CONFIG.value

  # The working directory has the format: {basedir}/{simdescr}/{measdescr}/{hparamdescr}.
  workdir = run_utils.get_workdir(config, _WORKDIR.value)
  simdir = run_utils.get_simdir(config, _WORKDIR.value)

  ckpt_dir = os.path.join(workdir, 'checkpoints')
  progress_dir = os.path.join(workdir, 'progress')
  movie_dir = os.path.join(workdir, 'movies')
  os.makedirs(ckpt_dir, exist_ok=True)
  os.makedirs(progress_dir, exist_ok=True)
  os.makedirs(movie_dir, exist_ok=True)

  # Save config.
  with open(os.path.join(workdir, 'config.txt'), 'w') as f:
    f.write(str(config))

  # Get EHT object.
  fov = consts.rad_per_M(consts.sgra_mass, consts.sgra_distance) * config.sim.fov_M
  eht = EHT(
    config.sim.array,
    imsize=config.sim.num_alpha,
    fov=fov.value,
    tstart=config.sim.tstart,
    tstop=config.sim.tstop,
    tint=config.sim.tadv,
    tadv=config.sim.tadv)

  # Get geodesics with simulation parameters.
  # These will be used to generate the ground-truth emissivity field.
  sim_geos = run_utils.get_geodesics(config, spin=config.sim.spin)

  # Get simulated emissivity field data.
  t_frames = eht.times * units.hr
  random_state = np.random.RandomState(config.seed)
  emission_list, t_injection_list, image_plane = run_utils.get_simulation_data(
    config, t_frames, sim_geos, random_state)

  # Rescale flux.
  for i in range(len(emission_list)):
    emission_list[i].data = emission_list[i].data * config.sim.flux_multiplier
  image_plane = image_plane * config.sim.flux_multiplier

  emission_coords = run_utils.get_emission_coords(emission_list[0])
  # Simulate true dynamic 3D emissivity.
  true_emission_per_t = emission.propagate_emissions(
    emission_list,
    t_injection_list,
    sim_geos,
    t_frames,
    coords=emission_coords,
    fac_subkep=config.sim.fac_subkep,
    beta_phi=config.sim.beta,
    beta_r=config.sim.beta,
    dt0=0.001 * units.hr,
    M=consts.sgra_mass,
    retrograde=config.sim.retrograde)
  logging.info('Simulated true dynamic 3D emissivity.')

  # Save data products.
  np.save(os.path.join(simdir, 'image_plane.npy'), image_plane)
  np.save(os.path.join(simdir, 'true_emission_per_t.npy'), true_emission_per_t)
  np.save(os.path.join(simdir, 't_injection_list.npy'), t_injection_list.value)
  np.save(os.path.join(simdir, 'emission_list.npy'), emission_list)
  np.save(os.path.join(simdir, 't_frames.npy'), t_frames.value)

  # Create networks.
  rmin = float(sim_geos.r.min())  # minimum recovery radius
  rmax = 0.6 * config.sim.fov_M  # maximum recovery radius
  emission_predictor, velocity_predictor = run_utils.get_predictors(config, rmin, rmax)

  train(emission_predictor, velocity_predictor,
        emission_list, t_injection_list, image_plane, t_frames,
        true_emission_per_t, eht)


if __name__ == '__main__':
  app.run(main)