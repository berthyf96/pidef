import ml_collections
import numpy as np


def get_config():
  config = ml_collections.ConfigDict()

  config.seed = 8

  config.sim = sim = ml_collections.ConfigDict()
  sim.M = 1.
  sim.spin = 0.2
  sim.fov_M = 25.
  sim.inclination_deg = 60.
  sim.slow_light = False
  sim.retrograde = False
  sim.doppler = True
  sim.beta = 0.9  # 1 => full sub-Kepleriian, 0 => full infall
  sim.fac_subkep = 0.7
  sim.num_emissions = 20
  sim.min_orbit_radius = 7.  # min. orbital radius of emission hot spot
  sim.max_orbit_radius = 8.  # max. orbital radius of emission hot spot
  sim.min_std = 0.5  # min. stddev of emission hot spot
  sim.max_std = 1.  # max. stddev of emission hot spot
  sim.type = 'simple'  # 'simple' | 'grmhd'
  sim.num_alpha = 100  # discretization of alpha in image plane
  sim.num_beta = 100  # discretization of beta in image plane
  sim.tadv = 102  # tadv (seconds) parameters for EHT observations
  sim.tstart = 12.5  # UTC hours
  sim.tstop = 13.5  # UTC hours (best Sgr A* observing time: 12.5-14.3)
  sim.array = 'eht_arrays/EHT2017.txt'
  sim.station_noise = False
  sim.flux_multiplier = 0.02  # corresponds to a mean total flux of ~2.3 Jy, which is that of Sgr A*

  config.opt = opt = ml_collections.ConfigDict()
  opt.beta = 1.
  opt.fac_subkep = 1.
  opt.spin = 0.2
  opt.dt_hr = 0.01
  opt.data_weight = 1.
  opt.image_weight = 0.
  opt.vis_weight = 1.
  opt.amp_weight = 0.
  opt.cp_weight = 0.
  opt.logca_weight = 0.
  opt.flux_weight = 0.
  opt.l1_pinn_loss = True
  opt.blur_for_pinn_loss = True
  opt.pinn_weight = 1.
  opt.anneal_pinn_weight = False
  opt.velo_weight = 1.
  opt.anneal_velo_weight = False
  opt.warmup_velo_steps = 0
  opt.init_velo_weight = 1.
  opt.velo_weight_decay_rate = 0.001
  opt.velo_loss_grid = 'geos'
  opt.normal_observer = True
  opt.learned_doppler = True
  opt.batch_size = 6
  opt.val_freq = 5000
  opt.ckpt_freq = 10000
  opt.progress_freq = 1000
  opt.log_freq = 10
  opt.movie_freq = 10000
  opt.niter = 100000
  opt.nckpt = 20
  opt.emission_lr_init = 1e-4
  opt.emission_lr_final = 1e-6
  opt.velocity_lr_init = 1e-4
  opt.velocity_lr_final = 1e-6
  opt.lr_decay_schedule = 'linear'  # 'linear' | 'cosine'
  opt.lr_decay_begin = 0  # for 'linear' schedule
  opt.lr_decay_steps = 100000  # for 'cosine' and 'linear' schedule
  opt.grad_clip = ml_collections.config_dict.FieldReference(None, field_type=float)
  opt.use_target_velocity = False  # if True, use target velocity for dynamics loss

  config.net = net = ml_collections.ConfigDict()
  net.emission_width = 256
  net.velocity_width = 128
  net.emission_z_width = ml_collections.config_dict.FieldReference(np.inf, field_type=float)
  net.velocity_z_width = ml_collections.config_dict.FieldReference(np.inf, field_type=float)
  net.velocity_residual = False
  net.velocity_fill_unsupervised = False
  net.velocity_out_r_scale = 1.
  net.velocity_out_phi_scale = 1.
  net.emission_posenc_deg = 3
  net.velocity_posenc_deg = 1
  net.time_posenc_deg = 3
  net.batch_norm = False
  net.coordinate_type = 'r'  # 'cartesian' | 'spherical' | 'r' | 'r+theta'
  config.ode = ode = ml_collections.ConfigDict()
  ode.solver = 'Tsit5'
  ode.stepsize_controller = 'ConstantStepSize'
  ode.dt0_hr = 0.001  # if <= 0, use automatic dt0
  ode.rtol = 1e-3  # rtol for diffrax.PIDController
  ode.atol = 1e-6  # atol for diffrax.PIDController
  # NOTE: BacksolveAdjoint currently leads to CustomVJPException, but anyway
  # RecursiveCheckpointAdjoint is preferred.
  ode.adjoint_method = 'RecursiveCheckpointAdjoint'  
  ode.adjoint_solver = 'Tsit5'
  ode.adjoint_stepsize_controller = 'PIDController'
  ode.adjoint_rms_seminorm = False  # can increase efficiency of BacksolveAdjoint
  ode.adjoint_rtol = 1e-3
  ode.adjoint_atol = 1e-6

  return config