#!/bin/sh

python run.py \
  --config config.py \
  --workdir runs \
  --config.sim.array 'eht_arrays/ngEHT.txt' \
  --config.sim.flux_multiplier 0.02 \
  --config.opt.fac_subkep 1. \
  --config.opt.beta 1. \
  --config.net.emission_z_width 3 \
  --config.opt.pinn_weight 10 \
  --config.opt.init_velo_weight 100 \
  --config.opt.anneal_velo_weight \
  --config.opt.velo_weight 1e-4 \
  --config.opt.emission_lr_init 1e-4 \
  --config.opt.emission_lr_final 1e-6 \
  --config.opt.velocity_lr_init 1e-4 \
  --config.opt.velocity_lr_final 1e-6 \
  --config.opt.lr_decay_steps 100000 \
  --config.opt.niter 100000