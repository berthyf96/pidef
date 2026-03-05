"""A library of forward models with associated log-likelihood functions."""
import ehtim as eh
from ehtim.obsdata import Obsdata
from ehtim.observing.pulses import trianglePulse2D
import ehtim.imaging.starwarps as sw
import jax.numpy as jnp
import numpy as np

# fix one bug in ehtim
class ObsdataWrapper(Obsdata):
  def tlist(self, conj=False, t_gather=0., scan_gather=False):
    datalist = super().tlist(conj=conj, t_gather=t_gather, scan_gather=scan_gather)
    try:
      return np.array(datalist, dtype=self.data.dtype)
    except:
      return np.array(datalist, dtype=object)


def pad_data(data_list, fill_value=np.nan):
  """Pad a list of 1D arrays to form a 2D array of shape [T, f]."""
  max_len = max(len(data) for data in data_list)
  padded_array = np.full((len(data_list), max_len), fill_value, dtype=data_list[0].dtype)
  for i, data in enumerate(data_list):
    padded_array[i, :len(data)] = data
  return padded_array


class EHT:
  """EHT measurements with visibility amplitudes and closure phases. Assumes grayscale, square images."""
  def __init__(self, array, imsize, fov, tstart=12.5, tstop=14.3, tint=102, tadv=102):
    """Initialize `EHT` module.

    Args:
      array: path to the EHT array file.
      imsize: image size (pixels).
      fov: field of view (radians).
      tstart: observation start time (UTC hours).
      tstop: observation stop time (UTC hours).
      tint: integration time (seconds).
      tadv: advance time (seconds).
    """
    # ref_obs, A_vis, times, cp_index, cp_conjugate, camp_index, camp_conjugate = self.process_obs(
    #   array, imsize, tstart, tstop, tint, tadv)
    ref_obs_list, ref_obs, times, A_vis_expanded, A_cp, A_logca = self.process_obs(
      array, imsize, fov, tstart, tstop, tint, tadv)
    self.ref_obs_list = ref_obs_list
    self.ref_obs = ref_obs
    self.imsize = imsize
    self.fov = fov
    self.A_vis_expanded = A_vis_expanded  # [T, 2f, HxW]
    self.times = times  # [T]
    self.T = len(times)
    self.A_cp = A_cp
    self.A_logca = A_logca

  def process_obs(self, array, imsize, fov, tstart, tstop, tint, tadv):
    self.uvarray     = uvarray     = eh.array.load_txt(array)
    self.psize       = fov / imsize
    self.pulse       = pulse       = trianglePulse2D
    self.ra          = ra          =  17.761120
    self.dec         = dec         = -29.007797
    self.rf          = rf          = 230 * 1e9
    self.bw          = bw          = 2 * 1e9 # for EHT, 16 * 1e9 for the ngEHT
    self.mjd         = mjd         = 60775   # (10/04/2025)
    self.source      = source      = 'SgrA'
    self.tint        = tint
    self.tadv        = tadv
    self.tau         = tau         = 0.1
    self.taup        = 0.1
    self.polrep_obs  = polrep_obs  = 'stokes'
    self.elevmin     = elevmin     = 10.0 
    self.elevmax     = elevmax     = 85.0
    self.timetype    = timetype    = 'UTC'

    # SgrA best times (in UTC): 12.5 - 14.2
    self.tstart = tstart
    self.tstop = tstop

    ref_obs = uvarray.obsdata(
      ra, dec, rf, bw, tint, tadv, tstart, tstop,
      mjd=mjd, polrep=polrep_obs, tau=tau, timetype=timetype,
      elevmin=elevmin, elevmax=elevmax,
      no_elevcut_space=False,
      fix_theta_GMST=False)
    ref_obs.__class__ = ObsdataWrapper

    # Dummy image with image metadata
    ref_im = eh.image.make_empty(
      imsize, fov, ra, dec, rf, source, pulse=pulse, mjd=mjd)

    # Get time for each frame
    time_list = []
    for t_obs in ref_obs.tlist():
      time_list.append(t_obs[0]['time'])

    # Get forward models for each frame
    A_vis_expanded_list = []
    A_cp_list = []
    A_logca_list = []
    obs_list = sw.splitObs(ref_obs)
    for t_obs in obs_list:
      # uv = np.hstack((t_obs['u'].reshape(-1, 1), t_obs['v'].reshape(-1, 1)))
      # A_vis = obsh.ftmatrix(psize, resolution, resolution, uv, pulse=pulse, mask=[])

      # Visibilities
      _, _, A_vis = eh.imaging.imager_utils.chisqdata_vis(
        t_obs, ref_im, mask=[])
      A_vis_expanded = jnp.concatenate((A_vis.real, A_vis.imag), axis=0)  # [2f, HxW]
      A_vis_expanded_list.append(A_vis_expanded)

      # Closure phases
      _, _, A_cp = eh.imaging.imager_utils.chisqdata_cphase(t_obs, ref_im, mask=[])
      A_cp_list.append(A_cp)

      # Log closure amplitudes
      _, _, A_logca = eh.imaging.imager_utils.chisqdata_logcamp(t_obs, ref_im, mask=[])
      A_logca_list.append(A_logca)

    A_vis_expanded = np.stack(A_vis_expanded_list, axis=0) # [T, 2f, HxW]
    A_cp = np.stack(A_cp_list, axis=0) # [T, 3, fCP, D]
    A_logca = np.stack(A_logca_list, axis=0) # [T, 4, fCA, D]
    times = np.array(time_list)  # [T]

    return obs_list, ref_obs, times, A_vis_expanded, A_cp, A_logca

  def measure(self, frames, ampcal, phasecal, dcal):
    if not ampcal:
      # These are measured from 2017 campaign. For future arrays, one thing 
      # you could do is add a realistic value for the gains of the missing antennas
      # (for example the average value from the ones you have),
      # or just pass a single value for all antennas, instead of passing
      # a dictionary.
      # GAIN_OFFSET = {'AA': 0.029,'AP': 0.028,'AZ': 0.045,'JC': 0.020,'LM': 0.147,
      # 			   'PV': 0.050,'SM': 0.019,'SP': 0.052,'SR': 0.0}
      # GAINP =       {'AA': 0.054,'AP': 0.045,'AZ': 0.056,'JC': 0.030,'LM': 0.124,
      # 			   'PV': 0.075,'SM': 0.028,'SP': 0.095,'SR': 0.0}
      GAIN_OFFSET = {
          'ALMA': 0.029,
          'APEX': 0.028,
          'SMT': 0.045,
          'JCMT': 0.020,
          'LMT': 0.147,
          'PV': 0.050,
          'SMA': 0.019,
          'SPT': 0.052,
          'SR': 0.0}  # ?
      GAINP = {
          'ALMA': 0.054,
          'APEX': 0.045,
          'SMT': 0.056,
          'JCMT': 0.030,
          'LMT': 0.124,
          'PV': 0.075,
          'SMA': 0.028,
          'SPT': 0.095,
          'SR': 0.0}
        
    if not dcal:
      # DOFF = {'AA':0.005, 'AP':0.005, 'AZ':0.01, 'LM':0.01, 'PV':0.01, 'SM':0.005,
      # 		'JC':0.01, 'SP':0.01, 'SR':0.01}
      DOFF = {
          'ALMA': 0.005,
          'APEX': 0.005,
          'SMT': 0.01,
          'LMT': 0.01,
          'PV': 0.01,
          'SMA': 0.005,
          'JCMT': 0.01,
          'SPT': 0.01,
          'SR': 0.01}

    ref_obs = self.ref_obs
    times = self.times
    mov = eh.movie.Movie(
      frames, times, self.psize, self.ra, self.dec, rf=self.rf, polrep='stokes',
      pol_prim=None, pulse=eh.PULSE_DEFAULT, source=self.source, mjd=self.mjd)

    # observe the image
    obs = mov.observe_same(
      ref_obs, ttype='direct', add_th_noise=True, taup=self.taup, jones=True,
      inv_jones=False, ampcal=ampcal, phasecal=phasecal, dcal=dcal,
      stabilize_scan_phase=True, stabilize_scan_amp=True,
      gain_offset=0.0, gainp=0.0, dterm_offset=0.0,
      rlratio_std=0.0,rlphase_std=0.0, seed=3, sigmat=0.25,
      verbose=False)

    # flux
    _, flux = self.estimate_flux(obs)
  
    # split obs into T observations
    obs_list = sw.splitObs(obs)

    vis_expanded_list, sigmavis_expanded_list = [], []
    amp_list, sigmaamp_list = [], []
    cp_list, sigmacp_list = [], []
    logca_list, sigmalogca_list = [], []
    for obs_t in obs_list:
      # visibilities
      # vis_list.append(obs_t.data['vis'])
      # sigmavis_list.append(obs_t.data['sigma'])
      vis = obs_t.data['vis']
      vis_expanded = np.concatenate((vis.real, vis.imag), axis=0)
      sigmavis = obs_t.data['sigma']
      sigmavis_expanded = np.concatenate((sigmavis, sigmavis), axis=0)
      vis_expanded_list.append(vis_expanded)
      sigmavis_expanded_list.append(sigmavis_expanded)

      # visibilities amplitude
      obs_t.add_amp(debias=True)
      amp_list.append(obs_t.amp['amp'])
      sigmaamp_list.append(obs_t.amp['sigma'])

      # closure phase
      obs_t.add_cphase(count='min')
      cp_list.append(obs_t.cphase['cphase'] * eh.DEGREE)
      sigmacp_list.append(obs_t.cphase['sigmacp'] * eh.DEGREE)
  
      # log closure amplitude
      obs_t.add_logcamp(debias=True, count='min')
      logca_list.append(obs_t.logcamp['camp'])
      sigmalogca_list.append(obs_t.logcamp['sigmaca'])

    vis_expanded = pad_data(vis_expanded_list)  # [T, f]
    sigmavis_expanded = pad_data(sigmavis_expanded_list)
    amp = pad_data(amp_list)
    sigmaamp = pad_data(sigmaamp_list)
    cp = pad_data(cp_list)
    sigmacp = pad_data(sigmacp_list)
    logca = pad_data(logca_list)
    sigmalogca = pad_data(sigmalogca_list)

    self.obs = obs
    self.obs_list = obs_list
    return vis_expanded, sigmavis_expanded, amp, sigmaamp, cp, sigmacp, logca, sigmalogca, flux

  def estimate_flux(self, obs):
    # estimate the total flux from the observation
    data = obs.unpack_bl('ALMA', 'APEX', 'amp')
    flux_per_frames = []
    for pair in data:
      amp = pair[0][1]
      flux_per_frames.append(amp)
    flux_per_frames = np.array(flux_per_frames) # [T,]
    flux = np.median(flux_per_frames)
    return flux, flux_per_frames

  def forward_vis(self, x, A_vis):
    return A_vis @ x.flatten()  # [f]
  
  def forward_cphase(self, x, A_cp):
    xvec = x.flatten()
    i1 = A_cp[0] @ xvec
    i2 = A_cp[1] @ xvec
    i3 = A_cp[2] @ xvec
    cphase = jnp.angle(i1 * i2 * i3)
    return cphase

  def forward_logcamp(self, x, A_logca):
    xvec = x.flatten()
    a1 = jnp.abs(A_logca[0] @ xvec)
    a2 = jnp.abs(A_logca[1] @ xvec)
    a3 = jnp.abs(A_logca[2] @ xvec)
    a4 = jnp.abs(A_logca[3] @ xvec)
    logcamp = jnp.log(a1) + jnp.log(a2) - jnp.log(a3) - jnp.log(a4)
    return logcamp

  def chi2_vis(self, x, y_vis_expanded, y_sigmavis_expanded, A_vis_expanded):
    """Chi^2 for complex visibilities (single frame)."""
    vis_expanded = self.forward_vis(x, A_vis_expanded)
    residual = vis_expanded - y_vis_expanded
    return jnp.mean(jnp.square(residual / y_sigmavis_expanded)) / 2

  def chi2_amp(self, x, y_amp, y_sigmaamp, A_vis_expanded):
    """Chi^2 for visibility amplitudes (single frame)."""
    vis_expanded = self.forward_vis(x, A_vis_expanded)
    n = vis_expanded.shape[0] // 2
    vis_real = vis_expanded[:n]
    vis_imag = vis_expanded[n:]
    amp = jnp.sqrt(vis_real**2 + vis_imag**2)
    residual = amp - y_amp
    return jnp.mean(jnp.square(residual / y_sigmaamp))
  
  def chi2_cphase(self, x, y_cphase, y_sigmacp, A_cp):
    """Chi^2 for closure phases (single frame)."""
    cphase = self.forward_cphase(x, A_cp)
    angle_residual = cphase - y_cphase
    return 2. * jnp.mean((1 - jnp.cos(angle_residual)) / jnp.square(y_sigmacp))
  
  def chi2_logcamp(self, x, y_logcamp, y_sigmalogcamp, A_logca):
    """Chi^2 for log closure amplitudes (single frame)."""
    logcamp = self.forward_logcamp(x, A_logca)
    residual = logcamp - y_logcamp
    return jnp.mean(jnp.square((residual) / y_sigmalogcamp))

  def chi2_flux(self, x, y_flux):
    flux = jnp.sum(x)
    return jnp.mean(jnp.square((flux - y_flux))) / 2