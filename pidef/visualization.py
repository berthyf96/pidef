# Adapted from bhnerf (https://github.com/aviadlevis/bhnerf)
# Original authors: Aviad Levis et al.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import ImageGrid
from pidef.utils import normalize
import jax
from jax import numpy as jnp


class VolumeVisualizer(object):
    def __init__(self, width, height, samples):
        """
        A Volume visualization class
        
        Parameters
        ----------
        width: int
            camera horizontal resolution.
        height: int
            camera vertical resolution.
        samples: int
            Number of integration points along a ray.
        """
        self.width = width
        self.height = height
        self.samples = samples 
        self._pts = None
        
    def set_view(self, cam_r, domain_r, azimuth, zenith, up=np.array([0., 0., 1.])):
        """
        Set camera view geometry
        
        Parameters
        ----------
        cam_r: float,
            Distance from the origin
        domain_r: float, 
            Maximum radius of the spherical domain
        azimuth: float, 
            Azimuth angle in radians
        zenith: float, 
            Zenith angle in radians
        up: array, default=[0,0,1]
            The up direction determines roll of the camera
        """
        camorigin = cam_r * np.array([np.cos(azimuth)*np.sin(zenith), 
                                      np.sin(azimuth)*np.sin(zenith), 
                                      np.cos(zenith)])
        self._viewmatrix = self.viewmatrix(camorigin, up, camorigin)
        fov = 1.06 * np.arctan(np.sqrt(3) * domain_r / cam_r)
        focal = .5 * self.width / jnp.tan(fov)
        rays_o, rays_d = self.generate_rays(
            self._viewmatrix, self.width, self.height, focal)
        
        near = cam_r - np.sqrt(3) * domain_r
        far  = cam_r + np.sqrt(3) * domain_r
    
        self._pts = self.sample_along_rays(rays_o, rays_d, near, far, self.samples)
        self.x, self.y, self.z = self._pts[...,0], self._pts[...,1], self._pts[...,2]
        self.d = jnp.linalg.norm(jnp.concatenate([jnp.diff(self._pts, axis=2), 
                                                  jnp.zeros_like(self._pts[...,-1:,:])], 
                                                  axis=2), axis=-1)

    def render(self, emission, facewidth, jit=False, bh_radius=0.0,
               linewidth=0.1, bh_albedo=[0, 0, 0], cmap='hot',
               wireframe=True, darkmode=False):
        """
        Render an image of the 3D emission
        
        Parameters
        ----------
        emission: 3D array 
            3D array with emission values
        jit: bool, default=False,
            Just in time compilation. Set true for rendering multiple frames.
            First rendering will take more time due to compilation.
        bh_radius: float, default=0.0
            Radius at which to draw a black hole (for visualization). 
            If bh_radius=0 then no black hole is drawn.
        facewidth: float or (float, float, float)
            If float: width of the enclosing cube face (same in x,y,z).
            If tuple/list of length 3: (x_width, y_width, z_width) of a rectangular box.
        linewidth: float, default=0.1
            width of the cube/box lines
        ...
        """
        if self._pts is None: 
            raise AttributeError('must set view before rendering')

        # --- NEW: normalize facewidth to a 3-vector (wx, wy, wz) ---
        if np.isscalar(facewidth):
            box_widths = jnp.array([facewidth, facewidth, facewidth], dtype=jnp.float32)
        else:
            fw = np.asarray(facewidth, dtype=np.float32)
            if fw.shape != (3,):
                raise ValueError("facewidth must be a scalar or an iterable of length 3 (wx, wy, wz).")
            box_widths = jnp.array(fw)

        # --- existing colormap logic (slightly cleaned) ---
        cm = plt.get_cmap(cmap) 
        # Normalize emission
        normalized_emission = emission / (jnp.amax(emission) + 1e-10)

        # Apply colormap and set alpha to normalized emission (or zero if emission is zero)
        emission_cm_rgb = cm(normalized_emission)[..., :3]
        alpha_channel = jnp.where(emission > 0, normalized_emission, 0.0)

        # Combine into RGBA
        emission_cm = jnp.concatenate([emission_cm_rgb, alpha_channel[..., None]], axis=-1)

        wireframe_color = jnp.array([1.0, 1.0, 1.0, 1e6]) if darkmode else jnp.array([0.0, 0.0, 0.0, 1e6])
        if wireframe:
            if jit:
                emission_cube = draw_cube_jit(emission_cm, self._pts, box_widths, linewidth, wireframe_color)
            else:
                emission_cube = draw_cube(emission_cm, self._pts, box_widths, linewidth, wireframe_color)
        else:
            emission_cube = emission_cm

        if bh_radius > 0:
            if jit:
                emission_cube = draw_bh_jit(emission_cube, self._pts, bh_radius, bh_albedo)
            else:
                emission_cube = draw_bh(emission_cube, self._pts, bh_radius, bh_albedo)

        # Half-widths for “inside box” mask
        inside_halfwidth = np.array(box_widths) / 2.0 - linewidth
        rendering = alpha_composite(
            emission_cube, self.d, self._pts, bh_radius, inside_halfwidth=inside_halfwidth
        )
        return rendering
    
    def viewmatrix(self, lookdir, up, position):
        """Construct lookat view matrix."""
        vec2 = normalize(lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def generate_rays(self, camtoworlds, width, height, focal):
        """Generating rays for all images."""
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(width, dtype=np.float32),  # X-Axis (columns)
            np.arange(height, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_dirs = np.stack(
            [(x - width * 0.5 + 0.5) / focal,
             -(y - height * 0.5 + 0.5) / focal, -np.ones_like(x)],
            axis=-1)
        directions = ((camera_dirs[..., None, :] *
                       camtoworlds[None, None, :3, :3]).sum(axis=-1))
        origins = np.broadcast_to(camtoworlds[None, None, :3, -1],
                                  directions.shape)

        return origins, directions

    def sample_along_rays(self, rays_o, rays_d, near, far, num_samples):
        t_vals = jnp.linspace(near, far, num_samples)
        pts = rays_o[..., None, :] + t_vals[None, None, :, None] * rays_d[..., None, :]
        return pts
    
    @property
    def coords(self):
        coords = None if self._pts is None else jnp.moveaxis(self._pts, -1, 0)
        return coords


def alpha_composite(emission, dists, pts, bh_rad, inside_halfwidth=7.5):
    emission = np.clip(emission, 0., 1.)
    color = emission[..., :-1] * dists[0, ..., None]
    alpha = emission[..., -1:] 
    
    # --- NEW: mask for points inside wireframe (cube OR rectangular box) ---
    if np.isscalar(inside_halfwidth):
        # Old behavior: isotropic cube
        inside = np.where(
            np.less(np.amax(np.abs(pts), axis=-1), inside_halfwidth),
            np.ones_like(pts[..., 0]),
            np.zeros_like(pts[..., 0])
        )
    else:
        # New behavior: axis-aligned rectangular prism
        ihw = np.asarray(inside_halfwidth)
        if ihw.shape != (3,):
            raise ValueError("inside_halfwidth must be scalar or length-3 iterable.")
        hx, hy, hz = ihw
        inside = (
            (np.abs(pts[..., 0]) < hx) &
            (np.abs(pts[..., 1]) < hy) &
            (np.abs(pts[..., 2]) < hz)
        ).astype(pts.dtype)

    # masks for points outside black hole
    bh = np.where(np.greater(np.linalg.norm(pts, axis=-1), bh_rad),
                  np.ones_like(pts[..., 0]),
                  np.zeros_like(pts[..., 0]))

    combined_mask = np.logical_and(inside, bh)

    rendering = np.zeros_like(color[:, :, 0, :])
    acc = np.zeros_like(color[:, :, 0, 0])
    outside_acc = np.zeros_like(color[:, :, 0, 0])
    for i in range(alpha.shape[-2]):
        ind = alpha.shape[-2] - i - 1

        # if pixels inside box and outside black hole, don't alpha composite
        rendering = rendering + combined_mask[..., ind, None] * color[..., ind, :]

        # else, alpha composite      
        outside_alpha = alpha[..., ind, :] * (1. - combined_mask[..., ind, None])
        rendering = rendering * (1. - outside_alpha) + color[..., ind, :] * outside_alpha 

        acc = alpha[..., ind, 0] + (1. - alpha[..., ind, 0]) * acc
        outside_acc = outside_alpha[..., 0] + (1. - outside_alpha[..., 0]) * outside_acc

    rendering += np.array([1., 1., 1.])[None, None, :] * (1. - acc[..., None])
    # Include final alpha for rendering transpareny.
    final_alpha = jnp.clip(acc, 0.0, 1.0)
    rgba = jnp.concatenate([rendering, final_alpha[..., None]], axis=-1)
    return rgba


@jax.jit
def draw_cube_jit(emission_cm, pts, facewidth, linewidth, linecolor):
    # facewidth: scalar -> cube, (wx, wy, wz) -> rectangular box
    fw = jnp.array(facewidth)

    if fw.shape == ():  # scalar
        box_widths = jnp.array([fw, fw, fw])
    else:
        # assume shape (3,)
        box_widths = fw

    halfwidths = box_widths / 2.0
    hx, hy, hz = halfwidths[0], halfwidths[1], halfwidths[2]

    vertices = jnp.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [-hx,  hy, -hz],
        [ hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [-hx,  hy,  hz],
        [ hx,  hy,  hz],
    ])

    dirs = jnp.array([
        [-1., 0., 0.],
        [ 1., 0., 0.],
        [ 0., -1., 0.],
        [ 0.,  1., 0.],
        [ 0.,  0., -1.],
        [ 0.,  0.,  1.]
    ])

    # Length of edge along each direction (x, x, y, y, z, z)
    lengths_for_dir = jnp.array([
        box_widths[0], box_widths[0],
        box_widths[1], box_widths[1],
        box_widths[2], box_widths[2],
    ])

    for i in range(vertices.shape[0]):

        for j in range(dirs.shape[0]):
            L = lengths_for_dir[j]
            # Draw line segments from each vertex
            line_seg_pts = vertices[i, None, :] + jnp.linspace(0.0, L, 64)[:, None] * dirs[j, None, :]

            for k in range(line_seg_pts.shape[0]):
                dists = jnp.linalg.norm(
                    pts - jnp.broadcast_to(line_seg_pts[k, None, None, None, :], pts.shape),
                    axis=-1
                )
                update = linecolor[None, None, None, :] * jnp.exp(-1. * dists / linewidth ** 2)[..., None]
                emission_cm += update

    # Zero out everything outside the rectangular box (with small linewidth margin)
    mask = (
        (jnp.abs(pts[..., 0]) <= halfwidths[0] + linewidth) &
        (jnp.abs(pts[..., 1]) <= halfwidths[1] + linewidth) &
        (jnp.abs(pts[..., 2]) <= halfwidths[2] + linewidth)
    )
    mask = jnp.broadcast_to(mask[..., None], emission_cm.shape)

    out = jnp.where(mask, emission_cm, jnp.zeros_like(emission_cm))
    return out


def draw_cube(emission_cm, pts, facewidth, linewidth, linecolor):
    # facewidth: scalar -> cube, (wx, wy, wz) -> rectangular box
    fw = jnp.array(facewidth)

    if fw.shape == ():  # scalar
        box_widths = jnp.array([fw, fw, fw])
    else:
        box_widths = fw

    halfwidths = box_widths / 2.0
    hx, hy, hz = halfwidths[0], halfwidths[1], halfwidths[2]

    vertices = jnp.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [-hx,  hy, -hz],
        [ hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [-hx,  hy,  hz],
        [ hx,  hy,  hz],
    ])

    dirs = jnp.array([
        [-1., 0., 0.],
        [ 1., 0., 0.],
        [ 0., -1., 0.],
        [ 0.,  1., 0.],
        [ 0.,  0., -1.],
        [ 0.,  0.,  1.]
    ])

    lengths_for_dir = jnp.array([
        box_widths[0], box_widths[0],
        box_widths[1], box_widths[1],
        box_widths[2], box_widths[2],
    ])

    for i in range(vertices.shape[0]):

        for j in range(dirs.shape[0]):
            L = lengths_for_dir[j]
            line_seg_pts = vertices[i, None, :] + jnp.linspace(0.0, L, 64)[:, None] * dirs[j, None, :]

            for k in range(line_seg_pts.shape[0]):
                dists = jnp.linalg.norm(
                    pts - jnp.broadcast_to(line_seg_pts[k, None, None, None, :], pts.shape),
                    axis=-1
                )
                update = linecolor[None, None, None, :] * jnp.exp(-1. * dists / linewidth ** 2)[..., None]
                emission_cm += update

    mask = (
        (jnp.abs(pts[..., 0]) <= halfwidths[0] + linewidth) &
        (jnp.abs(pts[..., 1]) <= halfwidths[1] + linewidth) &
        (jnp.abs(pts[..., 2]) <= halfwidths[2] + linewidth)
    )
    mask = jnp.broadcast_to(mask[..., None], emission_cm.shape)

    out = jnp.where(mask, emission_cm, jnp.zeros_like(emission_cm))
    return out


@jax.jit
def draw_bh_jit(emission, pts, bh_radius, bh_albedo):
    bh_albedo = jnp.array(bh_albedo)[None, None, None, :]
    lightdir = jnp.array([-1., -1., 1.])
    lightdir /= jnp.linalg.norm(lightdir, axis=-1, keepdims=True)
    bh_color = jnp.sum(lightdir * pts, axis=-1)[..., None] * bh_albedo
    emission = jnp.where(jnp.less(jnp.linalg.norm(pts, axis=-1, keepdims=True), bh_radius),
                    jnp.concatenate([bh_color, jnp.ones_like(emission[..., 3:])], axis=-1), emission)
    return emission


def draw_bh(emission, pts, bh_radius, bh_albedo):
    bh_albedo = jnp.array(bh_albedo)[None, None, None, :]
    lightdir = jnp.array([-1., -1., 1.])
    lightdir /= jnp.linalg.norm(lightdir, axis=-1, keepdims=True)
    bh_color = jnp.sum(lightdir * pts, axis=-1)[..., None] * bh_albedo
    emission = jnp.where(jnp.less(jnp.linalg.norm(pts, axis=-1, keepdims=True), bh_radius),
                    jnp.concatenate([bh_color, jnp.ones_like(emission[..., 3:])], axis=-1), emission)
    return emission


def image_movie(image_plane, t_frames=None, fps=6, cmap='hot', cbar=False, figsize=(6, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.axis('off')
    clim = (np.min(image_plane), np.max(image_plane))
    im = ax.imshow(image_plane[0], clim=clim, cmap=cmap)
    if t_frames is not None:
        ax.set_title(f'$t=${t_frames[0]:.2f}', fontsize=21)

    if cbar:
        norm = Normalize(vmin=clim[0], vmax=clim[1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    def update_img(n):
        im.set_data(image_plane[n])
        if t_frames is not None:
            ax.set_title(f'$t=${t_frames[n]:.2f}', fontsize=21)
        return

    plt.close(fig)
    return animation.FuncAnimation(fig, update_img, len(image_plane), interval=1e3 / fps)


def plot_image_grid(y, nrow=None, padding=0., title='', cmap='viridis',
                    clim=(None, None), cbar=False, figsize=(10, 10), fig=None, rect=111):
    """Plot an image grid with matplotlib, optionally into an existing figure.

    Args:
        y: Array of shape (N, H, W) or (N, H, W, C)
        nrow: Number of rows in the grid (optional)
        padding: Padding between images
        title: Figure title
        cmap: Color map
        clim: Color limits (vmin, vmax)
        figsize: Figure size if fig is None
        fig: Existing matplotlib Figure object (optional)
        rect: Subplot spec or rect in figure for ImageGrid (e.g., 111 or [0.1, 0.1, 0.8, 0.8])
    """
    images = np.clip(y, 0., 1.) if y.shape[-1] == 3 else y

    if fig is None:
        fig = plt.figure(figsize=figsize)
        show_plot = True
    else:
        show_plot = False

    if nrow is None:
        nrow = int(np.floor(np.sqrt(images.shape[0])))
    ncol = int(np.ceil(len(images) / nrow))

    grid = ImageGrid(fig, rect, nrows_ncols=(nrow, ncol), axes_pad=padding, cbar_mode='single', cbar_location='right',)

    for ax in grid:
        ax.set_axis_off()
    
    for ax, im in zip(grid, images):
        mappable = ax.imshow(im, cmap=cmap, norm=Normalize(*clim))

    if title and show_plot:
        fig.suptitle(title, fontsize=20)

    if cbar:
        grid.cbar_axes[0].colorbar(mappable)
    else:
        grid.cbar_axes[0].remove()

    if show_plot:
        plt.show()

    return fig


def make_grid(images, nrow=None, norm=(None, None), axis=0):
    """Turn list of images into a single grid image."""
    images_arr = np.array(images)
    images_arr = np.moveaxis(images_arr, axis, 0)

    n = len(images_arr)
    if nrow is None:
        nrow = int(np.floor(np.sqrt(n)))
    ncol = n // nrow

    is_rgb = False
    if len(images_arr[0].shape) == 3:
        if images_arr[0].shape[-1] != 3:
            raise ValueError('3D images must have 3 channels for RGB.')
        else:
            # Rescale to [0, 1].
            vmin = norm[0] if norm[0] is not None else images_arr.min()
            vmax = norm[1] if norm[1] is not None else images_arr.max()
            images_arr = (images_arr - vmin) / (vmax - vmin)
            is_rgb = True
    h, w = images_arr[0].shape[:2]

    grid = np.zeros((nrow * h, ncol * w, 3)) if is_rgb else np.zeros((nrow * h, ncol * w))
    for row in range(nrow):
        for col in range(ncol):
            image_idx = nrow * row + col
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = images_arr[image_idx]
    return grid


def cube_edges(x0, y0, z0, size):
    # Return lines between cube corners.
    x, y, z = [], [], []
    corners = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                        [0, 1, 1]]) * size + [x0, y0, z0]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]
    for i, j in edges:
        x += [corners[i][0], corners[j][0], None]
        y += [corners[i][1], corners[j][1], None]
        z += [corners[i][2], corners[j][2], None]
    return x, y, z
