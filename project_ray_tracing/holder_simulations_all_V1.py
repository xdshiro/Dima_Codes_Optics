"""
Integrating Sphere Simulation with pvtrace

This module constructs and ray-traces an integrating sphere setup, featuring:
  • Definition of the cylinder shape geometry in the integrating-sphere (holder and bottom hole).
  • Configurable optical properties for three sample cases: RZ11, Z7, and RZ13.
  • Computation of transmission (T), direct transmission (T_dir), reflection (R),
    absorption (A), as well as absorption (L_A), attenuation (L), and scattering (L_S)
    lengths from those coefficients.
  • Monkey-patch instructions to override pvtrace’s Fresnel refraction for
    full total-internal-reflection support without editing package files.
  • Custom surface delegate (PartialTopSurfaceMirror) for applying perfect mirrors
    to selected top/bottom regions while using Fresnel elsewhere.
  • Beam-source utilities for collimated and focused Gaussian beams.
  • Functions to convert pvtrace ray paths into voxelized 3D intensity arrays
    (`array_3D_intensity_from_dots`) and planar projections (`plane_intensity`).
  • A `main()` driver that runs the simulation, saves both raw positions and
    intensity volumes, and plots cross-sectional slices with Gaussian smoothing.

"""

#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
# To enable proper total internal reflection, you must update pvtrace’s fresnel_refraction function.
# When you run the script, you will see a warning like:
#   …\pvtrace\material\utils.py:39: RuntimeWarning: invalid value encountered in sqrt
# Open that file, locate the fresnel_refraction definition, and replace it with:
#
# def fresnel_refraction(direction, normal, n1, n2):
#     vector = np.array(direction, dtype=float)
#     normal = np.array(normal, dtype=float)
#     ratio = n1 / n2
#     dot = np.dot(vector, normal)
#     discr = 1 - ratio**2 * (1 - dot**2)
#     c = np.sqrt(discr) if discr >= 0 else 0.0
#     sign = -1.0 if dot < 0 else 1.0
#     return ratio * vector + sign * (c - sign * ratio * dot) * normal
#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
import random
import logging
import pickle
import warnings
from scipy.ndimage import gaussian_filter
import numpy as np
# Ensure compatibility for older code expecting np.float and np.int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

import matplotlib.pyplot as plt
import functools
import collections
from numpy.random import Generator, PCG64, SeedSequence

# Custom modules
import pvtrace as pv
import helper_cross_sections as cs
from pvtrace.geometry.cylinder import Cylinder
from pvtrace.common.errors import GeometryError
from pvtrace.geometry.utils import norm

# ------------------------------------------------------------------------------
# Suppress warnings from NumPy and pvtrace internals
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", category=np.ComplexWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------------------
# Suppress verbose logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)





# --------------------------------------------------------------------------------
# Random Number Generator: reproducible for scattering calculations
rng = Generator(PCG64(SeedSequence(103)))

# --------------------------------------------------------------------------------
# Geometry Parameters (cm)
d_bottom_hole = 2
h_bottom_hole = 0.5
d_holder = 4.2
h_holder = 2.5

#--------------------------------------------------------------------------------
# Optical Properties: select one case by uncommenting
#-------------------------------------------------------------------------------
# === Case RZ11 ===
# total_energy = 3.96
# T = 0.27 / total_energy       # transmitted fraction
# T_dir = 0.18 / total_energy   # direct transmission fraction
# R = 0.50 / total_energy       # reflected fraction

# # === Case Z7 ===
total_energy = 8.41
T = 5.54 / total_energy       # transmitted fraction
T_dir = (3.62 - 0.03) / total_energy  # direct transmission fraction
R = (1.61 - 0.03) / total_energy      # reflected fraction

# # === Case RZ13 ===
# total_energy = 5.42
# T = 0.02 / total_energy       # transmitted fraction
# T_dir = 0.01 / total_energy   # direct transmission fraction
# R = 0.20 / total_energy       # reflected fraction


total_energy = total_energy  # ensure variable is defined
A = 1.0 - T - R
assert np.isclose(T + R + A, 1.0), "Optical fractions T, R, A must sum to 1."
length_m = h_holder # cm
TF = 1.0 - A
L_A = -length_m / np.log(TF)
L = -length_m / np.log(T)
L_S = L * L_A / (L_A - L)
print(L_A, L, L_S)
# --------------------------------------------------------------------------------
# Beam Source Parameters
r_source = 2.5  # cm
r_focus = 1e-3
dist_above_holder = h_holder + 4.6
focus_positions = [-h_holder]


# --------------------------------------------------------------------------------
# Custom Surface Delegate: PartialTopSurfaceMirror
class PartialTopSurfaceMirror(pv.FresnelSurfaceDelegate):
    """Partial mirror on top/bottom of cylinder; curved sides perfect reflector."""

    def reflected_direction(self, surface, ray, geometry, container, adjacent):
        n = np.array(geometry.normal(ray.position))
        d = np.array(ray.direction)
        return tuple(d - 2.0 * np.dot(d, n) * n)

    def reflectivity(self, surface, ray, geometry, container, adjacent):
        x, y, z = ray.position
        # Top face
        if getattr(geometry, 'radius', None) == d_holder / 2 and geometry.length == h_holder:
            if np.isclose(z, h_holder / 2) and x ** 2 + y ** 2 < (d_holder / 2) ** 2:
                return super().reflectivity(surface, ray, geometry, container, adjacent)
            if np.isclose(z, -h_holder / 2) and x ** 2 + y ** 2 < (d_bottom_hole / 2) ** 2:
                return super().reflectivity(surface, ray, geometry, container, adjacent)
            return 1.0
        # Bottom hole
        if getattr(geometry, 'radius', None) == d_bottom_hole / 2 and geometry.length == h_bottom_hole:
            if np.isclose(z, h_bottom_hole / 2) or np.isclose(z, -h_bottom_hole / 2):
                return super().reflectivity(surface, ray, geometry, container, adjacent)
            return 1.0
        return 1.0


class CylinderRough(Cylinder):
    """A cylinder that has a different definition of the normal method."""

    def normal(self, surface_point):
        def theta():
            theta_degree = random.uniform(0, 15)
            theta_rad = np.radians(theta_degree)
            return theta_rad

        def random_vector_with_angle_to_z(theta_rad):
            """Return a random unit vector with a given angle to the (0,0,-1) vector."""

            # Generate a random azimuthal angle in [0, 2*pi]
            phi = random.uniform(0, 2 * np.pi)

            # Spherical to Cartesian conversion for unit vector
            x = np.sin(theta_rad) * np.cos(phi)
            y = np.sin(theta_rad) * np.sin(phi)
            z = np.cos(theta_rad)

            # Since we want the angle with (0,0,-1), we flip the z coordinate
            z = z
            return (x, y, z)

        """Override the normal method with a new definition."""
        z = surface_point[2]
        if np.isclose(z, -0.5 * self.length):

            return (0.0, 0.0, -1.0)
        elif np.isclose(z, 0.5 * self.length):
            theta_rad = theta()
            normal_ = random_vector_with_angle_to_z(theta_rad)
            # print(normal_)
            return normal_
            # return (0.0, 0.0, 1.0)
        elif np.isclose(self.radius, np.sqrt(np.sum(np.array(surface_point[:2]) ** 2))):
            v = np.array(surface_point) - np.array([0.0, 0.0, surface_point[2]])
            n = tuple(norm(v).tolist())
            return n
        else:
            raise GeometryError("Not a surface point.")


# --------------------------------------------------------------------------------
# Utility Functions

def cylindrical_to_cart(r, phi, z=0.0):
    x = r * np.cos(phi);
    y = r * np.sin(phi)
    return (x, y, z) if np.isscalar(x) else np.column_stack((x, y, z))


def collimated_beam(r):
    if r <= 0: raise ValueError(f"Aperture radius must be positive, got {r}")
    x, y = rng.normal(0, r, 2)
    while x ** 2 + y ** 2 > r ** 2: x, y = rng.normal(0, r, 2)
    return (x, y, 0.0)


random_counter = 0
random_seed = 0


# --------------------------------------------------------------------------------
# Scene & Structure Construction
def structure_sample(parent, absor=1.0, scat=1.0):
    holder = pv.Node(
        name="holder",
        geometry=CylinderRough(radius=d_holder / 2, length=h_holder,
                               material=pv.Material(refractive_index=1.1,
                                                    surface=pv.Surface(delegate=PartialTopSurfaceMirror()),
                                                    components=[pv.Absorber(coefficient=absor),
                                                                pv.Scatterer(coefficient=scat)]))
        , parent=parent)
    holder.translate([0, 0, h_holder / 2])
    bottom = pv.Node(name="bottom_hole",
                     geometry=pv.Cylinder(radius=d_bottom_hole / 2, length=h_bottom_hole,
                                          material=pv.Material(refractive_index=1.0,
                                                               surface=pv.Surface(delegate=PartialTopSurfaceMirror()))),
                     parent=parent)
    bottom.translate([0, 0, -h_bottom_hole / 2])
    return holder, bottom


def light_beam(parent, focus_z):
    light = pv.Node(name="Light",
                    light=pv.Light(direction=functools.partial(direction, focus_z),
                                   position=functools.partial(position, focus_z), wavelength=lambda: 455),
                    parent=parent)
    light.translate([0, 0, dist_above_holder])
    light.rotate(np.pi, [1, 0, 0])
    return light


def pv_scene_real(absor=1e-10, scat=1e-10, focus=1e9):
    world = pv.Node(name="world", geometry=pv.Sphere(radius=17, material=pv.Material(refractive_index=1.0)))
    structure_sample(world, absor, scat)
    light_beam(world, focus)
    return pv.Scene(world)


def plane_intensity(positions, plane_vec=(0, 0, 1), plane_dot=(0, 0, 0), x_res=21, y_res=21,
                    x_max_min=(-1, 1), y_max_min=(-1, 1)):
    a, b, c = plane_vec;
    d = -np.dot(plane_vec, plane_dot)
    plane = np.array([a, b, c, d]);
    pts = []
    for path in positions:
        for p1, p2 in zip(path[:-1], path[1:]):
            n1, n2 = np.dot(p1, plane[:3]) + d, np.dot(p2, plane[:3]) + d
            if n1 * n2 < 0:
                t = - (np.dot(plane[:3], p1) + d) / np.dot(plane[:3], p2 - p1)
                pts.append(p1 + t * (p2 - p1))
    pts = np.array(pts);
    dx, dy = (x_max_min[1] - x_max_min[0]) / (x_res - 1), (y_max_min[1] - y_max_min[0]) / (y_res - 1)
    idx = np.rint(pts[:, :2] * [1 / dx, 1 / dy]).astype(int)
    cnt = collections.Counter(map(tuple, idx))
    I = np.zeros((x_res, y_res), int)
    for (i, j), v in cnt.items():
        if 0 <= i < x_res and 0 <= j < y_res: I[i, j] = v
    return pts, I


# def positions_directions(focus_z):
#     """
#     Compute ray origin and direction toward a focal plane z = focus_z.
#     Alternates RNG seed for reproducibility and decorrelation.
#     """
#     global random_counter, random_seed
#     if random_counter % 2 == 0:
#         random_seed += 1
#     gen = np.random.default_rng(random_seed)
#     # Sample source point
#     x0, y0 = gen.normal(0, r_source, 2)
#     while x0 ** 2 + y0 ** 2 > r_source ** 2:
#         x0, y0 = gen.normal(0, r_source, 2)
#     # Sample focal spot point with jitter
#     xf, yf = gen.normal(0, r_focus, 2)
#     zf = focus_z + gen.uniform(-1e-3, 1e-3)
#     v = np.array([xf - x0, yf - y0, zf])
#     v /= np.linalg.norm(v)
#     random_counter += 1
#     return (x0, y0, 0.0), tuple(v)

def positions_directions(focus):
    """
    Generate a random starting point on the source aperture and a beam direction
    aimed roughly toward (0, 0, focus) with Gaussian angular spread.

    Parameters
    ----------
    focus : float
        z‐coordinate of the focal plane.

    Returns
    -------
    coordinates : tuple of float
        (x, y, 0) starting position on the source plane.
    direction : ndarray of shape (3,)
        Unit vector giving the ray direction.
    """
    global random_counter, random_seed, r_source, r_focus

    # Alternate seed every other call for decorrelation
    if random_counter % 2 == 0:
        random_seed += 1
    rng = np.random.default_rng(random_seed)

    # Sample (x, y) uniformly within a circle of radius r
    theta = rng.uniform(0, 2 * np.pi)
    rho = np.sqrt(rng.uniform(0, 1)) * r_source
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    coordinates = (x, y, 0)

    # Define the nominal focus point
    z0 = focus

    # Base direction vector pointing straight toward (x, y, z0)
    base_dir = np.array([0.0, 0.0, z0])
    base_dir /= np.linalg.norm(base_dir)

    # Gaussian angular spread around the cone axis
    cone_sigma = 0.184 * np.pi  # standard deviation of angular deviation
    delta_theta = rng.normal(0, cone_sigma)
    delta_phi = rng.uniform(0, 2 * np.pi)

    # Construct a small perturbation in local spherical coords
    perturb = np.array([
        np.sin(delta_theta) * np.cos(delta_phi),
        np.sin(delta_theta) * np.sin(delta_phi),
        np.cos(delta_theta)
    ])

    # Combine and renormalize to get the final direction
    direction = base_dir + perturb
    direction /= np.linalg.norm(direction)

    random_counter += 1
    return coordinates, direction


def position(focus_z):
    """Return ray origin position for given focus_z."""
    return positions_directions(focus_z)[0]


def direction(focus_z):
    """Return ray direction unit vector for given focus_z."""
    return positions_directions(focus_z)[1]


# --------------------------------------------------------------------------------
# 3D Intensity Grid Utilities

def array_3D_intensity_from_dots(dots, x_res, y_res, z_res,
                                 x_max_min=(-1, 1), y_max_min=(-1, 1), z_max_min=(-1, 1)):
    """
    Build a 3D intensity array by counting dot occurrences in grid voxels.

    Parameters:
    - dots: (N,3) array of world-coordinate points
    - x_res, y_res, z_res: grid resolution
    - x_max_min, y_max_min, z_max_min: domain bounds tuples

    Returns:
    - I: (x_res, y_res, z_res) numpy array of counts
    """
    # Compute centering offsets
    x_cen = int((-x_max_min[0]) / (x_max_min[1] - x_max_min[0]) * x_res)
    y_cen = int((-y_max_min[0]) / (y_max_min[1] - y_max_min[0]) * y_res)
    z_cen = int((-z_max_min[0]) / (z_max_min[1] - z_max_min[0]) * z_res)
    # Shift dots to grid indices
    idxs = (dots + (x_cen, y_cen, z_cen)).round().astype(int)
    # Initialize intensity grid
    I = np.zeros((x_res, y_res, z_res), dtype=int)
    for x, y, z in idxs:
        if 0 <= x < x_res and 0 <= y < y_res and 0 <= z < z_res:
            I[x, y, z] += 1
    return I


def array_3D_intensity_from_dots_avg(dots_3D):
    """
    Smooth a 3D intensity volume by averaging each voxel with its 26 neighbors.

    Parameters:
    - dots_3D: (x_res, y_res, z_res) 3D numpy intensity array

    Returns:
    - I_avg: smoothed 3D intensity array
    """
    # Pad with linear ramp to avoid edge artifacts
    pad = np.pad(dots_3D, pad_width=1, mode='linear_ramp')
    I_avg = np.zeros_like(dots_3D, dtype=float)
    # Local averaging over 3x3x3 window
    for i in range(I_avg.shape[0]):
        for j in range(I_avg.shape[1]):
            for k in range(I_avg.shape[2]):
                I_avg[i, j, k] = pad[i:i + 3, j:j + 3, k:k + 3].sum() / 27.0
    return I_avg


def lines_dots(positions, x_res, y_res, z_res,
               x_max_min=(-1, 1), y_max_min=(-1, 1), z_max_min=(-1, 1),
               length_line=1.0, res_line=4):
    """
    Discretize each ray path into voxel indices by sampling points along
    each segment, clipping to the given domain, scaling to grid, and
    returning unique integer indices.

    Parameters
    ----------
    positions : list of (N_i, 3) arrays
        Each entry is a sequence of 3D points describing a ray path.
    x_res, y_res, z_res : int
        Number of voxels in each dimension.
    x_max_min, y_max_min, z_max_min : tuple of float
        (min, max) bounds in world coordinates for clipping.
    length_line : float
        Reference length for sampling density.
    res_line : int
        Base number of samples per unit length.

    Returns
    -------
    voxels : (M, 3) ndarray of int
        Unique voxel indices visited by any ray.
    """
    # Compute world‐to‐grid scale factors
    dx = (x_max_min[1] - x_max_min[0]) / (x_res - 1)
    dy = (y_max_min[1] - y_max_min[0]) / (y_res - 1)
    dz = (z_max_min[1] - z_max_min[0]) / (z_res - 1)
    sx, sy, sz = 1.0 / dx, 1.0 / dy, 1.0 / dz

    def sample_segment(p1, p2):
        """Sample points uniformly along the segment p1→p2, clipped to bounds."""
        vec = p2 - p1
        L = np.linalg.norm(vec)
        npts = max(2, int(res_line * L / length_line))
        t = np.linspace(0, 1, npts)[:, None]  # shape (npts, 1)
        pts = p1 + t * vec  # shape (npts, 3)

        # Clip to world bounds
        mask = (
                (pts[:, 0] >= x_max_min[0]) & (pts[:, 0] <= x_max_min[1]) &
                (pts[:, 1] >= y_max_min[0]) & (pts[:, 1] <= y_max_min[1]) &
                (pts[:, 2] >= z_max_min[0]) & (pts[:, 2] <= z_max_min[1])
        )
        return pts[mask]

    all_voxels = []
    # Loop over each ray path
    for path in positions:
        # Walk segments between consecutive points
        for p1, p2 in zip(path[:-1], path[1:]):
            pts = sample_segment(p1, p2)
            if pts.size == 0:
                continue
            # Scale to grid and round to nearest voxel
            idx = np.rint(pts * [sx, sy, sz]).astype(int)
            # Keep only unique indices per segment
            uniq = np.unique(idx, axis=0)
            all_voxels.append(uniq)

    if not all_voxels:
        return np.empty((0, 3), dtype=int)

    # Concatenate all segments and return
    return np.vstack(all_voxels)


# --------------------------------------------------------------------------------
# Main Driver: simulation and plotting
# --------------------------------------------------------------------------------
def main():
    """
    Render and analyze ray-tracing results for a series of focus positions.

    For each focus in `focus_positions`:
      1. Create a PVTrace scene with given absorption and scattering lengths.
      2. Trace `number_rays` through the scene.
      3. Voxelize the resulting ray paths into a 3D intensity grid.
      4. Save the 3D data (`.npy`) and the raw ray positions (`.pkl`).
      5. Extract three orthogonal slices (XY at mid-y, XY at mid-z, XZ at z=0).
      6. Apply a Gaussian filter (σ=4) with spline36 interpolation to each slice.
      7. Display the filtered slices side-by-side.


    Output files (per focus):
      Z7_f{focus:.2f}_r{r_source:.2f}_{x_res}res_{number_rays}rays.npy
      Z7_f{focus:.2f}_r{r_source:.2f}_{x_res}res_{number_rays}rays_pos.pkl
    """
    number_rays = 100
    show_3d = True
    save_files = True
    x_res = y_res = z_res = 221

    # Define physical domain bounds for x, y, z
    xM = (-d_holder / 2 - 0.1, d_holder / 2 + 0.1)
    yM = xM
    zM = (-h_bottom_hole * 1.000001, h_holder * 1.000001)

    for focus in focus_positions:
        # 1) Build and render the scene
        scene = pv_scene_real(
            absor=1. / L_A,
            scat=1. / L_S,
            focus=focus + dist_above_holder
        )
        positions = cs.scene_render_and_positions(
            scene,
            rays_number=number_rays,
            show_3d=show_3d,
            random_seed=2
        )

        # 2) Voxelize ray paths into 3D intensity grid
        dots = lines_dots(
            positions,
            x_res, y_res, z_res,
            xM, yM, zM,
            res_line=int(np.sqrt(x_res**2 + y_res**2 + z_res**2)),
            length_line=1
        )
        dots_3D = array_3D_intensity_from_dots(
            dots,
            x_res, y_res, z_res,
            xM, yM, zM
        )
        # 3) Save the results to disk
        if save_files:
            tag = f"Z7_f{focus:.2f}_r{r_source:.2f}_{x_res}res_{number_rays}rays"
            np.save(f"{tag}.npy", dots_3D)

        # 4) Extract slices and apply Gaussian smoothing
        slices = [
            dots_3D[:, y_res // 2, :],   # XY at middle y
            dots_3D[:, :, z_res // 2],   # XY at middle z
            dots_3D[:, :, 0]             # XZ at y=0
        ]
        titles = [
            "XZ mid-plane (y=0)",
            "XY mid-plane (z=h/2)",
            "XY plane (z=0)"
        ]

        # 5) Plot filtered slices side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, slc, title in zip(axes, slices, titles):
            # Transpose so axes align correctly under imshow
            im = gaussian_filter(slc.T.astype(float), sigma=4)
            ax.imshow(im, cmap='hot', interpolation='spline36', origin='lower')
            ax.set_title(title)
            ax.axis('off')

        # plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
