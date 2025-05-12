"""
flower_beam_simulation.py

Generate and visualize “flower” beam singularity structures with 4, 5, or 6 lobes.

Features:
  - Choose between 4-, 5-, or 6-petal flower beams via `knot_or_flower` flag
  - Specify petal sizes (full, half, or absent) with the `foils` list of int codes:
        2 = full-sized petal, 1 = half-sized petal, 0 = no petal
  - Compute 2D field slice and 3D singularity dot plots of the resulting beam
  - Optional turbulence propagation via Kolmogorov phase-screen model
  - Center-plane propagation and beam expansion to extract 3D structure
  - Adjustable plotting, saving, and verbosity flags

Usage:
    1. Set `knot_or_flower` to one of:
           unknot_4_any, unknot_5_any, unknot_6_any
    2. Define `foils` as a list of lists, e.g. [[2,2,1,2,0]] for a 5-petal beam
    3. Toggle `plot`, `plot_3d`, and `print_values` as needed
    4. Run: python flower_beam_simulation.py
"""


from extra_functions_package.all_knots_functions import *
import math
from tqdm import trange



# ----------------------------------------------------------------------
# Configuration and Control Flags
# ----------------------------------------------------------------------

# generating structure: you can choose out of 4 5 and 6-lobes flower beams
# # # # # # # # # # # # # # # # # # # ## # # # # # # # # ## # # # # # # # # #
# knot_or_flower = unknot_4_any
knot_or_flower = unknot_5_any
# knot_or_flower = unknot_6_any
# foils array defines the sizes of the lobes of each flower beam. 2 - full size, 1 - half size, 0 - no petal
# type the flower beam based on the amount of lobes
foils = [[2, 2, 1, 2, 0]]

# Colormap for visualization
custom_blues = plt.cm.gist_earth


plot = 1  # Field plotting
plot_3d = 1  # 3D dot plotting
print_values = 0  # Verbose output
# ----------------------------------------------------------------------
# Simulation Parameters
# ----------------------------------------------------------------------
# Mesh boundaries for the 3D knot
x_lim_3D_knot, y_lim_3D_knot, z_lim_3D_knot = (-7.0, 7.0), (-7.0, 7.0), (-2.0, 2.0)
res_x_3D_knot, res_y_3D_knot, res_z_3D_knot = 256, 256, 1

# Beam and propagation parameters
lmbda = 532e-9  # Wavelength (m)
L_prop = 270  # Propagation distance (m)
knot_length = 212.58897655870774 / 2 * 1.4  # Detector distance from knot center
center_plane = 1  # Flag for center plane propagation
width0 = 6.0e-3 / np.sqrt(2)  # Beam width (m)
xy_lim_2D_origin = (-35.0e-3, 35.0e-3)  # 2D window limits (m)
scale = 1
res_xy_2D_origin = int(scale * 300)  # 2D resolution
res_z = int(scale * 64)  # z-resolution for knot
crop = int(scale * 185)  # Crop size for propagation
crop_3d = int(scale * 100)  # Crop size for knot extraction
new_resolution = (64, 64)  # Final knot resolution

multiplier1 = [1]
multiplier2 = [1]


# Foil configurations (each is a list of integers representing angle sizes)
seed = 1
# amount of samples to generate
SAMPLES = 1
# ----------------------------------------------------------------------
# Pre-calculate 2D and 3D Meshes
# ----------------------------------------------------------------------
# 3D mesh for knot generation
x_3D_knot = np.linspace(*x_lim_3D_knot, res_x_3D_knot)
y_3D_knot = np.linspace(*y_lim_3D_knot, res_y_3D_knot)
z_3D_knot = np.linspace(*z_lim_3D_knot, res_z_3D_knot) if res_z_3D_knot != 1 else 0
mesh_3D_knot = np.meshgrid(x_3D_knot, y_3D_knot, z_3D_knot, indexing='ij')

# 2D mesh for initial field
x_2D_origin = np.linspace(*xy_lim_2D_origin, res_xy_2D_origin)
y_2D_origin = np.linspace(*xy_lim_2D_origin, res_xy_2D_origin)
mesh_2D_original = np.meshgrid(x_2D_origin, y_2D_origin, indexing='ij')

extend = [*xy_lim_2D_origin, *xy_lim_2D_origin]
xy_lim_2D_crop = list(np.array(xy_lim_2D_origin) / res_xy_2D_origin * crop)
extend_crop = [*xy_lim_2D_crop, *xy_lim_2D_crop]
xy_lim_2D_crop3d = list(np.array(xy_lim_2D_crop) / crop * crop_3d)
extend_crop3d = [*xy_lim_2D_crop3d, *xy_lim_2D_crop3d]
pxl_scale = (xy_lim_2D_origin[1] - xy_lim_2D_origin[0]) / (res_xy_2D_origin - 1)
D_window = (xy_lim_2D_origin[1] - xy_lim_2D_origin[0])
perfect_scale = lmbda * np.sqrt(L_prop ** 2 + (D_window / 2) ** 2) / D_window
if print_values:
    print(
        f'dx={pxl_scale * 1e6:.2f} um, perfect={perfect_scale * 1e6:.2f} um, res required={math.ceil(D_window / perfect_scale + 1)}')

# ----------------------------------------------------------------------
# Main Simulation Loop
# ----------------------------------------------------------------------

k0 = 2 * np.pi / lmbda
L0 = 9e10  # Outer scale
l0 = 5e-3 * 1e-10  # Inner scale

# Propagation positions
z0 = knot_length * (1 - center_plane) + L_prop
prop1 = L_prop
prop2 = knot_length * (1 - center_plane)

beam_par = (0, 0, width0, lmbda)

# Turbulence parameters and phase screen settings



psh_par_0 = (1 * 1e100, res_xy_2D_origin, pxl_scale, L0, l0 * 1e100)
psh_par = psh_par_0

# Loop over each foil configuration
for foil in foils:
    foil_str = ''.join(map(str, foil))
    if print_values:
        print("Processing foil:", foil)
    values = knot_or_flower(mesh_3D_knot, braid_func=braid, plot=True,
                          angle_size=foil, cmap=custom_blues)
    field_before_prop = field_knot_from_weights(values, mesh_2D_original, width0, k0=k0, x0=0, y0=0, z0=z0)

    for indx in trange(SAMPLES, desc="Propagation Progress"):
        field_after_turb = propagation_ps(field_before_prop, beam_par, psh_par, prop1,
                                          multiplier=multiplier1, screens_num=1, seed=seed)
        if center_plane == 1:
            field_center = field_after_turb
        else:
            field_center = propagation_ps(field_after_turb, beam_par, psh_par_0, prop2,
                                          multiplier=multiplier2, screens_num=1, seed=seed)
        field_center = field_center / np.sqrt(np.sum(np.abs(field_center) ** 2))

        field_z_crop = field_center[
                       res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
                       res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2
                       ]

        if plot:
            plot_field_both(field_z_crop, extend=extend_crop)

        field_3d = beam_expander(field_z_crop, beam_par, psh_par_0, distance_both=knot_length, steps_one=res_z // 2)

        x_cent, y_cent = crop // 2, crop // 2
        if print_values:
            print(f'Center: {x_cent}, {y_cent} (crop size: {crop})')

        field_3d_crop = field_3d[
                        x_cent - crop_3d // 2: x_cent + crop_3d // 2,
                        y_cent - crop_3d // 2: y_cent + crop_3d // 2,
                        :
                        ]
        field_3d_crop = field_3d_crop[:, :, :-1]

        dots_init_dict, dots_init = sing.get_singularities(np.angle(field_3d_crop), axesAll=False, returnDict=True)

        dots_cut_non_unique = cut_circle_dots(dots_init, crop_3d // 2, crop_3d // 2, crop_3d // 2)
        view = np.ascontiguousarray(dots_cut_non_unique).view(
            np.dtype((np.void, dots_cut_non_unique.dtype.itemsize * dots_cut_non_unique.shape[1]))
        )
        _, idx = np.unique(view, return_index=True)
        dots_cut = dots_cut_non_unique[idx]

        original_resolution = (crop_3d, crop_3d)
        scale_x = new_resolution[0] / original_resolution[0]
        scale_y = new_resolution[1] / original_resolution[1]
        xy = dots_cut[:, :2]
        z = dots_cut[:, 2]
        scaled_xy = np.rint(xy * [scale_x, scale_y]).astype(int)
        scaled_data = np.column_stack((scaled_xy, z))
        if plot_3d:
            dots_bound = [[0, 0, 0], [crop_3d, crop_3d, res_z + 1]]
            pl.plotDots(dots_cut, dots_bound, color='black', show=True, size=10)