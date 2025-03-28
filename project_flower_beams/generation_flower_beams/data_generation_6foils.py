from extra_functions_package.all_knots_functions import *
import math
from tqdm import trange

"""
README:
------
This script performs a simulation to generate a 6-petal flower beam structure through turbulence.
 It:
  - Sets up 3D and 2D meshes for the knot.
  - Propagates the beam through phase screens.
  - Extracts the field (and its singularities) after propagation.
  - Optionally plots and saves intermediate fields, spectra, and singularity data.

Controlling features (set in the configuration section below):
  - Plotting: Toggle field plotting (plot), 3D dot plotting (plot_3d).
  - Verbose output: Toggle detailed prints with print_values.
  - Saving: Toggle saving of fields, spectra, and dots via save and spectrum_save.
  - Filtering: Enable filtering of isolated singularity points via filter_flag.

Make sure that the package “extra_functions_package.all_knots_functions” is in your PYTHONPATH.
"""


# ----------------------------------------------------------------------
# Utility Function: Filter Isolated Points
# ----------------------------------------------------------------------
def filter_isolated_points(dots):
    """
    Filters out points that do not have any neighbors within a 2-step radius.

    Parameters:
        dots (numpy.ndarray): Array of integer coordinates with shape (N, 3).

    Returns:
        numpy.ndarray: Filtered array containing only points with at least one neighbor.
    """
    filtered_dots = []
    dots_set = set(map(tuple, dots))
    for dot in dots:
        x, y, z = dot
        neighbor_count = 0
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dz in range(-3, 4):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    if (x + dx, y + dy, z + dz) in dots_set:
                        neighbor_count += 1
                    if neighbor_count > 2:
                        filtered_dots.append(dot)
                        break
                if neighbor_count > 2:
                    break
            if neighbor_count > 2:
                break
    return np.array(filtered_dots)


# ----------------------------------------------------------------------
# Configuration and Control Flags
# ----------------------------------------------------------------------

# generating structure:
knot_or_flower = unknot_6_any
# foils array defines the sizes of the lobes of each flower beam. 2 - full size, 1 - half size, 0 - no petal
foils = [[2, 2, 2, 2, 2, 2], [2, 1, 2, 2, 0, 2]]

# Colormap for visualization
custom_blues = plt.cm.gist_earth

# Control Flags (set to 1/True to enable)
SAMPLES = 1
plot = 1  # Field plotting
plot_3d = 1  # 3D dot plotting
print_values = 0  # Verbose output
save = True  # Save generated files
spectrum_save = 1  # Save spectrum data
no_turb = 0  # Disable turbulence if set to 1
filter_flag = False  # Apply filtering to singularities
centering = False  # Use default centering for the knot

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

# Turbulence simulation (phase screens) parameters
screens_num1 = 3
multiplier1 = [1] * screens_num1
screens_num2 = 1
multiplier2 = [1] * screens_num2

# Rytov parameter and foil configurations
Rytovs = [0.05]
# Foil configurations (each is a list of integers representing angle sizes)
seed = 1

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
for Rytov in Rytovs:
    k0 = 2 * np.pi / lmbda
    Cn2 = Cn2_from_Rytov(Rytov, k0, L_prop)
    L0 = 9e10  # Outer scale
    l0 = 5e-3 * 1e-10  # Inner scale

    # Propagation positions
    z0 = knot_length * (1 - center_plane) + L_prop
    prop1 = L_prop
    prop2 = knot_length * (1 - center_plane)

    beam_par = (0, 0, width0, lmbda)

    if print_values:
        print("Starting simulation for Rytov =", Rytov)

    # Turbulence parameters and phase screen settings
    r0 = r0_from_Cn2(Cn2, k0, dz=L_prop)
    if print_values:
        print(f"r0 = {r0}, 2w0/r0 = {2 * width0 / r0}")
    screens_num = screens_number(Cn2, k0, dz=L_prop)
    if print_values:
        print("Phase screens required:", screens_num)
    ryt = rytov(Cn2, k0, L_prop)
    if print_values:
        print("Rytov variance:", ryt)
    zR = k0 * width0 ** 2
    if print_values:
        print("Rayleigh Range (Zr) =", zR, "m")
    psh_par = (r0, res_xy_2D_origin, pxl_scale, L0, l0)
    psh_par_0 = (r0 * 1e100, res_xy_2D_origin, pxl_scale, L0, l0 * 1e100)
    if no_turb:
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
                                              multiplier=multiplier1, screens_num=screens_num1, seed=seed)
            if center_plane == 1:
                field_center = field_after_turb
            else:
                field_center = propagation_ps(field_after_turb, beam_par, psh_par_0, prop2,
                                              multiplier=multiplier2, screens_num=screens_num2, seed=seed)
            field_center = field_center / np.sqrt(np.sum(np.abs(field_center) ** 2))

            field_z_crop = field_center[
                           res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
                           res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2
                           ]

            if plot:
                plot_field_both(field_z_crop, extend=extend_crop)
            if save:
                np.save(f'foil6_field_XY_{Rytov}_{foil_str}.npy', field_z_crop)

            if spectrum_save:
                p1, p2 = 0, 10
                l1, l2 = -10, 10
                moments = {'p': (p1, p2), 'l': (l1, l2)}
                spectrum = cbs.LG_spectrum(field_center, **moments, mesh=mesh_2D_original, plot=False,
                                           width=width0, k0=k0, functions=LG_simple, x0=0, y0=0)
                spectrum = spectrum / np.sqrt(np.sum(np.abs(spectrum) ** 2)) * 100
                pl.plot_2D(np.abs(spectrum), x=np.arange(l1 - 0.5, l2 + 1.5),
                           y=np.arange(p1 - 0.5, p2 + 1.5), interpolation='none', grid=True,
                           xname='l', yname='p', map=custom_blues, show=False)
                plt.yticks(np.arange(p1, p2 + 1))
                plt.xticks(np.arange(l1, l2 + 1))
                plt.show()
                if save:
                    np.save(f'foil6_spectr_XY__{Rytov}_{foil_str}.npy', spectrum)

            field_3d = beam_expander(field_z_crop, beam_par, psh_par_0, distance_both=knot_length, steps_one=res_z // 2)
            if centering:
                x_cent_R, y_cent_R = find_center_of_intensity(field_z_crop)
                x_cent, y_cent = int(x_cent_R), int(y_cent_R)
            else:
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
            if filter_flag:
                dots_init = filter_isolated_points(dots_init)
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
            if save:
                np.save(f'foil6_dots_{Rytov}_{foil_str}.npy', dots_cut)
            if plot_3d:
                dots_bound = [[0, 0, 0], [crop_3d, crop_3d, res_z + 1]]
                pl.plotDots(dots_cut, dots_bound, color='black', show=True, size=10)