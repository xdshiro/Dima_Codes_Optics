"""
README:
------
This package is designed for calculating Strehl Ratio (SR) values and provides functions to compute
other important turbulence parameters. It is not intended for full beam propagation simulations.
Key turbulence parameters include:
  - Fried parameter (r0): A measure of the coherence diameter of the atmosphere.
  - Refractive index structure parameter (Cn2): Quantifies the strength of atmospheric turbulence.

Features:
  - Calculation of SR values using a Gaussian Fourier method.
  - Computation of turbulence metrics such as r0 and Cn2.
  - External control over the number of calculation epochs.
  - Optional plotting functionality to visualize fields and turbulence screens.
  - you can set up the !!reversed propagation!! of the beam (from -L to 0 instead of
  0 to L). For that you need to swap the part of the code in the end of run_simulation

Usage:
  1. Adjust the simulation parameters at the bottom of the file.
  2. Set the desired number of epochs and toggle plotting via the provided flags.
  3. Run the script to calculate the SR values and other turbulence parameters.

Note:
  Ensure that the "extra_functions_package" is available in your Python path.
"""

from extra_functions_package.all_knots_functions import *


def crop_field_3d(field_3d, crop_percentage):
    """
    Crop a 3D field along the x and y dimensions by a specified percentage.

    Parameters:
        field_3d (numpy.ndarray): 3D array representing the field.
        crop_percentage (float): Percentage of the field dimensions to crop.

    Returns:
        tuple: (cropped_field, new_x_size, new_y_size)
    """
    x_size, y_size, z_size = field_3d.shape

    # Determine center indices
    center_x, center_y = x_size // 2, y_size // 2

    # Calculate crop size based on the percentage
    crop_x = int(crop_percentage * x_size / 100)
    crop_y = int(crop_percentage * y_size / 100)

    # Calculate boundaries ensuring they do not exceed array limits
    start_x = max(center_x - crop_x // 2, 0)
    end_x = min(center_x + crop_x // 2, x_size)
    start_y = max(center_y - crop_y // 2, 0)
    end_y = min(center_y + crop_y // 2, y_size)

    # Crop the array
    cropped_field = field_3d[start_x:end_x, start_y:end_y, :]

    return cropped_field, end_x - start_x, end_y - start_y


def run_simulation(L_prop, width0, xy_lim_2D, res_xy_2D, Rytov, l0, L0, screens_nums, knot_length, epochs=1,
                   plot=False):
    """
    Calculate the Strehl Ratio (SR) and turbulence parameters using the Gaussian Fourier method.

    Parameters:
        L_prop (float): Propagation distance.
        width0 (float): Initial beam width.
        xy_lim_2D (tuple): Limits for the 2D grid (min, max).
        res_xy_2D (int): Number of grid points in each dimension.
        Rytov (float): Rytov variance parameter.
        l0 (float): Turbulence inner scale.
        L0 (float): Turbulence outer scale.
        screens_nums (int): Number of phase screens.
        knot_length (int): Parameter controlling knot length (usage defined by simulation).
        epochs (int, optional): Number of calculation epochs. Default is 1.
        plot (bool, optional): If True, plotting functions are executed. Default is False.

    Returns:
        tuple: (SR, currents_SR) simulation results.
    """
    # Define beam parameters
    lmbda = 532e-9  # wavelength in meters
    l, p = 0, 0  # beam mode parameters (e.g., for Gaussian beams)
    beam_par = (l, p, width0, lmbda)
    k0 = 2 * np.pi / lmbda

    # Create 2D spatial grid and mesh
    xy_2D = np.linspace(*xy_lim_2D, res_xy_2D)
    mesh_2D = np.meshgrid(xy_2D, xy_2D, indexing='ij')
    pxl_scale = (xy_lim_2D[1] - xy_lim_2D[0]) / (res_xy_2D - 1)

    # Calculate turbulence parameters:
    # Cn2: Refractive index structure parameter
    Cn2 = Cn2_from_Rytov(Rytov, k0, L_prop)
    # r0: Fried parameter (coherence diameter)
    r0 = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=L_prop)
    print(f"\n[INFO] Fried parameter (r0): {r0:.4e}")
    print(f"[INFO] Beam width to r0 ratio (2*w0/r0): {2 * width0 / r0:.4e}")

    psh_par = (r0, res_xy_2D, pxl_scale, L0, l0)

    ryt = rytov(Cn2, k0, L_prop)
    print(f"[INFO] Rytov variance: {ryt:.4e}")
    print(f"[INFO] Cn2 (refractive index structure parameter) from Rytov: {Cn2_from_Rytov(Rytov, k0, L_prop):.4e}")

    # Generate Gaussian beam (renamed for clarity)
    gaussian_beam = LG_simple(*mesh_2D, z=0, l=l, p=p, width=width0, k0=k0, x0=0, y0=0, z0=0)

    # Optionally plot the initial beam field
    if plot:
        plot_field_both(gaussian_beam, extend=None)

    # Generate a phase screen and optionally plot its amplitude and phase
    phase_screen = psh_wrap(psh_par, seed=1)
    if plot:
        plot_field(phase_screen, extend=None)

    # Propagate the beam through the turbulence screens (if needed for calculation)
    field_prop = propagation_ps(gaussian_beam, beam_par, psh_par, L_prop, screens_num=screens_nums)
    if plot:
        plot_field_both(field_prop)

    # Calculate the Strehl Ratio using the Gaussian Fourier method with the specified number of epochs
    SR, currents_SR = SR_gauss_fourier(
        mesh_2D, L_prop, beam_par, psh_par,
        epochs=epochs,
        screens_num=screens_nums,
        max_cut=False,
        pad_factor=4
    )
    # you can set the reversed gaussian regime
    # beam starts propagation from z=-L

    # SR, currents_SR = SR_reversed_gauss_fourier(
    #     mesh_2D, L_prop, beam_par, psh_par,
    #     epochs=epochs,
    #     screens_num=screens_nums,
    #     max_cut=False,
    #     pad_factor=4
    # )

    return SR, currents_SR


# =============================================================================
# Simulation Parameter Definitions
# =============================================================================
L_prop_values = [270]
knot_length = 100
width0_values = [6e-3 / np.sqrt(2)]
xy_lim_2D_values = [(-45.0e-3, 45.0e-3)]
res_xy_2D_values = [301]
Rytov_values = [0.15]  # Example values for different simulation cases
l0_values = [3e-3]
L0_values = [10]
screens_numss = [1]
# amount of epochs should be high (>500 at least)
simulation_epochs = 5  # Set the number of epochs (adjust as needed)
enable_plotting = True  # Set to True to enable plotting, False to disable

# Ensure all parameter lists have the same length by repeating single-element lists
max_len = max(len(L_prop_values), len(width0_values), len(xy_lim_2D_values),
              len(res_xy_2D_values), len(Rytov_values), len(l0_values), len(L0_values))

L_prop_values = L_prop_values if len(L_prop_values) > 1 else L_prop_values * max_len
width0_values = width0_values if len(width0_values) > 1 else width0_values * max_len
xy_lim_2D_values = xy_lim_2D_values if len(xy_lim_2D_values) > 1 else xy_lim_2D_values * max_len
res_xy_2D_values = res_xy_2D_values if len(res_xy_2D_values) > 1 else res_xy_2D_values * max_len
Rytov_values = Rytov_values if len(Rytov_values) > 1 else Rytov_values * max_len
l0_values = l0_values if len(l0_values) > 1 else l0_values * max_len
L0_values = L0_values if len(L0_values) > 1 else L0_values * max_len

# Zip parameters together for iterative simulation runs
parameter_sets = list(
    zip(L_prop_values, width0_values, xy_lim_2D_values, res_xy_2D_values, Rytov_values, l0_values, L0_values))

currents_list = []
currents_SR_list = []

# =============================================================================
# Run Simulations
# =============================================================================
for params in parameter_sets:
    for screens_nums in screens_numss:
        print("\n==============================")
        print("Running simulation with the following parameters:")
        print(f"  Propagation distance (L_prop): {params[0]}")
        print(f"  Beam width (width0): {params[1]:.4e}")
        print(f"  2D grid limits (xy_lim_2D): {params[2]}")
        print(f"  Grid resolution (res_xy_2D): {params[3]}")
        print(f"  Rytov parameter: {params[4]}")
        print(f"  Turbulence inner scale (l0): {params[5]:.4e}")
        print(f"  Turbulence outer scale (L0): {params[6]}")
        print(f"  Number of phase screens: {screens_nums}")
        print("==============================")

        # External control for simulation epochs and plotting option


        currents, currents_SR = run_simulation(
            *params,
            screens_nums=screens_nums,
            knot_length=knot_length,
            epochs=simulation_epochs,
            plot=enable_plotting
        )
        currents_list.append(currents)
        currents_SR_list.append(currents_SR)

print("\n[RESULT] Simulation currents list:")
print(currents_list)

# Optional: Save simulation results to files
# np.save('arrays_scin.npy', np.array(currents_list))
# np.save(f'arrays_SR_L{L_prop_values[0]}.npy', np.array(currents_SR_list))