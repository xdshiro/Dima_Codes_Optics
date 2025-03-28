"""
Braid Knot Field Calculation and Visualization

This module provides functions to calculate and visualize knot fields using braid functions
and Laguerre-Gaussian (LG) modes. The main objective is to generate various configurations
of braided fields, such as trefoils, hopf links, and borromean rings, and to visualize their
complex field structures. The calculations utilize a combination of braid and LG functions to
produce unique field configurations that are further analyzed for their significant LG mode components.

Key Features:
-------------
1. **General Braid Field Calculation**:
    - `general_braid_function()`: A versatile function to compute braided fields based on a
      user-defined set of parameters including rotations, shifts, and braid properties.

2. **Preconfigured Knot and Braid Functions**:
    - Specific implementations like `trefoil_standard`, `hopf_standard`, and `borromean`
      provide ready-to-use braid fields with typical configurations such as different widths
      and angles.

3. **Lobe Modification Utilities**:
    - Functions like `lobe_modify()` allow for the selective scaling or removal of lobes in
      the field, providing additional control over field configurations.
    - `unknot_general()` serves as a function to create "unknotted" braids with customizable
      lobe properties.

4. **Optimized Knot Configurations**:
    - Optimized functions, such as `trefoil_optimized` and `hopf_optimized`, use precomputed
      weights to quickly generate field configurations without recalculating every time.

5. **Combination and Reconstruction**:
    - `field_combination_LG()` and `field_knot_from_weights()` help in combining different
      LG modes and reconstructing the knot field from weighted LG modes respectively.

Usage Example:
--------------
The module is intended to be used as a library where different knot fields can be generated,
modified, and visualized. The script includes a `__main__` section demonstrating how to
generate and visualize a trefoil knot configuration
"""
from extra_functions_package.functions_turbulence import *
import extra_functions_package.singularities as sing
import extra_functions_package.plotings as pl
import extra_functions_package.center_beam_search as cbs
import numpy as np


def u(x, y, z):
    numerator = x ** 2 + y ** 2 + z ** 2 - 1 + 2j * z
    denominator = x ** 2 + y ** 2 + z ** 2 + 1
    return numerator / denominator


def v(x, y, z):
    numerator = 2 * (x + 1j * y)
    denominator = x ** 2 + y ** 2 + z ** 2 + 1
    return numerator / denominator


def braid(x, y, z, angle=0, pow_cos=1, pow_sin=1, theta=0, a_cos=1, a_sin=1):
    def cos_v(x, y, z, power=1):
        return (v(x, y, z) ** power + np.conj(v(x, y, z)) ** power) / 2

    def sin_v(x, y, z, power=1):
        return (v(x, y, z) ** power - np.conj(v(x, y, z)) ** power) / 2j

    angle_3D = np.ones(np.shape(z)) * angle
    a_cos_3D = np.ones(np.shape(z)) * a_cos
    a_sin_3D = np.ones(np.shape(z)) * a_sin

    return u(x, y, z) * np.exp(1j * theta) - (
            cos_v(x, y, z, pow_cos) / a_cos_3D + 1j
            * sin_v(x, y, z, pow_sin) / a_sin_3D) * np.exp(1j * angle_3D)


def rotate_mesh_grids(mesh_3D, rotations):
    rotated_grids = [rotate_meshgrid(*mesh_3D, *rotation) for rotation in rotations]
    return rotated_grids


def calculate_braid_product(xyz_array, braid_func, angle_array, pow_cos_array, pow_sin_array, theta_array, a_cos_array,
                            a_sin_array, conj_array):
    ans = 1
    for i, xyz in enumerate(xyz_array):
        braid_value = braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                 a_cos_array[i], a_sin_array[i])
        if conj_array[i]:
            ans *= np.conjugate(braid_value)
        else:
            ans *= braid_value
    return ans


def general_braid_function(mesh_3D, braid_func, modes_cutoff, plot, power_cos, power_sin, width, angle_array,
                           theta_array=None, rotations=None, shifts=None, moments={'p': (0, 6), 'l': (-6, 6)}):
    """
    General function for calculating braid field and Laguerre-Gaussian spectrum.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.
    power_cos : int
        Power for cosine components in the braid.
    power_sin : int
        Power for sine components in the braid.
    width : float
        Width for the Laguerre-Gaussian mode.
    angle_array : list of floats
        Starting angles for each braid in radians.
    theta_array : list of floats, optional
        Phases for each braid. Default is None.
    rotations : list of tuples, optional
        Rotation angles for each braid in degrees. Default is None (no rotations).
    shifts : list of tuples, optional
        Shifts to apply along x, y, z axes for each braid. Default is None (no shifts).
    moments : dict, optional
        Range for radial ('p') and azimuthal ('l') quantum numbers. Default is {'p': (0, 6), 'l': (-6, 6)}.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    # Setting default values for rotations, shifts, and theta if not provided
    if rotations is None:
        rotations = [(0, 0, 0)] * len(angle_array)
    if shifts is None:
        shifts = [(0, 0, 0)] * len(angle_array)
    if theta_array is None:
        theta_array = [0] * len(angle_array)

    # Rotate and shift mesh grids based on the given rotations and shifts
    mesh_3D_list = [rotate_meshgrid(*mesh_3D, *map(np.radians, rotation)) for rotation in rotations]
    mesh_3D_list = [(x + shift[0], y + shift[1], z + shift[2]) for (x, y, z), shift in zip(mesh_3D_list, shifts)]

    # Prepare parameters for braid calculation
    xyz_array = [(grid[0], grid[1], grid[2]) for grid in mesh_3D_list]
    pow_cos_array = [power_cos] * len(angle_array)
    pow_sin_array = [power_sin] * len(angle_array)
    a_cos_array = [1] * len(angle_array)
    a_sin_array = [1] * len(angle_array)
    conj_array = [0] * len(angle_array)

    # Calculate the braid product using the helper function
    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])

    # Multiply by the radial factor
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** (power_cos + power_sin)

    # Multiply by Laguerre-Gaussian mode
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=width, k0=1, x0=0, y0=0, z0=0)

    # Calculate the spectrum and filter modes
    x_2D = mesh_3D[0][:, :, 0]
    y_2D = mesh_3D[1][:, :, 0]
    values = cbs.LG_spectrum(ans[:, :, mesh_3D[2].shape[2] // 2], p=moments['p'], l=moments['l'],
                             mesh=(x_2D, y_2D), plot=plot, width=width, k0=1)

    l_save = []
    p_save = []
    weight_save = []
    moment0 = moments['l'][0]
    for l, p_array in enumerate(values):
        for p, value in enumerate(p_array):
            if abs(value) > modes_cutoff * abs(values).max():
                l_save.append(l + moment0)
                p_save.append(p)
                weight_save.append(value)

    weight_save = np.array(weight_save)
    weight_save /= np.sqrt(np.sum(weight_save ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save.tolist()}

    return weights_important


def hopf_standard_16(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Calculate braid product using standard parameters for hopf_standard_16.

    Parameters:
    - mesh_3D (tuple of np.array): 3D mesh grids (X, Y, Z).
    - braid_func (function): Function to compute the braid at each grid (default: braid).
    - modes_cutoff (float): Cutoff value to filter out insignificant modes (default: 0.01).
    - plot (bool): Whether to plot the resulting field (default: False).

    Returns:
    - dict: Filtered modes with 'l', 'p', and 'weight' representing the modes and their respective weights.
    """
    return general_braid_function(mesh_3D, braid_func, modes_cutoff, plot,
                                  power_cos=1, power_sin=1, width=1.6, angle_array=[0, np.pi])


def hopf_standard_14(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Calculate braid product using standard parameters for hopf_standard_14.

    Parameters:
    - mesh_3D (tuple of np.array): 3D mesh grids (X, Y, Z).
    - braid_func (function): Function to compute the braid at each grid (default: braid).
    - modes_cutoff (float): Cutoff value to filter out insignificant modes (default: 0.01).
    - plot (bool): Whether to plot the resulting field (default: False).

    Returns:
    - dict: Filtered modes with 'l', 'p', and 'weight' representing the modes and their respective weights.
    """
    return general_braid_function(mesh_3D, braid_func, modes_cutoff, plot,
                                  power_cos=1, power_sin=1, width=1.4,
                                  angle_array=[0, np.pi], rotations=None)


def hopf_standard_18(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Standard Hopf field with specific parameters.

    Parameters:
    mesh_3D (tuple): A tuple of 3D mesh grids (x, y, z).
    braid_func (function, optional): The braid function to be applied. Defaults to braid.
    modes_cutoff (float, optional): Threshold for mode cutoff analysis. Defaults to 0.01.
    plot (bool, optional): Whether to plot the final field or not. Defaults to False.

    Returns:
    dict: A dictionary containing important mode weights and their indices.
    """
    return general_braid_function(mesh_3D, braid_func, modes_cutoff, plot,
                                  power_cos=1, power_sin=1, width=1.8,
                                  angle_array=[0, np.pi], rotations=None)


def hopf_30both(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Hopf field with rotations of ±30 degrees along the x-axis.

    Parameters:
    mesh_3D (tuple): A tuple of 3D mesh grids (x, y, z).
    braid_func (function, optional): The braid function to be applied. Defaults to braid.
    modes_cutoff (float, optional): Threshold for mode cutoff analysis. Defaults to 0.01.
    plot (bool, optional): Whether to plot the final field or not. Defaults to False.

    Returns:
    dict: A dictionary containing important mode weights and their indices.
    """
    return general_braid_function(mesh_3D, braid_func, modes_cutoff, plot,
                                  power_cos=1, power_sin=1, width=1.6,
                                  angle_array=[0, np.pi], rotations=[(np.radians(30), 0, 0), (np.radians(-30), 0, 0)])


def hopf_30oneZ(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Hopf field with a 30-degree rotation along the z-axis for one braid, and shifts along the z-axis for both braids.

    Parameters:
    mesh_3D (tuple): A tuple of 3D mesh grids (x, y, z).
    braid_func (function, optional): The braid function to be applied. Defaults to braid.
    modes_cutoff (float, optional): Threshold for mode cutoff analysis. Defaults to 0.01.
    plot (bool, optional): Whether to plot the final field or not. Defaults to False.

    Returns:
    dict: A dictionary containing important mode weights and their indices.
    """
    return general_braid_function(mesh_3D, braid_func, modes_cutoff, plot,
                                  power_cos=1, power_sin=1, width=1.6,
                                  angle_array=[0, np.pi], rotations=[(0, 0, np.radians(30)), (0, 0, 0)],
                                  shifts=[(0, 0, -0.3), (0, 0, 0.3)])


def hopf_optimized(*args, **kwargs):
    """
    Optimized version of a Hopf braid function.

    This function provides a precomputed set of important modes for a specific Hopf braid configuration.
    It returns a dictionary containing the `l`, `p`, and `weight` values of important modes,
    where weights are normalized.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        A 3D mesh grid used for the calculation.

    braid_func : callable, optional (default=braid)
        The function to generate the braid field.

    modes_cutoff : float, optional (default=0.01)
        The cutoff value for mode significance.

    plot : bool, optional (default=False)
        Whether to plot the result or not.

    Returns:
    --------
    weights_important : dict
        A dictionary containing the important modes with:
        - 'l': list of azimuthal quantum numbers.
        - 'p': list of radial quantum numbers.
        - 'weight': list of normalized weights for each mode.
    """
    # Precomputed important modes for this optimized braid configuration
    l_save = [0, 0, 0, 2]
    p_save = [0, 1, 2, 0]
    weight_save = [2.96, -6.23, 4.75, -5.49]

    # Normalize the weights to sum to 100% in magnitude
    weight_save = np.array(weight_save)
    weight_save /= np.sqrt(np.sum(weight_save ** 2)) * 100

    # Package the important mode data in a dictionary
    weights_important = {
        'l': l_save,
        'p': p_save,
        'weight': weight_save.tolist()
    }

    return weights_important


# Specific implementations of the general braid function
def hopf_pm_03_z(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Function for the Hopf braid configuration with ±0.3 shifts along the z-axis.
    """
    return general_braid_function(
        mesh_3D, braid_func, modes_cutoff, plot,
        power_cos=1, power_sin=1, width=1.6,
        angle_array=[0, np.pi],
        shifts=[(0, 0, -0.3), (0, 0, 0.3)]
    )


def hopf_4foil(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Function for the four-foil Hopf braid configuration.
    """
    return general_braid_function(
        mesh_3D, braid_func, modes_cutoff, plot,
        power_cos=2, power_sin=2, width=1.0,
        angle_array=[0, np.pi]
    )


def hopf_6foil(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Function for the six-foil Hopf braid configuration.
    """
    return general_braid_function(
        mesh_3D, braid_func, modes_cutoff, plot,
        power_cos=3, power_sin=3, width=0.75,
        angle_array=[0, np.pi],
        moments={'p': (0, 7), 'l': (-7, 7)}
    )


def hopf_30oneX(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Function for the Hopf braid configuration with 20 degrees rotation along the x-axis.
    """
    return general_braid_function(
        mesh_3D, braid_func, modes_cutoff, plot,
        power_cos=1, power_sin=1, width=1.6,
        angle_array=[0, np.pi],
        rotations=[(20, 0, 0), (0, 0, 0)]
    )


def hopf_15oneZ(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Function for the Hopf braid configuration with 15 degrees rotation along the z-axis.
    """
    return general_braid_function(
        mesh_3D, braid_func, modes_cutoff, plot,
        power_cos=1, power_sin=1, width=1.6,
        angle_array=[0, np.pi],
        rotations=[(0, 0, 15), (0, 0, 0)],
        shifts=[(0, 0, -0.3), (0, 0, 0.3)]
    )


def hopf_dennis(*args, **kwargs):
    """
    Function for the Dennis Hopf braid configuration with precomputed weights.
    """
    l_save = [0, 0, 0, 2]
    p_save = [0, 1, 2, 0]
    weight_save = [2.63, -6.32, 4.21, -5.95]
    weight_save = np.array(weight_save)
    weight_save /= np.sqrt(np.sum(weight_save ** 2)) * 100
    return {'l': l_save, 'p': p_save, 'weight': weight_save.tolist()}



def lobe_modify(mesh, angle_ranges, scale_factor):
    """
    Modify specific lobes of the mesh by either removing or scaling.

    Parameters:
    -----------
    mesh : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    angle_ranges : list of tuples
        List of angle ranges (angle1, angle2) for phase modification.
    scale_factor : float
        Factor by which to scale the mesh. Set to 0 to remove the lobe.

    Returns:
    --------
    mesh2 : tuple of ndarrays
        Modified mesh with the specified lobe either removed or scaled.
    """
    # Create a copy of the mesh to modify
    mesh2 = list(np.copy(mesh))

    # Apply modifications for each angle range
    for angle1, angle2 in angle_ranges:
        phase = np.angle(mesh[0] + 1j * mesh[1])
        phase_mask = (phase > angle1) & (phase <= angle2)

        # Apply scaling or removal based on scale_factor
        if scale_factor == 0:
            # Removing the lobe by setting values to zero
            for i in range(3):
                mesh2[i][phase_mask] = 0
        else:
            # Scaling the selected lobe
            mesh2[0][phase_mask] *= scale_factor
            mesh2[1][phase_mask] *= scale_factor

    return tuple(mesh2)



def lobe_remove(mesh, angle1, angle2, rot_x, rot_y, rot_z):
    # helper function to remove the lobe/petal in Flower beams.
    # it's defined by mesh2[0][phase_mask] = mesh2_flat[0] * 100 here. more than 50 is enough
    A, B = angle1, angle2
    phase = np.angle(mesh[0] + 1j * mesh[1])
    phase_mask = (phase > A) & (phase <= B)

    mesh2_flat = rotate_meshgrid(
        mesh[0][phase_mask],
        mesh[1][phase_mask],
        mesh[2][phase_mask],
        np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)
    )
    mesh2 = np.copy(mesh)
    mesh2[0][phase_mask] = mesh2_flat[0] * 100
    mesh2[1][phase_mask] = mesh2_flat[1] * 100
    mesh2[2][phase_mask] = mesh2_flat[2]
    return mesh2


def lobe_smaller(mesh, angle1, angle2, rot_x, rot_y, rot_z):
    # helper function to make the lobe/petal in Flower beams smaller.
    # the size is 1.5 rn, it's defined in  mesh2[0][phase_mask] = mesh2_flat[0] * 1.5
    A, B = angle1, angle2
    phase = np.angle(mesh[0] + 1j * mesh[1])
    phase_mask = (phase > A) & (phase <= B)

    mesh2_flat = rotate_meshgrid(
        mesh[0][phase_mask],
        mesh[1][phase_mask],
        mesh[2][phase_mask],
        np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)
    )
    mesh2 = np.copy(mesh)
    mesh2[0][phase_mask] = mesh2_flat[0] * 1.5
    mesh2[1][phase_mask] = mesh2_flat[1] * 1.5
    mesh2[2][phase_mask] = mesh2_flat[2]
    return mesh2

def unknot_general(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False, power=4, lobe_sizes=(2, 2, 2, 2),
                   num_lobes=4):
    """
    General function to create unknotted braids with specific lobe modifications.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.
    power : int
        Power for cosine and sine components in the braid.
    lobe_sizes : tuple of int
        A tuple specifying modifications to each lobe:
        - 2: Leave untouched
        - 1: Scale down (factor of 1.5)
        - 0: Remove
    num_lobes : int
        Number of lobes (e.g., 4 or 6) to specify the type of configuration.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    # Initial mesh rotation (if needed)
    mesh_3D_new = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(0), np.radians(0))

    # Define angle ranges for each lobe based on the number of lobes
    if num_lobes == 4:
        lobe_definitions = {
            0: [(-np.pi / 4, np.pi / 4)],  # Lobe 0
            1: [(np.pi / 4, 3 * np.pi / 4)],  # Lobe 1
            2: [(-4 * np.pi / 4, -3 * np.pi / 4), (3 * np.pi / 4, 4 * np.pi / 4)],  # Lobe 2 (two intervals)
            3: [(-3 * np.pi / 4, -np.pi / 4)]  # Lobe 3
        }
    elif num_lobes == 6:
        lobe_definitions = {
            0: [(-np.pi / 6, np.pi / 6)],  # Lobe 0
            1: [(np.pi / 6, np.pi / 2)],  # Lobe 1
            2: [(np.pi / 2, 5 * np.pi / 6)],  # Lobe 2
            3: [(5 * np.pi / 6, 7 * np.pi / 6)],  # Lobe 3
            4: [(7 * np.pi / 6, 3 * np.pi / 2)],  # Lobe 4
            5: [(3 * np.pi / 2, 11 * np.pi / 6)]  # Lobe 5
        }
    else:
        raise ValueError("Unsupported number of lobes. Only 4 or 6 lobes are supported.")

    # Apply modifications based on lobe_sizes
    for lobe_idx, size in enumerate(lobe_sizes):
        if size == 2:  # Leave lobe untouched
            continue
        elif size == 1:  # Scale down
            mesh_3D_new = lobe_modify(mesh_3D_new, lobe_definitions[lobe_idx], scale_factor=1.5)
        elif size == 0:  # Remove
            mesh_3D_new = lobe_modify(mesh_3D_new, lobe_definitions[lobe_idx], scale_factor=100)

    # Setup the braid parameters
    angle_array = [0]
    pow_cos_array = [power]
    pow_sin_array = [power]
    theta_array = [0.0]
    a_cos_array = [1]
    a_sin_array = [1]
    conj_array = [0]

    # Compute the braid field
    ans = 1
    for i, xyz in enumerate([(mesh_3D_new[0], mesh_3D_new[1], mesh_3D_new[2])]):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])

    # Multiply by the radial factor
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** power

    # Set the appropriate width for the Laguerre-Gaussian mode
    ws = {
        0: 3, 1: 2.6, 2: 2.6 ** (1 / 2), 3: 1.2,
        4: 0.9, 5: 0.75, 6: 0.65
    }
    w = ws[power]
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    # Calculate the Laguerre-Gaussian spectrum and find significant modes
    moments = {'p': (0, 10), 'l': (-10, 10)}
    _, _, res_z_3D = np.shape(mesh_3D_new[0])
    x_2D = mesh_3D[0][:, :, 0]
    y_2D = mesh_3D[1][:, :, 0]

    if plot:
        plot_field_both(ans[:, :, res_z_3D // 2])

    values = cbs.LG_spectrum(
        ans[:, :, res_z_3D // 2], **moments, mesh=(x_2D, y_2D), plot=True, width=w, k0=1
    )

    # Extract significant modes
    l_save, p_save, weight_save = [], [], []
    moment0 = moments['l'][0]

    for l, p_array in enumerate(values):
        for p, value in enumerate(p_array):
            if abs(value) > modes_cutoff * abs(values).max():
                l_save.append(l + moment0)
                p_save.append(p)
                weight_save.append(value)

    # Normalize the weights
    weight_save = np.array(weight_save)
    weight_save /= np.sqrt(np.sum(weight_save ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save.tolist()}

    return weights_important


# Example usage of `unknot_general` with specific lobe modifications for different lobe configurations

def unknot_4(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False, lobe_sizes=(2, 2, 2, 2)):
    """
    Function to create a 4-fold unknot with specific lobe modifications.
    """
    return unknot_general(mesh_3D, braid_func, modes_cutoff, plot, power=4, lobe_sizes=lobe_sizes, num_lobes=4)


def unknot_6(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False, lobe_sizes=(2, 2, 2, 2, 2, 2)):
    """
    Function to create a 6-fold unknot with specific lobe modifications.
    """
    return unknot_general(mesh_3D, braid_func, modes_cutoff, plot, power=6, lobe_sizes=lobe_sizes, num_lobes=6)


def unknot_4_any(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False,
                 angle_size=(2, 2, 2, 2), cmap='jet'):
    # angle_size=((1, 0), (3, 1), (4, 0))):
    mesh_3D_new = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    for angle, size in enumerate(angle_size):
        angles_dict = {
            0: [(-np.pi / 4, np.pi / 4)],
            1: [(np.pi / 4, 3 * np.pi / 4)],
            2: [(-4 * np.pi / 4, -3 * np.pi / 4), (3 * np.pi / 4, 4 * np.pi / 4)],
            3: [(-3 * np.pi / 4, -np.pi / 4)]
        }
        for ang in angles_dict[angle]:
            if size == 2:
                continue
            if size == 1:
                mesh_3D_new = lobe_smaller(mesh_3D_new, ang[0], ang[1], rot_x=0, rot_y=0, rot_z=0)
            elif size == 0:
                mesh_3D_new = lobe_remove(mesh_3D_new, ang[0], ang[1], rot_x=0, rot_y=0, rot_z=0)
            else:
                print(f"Invalid size {size} for angle {angle}")

    # angles = [1, 2, 3, 4]
    # sizes = [1, 2]
    #
    # A, B = -np.pi / 4, np.pi / 4
    # mesh_3D_new = lobe_smaller(mesh_3D_new, A, B, rot_x=0, rot_y=0, rot_z=0)
    # A, B = -4 * np.pi / 4, -3 * np.pi / 4
    # mesh_3D_new = lobe_remove(mesh_3D_new, A, B, rot_x=0, rot_y=0, rot_z=0)
    # A, B = 3 * np.pi / 4, 4 * np.pi / 4
    # mesh_3D_new = lobe_remove(mesh_3D_new, A, B, rot_x=0, rot_y=0, rot_z=0)
    # A, B = -3 * np.pi / 6, -1 * np.pi / 6
    # mesh_3D_new2 = rotate_part(mesh_3D_new2, A, B, rot_x=0, rot_y=0, rot_z=0)

    xyz_array = [
        (mesh_3D_new[0], mesh_3D_new[1], mesh_3D_new[2]),
    ]
    # starting angle for each braid
    angle_array = np.array([0])
    # powers in cos in sin
    power = 4
    pow_cos_array = [power]
    pow_sin_array = [power]
    # conjugating the braid (in "Milnor" space)
    conj_array = [0]
    # moving x+iy (same as in the paper)
    theta_array = [0.0 * np.pi, 0 * np.pi]
    # braid scaling
    a_cos_array = [1]
    a_sin_array = [1]

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** power
    ws = {
        0: 3,
        1: 2.6,
        # 2: 1.6,
        2: 2.6 ** (1 / 2),
        3: 1.2,
        4: 0.85,
        5: 0.75,
        6: 0.65,
    }
    w = ws[power]
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    moments = {'p': (0, 10), 'l': (-10, 10)}

    _, _, res_z_3D = np.shape(mesh_3D_new[0])
    x_2D = mesh_3D[0][:, :, 0]
    y_2D = mesh_3D[1][:, :, 0]
    if plot:
        plot_field_both(ans[:, :, res_z_3D // 2])
    values = cbs.LG_spectrum(
        ans[:, :, res_z_3D // 2], **moments, mesh=(x_2D, y_2D), plot=plot, width=w, k0=1, cmap=cmap
    )
    l_save = []
    p_save = []
    weight_save = []
    moment0 = moments['l'][0]
    for l, p_array in enumerate(values):
        for p, value in enumerate(p_array):
            if abs(value) > modes_cutoff * abs(values).max():
                l_save.append(l + moment0)
                p_save.append(p)
                weight_save.append(value)
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important



def unknot_5_any(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False,
                 angle_size=(2, 2, 2, 2, 2), cmap='jet'):
    """
    Create a 5-lobe structure from the input 3D mesh.

    Parameters:
      mesh_3D      : tuple of arrays representing the 3D meshgrid.
      braid_func   : function for the braid operation.
      modes_cutoff : threshold for selecting mode weights.
      plot         : whether to produce plots.
      angle_size   : a 5-element tuple controlling modifications in each lobe.
                     For each element:
                        2 = do nothing,
                        1 = lobe_smaller,
                        0 = lobe_remove.
      cmap         : colormap to use for plotting.

    Returns:
      weights_important : dictionary with keys 'l', 'p', and 'weight' describing the modes.
    """
    # Optionally rotate the original mesh (here with zero rotations)
    mesh_3D_new = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(0), np.radians(0))

    # Partition the full circle into 5 equal sectors.
    # Here we divide the interval [-pi, pi] into 5 equal parts.
    # The boundaries are: -pi, -pi + 2π/5, -pi + 4π/5, -pi + 6π/5, -pi + 8π/5, π.
    angles_dict = {
        0: [(-np.pi, -np.pi + 2 * np.pi / 5)],               # Sector 0: [-π, -π+0.4π]
        1: [(-np.pi + 2 * np.pi / 5, -np.pi + 4 * np.pi / 5)],   # Sector 1: [-π+0.4π, -π+0.8π]
        2: [(-np.pi + 4 * np.pi / 5, -np.pi + 6 * np.pi / 5)],   # Sector 2: [-π+0.8π, -π+1.2π] => centered at 0
        3: [(-np.pi + 6 * np.pi / 5, -np.pi + 8 * np.pi / 5)],   # Sector 3: [-π+1.2π, -π+1.6π]
        4: [(-np.pi + 8 * np.pi / 5, np.pi)]                   # Sector 4: [-π+1.6π, π]
    }

    # Modify the mesh in each sector according to the corresponding entry in angle_size.
    for angle, size in enumerate(angle_size):
        for ang in angles_dict[angle]:
            if size == 2:
                continue
            elif size == 1:
                mesh_3D_new = lobe_smaller(mesh_3D_new, ang[0], ang[1],
                                           rot_x=0, rot_y=0, rot_z=0)
            elif size == 0:
                mesh_3D_new = lobe_remove(mesh_3D_new, ang[0], ang[1],
                                          rot_x=0, rot_y=0, rot_z=0)
            else:
                print(f"Invalid size {size} for angle index {angle}")

    # Build the field using the braid function and additional factors.
    xyz_array = [
        (mesh_3D_new[0], mesh_3D_new[1], mesh_3D_new[2]),
    ]
    # Starting angle for each braid
    angle_array = np.array([0])
    # Set the power (which now matches the 5-lobe structure)
    power = 5
    pow_cos_array = [power]
    pow_sin_array = [power]
    # Flag to indicate conjugation
    conj_array = [0]
    # Additional phase shifts (as in your paper)
    theta_array = [0.0 * np.pi, 0 * np.pi]
    # Braid scaling factors
    a_cos_array = [1]
    a_sin_array = [1]

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i],
                                           pow_cos_array[i], pow_sin_array[i],
                                           theta_array[i], a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i],
                              pow_cos_array[i], pow_sin_array[i],
                              theta_array[i], a_cos_array[i], a_sin_array[i])

    # Multiply by a radial scaling factor
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** power

    # Use a Laguerre–Gaussian (LG) envelope (adjust parameters as needed)
    ws = {
        0: 3,
        1: 2.6,
        2: 2.6 ** (1 / 2),
        3: 1.2,
        4: 0.85,
        5: 0.75,
        6: 0.65,
    }
    w = ws[power]
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    # Set the moments for the spectrum calculation.
    moments = {'p': (0, 10), 'l': (-10, 10)}
    _, _, res_z_3D = np.shape(mesh_3D_new[0])
    x_2D = mesh_3D[0][:, :, 0]
    y_2D = mesh_3D[1][:, :, 0]

    if plot:
        plot_field_both(ans[:, :, res_z_3D // 2])

    # Calculate the LG spectrum
    values = cbs.LG_spectrum(ans[:, :, res_z_3D // 2],
                             **moments, mesh=(x_2D, y_2D),
                             plot=plot, width=w, k0=1, cmap=cmap)

    # Extract significant mode components.
    l_save = []
    p_save = []
    weight_save = []
    moment0 = moments['l'][0]
    for l, p_array in enumerate(values):
        for p, value in enumerate(p_array):
            if abs(value) > modes_cutoff * abs(values).max():
                l_save.append(l + moment0)
                p_save.append(p)
                weight_save.append(value)
    weight_save = np.array(weight_save) / (np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100)
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}

    return weights_important


def unknot_6_any(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False,
                 angle_size=(2, 2, 2, 2, 2, 2), cmap='jet'):
    """
    Create a 6-lobe structure from the input 3D mesh.

    Parameters:
      mesh_3D      : tuple of arrays representing the 3D meshgrid.
      braid_func   : function for the braid operation.
      modes_cutoff : threshold for selecting mode weights.
      plot         : whether to produce plots.
      angle_size   : a 6-element tuple controlling modifications in each lobe.
                     For each element:
                        2 = do nothing,
                        1 = lobe_smaller,
                        0 = lobe_remove.
      cmap         : colormap to use for plotting.

    Returns:
      weights_important : dictionary with keys 'l', 'p', and 'weight' describing the modes.
    """

    # Optionally rotate the original mesh (here with zero rotations)
    mesh_3D_new = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(0), np.radians(0))

    # Define the angular sectors for the 6 lobes.
    angles_dict = {
        0: [(-np.pi / 6, np.pi / 6)],  # Lobe centered at 0
        1: [(np.pi / 6, np.pi / 2)],  # Lobe centered at π/3
        2: [(np.pi / 2, 5 * np.pi / 6)],  # Lobe centered at 2π/3
        3: [(5 * np.pi / 6, np.pi), (-np.pi, -5 * np.pi / 6)],  # Lobe centered at π (split due to periodicity)
        4: [(-5 * np.pi / 6, -np.pi / 2)],  # Lobe centered at -2π/3
        5: [(-np.pi / 2, -np.pi / 6)]  # Lobe centered at -π/3
    }

    # Modify the mesh in each lobe according to the corresponding entry in angle_size.
    for angle, size in enumerate(angle_size):
        for ang in angles_dict[angle]:
            if size == 2:
                continue
            elif size == 1:
                mesh_3D_new = lobe_smaller(mesh_3D_new, ang[0], ang[1],
                                           rot_x=0, rot_y=0, rot_z=0)
            elif size == 0:
                mesh_3D_new = lobe_remove(mesh_3D_new, ang[0], ang[1],
                                          rot_x=0, rot_y=0, rot_z=0)
            else:
                print(f"Invalid size {size} for angle index {angle}")

    # Build the field using the braid function and additional factors.
    xyz_array = [
        (mesh_3D_new[0], mesh_3D_new[1], mesh_3D_new[2]),
    ]
    # Starting angle for each braid
    angle_array = np.array([0])
    # Powers in cosine and sine terms
    power = 6
    pow_cos_array = [power]
    pow_sin_array = [power]
    # Flag to indicate conjugation
    conj_array = [0]
    # Additional phase shifts (as in your paper)
    theta_array = [0.0 * np.pi, 0 * np.pi]
    # Braid scaling factors
    a_cos_array = [1]
    a_sin_array = [1]

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i],
                                           pow_cos_array[i], pow_sin_array[i],
                                           theta_array[i], a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i],
                              pow_cos_array[i], pow_sin_array[i],
                              theta_array[i], a_cos_array[i], a_sin_array[i])

    # Multiply by a radial scaling factor
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** power

    # Use a Laguerre–Gaussian (LG) envelope (adjust parameters as needed)
    ws = {
        0: 3,
        1: 2.6,
        2: 2.6 ** (1 / 2),
        3: 1.2,
        4: 0.85,
        5: 0.75,
        6: 0.65,
    }
    w = ws[power]
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    # Set the moments for the spectrum calculation.
    moments = {'p': (0, 10), 'l': (-10, 10)}
    _, _, res_z_3D = np.shape(mesh_3D_new[0])
    x_2D = mesh_3D[0][:, :, 0]
    y_2D = mesh_3D[1][:, :, 0]

    if plot:
        plot_field_both(ans[:, :, res_z_3D // 2])

    # Calculate the LG spectrum
    values = cbs.LG_spectrum(ans[:, :, res_z_3D // 2],
                             **moments, mesh=(x_2D, y_2D),
                             plot=plot, width=w, k0=1, cmap=cmap)

    # Extract significant mode components.
    l_save = []
    p_save = []
    weight_save = []
    moment0 = moments['l'][0]
    for l, p_array in enumerate(values):
        for p, value in enumerate(p_array):
            if abs(value) > modes_cutoff * abs(values).max():
                l_save.append(l + moment0)
                p_save.append(p)
                weight_save.append(value)
    weight_save = np.array(weight_save) / (np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100)
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}

    return weights_important


#
# def unknot_4_any(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False,
#                  angle_size=(2, 2, 2, 2), cmap='jet'):
# 	# angle_size=((1, 0), (3, 1), (4, 0))):
# 	mesh_3D_new = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
# 	for angle, size in enumerate(angle_size):
# 		angles_dict = {
# 			0: [(-np.pi / 4, np.pi / 4)],
# 			1: [(np.pi / 4, 3 * np.pi / 4)],
# 			2: [(-4 * np.pi / 4, -3 * np.pi / 4), (3 * np.pi / 4, 4 * np.pi / 4)],
# 			3: [(-3 * np.pi / 4, -np.pi / 4)]
# 		}
# 		for ang in angles_dict[angle]:
# 			if size == 2:
# 				continue
# 			if size == 1:
# 				mesh_3D_new = lobe_smaller(mesh_3D_new, ang[0], ang[1], rot_x=0, rot_y=0, rot_z=0)
# 			elif size == 0:
# 				mesh_3D_new = lobe_remove(mesh_3D_new, ang[0], ang[1], rot_x=0, rot_y=0, rot_z=0)
# 			else:
# 				print(f"Invalid size {size} for angle {angle}")
#
# 	# angles = [1, 2, 3, 4]
# 	# sizes = [1, 2]
# 	#
# 	# A, B = -np.pi / 4, np.pi / 4
# 	# mesh_3D_new = lobe_smaller(mesh_3D_new, A, B, rot_x=0, rot_y=0, rot_z=0)
# 	# A, B = -4 * np.pi / 4, -3 * np.pi / 4
# 	# mesh_3D_new = lobe_remove(mesh_3D_new, A, B, rot_x=0, rot_y=0, rot_z=0)
# 	# A, B = 3 * np.pi / 4, 4 * np.pi / 4
# 	# mesh_3D_new = lobe_remove(mesh_3D_new, A, B, rot_x=0, rot_y=0, rot_z=0)
# 	# A, B = -3 * np.pi / 6, -1 * np.pi / 6
# 	# mesh_3D_new2 = rotate_part(mesh_3D_new2, A, B, rot_x=0, rot_y=0, rot_z=0)
#
# 	xyz_array = [
# 		(mesh_3D_new[0], mesh_3D_new[1], mesh_3D_new[2]),
# 	]
# 	# starting angle for each braid
# 	angle_array = np.array([0])
# 	# powers in cos in sin
# 	power = 4
# 	pow_cos_array = [power]
# 	pow_sin_array = [power]
# 	# conjugating the braid (in "Milnor" space)
# 	conj_array = [0]
# 	# moving x+iy (same as in the paper)
# 	theta_array = [0.0 * np.pi, 0 * np.pi]
# 	# braid scaling
# 	a_cos_array = [1]
# 	a_sin_array = [1]
#
# 	ans = 1
# 	for i, xyz in enumerate(xyz_array):
# 		if conj_array[i]:
# 			ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
# 			                               a_cos_array[i], a_sin_array[i]))
# 		else:
# 			ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
# 			                  a_cos_array[i], a_sin_array[i])
# 	R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
# 	ans *= (1 + R ** 2) ** power
# 	ws = {
# 		0: 3,
# 		1: 2.6,
# 		# 2: 1.6,
# 		2: 2.6 ** (1 / 2),
# 		3: 1.2,
# 		4: 0.85,
# 		5: 0.75,
# 		6: 0.65,
# 	}
# 	w = ws[power]
# 	ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)
#
# 	moments = {'p': (0, 10), 'l': (-10, 10)}
#
# 	_, _, res_z_3D = np.shape(mesh_3D_new[0])
# 	x_2D = mesh_3D[0][:, :, 0]
# 	y_2D = mesh_3D[1][:, :, 0]
# 	if plot:
# 		plot_field_both(ans[:, :, res_z_3D // 2])
# 	values = cbs.LG_spectrum(
# 		ans[:, :, res_z_3D // 2], **moments, mesh=(x_2D, y_2D), plot=True, width=w, k0=1, cmap=cmap
# 	)
# 	l_save = []
# 	p_save = []
# 	weight_save = []
# 	moment0 = moments['l'][0]
# 	for l, p_array in enumerate(values):
# 		for p, value in enumerate(p_array):
# 			if abs(value) > modes_cutoff * abs(values).max():
# 				l_save.append(l + moment0)
# 				p_save.append(p)
# 				weight_save.append(value)
# 	weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
# 	weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
# 	return weights_important

def borromean(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Function for generating the Borromean rings braid field.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    rotations = [(0, 0, 0), (0, 0, 30), (0, 0, 0)]
    shifts = [(0, 0, 0)] * 3
    angle_array = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    width = 1.4
    power_cos = power_sin = 1

    return general_braid_function(mesh_3D, braid_func, modes_cutoff, plot, power_cos, power_sin, width, angle_array,
                                  rotations=rotations, shifts=shifts)


def loops_x2(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Function for generating a braid field representing two loops.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    rotations = [(-40, 0, 0), (40, 0, 0), (0, 0, 0)]
    shifts = [(0.3, 0, 0), (-0.3, 0, 0), (0, 0, 0)]
    angle_array = [0, -np.pi, np.pi / 2]
    width = 1.5
    power_cos = power_sin = 1

    return general_braid_function(mesh_3D, braid_func, modes_cutoff, plot, power_cos, power_sin, width, angle_array,
                                  rotations=rotations, shifts=shifts)


def whitehead(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Function for generating a braid field representing the Whitehead link.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    rotations = [(0, 0, 0), (0, 0, 0), (0, -45, 0)]
    shifts = [(0, 0, 0), (0, 0, 0), (0, -0.5, 0)]
    angle_array = [0, 2 * np.pi / 2, np.pi / 2]
    width = 1.3
    power_cos = power_sin = 1

    return general_braid_function(mesh_3D, braid_func, modes_cutoff, plot, power_cos, power_sin, width, angle_array,
                                  rotations=rotations, shifts=shifts)


def hopf_new0(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Function for generating a braid field for the Hopf link.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    rotations = [(0, 0, 0), (0, 0, 0)]
    shifts = [(0, 0, 0)] * 2
    angle_array = [0, np.pi]
    width = 1.8
    power_cos = power_sin = 1

    return general_braid_function(mesh_3D, braid_func, modes_cutoff, plot, power_cos, power_sin, width, angle_array,
                                  rotations=rotations, shifts=shifts)


def trefoil_dennis(*args, **kwargs):
    """
    Function representing the Dennis version of the Trefoil braid field.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    l_save = [0, 0, 0, 0, 3]
    p_save = [0, 1, 2, 3, 0]
    weight_save = [1.51, -5.06, 7.23, -2.03, -3.97]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important


def trefoil_optimized_math_many(*args, **kwargs):
    """
    Optimized version of the Trefoil braid with multiple components, using complex coefficients.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    l_save = [-3, -3, -3, 0, 0, 0, 0, 3, 3, 3, 3]
    p_save = [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3]
    weight_save = [(0.00021092227580502785 + 7.920886118023887e-07j), (-0.0008067955939880228 + 3.607965773547045e-06j),
                   (0.00012245355229622952 + 1.0489244513922499e-07j),
                   (0.0011924249760862221 + 1.1372720865198612e-05j), (-0.002822503524696713 + 8.535015090975148e-06j),
                   (0.0074027513552253985 + 5.475152609562587e-06j), (-0.0037869189890120283 + 8.990311302510444e-06j),
                   (-0.0043335243263204586 + 8.720849197446247e-07j),
                   (0.00013486021497321202 + 3.7649803780475746e-06j),
                   (-0.0002255962470142481 + 3.4852313165379194e-07j),
                   (-9.910118914013346e-05 + 1.5151969296215156e-08j)]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important


def trefoil_optimized_math_many_095(*args, **kwargs):
    """
    Optimized version of the Trefoil braid with a different set of complex coefficients.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    l_save = [-3, -3, -3, 0, 0, 0, 0, 3, 3, 3, 3]
    p_save = [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3]
    weight_save = [(0.0003480344230302087 - 2.24808914259125e-08j), (-0.0008202669878716706 + 3.440862545658389e-06j),
                   (0.00010265094423308904 + 2.31922738071223e-07j), (0.0011967299943849222 + 7.739735031831317e-06j),
                   (-0.0038439481283424996 + 6.262024157567451e-06j), (0.007992793393037656 + 2.6957311985005077e-06j),
                   (-0.0038705109086829936 + 6.2293722796143055e-06j),
                   (-0.0038065491189479697 + 2.3047912587806214e-08j), (0.00020229977021637684 + 3.5303977611434e-06j),
                   (-0.00019990398151559197 + 3.711187821973153e-07j),
                   (-8.519625649010964e-05 + 1.3228967779423537e-08j)]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important


def trefoil_optimized_math_5(*args, **kwargs):
    """
    Optimized version of the Trefoil braid with fewer components and real coefficients.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    l_save = [0, 0, 0, 0, 3]
    p_save = [0, 1, 2, 3, 0]
    weight_save = [1.20, -2.85, 7.48, -3.83, -4.38]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important


def trefoil_standard_XX(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False, width=1.5):
    """
    Calculates the braid field for a specific trefoil configuration and computes the Laguerre-Gaussian spectrum.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z) representing the spatial coordinates.
    braid_func : callable, optional
        The function used to generate the braid field. Default is `braid`.
    modes_cutoff : float, optional
        Cutoff value to determine the significance of calculated modes. Only modes above this cutoff are retained.
        Default is 0.01.
    plot : bool, optional
        Whether to plot the resulting field and spectrum. Default is False.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes:
        - 'l': List of azimuthal quantum numbers.
        - 'p': List of radial quantum numbers.
        - 'weight': List of relative weights for each mode.

    Notes:
    ------
    - The parameter `XX` represents the width parameter used in the general braid calculation.
    - Each function calls the `general_braid_function` with specific parameters for `width` and braid characteristics.
    """
    # Parameters for general braid function
    power_cos = 1.5
    power_sin = 1.5
    angle_array = [0, np.pi]  # Starting angles for each braid in radians
    rotations = [(0, 0, 0), (0, 0, 0)]  # No rotation
    shifts = [(0, 0, 0), (0, 0, 0)]  # No shift

    # Call the general_braid_function to calculate the braid field and spectrum
    weights_important = general_braid_function(
        mesh_3D=mesh_3D,
        braid_func=braid_func,
        modes_cutoff=modes_cutoff,
        plot=plot,
        power_cos=power_cos,
        power_sin=power_sin,
        width=width,
        angle_array=angle_array,
        theta_array=[0, 0],
        rotations=rotations,
        shifts=shifts,
        moments={'p': (0, 6), 'l': (-6, 6)}
    )

    return weights_important


def trefoil_standard_15(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Trefoil configuration with Laguerre-Gaussian width of 1.5.
    """
    return general_braid_function(
        mesh_3D=mesh_3D,
        braid_func=braid_func,
        modes_cutoff=modes_cutoff,
        plot=plot,
        power_cos=1.5,
        power_sin=1.5,
        width=1.5,
        angle_array=[0, np.pi]
    )


def trefoil_standard_13(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Trefoil configuration with Laguerre-Gaussian width of 1.3.
    """
    return general_braid_function(
        mesh_3D=mesh_3D,
        braid_func=braid_func,
        modes_cutoff=modes_cutoff,
        plot=plot,
        power_cos=1.5,
        power_sin=1.5,
        width=1.3,
        angle_array=[0, np.pi]
    )


def trefoil_standard_125(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Trefoil configuration with Laguerre-Gaussian width of 1.25.
    """
    return general_braid_function(
        mesh_3D=mesh_3D,
        braid_func=braid_func,
        modes_cutoff=modes_cutoff,
        plot=plot,
        power_cos=1.5,
        power_sin=1.5,
        width=1.25,
        angle_array=[0, np.pi]
    )


def trefoil_standard_12(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Trefoil configuration with Laguerre-Gaussian width of 1.2.
    """
    return general_braid_function(
        mesh_3D=mesh_3D,
        braid_func=braid_func,
        modes_cutoff=modes_cutoff,
        plot=plot,
        power_cos=1.5,
        power_sin=1.5,
        width=1.2,
        angle_array=[0, np.pi]
    )


def trefoil_standard_11(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Trefoil configuration with Laguerre-Gaussian width of 1.1.
    """
    return general_braid_function(
        mesh_3D=mesh_3D,
        braid_func=braid_func,
        modes_cutoff=modes_cutoff,
        plot=plot,
        power_cos=1.5,
        power_sin=1.5,
        width=1.1,
        angle_array=[0, np.pi]
    )


def trefoil_standard_115(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Trefoil configuration with Laguerre-Gaussian width of 1.15.
    """
    return general_braid_function(
        mesh_3D=mesh_3D,
        braid_func=braid_func,
        modes_cutoff=modes_cutoff,
        plot=plot,
        power_cos=1.5,
        power_sin=1.5,
        width=1.15,
        angle_array=[0, np.pi]
    )


def trefoil_standard_105(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Trefoil configuration with Laguerre-Gaussian width of 1.05.
    """
    return general_braid_function(
        mesh_3D=mesh_3D,
        braid_func=braid_func,
        modes_cutoff=modes_cutoff,
        plot=plot,
        power_cos=1.5,
        power_sin=1.5,
        width=1.05,
        angle_array=[0, np.pi]
    )


def fivefoil_standard_08(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    """
    Fivefoil configuration with Laguerre-Gaussian width of 0.8.
    """
    return general_braid_function(
        mesh_3D=mesh_3D,
        braid_func=braid_func,
        modes_cutoff=modes_cutoff,
        plot=plot,
        power_cos=2.5,
        power_sin=2.5,
        width=0.8,
        angle_array=[0, np.pi]
    )


def trefoil_optimized(*args, **kwargs):
    """
    Function representing an optimized version of the Trefoil braid field.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    l_save = [0, 0, 0, 0, 3]
    p_save = [0, 1, 2, 3, 0]
    weight_save = [1.29, -3.95, 7.49, -3.28, -3.98]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important


def trefoil_optimized_new(*args, **kwargs):
    """
    Function representing a new optimized version of the Trefoil braid field.

    Parameters:
    -----------
    mesh_3D : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    braid_func : callable
        The function to generate the braid field.
    modes_cutoff : float
        Cutoff value for significant modes.
    plot : bool
        Whether to plot the resulting field or not.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    l_save = [0, 0, 0, 0, 3, 0, 0]
    p_save = [0, 1, 2, 3, 0, 4, 5]
    weight_save = [1.05, -4.05, 7.48, -3.03, -4.03, -1.19, 0.0]  # Using the last version provided in the comments
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important


def field_combination_LG(l, p, weights, values=0, mesh=0, w_real=0, k0=1, x0=0, y0=0, z0=0):
    """
    Combines LG modes based on quantum numbers and weights.

    Parameters:
    -----------
    l : list of int
        Azimuthal quantum numbers.
    p : list of int
        Radial quantum numbers.
    weights : list of float
        Weights for each mode.
    values, mesh, w_real, k0, x0, y0, z0 : optional
        Unused in this function, but kept for consistency with the API.

    Returns:
    --------
    weights_important : dict
        A dictionary containing significant modes with keys 'l', 'p', and 'weight'.
    """
    l_save = l
    p_save = p
    weight_save = weights
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important


def field_knot_from_weights(values, mesh, w_real, k0=1, x0=0, y0=0, z0=0):
    """
    Reconstructs the knot field from weighted LG modes.

    Parameters:
    -----------
    values : dict
        A dictionary containing quantum numbers and weights with keys 'l', 'p', and 'weight'.
    mesh : tuple of ndarrays
        The 3D mesh grids (x, y, z).
    w_real : float
        Width parameter for the Laguerre-Gaussian modes.
    k0, x0, y0, z0 : float, optional
        Parameters for the Laguerre-Gaussian function.

    Returns:
    --------
    field_new : ndarray
        The reconstructed field from weighted LG modes, normalized to max absolute value 1.
    """
    res = np.shape(mesh[0])
    field_new = np.zeros(res).astype(np.complex128)

    # Iterate over the LG modes
    for i in range(len(values['l'])):
        l, p, weight = values['l'][i], values['p'][i], values['weight'][i]
        # Add each mode to the field with the corresponding weight
        field_new += weight * LG_simple(*mesh, l=l, p=p, width=w_real, k0=k0, x0=x0, y0=y0, z0=z0)

    # Normalize the field to have max absolute value 1
    field_new = field_new / np.abs(field_new).max()
    return field_new


if __name__ == "__main__":
    """
    Main script to calculate and visualize the knot field using specified braid functions and Laguerre-Gaussian (LG) modes.
    The script generates a 3D mesh grid, computes the field for a selected braid, and plots the resulting knot field.
    """

    # Define the spatial limits for the 3D mesh grid in the x, y, and z directions
    x_lim_3D_knot, y_lim_3D_knot, z_lim_3D_knot = (-7.0, 7.0), (-7.0, 7.0), (-2.0, 2.0)

    # Define the resolution for the 3D mesh grid in the x, y, and z directions
    res_x_3D_knot, res_y_3D_knot, res_z_3D_knot = 90, 90, 40
    # Alternative higher resolution settings (commented out)
    # res_x_3D_knot, res_y_3D_knot, res_z_3D_knot = 120, 120, 90

    # Create the z-coordinates based on the resolution
    if res_z_3D_knot != 1:
        z_3D_knot = np.linspace(*z_lim_3D_knot, res_z_3D_knot)
    else:
        z_3D_knot = 0

    # Parameters for the LG mode
    width0 = 1.6  # Width parameter for LG mode
    k0 = 1  # Wave number
    z0 = 0  # z-coordinate shift

    # Generate the x and y coordinates and create a 3D mesh grid
    x_3D_knot, y_3D_knot = np.linspace(*x_lim_3D_knot, res_x_3D_knot), np.linspace(*y_lim_3D_knot, res_y_3D_knot)
    mesh_3D_knot = np.meshgrid(x_3D_knot, y_3D_knot, z_3D_knot, indexing='ij')

    # Choose the braid function to calculate LG modes and knot field
    # Uncomment the desired braid field to calculate
    # values = unknot_4_any(mesh_3D_knot, braid_func=braid, plot=True, angle_size=(2, 1, 2, 0))
    # values = trefoil_optimized_new(mesh_3D_knot, braid_func=braid, plot=True)
    # values = unknot_6unknot_4unknot_4_any(mesh_3D_knot, braid_func=braid, plot=True)
    values = trefoil_standard_13(mesh_3D_knot, braid_func=braid, plot=True)

    # Reconstruct the field from the given weights and LG mode parameters
    field = field_knot_from_weights(
        values, mesh_3D_knot, width0, k0=k0, x0=0, y0=0, z0=z0
    )

    # Plot the field at the center plane along the z-axis
    plot_along_Z = 0  # Set to 1 to enable plotting along multiple planes
    plot_field_both(field[:, :, res_z_3D_knot // 2])

    # Optionally plot the field along different z-planes
    if plot_along_Z:
        plot_field_both(field[:, :, res_z_3D_knot // 4])
        plot_field_both(field[:, :, 0])

    # Optionally plot the 3D singularity points of the calculated field
    plot_3D = 0  # Set to 1 to enable 3D plotting
    if plot_3D:
        dots_bound = [
            [0, 0, 0],  # Lower boundary of the 3D plot
            [res_x_3D_knot, res_y_3D_knot, res_z_3D_knot],  # Upper boundary of the 3D plot
        ]
        # Get singularity points and plot them
        dots_init_dict, dots_init = sing.get_singularities(np.angle(field), axesAll=True, returnDict=True)
        pl.plotDots(dots_init, dots_bound, color='black', show=True, size=10)
