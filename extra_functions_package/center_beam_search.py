"""
Field Processing for Laser Beams with LG Modes

This script provides a set of tools and a workflow for processing laser beam fields, specifically those represented by Laguerre-Gaussian (LG) modes. The main functionality includes reading and processing 2D complex fields from MATLAB files, normalizing the fields, interpolating them for improved resolution, finding the beam center, and removing beam tilts. The processed fields can then be cropped for use in further 3D calculations or analysis.

Key Features:
1. **Field Reading and Normalization**: The script reads field data from MATLAB files and normalizes it to prepare for further processing.
2. **Beam Width Estimation**: An approximate beam width (waist) is calculated based on the input field.
3. **Interpolation for Center Finding**: The field is interpolated to a higher resolution to facilitate accurate beam center finding.
4. **Beam Center Detection**: The beam center is found by shifting the field along the x and y axes to minimize variance.
5. **Tilt Removal**: Any tilt in the beam is removed to standardize the field for further analysis.
6. **Field Cropping**: The field is cropped around the detected center to a specified resolution, ready for use in simulations or further processing.
7. **Plotting and Visualization**: Intermediate results can be plotted for visualization to help understand the beam properties and transformations at each step.

Functions and Workflow:
- `main_field_processing(...)`: This is the main function of the script, which integrates all the processing steps.
  - Reads a 2D complex field from a specified MATLAB file.
  - Normalizes the field and estimates the beam width.
  - Uses interpolation to increase the field resolution for accurate center finding.
  - Finds the beam center and removes any tilt present in the field.
  - Crops the field around the detected center to the specified resolution.
  - If enabled, plots intermediate fields to visualize the changes made during processing.

- **Helper Functions**:
  - `read_field_2D_single(path, field=None)`: Reads a 2D array from a MATLAB `.mat` file and converts it to a numpy array.
  - `normalization_field(field)`: Normalizes the input field to ensure the total energy remains consistent.
  - `find_beam_waist(field, mesh=None)`: Estimates the beam width for the LG beam.
  - `field_interpolation(...)`: Interpolates the input field to a new resolution for better center detection or other processing steps.
  - `center_beam_finding(...)`: Finds the center of the beam by shifting along the x and y axes to minimize variance.
  - `remove_tilt(field, mesh, eta, gamma, k=1)`: Removes any tilt present in the beam.
  - `plot_field(field, save=None)`: Plots the intensity and phase of the input field, which is useful for visualization.

- **Usage**:
  The `main_field_processing` function can be used to handle various fields from experiments or simulations that are saved as `.mat` files. It processes these fields through a sequence of steps involving normalization, center finding, tilt removal, and cropping. The resulting field is then ready for further analysis, such as studying LG spectra or other physical properties of the beam.

Example Usage:
```python
if __name__ == '__main__':
    path = 'Uz_trefoilunmod_exp.mat'
    main_field_processing(
        path,
        field_name='Uz',
        plotting=False,
        resolution_iterpol_center=(80, 80),
        stepXY=(1, 1),
        zero_pad=0,
        xMinMax_frac_center=(-1., 1.),
        yMinMax_frac_center=(-1., 1.),
        resolution_interpol_working=(80, 80),
        xMinMax_frac_working=(-1, 1),
        yMinMax_frac_working=(-1, 1),
        resolution_crop=(65, 65),
        moments_init={'p': (0, 6), 'l': (-5, 3)},
        moments_center={'p': (0, 5), 'l': (-4, 2)}
    )
```
This example processes a beam saved in the MATLAB file `Uz_trefoilunmod_exp.mat`, detects its center, removes any tilt, and crops it for further analysis.

Requirements:
- This script requires the `numpy`, `matplotlib`, and `scipy.io` libraries.
- The field processing also depends on external packages (`fg`, `bp`, `pl`) that contain utility functions for field generation, beam profiles, and plotting.
- `.mat` files containing the complex fields must be available for input.

Notes:
- The accuracy of the beam center detection and tilt removal steps depends on the initial resolution and quality of the input fields. Interpolation is used to increase accuracy, but care should be taken when setting parameters for these operations.
- The field must be normalized before performing any operations that depend on energy conservation.
- The default parameters are provided to work with typical LG beams, but they can be adjusted to fit specific experimental setups or requirements.

"""

import extra_functions_package.plotings as pl
import extra_functions_package.beams_and_pulses as bp

from extra_functions_package.data_generation_old import *


def LG_spectre_coeff(field, l, p, xM=(-1, 1), yM=(-1, 1), width=1., k0=1., mesh=None, functions=bp.LG_simple, **kwargs):
    """
    Function calculates a single coefficient of LG_l_p in the LG spectrum of the field
    :param field: complex electric field
    :param l: azimuthal index of LG beam
    :param p: radial index of LG beam
    :param xM: x boundaries for an LG beam (if Mesh is None)
    :param yM: y boundaries for an LG beam (if Mesh is None)
    :param width: LG beam width
    :param k0: k0 in LG beam but I believe it doesn't affect anything since we are in z=0
    :param mesh: mesh for LG beam. if None, xM and yM are used
    :return: complex weight of LG_l_p in the spectrum
    """

    if mesh is None:
        shape = np.shape(field)
        mesh = fg.create_mesh_XY(xMinMax=xM, yMinMax=yM, xRes=shape[0], yRes=shape[1])
        dS = ((xM[1] - xM[0]) / (shape[0] - 1)) * ((yM[1] - yM[0]) / (shape[1] - 1))
    else:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        dS = (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
    LGlp = functions(*mesh, l=l, p=p, width=width, k0=k0, **kwargs)

    return np.sum(field * np.conj(LGlp)) * dS


def displacement_lateral(field_func, mesh, r_0, eta, **kwargs):
    """
    Shift the field laterally by a given radius and angle.

    Parameters:
    field_func (callable): Function representing the 2D field.
    mesh (tuple): Mesh for the final field.
    r_0 (float): Radius-vector in polar coordinates (shift distance).
    eta (float): Angle in polar coordinates (shift direction in radians).
    **kwargs: Additional keyword arguments to pass to the field function.

    Returns:
    ndarray: Array with the shifted field.
    """
    xArray, yArray = fg.arrays_from_mesh(mesh)
    # Shift coordinates by r_0 along angle eta
    xArray, yArray = xArray - r_0 * np.cos(eta), yArray - r_0 * np.sin(eta)
    mesh_new = np.meshgrid(xArray, yArray, indexing='ij')
    return field_func(*mesh_new, **kwargs)


def displacement_deflection(field_func, mesh, eta, gamma, k=1.0, **kwargs):
    """
    Tilt the field using angles eta and gamma.

    Parameters:
    field_func (callable): Function representing the 2D field.
    mesh (tuple): Mesh for the final field.
    eta (float): Angle in polar coordinates for tilt direction (radians).
    gamma (float): Angle in polar coordinates for tilt magnitude (radians).
    k (float): Wave number. Default is 1.0.
    **kwargs: Additional keyword arguments to pass to the field function.

    Returns:
    ndarray: Array with the tilted field.
    """
    rho = fg.rho(*mesh)
    phi = fg.phi(*mesh)
    # Apply deflection to the field
    field_change = np.exp(1j * k * rho * np.sin(gamma) * np.cos(phi - eta))
    return field_func(*mesh, **kwargs) * field_change


def remove_tilt(field, mesh, eta, gamma, k=1.0):
    """
    Remove the tilt from the given field.

    Parameters:
    field (ndarray): Initial field array.
    mesh (tuple): Mesh for the initial field.
    eta (float): Angle in polar coordinates for tilt direction (radians).
    gamma (float): Angle in polar coordinates for tilt magnitude (radians).
    k (float): Wave number. Default is 1.0.

    Returns:
    ndarray: Field array with the tilt removed.
    """
    rho = fg.rho(*mesh)
    phi = fg.phi(*mesh)
    # Remove deflection from the field
    field_change = np.exp(-1j * k * rho * np.sin(gamma) * np.cos(phi - eta))
    return field * field_change


def remove_shift(mesh, x0, y0):
    """
    Remove the shift from the given mesh.

    Parameters:
    mesh (tuple): Initial mesh array.
    x0 (float): Shift along the x-axis.
    y0 (float): Shift along the y-axis.

    Returns:
    tuple: New mesh with the shift removed.
    """
    xArray, yArray = fg.arrays_from_mesh(mesh)
    xArray, yArray = xArray - x0, yArray - y0
    mesh_new = np.meshgrid(xArray, yArray, indexing='ij')
    return mesh_new


def LG_transition_matrix_real_space(operator, l, p, l0, p0, xM=(-1, 1), yM=(-1, 1), shape=(100, 100),
                                    width=1.0, k0=1.0, mesh=None, **kwargs):
    """
    Calculate the transition element for an LG beam using a specified operator.

    Parameters:
    operator (callable): Operator to be applied (e.g., T_LD, T_DD, or their combination).
    l (int): Azimuthal index of the target LG beam.
    p (int): Radial index of the target LG beam.
    l0 (int): Azimuthal index of the initial LG beam.
    p0 (int): Radial index of the initial LG beam.
    xM (tuple): x boundaries for LG beam (if mesh is None). Default is (-1, 1).
    yM (tuple): y boundaries for LG beam (if mesh is None). Default is (-1, 1).
    shape (tuple): Size of the new mesh (if mesh is None). Default is (100, 100).
    width (float): LG beam width. Default is 1.0.
    k0 (float): Parameter k0 in LG beam. Default is 1.0.
    mesh (tuple, optional): Mesh for LG beam. If None, xM and yM are used to create a mesh.
    **kwargs: Additional keyword arguments to pass to the operator.

    Returns:
    complex: Value of the transition element.
    """
    if mesh is None:
        mesh = fg.create_mesh_XY(xM, yM, xRes=shape[0], yRes=shape[1])
        dS = ((xM[1] - xM[0]) / (shape[0] - 1)) * ((yM[1] - yM[0]) / (shape[1] - 1))
    else:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        dS = (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])

    operatorOnLG = operator(bp.LG_simple, mesh, z=0, l=l0, p=p0, width=width, k0=k0, x0=0, y0=0, z0=0, **kwargs)
    LGlp = bp.LG_simple(*mesh, l=l, p=p, width=width, k0=k0)
    return np.sum(operatorOnLG * np.conj(LGlp)) * dS


def shift_tilt_combined(field_func, mesh, r_0, eta, eta2, gamma, k=1.0, **kwargs):
    """
    Combine both shift and tilt operations on the field.

    Parameters:
    field_func (callable): Function representing the 2D field.
    mesh (tuple): Mesh for the final field.
    r_0 (float): Radius-vector in polar coordinates for shift distance.
    eta (float): Angle in polar coordinates for shift direction (radians).
    eta2 (float): Angle in polar coordinates for tilt direction (radians).
    gamma (float): Angle in polar coordinates for tilt magnitude (radians).
    k (float): Wave number. Default is 1.0.
    **kwargs: Additional keyword arguments to pass to the field function.

    Returns:
    ndarray: Array with the shifted and tilted field.
    """
    xArray, yArray = fg.arrays_from_mesh(mesh)
    # Apply lateral shift
    xArray, yArray = xArray - r_0 * np.cos(eta), yArray - r_0 * np.sin(eta)
    mesh_new = np.meshgrid(xArray, yArray, indexing='ij')
    # Apply deflection
    field_change = np.exp(1j * k * fg.rho(*mesh) * np.sin(gamma) * np.cos(fg.phi(*mesh) - eta2))
    return field_func(*mesh_new, **kwargs) * field_change


def variance_V_helper(Pl, lArray):
    """
    Helper function to calculate the variance for a single transition.

    Parameters:
    Pl (array-like): Probabilities associated with each value in lArray.
    lArray (array-like): Array of l values.

    Returns:
    float: Variance calculated using equation (23).
    """
    sum1, sum2 = 0, 0
    for p, l in zip(Pl, lArray):
        sum1 += p * l ** 2
        sum2 += p * l
    return sum1 - sum2 ** 2


def variance_single_transition_combined(field, mesh, r, eta, eta2, gamma,
                                        displacement_function=shift_tilt_combined,
                                        p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                                        width=1.0, k0=1.0):
    """
    Calculate the final variance for combined transitions.

    Parameters:
    field (ndarray): Complex electric field.
    mesh (tuple): Mesh for the calculation.
    r (float): Radius for the displacement.
    eta (float): Angle for lateral displacement (radians).
    eta2 (float): Angle for tilt direction (radians).
    gamma (float): Tilt magnitude (radians).
    displacement_function (callable): Function to apply both shift and tilt.
    p (tuple): Range of radial indices for integration (inclusive).
    l (tuple): Range of azimuthal indices for integration (inclusive).
    p0 (tuple): Range of initial radial indices for integration (inclusive).
    l0 (tuple): Range of initial azimuthal indices for integration (inclusive).
    width (float): LG beam width. Default is 1.0.
    k0 (float): LG beam parameter k0. Default is 1.0.

    Returns:
    float: The value of the variance.
    """
    p1, p2 = p
    l1, l2 = l
    p01, p02 = p0
    l01, l02 = l0
    Pl = np.zeros(l2 - l1 + 1)
    for ind_l, l in enumerate(np.arange(l1, l2 + 1)):
        sum_ = 0
        for p in np.arange(p1, p2 + 1):
            sum_inner = 0
            for l0 in np.arange(l01, l02 + 1):
                for p0 in np.arange(p01, p02 + 1):
                    element_ = LG_transition_matrix_real_space(displacement_function, l=l, p=p,
                                                               l0=l0, p0=p0, mesh=mesh,
                                                               r_0=r, eta=eta, eta2=eta2, gamma=gamma,
                                                               width=width, k0=k0)
                    value = LG_spectre_coeff(field, l=l0, p=p0, mesh=mesh, width=width, k0=k0)
                    sum_inner += (element_ * value)
            sum_ += np.abs(sum_inner) ** 2
        Pl[ind_l] = sum_
    V = variance_V_helper(Pl, np.arange(l1, l2 + 1))
    return V


def LG_spectrum(beam, l=(-3, 3), p=(0, 5), xM=(-1, 1), yM=(-1, 1), width=1.0, k0=1.0, mesh=None, plot=True,
                functions=bp.LG_simple, **kwargs):
    """
    Calculate the LG spectrum of a given beam.

    Parameters:
    beam (ndarray): Complex electric field representing the beam.
    l (tuple): Range of azimuthal indices for the spectrum (inclusive).
    p (tuple): Range of radial indices for the spectrum (inclusive).
    xM (tuple): x boundaries for LG beam (if mesh is None). Default is (-1, 1).
    yM (tuple): y boundaries for LG beam (if mesh is None). Default is (-1, 1).
    width (float): LG beam width. Default is 1.0.
    k0 (float): Parameter k0 in LG beam. Default is 1.0.
    mesh (tuple, optional): Mesh for LG beam. If None, xM and yM are used to create a mesh.
    plot (bool): Whether to plot the spectrum. Default is True.
    functions (callable): Function used to calculate LG modes. Default is bp.LG_simple.
    **kwargs: Additional keyword arguments to pass to the LG calculation function.

    Returns:
    ndarray: Complex LG spectrum of the beam.
    """
    l1, l2 = l
    p1, p2 = p
    spectrum = np.zeros((l2 - l1 + 1, p2 - p1 + 1), dtype=complex)
    for l in np.arange(l1, l2 + 1):
        for p in np.arange(p1, p2 + 1):
            value = LG_spectre_coeff(beam, l=l, p=p, xM=xM, yM=yM, width=width, k0=k0, mesh=mesh,
                                     functions=functions, **kwargs)
            spectrum[l - l1, p] = value
    if plot:
        pl.plot_2D(np.abs(spectrum), x=np.arange(l1 - 0.5, l2 + 1 + 0.5), y=np.arange(p1 - 0.5, p2 + 1 + 0.5),
                   interpolation='none', grid=True, xname='l', yname='p', show=False)
        plt.yticks(np.arange(p1, p2 + 1))
        plt.xticks(np.arange(l1, l2 + 1))
        plt.show()
    return spectrum


def beam_full_center(beam, mesh, stepEG=None, stepXY=None, displacement_function=shift_tilt_combined,
                     p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                     width=1.0, k0=1.0, x=None, y=None, eta2=0.0, gamma=0.0, threshold=0.99,
                     print_info=False):
    """
    Find the center of the beam by minimizing the variance using shifts and tilts.

    Parameters:
    beam (ndarray): Complex electric field representing the beam.
    mesh (tuple): Mesh for the calculation.
    stepEG (tuple, optional): Step sizes for eta2 and gamma (tilt). Default is (0.1, 0.1).
    stepXY (tuple, optional): Step sizes for x and y shifts. If None, calculated from mesh.
    displacement_function (callable): Function to apply both shift and tilt.
    p (tuple): Range of radial indices for integration (inclusive).
    l (tuple): Range of azimuthal indices for integration (inclusive).
    p0 (tuple): Range of initial radial indices for integration (inclusive).
    l0 (tuple): Range of initial azimuthal indices for integration (inclusive).
    width (float): LG beam width. Default is 1.0.
    k0 (float): LG beam parameter k0. Default is 1.0.
    x (float, optional): Initial x coordinate for search. Default is center of mesh.
    y (float, optional): Initial y coordinate for search. Default is center of mesh.
    eta2 (float): Initial tilt angle (radians). Default is 0.0.
    gamma (float): Initial tilt magnitude (radians). Default is 0.0.
    threshold (float): Threshold to stop search. Default is 0.99.
    print_info (bool): Whether to print intermediate results. Default is False.

    Returns:
    tuple: Final coordinates (x, y, eta2, gamma).
    """
    if stepEG is None:
        stepEG = (0.1, 0.1)
    if stepXY is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        stepXY = (xArray[1] - xArray[0], yArray[1] - yArray[0])
    if x is None:
        xArray, _ = fg.arrays_from_mesh(mesh)
        x = xArray[len(xArray) // 2]
    if y is None:
        _, yArray = fg.arrays_from_mesh(mesh)
        y = yArray[len(yArray) // 2]

    def search(coordinate):
        nonlocal x, y, eta2, gamma, var0
        varIt = var0
        signX = 1
        correctWay = False
        while True:
            if coordinate == 'x':
                x += signX * stepXY[0]
            elif coordinate == 'y':
                y += signX * stepXY[1]
            elif coordinate == 'eta2':
                eta2 += signX * stepEG[0]
            elif coordinate == 'gamma':
                gamma += signX * stepEG[1]

            var = variance_single_transition_combined(beam, mesh, r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                                      eta2=eta2, gamma=gamma,
                                                      displacement_function=displacement_function,
                                                      p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
            if print_info:
                print(f'x={round(x, 3)}, y={round(y, 3)},'
                      f' eta={round(eta2 * 180 / np.pi, 3)}*, gamma={round(gamma * 180 / np.pi, 3)}*, var={var}')
            if var < (varIt * threshold):
                varIt = var
                correctWay = True
            else:
                if coordinate == 'x':
                    x -= signX * stepXY[0]
                elif coordinate == 'y':
                    y -= signX * stepXY[1]
                elif coordinate == 'eta2':
                    eta2 -= signX * stepEG[0]
                elif coordinate == 'gamma':
                    gamma -= signX * stepEG[1]
                if correctWay:
                    break
                else:
                    correctWay = True
                    signX *= -1

        return varIt

    var0 = variance_single_transition_combined(beam, mesh, r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                               eta2=eta2, gamma=gamma,
                                               displacement_function=displacement_function,
                                               p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)

    while True:
        search(coordinate='x')
        search(coordinate='y')
        search(coordinate='eta2')
        varEG = search(coordinate='gamma')
        if varEG == var0:
            print('FINAL COORDINATES:')
            print(f'x={round(x, 5)}, y={round(y, 5)},'
                  f' eta={round(eta2 * 180 / np.pi, 5)}*,'
                  f' gamma={round(gamma * 180 / np.pi, 5)}*, var={var0}')
            return x, y, eta2, gamma
        else:
            var0 = varEG


def find_width(beam, mesh, widthStep=0.1, l=(-8, 8), p=(0, 9), width=1.0, k0=1.0, print_steps=True):
    """
    Find the approximate beam waist (at any position, for any beam represented as a sum of LG modes).

    Parameters:
    beam (ndarray): Complex electric field representing the beam.
    mesh (tuple): Mesh for the calculation.
    widthStep (float): Precision step for beam width search (not highly accurate).
    l (tuple): Range of azimuthal indices to cover the entire spectrum.
    p (tuple): Range of radial indices to cover the entire spectrum.
    width (float): Initial estimate of the beam width.
    k0 (float): Wave number (does not affect calculation since we work at z=0).
    print_steps (bool): Whether to print intermediate values during search.

    Returns:
    float: Approximate beam width.
    """
    minSpec = np.sum(np.abs(LG_spectrum(beam, l=l, p=p, mesh=mesh, width=width, k0=k0, plot=False)) ** (1 / 2))
    correctWay = False
    direction = +1
    while True:
        width += direction * widthStep
        spec = np.sum(np.abs(LG_spectrum(beam, l=l, p=p, mesh=mesh, width=width, k0=k0, plot=False)) ** (1 / 2))
        if print_steps:
            print(width, spec)
        if spec < minSpec:
            minSpec = spec
            correctWay = True
        else:
            width -= direction * widthStep
            if correctWay:
                break
            else:
                correctWay = True
                direction *= -1
    return width


def beam_center_coordinates(beam, mesh, stepXY=None, stepEG=None,
                            p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                            width=1.0, k0=1.0, x=None, y=None, eta=0.0, gamma=0.0,
                            shift=True, tilt=True, fast=False):
    """
    Find the beam center coordinates by applying shifts and tilts.

    Parameters:
    beam (ndarray): Complex electric field representing the beam.
    mesh (tuple): Mesh for the calculation.
    stepXY (tuple, optional): Step sizes for x and y shifts. If None, calculated from mesh.
    stepEG (tuple, optional): Step sizes for eta and gamma (tilt). Default is None.
    p (tuple): Range of radial indices for integration (inclusive).
    l (tuple): Range of azimuthal indices for integration (inclusive).
    p0 (tuple): Range of initial radial indices for integration (inclusive).
    l0 (tuple): Range of initial azimuthal indices for integration (inclusive).
    width (float): LG beam width. Default is 1.0.
    k0 (float): LG beam parameter k0. Default is 1.0.
    x (float, optional): Initial x coordinate for search. Default is center of mesh.
    y (float, optional): Initial y coordinate for search. Default is center of mesh.
    eta (float): Initial tilt angle (radians). Default is 0.0.
    gamma (float): Initial tilt magnitude (radians). Default is 0.0.
    shift (bool): Whether to apply shifting. Default is True.
    tilt (bool): Whether to apply tilting. Default is True.
    fast (bool): Whether to perform a fast search without multiple iterations. Default is False.

    Returns:
    tuple: Updated mesh and beam after centering.
    """
    if fast or not shift or not tilt:
        print(f'Not the perfect center: shift={shift}, tilt={tilt}, fast={fast}')
        if shift:
            x, y = center_beam_finding(beam, mesh, x=x, y=y, stepXY=stepXY,
                                       p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
        if tilt:
            eta, gamma = tilt_beam_finding(beam, mesh, eta=eta, gamma=gamma, stepEG=stepEG,
                                           p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
    else:
        meshNew = mesh
        while True:
            print('New cycle')
            x, y = center_beam_finding(beam, meshNew, x=x, y=y, stepXY=stepXY,
                                       p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
            eta, gamma = tilt_beam_finding(beam, mesh, eta=eta, gamma=gamma, stepEG=stepEG,
                                           p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)

            meshNew = remove_shift(meshNew, -x, -y)
            pl.plot_2D(np.angle(beam))
            pl.plot_2D(np.abs(beam))
            print(x, y, eta, gamma)
            if (x, y, eta, gamma) == (0, 0, 0, 0):
                break
            x, y, eta, gamma = [0] * 4

    print(f'x={x}, y={y}, eta={eta / np.pi * 180}, gamma={gamma / np.pi * 180}')
    return mesh, beam


# Old functions that may not be functional, but could be useful in some cases
# These functions were used for the algorithm implementation for separated shift and tilt, not combined.
# They might be useful if the fields have only shift or only tilt.
####################################################################################

def variance_single_transition(field, mesh, r, eta, displacement_function=displacement_lateral,
                               p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                               width=1.0, k0=1.0):
    """
    Calculate the variance for a single transition (for either shift or tilt).

    Parameters:
    field (ndarray): Complex electric field representing the beam.
    mesh (tuple): Mesh for the calculation.
    r (float): Radius for the displacement.
    eta (float): Angle for displacement (radians).
    displacement_function (callable): Function to apply either shift or tilt.
    p (tuple): Range of radial indices for integration (inclusive).
    l (tuple): Range of azimuthal indices for integration (inclusive).
    p0 (tuple): Range of initial radial indices for integration (inclusive).
    l0 (tuple): Range of initial azimuthal indices for integration (inclusive).
    width (float): LG beam width. Default is 1.0.
    k0 (float): LG beam parameter k0. Default is 1.0.

    Returns:
    float: The value of the variance.
    """
    p1, p2 = p
    l1, l2 = l
    p01, p02 = p0
    l01, l02 = l0
    Pl = np.zeros(l2 - l1 + 1)
    for ind_l, l in enumerate(np.arange(l1, l2 + 1)):
        sum_ = 0
        for p in np.arange(p1, p2 + 1):
            sum_inner = 0
            for l0 in np.arange(l01, l02 + 1):
                for p0 in np.arange(p01, p02 + 1):
                    if displacement_function is displacement_lateral:
                        element_ = LG_transition_matrix_real_space(displacement_function, l=l, p=p,
                                                                   l0=l0, p0=p0,
                                                                   mesh=mesh, r_0=r, eta=eta,
                                                                   width=width, k0=k0)
                    else:
                        element_ = LG_transition_matrix_real_space(displacement_function, l=l, p=p,
                                                                   l0=l0, p0=p0,
                                                                   mesh=mesh, eta=r, gamma=eta,
                                                                   width=width, k0=k0)
                    value = LG_spectre_coeff(field, l=l0, p=p0, mesh=mesh, width=width, k0=k0)
                    sum_inner += (element_ * value)
            sum_ += np.abs(sum_inner) ** 2
        Pl[ind_l] = sum_
    V = variance_V_helper(Pl, np.arange(l1, l2 + 1))
    return V


def variance_map_shift(beam, mesh, displacement_function=displacement_lateral,
                       resolution_V=(4, 4), xBound=(-1, 1), yBound=(-1, 1), width=1.0, k0=1.0,
                       p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3)):
    """
    Generate a variance map for beam shifts.

    Parameters:
    beam (ndarray): Complex electric field representing the beam.
    mesh (tuple): Mesh for the calculation.
    displacement_function (callable): Function to apply shift to the beam.
    resolution_V (tuple): Resolution of the variance map.
    xBound (tuple): Bounds for x-coordinate shifts.
    yBound (tuple): Bounds for y-coordinate shifts.
    width (float): LG beam width. Default is 1.0.
    k0 (float): LG beam parameter k0. Default is 1.0.
    p (tuple): Range of radial indices for integration (inclusive).
    l (tuple): Range of azimuthal indices for integration (inclusive).
    p0 (tuple): Range of initial radial indices for integration (inclusive).
    l0 (tuple): Range of initial azimuthal indices for integration (inclusive).

    Returns:
    ndarray: Variance map for the beam shifts.
    """
    V = np.zeros(resolution_V)
    xArray = np.linspace(*xBound, resolution_V[0])
    yArray = np.linspace(*yBound, resolution_V[1])
    for i, x in enumerate(xArray):
        print('Main Coordinate x: ', i)
        for j, y in enumerate(yArray):
            print('y: ', j)
            r = fg.rho(x, y)
            eta = np.angle(x + 1j * y)
            V[i, j] = variance_single_transition(beam, mesh, r, eta,
                                                 displacement_function=displacement_function,
                                                 width=width, k0=k0,
                                                 p=p, l=l, p0=p0, l0=l0)
    return V


def variance_map_tilt(beam, mesh, displacement_function=displacement_deflection,
                      resolution_V=(4, 4), etaBound=(-1, 1), gammaBound=(-1, 1), width=1.0, k0=1.0,
                      p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3)):
    """
    Generate a variance map for beam tilts.

    Parameters:
    beam (ndarray): Complex electric field representing the beam.
    mesh (tuple): Mesh for the calculation.
    displacement_function (callable): Function to apply tilt to the beam.
    resolution_V (tuple): Resolution of the variance map.
    etaBound (tuple): Bounds for eta angle.
    gammaBound (tuple): Bounds for gamma angle.
    width (float): LG beam width. Default is 1.0.
    k0 (float): LG beam parameter k0. Default is 1.0.
    p (tuple): Range of radial indices for integration (inclusive).
    l (tuple): Range of azimuthal indices for integration (inclusive).
    p0 (tuple): Range of initial radial indices for integration (inclusive).
    l0 (tuple): Range of initial azimuthal indices for integration (inclusive).

    Returns:
    ndarray: Variance map for the beam tilts.
    """
    V = np.zeros(resolution_V)
    etaArray = np.linspace(*etaBound, resolution_V[0])
    gammaArray = np.linspace(*gammaBound, resolution_V[1])
    for i, eta in enumerate(etaArray):
        print('Main Coordinate eta: ', i)
        for j, gamma in enumerate(gammaArray):
            print('gamma: ', j)
            V[i, j] = variance_single_transition(beam, mesh, r=eta, eta=gamma,
                                                 displacement_function=displacement_function,
                                                 width=width, k0=k0,
                                                 p=p, l=l, p0=p0, l0=l0)
    return V


def find_beam_waist(field, mesh=None):
    """
    Wrapper for the beam waist finder. More details in knots_ML.center_beam_search.

    Parameters:
    field (ndarray): Complex electric field representing the beam.
    mesh (tuple, optional): Mesh for the calculation. If None, a mesh is created from the field shape.

    Returns:
    float: Approximate beam waist.
    """
    shape = np.shape(field)
    if mesh is None:
        mesh = fg.create_mesh_XY(xRes=shape[0], yRes=shape[1])
    width = find_width(field, mesh=mesh, width=shape[1] // 4, widthStep=1, print_steps=False)
    return width


def center_beam_finding(beam, mesh, stepXY=None, displacement_function=displacement_lateral,
                        p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                        width=1, k0=1, x=None, y=None):
    """
    Find the center of the beam by shifting along the x and y axes to minimize variance.

    Parameters:
    beam (ndarray): Complex electric field representing the beam.
    mesh (tuple): Mesh for the calculation.
    stepXY (tuple, optional): Step sizes for x and y shifts. If None, calculated from mesh.
    displacement_function (callable): Function to apply lateral shift.
    p (tuple): Range of radial indices for integration (inclusive).
    l (tuple): Range of azimuthal indices for integration (inclusive).
    p0 (tuple): Range of initial radial indices for integration (inclusive).
    l0 (tuple): Range of initial azimuthal indices for integration (inclusive).
    width (float): LG beam width. Default is 1.
    k0 (float): LG beam parameter k0. Default is 1.
    x (float, optional): Initial x coordinate for search. Default is center of mesh.
    y (float, optional): Initial y coordinate for search. Default is center of mesh.

    Returns:
    tuple: x and y coordinates of the beam center.
    """
    if stepXY is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        stepXY = xArray[1] - xArray[0], yArray[1] - yArray[0]
    if x is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        x = xArray[len(xArray) // 2]
    if y is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        y = yArray[len(yArray) // 2]

    def search(xFlag):
        nonlocal x, y, var0
        varIt = var0
        signX = 1
        correctWay = False
        while True:
            if xFlag:
                x = x + signX * stepXY[0]
            else:
                y = y + signX * stepXY[1]
            var = variance_single_transition(beam, mesh=mesh, displacement_function=displacement_function,
                                             r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                             p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
            if var < varIt:
                varIt = var
                correctWay = True
            else:
                if xFlag:
                    x = x - signX * stepXY[0]
                else:
                    y = y - signX * stepXY[1]
                if correctWay:
                    break
                else:
                    correctWay = True
                    signX *= -1

        return varIt

    var0 = variance_single_transition(beam, mesh=mesh, r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                      p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)

    while True:
        search(xFlag=True)
        varXY = search(xFlag=False)
        if varXY == var0:
            return x, y
        else:
            var0 = varXY


def tilt_beam_finding(beam, mesh, stepEG=None, displacement_function=displacement_deflection,
                      p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                      width=1, k0=1, eta=0., gamma=0.):
    """
    Find the tilt of the beam by adjusting eta and gamma to minimize variance.

    Parameters:
    beam (ndarray): Complex electric field representing the beam.
    mesh (tuple): Mesh for the calculation.
    stepEG (tuple, optional): Step sizes for eta and gamma adjustments. Default is (0.1, 0.1).
    displacement_function (callable): Function to apply deflection.
    p (tuple): Range of radial indices for integration (inclusive).
    l (tuple): Range of azimuthal indices for integration (inclusive).
    p0 (tuple): Range of initial radial indices for integration (inclusive).
    l0 (tuple): Range of initial azimuthal indices for integration (inclusive).
    width (float): LG beam width. Default is 1.
    k0 (float): LG beam parameter k0. Default is 1.
    eta (float): Initial eta angle (radians). Default is 0.
    gamma (float): Initial gamma angle (radians). Default is 0.

    Returns:
    tuple: eta and gamma values after optimization.
    """
    if stepEG is None:
        stepEG = 0.1, 0.1

    def search(etaFlag):
        nonlocal eta, gamma, var0
        varIt = var0
        signX = 1
        correctWay = False
        while True:
            if etaFlag:
                eta = eta + signX * stepEG[0]
            else:
                gamma = gamma + signX * stepEG[1]
            var = variance_single_transition(beam, mesh, eta, gamma,
                                             displacement_function=displacement_function,
                                             p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)

            print(f'eta={eta}, gamma={gamma}, var={var}')
            if var < varIt:
                varIt = var
                correctWay = True
            else:
                if etaFlag:
                    eta = eta - signX * stepEG[0]
                else:
                    gamma = gamma - signX * stepEG[1]
                if correctWay:
                    break
                else:
                    correctWay = True
                    signX *= -1

        return varIt

    var0 = variance_single_transition(beam, mesh, eta, gamma,
                                      displacement_function=displacement_function,
                                      p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)

    while True:
        search(etaFlag=True)
        varEG = search(etaFlag=False)
        if varEG == var0:
            return eta, gamma
        else:
            var0 = varEG


def field_interpolation(field, mesh=None, resolution=(100, 100),
                        xMinMax_frac=(1, 1), yMinMax_frac=(1, 1), fill_value=True):
    """
    Interpolate the given field to a new resolution and size.

    This function is a wrapper for the `fg.interpolation_complex` function, which handles the complex field interpolation.
    The output field will have the specified resolution and dimensions, which are set as fractions of the original
     dimensions.

    Parameters:
    field (ndarray): The input field to be interpolated.
    mesh (tuple, optional): The mesh of the field. If None, a mesh is created based on the field dimensions.
    resolution (tuple): The resolution (number of points) for the new interpolated field. Default is (100, 100).
    xMinMax_frac (tuple): Fractions of the original x-dimension to determine the new x boundaries. Default is (1, 1).
    yMinMax_frac (tuple): Fractions of the original y-dimension to determine the new y boundaries. Default is (1, 1).
    fill_value (bool): Whether to use a fill value for out-of-bounds interpolation. Default is True.

    Returns:
    tuple: Interpolated field and the new mesh.
    """
    shape = np.shape(field)
    if mesh is None:
        mesh = fg.create_mesh_XY(xRes=shape[0], yRes=shape[1])
    interpol_field = fg.interpolation_complex(field, mesh=mesh, fill_value=fill_value)
    xMinMax = int(shape[0] // 2 * xMinMax_frac[0]), int(shape[0] // 2 * xMinMax_frac[1])
    yMinMax = int(shape[1] // 2 * yMinMax_frac[0]), int(shape[1] // 2 * yMinMax_frac[1])
    xyMesh_interpol = fg.create_mesh_XY(
        xRes=resolution[0], yRes=resolution[1],
        xMinMax=xMinMax, yMinMax=yMinMax)
    return interpol_field(*xyMesh_interpol), xyMesh_interpol


def normalization_field(field):
    """
    Normalize the given field.

    This function normalizes the input field such that the sum of the squared magnitudes is equal to 1.
    This is commonly used to prepare fields for further calculations, such as beam center finding.

    Parameters:
    field (ndarray): The input field to be normalized.

    Returns:
    ndarray: The normalized field.
    """
    field_norm = field / np.sqrt(np.sum(np.abs(field) ** 2))
    return field_norm


def read_field_2D_single(path, field=None):
    """
    Read a 2D field from a MATLAB `.mat` file.

    This function reads a 2D array from a MATLAB `.mat` file and converts it to a numpy array.
    If the `field` parameter is not provided, the function will attempt to find a suitable 2D field automatically.

    Parameters:
    path (str): Full path to the `.mat` file.
    field (str, optional): The name of the field to read. If None, it tries to find a suitable field automatically.

    Returns:
    ndarray: The 2D field read from the file.
    """
    field_read = sio.loadmat(path, appendmat=False)
    if field is None:
        for field_check in field_read:
            if len(np.shape(np.array(field_read[field_check]))) == 2:
                field = field_check
                break
    return np.array(field_read[field])


def plot_field(field, save=None):
    """
    Plot the intensity and phase of a given field.

    This function creates a plot of the intensity and phase of the input field side by side.
    The intensity (magnitude) and phase are useful for understanding the spatial characteristics of the field.

    Parameters:
    field (ndarray): The input field to be plotted.
    save (str, optional): Path to save the plot as a `.png` file. If None, the plot is not saved. Default is None.

    Returns:
    None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    image1 = ax1.imshow(np.abs(field))
    ax1.set_title('|E| (Intensity)')
    plt.colorbar(image1, ax=ax1, shrink=0.4, pad=0.02, fraction=0.1)
    image2 = ax2.imshow(np.angle(field), cmap='jet')
    ax2.set_title('Phase(E)')
    plt.colorbar(image2, ax=ax2, shrink=0.4, pad=0.02, fraction=0.1)
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, format='png')
    plt.show()


def main_field_processing(
        path,
        field_name=None,
        plotting=True,
        resolution_iterpol_center=(70, 70),
        stepXY=(3, 3),
        zero_pad=0,
        xMinMax_frac_center=(1, 1),
        yMinMax_frac_center=(1, 1),
        resolution_interpol_working=(150, 150),
        xMinMax_frac_working=(1, 1),
        yMinMax_frac_working=(1, 1),
        resolution_crop=(120, 120),
        moments_init=None,
        moments_center=None,
):
    """
    Process a field read from a MATLAB file to prepare it for further analysis.

    This function performs a series of operations on the field, including reading, normalization, beam width estimation,
    interpolation for center finding, center detection, rescaling, tilt removal, and cropping for final use in calculations.

    Parameters:
    path (str): Path to the MATLAB file containing the field data.
    field_name (str, optional): Name of the field to be read. If None, the function will attempt to find a suitable field.
    plotting (bool): Whether to plot intermediate results for visualization. Default is True.
    resolution_iterpol_center (tuple): Resolution for interpolating the field for center detection. Default is (70, 70).
    stepXY (tuple): Step sizes for x and y coordinate adjustments when finding the beam center. Default is (3, 3).
    zero_pad (int): Padding size for the field when interpolating. Default is 0.
    xMinMax_frac_center (tuple): Fractional rescaling along the x-axis for center detection. Default is (1, 1).
    yMinMax_frac_center (tuple): Fractional rescaling along the y-axis for center detection. Default is (1, 1).
    resolution_interpol_working (tuple): Resolution for the final interpolated field before cropping. Default is (150, 150).
    xMinMax_frac_working (tuple): Fractional rescaling along the x-axis for final working resolution. Default is (1, 1).
    yMinMax_frac_working (tuple): Fractional rescaling along the y-axis for final working resolution. Default is (1, 1).
    resolution_crop (tuple): Final resolution after cropping the field around its center. Default is (120, 120).
    moments_init (dict, optional): Initial moments for LG spectrum analysis. Default is {'p': (0, 6), 'l': (-4, 4)}.
    moments_center (dict, optional): Moments for center finding. Default is {'p0': (0, 6), 'l0': (-4, 4)}.

    Returns:
    None
    """
    # Set default values for moments if not provided
    if moments_init is None:
        moments_init = {'p': (0, 6), 'l': (-4, 4)}
    if moments_center is None:
        moments_center = {'p0': (0, 6), 'l0': (-4, 4)}

    # Read the field from the MATLAB file
    field_init_all = read_field_2D_single(path, field=field_name)
    xy_coordinates = []

    # Process each 2D field in the dataset
    for i in range(0, field_init_all.shape[2], 1):
        field_init = field_init_all[:, :, i]
        if plotting:
            plot_field(field_init)

        # Normalize the field
        field_norm = normalization_field(field_init)
        if plotting:
            plot_field(field_norm)

        # Create the initial mesh
        mesh_init = fg.create_mesh_XY(xRes=np.shape(field_norm)[0], yRes=np.shape(field_norm)[1])

        # Estimate the beam width
        width = float(find_beam_waist(field_norm, mesh=mesh_init))
        if plotting:
            print(f'Approximate beam waist: {width}')

        # Interpolate the field for center finding
        field_interpol, mesh_interpol = field_interpolation(
            field_norm, mesh=mesh_init,
            resolution=(resolution_iterpol_center[0] - zero_pad * 2, resolution_iterpol_center[1] - zero_pad * 2),
            xMinMax_frac=xMinMax_frac_center, yMinMax_frac=yMinMax_frac_center
        )

        # Apply zero padding if specified
        if zero_pad:
            field_interpol = np.pad(field_interpol, zero_pad, 'constant')
            shape = np.shape(field_norm)
            xMinMax = int(shape[0] // 2 * xMinMax_frac_center[0]), int(shape[0] // 2 * xMinMax_frac_center[1])
            yMinMax = int(shape[1] // 2 * yMinMax_frac_center[0]), int(shape[1] // 2 * yMinMax_frac_center[1])
            mesh_interpol = fg.create_mesh_XY(
                xRes=resolution_iterpol_center[0], yRes=resolution_iterpol_center[1],
                xMinMax=xMinMax, yMinMax=yMinMax)

        if plotting:
            plot_field(field_interpol)

        # Find the beam center
        x, y = center_beam_finding(field_interpol, mesh_interpol,
                                   stepXY=stepXY, displacement_function=displacement_lateral,
                                   **moments_center,
                                   width=width, k0=1, x=None, y=None)
        xy_coordinates.append((x, y))
        eta = 0
        gamma = 0

        print(f'Coordinates: x={x}, y={y}; Tilt: eta={eta}, gamma={gamma}')

        # Interpolate the field to the desired working resolution for 3D calculations
        field_interpol2, mesh_interpol2 = field_interpolation(
            field_norm, mesh=mesh_init, resolution=resolution_interpol_working,
            xMinMax_frac=xMinMax_frac_working, yMinMax_frac=yMinMax_frac_working, fill_value=False
        )
        if plotting:
            plot_field(field_interpol2)

        # Remove tilt from the field
        field_untilted = remove_tilt(field_interpol2, mesh_interpol2, eta=-eta, gamma=gamma, k=1)
        if plotting:
            plot_field(field_untilted)

        # Scale the beam center
        shape = np.shape(field_untilted)
        scaling_factor2 = 1. / np.shape(field_interpol)[0] * shape[0]
        x = int(x * scaling_factor2)
        y = int(y * scaling_factor2)

        # Crop the field around the center
        field_cropped = field_untilted[
                        shape[0] // 2 - x - resolution_crop[0] // 2:shape[0] // 2 - x + resolution_crop[0] // 2,
                        shape[1] // 2 - y - resolution_crop[1] // 2:shape[1] // 2 - y + resolution_crop[1] // 2]
        if plotting:
            plot_field(field_cropped)

        # Create mesh for the cropped field
        mesh = fg.create_mesh_XY(xRes=np.shape(field_cropped)[0], yRes=np.shape(field_cropped)[1])
        field = field_cropped

    print(xy_coordinates)
    np.save('coordinates_unmod2.npy', np.array(xy_coordinates))


####################################################################################


if __name__ == '__main__':
    exit()
# path = f'any_mat_file.mat'
# main_field_processing(
# 	path,
# 	field_name='Uz',
# 	plotting=False,
# 	resolution_iterpol_center=(80, 80),
# 	stepXY=(1, 1),
# 	zero_pad=0,
# 	xMinMax_frac_center=(-1., 1.),
# 	yMinMax_frac_center=(-1., 1.),
# 	resolution_interpol_working=(80, 80),
# 	xMinMax_frac_working=(-1, 1),
# 	yMinMax_frac_working=(-1, 1),
# 	resolution_crop=(65, 65),
# 	moments_init={'p': (0, 6), 'l': (-5, 3)},
# 	moments_center={'p': (0, 5), 'l': (-4, 2)})
