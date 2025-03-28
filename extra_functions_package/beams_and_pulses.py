"""
This module includes different optical beam shapes.

The module provides functions to generate and manipulate various optical beam configurations,
including classic Laguerre-Gaussian beams, trefoil knot configurations, Hopf link configurations,
and Milnor polynomials. Additionally, it allows for creating fields by combining multiple LG beams.

Functions:
    - LG_simple: Generates a classic Laguerre-Gaussian (LG) beam.
    - trefoil: Constructs a field based on a trefoil knot configuration using a combination of LG beams.
    - hopf: Constructs a field based on a Hopf link configuration using a combination of LG beams.
    - milnor_Pol_u_v_any: Creates a Milnor polynomial of the form u^a - v^b.
    - LG_combination: Creates a field by combining multiple LG beams according to specified coefficients and modes.

Example Usage:
    The following example demonstrates how to create a Milnor polynomial field and visualize it using the provided functions:

    ```python
    import my_functions.plotings as pl
    import extra_functions_package.functions_general as fg
    
    xyzMesh = fg.create_mesh_XYZ(4, 4, 1, zMin=None)
    beam = milnor_Pol_u_v_any(xyzMesh, uOrder=2, vOrder=2, H=1)
    pl.plot_2D(np.abs(beam[:, :, 20]))
    ```
    
    This example creates a mesh grid, generates a Milnor polynomial field with specified parameters, and plots a 2D slice of the beam.
"""

import numpy as np
from scipy.special import assoc_laguerre
import extra_functions_package.functions_general as fg
import math

np.seterr(divide='ignore', invalid='ignore')

def LG_simple(x, y, z=0, l=1, p=0, width=1, k0=1, x0=0, y0=0, z0=0, **kwargs):
    """
    Generates a classic Laguerre-Gaussian (LG) beam.

    Parameters:
        x, y, z: Spatial coordinates (floats or ndarrays).
        l: Azimuthal index (integer).
        p: Radial index (integer).
        width: Beam waist (float).
        k0: Wave number (float).
        x0, y0, z0: Center of the beam in x, y, and z (floats).

    Returns:
        Complex ndarray representing the LG beam.
    """

    def laguerre_polynomial(x, l, p):
        return assoc_laguerre(x, p, l)

    x = x - x0
    y = y - y0
    z = z - z0
    zR = (k0 * width ** 2)  # Rayleigh range

    rho = fg.rho(x, y)
    phi = fg.phi(x, y)
    gaussian_envelope = np.exp(-rho ** 2 / (2 * width ** 2 * (1 + 1j * z / zR)))
    laguerre_component = laguerre_polynomial(rho ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)), np.abs(l), p)
    amplitude_factor = (np.sqrt(math.factorial(p) / (np.pi * math.factorial(np.abs(l) + p)))
                        * rho ** np.abs(l) * np.exp(1j * l * phi)
                        / (width ** (np.abs(l) + 1) * (1 + 1j * z / zR) ** (np.abs(l) + 1))
                        * ((1 - 1j * z / zR) / (1 + 1j * z / zR)) ** p)
    E = amplitude_factor * gaussian_envelope * laguerre_component

    return E

def trefoil(*x, w, width=1, k0=1, aCoeff=None, coeffPrint=False, **kwargs):
    """
    Constructs a field based on a trefoil knot configuration using a combination of LG beams.

    Parameters:
        *x: Spatial coordinates (floats or ndarrays).
        w: Weight parameter (float).
        width: Beam waist (float).
        k0: Wave number (float).
        aCoeff: Coefficients for the LG beam combination (list or None).
        coeffPrint: Flag to print the coefficients (boolean).

    Returns:
        Complex ndarray representing the trefoil knot configuration.
    """
    H = 1.0
    if aCoeff is not None:
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    else:
        a00 = 1 * (H ** 6 - H ** 4 * w ** 2 - 2 * H ** 2 * w ** 4 + 6 * w ** 6) / H ** 6
        a01 = (w ** 2 * (1 * H ** 4 + 4 * w ** 2 * H ** 2 - 18 * w ** 4)) / H ** 6
        a02 = (- 2 * w ** 4 * (H ** 2 - 9 * w ** 2)) / H ** 6
        a03 = (-6 * w ** 6) / H ** 6
        a30 = (-8 * np.sqrt(6) * w ** 3) / H ** 3
        aCoeff = [a00, a01, a02, a03, a30]
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    if coeffPrint:
        print(aCoeff)
        print(f'a00 -> a01 -> a02 ->... -> a0n -> an0:')
        for i, a in enumerate(aCoeff):
            print(f'a{i}: {a:.3f}', end=',\t')
    field = sum(aCoeff[i] * LG_simple(*x, l=i if i < 4 else 3, p=i if i < 4 else 0, width=width, k0=k0, **kwargs)
                for i in range(len(aCoeff)))
    return field

def hopf(*x, w, width=1, k0=1, aCoeff=None, coeffPrint=False, **kwargs):
    """
    Constructs a field based on a Hopf link configuration using a combination of LG beams.

    Parameters:
        *x: Spatial coordinates (floats or ndarrays).
        w: Weight parameter (float).
        width: Beam waist (float).
        k0: Wave number (float).
        aCoeff: Coefficients for the LG beam combination (list or None).
        coeffPrint: Flag to print the coefficients (boolean).

    Returns:
        Complex ndarray representing the Hopf link configuration.
    """
    if aCoeff is not None:
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    else:
        a00 = 1 - 2 * w ** 2 + 2 * w ** 4
        a01 = 2 * w ** 2 - 4 * w ** 4
        a02 = 2 * w ** 4
        a20 = 4 * np.sqrt(2) * w ** 2
        aCoeff = [a00, a01, a02, a20]
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    if coeffPrint:
        print(aCoeff)
        print(f'a00 -> a01 -> a02 ->... -> a0n -> an0:')
        for i, a in enumerate(aCoeff):
            print(f'a{i}: {a:.3f}', end=',\t')
        print()
    field = sum(aCoeff[i] * LG_simple(*x, l=i if i < 3 else 2, p=i if i < 3 else 0, width=width, k0=k0, **kwargs)
                for i in range(len(aCoeff)))
    return field

def milnor_Pol_u_v_any(mesh, uOrder, vOrder, H=1):
    """
    Creates a Milnor polynomial of the form u^a - v^b.

    Parameters:
        mesh: Tuple of x, y, z coordinates (floats or ndarrays).
        uOrder: Order of u (integer).
        vOrder: Order of v (integer).
        H: Constant parameter (float).

    Returns:
        Complex ndarray representing the Milnor polynomial field.
    """
    x, y, z = mesh
    R = fg.rho(x, y)
    f = fg.phi(x, y)
    u = (-H ** 2 + R ** 2 + 2j * z * H + z ** 2) / (H ** 2 + R ** 2 + z ** 2)
    v = (2 * R * H * np.exp(1j * f)) / (H ** 2 + R ** 2 + z ** 2)
    return u ** uOrder - v ** vOrder

def LG_combination(*mesh, coefficients, modes, width=1, **kwargs):
    """
    Creates a field by combining multiple LG beams according to specified coefficients and modes.

    Parameters:
        *mesh: Spatial coordinates mesh (floats or ndarrays).
        coefficients: List of coefficients for the LG beam combination (list of floats).
        modes: List of tuples representing (l, p) modes (list of tuples).
        width: Beam waist (float or list of floats).

    Returns:
        Complex ndarray representing the combined LG beam field.
    """
    field = 0
    if isinstance(width, int):
        width = [width] * len(modes)
    for num, coefficient in enumerate(coefficients):
        field += coefficient * LG_simple(*mesh, l=modes[num][0], p=modes[num][1], width=width[num], **kwargs)
    return field

if __name__ == '__main__':
    """
    Example demonstrating the creation of a Milnor polynomial field and visualization.

    The following code creates a mesh grid, generates a Milnor polynomial field with specified parameters,
    and plots a 2D slice of the beam at a particular z-coordinate.
    """
    import extra_functions_package.plotings as pl

    xyzMesh = fg.create_mesh_XYZ(4, 4, 1, zMin=None)
    beam = milnor_Pol_u_v_any(xyzMesh, uOrder=2, vOrder=2, H=1)
    pl.plot_2D(np.abs(beam[:, :, 20]))
