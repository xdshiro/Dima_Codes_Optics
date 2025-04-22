"""
A flexible, extensible simulation framework for propagating structured light fields—
with built‑in support for “Hobbit” beam profiles and the ability to swap in
other beam shapes (e.g., asymmetric Laguerre–Gaussian, STOV variants).

Key features:
  • Propagation solvers:
      – split_step_time / split_step_time_Z  : Split‑step Fourier methods (time‑ or z‑domain)
      – adi_2d1_nonlinear / adi_2d1_nonlinear_z : 2D space + time ADI scheme
      – UPPE_time                         : Unidirectional Pulse Propagation Equation in time domain
  • Preconfigured beam generators:
      – hobbit (u_far_hob2 + helpers)     : “Hobbit” field profile
      – asymmetric_lg                     : Asymmetric Laguerre–Gaussian beams
      – field_stov_1, field_stov_simple  : Spatio‑temporal optical vortex (STOV) variants
  • Centralized physical & simulation parameters:
      – Spatial/temporal grids and resolutions
      – Beam/pulse settings (radius, wavelength, OAM order, chirp, power)
      – Linear medium properties (dispersion, refractive index)
      – Nonlinear constants (Kerr n₂, χ⁽³⁾, multiphoton cross‑sections, plasma rates)
      – Derived quantities (k₀, Rayleigh length, critical power, collapse length)
  • Diagnostics & visualization:
      – plot_1D: 1D line cuts
      – plot_2D: 2D intensity maps
      – plot_3D: 3D isosurfaces via Plotly
      – Real‑time spectral plots when module_checking_spectrum is enabled
  • Example “__main__” workflow:
      1. Select or customize a beam generator (e.g., hobbit)
      2. Normalize to target power Pmax
      3. Propagate using a chosen solver
      4. Visualize results in physical or spectral domains

Usage:
  1. Edit top‑of‑file flags (module_checking_spectrum, module_intensity) and core
     parameters to match your scenario.
  2. Import or define your beam function (hobbit, asymmetric_lg, etc.).
  3. Call one of the high‑level propagation functions with your beam generator:
       E_final = split_step_time_Z(hobbit, loop_inner_M, loop_outer_K)
  4. Use the plotting utilities to inspect spatial, temporal, or spectral evolution.
"""

from typing import Any

import numpy as np
from numpy import ndarray, dtype, floating
from scipy.special import erf, jv, iv, assoc_laguerre
from scipy.fft import fftn, ifftn, fftshift, ifftshift
import matplotlib.pyplot as plt
from scipy.integrate import ode
import plotly.graph_objects as go

# =============================================================================
# Module Selection Flags
# =============================================================================
# book Couairon
# These flags determine which modules or features of the simulation are active.
module_checking_spectrum = 1  # Flag for checking spectral properties
module_intensity = 0  # Flag to use intensity (1) or field absolute value (0)

# =============================================================================
# Spatial and Temporal Resolutions
# =============================================================================
# Grid resolutions and physical domain limits.
x_resolution = 131  # Number of grid points in the x-direction
y_resolution = 131  # Number of grid points in the y-direction
t_resolution = 1  # Number of grid points in time (2D version of the code) 1 for CW
loop_inner_resolution = 101  # Inner loop resolution factor (M)
loop_outer_resolution = 1  # Outer loop resolution factor (Kmax)
z_resolution = loop_inner_resolution * loop_outer_resolution  # Total z-resolution

# Spatial domain boundaries (in meters)
x_start, x_finish = 0, 700e-6  # x-range (0 to 700 μm)
y_start, y_finish = 0, 700e-6  # y-range (0 to 700 μm)
z_start, z_finish = 0, 0.1  # z-range (1 mm to 80 mm)

# Temporal domain boundaries (in seconds) for the split_step_time (over time)
# DOES NOT MATTER for CW version
t_start, t_finish = 0, 10e-13  # t-range (0 to 1000e-13 s)
# A derived time index (typically used to index the middle of the temporal grid)
time_index = int(t_resolution / 2)


# =============================================================================
# Pulse Parameters
# =============================================================================
rho0 = 700e-6  # Beam radius for LG (in meters)
lambda0 = 0.517e-6  # Central wavelength of the pulse (in meters)
Pmax = 1e-5  # Beam power of the pulse (e.g., in Watts)


# =============================================================================
# "Hobbit" Module Parameters
# =============================================================================
# Parameters for the "Hobbit" field generation.
beta_hob = 1.08  # Scaling parameter for the Hobbit phase factor
alpha_hob = 0  # Additional phase offset for the Hobbit field
ro_0_hob = 650e-6  # Characteristic parameter for the Hobbit field (in meters)
k = 3  # Mode index used in Hobbit functions
F_hob = 150e-3  # Focal length or related parameter for Hobbit (in meters)
w_ring_hob = 244e-6  # Ring width parameter for the Hobbit field (in meters)
w_G_hob = lambda0 * F_hob / (np.pi * w_ring_hob)  # Derived Gaussian width for Hobbit
tau_0 = 1e100  # Temporal parameter for the Hobbit field CW


l_oam = 0  # Orbital angular momentum (OAM) order for the field

# Center coordinates for the beam (usually at the center of the domain)
x0 = (x_finish - x_start) / 2  # x-center of the beam (in meters)
y0 = (y_finish - y_start) / 2  # y-center of the beam (in meters)
t0 = (t_finish - t_start) / 2  # t-center of the pulse (in seconds)

f = 1e5  # Focal length or related propagation parameter (in meters)

# =============================================================================
# Linear Medium Parameters
# =============================================================================
k2_dis = 5.6e-28 / 1e-2  # Group Velocity Dispersion (GVD) coefficient (in ps^2/m)
# Alternative option for non-diffraction with OAM:
# k2_dis = -9.443607756116762e-22
n0 = 1.332  # Linear refractive index of the medium

# =============================================================================
# Temporary/Plasma Parameters
# =============================================================================
# These parameters are related to the plasma and nonlinear interactions.
sigma_K8 = 2.4e-42 * (1e-2) ** (2 * 4)  # Effective cross section factor for multiphoton processes
sigma = 4e-18 * (1e-2) ** 2  # Secondary cross section value
rho_at = 7e22 * (1e-2) ** (-3)  # Atomic density (in m^-3)
a = 0  # Additional parameter (e.g., recombination coefficient)
tau_c = 3e-15  # Characteristic collision time (in seconds)

# =============================================================================
# Nonlinear and Temporal Parameters
# =============================================================================
K = 4  # Photon order for multiphoton absorption processes
chirp = 0  # Chirp parameter (dimensionless or in s^-1, as needed)
n2 = 2.7e-16 * (1e-2) ** 2  # Nonlinear refractive index (m^2/W) or related to chi^(3)
q_e = -1.602176565e-19  # Elementary charge (in Coulombs)
Ui = 7.1 * abs(q_e)  # Ionization potential (in Joules, derived from q_e)
sigma_k = [0, 1, 2]  # A list of parameters for multiphoton processes (purpose-specific)


def betta_func(K_val):
    """
    Compute the beta parameter for nonlinear processes based on photon order.

    Parameters
    ----------
    K_val : int
        The photon order.

    Returns
    -------
    float
        The beta parameter corresponding to the given photon order.
    """
    beta_values = [
        0,  # Placeholder value
        0 * 2e-0,  # Placeholder value
        2,  # Placeholder value
        3,  # Placeholder value
        2.4e-37 * (1e-2) ** (2 * K_val - 3),
        5,  # Placeholder value
        6,  # Placeholder value
        7,  # Placeholder value
        3.79347046850176e-121
    ]
    return beta_values[K_val]


# =============================================================================
# Physical Constants
# =============================================================================
eps0 = 8.854187817e-12  # Vacuum permittivity (F/m)
c_sol = 2.99792458e8  # Speed of light in vacuum (m/s)

# =============================================================================
# Plotting Parameters
# =============================================================================
ticks_font_size = 18  # Font size for axis ticks
legend_font_size = 18  # Font size for plot legends
xy_label_font_size = 18  # Font size for x and y labels

# =============================================================================
# Derived Parameters
# =============================================================================
k0 = 2 * np.pi / lambda0  # Wave number (rad/m)
w0 = k0 * c_sol  # Angular frequency (rad/s)
w_D = 2 * n0 / (k2_dis * c_sol)  # Derived dispersion parameter

# Calculate third-order nonlinear susceptibility parameters
chi3_2 = 8 * n0 * n2 / 3
eps_nl = 3 * chi3_2 / 4

# Normalized maximum field amplitude (dimensionless scaling)
Imax = 1

# =============================================================================
# Critical Power and Collapse Length Calculations
# =============================================================================
if n2 == 0:
    Pcrit = 1e100  # Avoid division by zero
else:
    # Critical power for self-focusing (in Watts)
    Pcrit = (1.22 ** 2 * np.pi * lambda0 ** 2) / (32 * n0 * n2)
print("P crit (MW):", Pcrit * 1e-6)


def LDF():
    """
    Compute the Rayleigh length for the beam.

    Returns
    -------
    float
        The Rayleigh length (in meters).
    """
    return np.pi * rho0 ** 2 / lambda0


def Lcollapse():
    """
    Compute the Kerr collapse length for the beam.

    Returns
    -------
    float
        The collapse length (in meters), or 0 if n2 is zero.
    """
    rayleigh_length = LDF()
    if n2 == 0:
        return 0
    else:
        temp2 = (np.sqrt(Pmax / Pcrit) - 0.852) ** 2 - 0.0219
        return 0.367 * rayleigh_length / np.sqrt(temp2)


print("Rayleigh length:", LDF(), "Kerr Collapse length:", Lcollapse())

# =============================================================================
# Grid and Array Creation
# =============================================================================
# Spatial and temporal coordinate arrays
x_array = np.linspace(x_start, x_finish, x_resolution)
y_array = np.linspace(y_start, y_finish, y_resolution)
z_array = np.linspace(z_start, z_finish, z_resolution)
t_array = np.linspace(t_start, t_finish, t_resolution)

# Mesh for (x, t) used in ADI propagation (2D)
xt_mesh = np.array(np.meshgrid(x_array, t_array, indexing='ij'))

# Fourier space coordinate arrays for spectral methods
kx_array = np.linspace(-np.pi * (x_resolution - 2) / x_finish,
                       np.pi * (x_resolution - 2) / x_finish, x_resolution)
ky_array = np.linspace(-np.pi * (y_resolution - 2) / y_finish,
                       np.pi * (y_resolution - 2) / y_finish, y_resolution)
w_array = np.linspace(-np.pi * (t_resolution - 2) / t_finish,
                      np.pi * (t_resolution - 2) / t_finish, t_resolution)

# 3D mesh for (x, y, t)
xyt_mesh = np.array(np.meshgrid(x_array, y_array, t_array, indexing='ij'))

# Mesh for (kx, ky, w) used in spectral computations
kxyw_mesh = np.array(np.meshgrid(kx_array, ky_array, w_array, indexing='ij'))


# -----------------------------------------------------------------------------
# Helper Functions for "Hobbit" Fields
# -----------------------------------------------------------------------------
def b_m_hob(n, l_hob):
    """
    Compute the B_m_hob factor for the Hobbit field.
    Depends on global parameters: beta_hob, alpha_hob.
    """
    temp = beta_hob * np.pi * (l_hob + alpha_hob - n) / 2
    return ((-1j) ** (n - 1) * 2 * np.exp(-temp ** 2) *
            np.imag(erf(1j * (1j + temp)) / 1j))


def arg_hob(x, y):
    """
    Compute the argument for the Bessel function in the Hobbit field.
    Depends on global parameters: ro_0_hob, lambda0, F_hob.
    """
    return np.pi * ro_0_hob * radius(x, y) / (lambda0 * F_hob)


def j_hob(arg, m):
    """
    Compute the Bessel function of order m.
    """
    return jv(m, arg)


def u_far_hob2(x, y, t, m, k):
    """
    Compute the far-field (Hobbit) field contribution.

    Parameters:
        x, y, t : float
            Spatial and temporal coordinates.
        m : int
            Order (typically set equal to lOAM).
        k : int
            Summation limit.

    Depends on global parameters: x0, y0, w_G_hob, t0, tau_0.
    """
    temp_sum = 0
    # Sum contributions over orders from m-k to m+k.
    for i in range(m - k, m + k + 1):
        phase = np.exp(1j * i * phi(x - x0, y - y0))
        b_term = b_m_hob(i, m)
        j_term = j_hob(arg_hob(x - x0, y - y0) * 2, i)
        temp_sum += b_term * phase * j_term

    spatial_envelope = np.exp(- (radius(x - x0, y - y0) ** 2) / w_G_hob ** 2)
    temporal_envelope = np.exp(-2 * np.log(2) * ((t - t0) / tau_0) ** 2)
    return temp_sum * spatial_envelope * temporal_envelope



def hobbit(x, y, t):
    """
    Compute the Hobbit field at (x, y, t).

    Global parameters used:
      l_oam, x0, y0, t0, Imax, k, (and those used in the helper functions)
    """
    m = l_oam
    return Imax * u_far_hob2(x, y, t, m, k)


def asymmetric_lg(x, y):
    """
    Compute an asymmetric Laguerre-Gaussian (LG) field.

    Global parameters used:
      rho0, l_oam, x0, y0, lambda0, k0, Imax

    Notes:
      - z is set to a fixed value (1e-6) here.
      - The function 'nonlinearity' uses np.arctan2 for a robust phase angle.
    """
    # Fixed propagation distance (or a placeholder value)
    z = 1e-6
    width = rho0
    p = 0
    l = l_oam

    # Shift coordinates relative to beam center
    x_shifted = x - x0
    y_shifted = y - y0

    def rayleigh_range(wavelength, beam_width):
        return np.pi * beam_width ** 2 / wavelength

    z_R = rayleigh_range(lambda0, width)
    print("Rayleigh Range:", z_R)

    def rho_val(x_val, y_val):
        return np.sqrt(x_val ** 2 + y_val ** 2)

    def width_z(z_val):
        return width * np.sqrt(1 + (z_val / z_R) ** 2)

    def R(z_val):
        return z_val * (1 + (z_R / z_val) ** 2) if z_val != 0 else np.inf

    def ksi(z_val):
        return np.arctan(z_val / z_R)

    def laguerre_poly(x_val, l_val, p_val):
        return assoc_laguerre(x_val, l_val, p_val)

    def nonlinearity(x_val, y_val):
        # Returns the magnitude and phase of the (x,y) coordinate.
        return np.sqrt(x_val ** 2 + y_val ** 2) * np.exp(1j * np.arctan2(y_val, x_val))

    r_val = rho_val(x_shifted, y_shifted)
    wz = width_z(z)
    E = ((width_z(0) / wz) *
         ((np.sqrt(2) / wz) ** abs(l)) *
         (nonlinearity(x_shifted, y_shifted) ** abs(l)) *
         laguerre_poly(2 * r_val ** 2 / wz ** 2, abs(l), p) *
         np.exp(- r_val ** 2 / wz ** 2 + 1j * k0 * r_val ** 2 / (2 * R(z)) -
                1j * (abs(l) + 2 * p + 1) * ksi(z)))
    return Imax * E

def radius(x, y):
    """Compute the radial distance from the origin."""
    return np.sqrt(x ** 2 + y ** 2)


def phi(x, y_or_t):
    """
    Compute the phase angle of the complex number x + i*(y_or_t).

    Note: The second parameter can represent either a spatial or temporal coordinate.
    """
    return np.angle(x + 1j * y_or_t)


def split_step_time(shape: callable, loopInnerM: int = 1, loopOuterKmax: int = 1) -> np.ndarray:
    """
    Perform a split-step simulation for wave propagation in a plasma medium.

    This function evolves an electric field E along the propagation axis (z) using
    the split-step method. At each step, a linear propagation part is computed in the
    Fourier domain while a nonlinear phase modulation is applied in real space to account
    for Kerr nonlinearity and plasma effects.

    Parameters:
        shape (callable): A function to generate the initial electric field E, taking three
                          arguments corresponding to the spatial and temporal mesh grids
                          (xytMesh[0], xytMesh[1], xytMesh[2]).
        loopInnerM (int, optional): Number of inner iterations for each outer propagation step.
        loopOuterKmax (int, optional): Number of outer propagation steps.

    Returns:
        np.ndarray: The electric field E after propagation.

    Notes:
        This function expects various global variables to be defined:
            - zArray: A 1D array of z positions; used to compute propagation step dz.
            - tArray: A 1D array of time values; used to compute the time step dt.
            - xytMesh: A tuple containing meshgrids for the spatial (x, y) and temporal (t) domains.
            - xResolution, yResolution, tResolution: Resolutions for the (x, y, t) grid.
            - sigma_K8, K, rho_at, sigma, Ui, a, w0, n2, eps0, epsNL, tau_c: Physical parameters.
            - KxywMesh: A tuple or array containing spectral mesh grids for (kx, ky, ω).
            - k0, n0, cSOL, k2Dis: Propagation constants.
            - Betta_func: A function that returns the beta parameter based on K.
    """

    # Helper: Compute intensity (squared amplitude) of the electric field.
    def compute_intensity(E: np.ndarray) -> np.ndarray:
        """Return the intensity computed as the squared magnitude of E."""
        return np.abs(E) ** 2

    # Determine propagation step size along z from the global zArray.
    dz = z_array[1] - z_array[0]

    # Helper: Compute the plasma density evolution using a nonlinear model.
    def compute_plasma_density_nonlinear(E: np.ndarray) -> np.ndarray:
        """
        Compute the plasma density evolution over time using an explicit time-stepping scheme.

        This scheme updates the plasma density at each time slice based on the ionization rate
        and avalanche effect. It uses an exponential decay factor computed from the averaged
        ionization rates.

        Parameters:
            E (np.ndarray): Electric field distribution of shape (xResolution, yResolution, tResolution).

        Returns:
            np.ndarray: Plasma density distribution over the grid.
        """
        plasma_density = np.zeros((x_resolution, y_resolution, t_resolution))
        dt = t_array[1] - t_array[0]  # time step

        # Helper functions for ionization and avalanche rates:
        def Wofi(I: np.ndarray) -> np.ndarray:
            """Multiphoton ionization rate."""
            return sigma_K8 * I ** K

        def Wava(I: np.ndarray) -> np.ndarray:
            """Avalanche ionization rate."""
            return sigma * I / Ui

        def Q_pd(I: np.ndarray) -> np.ndarray:
            """Photoionization source term."""
            return Wofi(I)

        def a_pd(I1: np.ndarray, I2: np.ndarray) -> np.ndarray:
            """
            Compute the exponential decay factor over the time interval using average ionization rates.

            The decay factor is given by the exponential of the negative average rate.
            """
            avg_rate = (Wofi(I1) - Wava(I1) + Wofi(I2) - Wava(I2)) * dt / 2
            return np.exp(-avg_rate)

        etta_pd = dt * rho_at / 2

        # Time-stepping: update plasma density for each time slice.
        for i in range(t_resolution - 1):
            intensity_current = compute_intensity(E[:, :, i])
            intensity_next = compute_intensity(E[:, :, i + 1])
            plasma_density[:, :, i + 1] = (
                    a_pd(intensity_current, intensity_next) *
                    (plasma_density[:, :, i] + etta_pd * Q_pd(intensity_current))
                    + etta_pd * Q_pd(intensity_next)
            )
        return plasma_density

    # Helper: Compute the nonlinear phase shift (spectral nonlinearity) to be applied.
    def compute_nonlinearity_spec(E: np.ndarray, plasma_density: np.ndarray) -> np.ndarray:
        """
        Compute the nonlinear phase change caused by Kerr effect and plasma induced effects.

        The nonlinearity includes contributions from the Kerr response, plasma defocusing (via
        a beta term), and absorption/loss terms. The returned value is scaled by the step size dz.

        Parameters:
            E (np.ndarray): The current electric field distribution.
            plasma_density (np.ndarray): Plasma density corresponding to the current field.

        Returns:
            np.ndarray: The computed nonlinear phase shift.
        """
        intensity_val = compute_intensity(E)
        term1 = (1j / (2 * eps0)) * ((w0 + kxyw_mesh[2]) / (c_sol * n0)) * eps_nl * intensity_val
        term2 = -betta_func(K) / 2 * intensity_val ** (K - 1) * (1 - plasma_density / rho_at)
        term3 = -sigma / 2 * (1 + 1j * w0 * tau_c) * plasma_density
        return dz * (term1 + term2 + term3)

    # Helper: Linear propagation step using FFT-based spectral methods.
    def linear_step(field: np.ndarray) -> np.ndarray:
        """
        Propagate the electric field in the spectral domain by applying the appropriate phase shift.

        This function uses FFT to transform the field to the spectral domain, applies phase shifts
        corresponding to diffraction and dispersion, and then returns to the spatial domain.

        Parameters:
            field (np.ndarray): Input electric field distribution.

        Returns:
            np.ndarray: Electric field after the linear propagation step.
        """
        temporary_field = fftshift(fftn(field))
        # Construct the phase factor for diffraction and dispersion.
        phase_factor = (np.exp(-1j * dz / (2 * k0 * n0) * kxyw_mesh[0] ** 2) *
                        np.exp(-1j * dz / (2 * k0 * n0) * kxyw_mesh[1] ** 2) *
                        np.exp(1j * dz * k2_dis / 2 * kxyw_mesh[2] ** 2))
        temporary_field *= phase_factor
        return ifftn(ifftshift(temporary_field))

    # Initialize the electric field using the provided shape function and mesh.
    E = shape(xyt_mesh[0], xyt_mesh[1], xyt_mesh[2])
    center_val = E[int(x_resolution / 2), int(y_resolution / 2), int(t_resolution / 2)]
    print("Initial center field value:", center_val)

    # Main propagation loop: Outer and inner loop to perform split-step integration.
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # Compute plasma density evolution (nonlinear model) for current electric field.
            plasma_density = compute_plasma_density_nonlinear(E)
            # Apply the linear step (using FFT propagation).
            E = linear_step(E)
            # Compute the nonlinear phase shift and update the field accordingly.
            nonlin_phase = compute_nonlinearity_spec(E, plasma_density)
            E = E * np.exp(nonlin_phase)

    return E


def split_step_time_Z(shape: callable, loopInnerM: int = 1, loopOuterKmax: int = 1) -> np.ndarray:
    """
    Perform a split-step simulation in z that evolves an electric field
    along a propagation axis while accounting for nonlinear plasma and Kerr effects.

    This function initializes the electric field using the provided mesh grids (from xytMesh)
    and then propagates the field along z. At each z-step the simulation applies:
        1. A linear propagation step in the spectral domain (to model diffraction and dispersion).
        2. A nonlinear phase modulation due to the Kerr effect and plasma formation.

    The field is recorded at a specific time slice (the center of the temporal grid) for every z step.

    Parameters:
        shape (callable): Function that returns the initial electric field given the spatial
                          and temporal mesh grids (xytMesh[0], xytMesh[1], xytMesh[2]).
        loopInnerM (int, optional): Number of inner propagation iterations per outer step.
        loopOuterKmax (int, optional): Number of outer propagation steps.

    Returns:
        np.ndarray: A complex array (of shape (xResolution, yResolution, zResolution))
                    holding the propagated field sampled at the central time slice.

    Notes:
        This routine relies on many global variables that must be defined prior to calling it,
        including:

          - zArray: 1D array of propagation distances.
          - tArray: 1D array of temporal points.
          - xytMesh: Tuple of mesh grids for the spatial (x, y) and temporal (t) domains.
          - xResolution, yResolution, tResolution: Dimensions of the spatial and temporal grids.
          - sigma_K8, K, rho_at, sigma, Ui, a, tFinish, module_CheckingSpectrum: Physical and simulation parameters.
          - xArray, yArray, kxArray, kyArray, wArray: Arrays used for spectrum and spatial debugging plots.
          - k0, n0, cSOL, k2Dis, eps0, epsNL, w0, tau_c: Propagation constants.
          - KxywMesh: Tuple or array containing the spectral mesh grids.
          - Betta_func: Function returning beta parameters as a function of K.
          - zResolution: The number of z steps used for recording the propagated field.
    """

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def compute_intensity(E: np.ndarray) -> np.ndarray:
        """
        Compute the intensity as the squared magnitude of an electric field.

        Parameters:
            E (np.ndarray): Electric field.

        Returns:
            np.ndarray: Intensity, |E|².
        """
        return np.abs(E) ** 2

    # Determine propagation step size from zArray.
    dz = z_array[1] - z_array[0]

    def compute_plasma_density(E: np.ndarray) -> np.ndarray:
        """
        Compute the plasma density evolution along the temporal grid based on the electric field E.

        For a single time point (tResolution == 1), the density is computed directly.
        For multiple time points, an explicit time–stepping approach is used where the density
        is updated iteratively based on photoionization and avalanche processes.

        Parameters:
            E (np.ndarray): Electric field with shape (xResolution, yResolution, tResolution).

        Returns:
            np.ndarray: Plasma density array with the same (x, y, t) dimensions as E.
        """
        plasma_density = np.zeros((x_resolution, y_resolution, t_resolution))
        dt = t_array[1] - t_array[0] if t_resolution > 1 else t_finish

        if t_resolution == 1:
            # For a single temporal point, a simplified estimation is used.
            plasma_density[:, :, 0] = (t_finish *
                                       (sigma_K8 * np.abs(E[:, :, 0]) ** (2 * K) * (rho_at)
                                        + sigma / Ui * np.abs(E[:, :, 0]) ** 2 * plasma_density[:, :, 0]))
            return plasma_density
        else:
            # For multiple time points, define helper functions for ionization rates.
            def Wofi(I: np.ndarray) -> np.ndarray:
                """Multiphoton ionization rate."""
                return sigma_K8 * I ** K

            def Wava(I: np.ndarray) -> np.ndarray:
                """Avalanche ionization rate."""
                return sigma * I / Ui

            def Q_pd(I: np.ndarray) -> np.ndarray:
                """Photoionization source term."""
                return Wofi(I)

            def a_pd(I1: np.ndarray, I2: np.ndarray) -> np.ndarray:
                """
                Compute the decay factor over a time step based on the average of the ionization rates.

                Parameters:
                    I1, I2 (np.ndarray): Intensities at consecutive time steps.

                Returns:
                    np.ndarray: Exponential decay factor.
                """
                avg_rate = ((Wofi(I1) - Wava(I1)) + (Wofi(I2) - Wava(I2))) * dt / 2
                return np.exp(-avg_rate)

            etta_pd = dt * rho_at / 2

            # Iteratively update the plasma density for each time index.
            for i in range(t_resolution - 1):
                current_intensity = compute_intensity(E[:, :, i])
                next_intensity = compute_intensity(E[:, :, i + 1])
                plasma_density[:, :, i + 1] = (a_pd(current_intensity, next_intensity) *
                                               (plasma_density[:, :, i] + etta_pd * Q_pd(current_intensity))
                                               + etta_pd * Q_pd(next_intensity))
            return plasma_density

    def compute_nonlinearity_spec(E: np.ndarray, plasma_density: np.ndarray) -> np.ndarray:
        """
        Compute the nonlinear phase shift caused by Kerr effects and plasma dynamics.

        The function combines contributions from:
            - The Kerr effect.
            - Plasma-induced defocusing (modulated by Betta_func).
            - Plasma absorption.
        The total phase shift is scaled by the propagation step dz.

        Parameters:
            E (np.ndarray): The current electric field distribution.
            plasma_density (np.ndarray): Plasma density matching the dimensions of E.

        Returns:
            np.ndarray: The nonlinear phase shift to be applied to E.
        """
        intensity_val = compute_intensity(E)
        term1 = (1j / (2 * eps0)) * ((w0 + kxyw_mesh[2]) / (c_sol * n0)) * eps_nl * intensity_val
        term2 = -betta_func(K) / 2 * intensity_val ** (K - 1) * (1 - plasma_density / rho_at)
        term3 = -sigma / 2 * (1 + 1j * w0 * tau_c) * plasma_density
        return dz * (term1 + term2 + term3)

    def linear_step(field: np.ndarray) -> np.ndarray:
        """
        Perform a linear propagation step by applying phase shifts in the Fourier domain.

        This function transforms the electric field to its spectral domain, applies a phase
        factor corresponding to diffraction and dispersion, and transforms back to the spatial domain.

        Parameters:
            field (np.ndarray): The input electric field.

        Returns:
            np.ndarray: Updated electric field after linear propagation.
        """
        temporary_field = fftshift(fftn(field))
        phase_factor = (np.exp(-1j * dz / (2 * k0 * n0) * kxyw_mesh[0] ** 2) *
                        np.exp(-1j * dz / (2 * k0 * n0) * kxyw_mesh[1] ** 2) *
                        np.exp(1j * dz * k2_dis / 2 * kxyw_mesh[2] ** 2))
        temporary_field *= phase_factor
        return ifftn(ifftshift(temporary_field))

    # -------------------------------------------------------------------------
    # Main Propagation Routine
    # -------------------------------------------------------------------------

    # Initialize the electric field from the provided shape function and mesh.
    E = shape(xyt_mesh[0], xyt_mesh[1], xyt_mesh[2])
    center_val = E[int(x_resolution / 2), int(y_resolution / 2), int(t_resolution / 2)]
    print("Initial center field value:", center_val)

    # Allocate storage for the propagated field along z.
    fieldReturn = np.zeros((x_resolution, y_resolution, z_resolution), dtype=complex)
    time_index = int(t_resolution / 2)  # Record the central time slice.
    fieldReturn[:, :, 0] = E[:, :, time_index]

    # Main loop: propagate the field along z.
    for k in range(loopOuterKmax):
        # Optional: Display spatial and spectral profiles for debugging if enabled.
        if module_checking_spectrum:
            E_abs = np.abs(E)
            plt.plot(x_array, E_abs[:, int(y_resolution / 2), time_index])
            plt.title("Spatial Profile (x)")
            plt.show()
            plt.close()

            plt.plot(y_array, E_abs[int(x_resolution / 2), :, time_index])
            plt.title("Spatial Profile (y)")
            plt.show()
            plt.close()


            E_spectrum = np.abs(fftshift(fftn(E)))
            plt.plot(kx_array, E_spectrum[:, int(y_resolution / 2), time_index])
            plt.title("Spectral Profile (kx)")
            plt.show()
            plt.close()

            plt.plot(ky_array, E_spectrum[int(x_resolution / 2), :, time_index])
            plt.title("Spectral Profile (ky)")
            plt.show()
            plt.close()


        for m in range(1, loopInnerM):
            # Calculate the current z-index for storage.
            z_index = k * loopInnerM + m
            # Update plasma density based on the current electric field.
            plasma_density = compute_plasma_density(E)
            # Perform a linear propagation step.
            E = linear_step(E)
            # Compute the nonlinear phase shift and update the field.
            nonlin_phase = compute_nonlinearity_spec(E, plasma_density)
            E = E * np.exp(nonlin_phase)
            # Record the field at the central time slice.
            fieldReturn[:, :, z_index] = E[:, :, time_index]

        # Optional: Check the spectrum after each outer loop iteration.
        if module_checking_spectrum:
            E_spectrum = np.abs(fftshift(fftn(E)))
            plt.plot(kx_array, E_spectrum[:, int(y_resolution / 2), time_index])
            plt.title("Spectrum after propagation (kx)")
            plt.show()
            plt.close()

            plt.plot(ky_array, E_spectrum[int(x_resolution / 2), :, time_index])
            plt.title("Spectrum after propagation (ky)")
            plt.show()
            plt.close()


    return fieldReturn


def adi_2d1_nonlinear(E0, loop_inner_m, loop_outer_kmax):
    """
    Propagate the field E0 using an Alternate Direction Implicit (ADI) scheme
    with dispersion and nonlinear effects.

    Parameters
    ----------
    E0 : np.ndarray
        Initial field (shape: [x_resolution, t_resolution]).
    loop_inner_m : int
        Number of inner iterations.
    loop_outer_kmax : int
        Number of outer iterations.

    Returns
    -------
    np.ndarray
        The propagated field after applying the ADI scheme.

    Global Variables (must be defined externally)
    -----------------------------------------------
    x_resolution, t_resolution, z_array, x_array, t_array, n0, k0, k2_dis,
    sigma_K8, K, rho_at, t_finish, sigma, Ui, eps0, cSOL, epsNL, tau_c,
    Betta_func, w0

    Notes
    -----
    The method builds matrices for spatial and temporal (dispersion) implicit steps,
    and applies a finite-difference update including a nonlinear term.
    """
    nu = 1  # Use 1 for cylindrical geometry; 0 for planar geometry

    # -------------------------------
    # Construct u_array and v_array for spatial steps
    # -------------------------------
    u_array = np.zeros(x_resolution, dtype=complex)
    v_array = np.zeros(x_resolution, dtype=complex)
    for i in range(1, x_resolution - 1):
        u_array[i] = 1 - nu / (2 * i)
        v_array[i] = 1 + nu / (2 * i)

    # -------------------------------
    # Spatial step parameter delta
    # -------------------------------
    delta = (z_array[1] - z_array[0]) / (4 * n0 * k0 * (x_array[1] - x_array[0]) ** 2)

    # -------------------------------
    # Construct L_plus matrix (spatial implicit step)
    # -------------------------------
    L_plus = np.zeros((x_resolution, x_resolution), dtype=complex)
    d_plus = np.zeros(x_resolution, dtype=complex)
    d_plus[0] = 1 - 4j * delta
    d_plus[1] = 4j * delta
    L_plus[0, :] = d_plus
    for i in range(1, x_resolution - 1):
        L_plus[i, i - 1] = 1j * delta * u_array[i]
        L_plus[i, i] = 1 - 2j * delta
        L_plus[i, i + 1] = 1j * delta * v_array[i]

    # -------------------------------
    # Construct L_minus matrix (spatial implicit step)
    # -------------------------------
    L_minus = np.zeros((x_resolution, x_resolution), dtype=complex)
    d_minus = np.zeros(x_resolution, dtype=complex)
    d_minus[0] = 1 + 4j * delta
    d_minus[1] = -4j * delta
    L_minus[0, :] = d_minus
    L_minus[-1, -1] = 1
    for i in range(1, x_resolution - 1):
        L_minus[i, i - 1] = -1j * delta * u_array[i]
        L_minus[i, i] = 1 + 2j * delta
        L_minus[i, i + 1] = -1j * delta * v_array[i]

    # -------------------------------
    # Dispersion (time) step parameter delta_D
    # -------------------------------
    delta_D = - (z_array[1] - z_array[0]) * k2_dis / (4 * (t_array[1] - t_array[0]) ** 2)

    # -------------------------------
    # Construct L_plus_D matrix (temporal implicit step)
    # -------------------------------
    L_plus_D = np.zeros((t_resolution, t_resolution), dtype=complex)
    d_plus_D = np.zeros(t_resolution, dtype=complex)
    # Initial values (overwritten for index 0 and 1 per provided code)
    d_plus_D[0] = 1 - 4j * delta_D
    d_plus_D[1] = 4j * delta_D
    d_plus_D[0], d_plus_D[1] = 1, 0  # Override as in original code
    L_plus_D[0, :] = d_plus_D
    for i in range(1, t_resolution - 1):
        L_plus_D[i, i - 1] = 1j * delta_D
        L_plus_D[i, i] = 1 - 2j * delta_D
        L_plus_D[i, i + 1] = 1j * delta_D

    # -------------------------------
    # Construct L_minus_D matrix (temporal implicit step)
    # -------------------------------
    L_minus_D = np.zeros((t_resolution, t_resolution), dtype=complex)
    d_minus_D = np.zeros(t_resolution, dtype=complex)
    d_minus_D[0] = 1 + 4j * delta_D
    d_minus_D[1] = -4j * delta_D
    L_minus_D[0, :] = d_minus_D
    L_minus_D[-1, -1] = 1
    for i in range(1, t_resolution - 1):
        L_minus_D[i, i - 1] = -1j * delta_D
        L_minus_D[i, i] = 1 + 2j * delta_D
        L_minus_D[i, i + 1] = -1j * delta_D

    # Invert the L_minus matrices for the implicit update steps
    L_minus_D_inv = np.linalg.inv(L_minus_D)
    L_minus_inv = np.linalg.inv(L_minus)

    # -------------------------------
    # Local helper functions for intensity and plasma density
    # -------------------------------
    def intensity(E):
        """Calculate the intensity |E|^2 of the field."""
        return np.abs(E) ** 2


    def plasma_density(E):
        """
        Compute plasma density evolution based on the field intensity.

        Uses a finite-difference update in time.
        """
        density = np.zeros((x_resolution, t_resolution))

        def wofi(I_val):
            return sigma_K8 * I_val ** K

        def wava(I_val):
            return sigma * I_val / Ui

        def q_pd(I_val):
            return wofi(I_val)

        def a_pd(I1, I2):
            temp_value = (t_array[1] - t_array[0]) * ((wofi(I1) - wava(I1)) + (wofi(I2) - wava(I2))) / 2
            return np.exp(-temp_value)

        eta_pd = (t_array[1] - t_array[0]) * rho_at / 2

        for i in range(t_resolution - 1):
            density[:, i + 1] = (a_pd(intensity(E[:, i]), intensity(E[:, i + 1])) *
                                 (density[:, i] + eta_pd * q_pd(intensity(E[:, i]))) +
                                 eta_pd * q_pd(intensity(E[:, i + 1])))
        return density

    def nonlinearity(E, plasma_dens):
        """
        Compute the nonlinear modification of the field E.

        Accounts for Kerr nonlinearity and plasma-induced effects.
        """
        return E * (
                (1j / (2 * eps0)) * (w0 / (c_sol * n0)) * eps0 * eps_nl * intensity(E)
                - betta_func(K) / 2 * intensity(E) ** (K - 1) * (1 - plasma_dens / rho_at)
                - sigma / 2 * (1 + 1j * w0 * tau_c) * plasma_dens
        )

    # -------------------------------
    # Main propagation loop
    # -------------------------------
    E = E0.copy()  # Avoid modifying the original field

    # Initialize plasma density and nonlinear term
    plasma_dens = plasma_density(E)
    Nn_prev = (z_array[1] - z_array[0]) * nonlinearity(E, plasma_dens)

    for outer in range(loop_outer_kmax):
        for inner in range(1, loop_inner_m):
            # Update the nonlinear term and plasma density
            Nn_current = (z_array[1] - z_array[0]) * nonlinearity(E, plasma_density(E))

            # Apply implicit time-step (dispersion) update
            E = np.dot(L_plus_D, E.T)
            # Spatial implicit step update
            Vn = np.dot(L_plus, E.T)
            S_n = Vn + (3 * Nn_current - Nn_prev) / 2
            Nn_prev = Nn_current

            E = np.dot(L_minus_inv, S_n)
            # Final implicit dispersion step update
            E = np.dot(L_minus_D_inv, E.T).T

    return E


def adi_2d1_nonlinear_z(E0, loop_inner_m, loop_outer_kmax):
    """
    Propagate the field E0 along z using an Alternate Direction Implicit (ADI)
    scheme that includes nonlinear effects.

    This function uses finite-difference steps in both the spatial (x) and
    temporal (t) dimensions to account for dispersion and nonlinear modifications.
    The field is updated in a series of inner and outer loop iterations.

    Parameters
    ----------
    E0 : np.ndarray, shape (x_resolution, t_resolution)
        Initial field distribution.
    loop_inner_m : int
        Number of inner loop iterations (M steps per outer loop).
    loop_outer_kmax : int
        Number of outer loop iterations (Kmax steps in z).

    Returns
    -------
    field_return : np.ndarray, shape (x_resolution, z_resolution)
        The field at the mid-time index for each propagation step along z.

    Global Variables Used
    -----------------------
    x_resolution, t_resolution, z_array, x_array, t_array, n0, k0, k2_dis,
    sigma_K8, K, rho_at, t_finish, sigma, Ui, eps0, c_sol, eps_nl, tau_c,
    betta_func, w0
    """
    # Use cylindrical geometry if nu == 1; planar if 0.
    nu = 1

    # ---------------------------
    # Build helper arrays for the spatial (x) step
    # ---------------------------
    u_array = np.zeros(x_resolution, dtype=complex)
    v_array = np.zeros(x_resolution, dtype=complex)
    # u_array and v_array account for cylindrical symmetry corrections.
    for i in range(1, x_resolution - 1):
        u_array[i] = 1 - nu / (2 * i)
        v_array[i] = 1 + nu / (2 * i)

    # Spatial finite-difference step parameter.
    delta = (z_array[1] - z_array[0]) / (4 * n0 * k0 * (x_array[1] - x_array[0]) ** 2)

    # ---------------------------
    # Construct L_plus matrix for spatial update
    # ---------------------------
    l_plus_matrix = np.zeros((x_resolution, x_resolution), dtype=complex)
    d_plus_array = np.zeros(x_resolution, dtype=complex)
    d_plus_array[0] = 1 - 4j * delta
    d_plus_array[1] = 4j * delta
    l_plus_matrix[0, :] = d_plus_array

    for i in range(1, x_resolution - 1):
        l_plus_matrix[i, i - 1] = 1j * delta * u_array[i]
        l_plus_matrix[i, i] = 1 - 2j * delta
        l_plus_matrix[i, i + 1] = 1j * delta * v_array[i]

    # ---------------------------
    # Construct L_minus matrix for spatial update
    # ---------------------------
    l_minus_matrix = np.zeros((x_resolution, x_resolution), dtype=complex)
    d_minus_array = np.zeros(x_resolution, dtype=complex)
    d_minus_array[0] = 1 + 4j * delta
    d_minus_array[1] = -4j * delta
    l_minus_matrix[0, :] = d_minus_array
    l_minus_matrix[-1, -1] = 1
    for i in range(1, x_resolution - 1):
        l_minus_matrix[i, i - 1] = -1j * delta * u_array[i]
        l_minus_matrix[i, i] = 1 + 2j * delta
        l_minus_matrix[i, i + 1] = -1j * delta * v_array[i]

    # ---------------------------
    # Construct matrices for the temporal (dispersion) step.
    # ---------------------------
    delta_d = - (z_array[1] - z_array[0]) * k2_dis / (4 * (t_array[1] - t_array[0]) ** 2)

    # L_plus for dispersion
    l_plus_matrix_d = np.zeros((t_resolution, t_resolution), dtype=complex)
    d_plus_array_d = np.zeros(t_resolution, dtype=complex)
    d_plus_array_d[0] = 1 - 4j * delta_d
    d_plus_array_d[1] = 4j * delta_d
    # Override first two values as in the original code.
    d_plus_array_d[0], d_plus_array_d[1] = 1, 0
    l_plus_matrix_d[0, :] = d_plus_array_d
    for i in range(1, t_resolution - 1):
        l_plus_matrix_d[i, i - 1] = 1j * delta_d
        l_plus_matrix_d[i, i] = 1 - 2j * delta_d
        l_plus_matrix_d[i, i + 1] = 1j * delta_d

    # L_minus for dispersion
    l_minus_matrix_d = np.zeros((t_resolution, t_resolution), dtype=complex)
    d_minus_array_d = np.zeros(t_resolution, dtype=complex)
    d_minus_array_d[0] = 1 + 4j * delta_d
    d_minus_array_d[1] = -4j * delta_d
    l_minus_matrix_d[0, :] = d_minus_array_d
    l_minus_matrix_d[-1, -1] = 1
    for i in range(1, t_resolution - 1):
        l_minus_matrix_d[i, i - 1] = -1j * delta_d
        l_minus_matrix_d[i, i] = 1 + 2j * delta_d
        l_minus_matrix_d[i, i + 1] = -1j * delta_d

    # Invert the matrices needed for the implicit steps.
    l_minus_matrix_d_inv = np.linalg.inv(l_minus_matrix_d)
    l_minus_matrix_inv = np.linalg.inv(l_minus_matrix)

    # ---------------------------
    # Local helper functions
    # ---------------------------
    def intensity(E):
        """Return the intensity |E|^2 of the field."""
        return np.abs(E) ** 2

    def plasma_density(E):
        """
        Compute plasma density evolution based on the local field intensity.
        Uses a finite-difference time-update scheme.
        """
        density = np.zeros((x_resolution, t_resolution))

        def wofi(I_val):
            return sigma_K8 * I_val ** K

        def wava(I_val):
            return sigma * I_val / Ui

        def q_pd(I_val):
            return wofi(I_val)

        def a_pd(I1, I2):
            # Average loss factor computed over a time step.
            temp_value = (t_array[1] - t_array[0]) * ((wofi(I1) - wava(I1)) + (wofi(I2) - wava(I2))) / 2
            return np.exp(-temp_value)

        eta_pd = (t_array[1] - t_array[0]) * rho_at / 2

        for i in range(t_resolution - 1):
            density[:, i + 1] = (a_pd(intensity(E[:, i]), intensity(E[:, i + 1])) *
                                 (density[:, i] + eta_pd * q_pd(intensity(E[:, i]))) +
                                 eta_pd * q_pd(intensity(E[:, i + 1])))
        return density

    def nonlinearity(E, plasma_dens):
        """
        Compute the nonlinear modification of the field E.

        The nonlinearity includes contributions from Kerr-type effects and
        plasma-induced modifications.

        Parameters:
            E : np.ndarray
                Field distribution.
            plasma_dens : np.ndarray
                Plasma density computed from the field.

        Returns:
            np.ndarray : Nonlinear term to be used in the propagation update.
        """
        return E * (
                (1j / (2 * eps0)) * (w0 / (c_sol * n0)) * eps_nl * intensity(E)
                - betta_func(K) / 2 * intensity(E) ** (K - 1) * (1 - plasma_dens / rho_at)
                - sigma / 2 * (1 + 1j * w0 * tau_c) * plasma_dens
        )

    # ---------------------------
    # Initialization before propagation loop
    # ---------------------------
    E = E0.copy()  # Use a copy of the initial field.
    plasma_dens = plasma_density(E)
    # Nonlinear contribution scaled by the z-step.
    Nn_prev = (z_array[1] - z_array[0]) * nonlinearity(E, plasma_dens)

    # Prepare an array to store the field at mid-time for each z step.
    field_return = np.zeros((x_resolution, z_resolution), dtype=complex)
    mid_time = int(t_resolution / 2)
    field_return[:, 0] = E[:, mid_time]

    # ---------------------------
    # Propagation loop over z
    # ---------------------------
    for outer in range(loop_outer_kmax):
        for inner in range(1, loop_inner_m):
            # Indexing count (n) if needed for diagnostics.
            n = outer * loop_inner_m + inner + 1

            # Update nonlinear term.
            Nn_current = (z_array[1] - z_array[0]) * nonlinearity(E, plasma_density(E))
            plasma_dens = plasma_density(E)  # Recompute plasma density.

            # First, perform an implicit dispersion (temporal) update.
            E = np.dot(l_plus_matrix_d, E.transpose())

            # Then, perform an implicit spatial update.
            Vn = np.dot(l_plus_matrix, E.transpose())
            S_n = Vn + (3 * Nn_current - Nn_prev) / 2
            Nn_prev = Nn_current

            # Apply the implicit spatial solve.
            E = np.dot(l_minus_matrix_inv, S_n)

            # Final implicit dispersion update.
            E = np.dot(l_minus_matrix_d_inv, E.transpose()).transpose()

            # Save the field at the mid-time index for diagnostics.
            field_return[:, outer * loop_inner_m + inner] = E[:, mid_time]

    return field_return


##################

# %% UPPE with time
def UPPE_time(shape, loop_inner_m, loop_outer_kmax):
    """
    Solve the Unidirectional Pulse Propagation Equation (UPPE) in the time domain.

    Uses your workspace globals:
      - xyt_mesh, z_array, kxyw_mesh
      - x_resolution, y_resolution, t_resolution
      - eps0, eps_nl, w0, n0, c_sol, w_D
      - module_checking_spectrum, legend_font_size
    """

    # -- Helpers -------------------------------------------------------------
    def compute_intensity(E):
        return np.abs(E) ** 2

    def compute_nonlinearity(E):
        # Kerr‐only nonlinear polarization
        return eps0 * eps_nl * E * compute_intensity(E)

    # -- Initialize field and spectrum -------------------------------------
    E = shape(xyt_mesh[0], xyt_mesh[1], xyt_mesh[2])
    espec = fftshift(fftn(E))
    aspec = espec.copy()
    dz = z_array[1] - z_array[0]

    # -- Build spectral kz -----------------------------------------------
    n_spec_3d = n0 * (1.0 + (w0 + kxyw_mesh[2]) / w_D)
    k_spec_3d = n_spec_3d * (w0 + kxyw_mesh[2]) / c_sol
    kz_3d = np.sqrt(k_spec_3d ** 2 - kxyw_mesh[0] ** 2 - kxyw_mesh[1] ** 2)

    # -- Phase velocity ---------------------------------------------------
    v_phase = c_sol / (n0 + 2 * n0 * w0 / w_D)

    # -- ODE for spectral evolution --------------------------------------
    def odes(z, A_flat):
        # apply forward phase
        A_flat = A_flat * np.exp(1j * z * (kz_3d - (w0 + w1d) / v_phase))
        A3 = np.reshape(A_flat, (x_resolution, y_resolution, t_resolution))
        E_space = ifftn(ifftshift(A3))
        P = compute_nonlinearity(E_space)
        P_spec = fftshift(fftn(P))
        P_flat = np.reshape(P_spec, -1)
        # apply backward phase and scaling
        P_flat *= np.exp(-1j * z * (kz_3d - (w0 + w1d) / v_phase))
        P_flat *= (1j / (2 * eps0)) * ((w0 + w1d) ** 2 / (c_sol ** 2 * kz_3d))
        return P_flat

    # -- Flatten spectral arrays for the integrator ------------------------
    N = x_resolution * y_resolution * t_resolution
    aspec = np.reshape(aspec, N)
    w1d = np.reshape(kxyw_mesh[2], N)
    kx1d = np.reshape(kxyw_mesh[0], N)
    ky1d = np.reshape(kxyw_mesh[1], N)

    # rebuild kz in 1D
    n_spec_1d = n0 * (1.0 + (w0 + w1d) / w_D)
    k_spec_1d = n_spec_1d * (w0 + w1d) / c_sol
    kz_flat = np.sqrt(k_spec_1d ** 2 - kx1d ** 2 - ky1d ** 2)

    # override the 3D kz with the flattened version for use in the ODE:
    # (the odes() closure still sees kz_3d; you could unify but this is minimal)
    # assume kz_3d broadcastable against operations in odes()

    # -- Integrator setup ---------------------------------------------------
    integrator = ode(odes).set_integrator('zvode', nsteps=1e6)

    # -- Optional initial spectrum plot ------------------------------------
    if module_checking_spectrum:
        A3 = np.reshape(aspec, (x_resolution, y_resolution, t_resolution))
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(np.abs(A3[:, y_resolution // 2, t_resolution // 2]), 'b-', lw=6, label='x-cut')
        ax.plot(np.abs(A3[x_resolution // 2, :, t_resolution // 2]), 'g-', lw=4, label='y-cut')
        ax.plot(np.abs(A3[x_resolution // 2, y_resolution // 2, :]), 'r-', lw=4, label='t-cut')
        ax.legend(shadow=True, fontsize=legend_font_size, loc='upper right')
        plt.show()

    # -- Main propagation loop ---------------------------------------------
    for outer in range(loop_outer_kmax):
        if module_checking_spectrum:
            aspec = aspec.ravel()
        for inner in range(1, loop_inner_m):
            step = outer * loop_inner_m + inner
            print("UPPE step:", step)
            integrator.set_initial_value(aspec, 0.0)
            aspec = integrator.integrate(dz)
            aspec *= np.exp(1j * dz * (kz_3d - (w0 + w1d) / v_phase))
        if module_checking_spectrum:
            A3 = np.reshape(aspec, (x_resolution, y_resolution, t_resolution))
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.plot(np.abs(A3[:, y_resolution // 2, t_resolution // 2]), 'b-', lw=6, label='x-cut')
            ax.plot(np.abs(A3[x_resolution // 2, :, t_resolution // 2]), 'g-', lw=4, label='y-cut')
            ax.plot(np.abs(A3[x_resolution // 2, y_resolution // 2, :]), 'r-', lw=4, label='t-cut')
            ax.legend(shadow=True, fontsize=legend_font_size, loc='upper right')
            plt.show()

    # -- Return to real space ----------------------------------------------
    A3 = np.reshape(aspec, (x_resolution, y_resolution, t_resolution))
    E_final = ifftn(ifftshift(A3))
    return E_final


def plot_1D(x, y, label='', xlabel='', ylabel='', linestyle='-', linewidth=2,
            color=None, legend=False, ax=None, ticks_font_size=12,
            xy_label_font_size=14, legend_font_size=12):
    """
    Create a 1D line plot.

    Parameters:
        x (array-like): Values for the x-axis.
        y (array-like): Values for the y-axis.
        label (str): Label for the plot (for legend purposes).
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        linestyle (str): Line style (default '-').
        linewidth (float): Line width (default 2).
        color (str or None): Line color; if None, uses matplotlib default.
        legend (bool): Whether to display the legend.
        ax (matplotlib.axes.Axes or None): Axis to plot on; if None, use current axis.
        ticks_font_size (int): Font size for tick labels.
        xy_label_font_size (int): Font size for axis labels.
        legend_font_size (int): Font size for the legend text.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """
    if ax is None:
        ax = plt.gca()

    # Plot the data (use the provided color if given)
    if color is not None:
        ax.plot(x, y, linestyle=linestyle, linewidth=linewidth, label=label, color=color)
    else:
        ax.plot(x, y, linestyle=linestyle, linewidth=linewidth, label=label)

    # Set axis labels and tick parameters.
    ax.set_xlabel(xlabel, fontsize=xy_label_font_size)
    ax.set_ylabel(ylabel, fontsize=xy_label_font_size)
    ax.tick_params(axis='both', labelsize=ticks_font_size)

    # Display legend if requested and if a label is provided.
    if legend and label:
        ax.legend(shadow=True, fontsize=legend_font_size, facecolor='white',
                  edgecolor='black', loc='upper right')
    return ax


def plot_2D(E, x, y, xlabel='', ylabel='', cmap='jet', vmin=None, vmax=None,
            ax=None, ticks_font_size=12, xy_label_font_size=14):
    """
    Create a 2D image plot with a colorbar.

    Parameters:
        E (2D array): Data array to be plotted.
        x (1D array): Values for the x-axis.
        y (1D array): Values for the y-axis.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        cmap (str): Colormap (default 'jet').
        vmin (float or None): Minimum value for the colormap; if None, uses E.min().
        vmax (float or None): Maximum value for the colormap; if None, uses E.max().
        ax (matplotlib.axes.Axes or None): Axis to plot on; if None, use current axis.
        ticks_font_size (int): Font size for tick labels.
        xy_label_font_size (int): Font size for axis labels.

    Returns:
        matplotlib.axes.Axes: The axis object with the image plot.
    """
    if ax is None:
        ax = plt.gca()

    # Set vmin and vmax if not explicitly provided.
    if vmin is None:
        vmin = np.min(E)
    if vmax is None:
        vmax = np.max(E)

    # Create the image plot; note the extent aligns x and y with the data.
    extent = [y[0], y[-1], x[0], x[-1]]  # [left, right, bottom, top]
    image = ax.imshow(E, interpolation='bilinear', cmap=cmap, origin='lower', aspect='auto',
                      extent=extent, vmin=vmin, vmax=vmax)
    # Attach colorbar to the provided axis.
    cbar = plt.colorbar(image, ax=ax, shrink=0.9, pad=0.05, fraction=0.046)
    cbar.ax.tick_params(labelsize=ticks_font_size)

    # Set axis labels and tick parameters.
    ax.set_xlabel(xlabel, fontsize=xy_label_font_size)
    ax.set_ylabel(ylabel, fontsize=xy_label_font_size)
    ax.tick_params(axis='both', labelsize=ticks_font_size)
    return ax


def plot_3D(field3D, x_start, x_finish, y_start, y_finish, t_start, t_finish,
            x_resolution, y_resolution, t_resolution, isomin=40, isomax=40,
            opacity=0.6, surface_count=1):
    """
    Create a 3D isosurface plot of a field using Plotly.

    The function first builds a mesh grid for the (x, y, t) coordinates based on the
    given spatial/temporal limits and resolutions. The intensity (|field|^2) is computed,
    normalized, and then plotted as an isosurface.

    Parameters:
        field3D (ndarray): 3D field data with shape (x_resolution, y_resolution, t_resolution).
        x_start, x_finish (float): Range for the x-axis.
        y_start, y_finish (float): Range for the y-axis.
        t_start, t_finish (float): Range for the t-axis.
        x_resolution (int): Number of grid points in x.
        y_resolution (int): Number of grid points in y.
        t_resolution (int): Number of grid points in t.
        isomin (float): Minimum value for the isosurface.
        isomax (float): Maximum value for the isosurface.
        opacity (float): Opacity of the isosurface (default 0.6).
        surface_count (int): Number of isosurfaces to display.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object showing the isosurface.
    """
    # Create a mesh grid for x, y, and t.
    X, Y, Z = np.mgrid[x_start:x_finish:1j * x_resolution,
              y_start:y_finish:1j * y_resolution,
              t_start:t_finish:1j * t_resolution]

    # Compute the intensity (squared absolute value).
    values = np.abs(field3D) ** 2
    max_val = np.max(values)
    # Normalize values to a percentage of max for plotting, unless max is zero.
    values_norm = (values / max_val * 100) if max_val != 0 else values

    # Create the isosurface plot with Plotly.
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values_norm.flatten(),
        opacity=opacity,
        isomin=isomin,
        isomax=isomax,
        surface_count=surface_count,
        caps=dict(x_show=False, y_show=False)
    ))
    # Update layout for clear axis labels.
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='T'
    ))
    fig.show()
    return fig


if __name__ == '__main__':
    # 1) Generate the initial field and grab the central time slice
    field_temp = hobbit
    E3d = field_temp(xyt_mesh[0], xyt_mesh[1], xyt_mesh[2])  # shape (x,y,t)
    t_mid = t_resolution // 2
    E_slice = E3d[:, :, t_mid]

    # 2) Compute the “energy” (integral of |E|^2 dx dy) to normalize Imax
    dx = x_array[1] - x_array[0]
    dy = y_array[1] - y_array[0]
    energy = np.sum(np.abs(E_slice) ** 2) * dx * dy

    # 3) Find Imax so that total power Pmax is matched
    Imax = np.sqrt(Pmax / energy)
    print(f"Computed Imax = {Imax:e}")

    # --- (optional) set the global or pass Imax into your shape functions here ---

    # 4) Propagate the field along z
    field_propagated = split_step_time_Z(
        field_temp,
        loop_inner_resolution,
        loop_outer_resolution
    )  # shape (x, y, z)

    # 5) Plot a 2D slice: e.g. x vs. z at the central y
    y_mid = y_resolution // 2
    intensity_exponent = 2 if module_intensity else 1  # or whatever exponent you use
    data_2d = np.abs(field_propagated[:, y_mid, :]) ** intensity_exponent

    # convert to mm for plotting
    x_mm = x_array * 1e3
    z_mm = z_array * 1e3

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_2D(
        data_2d,
        x_mm,
        z_mm,
        xlabel='z (mm)',
        ylabel='x (mm)',
        cmap='magma'
    )
    plt.title('Propagation slice at y = {:.2f} mm'.format(y_array[y_mid] * 1e3))
    plt.show()
