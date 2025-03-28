import numpy as np
from scipy.special import erf, jv, iv, assoc_laguerre

# =============================================================================
# Module Selection Flags
# =============================================================================
# These flags determine which modules or features of the simulation are active.
module_checking_spectrum = 0  # Flag for checking spectral properties
module_hobbit = 0  # Flag for enabling the "Hobbit" module
module_sum = 1  # Flag for summing fields
mod_4pulses = 1  # Flag for using four pulses
module_adi = 0  # Flag for using the ADI propagation method
module_paraxial = 1  # Flag for using paraxial approximation
module_nonrapaxial = 0  # Flag for non-paraxial propagation (if applicable)
module_phase = 0  # Flag for phase-related processing
module_initial = 0  # Flag for initial field setup
module_intensity = 1  # Flag to use intensity (1) or field absolute value (0)
module_3d = 0  # Flag for 3D simulations
save_results = 0  # Flag for saving simulation outputs
save_name = 'SUMm-1B_4Hp1e5_y100_phph1_-3'  # Base filename for saved outputs

# =============================================================================
# Spatial and Temporal Resolutions
# =============================================================================
# Grid resolutions and physical domain limits.
x_resolution = 171  # Number of grid points in the x-direction
y_resolution = 171  # Number of grid points in the y-direction
t_resolution = 1  # Number of grid points in time (2D version of the code)
loop_inner_resolution = 501  # Inner loop resolution factor (M)
loop_outer_resolution = 1  # Outer loop resolution factor (Kmax)
z_resolution = loop_inner_resolution * loop_outer_resolution  # Total z-resolution

# Spatial domain boundaries (in meters)
x_start, x_finish = 0, 700e-6  # x-range (0 to 700 μm)
y_start, y_finish = 0, 700e-6  # y-range (0 to 700 μm)
z_start, z_finish = 1e-3, 0.08  # z-range (1 mm to 80 mm)

# Temporal domain boundaries (in seconds)
t_start, t_finish = 0, 1000e-13  # t-range (0 to 1000e-13 s)

# Offsets for field positioning (in meters and seconds)
x1, x2 = 0 * 50e-6, 0 * -50e-6  # x-offsets for different fields (currently zero)
y1, y2 = 100e-6, -100e-6  # y-offsets for different fields
t1, t2 = 0 * 100e-15, 0 * -100e-15  # t-offsets for different fields (currently zero)

# A derived time index (typically used to index the middle of the temporal grid)
time_index = int(t_resolution / 2)

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
# Pulse Parameters
# =============================================================================
rho0 = 200e-6  # Beam radius or focal spot size (in meters)
tp = 200e-12  # Pulse duration (in seconds)
lambda0 = 0.517e-6  # Central wavelength of the pulse (in meters)
Pmax = 13.5e6  # Maximum power of the pulse (e.g., in Watts)

# =============================================================================
# "Hobbit" Module Parameters
# =============================================================================
# Parameters for the "Hobbit" field generation.
beta_hob = 1.08  # Scaling parameter for the Hobbit phase factor
alpha_hob = 0  # Additional phase offset for the Hobbit field
ro_0_hob = 650e-6  # Characteristic radius for the Hobbit field (in meters)
k = 3  # Mode index or photon number used in Hobbit functions
F_hob = 150e-3  # Focal length or related parameter for Hobbit (in meters)
w_ring_hob = 244e-6  # Ring width parameter for the Hobbit field (in meters)
w_G_hob = lambda0 * F_hob / (np.pi * w_ring_hob)  # Derived Gaussian width for Hobbit
tau_0 = 242e-4  # Temporal parameter for the Hobbit field

# =============================================================================
# STOV (Spatio-Temporal Optical Vortex) Parameters
# =============================================================================
y_radius = 800e-6  # Transverse radius for the STOV field in y (in meters)
x_stov_radius = rho0  # Transverse radius for the STOV field in x (set equal to rho0)
t_stov_radius = tp  # Temporal width for the STOV field (in seconds)
l_oam = 0  # Orbital angular momentum (OAM) order for the field
phase = np.pi * 1  # Global phase shift (in radians)
l_oam_sum1 = 1  # OAM order for the first summed field component
l_oam_sum2 = -1  # OAM order for the second summed field component

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
        0,
        0 * 2e-0,  # Placeholder value
        2,
        3,
        2.4e-37 * (1e-2) ** (2 * K_val - 3),
        5,
        6,
        7,
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

# Intensity mode: 1 = intensity, 2 = field amplitude (if needed)
Int_mode = module_intensity + 1

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


# -----------------------------------------------------------------------------
# Field Functions
# -----------------------------------------------------------------------------
def sum_fields(x, y, t):
    """
    Sum fields (Hobbit-type) at a given (x, y, t).

    Global parameters used:
      Imax, x1, y1, t1, x2, y2, t2,
      lOAMSUM1, lOAMSUM2, phase, MOD_4pulses, k
    """
    field = Imax * (
            u_far_hob2(x - x1, y - y1, t - t1, l_oam_sum1, k) +
            u_far_hob2(x - x2, y - y2, t - t2, l_oam_sum2, k) * np.exp(1j * phase)
    )
    if mod_4pulses:
        field += Imax * u_far_hob2(x - x2, y - y2, t - t2, -l_oam_sum2, k)
    return field


def field_simple_oam(x, y, t):
    """
    Compute a simple Optical Angular Momentum (OAM) field with a Gaussian envelope.

    Global parameters used:
      Imax, radius, x0, y0, rho0, k0, f, t0, tp, lOAM

    Note: Verify that the second call to radius uses the correct y-offset.
    """
    r = radius(x - x0, y - y0)
    envelope = np.exp(- r ** 2 / rho0 ** 2 - 1j * k0 * r ** 2 / (2 * f) - ((t - t0) / tp) ** 2)
    oam_term = ((x - x0) / rho0 + 1j * np.sign(l_oam) * (y - y0) / rho0) ** abs(l_oam)
    return Imax * envelope * oam_term


def field_stov_1(x, y, t):
    """
    Compute a STOV (Spatio-Temporal Optical Vortex) field (method 1).

    Global parameters used:
      lOAM, phi, x0, t0, y0, yRadius, radius
    """

    def H1(r):
        return (np.pi ** 1.5 * r / 4 * np.exp(- (2 * np.pi * r) ** 2 / 8) *
                (iv(0, (2 * np.pi * r) ** 2 / 8) - iv(1, (2 * np.pi * r) ** 2 / 8)))

    phase_factor = np.exp(-1j * l_oam * phi(x - x0, t - t0))
    spatial_factor = H1(radius(x - x0, t - t0))
    y_modulation = np.exp(- (y - y0) ** 2 / y_radius ** 2)
    return 2 * np.pi * (-1j) ** l_oam * phase_factor * spatial_factor * y_modulation


def hobbit(x, y, t):
    """
    Compute the Hobbit field at (x, y, t).

    Global parameters used:
      l_oam, x0, y0, t0, Imax, k, (and those used in the helper functions)
    """
    m = l_oam
    print(f"l_oam = {m}")
    return Imax * u_far_hob2(x, y, t, m, k)


def asymmetric_lg(x, y, t):
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


def field_stov_simple(x, y, t):
    """
    Compute a simple STOV field using spatio-temporal envelopes.

    Global parameters used:
      Imax, t0, t_stov_radius, l_oam, x0, x_stov_radius, rho0, tp, y0, y_radius, k0, f, radius
    """

    def y_dependence(y_val):
        return np.exp(- (y_val / y_radius) ** 2)

    def x_dependence(x_val):
        return np.exp(- (x_val / rho0) ** 2)

    def t_dependence(t_val):
        return np.exp(- (t_val / tp) ** 2)

    field_term = ((t - t0) / t_stov_radius + 1j * np.sign(l_oam) * (x - x0) / x_stov_radius) ** abs(l_oam)
    # Note: Using radius(x - x0, y - y0) instead of (y - x0) to ensure proper coordinate offsets.
    phase_term = np.exp(-1j * k0 * radius(x - x0, y - y0) ** 2 / (2 * f))
    return Imax * field_term * y_dependence(y - y0) * x_dependence(x - x0) * t_dependence(t - t0) * phase_term



def radius(x, y):
    """Compute the radial distance from the origin."""
    return np.sqrt(x**2 + y**2)

def phi(x, y_or_t):
    """
    Compute the phase angle of the complex number x + i*(y_or_t).

    Note: The second parameter can represent either a spatial or temporal coordinate.
    """
    return np.angle(x + 1j * y_or_t)


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
    delta = (z_array[1] - z_array[0]) / (4 * n0 * k0 * (x_array[1] - x_array[0])**2)

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
        L_plus[i, i]     = 1 - 2j * delta
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
        L_minus[i, i]     = 1 + 2j * delta
        L_minus[i, i + 1] = -1j * delta * v_array[i]

    # -------------------------------
    # Dispersion (time) step parameter delta_D
    # -------------------------------
    delta_D = - (z_array[1] - z_array[0]) * k2_dis / (4 * (t_array[1] - t_array[0])**2)

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
        L_plus_D[i, i]     = 1 - 2j * delta_D
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
        L_minus_D[i, i]     = 1 + 2j * delta_D
        L_minus_D[i, i + 1] = -1j * delta_D

    # Invert the L_minus matrices for the implicit update steps
    L_minus_D_inv = np.linalg.inv(L_minus_D)
    L_minus_inv   = np.linalg.inv(L_minus)

    # -------------------------------
    # Local helper functions for intensity and plasma density
    # -------------------------------
    def intensity(E):
        """Calculate the intensity |E|^2 of the field."""
        return np.abs(E)**2

    def plasma_density_initial(E):
        """
        Alternative plasma density calculation (not used in main loop).

        Uses the field intensity at the mid-time index.
        """
        density = np.zeros((x_resolution, t_resolution))
        for i in range(t_resolution):
            density[:, i] = 2 * (
                sigma_K8 * np.abs(E[:, t_resolution // 2])**(2 * K) *
                np.sqrt(np.pi / (8 * K)) * rho_at * t_finish * 0.1
            )
        return density

    def plasma_density(E):
        """
        Compute plasma density evolution based on the field intensity.

        Uses a finite-difference update in time.
        """
        density = np.zeros((x_resolution, t_resolution))

        def wofi(I_val):
            return sigma_K8 * I_val**K

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
            - betta_func(K) / 2 * intensity(E)**(K - 1) * (1 - plasma_dens / rho_at)
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
            plasma_dens = plasma_density(E)

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