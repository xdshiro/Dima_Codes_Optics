"""
STOV Simulation and Pulse Propagation Module

This file defines:
  - Module flags for selecting paraxial/nonparaxial, intensity, and 3D visualization.
  - Grid and domain parameters for spatial (x,y), temporal (t), and propagation (z) axes.
  - Physical constants and beam parameters (STOV definitions, OAM, pulse duration, etc.).
  - Three STOV field generator functions: Field_STOV_1, Field_STOV_2, Field_STOV_3.
  - Two propagation routines:
      * split_step_old_time: FFT-based split-step solver for paraxial propagation.
      * UPPE_time: ODE-based unidirectional pulse propagation equation solver (non-paraxial).
  - Plotting helpers for 1D line plots and 3D isosurfaces.
  - Example usage showing initialization and propagation slices.

Usage:
  Configure the module flags and parameters at the top, then call the
  desired STOV field generator and propagation function. Plot using the
  provided helpers or the example at the end of this file.

Dependencies:
  numpy, matplotlib, scipy (fft, integrate, special), plotly

"""

import numpy as np                     # Numerical computing
import matplotlib.pyplot as plt         # Plotting library
from scipy.fft import fftn, ifftn, fftshift, ifftshift  # Fourier transforms
from scipy.integrate import odeint, complex_ode, ode     # ODE solvers
from scipy.special import jv, iv       # Bessel and modified Bessel functions
import plotly.graph_objects as go       # Interactive 3D plotting

# %% Module flags
# These booleans toggle different simulation features:
module_checking_spectrum = 1    # Plot spectral diagnostics when enabled
module_Paraxial = 1            # Use paraxial approximation (True) vs non-paraxial
module_NonRapaxial = 0         # Complement of paraxial flag
module_Intensity = 1           # Plot intensity (1) vs. field amplitude (0)
module_3D = 1                  # Enable 3D plotting of results

# %% Grid resolutions and domains
# Spatial and temporal discretization parameters:
xResolution = 71                # Number of x points
yResolution = 71               # Number of y points
tResolution = 71                # Number of temporal points
# Split-step iteration counts:
loopInnerResolution, loopOuterResolution = 51, 1  # M (inner), Kmax (outer)
zResolution = loopInnerResolution * loopOuterResolution  # Total z-slices
# Coordinate ranges:
xStart, xFinish = 0, 4500e-6     # x from 0 to 4500 μm
yStart, yFinish = 0, 4500e-6     # y from 0 to 4500 μm
zStart, zFinish = 0, 1.5          # propagation from z=0 to 1.5 m
tStart, tFinish = 0, 400e-12      # time window from 0 to 400 fs

# %% Pulse parameters
# Defines the initial beam and vortex parameters:
rho0 = 800e-6                   # Beam waist radius (m)
tp = 70e-12                     # Pulse duration (s)
lambda0 = 0.775e-6              # Central wavelength (m)
Pmax = 4.5e9                    # Peak pulse power (W)
# STOV-specific radii and OAM:
yRadius = 800e-6
xSTOVRadius = rho0
tSTOVRadius = tp
lOAM = 1                         # Orbital angular momentum order
# Compute central coordinates of the beam:
x0 = (xFinish - xStart) / 2
y0 = (yFinish - yStart) / 2
t0 = (tFinish - tStart) / 2
# Focusing parameter (infinite focus approximation)
f = 1e20

# %% Medium parameters
# Linear dispersion and refractive index:
k2Dis = 2.0 * 1e-15**2 / 1e-2  # Group-velocity dispersion (ps^2/m)
n0 = 1.0                      # Linear refractive index
# %% Nonlinear parameters
K = 1                          # Number of photons in MPI
n2 = 0 * 5.57e-19 * 1e-2 ** 2  # Nonlinear refractive index (m^2/W); set zero to disable Kerr
Ui = 1.0                       # Ionization potential (J)

def Betta_func(K):
    """Return beta parameter for MPI; zeroed out for STOV."""
    Betta = [0, 0]
    return Betta[K]

# %% Physical constants
eps0 = 8.854187817e-12         # Vacuum permittivity (F/m)
cSOL = 2.99792458e8             # Speed of light in vacuum (m/s)

# %% Plotting parameters
ticksFontSize = 18             # Font size for axis ticks
legendFontSize = 18            # Font size for legends
xyLabelFontSize = 18           # Font size for axis labels

# %% Derived parameters
# Wave number and frequency:
k0 = 2 * np.pi / lambda0       # Wave number (rad/m)
w0 = k0 * cSOL                 # Angular frequency (rad/s)
# Dispersion parameter:
wD = 2 * n0 / (k2Dis * cSOL)
# Nonlinear susceptibility:
chi3_2 = 8 * n0 * n2 / 3
epsNL = 3 * chi3_2 / 4          # Third-order nonlinear permittivity
# Intensity exponent for plotting:
Int = module_Intensity + 1     # 2 if intensity, 1 if field amplitude
# Initial Imax placeholder:
Imax = 1
# Critical power for self-focusing (W):
if n2 == 0:
    Pcrit = 1e100               # Avoid divide by zero if no Kerr
else:
    Pcrit = (1.22**2 * np.pi * lambda0**2) / (32 * n0 * n2)
print("P crit (MW):", Pcrit * 1e-6)

# Functions for Rayleigh and collapse lengths:
def LDF():
    """Rayleigh length of the beam (m)."""
    return np.pi * rho0**2 / lambda0

def Lcollapse():
    """Kerr collapse length (m) or zero if n2=0."""
    temp1 = 0.367 * LDF()
    if n2 == 0:
        return 0
    temp2 = (np.sqrt(Pmax / Pcrit) - 0.852)**2 - 0.0219
    return temp1 / np.sqrt(temp2)

print("Rayleigh length:", LDF(), "Kerr Collapse length:", Lcollapse())

# %% Arrays creation
# Build coordinate arrays for spatial (x,y), propagation (z), and temporal (t) grids
xArray = np.linspace(xStart, xFinish, xResolution)  # x-coordinates
yArray = np.linspace(yStart, yFinish, yResolution)  # y-coordinates
zArray = np.linspace(zStart, zFinish, zResolution)  # z (propagation) coordinates
tArray = np.linspace(tStart, tFinish, tResolution)  # time coordinates

# Mesh for ADI (x,t) calculations
xtMesh = np.meshgrid(xArray, tArray, indexing='ij')  # shape (2, xResolution, tResolution)

# Spectral grids for FFT-based methods:
# kx and ky span ± π*(N−2)/span to avoid edge artifacts
kxArray = np.linspace(-np.pi*(xResolution-2)/xFinish,
                      np.pi*(xResolution-2)/xFinish,
                      xResolution)
kyArray = np.linspace(-np.pi*(yResolution-2)/yFinish,
                      np.pi*(yResolution-2)/yFinish,
                      yResolution)
# Angular frequency grid
wArray  = np.linspace(-np.pi*(tResolution-2)/tFinish,
                      np.pi*(tResolution-2)/tFinish,
                      tResolution)

# Full 3D spatial-temporal grid for field evaluation
xytMesh = np.meshgrid(xArray, yArray, tArray, indexing='ij')  # shape (3, x, y, t)

# 3D spectral grid for (kx, ky, ω)
KxywMesh = np.meshgrid(kxArray, kyArray, wArray, indexing='ij')  # shape (3, kx, ky, ω)





# %% STOV Field Generators

def Field_STOV_1(x, y, t):
    """
    STOV Variant 1: Uses modified Bessel envelopes in the (x,t) plane and a Gaussian profile in y.

    The field is given by:
      E ∝ (-i)^ℓ e^{-iℓφ(x,t)} H₁(r) · exp[-(y/yRadius)²]
    where H₁(r) = (π³/2 r/4) e^{- (2πr)²/8} [I₀((2πr)²/8) - I₁((2πr)²/8)].

    Parameters:
        x, y, t : float or np.ndarray
            Spatial and temporal coordinates.
    Returns:
        complex or np.ndarray: Electric field amplitude at (x,y,t).
    """
    def H1(r):
        # Radial envelope combining Gaussian and modified Bessel functions
        arg = (2 * np.pi * r)**2 / 8
        return (np.pi**1.5 * r / 4) * np.exp(-arg) * (iv(0, arg) - iv(1, arg))

    # Gaussian profile along y
    def y_profile(yy):
        return np.exp(- (yy / yRadius)**2)

    # Azimuthal phase factor for OAM and spatio-temporal vortex
    phase = np.exp(-1j * lOAM * phi(x - x0, t - t0))

    # Combine components
    r = radius(x - x0, t - t0)
    envelope = H1(r)
    return 2 * np.pi * (-1j)**lOAM * phase * envelope * y_profile(y - y0)


def Field_STOV_2(x, y, t):
    """
    STOV Variant 2: Gaussian transverse profile, hyperbolic-secant temporal envelope,
    and an OAM phase factor in the x-y plane.

    E ∝ Imax · exp[-r²/(2 ρ₀²) - i k₀ r²/(2f)] · sech[1.07 (t-t₀)/τₚ] · exp[i ℓ φ(x,y)]

    Parameters:
        x, y, t : float or np.ndarray
    Returns:
        complex or np.ndarray: Electric field amplitude.
    """
    # Radial distance in x-y
    r_xy = radius(x - x0, y - y0)
    # Spatial Gaussian and focusing phase
    spatial = np.exp(- r_xy**2/(2 * rho0**2) - 1j * k0 * r_xy**2/(2 * f))
    # Temporal sech envelope
    temporal = 1 / np.cosh(1.07 * (t - t0) / tp)
    # OAM azimuthal phase
    oam_phase = np.exp(1j * lOAM * phi(x - x0, y - y0))

    return Imax * spatial * temporal * oam_phase


# STOV simple: algebraic (t/tw + i x/xw)^|ℓ| with separable Gaussians

def Field_STOV_main(x, y, t):
    """
    STOV Variant 3: Algebraic vortex core in t-x plane combined with Gaussian envelopes
    in x, y, and t, and a quadratic focusing phase.

    E ∝ [ (t/τ_w + i sign(ℓ) x/ρ_w )^|ℓ| ]
          · exp[-(y/yRadius)²] · exp[-(x/ρ₀)²] · exp[-(t/τₚ)²]
          · exp[-i k₀ (radius(x,y))²/(2f)]
    """
    # Gaussian envelopes
    env_y = np.exp(-((y - y0) / yRadius)**2)
    env_x = np.exp(-((x - x0) / rho0)**2)
    env_t = np.exp(-((t - t0) / tp)**2)

    # Algebraic vortex core in t-x
    core = ((t - t0)/tSTOVRadius + 1j * np.sign(lOAM) * (x - x0)/xSTOVRadius)**abs(lOAM)
    # Quadratic focusing phase in x-y
    focus = np.exp(-1j * k0 * radius(x - x0, y - y0)**2 / (2 * f))

    return Imax * core * env_y * env_x * env_t * focus

# %% Utility functions (already defined)
def radius(x, y): return np.sqrt(x**2 + y**2)
def phi(x, t):    return np.angle(x + 1j * t)





# =============================================================================
# Split-Step Propagation in z (3D+1 Time)
# =============================================================================
def split_step_old_time(shape, loop_inner_m=1, loop_outer_kmax=1):
    """
    Perform split-step propagation along z for a 3D+1D field:
      - Linear diffraction and dispersion via FFTs
      - Nonlinear phase accumulation (Kerr and plasma) in real space

    Parameters:
        shape (callable): function generating E(x,y,t)
        loop_inner_m (int): number of inner split steps
        loop_outer_kmax (int): number of outer propagation steps
    Returns:
        np.ndarray: propagated field E(x,y,t) after zResolution steps
    """
    # Helper: intensity |E|^2
    def compute_intensity(E):
        return np.abs(E)**2

    dz = zArray[1] - zArray[0]  # propagation step

    # Nonlinear phase term per split-step
    def compute_nonlinear_phase(E):
        I = compute_intensity(E)
        # Kerr term
        term_kerr = (1j/(2*eps0)) * ((w0 + KxywMesh[2])/(cSOL*n0)) * epsNL * I
        # MPI/plasma term
        term_plasma = - Betta_func(K) * I**(K-1)
        return dz * (term_kerr + term_plasma)

    # Linear propagation in spectral domain
    def linear_step(field):
        F = fftshift(fftn(field))
        # Diffraction and dispersion phase factors
        phase = np.exp(-1j * dz/(2*k0*n0) * (KxywMesh[0]**2 + KxywMesh[1]**2))
        phase *= np.exp( 1j * dz*k2Dis/2 * KxywMesh[2]**2)
        return ifftn(ifftshift(F * phase))

    # Initialize field
    E = shape(xytMesh[0], xytMesh[1], xytMesh[2])
    # Optional center-point print for debugging
    print("Center E:", E[xResolution//2, yResolution//2, tResolution//2])

    # Main split-step loops
    for _ in range(loop_outer_kmax):
        for _ in range(loop_inner_m):
            E = linear_step(E)
            nl_phase = compute_nonlinear_phase(E)
            E = E * np.exp(nl_phase)
    return E


# =============================================================================
# UPPE Time-Domain Solver via ODE Integration
# =============================================================================
def UPPE_time(shape, loop_inner_m, loop_outer_kmax):
    """
    Solve the Unidirectional Pulse Propagation Equation (UPPE) using an ODE
    in the spectral domain.

    Parameters:
        shape (callable): function generating initial E(x,y,t)
        loop_inner_m (int): inner integration steps per z
        loop_outer_kmax (int): outer z-steps
    Returns:
        np.ndarray: propagated field E(x,y,t)
    """
    # Helper: intensity
    def compute_intensity(E):
        return np.abs(E)**2
    # Helper: nonlinear polarization P = eps0*epsNL*E*|E|^2
    def compute_nonlinear_polarization(E):
        return eps0 * epsNL * E * compute_intensity(E)

    # Initial field and its spectrum
    E0 = shape(xytMesh[0], xytMesh[1], xytMesh[2])
    spec0 = fftshift(fftn(E0)).ravel()
    dz = zArray[1] - zArray[0]

    # Flatten spectral coordinates
    N = xResolution * yResolution * tResolution
    w1D = KxywMesh[2].ravel()
    kx1D, ky1D = KxywMesh[0].ravel(), KxywMesh[1].ravel()
    n1D = n0 * (1 + (w0 + w1D)/wD)
    k1D = n1D * (w0 + w1D)/cSOL
    kz1D = np.sqrt(k1D**2 - kx1D**2 - ky1D**2)
    vPhase = cSOL/(n0 + 2*n0*w0/wD)

    # ODE function for spectral evolution
    def ODEs(z, A_flat):
        # forward phase shift
        A_flat = A_flat * np.exp(1j*z*(kz1D - (w0 + w1D)/vPhase))
        A3 = A_flat.reshape((xResolution, yResolution, tResolution))
        E_space = ifftn(ifftshift(A3))
        P = compute_nonlinear_polarization(E_space)
        P_spec = fftshift(fftn(P)).ravel()
        # backward phase and scaling
        P_spec *= np.exp(-1j*z*(kz1D - (w0 + w1D)/vPhase))
        P_spec *= (1j/(2*eps0)) * ((w0 + w1D)**2/(cSOL**2 * kz1D))
        return P_spec

    integrator = ode(ODEs).set_integrator('zvode', nsteps=1e6)
    A_flat = spec0.copy()

    # z-propagation loop
    for _ in range(loop_outer_kmax):
        for _ in range(1, loop_inner_m):
            integrator.set_initial_value(A_flat, 0)
            A_flat = integrator.integrate(dz)
            A_flat *= np.exp(1j*dz*(kz1D - (w0 + w1D)/vPhase))

    # reshape and inverse transform
    A3_final = A_flat.reshape((xResolution, yResolution, tResolution))
    return ifftn(ifftshift(A3_final))



def plot_1D(x, y, label='', xlabel='', ylabel='', linestyle='-', linewidth=2,
            color=None, legend=False, ax=None,
            ticks_font_size=18, xy_label_font_size=18, legend_font_size=18):
    """
    Create a 1D line plot with customizable style and labeling.

    Parameters:
        x (array-like): x-axis data.
        y (array-like): y-axis data.
        label (str): Legend label for the line.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        linestyle (str): Line style (default='-').
        linewidth (float): Width of the line (default=2).
        color (str or None): Line color; if None uses default cycle.
        legend (bool): Whether to display the legend (requires a non-empty label).
        ax (matplotlib.axes.Axes or None): Axes object to draw on; uses current axes if None.
        ticks_font_size (int): Font size for tick labels.
        xy_label_font_size (int): Font size for axis labels.
        legend_font_size (int): Font size for legend text.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """
    if ax is None:
        ax = plt.gca()
    # Plot line
    plot_kwargs = {'linestyle': linestyle, 'linewidth': linewidth}
    if color:
        plot_kwargs['color'] = color
    ax.plot(x, y, label=label, **plot_kwargs)

    # Configure axes
    ax.set_xlabel(xlabel, fontsize=xy_label_font_size)
    ax.set_ylabel(ylabel, fontsize=xy_label_font_size)
    ax.tick_params(labelsize=ticks_font_size)

    # Add legend if requested
    if legend and label:
        ax.legend(fontsize=legend_font_size, shadow=True,
                  facecolor='white', edgecolor='black', loc='upper right')
    return ax


def plot_3D(field3D, x_start=xStart, x_end=xFinish,
            y_start=yStart, y_end=yFinish,
            t_start=tStart, t_end=tFinish,
            x_res=xResolution, y_res=yResolution, t_res=tResolution,
            isomin=30, isomax=30, opacity=0.6, surface_count=1):
    """
    Create a 3D isosurface plot of the intensity |field3D|^2 using Plotly.

    Parameters:
        field3D (ndarray): Complex field array of shape (x_res, y_res, t_res).
        x_start, x_end (float): Range of x-axis (m).
        y_start, y_end (float): Range of y-axis (m).
        t_start, t_end (float): Range of t-axis (s).
        x_res, y_res, t_res (int): Grid resolutions.
        isomin, isomax (float): Intensity percentage bounds for isosurface.
        opacity (float): Surface opacity.
        surface_count (int): Number of distinct isosurfaces.

    Returns:
        plotly.graph_objects.Figure: The generated isosurface figure.
    """
    # Build coordinate grid
    X, Y, Z = np.mgrid[x_start:x_end:1j*x_res,
                       y_start:y_end:1j*y_res,
                       t_start:t_end:1j*t_res]
    # Compute normalized intensity (0-100)
    intensity = np.abs(field3D)**2
    max_val = intensity.max() if intensity.size else 1
    norm_intensity = intensity / max_val * 100

    # Create isosurface plot
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=norm_intensity.flatten(),
        isomin=isomin, isomax=isomax,
        opacity=opacity,
        surface_count=surface_count,
        caps=dict(x_show=False, y_show=False)
    ))
    # Axis labels
    fig.update_layout(scene=dict(
        xaxis_title='X [m]',
        yaxis_title='Y [m]',
        zaxis_title='T [s]'
    ))
    fig.show()
    return fig

if __name__ == '__main__':
    # === Real-Parameter Propagation & Visualization Example ===
    # Choose which STOV variant to propagate:
    field_generator = Field_STOV_main  # options: Field_STOV_1, Field_STOV_2, Field_STOV_3

    # 1) Initial STOV field slice at central y-plane
    fig, ax = plt.subplots(figsize=(8, 6))
    E_init = field_generator(xytMesh[0], xytMesh[1], xytMesh[2])
    # extract |E|^2 slice at y = center
    slice_init = np.abs(E_init[:, yResolution // 2, :]) ** 2
    # plot with time in fs and x in mm
    im = ax.imshow(
        slice_init,
        extent=[tStart * 1e15, tFinish * 1e15, xStart * 1e3, xFinish * 1e3],
        origin='lower', aspect='auto', cmap='magma'
    )
    ax.set_xlabel('Time (fs)', fontsize=xyLabelFontSize)
    ax.set_ylabel('X (mm)', fontsize=xyLabelFontSize)
    plt.colorbar(im, ax=ax, label='Intensity (arb. units)')
    plt.title('Initial STOV Intensity Slice', fontsize=legendFontSize)
    plt.show()

    # 2) Paraxial propagation via split-step method
    if module_Paraxial:
        E_parax = split_step_old_time(field_generator, loopInnerResolution, loopOuterResolution)
        fig, ax = plt.subplots(figsize=(8, 6))
        slice_parax = np.abs(E_parax[:, yResolution // 2, :]) ** 2
        im = ax.imshow(
            slice_parax,
            extent=[tStart * 1e15, tFinish * 1e15, xStart * 1e3, xFinish * 1e3],
            origin='lower', aspect='auto', cmap='magma'
        )
        ax.set_xlabel('Time (fs)', fontsize=xyLabelFontSize)
        ax.set_ylabel('X (mm)', fontsize=xyLabelFontSize)
        plt.colorbar(im, ax=ax, label='Intensity (arb. units)')
        plt.title('Split-Step (Paraxial) Propagated Slice', fontsize=legendFontSize)
        plt.show()

    # 3) Non-paraxial propagation via UPPE_time
    if module_NonRapaxial:
        E_nparax = UPPE_time(field_generator, loopInnerResolution, loopOuterResolution)
        fig, ax = plt.subplots(figsize=(8, 6))
        slice_nparax = np.abs(E_nparax[:, yResolution // 2, :]) ** 2
        im = ax.imshow(
            slice_nparax,
            extent=[tStart * 1e15, tFinish * 1e15, xStart * 1e3, xFinish * 1e3],
            origin='lower', aspect='auto', cmap='magma'
        )
        ax.set_xlabel('Time (fs)', fontsize=xyLabelFontSize)
        ax.set_ylabel('X (mm)', fontsize=xyLabelFontSize)
        plt.colorbar(im, ax=ax, label='Intensity (arb. units)')
        plt.title('UPPE (Non-Paraxial) Propagated Slice', fontsize=legendFontSize)
        plt.show()
