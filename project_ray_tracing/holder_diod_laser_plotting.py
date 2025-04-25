""""
Plotting Script for Ray‐Trace Intensity Volumes

This script loads a 3D intensity volume produced by the integrating-sphere ray trace,
crops it to a specified real-world region, smooths it, and then:

  1. Extracts and plots a central X–Z cross-section at the mid-Y plane.
  2. Computes the cumulative energy contained within a circular aperture
     as a function of Z and plots the result.
  3. Exports both the cross-section data and cumulative‐energy curve to CSV and Excel.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd

# ------------------------------------------------------------------------------
# User Settings
# ------------------------------------------------------------------------------
INPUT_FILE = r'Z7_f-2.50_r2.50_221res_10000rays.npy'   # Path to the .npy volume
RUN_NAME   = 'experiment1'                        # Identifier for saved outputs

# Real-world bounds corresponding to the full volume (in mm)
REAL_BOUNDS = {
    'x_min': -2.2, 'x_max':  2.2,
    'y_min': -2.2, 'y_max':  2.2,
    'z_min':  0.0, 'z_max':  3.0,
}

# Target crop bounds (in mm)
TARGET_BOUNDS = {
    'x_min': -2.1, 'x_max':  2.1,
    'y_min': -2.1, 'y_max':  2.1,
    'z_min':  0.5, 'z_max':  3.0,
}

# Gaussian smoothing sigma (voxels)
SMOOTH_SIGMA = 2

# Colormap for cross-section
CMAP = 'turbo'

# ------------------------------------------------------------------------------
def crop_volume(volume, real_bounds, target_bounds):
    """
    Crop a 3D array to a specified real-world subregion.

    Parameters
    ----------
    volume : ndarray, shape (Nx, Ny, Nz)
        The full intensity volume.
    real_bounds : dict
        {'x_min','x_max','y_min','y_max','z_min','z_max'} for the full volume.
    target_bounds : dict
        Same keys, specifying the desired crop region.

    Returns
    -------
    cropped : ndarray
        Sub-volume corresponding to target_bounds.
    axes : tuple of 3 arrays
        (x_coords, y_coords, z_coords) in mm for the cropped volume.
    """
    Nx, Ny, Nz = volume.shape
    # Generate linearly spaced coordinates for each axis
    x = np.linspace(real_bounds['x_min'], real_bounds['x_max'], Nx)
    y = np.linspace(real_bounds['y_min'], real_bounds['y_max'], Ny)
    z = np.linspace(real_bounds['z_min'], real_bounds['z_max'], Nz)

    # Find index ranges via searchsorted
    i0, i1 = np.searchsorted(x, [target_bounds['x_min'], target_bounds['x_max']])
    j0, j1 = np.searchsorted(y, [target_bounds['y_min'], target_bounds['y_max']])
    k0, k1 = np.searchsorted(z, [target_bounds['z_min'], target_bounds['z_max']])

    cropped = volume[i0:i1, j0:j1, k0:k1]
    return cropped, x[i0:i1], y[j0:j1], z[k0:k1]


def plot_cross_section(slice2d, x_coords, z_coords):
    """
    Plot the X–Z cross-section with a colorbar.

    Parameters
    ----------
    slice2d : ndarray, shape (Nx, Nz)
        Intensity at the mid-Y plane.
    x_coords : array, shape (Nx,)
        X coordinates in mm.
    z_coords : array, shape (Nz,)
        Z coordinates in mm.
    """
    # Normalize to [0, 100]%
    pct = slice2d / slice2d.max() * 100

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    im = ax.imshow(
        pct.T,
        origin='lower',
        extent=(x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]),
        aspect='equal',
        cmap=CMAP,
        interpolation='spline36',
        vmin=0, vmax=100
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cbar.ax.tick_params(labelsize=14)

    ax.set_xlabel('X (mm)', fontsize=16)
    ax.set_ylabel('Z (mm)', fontsize=16)
    ax.set_xticks(np.linspace(x_coords[0], x_coords[-1], 5))
    ax.set_yticks(np.linspace(z_coords[0], z_coords[-1], 4))
    ax.tick_params(labelsize=14)
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(z_coords[0], z_coords[-1])

    ax.set_title('X–Z Intensity Cross-Section (Y = mid)', fontsize=16)
    fig.tight_layout(pad=0.2)
    plt.show()


def compute_cumulative_energy(volume, x_coords, y_coords, z_coords):
    """
    Compute cumulative energy vs. Z within a circular aperture in X–Y.

    Parameters
    ----------
    volume : ndarray, shape (Nx, Ny, Nz)
        Cropped intensity volume.
    x_coords : array, shape (Nx,)
        X coordinates in mm.
    y_coords : array, shape (Ny,)
        Y coordinates in mm.
    z_coords : array, shape (Nz,)
        Z coordinates in mm.

    Returns
    -------
    z_coords : array, shape (Nz,)
        Same as input.
    cum_energy : array, shape (Nz,)
        Percentage of total energy contained at or below each Z.
    """
    # Build circular mask at each X–Y slice
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='xy')
    radius = (x_coords.max() - x_coords.min()) / 2
    mask = (xx**2 + yy**2) <= radius**2

    # Sum intensity within mask for each Z
    slice_sums = [(volume[:, :, k] * mask).sum() for k in range(volume.shape[2])]
    cum = np.cumsum(slice_sums)
    cum_pct = cum / cum[-1] * 100  # normalize to 100%
    return z_coords, cum_pct


def plot_cumulative_energy(z_coords, cum_energy):
    """
    Plot the cumulative energy distribution along Z.

    Parameters
    ----------
    z_coords : array, shape (Nz,)
        Z coordinates in mm.
    cum_energy : array, shape (Nz,)
        Cumulative energy (%) at each Z.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(z_coords, cum_energy, lw=2, label='Cumulative Energy')
    ax.axhline(0,   color='gray', linestyle='--', lw=0.8)
    ax.axhline(100, color='gray', linestyle='--', lw=0.8, label='100%')
    ax.set_xlabel('Z (mm)', fontsize=14)
    ax.set_ylabel('Cumulative Energy (%)', fontsize=14)
    ax.set_title('Cumulative Energy vs. Z', fontsize=16)
    ax.set_xticks(np.round(np.linspace(z_coords[0], z_coords[-1], 6), 2))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    fig.tight_layout(pad=0.2)
    plt.show()


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    save_energies = True
    # Load the 3D volume
    volume = np.load(INPUT_FILE)
    # Smooth the volume
    volume = gaussian_filter(volume.astype(float), sigma=SMOOTH_SIGMA)

    # Crop to the region of interest
    cropped, xs, ys, zs = crop_volume(volume, REAL_BOUNDS, TARGET_BOUNDS)

    # Extract central X–Z slice at mid-Y
    mid_j = len(ys) // 2
    slice_xz = cropped[:, mid_j, :]

    # Plot cross-section
    plot_cross_section(slice_xz, xs, zs)

    # Compute and plot cumulative energy
    z_vals, cum_en = compute_cumulative_energy(cropped, xs, ys, zs)
    plot_cumulative_energy(z_vals, cum_en)

    # Save results to CSV and Excel
    if save_energies:
        df_energy = pd.DataFrame({'Z (mm)': z_vals, 'Cumulative Energy (%)': cum_en})
        df_energy.to_csv(f'cumulative_energy_{RUN_NAME}.csv', index=False)
        df_energy.to_excel(f'cumulative_energy_{RUN_NAME}.xlsx', index=False, engine='openpyxl')

        df_slice = pd.DataFrame(slice_xz.T, index=zs, columns=xs)
        df_slice.to_csv(f'cross_section_{RUN_NAME}.csv')
        df_slice.to_excel(f'cross_section_{RUN_NAME}.xlsx', engine='openpyxl')

        print(f"Plots generated and data saved under prefix '{RUN_NAME}'.")