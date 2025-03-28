import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import assoc_laguerre
from matplotlib import colors


def LG_simple(x, y, z=0, l=1, p=0, width=1, k0=1, x0=0, y0=0, z0=0):
    """Compute the complex field E of a Laguerre–Gaussian beam."""

    def rho(*r):
        return np.sqrt(sum(val ** 2 for val in r))

    def phi(x, y):
        return np.angle(x + 1j * y)

    def laguerre_polynomial(x, l, p):
        return assoc_laguerre(x, p, l)

    # Shift coordinates
    x = x - x0
    y = y - y0
    z = z - z0

    # Rayleigh range
    zR = k0 * width ** 2

    E = (np.sqrt(math.factorial(p) / (np.pi * math.factorial(abs(l) + p))) *
         (rho(x, y) ** abs(l)) * np.exp(1j * l * phi(x, y)) /
         (width ** (abs(l) + 1) * (1 + 1j * z / zR) ** (abs(l) + 1)) *
         ((1 - 1j * z / zR) / (1 + 1j * z / zR)) ** p *
         np.exp(-rho(x, y) ** 2 / (2 * width ** 2 * (1 + 1j * z / zR))) *
         laguerre_polynomial(rho(x, y) ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)), abs(l), p)
         )
    return E


# -------------------------------------------------------------------
# Adjustable parameters for font sizes, figure layout, and colorbars
# -------------------------------------------------------------------
global_font_size = 16
title_font_size = 18
label_font_size = 16

# Spacing/margins: tweak these to control the overall layout
left_margin = 0.07
right_margin = 0.85
top_margin = 0.92
bottom_margin = 0.08
wspace_val = 0.2
hspace_val = 0.2

# Colorbar sizing: tweak these to change the colorbar width and spacing
cb_fraction = 0.04
cb_pad = 0.02

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': global_font_size
})

# ---------------------------------
# 1) Prepare grid and beam settings
# ---------------------------------
# Updated grid to span from -4 to 4
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)

# Define the LG beams to plot
beams = [
    {"l": 0, "p": 0},  # LG_00
    {"l": 0, "p": 1},  # LG_01
    {"l": 1, "p": 0},  # LG_10
    {"l": 1, "p": 1},  # LG_11
]

# ---------------------------------
# 2) Precompute amplitude and phase for each beam,
#    and determine global amplitude range.
# ---------------------------------
amps = []
phases = []
amp_min = float('inf')
amp_max = -float('inf')

for beam in beams:
    E = LG_simple(X, Y, l=beam["l"], p=beam["p"], width=1, k0=1)
    A = np.abs(E)
    P = np.angle(E)
    amps.append(A)
    phases.append(P)

    amp_min = min(amp_min, A.min())
    amp_max = max(amp_max, A.max())

norm_amp = colors.Normalize(vmin=amp_min, vmax=amp_max)
norm_phase = colors.Normalize(vmin=-np.pi, vmax=np.pi)

# ---------------------------------
# 3) Create figure with 2 rows (amplitude and phase)
#    and 4 columns (one per beam)
# ---------------------------------
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 6))

plt.subplots_adjust(
    left=left_margin,
    right=right_margin,
    top=top_margin,
    bottom=bottom_margin,
    wspace=wspace_val,
    hspace=hspace_val
)

# Ticks to use on axes
ticks = [-4, -2, 0, 2, 4]

# ---------------------------------
# 4) Plot amplitude (top row) and phase (bottom row)
# ---------------------------------
for idx, beam in enumerate(beams):
    l_val = beam["l"]
    p_val = beam["p"]

    # Top row: amplitude
    ax_amp = axes[0, idx]
    im_amp = ax_amp.imshow(
        amps[idx],
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        aspect='equal',
        cmap='magma',
        norm=norm_amp
    )
    ax_amp.set_title(f"Amplitude (l={l_val}, p={p_val})", fontsize=title_font_size)
    ax_amp.set_xticks([])  # Remove x ticks in the top row

    # Only the first column gets y ticks
    if idx == 0:
        ax_amp.set_yticks(ticks)
        ax_amp.set_ylabel("y", fontsize=label_font_size)
    else:
        ax_amp.set_yticks([])

    # Bottom row: phase
    ax_phase = axes[1, idx]
    im_phase = ax_phase.imshow(
        phases[idx],
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        aspect='equal',
        cmap='hsv',
        norm=norm_phase,
        interpolation='bilinear'
    )
    ax_phase.set_title(f"Phase (l={l_val}, p={p_val})", fontsize=title_font_size)
    ax_phase.set_xticks(ticks)
    ax_phase.set_xlabel("x", fontsize=label_font_size)

    # Only the first column gets y ticks
    if idx == 0:
        ax_phase.set_yticks(ticks)
        ax_phase.set_ylabel("y", fontsize=label_font_size)
    else:
        ax_phase.set_yticks([])

# -----------------------------------------------------------
# 5) Add shared colorbars: one for amplitude (top row)
#    and one for phase (bottom row)
# -----------------------------------------------------------
cbar_amp = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm_amp, cmap='magma'),
    ax=axes[0, :],
    orientation='vertical',
    fraction=cb_fraction,
    pad=cb_pad
)
cbar_amp.set_label("Amplitude", fontsize=label_font_size)

cbar_phase = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm_phase, cmap='hsv'),
    ax=axes[1, :],
    orientation='vertical',
    fraction=cb_fraction,
    pad=cb_pad
)
cbar_phase.set_label("Phase [rad]", fontsize=label_font_size)

plt.show()