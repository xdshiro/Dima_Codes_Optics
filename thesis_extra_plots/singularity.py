import numpy as np
import matplotlib.pyplot as plt

# Adjustable font size parameters
global_font_size = 34
title_font_size = 36
label_font_size = 36

# Set global font settings for Times New Roman and the specified global font size
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': global_font_size
})

# Define the grid: here we choose x and y from -1 to 1
x = np.linspace(-1.1, 1.1, 401)
y = np.linspace(-1.1, 1.1, 401)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Compute f = x + iy (which is simply Z)
F = Z

# Compute amplitude and phase
amplitude = np.abs(F)
phase = np.angle(F)

# Create a figure with two subplots for amplitude and phase
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot amplitude
im0 = ax[0].imshow(
    amplitude,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='equal',        # make axes square
    cmap='magma',
)
ax[0].set_title("Amplitude", fontsize=title_font_size)
ax[0].set_xlabel("x", fontsize=label_font_size)
ax[0].set_ylabel("y", fontsize=label_font_size)
ax[0].set_xticks([-1, 0, 1])
ax[0].set_yticks([-1, 0, 1])
cbar0 = fig.colorbar(im0, ax=ax[0], orientation='vertical')
cbar0.set_ticks([0, 1])
cbar0.set_ticklabels(['0', '1'])

# Plot phase
im1 = ax[1].imshow(
    phase,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='equal',        # make axes square
    cmap='hsv',
    interpolation='bilinear'
)
ax[1].set_title("Phase", fontsize=title_font_size)
ax[1].set_xlabel("x", fontsize=label_font_size)
ax[1].set_ylabel("y", fontsize=label_font_size)
ax[1].set_xticks([-1, 0, 1])
ax[1].set_yticks([-1, 0, 1])
cbar1 = fig.colorbar(im1, ax=ax[1], orientation='vertical')
# set ticks at -π, 0, +π
cbar1.set_ticks([-np.pi, 0, np.pi])
cbar1.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

plt.tight_layout()
plt.show()