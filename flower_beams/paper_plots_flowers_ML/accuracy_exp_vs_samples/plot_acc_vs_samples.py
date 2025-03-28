from extra_functions_package.all_knots_functions import *

# Set the font to Times New Roman for text and math
plt.rc('font', family='Times New Roman')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# ========== FONT SIZES ==========
axis_label_fontsize = 16 * 2       # 32
tick_label_fontsize = 16 * 2       # 32
legend_fontsize = 16 * 2           # 32

# ========== DATA ==========
accuracy = [16.5, 43.3, 49, 51.4, 54.1, 54.3, 56.3, 58.9, 60.4, 61.7, 62.2]

samples = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ========== PLOT SETTINGS ==========
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the data with markers and a line.
# Here we use the color '#3182bd' (the third color in your palette) for consistency.
ax.plot(samples, accuracy, marker='o', markersize=16, linestyle='-', linewidth=8,
        color='#3182bd',           # Line color
        markerfacecolor='#1f77b4', # Marker fill color (a different blue tone)
        markeredgecolor='black',   # Marker edge color
        label='Test Accuracy')

# Set the axis labels with the same font sizes as before.
ax.set_xlabel('Samples per Class (Training)', fontsize=axis_label_fontsize)
ax.set_ylabel('Accuracy on Test Dataset (%)', fontsize=axis_label_fontsize)

# Set the tick label sizes
ax.tick_params(axis='both', labelsize=tick_label_fontsize)
ax.set_xticks(range(0, 11))
# Optionally add a legend with the specified font size.
# ax.legend(fontsize=legend_fontsize)
# ax.set_ylim(0, 65)
# Add a light grid on the y-axis similar to your previous plot.
ax.grid(axis='y', linestyle='--', alpha=0.8)

plt.tight_layout()
plt.show()