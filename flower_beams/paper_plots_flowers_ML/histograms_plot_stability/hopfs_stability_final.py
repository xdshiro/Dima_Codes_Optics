from extra_functions_package.all_knots_functions import *
from scipy.stats import norm

plt.rc('font', family='Times New Roman')

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
# ========== FONT SIZES ==========
axis_label_fontsize = 16 * 2
tick_label_fontsize = 16 * 2
legend_fontsize = 16 * 2
bar_text_fontsize = 13 * 2

# ========== DATA ==========
knots = ['H1','H2','H3','H4','H5','H6','H7','H8','H9','T1','T2']

stabilities_005 = [101 - 80, 101 - 70 + (70 - 66) / 2, 101 - 81 + 0 / 2,
                   101 - 85 + 4 / 2, 101 - 75, 100 - 40 + (40 - 32) / 2,
                   101 - 79 + 2 / 2, 101 - 58 + 4 / 2, 101 - 73 + 4 / 2,
                   16, 67.3333333]

stabilities_015 = [101 - 95 + 2 / 2, 101 - 92 + 2 / 2, 4,
                   4 + 2 / 2, 5 + 0 / 2, 100 - 79 + (76 - 66) / 2,
                   2 + 2 / 2, 101 - 86 + 6 / 2, 101 - 95 + 2 / 2,
                   1.6666666, 21.666666]

stabilities_025 = [1, 3 + 2 / 2, 1 + 2 / 3,
                   2 + 2 / 3, 3 + 0 / 2, 5 + 6 / 2,
                   2 + 1 / 3, 4 + 0 / 2, 1 + 1 / 3,
                   1 + 0 / 2, 3 + 2 / 2]

avg_005 = np.array(stabilities_005).mean()
avg_015 = np.array(stabilities_015).mean()
avg_025 = np.array(stabilities_025).mean()

# Compute the overall average stability (mean of the three averages)
overall_avg = np.mean([avg_005, avg_015, avg_025])

print("Average stability (σ_R² = 0.05):", avg_005)
print("Average stability (σ_R² = 0.15):", avg_015)
print("Average stability (σ_R² = 0.25):", avg_025)
print("Overall average stability:", overall_avg)

def confidence_interval(p, n, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)
    p = np.array(p) / 100
    se = np.sqrt((p*(1 - p))/n)
    return z*se*100

# Compute confidence intervals
n_samples = 300
err_005 = confidence_interval(stabilities_005, n_samples)
err_015 = confidence_interval(stabilities_015, n_samples)
err_025 = confidence_interval(stabilities_025, n_samples)

# ========== LAYOUT SETTINGS ==========
fig, ax = plt.subplots(figsize=(26, 7))

# Increase bar width and reduce spacing between groups
width = 0.25
group_spacing = 0.8

N = len(knots)
ind = np.arange(N)*group_spacing  # Scale the x positions

colors = ['#deebf7', '#9ecae1', '#3182bd']

# ========== PLOT BARS WITH ERROR BARS ==========
bar1 = ax.bar(ind - width, stabilities_005, width,
              yerr=err_005, capsize=5, color=colors[0],
              edgecolor='black', label=r"$\sigma_R^2=0.05$")

bar2 = ax.bar(ind, stabilities_015, width,
              yerr=err_015, capsize=5, color=colors[1],
              edgecolor='black', label=r"$\sigma_R^2=0.15$")

bar3 = ax.bar(ind + width, stabilities_025, width,
              yerr=err_025, capsize=5, color=colors[2],
              edgecolor='black', label=r"$\sigma_R^2=0.25$")

def add_value_labels(rects, errors, extra_offset=0.5):
    """
    Place a text label on top of each bar at (height + error + extra_offset).
    """
    for rect, err in zip(rects, errors):
        height = rect.get_height()
        x_pos = rect.get_x() + rect.get_width()/2
        label_pos = height + err + extra_offset
        ax.text(x_pos, label_pos,
                f"{height:.1f}",
                ha='center', va='bottom',
                # fontweight='bold',
                fontsize=bar_text_fontsize)

# Add numeric labels above each bar
add_value_labels(bar1, err_005, extra_offset=0.5)
add_value_labels(bar2, err_015, extra_offset=0.5)
add_value_labels(bar3, err_025, extra_offset=0.5)

# ========== AXIS & LEGEND STYLING ==========
ax.set_xlabel('Knot Type', fontsize=axis_label_fontsize)
ax.set_ylabel('Recovered Knots (%)', fontsize=axis_label_fontsize)
ax.set_xticks(ind)
ax.set_xticklabels(knots, fontsize=tick_label_fontsize)
ax.tick_params(axis='y', labelsize=tick_label_fontsize)

ax.legend(fontsize=legend_fontsize)

ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()