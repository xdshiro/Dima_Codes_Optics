import matplotlib.pyplot as plt

# Create figure and axes
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Settings for both diagrams
distances = [0.5, 1.5]
titles = ['Short distance\n(small shift)', 'Long distance\n(large shift)']

for ax, d, title in zip(axes, distances, titles):
    # Plot two vertical lines at x=0 and x=d
    ax.plot([0, 0], [0, 1], linewidth=4)
    ax.plot([d, d], [0, 1], linewidth=4)

    # Draw arrow showing shift distance
    ax.annotate(
        '', xy=(0, -0.2), xytext=(d, -0.2),
        arrowprops=dict(arrowstyle='<->', lw=2)
    )
    ax.text(d / 2, -0.3, f'{d:.1f}', ha='center', va='top')

    # Formatting
    ax.set_xlim(-0.5, max(distances) + 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
