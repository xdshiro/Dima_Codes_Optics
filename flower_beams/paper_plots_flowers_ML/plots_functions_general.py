from extra_functions_package.all_knots_functions import *

import seaborn as sns

fopts = 52  # Font size for labels
tick_fopts = 52  # Font size for tick numbers
plt.rc('font', family='Times New Roman')

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

import matplotlib.colors as mcolors
# Define the colors you want in your transition

color_list = [
    (0.0,  'black'),         # at the bottom
    (0.3,  'royalblue'),          # a deep, more neutral blue
    # (0.4,  'blue'),          # pure blue (#0000FF)
    # (0.5,  'lightskyblue'),  # light blue tone
    (0.5,  'deepskyblue'),  # light blue tone
    (1.0,  'white')          # top
]
# Create a custom colormap
# Get the built-in "Blues_r" colormap.
blues_r = plt.colormaps.get_cmap("Blues_r")
# for x in [0.0, 0.125, 0.25, 0.5, 1.0]:
#     print(x, blues_r(x))

def black_to_blues_r():
	# 1) Grab the reversed "Blues" colormap, sampled with 256 points.
	blues_r = plt.cm.get_cmap("Blues_r", 256)
	part = 0.15
	# 2) We'll build a list of (x, color) control points for the new colormap.
	#    - From x=0.0 to x=0.1, go from black to blues_r(0) (the darkest blue).
	#    - From x=0.1 to x=1.0, smoothly map blues_r's entire 0→1 range.
	color_list = []

	# Black at 0.0
	color_list.append((0.0, (0, 0, 0, 1)))  # RGBA for black

	# At 0.1, switch to darkest blue from Blues_r
	darkest_blue = blues_r(0)  # typically a deep blue
	color_list.append((part, darkest_blue))

	# 3) From 0.1→1.0, sweep through the entire Blues_r (0→1).
	#    i=0 will map to new_x=0.1, i=255 will map to new_x=1.0
	#    old_val in [0..1] → new_val in [0.1..1]
	for i in range(256):
		old_val = i / 255.0
		new_val = part + (1 - part) * old_val
		color_list.append((new_val, blues_r(old_val)))

	# 4) Create the new LinearSegmentedColormap
	return mcolors.LinearSegmentedColormap.from_list(
		"black_to_blues_r",
		color_list,
		N=256
	)

my_cmap = black_to_blues_r()

def plotDots_foils_paper_by_phi(dots, dots_bound=None, show=True, size=15, width=185, fig=None, save=None, reso=100):
	colorLine = 'white'
	
	# Define angle sections (phi ranges)
	z1 = np.linspace(-1 / 4 * np.pi, 1 / 4 * np.pi, reso)
	z2 = np.linspace(1 / 4 * np.pi, 3 / 4 * np.pi, reso)
	z31 = np.linspace(3 / 4 * np.pi, np.pi, reso // 2)
	z32 = np.linspace(-np.pi, -3 / 4 * np.pi, reso // 2)
	z4 = np.linspace(-3 / 4 * np.pi, -1 / 4 * np.pi, reso)
	
	# Compute the angle phi for each dot
	phi = np.arctan2(dots[:, 1], dots[:, 0])  # arctan2(y, x) gives angles in [-pi, pi]
	
	# Separate dots into 4 sections based on phi
	dots_z1 = dots[(phi >= -1 / 4 * np.pi) & (phi < 1 / 4 * np.pi)]
	dots_z2 = dots[(phi >= 1 / 4 * np.pi) & (phi < 3 / 4 * np.pi)]
	dots_z31 = dots[(phi >= 3 / 4 * np.pi) | (phi < -3 / 4 * np.pi)]  # Handles wrap-around
	dots_z4 = dots[(phi >= -3 / 4 * np.pi) & (phi < -1 / 4 * np.pi)]
	
	# Assign colors
	colors = {
		"z1": 'red',
		"z2": 'blue',
		"z31": 'green',
		"z4": 'purple'
	}
	
	# Plot each section with corresponding color
	if fig is None:
		fig = pl.plot_3D_dots_go(dots_z1, marker={'size': size, 'color': colors['z1'],
		                                          'line': dict(width=width, color=colorLine)})
	else:
		pl.plot_3D_dots_go(dots_z1, fig=fig, marker={'size': size, 'color': colors['z1'],
		                                             'line': dict(width=width, color=colorLine)})
	
	pl.plot_3D_dots_go(dots_z2, fig=fig, marker={'size': size, 'color': colors['z2'],
	                                             'line': dict(width=width, color=colorLine)})
	pl.plot_3D_dots_go(dots_z31, fig=fig, marker={'size': size, 'color': colors['z31'],
	                                              'line': dict(width=width, color=colorLine)})
	pl.plot_3D_dots_go(dots_z4, fig=fig, marker={'size': size, 'color': colors['z4'],
	                                             'line': dict(width=width, color=colorLine)})
	
	# Set up the bounding box
	if dots_bound is None:
		dots_bound = dots
	pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01)
	
	# Save or show the figure
	if save is not None:
		fig.write_html(save)
	if show:
		fig.show()
	
	return fig


def plot_shifted_paper_grid_spectrum(spectrum, l1, l2, p1, p2, figsize=(10, 5.5),
                                     map=my_cmap, title='',
									l1_lim=None, l2_lim=None, p1_lim=None, p2_lim=None,
                                     grid=True, axis_equal=False, xname='l', yname='p',
                                     xlim=None, ylim=None, every_ticks=False,
									 xyLabelFontSize=fopts,
									ticksFontSize=tick_fopts
):
	# my_cmap   plt.cm.gist_earth figsize=(8.5, 6)
	"""
	Plot the spectrum with grid shifted by 0.5 and custom settings.

	Parameters:
		spectrum: 2D numpy array to plot.
		l1, l2: Start and end points for the x-axis.
		p1, p2: Start and end points for the y-axis.
		map: Colormap for the plot.
		title: Title of the plot.
		xyLabelFontSize: Font size for axis labels.
		ticksFontSize: Font size for tick numbers.
		grid: Boolean, whether to show the grid.
		axis_equal: Boolean, whether to set equal aspect ratio.
		xname, yname: Names for the x and y axes.
		xlim, ylim: Optional x and y axis limits.
	"""
	if l1_lim is None:
		l1_lim = l1
	if l2_lim is None:
		l2_lim = l2
	if p1_lim is None:
		p1_lim = p1
	if p2_lim is None:
		p2_lim = p2

	# Create shifted grid
	x = np.arange(l1 - 0.5, l2 + 1.5, 1)  # Shift grid in x by 0.5
	y = np.arange(p1 - 0.5, p2 + 1.5, 1)  # Shift grid in y by 0.5
	# x = np.arange(l1, l2 + 1, 1)  # Shift grid in x by 0.5
	# y = np.arange(p1, p2 + 1, 1)  # Shift grid in y by 0.5
	
	# Create the figure and axis
	# fig, ax = plt.subplots(figsize=(13.5, 6))
	fig, ax = plt.subplots(figsize=figsize)
	# print(np.sum((np.abs(spectrum)/100)**2))
	# Plot the spectrum with shifted extent
	image = ax.imshow(np.abs(spectrum).T / 100,
	                  interpolation='none', cmap=map,
	                  origin='lower', aspect='auto', vmin=0,
	                  extent=[x[0], x[-1], y[0], y[-1]])
	
	# Add colorbar
	#
	cbr = plt.colorbar(image, ax=ax, fraction=0.05, pad=0.01, aspect=10)

	cbr.set_ticks((0, 0.5))
	cbr.ax.tick_params(labelsize=ticksFontSize)  # Colorbar tick font size
	
	# Axis ticks and labels
	if every_ticks:
		ax.set_xticks(np.arange(l1_lim, l2_lim + 1, 1))  # Main x-axis ticks
	else:
		ax.set_xticks(np.arange(l1_lim, l2_lim + 1, 2))  # Main x-axis ticks
	ax.set_yticks(np.arange(p1, p2 + 1, 1))  # Main y-axis ticks
	ax.set_xlabel(xname, fontsize=xyLabelFontSize, fontstyle='italic', labelpad=-4)
	ax.set_ylabel(yname, fontsize=xyLabelFontSize, fontstyle='italic')
	plt.xticks(fontsize=ticksFontSize)
	plt.yticks(fontsize=ticksFontSize)
	plt.xlim(l1_lim, l2_lim)
	plt.ylim(p1_lim, p2_lim)
	
	# Grid settings with minor ticks shifted by 0.5
	# ax.grid(grid, which='major', linestyle='--', alpha=0.7)
	ax.xaxis.set_minor_locator(plt.MultipleLocator(1), )  # Minor ticks every 1 unit
	ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
	ax.grid(True, which='minor', linestyle='--', alpha=0.8)
	
	# Minor grid alignment shift
	ax.set_xticks(np.arange(l1_lim - 0.5, l2_lim + 1.5, 1), minor=True)  # Shifted minor x ticks
	ax.set_yticks(np.arange(p1_lim - 0.5, p2_lim + 1.5, 1), minor=True)  # Shifted minor y ticks
	# Minor ticks are disabled, but minor grid remains
	ax.tick_params(axis='x', which='minor', length=0, width=0, direction='in', labelsize=0)
	ax.tick_params(axis='y', which='minor', length=4, width=1, direction='in', labelsize=0)
	# Plot title
	# plt.title(title, fontweight="bold", fontsize=26)
	
	# Optional settings
	if axis_equal:
		ax.set_aspect('equal', adjustable='box')
	if xlim is not None:
		ax.set_xlim(xlim[0], xlim[1])
	if ylim is not None:
		ax.set_ylim(ylim[0], ylim[1])
	ax.set_aspect('equal')
	# Adjust layout and display the plot
	plt.tight_layout(pad=0.1)
	plt.show()


def plotDots_foils_paper_by_indices(dots, indices, dots_bound=None, show=True, size=15, width=185, fig=None, save=None,
									general_view=False, font_size=64):
	"""
	Plot dots in 3D space with coloring based on a sequence of indices dividing the array into 4 parts.

	Parameters:
		dots: numpy array of shape (n, 3) representing 3D points.
		indices: list of 4 integers defining the division points in the array.
		dots_bound: Optional, bounding box points for the plot.
		show: Boolean, whether to display the plot.
		size: Marker size for the dots.
		width: Line width for the markers.
		fig: Optional, existing plotly figure to add to.
		save: Optional, file name to save the plot as HTML.

	Returns:
		fig: Plotly figure object.
	"""
	colorLine = 'white'
	
	# Sort indices to ensure they divide the array correctly
	indices = sorted(indices)
	
	# Split the dots array based on the provided indices
	dots_part1 = dots[indices[0]:indices[1]]  # From first to second index
	dots_part2 = dots[indices[1]:indices[2]]  # Between second and third index
	dots_part3 = dots[indices[2]:indices[3]]  # Between third and fourth index
	dots_part4 = np.vstack((dots[:indices[0]], dots[indices[3]:]))  # Combine [0:indices[0]] and [indices[3]:]
	
	# Assign requested colors to each section
	colors = ['#ff0000', '#007dff', '#ff9900', '#19ff19']
	colors = ['#660000', '#000099', '#ffff00', '#134a0d']
	# red blue orange greed
	colors = ['#ff0000', '#000099', '#ff9900', '#134a0d']
	# blue orange greed red
	colors = ['#000099', '#ff9900', '#134a0d', '#ff0000']
	colors = ['#000099', '#ffcc00', '#166d13', '#ff0000']
	colors = ['#000099', '#ffb200', '#166d13', '#ff0000']


	colors = ['#000099', '#ffb200', '#166d13', '#ff0000']
	colors = ["#000000",  # black
			  "#808080",  # gray
			  "#9ecae1",  # light-ish blue
			  "#3182bd"]  # darker blue
	colors = [
		"#000000",  # Black
		"#555555",  # Darker gray
		"#6baed6",  # Medium blue from Blues
		"#08519c",  # Darker blue from Blues
	]
	colors = [
		"#6baed6",  # Medium blue from Blues
		"#666666",  # Medium gray (between #555555 and #808080)
		"#08519c",  # Darker blue from Blues
		"#000000",  # Black

	]
	dots_parts = [dots_part1, dots_part2, dots_part3, dots_part4]
	
	# Plot each section with corresponding color
	if fig is None:
		fig = pl.plot_3D_dots_go(dots_parts[0], marker={'size': size, 'color': colors[0],
		                                                'line': dict(width=width, color=colorLine)})
	for i in range(1, len(dots_parts)):
		pl.plot_3D_dots_go(dots_parts[i], fig=fig, marker={'size': size, 'color': colors[i],
		                                                   'line': dict(width=width, color=colorLine)})
	
	# Set up the bounding box
	if dots_bound is None:
		dots_bound = dots
	pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01)
	if general_view:
		# scale_zoom = 2.45
		scale_zoom = 2.1
		camera = dict(
			eye=dict(x=1.25 * scale_zoom, y=1.25 * scale_zoom, z=1.25 * scale_zoom),
			up=dict(x=0, y=0, z=1),
			center=dict(x=0, y=0, z=0)
		)
	else:
		scale_zoom = 2.1
		camera = dict(
			eye=dict(x=0, y=0, z=2 * scale_zoom),  # Top-down view
			up=dict(x=0, y=-1, z=0),  # Rotate around z-axis by 90 degrees
			center=dict(x=0, y=0, z=0)  # Keep the center of the plot unchanged
		)
	fig.update_layout(scene_camera=camera)
	fig.update_layout(
		scene=dict(
			xaxis=dict(
				title=dict(
					text='<i>x</i>',
					font=dict(
						family='Times New Roman',  # Italic Times New Roman
						size=font_size  # Font size
					)
				)
			),
			yaxis=dict(
				title=dict(
					text='<i>y</i>',
					font=dict(
						family='Times New Roman',
						size=font_size
					)
				)
			),
			zaxis=dict(
				title=dict(
					text='<i>z</i>',
					font=dict(
						family='Times New Roman',
						size=font_size
					)
				)
			)
		)
	)
	# Save or show the figure
	if save is not None:
		fig.write_html(save)
	if show:
		fig.show()
	
	return fig


def plot_black_dots_paper(dots, dots_bound=None, size=15, width=185, show=True, fig=None, save=None,
                          general_view=False, font_size=64):
	"""
	Plot dots in 3D space as black dots.

	Parameters:
		dots: numpy array of shape (n, 3) representing 3D points.
		size: Marker size for the dots.
		width: Line width for the markers.
		show: Boolean, whether to display the plot.
		fig: Optional, existing plotly figure to add to.
		save: Optional, file name to save the plot as HTML.

	Returns:
		fig: Plotly figure object.
	"""
	colorLine = 'white'
	# Set up default figure if none is provided
	if fig is None:
		fig = pl.plot_3D_dots_go(dots, marker={
			'size': size,
			'color': 'black',
			'line': dict(width=width, color=colorLine)  # Border color for the dots
		})
	else:
		pl.plot_3D_dots_go(dots, fig=fig, marker={
			'size': size,
			'color': 'black',
			'line': dict(width=width, color=colorLine)
		})
	if dots_bound is None:
		dots_bound = dots
	pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01)
	if general_view:
		# scale_zoom = 2.45
		scale_zoom = 2.1
		camera = dict(
			eye=dict(x=1.25 * scale_zoom, y=1.25 * scale_zoom, z=1.25 * scale_zoom),
			up=dict(x=0, y=0, z=1),
			center=dict(x=0, y=0, z=0)
		)
	else:
		scale_zoom = 2.1
		camera = dict(
			eye=dict(x=0, y=0, z=2 * scale_zoom),  # Top-down view
			up=dict(x=0, y=-1, z=0),  # Rotate around z-axis by 90 degrees
			center=dict(x=0, y=0, z=0)  # Keep the center of the plot unchanged
		)
	fig.update_layout(scene_camera=camera)
	fig.update_layout(
		scene=dict(
			xaxis=dict(
				title=dict(
					text='<i>x</i>',
					font=dict(
						family='Times New Roman',  # Italic Times New Roman
						size=font_size  # Font size
					)
				)
			),
			yaxis=dict(
				title=dict(
					text='<i>y</i>',
					font=dict(
						family='Times New Roman',
						size=font_size
					)
				)
			),
			zaxis=dict(
				title=dict(
					text='<i>z</i>',
					font=dict(
						family='Times New Roman',
						size=font_size
					)
				)
			)
		)
	)
	# Save or show the figure
	if save is not None:
		fig.write_html(save)
	if show:
		fig.show()
	
	return fig


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_field_both_paper(E, extend=None, colorbars='phase', figsize=(10, 9),
						  xyLabelFontSize=fopts, ticksFontSize=tick_fopts):
	"""
	Plots:
	  - Phase of E on the main Axes
	  - Amplitude of E on the inset Axes

	colorbars can be:
	  'none'       -> No colorbars
	  'amplitude'  -> Only amplitude colorbar
	  'phase'      -> Only phase colorbar
	  'both'       -> Both colorbars
	"""

	# Create figure and main axis
	fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

	# -- PHASE PLOT on main Axes --
	im_phase = ax.imshow(np.angle(E).T,
						 extent=extend,
						 cmap='gray',
						 vmin=-np.pi,
						 vmax=np.pi)

	# Create PHASE colorbar only if requested
	if colorbars in ('phase', 'both'):
		cbar_phase = fig.colorbar(im_phase, ax=ax,fraction=0.1, pad=0.01, aspect=20)
		# cbar_phase = fig.colorbar(im_phase, ax=ax, fraction=0.046, pad=0.01)
		cbar_phase.ax.tick_params(labelsize=ticksFontSize)
		cbar_phase.set_ticks([-np.pi, 0, np.pi])
		cbar_phase.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
	# Else, no colorbar for phase

	# Format main Axes
	ax.set_xlabel('$x/w_0$', fontsize=xyLabelFontSize, labelpad=0)
	ax.set_ylabel('$y/w_0$', fontsize=xyLabelFontSize, labelpad=-18)
	ax.tick_params(axis='both', labelsize=ticksFontSize)
	ax.set_xticks([-3, 0, 3])
	ax.set_yticks([-3, 0, 3])

	# -- AMPLITUDE INSET --
	inset_ax = inset_axes(ax,
						  width="35%",
						  height="35%",
						  loc="upper left",
						  # loc="upper right",
						  bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
						  # bbox_to_anchor=(-0.1, 0.0, 1.0, 1.0),
						  bbox_transform=ax.transAxes)

	im_amp = inset_ax.imshow(np.abs(E).T,
							 extent=extend,
							 cmap=my_cmap)  # or your custom my_cmap

	inset_ax.set_xticks([])
	inset_ax.set_yticks([])

	# Optional: translucent (or none) facecolor
	inset_ax.set_facecolor('none')
	# Give inset a white border
	for spine in inset_ax.spines.values():
		spine.set_edgecolor('white')
		spine.set_linewidth(1)
	if colorbars in ('amplitude', 'both'):
		# Create an inset axis for the amplitude colorbar on the right of the inset
		cax = inset_axes(inset_ax,
						 width="10%",  # Adjust width as needed
						 height="80%",  # Same height as inset_ax
						 loc='lower left',
						 bbox_to_anchor=(1.02, 0.1, 1, 1),

						 bbox_transform=inset_ax.transAxes,
						 borderpad=0)
		# Use the amplitude image (im_amp) for the colorbar
		cbar_amp = fig.colorbar(im_amp, cax=cax)
		cbar_amp.set_ticks([0, 1])
		cbar_amp.ax.tick_params(labelsize=ticksFontSize)
	# # Create AMPLITUDE colorbar only if requested
	# if colorbars in ('amplitude', 'both'):
	# 	# Place colorbar on the main axis's side as well:
	# 	cbar_amp = fig.colorbar(im_phase, ax=ax,fraction=0.1, pad=0.01, aspect=20)
	# 	cbar_amp.set_ticks([0, 1])
	# 	cbar_amp.ax.tick_params(labelsize=ticksFontSize)
	# 	# Optionally set amplitude ticks:
	# 	# cbar_amp.set_ticks([0, 1])
	# 	# cbar_amp.set_label('|E|', fontsize=xyLabelFontSize)
	# # ax.set_anchor('SW')
	plt.show()

def plot_field_both_paper_old_version(E, extend=None):
	# Set Times New Roman as the font globally
	
	xyLabelFontSize = fopts
	ticksFontSize = tick_fopts
	
	fig, ax = plt.subplots(1, 2, figsize=(13.5, 6))

	# Plot |Amplitude|
	im0 = ax[0].imshow(np.abs(E).T, extent=extend, cmap='hot', interpolation='nearest')
	cbar0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.01)
	cbar0.ax.tick_params(labelsize=ticksFontSize)  # Set colorbar tick font size
	cbar0.set_ticks([0, 1])  # Custom ticks for |Amplitude|
	ax[0].set_xlabel('x (mm)', fontsize=xyLabelFontSize)  # X-axis label
	ax[0].set_ylabel('y (mm)', fontsize=xyLabelFontSize)  # Y-axis label
	ax[0].tick_params(axis='both', labelsize=ticksFontSize)  # Set axis tick font size

	# Plot Phase
	# im1 = ax[1].imshow(np.angle(E).T, extent=extend, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
	im1 = ax[1].imshow(np.angle(E).T, extent=extend, cmap='hsv', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
	cbar1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.01)
	cbar1.ax.tick_params(labelsize=ticksFontSize)  # Set colorbar tick font size
	cbar1.set_ticks([-np.pi, 0, np.pi])  # Custom ticks for Phase
	cbar1.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])  # Use LaTeX for Phase ticks
	ax[1].set_xlabel('x (mm)', fontsize=xyLabelFontSize)  # X-axis label
	ax[1].set_ylabel('y (mm)', fontsize=xyLabelFontSize)  # Y-axis label
	ax[1].tick_params(axis='both', labelsize=ticksFontSize)  # Set axis tick font size

	# Adjust layout
	plt.tight_layout()
	
	# Display the figure
	plt.show()


def plot_confusion_matrix(cm, class_labels, label='', figsize=(20, 18), cmap="Blues",
                          annot_fontsize=fopts):
	"""
	Plot the confusion matrix with custom font sizes and class labels.

	Parameters:
		cm: 2D numpy array
			The confusion matrix to be plotted.
		class_labels: list
			List of class labels to use as x and y ticks.
		figsize: tuple
			Size of the plot figure.
		cmap: str
			Colormap for the heatmap.
			:param annot_fontsize: size of the confusion numbers
	"""
	cm = cm / cm.sum(axis=1, keepdims=True) * 100
	xyLabelFontSize = fopts
	ticksFontSize = tick_fopts
	# Create the figure
	plt.figure(figsize=figsize)
	vmin = 0
	vmax = 100
	# Plot the heatmap
	# fmt="d"
	ax = sns.heatmap(cm, annot=True, fmt=".1f", xticklabels=class_labels, yticklabels=class_labels,
	            cmap=cmap, annot_kws={"size": annot_fontsize}, vmin=vmin, vmax=vmax,
                cbar_kws={"shrink": 0.9, "aspect": 20,
                          "ticks": np.linspace(vmin, vmax, 5),
                          "pad": 0.01,
                          })

	cbar = plt.gcf().axes[-1]  # Get the colorbar axis
	cbar.tick_params(labelsize=ticksFontSize)  # Set the font size for colorbar ticks
	cbar.set_ylabel("Accuracy (%)", fontsize=xyLabelFontSize)  # Set label and font size
	# Add labels and customize font sizes
	plt.xlabel('Predicted Knots', fontsize=xyLabelFontSize)
	plt.ylabel('True Knots', fontsize=xyLabelFontSize)
	plt.xticks(fontsize=ticksFontSize)
	plt.yticks(fontsize=ticksFontSize)
	ax.set_aspect('equal')
	plt.title(label, fontsize=xyLabelFontSize, fontweight="bold")
	# plt.subplots_adjust(right=0.95)  # Shrink the right margin
	plt.tight_layout()
	plt.show()


def plot_confusion_matrix_big(cm, class_labels, label='', figsize=(40 ,35.5), cmap="Blues",
						  annot_fontsize=fopts):
	"""
	Plot the confusion matrix with custom font sizes and class labels.

	Parameters:
		cm: 2D numpy array
			The confusion matrix to be plotted.
		class_labels: list
			List of class labels to use as x and y ticks.
		figsize: tuple
			Size of the plot figure.
		cmap: str
			Colormap for the heatmap.
			:param annot_fontsize: size of the confusion numbers
	"""
	cm = cm / cm.sum(axis=1, keepdims=True) * 100
	xyLabelFontSize = fopts
	ticksFontSize = tick_fopts
	ticksFontSizeXY = ticksFontSize / 1.35
	# Create the figure
	plt.figure(figsize=figsize)
	vmin = 0
	vmax = 100
	# Plot the heatmap
	ax = sns.heatmap(cm, annot=False, fmt="d", xticklabels=class_labels, yticklabels=class_labels,
					 cmap=cmap, vmin=vmin, vmax=vmax,
					 linewidths=0.2,  # controls grid line thickness
					 linecolor=(0, 0, 0, 0.3),  # 30% opaque black
					 # linecolor='black',  # or 'white', etc.
					 cbar_kws={"shrink": 0.92, "aspect": 40,
							   "ticks": np.linspace(vmin, vmax, 5),
							   "pad": 0.001,
							   })

	cbar = plt.gcf().axes[-1]  # Get the colorbar axis
	cbar.tick_params(labelsize=ticksFontSize)  # Set the font size for colorbar ticks
	cbar.set_ylabel("Accuracy (%)", fontsize=xyLabelFontSize)  # Set label and font size
	# Add labels and customize font sizes
	plt.xlabel('Predicted Knots', fontsize=xyLabelFontSize)
	plt.ylabel('True Knots', fontsize=xyLabelFontSize)
	plt.xticks(fontsize=ticksFontSizeXY, rotation=90)
	plt.yticks(fontsize=ticksFontSizeXY, rotation=0)
	ax.set_aspect('equal')
	plt.title(label, fontsize=xyLabelFontSize, fontweight="bold")
	# plt.subplots_adjust(right=0.95)  # Shrink the right margin
	plt.tight_layout(pad=0.4)
	plt.show()
