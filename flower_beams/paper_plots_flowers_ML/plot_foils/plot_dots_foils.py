from flower_beams.paper_plots_flowers_ML.plots_functions_general import *


foils_paths = [
	'data_foils_noturb/foil4_dots_XY_noturb_[2, 2, 2, 2]_sorted.npy',
	'data_foils_noturb/foil4_dots_XY_noturb_[2, 2, 2, 0]_sorted.npy',
	'data_foils_noturb/foil4_dots_XY_noturb_[2, 1, 2, 0]_sorted.npy',
]
indices_all = [[70, 230, 390, 550],
               [70, 71, 230, 395],
               [62, 63, 210, 280],
               ]
# foils_paths = [
# 	'data_foils_noturb/foil4_dots_XY_noturb_[2, 1, 2, 0]_sorted.npy',
# ]
#
# indices_all = [[62, 63, 210, 280],
#                ]
def sort_dots_to_create_line_with_threshold(dots, TH):
	"""
	Sort a 3D array of points to create a line in 3D space such that
	each point is the closest to the previous one, while discarding points
	with distances greater than a threshold.

	Parameters:
		dots: numpy array of shape (n, 3) representing 3D points.
		TH: float, the maximum allowable distance between consecutive points.

	Returns:
		sorted_dots: numpy array of shape (m, 3) representing the sorted 3D points.
					 m <= n since points beyond the threshold are discarded.
	"""
	# Make a copy of the array to avoid modifying the original
	dots = dots.copy()
	
	# Start with the first point
	sorted_dots = [dots[0]]
	remaining_dots = np.delete(dots, 0, axis=0)
	
	# Iteratively find the closest point
	while len(remaining_dots) > 0:
		last_point = sorted_dots[-1]
		distances = np.linalg.norm(remaining_dots - last_point, axis=1)  # Euclidean distances
		closest_idx = np.argmin(distances)  # Index of the closest point
		
		# Check if the closest distance is within the threshold
		if distances[closest_idx] <= TH:
			# Add the point to the sorted list and remove it from remaining
			sorted_dots.append(remaining_dots[closest_idx])
			remaining_dots = np.delete(remaining_dots, closest_idx, axis=0)
		else:
			# Terminate the sorting process if no valid points are within the threshold
			break
	
	return np.array(sorted_dots)


def plot_3d_line(points, color='blue', linewidth=2):
	"""
	Plot a 3D line connecting the given points.

	Parameters:
		points: numpy array of shape (n, 3) representing the sorted 3D points.
		color: Color of the line (default is 'blue').
		linewidth: Thickness of the line (default is 2).
	"""
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')
	
	# Extract x, y, z coordinates
	x, y, z = points[:, 0], points[:, 1], points[:, 2]
	
	# Plot the line connecting the points
	ax.plot3D(x, y, z, color=color, linewidth=linewidth)
	
	# Optionally, scatter the points for visualization
	ax.scatter3D(x, y, z, color='red', s=50, label='Points')
	
	# Set labels and title
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('3D Line Connecting Points')
	
	# Add a legend
	ax.legend()
	
	# Show the plot
	plt.show()


for path, indices in zip(foils_paths, indices_all):
	# foil4_dots_sorted = np.roll(np.load(path), 100, axis=0)
	foil4_dots_sorted = np.load(path)
	dots_bound = [
		[-100, -100, 0],


		[100, 100, 129],
	]
	# foil4_dts_sorted = sort_dots_to_create_line_with_threshold(foil4_dots, TH=30)
	# np.save('foil4_dots_XY_noturb_[2, 2, 2, 2]_sorted.npy', foil4_dots_sorted)
	# plot_3d_line(foil4_dots_sorted)
	print(len(foil4_dots_sorted))
	# B&W
	# plot_black_dots_paper(foil4_dots_sorted, dots_bound=dots_bound)
	# plotDots_foils_paper_by_indices(foil4_dots_sorted, indices, dots_bound=dots_bound)
	plotDots_foils_paper_by_indices(foil4_dots_sorted, indices, dots_bound=dots_bound, general_view=True,
									font_size=128)
# plotDots_foils_paper_by_phi(foil4
# _dots, dots_bound, show=True, size=10)
# print(foil4_dots)
# foil4_field = foil4_field / np.max(np.abs(foil4_field))
# XY_max = 30e-3 * 185 / 300 * 1e3
# X = [-XY_max, XY_max]
# Y = [-XY_max, XY_max]
# plot_field_both_paper(foil4_field, extend=[*X, *Y])
