from project_flower_beams.paper_plots_flowers_ML.plots_functions_general import *



foils_paths = [
	'5foils/foil5_dots_1e-30_[2, 2, 2, 2, 2].npy',
	'5foils/foil5_dots_1e-30_[2, 1, 2, 0, 2].npy',
	'6foils/foil6_dots_1e-30_[2, 2, 2, 2, 2, 2].npy',
	'6foils/foil6_dots_1e-30_[2, 1, 2, 2, 0, 2].npy',
]


for path in foils_paths:
	foil4_dots = np.load(path) - np.array([100, 100, 64])
	dots_bound = [
		[-100, -100, -64],
		[100, 100, 64],
	]
	plot_black_dots_paper(foil4_dots * [-1, 1, -1], dots_bound=dots_bound, font_size=64*1)
	# plot_black_dots_paper(foil4_dots * [-1, 0, 0], dots_bound=dots_bound, general_view=True)
