from project_flower_beams.paper_plots_flowers_ML.plots_functions_general import *

foils_paths = [
	'data_trefoils_turb\\hopf_dots_1e-40_trefoil_standard_12.npy',

]




for path in foils_paths:
	# foil4_dots_sorted = np.roll(np.load(path), 100, axis=0)
	foil4_dots = np.load(path) - np.array([100, 100, 0])
	dots_bound = [
		[-100, -100, 0],
		[100, 100, 129],
	]

	plot_black_dots_paper(foil4_dots, dots_bound=dots_bound, general_view=True, font_size=48)
	# plot_black_dots_paper(foil4_dots, dots_bound=dots_bound, general_view=False, font_size=48)
