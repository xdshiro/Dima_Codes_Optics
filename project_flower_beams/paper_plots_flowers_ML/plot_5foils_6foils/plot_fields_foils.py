from project_flower_beams.paper_plots_flowers_ML.plots_functions_general import *

foils_paths = [
	'5foils/foil5_field_XY_1e-30_[2, 2, 2, 2, 2].npy',
	'5foils/foil5_field_XY_1e-30_[2, 1, 2, 0, 2].npy',
	'6foils/foil6_field_XY_1e-30_[2, 2, 2, 2, 2, 2].npy',
	'6foils/foil6_field_XY_1e-30_[2, 1, 2, 2, 0, 2].npy',
]

for path in foils_paths:
	foil4_field = np.load(path)
	foil4_field = foil4_field / np.max(np.abs(foil4_field))
	XY_max = 35e-3 * 185 / 300 * 1e3 / 6 * np.sqrt(2)
	X = [-XY_max, XY_max]
	Y = [-XY_max, XY_max]
	plot_field_both_paper(foil4_field, extend=[*X, *Y])
