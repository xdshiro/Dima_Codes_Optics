from project_flower_beams.paper_plots_flowers_ML.plots_functions_general import *

foils_paths = [
	'data_foils_turb/foil4_dots_XY__0.05_[2, 2, 2, 2].npy',
	# 'data_foils_turb/foil4_dots_XY__0.05_[2, 2, 2, 0].npy',
	'data_foils_turb/foil4_dots_XY__0.05_[2, 1, 2, 0].npy',
	'data_foils_turb/foil4_dots_XY__0.15_[2, 2, 2, 2].npy',
	# 'data_foils_turb/foil4_dots_XY__0.15_[2, 2, 2, 0].npy',
	'data_foils_turb/foil4_dots_XY__0.15_[2, 1, 2, 0].npy',
	'data_foils_turb/foil4_dots_XY__0.25_[2, 2, 2, 2].npy',
	# 'data_foils_turb/foil4_dots_XY__0.25_[2, 2, 2, 0].npy',
	'data_foils_turb/foil4_dots_XY__0.25_[2, 1, 2, 0].npy',
]
foils_paths = [
	'data_foils_turb/foil4_dots_XY__0.05_[2, 2, 2, 2].npy',

]


# foils_paths = [
# 	'foil4_dots_XY_noturb_[2, 2, 2, 2]_sorted.npy',
# ]





for path in foils_paths:
	# foil4_dots_sorted = np.roll(np.load(path), 100, axis=0)
	foil4_dots = np.load(path) - np.array([100, 100, 0])
	dots_bound = [
		[-100, -100, 0],
		[100, 100, 129],
	]
	# foil4_dts_sorted = sort_dots_to_create_line_with_threshold(foil4_dots, TH=30)
	# np.save('foil4_dots_XY_noturb_[2, 2, 2, 2]_sorted.npy', foil4_dots_sorted)
	# plot_3d_line(foil4_dots_sorted)
	plot_black_dots_paper(foil4_dots, dots_bound=dots_bound, font_size=64)
	# plot_black_dots_paper(foil4_dots, dots_bound=dots_bound, general_view=True)
# plotDots_foils_paper_by_phi(foil4
# _dots, dots_bound, show=True, size=10)
# print(foil4_dots)
# foil4_field = foil4_field / np.max(np.abs(foil4_field))
# XY_max = 30e-3 * 185 / 300 * 1e3
# X = [-XY_max, XY_max]
# Y = [-XY_max, XY_max]
# plot_field_both_paper(foil4_field, extend=[*X, *Y])
