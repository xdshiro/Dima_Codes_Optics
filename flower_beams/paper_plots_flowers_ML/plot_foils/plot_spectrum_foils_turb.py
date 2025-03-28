from flower_beams.paper_plots_flowers_ML.plots_functions_general import *

foils_paths = [
	'data_foils_turb/foil4_spectr_XY__0.05_[2, 2, 2, 2].npy',
	# 'data_foils_turb/foil4_spectr_XY__0.05_[2, 2, 2, 0].npy',
	'data_foils_turb/foil4_spectr_XY__0.05_[2, 1, 2, 0].npy',
	'data_foils_turb/foil4_spectr_XY__0.15_[2, 2, 2, 2].npy',
	# 'data_foils_turb/foil4_spectr_XY__0.15_[2, 2, 2, 0].npy',
	'data_foils_turb/foil4_spectr_XY__0.15_[2, 1, 2, 0].npy',
	'data_foils_turb/foil4_spectr_XY__0.25_[2, 2, 2, 2].npy',
	# 'data_foils_turb/foil4_spectr_XY__0.25_[2, 2, 2, 0].npy',
	'data_foils_turb/foil4_spectr_XY__0.25_[2, 1, 2, 0].npy',
]

# foils_paths = [
# 	'data_foils_turb/foil4_spectr_XY__0.05_[2, 2, 2, 2].npy',
# ]


for path in foils_paths:
	foil4_spectrum_sorted = np.load(path)
	
	# plot_shifted_paper_grid_spectrum(foil4_spectrum_sorted, -10, 10, 0, 10
	# 								 , l1_lim=-6, l2_lim=6, p1_lim=0, p2_lim=6)
	plot_shifted_paper_grid_spectrum(foil4_spectrum_sorted, -10, 10, 0, 10, figsize=(10 * 1.4, 5.5 * 1.3))

# plotDots_foils_paper_by_phi(foil4
# _dots, dots_bound, show=True, size=10)
# print(foil4_dots)
# foil4_field = foil4_field / np.max(np.abs(foil4_field))
# XY_max = 30e-3 * 185 / 300 * 1e3
# X = [-XY_max, XY_max]
# Y = [-XY_max, XY_max]
# plot_field_both_paper(foil4_field, extend=[*X, *Y])
