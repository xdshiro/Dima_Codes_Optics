from flower_beams.paper_plots_flowers_ML.plots_functions_general import *

foils_paths = [
	'5foils/foil5_spectr_XY__1e-30_[2, 2, 2, 2, 2].npy',
	'5foils/foil5_spectr_XY__1e-30_[2, 1, 2, 0, 2].npy',
	'6foils/foil6_spectr_XY__1e-30_[2, 2, 2, 2, 2, 2].npy',
	'6foils/foil6_spectr_XY__1e-30_[2, 1, 2, 2, 0, 2].npy',
]

for path in foils_paths:
	foil4_spectrum_sorted = np.load(path)
	# plot_shifted_paper_grid_spectrum(foil4_spectrum_sorted, -10, 10, 0, 10
	# 								 , l1_lim=-6, l2_lim=6, p1_lim=0, p2_lim=6
	# 								 , figsize=(10, 5.5))
	plot_shifted_paper_grid_spectrum(foil4_spectrum_sorted, -10, 10, 0, 10, figsize=(10 * 1.5, 5.5 * 1.5))

