from flower_beams.paper_plots_flowers_ML.plots_functions_general import *


foils_paths = [
	'data_trefoils_turb\\hopf_spectr_before_1e-40_trefoil_standard_12.npy',
]

# foils_paths = [
# 	'foil4_spectr_XY_noturb_[2, 1, 2, 0].npy',
# ]


for path in foils_paths:
	foil4_spectrum_sorted = np.load(path)

	plot_shifted_paper_grid_spectrum(foil4_spectrum_sorted, -6, 6, 0, 6, every_ticks=False)

