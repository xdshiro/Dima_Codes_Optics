from project_flower_beams.paper_plots_flowers_ML.plots_functions_general import *


foils_paths = [
	'data_all_hopfs_basis\\hopf_spectr_before_1e-40_standard_14.npy',
	'data_all_hopfs_basis\\hopf_spectr_before_1e-40_standard_16.npy',
	'data_all_hopfs_basis\\hopf_spectr_before_1e-40_standard_18.npy',
	'data_all_hopfs_basis\\hopf_spectr_before_1e-40_30both.npy',
	'data_all_hopfs_basis\\hopf_spectr_before_1e-40_30oneZ.npy',
	'data_all_hopfs_basis\\hopf_spectr_before_1e-40_optimized.npy',
	'data_all_hopfs_basis\\hopf_spectr_before_1e-40_pm_03_z.npy',
	'data_all_hopfs_basis\\hopf_spectr_before_1e-40_30oneX.npy',
	'data_all_hopfs_basis\\hopf_spectr_before_1e-40_15oneZ.npy',
	'data_all_hopfs_basis\\hopf_spectr_before_1e-40_trefoil_standard_12.npy',
	'data_all_hopfs_basis\\hopf_spectr_before_1e-40_trefoil_optimized.npy',
]

for path in foils_paths:
	foil4_spectrum_sorted = np.load(path)

	plot_shifted_paper_grid_spectrum(foil4_spectrum_sorted, -6, 6, 0, 6, every_ticks=False)
