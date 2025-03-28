from project_flower_beams.paper_plots_flowers_ML.plots_functions_general import *

knots = [
	'standard_14',  # 1
	'standard_16',  # 2
	'standard_18',  # 3
	'30both',  # 4
	'30oneZ',  # 5
	'optimized',  # 6
	'pm_03_z',  # 7
	'30oneX',  # 11
	'15oneZ',
	'trefoil_standard_12',
	'trefoil_optimized',
]
foils_paths = [
	'data_all_hopfs_basis\\hopf_field_1e-40_standard_14.npy',
	'data_all_hopfs_basis\\hopf_field_1e-40_standard_16.npy',
	'data_all_hopfs_basis\\hopf_field_1e-40_standard_18.npy',
	'data_all_hopfs_basis\\hopf_field_1e-40_30both.npy',
	'data_all_hopfs_basis\\hopf_field_1e-40_30oneZ.npy',
	'data_all_hopfs_basis\\hopf_field_1e-40_optimized.npy',
	'data_all_hopfs_basis\\hopf_field_1e-40_pm_03_z.npy',
	'data_all_hopfs_basis\\hopf_field_1e-40_30oneX.npy',
	'data_all_hopfs_basis\\hopf_field_1e-40_15oneZ.npy',
	'data_all_hopfs_basis\\hopf_field_1e-40_trefoil_standard_12.npy',
	'data_all_hopfs_basis\\hopf_field_1e-40_trefoil_optimized.npy',

]

foils_paths = [
	'data_all_hopfs_basis\\hopf_field_1e-40_standard_14.npy',
]
for path in foils_paths:
	foil4_field = np.load(path)
	foil4_field = foil4_field / np.max(np.abs(foil4_field))
	XY_max = (30e-3 * 185 / 300 * 1e3) / 6 * np.sqrt(2)
	X = [-XY_max, XY_max]
	Y = [-XY_max, XY_max]
	# plot_field_both_paper(foil4_field, extend=[*X, *Y], colorbars='amplitude')
	plot_field_both_paper(foil4_field, extend=[*X, *Y], colorbars='both')
