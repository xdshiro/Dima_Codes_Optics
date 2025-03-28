from flower_beams.paper_plots_flowers_ML.plots_functions_general import *

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

sigma = 0.25
foils_paths = [
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_standard_14.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_standard_16.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_standard_18.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_30both.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_30oneZ.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_optimized.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_pm_03_z.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_30oneX.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_15oneZ.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_trefoil_standard_12.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_trefoil_optimized.npy',

]
foils_paths = [
	# f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_standard_14.npy',
	### f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_standard_16.npy',
	# f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_standard_18.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_30both.npy',
	# f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_30oneZ.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_optimized.npy',
	# f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_pm_03_z.npy',
	f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_30oneX.npy',
	# f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_15oneZ.npy',
	# f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_trefoil_standard_12.npy',
	# f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_trefoil_optimized.npy',
]


for path in foils_paths:
	foil4_field = np.load(path)
	foil4_field = foil4_field / np.max(np.abs(foil4_field))
	XY_max = 30e-3 * 185 / 300 * 1e3 / 6 * np.sqrt(2)
	X = [-XY_max, XY_max]
	Y = [-XY_max, XY_max]
	plot_field_both_paper(foil4_field, extend=[*X, *Y], colorbars='both')
