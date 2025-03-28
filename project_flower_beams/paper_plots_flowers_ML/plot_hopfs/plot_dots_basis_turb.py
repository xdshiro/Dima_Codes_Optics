from project_flower_beams.paper_plots_flowers_ML.plots_functions_general import *



sigma = 0.25
foils_paths = [
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_standard_14.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_standard_16.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_standard_18.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_30both.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_30oneZ.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_optimized.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_pm_03_z.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_30oneX.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_15oneZ.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_trefoil_standard_12.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_trefoil_optimized.npy',

]
foils_paths = [
	# f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_standard_14.npy',
	### f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_standard_16.npy',
	# f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_standard_18.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_30both.npy',
	# f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_30oneZ.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_optimized.npy',
	# f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_pm_03_z.npy',
	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_30oneX.npy',
	# f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_15oneZ.npy',
	# f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_trefoil_standard_12.npy',
	# f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_trefoil_optimized.npy',
]
# foils_paths = [
# 	# f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_standard_14.npy',
# 	f'data_all_hopfs_basis_turb\\hopf_dots_{sigma}_standard_16.npy',
#
# ]


for path in foils_paths:
	# foil4_dots_sorted = np.roll(np.load(path), 100, axis=0)
	foil4_dots = np.load(path) - np.array([100, 100, 0])
	dots_bound = [
		[-100, -100, 0],
		[100, 100, 129],
	]

	plot_black_dots_paper(foil4_dots, dots_bound=dots_bound, general_view=True)
