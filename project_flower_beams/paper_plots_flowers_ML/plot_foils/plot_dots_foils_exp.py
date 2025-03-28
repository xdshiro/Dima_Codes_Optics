from project_flower_beams.paper_plots_flowers_ML.plots_functions_general import *



foils_paths = [
	'data_foils_exp/foil4_2222_exp_turb_dots.npy',
]
# foils_paths = [
# 	'data_foils_exp/foil4_2222_exp_noturb_dots.npy',
# ]
# foils_paths = [
# 	'data_foils_exp/foil4_dots_XY_noturb_[2, 2, 2, 2]_sorted.npy',
# ]





for path in foils_paths:
	# foil4_dots_sorted = np.roll(np.load(path), 100, axis=0)
	# foil4_dots = np.load(path) - np.array([50, 50, 0])
	# exp
	foil4_dots = (np.load(path) - np.array([50, 50, 32])) * [1, 1, -1] + np.array([0, 0, 32])
	# foil4_dots = np.load(path) #- np.array([100, 100, 0])
	# print(foil4_dots)
	# exit()
	dots_bound = [
		[-50, -50, 0],
		[50, 50, 65],
	]
	# foil4_dts_sorted = sort_dots_to_create_line_with_threshold(foil4_dots, TH=30)
	# np.save('foil4_dots_XY_noturb_[2, 2, 2, 2]_sorted.npy', foil4_dots_sorted)
	# plot_3d_line(foil4_dots_sorted)
	plot_black_dots_paper(foil4_dots.T, dots_bound=dots_bound)
	plot_black_dots_paper(foil4_dots, dots_bound=dots_bound, general_view=True)
