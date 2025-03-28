from project_flower_beams.paper_plots_flowers_ML.plots_functions_general import *


foils_paths = [
	# 'hopf_dots_0.05_trefoil_standard_12.npy',
	'data_trefoils_turb\\hopf_dots_0.15_trefoil_standard_12.npy',
	# 'hopf_dots_0.25_trefoil_standard_12.npy',

]

for path in foils_paths:
	foil4_dots = np.load(path) - np.array([100, 100, 0])
	# element = [-10, -56, 122]
	# positions = np.where((foil4_dots == element).all(axis=1))[0]
	# foil4_dots = np.delete(foil4_dots, positions, axis=0)
	# np.save('hopf_dots_0.15_trefoil_standard_12.npy', foil4_dots + 1 * np.array([100, 100, 0]))
	# exit()
	dots_bound = [
		[-100, -100, 0],
		[100, 100, 129],
	]
	
	plot_black_dots_paper(foil4_dots, dots_bound=dots_bound, general_view=True, font_size=48)
	# plot_black_dots_paper(foil4_dots, dots_bound=dots_bound, general_view=False, font_size=48)
