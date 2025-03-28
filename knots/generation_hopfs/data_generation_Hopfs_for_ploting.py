import matplotlib.pyplot as plt

from functions.all_knots_functions import *
import os
import pickle
import csv
import json
from tqdm import trange
import itertools
from matplotlib.colors import LinearSegmentedColormap


def filter_isolated_points_1(dots):
	"""
	Filters out points that do not have any neighbors within a 1-integer radius.

	Parameters:
		dots (numpy.ndarray): Array of integer coordinates with shape (N, 3).

	Returns:
		numpy.ndarray: Filtered array containing only points with at least one neighbor.
	"""
	dots_set = set(map(tuple, dots))  # Convert to set for faster lookup
	filtered_dots = []
	
	for dot in dots:
		x, y, z = dot
		# Check all 26 possible neighbors
		has_neighbors = any(
			(x + dx, y + dy, z + dz) in dots_set
			for dx in [-1, 0, 1]
			for dy in [-1, 0, 1]
			for dz in [-1, 0, 1]
			if not (dx == 0 and dy == 0 and dz == 0)
		)
		if has_neighbors:
			filtered_dots.append(dot)
	
	return np.array(filtered_dots)


def filter_isolated_points(dots):
	"""
	Filters out points that do not have any neighbors within a 2-integer radius.

	Parameters:
		dots (numpy.ndarray): Array of integer coordinates with shape (N, 3).

	Returns:
		numpy.ndarray: Filtered array containing only points with at least one neighbor within a 2-step radius.
	"""
	filtered_dots = []
	dots_set = set(map(tuple, dots))  # Convert dots to a set for fast lookup
	
	for dot in dots:
		x, y, z = dot
		neighbor_count = 0  # Initialize neighbor count
		
		# Check all possible neighbors within a 2-step radius
		for dx in range(-5, 6):  # -2 to 2 (inclusive)
			for dy in range(-5, 6):
				for dz in range(-5, 6):
					if dx == 0 and dy == 0 and dz == 0:  # Skip the current dot itself
						continue
					if (x + dx, y + dy, z + dz) in dots_set:
						neighbor_count += 1
					if neighbor_count > 1:  # Stop checking once we find more than 2 neighbors
						filtered_dots.append(dot)
						break
				if neighbor_count > 1:
					break
			if neighbor_count > 1:
				break
	
	return np.array(filtered_dots)


original_cmap = plt.cm.YlGnBu
original_cmap = plt.cm.tab20c
white_fraction = 0.02  # 1% for white

# Extract the colors from the original colormap and add white at the beginning
colors = original_cmap(np.linspace(0, 1, 256))
colors[:int(256 * white_fraction)] = [1, 1, 1, 1]  # Set the first 1% to white

# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list("CustomYlGnBu", colors)
custom_blues = custom_cmap
custom_blues = plt.cm.gist_earth
foils = list(itertools.product(range(3), repeat=4))

SAMPLES = 1
indx_plus = 0

plot = 0
plot_short = 0
plot_3d = 0
print_coeff = 0

print_values = 0
centering = 0
seed = None  # does work with more than 1 phase screen
# seed = 1
no_last_plane = True

# folder = 'data_basis_delete'
# folder = 'data_no_centers_32114'
# folder = 'data_low_10'

spectrum_save = 1
no_turb = 0

# meshes and boundaries for getting a knot
x_lim_3D_knot, y_lim_3D_knot, z_lim_3D_knot = (-7.0, 7.0), (-7.0, 7.0), (-2.0, 2.0)
res_x_3D_knot, res_y_3D_knot, res_z_3D_knot = 256, 256, 1

# beam
lmbda = 532e-9  # wavelength
L_prop = 270  # propagation distance
knot_length = 212.58897655870774 / 2 * 1.4  # we need RALEYIG!!!!!!!!  # 1000 how far is detector from the knot center
center_plane = 0.3
center_plane = 1
########################################################
width0 = 6.0e-3 / np.sqrt(2)  # beam width
xy_lim_2D_origin = (-30.0e-3, 30.0e-3)  # window size to start with
# xy_lim_2D_origin = (-50.0e-3, 50.0e-3)  # window size to start with
scale = 2
res_xy_2D_origin = int(scale * 300)  # resolution

res_z = int(scale * 64)  # resolution of the knot is res_z+1
res_z = int(scale * 64)  # resolution of the knot is res_z+1
crop = int(scale * 185)  # for the knot propagation
crop_3d = int(scale * 100)  # for the knot
new_resolution = (int(scale * 100), int(scale * 100))  # resolution of the knot to save
# new_resolution = (32, 32)  # resolution of the knot to save

screens_num1 = 3
multiplier1 = [1] * screens_num1
screens_num2 = 1
multiplier2 = [1] * screens_num2

knot_types = {
	'standard_14': hopf_standard_14,  # 1
	'standard_16': hopf_standard_16,  # 2
	'standard_18': hopf_standard_18,  # 3
	'30both': hopf_30both,  # 4
	'30oneZ': hopf_30oneZ,  # 5
	'optimized': hopf_optimized,  # 6
	'pm_03_z': hopf_pm_03_z,  # 7
	# '4foil': hopf_4foil,  # 8
	# '6foil': hopf_6foil,  # 9
	# 'stand4foil': hopf_stand4foil,  # 10
	'30oneX': hopf_30oneX,  # 11
	'15oneZ': hopf_15oneZ,
	'dennis': hopf_dennis,
	'trefoil_standard_12': trefoil_standard_12,
	'trefoil_optimized': trefoil_optimized,
}
# turbulence
# Cn2 = 1.35e-13  # turbulence strength  is basically in the range of 10−17–10−12 m−2/3
# Cn2 = 3.21e-14
# Cn2s = [5e-15, 1e-14, 5e-15, 1e-13]
# Cn2 = Cn2s[0]
knots = [
	'standard_14', 'standard_16', 'standard_18', '30both', '30oneZ',
	'optimized', 'pm_03_z', '30oneX', '15oneZ', 'dennis',
	'trefoil_standard_12', 'trefoil_optimized'
]
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
	# 'dennis',
	'trefoil_standard_12',
	'trefoil_optimized',
]
Rytovs = [0.05]  # , 0.2]
Rytovs = [0.03, 0.052, 0.091]  # 135
# Rytovs = [0.0000005]  # 540
Rytovs = [0.05, 0.15, 0.25]
Rytovs = [0.05, 0.15, 0.25]
# Rytovs = [0.00000000000000000000015]
# Rytovs = [1e-40]

for Rytov in Rytovs:
	# folder = f'standard_vs_WWW_trefoil_vs_rytov_{Rytov}_100_1.4zR_c03_v1'
	# folder = f'4foils_L{L_prop}_{Rytov}_{SAMPLES}_64x64x64_v1'
	# folder = f'optimized_trefoil_vs_rytov_{Rytov}_100_center_plane_v2'
	k0 = 2 * np.pi / lmbda  # wave number
	Cn2 = Cn2_from_Rytov(Rytov, k0, L_prop)
	# # # # # Cn2 = 3.21e-15
	# Cn2 = 3.21e-40
	# https://www.mdpi.com/2076-3417/11/22/10548
	L0 = 9 * 1e10  # outer scale
	l0 = 5e-3 * 1e-10  # inner scale
	
	# propagation parameters
	z0 = knot_length * (1 - center_plane) + L_prop  # the source position
	prop1 = L_prop  # z0-prop1 - detector position
	prop2 = knot_length * (1 - center_plane)  # z0-prop1-pro2 - knot center (assumed)
	
	# extra values (physical)
	
	beam_par = (0, 0, width0, lmbda)
	
	# extra values (simulations)
	x_3D_knot, y_3D_knot = np.linspace(*x_lim_3D_knot, res_x_3D_knot), np.linspace(*y_lim_3D_knot, res_y_3D_knot)
	if res_z_3D_knot != 1:
		z_3D_knot = np.linspace(*z_lim_3D_knot, res_z_3D_knot)
	else:
		z_3D_knot = 0
	mesh_3D_knot = np.meshgrid(x_3D_knot, y_3D_knot, z_3D_knot, indexing='ij')
	x_2D_origin = np.linspace(*xy_lim_2D_origin, res_xy_2D_origin)
	y_2D_origin = np.linspace(*xy_lim_2D_origin, res_xy_2D_origin)
	mesh_2D_original = np.meshgrid(x_2D_origin, y_2D_origin, indexing='ij')
	
	x_2D_origin_spec = np.linspace(*(np.array(xy_lim_2D_origin) * 1.2), res_xy_2D_origin)
	y_2D_origin_spec = np.linspace(*(np.array(xy_lim_2D_origin) * 1.2), res_xy_2D_origin)
	mesh_2D_original_spec = np.meshgrid(x_2D_origin_spec, y_2D_origin_spec, indexing='ij')
	# boundary_3D = [[0, 0, 0], [res_x_3D, res_y_3D, res_z_3D]]  # boundaries for 3d knot
	extend = [*xy_lim_2D_origin, *xy_lim_2D_origin]  # boundaries for 2d plot
	xy_lim_2D_crop = list(np.array(xy_lim_2D_origin) / res_xy_2D_origin * crop)
	extend_crop = [*xy_lim_2D_crop, *xy_lim_2D_crop]  # boundaries for 2d plot after crop
	xy_lim_2D_crop3d = list(np.array(xy_lim_2D_crop) / crop * crop_3d)
	extend_crop3d = [*xy_lim_2D_crop3d, *xy_lim_2D_crop3d]  # boundaries for 2d plot after crop3d
	pxl_scale = (xy_lim_2D_origin[1] - xy_lim_2D_origin[0]) / (res_xy_2D_origin - 1)
	D_window = (xy_lim_2D_origin[1] - xy_lim_2D_origin[0])
	perfect_scale = lmbda * np.sqrt(L_prop ** 2 + (D_window / 2) ** 2) / D_window
	if print_values:
		print(f'dx={pxl_scale * 1e6: .2f}um, perfect={perfect_scale * 1e6: .2f}um,'
		      f' resolution required={math.ceil(D_window / perfect_scale + 1)}')
	
	# extra values (turbulence)
	r0 = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=L_prop)
	if print_values:
		print(f'r0 parameter: {r0}, 2w0/r0={2 * width0 / r0}')
	screens_num = screens_number(Cn2, k0, dz=L_prop)
	if print_values:
		print(f'Number of screen required: {screens_num}')
	ryt = rytov(Cn2, k0, L_prop)
	if print_values:
		print(f'SR={np.exp(-ryt)} (Rytov), Rytov={ryt}')
	zR = (k0 * width0 ** 2)
	if print_values:
		print(f'Rayleigh Range (Zr) = {zR} (m)')
	psh_par = (r0, res_xy_2D_origin, pxl_scale, L0, l0)
	psh_par_0 = (r0 * 1e100, res_xy_2D_origin, pxl_scale, L0, l0 * 1e100)
	
	if no_turb:
		psh_par = psh_par_0
	#
	# folder_path = os.path.join("..", folder)
	# if not os.path.exists(folder_path):
	# 	os.makedirs(folder_path)
	# folder_path = os.path.join("..", folder, "fields")
	# if not os.path.exists(folder_path):
	# 	os.makedirs(folder_path)
	#
	# getting the knot
	for knot in knots:
		print(knot)
		values = knot_types[knot](mesh_3D_knot, braid_func=braid, plot=False)
		# if os.path.exists(f'../{folder}/{knot}.pkl'):
		#     with open(f'../{folder}/{knot}.pkl', 'rb') as file:
		#         values = pickle.load(file)
		# else:
		#     values = unknot_4_any(mesh_3D_knot, braid_func=braid, plot=True,
		#                       angle_size=foil, cmap=custom_blues)
		#     with open(f'../{folder}/{knot}.pkl', 'wb') as file:
		#         pickle.dump(values, file)
		# printing values
		if print_coeff:
			values_print = np.real(values['weight']) / np.sqrt(np.sum(np.real(values['weight']) ** 2)) * 10
			values_print_formatted = [f'{val:.2f}' for val in values_print]
			print(f'vales l: {values["l"]}, p: {values["p"]}, weights: {values_print_formatted}')
		
		# building the knot from the coefficients
		field_before_prop = field_knot_from_weights(
			values, mesh_2D_original, width0, k0=k0, x0=0, y0=0, z0=z0
		)
		field_before_prop_spec_before = field_knot_from_weights(
			values, mesh_2D_original_spec, width0, k0=k0, x0=0, y0=0, z0=z0
		)
		p1, p2 = 0, 6
		l1, l2 = -6, 6
		
		moments = {'p': (p1, p2), 'l': (l1, l2)}
		# mesh_2D =
		spectrum1 = cbs.LG_spectrum(
			field_before_prop_spec_before, **moments, mesh=mesh_2D_original_spec, plot=False, width=width0, k0=k0,
			functions=LG_simple, x0=0, y0=0, z0=z0
		)
		# print(values)
		spectrum1 = spectrum1 / np.sqrt(np.sum(np.abs(spectrum1) ** 2)) * 100
		vmin, vmax = 0, np.abs(spectrum1).max()
		# print(spectrum)
		pl.plot_2D(np.abs(spectrum1),
		           x=np.arange(l1 - 0.5, l2 + 1 + 0.5),
		           y=np.arange(p1 - 0.5, p2 + 1 + 0.5),
		           interpolation='none', grid=True, xname='l', yname='p', map=custom_blues, show=False,
		           vmin=vmin, vmax=vmax)
		plt.yticks(np.arange(p1, p2 + 1))
		plt.xticks(np.arange(l1, l2 + 1))
		plt.show()
		np.save(f'hopf_spectr_before_{Rytov}_{knot}.npy', spectrum1)
		
		# if plot:
		# plot_field_both(field_before_prop)
		for indx in trange(SAMPLES, desc="Progress"):
			# propagating in the turbulence prop1
			field_after_turb = propagation_ps(
				field_before_prop, beam_par, psh_par, prop1, multiplier=multiplier1, screens_num=screens_num1, seed=seed
			)
			# if plot and not plot_short:
			#     plot_field_both(field_after_turb, extend=extend)
			if center_plane == 1:
				field_center = field_after_turb
			else:
				field_center = propagation_ps(
					field_after_turb, beam_par, psh_par_0, prop2, multiplier=multiplier2, screens_num=screens_num2,
					seed=seed
				)
			###########################
			field_center = field_center / np.sqrt(np.sum(np.abs(field_center) ** 2))
			###########################
			# if plot and not plot_short:
			# plot_field_both(field_center, extend=extend)
			
			field_z_crop = field_center[
			               res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
			               res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
			               ]
			# if 1:
			# 	filename = f'../{folder}/fields/data_{knot}_{indx + indx_plus}.npy'  # l rows, p columns
			#
			# 	np.save(filename, field_z_crop)
			if plot:
				plot_field_both(field_z_crop, extend=extend_crop)
			np.save(f'hopf_field_{Rytov}_{knot}.npy', field_z_crop)
			if spectrum_save:
				if centering:
					x_cent_R_big, y_cent_R_big = find_center_of_intensity(field_center)
					x_cent_big, y_cent_big = x_cent_R_big, y_cent_R_big
					
					x_cent_big_r = x_2D_origin[0] + (x_2D_origin[-1] - x_2D_origin[0]) / res_xy_2D_origin * round(
						x_cent_big)
					
					y_cent_big_r = y_2D_origin[0] + (y_2D_origin[-1] - y_2D_origin[0]) / res_xy_2D_origin * round(
						y_cent_big)
				else:
					x_cent_big_r = 0
					y_cent_big_r = 0
				
				# mesh_2D =
				spectrum = cbs.LG_spectrum(
					field_center, **moments, mesh=mesh_2D_original, plot=False, width=width0, k0=k0,
					functions=LG_simple, x0=x_cent_big_r, y0=y_cent_big_r
				)
				# print(values)
				print('\n', np.sum(np.abs(values['weight']) ** 2))
				spectrum = spectrum / np.sqrt(np.sum(np.abs(spectrum) ** 2)) * 100
				# print(spectrum)
				# pl.plot_2D(np.abs(spectrum),
				#            x=np.arange(l1 - 0.5, l2 + 1 + 0.5),
				#            y=np.arange(p1 - 0.5, p2 + 1 + 0.5),
				#            interpolation='none', grid=True, xname='l', yname='p', map=custom_blues,
				#            vmin=vmin, vmax=vmax,show=False)
				# plt.yticks(np.arange(p1, p2 + 1))
				# plt.xticks(np.arange(l1, l2 + 1))
				pl.plot_2D(np.abs(spectrum),
				           x=np.arange(l1 - 0.5, l2 + 1 + 0.5),
				           y=np.arange(p1 - 0.5, p2 + 1 + 0.5),
				           interpolation='none', grid=True, xname='l', yname='p', map=custom_blues,
				           show=False)
				plt.yticks(np.arange(p1, p2 + 1))
				plt.xticks(np.arange(l1, l2 + 1))
				plt.show()
				np.save(f'hopf_spectr_{Rytov}_{knot}.npy', spectrum)
				
				# if 1:
				#     filename = f'../{folder}/data_{knot}_spectr.csv'  # l rows, p columns
				#     spectrum_list = (
				#             [moments['l'][0], moments['l'][1], moments['p'][0], moments['p'][1]] + [indx + indx_plus] +
				#             [[x.real, x.imag] for x in spectrum.flatten()]
				#     )
				#     dots_json = json.dumps(spectrum_list)
				#     with open(filename, 'a', newline='') as file:
				#         writer = csv.writer(file)
				#         writer.writerow([dots_json])
			# plot_field_both(field_center)
			# exit()
			field_3d = beam_expander(field_z_crop, beam_par, psh_par_0, distance_both=knot_length, steps_one=res_z // 2)
			
			if centering:
				x_cent_R, y_cent_R = find_center_of_intensity(field_z_crop)
				x_cent, y_cent = int(x_cent_R), int(y_cent_R)
			else:
				x_cent, y_cent = crop // 2, crop // 2
			
			if print_values:
				print(f'Centers: {x_cent}, {y_cent} out of {crop}')
			
			field_3d_crop = field_3d[
			                x_cent - crop_3d // 2: x_cent + crop_3d // 2,
			                y_cent - crop_3d // 2: y_cent + crop_3d // 2,
			                :
			                ]
			if no_last_plane:
				field_3d_crop = field_3d_crop[:, :, :-1]
			if plot_3d and not plot_short:
				plot_field_both(field_3d_crop[:, :, res_z // 2], extend=extend_crop3d)
			dots_init_dict, dots_init = sing.get_singularities(np.angle(field_3d_crop), axesAll=False, returnDict=True)
			
			dots_cut_non_unique = cut_circle_dots(dots_init, crop_3d // 2, crop_3d // 2, crop_3d // 2)
			
			# check if there is no same points
			# Creating a view with a compound data type
			view = np.ascontiguousarray(dots_cut_non_unique).view(
				np.dtype((np.void, dots_cut_non_unique.dtype.itemsize * dots_cut_non_unique.shape[1]))
			)
			# Using np.unique with the view
			_, idx = np.unique(view, return_index=True)
			dots_cut = dots_cut_non_unique[idx]
			
			# decreasing the resolution of knots
			# knot_resolution = [crop_3d, crop_3d, res_z + 1]
			original_resolution = (crop_3d, crop_3d)
			
			scale_x = new_resolution[0] / original_resolution[0]
			scale_y = new_resolution[1] / original_resolution[1]
			xy = dots_cut[:, :2]  # First two columns (x and y)
			z = dots_cut[:, 2]  # Third column
			scaled_xy = xy * [scale_x, scale_y]
			scaled_xy = np.rint(scaled_xy).astype(int)
			scaled_data = np.column_stack((scaled_xy, z))
			dots_cut = filter_isolated_points(dots_cut)
			np.save(f'hopf_dots_{Rytov}_{knot}.npy', dots_cut)
			# print(len(dots_cut))
			################################################################################################
			# print(len(dots_cut))
			if plot_3d:
				dots_bound = [
					[0, 0, 0],
					[crop_3d, crop_3d, res_z + 1],
				]
				pl.plotDots(dots_cut, dots_bound, color='black', show=True, size=10)
			
			# break
			# if no_last_plane:
			#     knot_resolution = [new_resolution[0], new_resolution[0], res_z]
			# else:
			#     knot_resolution = [new_resolution[0], new_resolution[0], res_z + 1]
			# dots_cut_modified = np.vstack([[indx + indx_plus, 0, 0], knot_resolution, scaled_data])
			# if 1:
			#     filename = f'../{folder}/data_{knot}.csv'
			#     dots_json = json.dumps(dots_cut_modified.tolist())
			#     with open(filename, 'a', newline='') as file:
			#         writer = csv.writer(file)
			#         writer.writerow([dots_json])
