"""
This script reads the field from a .mat file, performs necessary pre-processing procedures, and
creates a 3D array of singularity dots.

## Main Functions

### main_field_processing
1) Reads the field from a MATLAB file.
2) Converts the field into a numpy array.
3) Normalizes the field.
4) Finds the beam waist.
5) Rescales the field using interpolation for faster processing.
6) Finds the beam center.
7) Rescales the field to the desired scale for 3D calculations.
8) Removes the tilt and shift.

### main_dots_building
1) Reads a 2D field and plots it.
2) Performs double-sided propagation to generate a 3D field.
3) Crops the 3D field for faster singularity calculation.
4) Detects singularity dots in the 3D field using cross-sections.
5) Crops the dots within a specified radius.
6) Moves the dots to the corner to create a smaller 3D array.
7) Applies dot simplification algorithms to reduce the number of dots.

## Import Statements

- External Libraries:
  - os: For handling directory operations.
  - math: For mathematical operations.
  - numpy: For numerical operations.
  - pandas: For data manipulation.
  - scipy.io: For handling MATLAB files.
  - matplotlib.pyplot: For plotting.

- Custom Modules:
  - my_functions.singularities: Functions for handling singularities.
  - my_functions.functions_general: General utility functions.
  - functions.dots_processing: Functions for processing dots.
  - functions.center_beam_search: Functions for beam search.
  - my_functions.beams_and_pulses: Functions related to beams and pulses.

## Functions

### read_field_2D_single
Reads a 2D array from a MATLAB file and converts it into a numpy array.

### normalization_field
Normalizes the field for beam center finding.

### plot_field
Plots the intensity and phase of a field in a single plot.

### plot_field_3D_multi_planes
Plots the intensity and phase of a 3D field at different z-planes.

### find_beam_waist
Finds the approximate beam waist using a specified mesh.

    This function uses the `find_width` function from the `center_beam_search` module to approximate the beam waist.
    If a mesh grid is not provided, it creates one using the field dimensions.

    Parameters:
        field (np.ndarray): 2D field to find the beam waist.
        mesh (tuple, optional): Mesh grid for the field. If None, it will be created automatically.

    Returns:
        float: Approximate beam waist value.

### field_interpolation
Interpolates a field to a new resolution.

    This function is a wrapper for `interpolation_complex` from the `functions_general` module. It interpolates the
    given field to the specified resolution and rescales its dimensions according to the provided fractions.

    Parameters:
        field (np.ndarray): Input 2D field to be interpolated.
        mesh (tuple, optional): Mesh grid for the field. If None, it will be created automatically.
        resolution (tuple): New resolution for the field.
        xMinMax_frac (tuple): Fraction of the original x dimension to keep.
        yMinMax_frac (tuple): Fraction of the original y dimension to keep.
        fill_value (bool): Fill value for interpolation.

    Returns:
        tuple: Interpolated field and the new mesh grid.

### one_plane_propagator
Performs double-sided propagation for generating a 3D field from a 2D field.

    This function uses double-sided propagation to generate a 3D field from an input 2D field. It propagates the field
    in both positive and negative z-directions.

    Parameters:
        field (np.ndarray): 2D complex field to propagate.
        dz (float): Step size along the z-axis.
        stepsNumber_p (int): Number of forward propagation steps.
        stepsNumber_m (int, optional): Number of backward propagation steps. If None, set equal to `stepsNumber_p`.
        n0 (float): Refractive index.
        k0 (float): Wave number.

    Returns:
        np.ndarray: 3D complex field resulting from the propagation.

### dots_rounded
Rounds the coordinates of the dots for further processing.

### files_list
Generates a list of files with a specified extension in a directory.

The `__main__` section provides examples and is mainly used for testing and processing specific directories and files.

Example Usage:
    The following example demonstrates the process of reading a field, processing it, and creating singularity dots:

    ```python
    directory_field = f'data/test/'
    directory_field_saved = f'data/test/saved/'
    if not os.path.isdir(directory_field_saved):
        os.makedirs(directory_field_saved)

    files = files_list(directory_field, end='.mat')
    for file in files:
        field2D, _ = main_field_processing(
            path=directory_field + file,
            plotting=False,
            resolution_iterpol_center=(100, 100),
            xMinMax_frac_center=(1, 1),
            yMinMax_frac_center=(1, 1),
            resolution_interpol_working=(115, 115),
            xMinMax_frac_working=(1.2, 1.2),
            yMinMax_frac_working=(1.2, 1.2),
            resolution_crop=(100, 100),
            moments_init={'p': (0, 5), 'l': (-4, 4)},
            moments_center={'p0': (0, 5), 'l0': (-4, 4)},
        )
        file_save = directory_field_saved + file[:-4] + '.npy'
        np.save(file_save, field2D)
    ```
    
    This example reads `.mat` files, processes the fields, and saves the output as `.npy` files for further usage.
"""

import os
import math
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import extra_functions_package.singularities as sing
import extra_functions_package.functions_general as fg
import extra_functions_package.dots_processing as dp
import extra_functions_package.center_beam_search as cbs
from os import listdir
from os.path import isfile, join


def read_field_2D_single(path, field=None):
	"""
	Reads a 2D array from a MATLAB (.mat) file and converts it into a numpy array.

	If the field name is not specified, the function will automatically try to determine the correct field.

	Parameters:
		path (str): Full path to the .mat file.
		field (str, optional): The name of the field to read. If None, the function will try to determine it automatically.

	Returns:
		np.ndarray: 2D array representing the field.
	"""
	field_read = sio.loadmat(path, appendmat=False)
	if field is None:
		for field_check in field_read:
			if len(np.shape(np.array(field_read[field_check]))) == 2:
				field = field_check
				break
	return np.array(field_read[field])


def normalization_field(field):
	"""
	Normalizes the input field for beam center finding.

	Parameters:
		field (np.ndarray): Input field to be normalized.

	Returns:
		np.ndarray: Normalized field.
	"""
	field_norm = field / np.sqrt(np.sum(np.abs(field) ** 2))
	return field_norm


def plot_field(field, save=None):
	"""
	Plots the intensity and phase of a 2D field.

	Parameters:
		field (np.ndarray): Input field to plot.
		save (str, optional): File path to save the plot. If None, the plot will be displayed but not saved.
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
	image1 = ax1.imshow(np.abs(field))
	ax1.set_title('|E|')
	plt.colorbar(image1, ax=ax1, shrink=0.4, pad=0.02, fraction=0.1)
	image2 = ax2.imshow(np.angle(field), cmap='jet')
	ax2.set_title('Phase(E)')
	plt.colorbar(image2, ax=ax2, shrink=0.4, pad=0.02, fraction=0.1)
	plt.tight_layout()
	if save is not None:
		fig.savefig(save, format='png')
	plt.show()


def plot_field_3D_multi_planes(field3D, number=6, columns=3):
	"""
	Plots the intensity and phase of a 3D field at different z-planes.

	Parameters:
		field3D (np.ndarray): 3D complex field to plot.
		number (int): Number of slices to plot.
		columns (int): Number of columns for subplot arrangement.
	"""
	fig, axis = plt.subplots(math.ceil(number / columns), columns, figsize=(10, 3 * math.ceil(number / columns)))
	reso_z = np.shape(field3D)[2]
	for i, ax_r in enumerate(axis):
		for j, ax in enumerate(ax_r):
			index = int((reso_z - 1) / (number - 1) * (columns * i + j))
			image = ax.imshow(np.abs(field3D[:, :, index]))
			ax.set_title(f'|E|, index z={index}')
			plt.colorbar(image, ax=ax, shrink=0.4, pad=0.02, fraction=0.1)
	plt.tight_layout()
	plt.show()
	
	fig, axis = plt.subplots(math.ceil(number / columns), columns, figsize=(10, 3 * math.ceil(number / columns)))
	for i, ax_r in enumerate(axis):
		for j, ax in enumerate(ax_r):
			index = int((reso_z - 1) / (number - 1) * (columns * i + j))
			image = ax.imshow(np.angle(field3D[:, :, index]), cmap='jet')
			ax.set_title(f'Phase(E), index z={index}')
			plt.colorbar(image, ax=ax, shrink=0.4, pad=0.02, fraction=0.1)
	plt.tight_layout()
	plt.show()


def find_beam_waist(field, mesh=None):
	"""
	Finds the approximate beam waist using a specified mesh.

	This function uses the `find_width` function from the `center_beam_search` module to approximate the beam waist.
	If a mesh grid is not provided, it creates one using the field dimensions.

	Parameters:
		field (np.ndarray): 2D field to find the beam waist.
		mesh (tuple, optional): Mesh grid for the field. If None, it will be created automatically.

	Returns:
		float: Approximate beam waist value.
	"""
	shape = np.shape(field)
	if mesh is None:
		mesh = fg.create_mesh_XY(xRes=shape[0], yRes=shape[1])
	width = cbs.find_width(field, mesh=mesh, width=shape[1] // 8, widthStep=1, print_steps=False)
	return width


def field_interpolation(field, mesh=None, resolution=(100, 100),
                        xMinMax_frac=(1, 1), yMinMax_frac=(1, 1), fill_value=True):
	"""
	Interpolates a field to a new resolution.

	This function is a wrapper for `interpolation_complex` from the `functions_general` module. It interpolates the
	given field to the specified resolution and rescales its dimensions according to the provided fractions.

	Parameters:
		field (np.ndarray): Input 2D field to be interpolated.
		mesh (tuple, optional): Mesh grid for the field. If None, it will be created automatically.
		resolution (tuple): New resolution for the field.
		xMinMax_frac (tuple): Fraction of the original x dimension to keep.
		yMinMax_frac (tuple): Fraction of the original y dimension to keep.
		fill_value (bool): Fill value for interpolation.

	Returns:
		tuple: Interpolated field and the new mesh grid.
	"""
	shape = np.shape(field)
	if mesh is None:
		mesh = fg.create_mesh_XY(xRes=shape[0], yRes=shape[1])
	interpol_field = fg.interpolation_complex(field, mesh=mesh, fill_value=fill_value)
	xMinMax = int(-shape[0] // 2 * xMinMax_frac[0]), int(shape[0] // 2 * xMinMax_frac[1])
	yMinMax = int(-shape[1] // 2 * yMinMax_frac[0]), int(shape[1] // 2 * yMinMax_frac[1])
	xyMesh_interpol = fg.create_mesh_XY(
		xRes=resolution[0], yRes=resolution[1],
		xMinMax=xMinMax, yMinMax=yMinMax)
	return interpol_field(*xyMesh_interpol), xyMesh_interpol


def one_plane_propagator(field, dz, stepsNumber_p, stepsNumber_m=None, n0=1, k0=1):
	"""
	Performs double-sided propagation for generating a 3D field from a 2D field.

	This function uses double-sided propagation to generate a 3D field from an input 2D field. It propagates the field
	in both positive and negative z-directions.

	Parameters:
		field (np.ndarray): 2D complex field to propagate.
		dz (float): Step size along the z-axis.
		stepsNumber_p (int): Number of forward propagation steps.
		stepsNumber_m (int, optional): Number of backward propagation steps. If None, set equal to `stepsNumber_p`.
		n0 (float): Refractive index.
		k0 (float): Wave number.

	Returns:
		np.ndarray: 3D complex field resulting from the propagation.
	"""
	if stepsNumber_m is None:
		stepsNumber_m = stepsNumber_p
	fieldPropMinus = fg.propagator_split_step_3D_linear(field, dz=-dz, zSteps=stepsNumber_p, n0=n0, k0=k0)
	fieldPropPLus = fg.propagator_split_step_3D_linear(field, dz=dz, zSteps=stepsNumber_m, n0=n0, k0=k0)
	fieldPropTotal = np.concatenate((np.flip(fieldPropMinus, axis=2), fieldPropPLus[:, :, 1:]), axis=2)
	return fieldPropTotal


def main_field_processing(
		path,
		plotting=True,
		resolution_iterpol_center=(70, 70),
		xMinMax_frac_center=(1, 1),
		yMinMax_frac_center=(1, 1),
		resolution_interpol_working=(150, 150),
		xMinMax_frac_working=(1, 1),
		yMinMax_frac_working=(1, 1),
		resolution_crop=(120, 120),
		moments_init=None,
		moments_center=None,
):
	"""
	Processes the field data from a MATLAB file, applies normalization, interpolation, and tilt corrections.

	This function reads the field from a MATLAB file, normalizes it, finds the beam waist, rescales the field,
	and removes the tilt and shift. The processed field is then cropped around the beam center.

	Parameters:
		path (str): Path to the MATLAB file containing the field.
		plotting (bool): Whether to generate plots for visualization.
		resolution_iterpol_center (tuple): Resolution for finding the beam center.
		xMinMax_frac_center (tuple): Fraction of the original x dimension to keep for beam center finding.
		yMinMax_frac_center (tuple): Fraction of the original y dimension to keep for beam center finding.
		resolution_interpol_working (tuple): Final resolution before cropping.
		xMinMax_frac_working (tuple): Fraction of the original x dimension to keep for final processing.
		yMinMax_frac_working (tuple): Fraction of the original y dimension to keep for final processing.
		resolution_crop (tuple): Final resolution of the cropped field.
		moments_init (dict, optional): Initial moments for LG spectrum analysis.
		moments_center (dict, optional): Moments for beam center finding.

	Returns:
		tuple: Processed 2D complex field and the corresponding mesh.
	"""
	if moments_init is None:
		moments_init = {'p': (0, 6), 'l': (-4, 4)}
	if moments_center is None:
		moments_center = {'p0': (0, 4), 'l0': (-4, 2)}
	
	# Read the field from the MATLAB file
	field_init = read_field_2D_single(path)
	if plotting:
		plot_field(field_init)
	
	# Normalize the field
	field_norm = normalization_field(field_init)
	if plotting:
		plot_field(field_norm)
	
	# Create the initial mesh
	mesh_init = fg.create_mesh_XY(xRes=np.shape(field_norm)[0], yRes=np.shape(field_norm)[1])
	
	# Find the beam waist
	width = find_beam_waist(field_norm, mesh=mesh_init)
	if plotting:
		print(f'Approximate beam waist: {width}')
	
	# Interpolate the field for faster processing
	field_interpol, mesh_interpol = field_interpolation(
		field_norm, mesh=mesh_init, resolution=resolution_iterpol_center,
		xMinMax_frac=xMinMax_frac_center, yMinMax_frac=yMinMax_frac_center
	)
	if plotting:
		plot_field(field_interpol)
	
	# Rescale the beam width
	width = width / np.shape(field_norm)[0] * np.shape(field_interpol)[0]
	
	# Plot LG spectrum for selecting moments
	if plotting:
		_ = cbs.LG_spectrum(field_interpol.T, **moments_init, mesh=mesh_interpol, plot=plotting, width=width, k0=1)
	
	# Find the beam center
	x, y, eta, gamma = cbs.beamFullCenter(
		field_interpol, mesh_interpol,
		stepXY=(1, 1), stepEG=(3 / 180 * np.pi, 0.25 / 180 * np.pi),
		x=0, y=0, eta2=0., gamma=0.,
		**moments_center, threshold=1, width=width, k0=1, print_info=plotting
	)
	
	# Interpolate the field for 3D calculations
	field_interpol2, mesh_interpol2 = field_interpolation(
		field_norm, mesh=mesh_init, resolution=resolution_interpol_working,
		xMinMax_frac=xMinMax_frac_working, yMinMax_frac=yMinMax_frac_working, fill_value=False
	)
	if plotting:
		plot_field(field_interpol2)
	
	# Remove the tilt from the field
	field_untilted = cbs.removeTilt(field_interpol2, mesh_interpol2, eta=-eta, gamma=gamma, k=1)
	if plotting:
		plot_field(field_untilted)
	
	# Scale the beam center coordinates
	shape = np.shape(field_untilted)
	x = int(x / np.shape(field_interpol)[0] * shape[0])
	y = int(y / np.shape(field_interpol)[1] * shape[1])
	
	# Crop the field around the beam center
	field_cropped = field_untilted[
	                shape[0] // 2 - x - resolution_crop[0] // 2:shape[0] // 2 - x + resolution_crop[0] // 2,
	                shape[1] // 2 - y - resolution_crop[1] // 2:shape[1] // 2 - y + resolution_crop[1] // 2
	                ]
	if plotting:
		plot_field(field_cropped)
	
	# Create the final mesh
	mesh = fg.create_mesh_XY(xRes=np.shape(field_cropped)[0], yRes=np.shape(field_cropped)[1])
	field = field_cropped
	print(f'field finished: {path[-20:]}')
	return field, mesh


def main_dots_building(
		field2D,
		plotting=True,
		dz=5,
		steps_both=25,
		resolution_crop=(100, 100),
		r_crop=30
):
	"""
	Builds the 3D field and detects singularity dots.

	This function takes a 2D field, propagates it to create a 3D field, and detects singularity dots in the 3D field.
	It also applies cropping and dot simplification techniques.

	Parameters:
		field2D (np.ndarray): 2D complex field.
		plotting (bool): Whether to generate plots for visualization.
		dz (float): Step size along the z-axis for propagation.
		steps_both (int): Number of propagation steps in both directions.
		resolution_crop (tuple): Resolution for cropping the 3D field.
		r_crop (int): Radius for cropping dots.

	Returns:
		tuple: Raw and filtered singularity dots.
	"""
	if plotting:
		plot_field(field2D)
	
	# Perform double-sided propagation to create a 3D field
	field3D = one_plane_propagator(field2D, dz=dz, stepsNumber_p=steps_both, stepsNumber_m=None, n0=1, k0=1)
	if plotting:
		plot_field_3D_multi_planes(field3D, number=6, columns=3)
	
	# Crop the 3D field for faster dot calculation
	shape = np.shape(field3D)
	if shape[0] >= resolution_crop[0] and shape[1] >= resolution_crop[1]:
		field3D_cropped = field3D[
		                  shape[0] // 2 - resolution_crop[0] // 2:shape[0] // 2 + resolution_crop[0] // 2,
		                  shape[1] // 2 - resolution_crop[1] // 2:shape[1] // 2 + resolution_crop[1] // 2,
		                  :
		                  ]
	else:
		field3D_cropped = field3D
		print('Resolution is lower than the crop resolution')
	
	# Detect singularity dots in the 3D field
	dots_init_dict, dots_init = sing.get_singularities(np.angle(field3D_cropped), axesAll=True, returnDict=True)
	dp.plotDots(dots_init, dots_init, color='black', show=plotting, size=10)
	
	# Crop dots within a specified radius
	x0, y0 = resolution_crop[0] // 2, resolution_crop[1] // 2
	dots_cropped_dict = {dot: 1 for dot in dots_init_dict if
	                     np.sqrt((dot[0] - x0) ** 2 + (dot[1] - y0) ** 2) < r_crop}
	dp.plotDots(dots_cropped_dict, dots_cropped_dict, color='black', show=plotting, size=10)
	
	# Move dots to the corner to create a smaller 3D array
	dots_moved_dict = {
		(dot[0] - (x0 - r_crop), dot[1] - (y0 - r_crop), dot[2]): 1 for dot in dots_cropped_dict}
	dp.plotDots(dots_moved_dict, dots_moved_dict, color='grey', show=plotting, size=12)
	
	# Apply dot simplification algorithms
	dots_filtered = dp.filtered_dots(dots_moved_dict)
	dp.plotDots(dots_filtered, dots_filtered, color='grey', show=plotting, size=12)
	dots_filtered_rounded = dots_rounded(dots_filtered, resolution_crop, x0=0, y0=0, z0=0)
	dots_filtered_dict = {tuple(dot): 1 for dot in dots_filtered_rounded}
	dots_filtered_twice = dp.filtered_dots(dots_filtered_dict, single_dot=True)
	dp.plotDots(dots_filtered_twice, dots_filtered_twice, color='red', show=plotting, size=12)
	
	# Get raw dots after filtering
	dots_raw = dp.filtered_dots(dots_moved_dict, single_dot=True)
	return dots_raw, dots_filtered_twice


def dots_rounded(dots, resolution_crop, x0=None, y0=None, z0=0):
	"""
	Rounds the coordinates of the dots for further processing.

	Parameters:
		dots (np.ndarray): Array of dot coordinates to be rounded.
		resolution_crop (tuple): Resolution for cropping.
		x0 (int, optional): x-coordinate of the center. Defaults to half the resolution.
		y0 (int, optional): y-coordinate of the center. Defaults to half the resolution.
		z0 (int): z-coordinate offset. Defaults to 0.

	Returns:
		np.ndarray: Rounded coordinates of the dots.
	"""
	if x0 is None:
		x0 = resolution_crop[0] // 2
	if y0 is None:
		y0 = resolution_crop[1] // 2
	dots_centered = dots - [x0, y0, z0]
	dots_rounded = dots_centered.astype(int)
	return dots_rounded


def files_list(mypath, end='.mat'):
	"""
	Generates a list of files with a specified extension in a directory.

	Parameters:
		mypath (str): Path to the directory containing files.
		end (str): File extension to filter by. Defaults to '.mat'.

	Returns:
		list: List of filenames with the specified extension.
	"""
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(end)]
	return onlyfiles


if __name__ == '__main__':
	"""
	Main script for processing fields and building singularity dots.

	This script processes `.mat` files to extract field data, normalizes them, performs interpolation, and propagates
	the fields to generate 3D representations. It also detects and saves singularity dots found in the 3D fields.
	"""
	# Define directories for input and output data
	directory_field = f'data/test/'
	directory_field_saved = f'data/test/saved/'
	directory_field_saved_dots = f'data/test/saved/dots/'
	directory_field_saved_plots = f'data/test/saved/plots/'
	
	# Create output directories if they do not exist
	if not os.path.isdir(directory_field_saved):
		os.makedirs(directory_field_saved)
	if not os.path.isdir(directory_field_saved_dots):
		os.makedirs(directory_field_saved_dots)
	if not os.path.isdir(directory_field_saved_plots):
		os.makedirs(directory_field_saved_plots)
	
	# Process `.mat` files to extract and save 2D field data
	file_processing_ = True
	if file_processing_:
		files = files_list(directory_field, end='.mat')
		print(files)
		for file in files:
			print(file)
			field2D, _ = main_field_processing(
				path=directory_field + file,
				plotting=False,
				resolution_iterpol_center=(100, 100),
				xMinMax_frac_center=(1, 1),
				yMinMax_frac_center=(1, 1),
				resolution_interpol_working=(115, 115),
				xMinMax_frac_working=(1.2, 1.2),
				yMinMax_frac_working=(1.2, 1.2),
				resolution_crop=(100, 100),
				moments_init={'p': (0, 5), 'l': (-4, 4)},
				moments_center={'p0': (0, 5), 'l0': (-4, 4)},
			)
			file_save = directory_field_saved + file[:-4] + '.npy'
			print(file_save)
			np.save(file_save, field2D)
			plot_field(field2D, save=directory_field_saved_plots + file[:-4] + '.png')
	
	# Build singularity dots from saved 2D fields
	dots_building_ = True
	resolution_crop = (70, 70)
	if dots_building_:
		files = files_list(directory_field_saved, end='.npy')
		print(files)
		for file in files:
			print(file)
			field2D = np.load(directory_field_saved + file)
			dots_raw, dots_filtered = main_dots_building(
				field2D=field2D,
				plotting=False,
				dz=5.5,
				steps_both=25,
				resolution_crop=resolution_crop,
				r_crop=25
			)
			file_save_dots_raw = directory_field_saved_dots + 'raw_' + file[:-4] + '.npy'
			file_save_dots_filtered = directory_field_saved_dots + 'filtered_' + file[:-4] + '.npy'
			dp.plotDots(dots_raw, dots_raw, color='green', show=False, size=15,
			            save=directory_field_saved_plots + file[:-4] + '_3D.html')
			np.save(file_save_dots_raw, dots_raw)
			np.save(file_save_dots_filtered, dots_filtered)
