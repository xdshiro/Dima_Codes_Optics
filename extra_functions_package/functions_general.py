"""

This module is part of a custom package developed for computational physics and optics simulations.
It provides a suite of utility functions designed to facilitate tasks such as mesh generation, field interpolation,
numerical integration, and optical field propagation. This package is intended for use within our research group
and can be extended or modified as needed for specific projects.

Key Features:
-------------
1. **General Mathematical Utilities**
   - `rho(*r)`: Computes the magnitude of a vector.
   - `phi(x, y)`: Computes the angle in a 2D plane.
   - `distance_between_points(point1, point2)`: Calculates the distance between two points in any dimension.

2. **Mesh Generation**
   - `create_mesh_XYZ(...)`: Generates a 3D meshgrid with optional randomness for simulating complex environments.
   - `create_mesh_XY(...)`: Generates a 2D meshgrid using a tuple for boundaries.
   - `create_mesh_XY_old(...)`: Legacy function for generating 2D meshgrids.

3. **Field Interpolation**
   - `interpolation_real(...)`: Converts a real 2D array into a continuous interpolating function.
   - `interpolation_complex(...)`: Converts a complex 2D array into a continuous interpolating function.

4. **Numerical Integration**
   - `integral_of_function_1D(...)`: Performs 1D integration for real or complex functions, useful for analyzing wavefunctions.

5. **Data Handling**
   - `reading_file_mat(...)`: Loads `.mat` files and extracts specific fields for analysis.

6. **Dot and Grid Manipulations**
   - `dots_move_center(dots)`: Centralizes dot positions for symmetry adjustments.
   - `dots3D_rescale(dots, mesh)`: Maps 3D dot indices to physical coordinates based on a mesh.

7. **Optical Field Propagation**
   - `propagator_split_step_3D(...)`: Simulates 3D field propagation using the split-step Fourier method.
   - `propagator_split_step_3D_linear(...)`: Linear propagation variant for optical fields.
   - `one_plane_propagator(...)`: Handles forward and backward propagation from a single plane.

8. **Filtering and Noise Addition**
   - `cut_filter(...)`: Applies spatial filters (circular/rectangular) to fields.
   - `random_list(...)`: Adds random perturbations to values, useful for Monte Carlo-like simulations.

9. **Utility Functions**
   - `arrays_from_mesh(mesh, indexing='ij')`: Extracts coordinate arrays from meshgrids for further processing.

Usage Notes:
------------
- Ensure all dependencies (`numpy`, `scipy`) are installed before using the module.
- Designed for advanced users familiar with computational physics and numerical methods.
- Extendable for new projects within the research group.
"""
import numpy as np

from scipy.interpolate import CloughTocher2DInterpolator
from scipy import integrate
import scipy.io as sio


def rho(*r):
	"""
	Calculates the magnitude of a vector.
	Parameters:
		*r: Components of the vector [x1, x2, x3, ...].

	Returns:
		Magnitude of the vector |r|.
	"""
	ans = 0
	for x in r:
		ans += x ** 2
	return np.sqrt(ans)


def phi(x, y):
	"""
	Computes the angle phi in the plane.

	Parameters:
		x, y: Coordinates in the plane.

	Returns:
		Angle phi.
	"""
	return np.angle(x + 1j * y)


def dots_move_center(dots):
	"""
	Moves a set of dots to the center of the object.

	Parameters:
		dots: Array of dot coordinates.

	Returns:
		Dots translated to the center.
	"""
	center = np.sum(dots, axis=0) / len(dots)
	return dots - center


def distance_between_points(point1, point2):
	"""
	Calculates the distance between two points in any dimension.

	Parameters:
		point1: Coordinates of the first point.
		point2: Coordinates of the second point.

	Returns:
		Geometrical distance between the points.
	"""
	deltas = np.array(point1) - np.array(point2)
	return rho(*deltas)


def create_mesh_XYZ(xMax, yMax, zMax, xRes=40, yRes=40, zRes=40,
                    xMin=None, yMin=None, zMin=None, indexing='ij', random=(None, None, None), **kwargs):
	"""
	Creates a 3D mesh using np.meshgrid.

	Parameters:
		xMax, yMax, zMax: Maximum values for the meshgrid along x, y, and z.
		xRes, yRes, zRes: Resolution along x, y, and z.
		xMin, yMin, zMin: Minimum values for the meshgrid along x, y, and z.
		indexing: 'ij' is the classic matrix (0,0) left top.
		random: Tuple to apply randomness to the mesh points.

	Returns:
		3D meshgrid.
	"""
	if xMin is None:
		xMin = -xMax
	if yMin is None:
		yMin = -yMax
	if zMin is None:
		zMin = -zMax
	if random[0] is None:
		xArray = np.linspace(xMin, xMax, xRes)
	else:
		xArray = np.sort((xMax + xMin) / 2 + (np.random.rand(xRes) - 0.5) * (xMax - xMin))
	# xArray = ((xMax + xMin) / 2 + (np.random.rand(xRes) - 0.5) * (xMax - xMin))
	
	if random[1] is None:
		yArray = np.linspace(yMin, yMax, yRes)
	else:
		yArray = np.sort((yMax + yMin) / 2 + (np.random.rand(yRes) - 0.5) * (yMax - yMin))
	# yArray = ((yMax + yMin) / 2 + (np.random.rand(yRes) - 0.5) * (yMax - yMin))
	if random[2] is None:
		zArray = np.linspace(zMin, zMax, zRes)
	else:
		zArray = np.sort((zMax + zMin) / 2 + (np.random.rand(zRes) - 0.5) * (zMax - zMin))
	# zArray = ((zMax + zMin) / 2 + (np.random.rand(zRes) - 0.5) * (zMax - zMin))
	return np.meshgrid(xArray, yArray, zArray, indexing=indexing, **kwargs)


def create_mesh_XY_old(xMax, yMax, xRes=50, yRes=50,
                       xMin=None, yMin=None, indexing='ij', **kwargs):
	"""
	Creates a 2D mesh using np.meshgrid with old parameters.

	Parameters:
		xMax, yMax: Maximum values for the meshgrid along x and y.
		xRes, yRes: Resolution along x and y.
		xMin, yMin: Minimum values for the meshgrid along x and y.
		indexing: 'ij' is the classic matrix (0,0) left top.

	Returns:
		2D meshgrid.
	"""
	if xMin is None:
		xMin = -xMax
	if yMin is None:
		yMin = -yMax
	xArray = np.linspace(xMin, xMax, xRes)
	yArray = np.linspace(yMin, yMax, yRes)
	return np.meshgrid(xArray, yArray, indexing=indexing, **kwargs)


def create_mesh_XY(xMinMax=None, yMinMax=None, xRes=50, yRes=50,
                   indexing='ij', **kwargs):
	"""
	Creates a 2D mesh using np.meshgrid with new parameters.

	Parameters:
		xMinMax: Tuple [xMin, xMax] for the boundaries along x.
		yMinMax: Tuple [yMin, yMax] for the boundaries along y.
		xRes, yRes: Resolution along x and y.
		indexing: 'ij' is the classic matrix (0,0) left top.

	Returns:
		2D meshgrid.
	"""
	if xMinMax is None:
		xMinMax = (0 - xRes // 2, 0 + xRes // 2)
	if yMinMax is None:
		yMinMax = (0 - yRes // 2, 0 + yRes // 2)
	xArray = np.linspace(*xMinMax, xRes)
	yArray = np.linspace(*yMinMax, yRes)
	return np.meshgrid(xArray, yArray, indexing=indexing, **kwargs)


def interpolation_real(field, xArray=None, yArray=None, **kwargs):
	"""
	Interpolates any real 2D matrix into a function.

	Parameters:
		field: Initial real 2D array.
		xArray: X interval (range).
		yArray: Y interval (range).
		kwargs: Extra parameters for CloughTocher2DInterpolator.

	Returns:
		CloughTocher2DInterpolator object.
	"""
	xResolution, yResolution = np.shape(field)
	if xArray is None:
		xArray = list(range(xResolution))
	if yArray is None:
		yArray = list(range(yResolution))
	xArrayFull = np.zeros(xResolution * yResolution)
	yArrayFull = np.zeros(xResolution * yResolution)
	fArray1D = np.zeros(xResolution * yResolution)
	for i in range(xResolution * yResolution):
		xArrayFull[i] = xArray[i // yResolution]
		yArrayFull[i] = yArray[i % yResolution]
		fArray1D[i] = field[i // yResolution, i % yResolution]
	return CloughTocher2DInterpolator(list(zip(xArrayFull, yArrayFull)), fArray1D, **kwargs)


# function interpolate complex 2D array of any data_weak into the function(x, y)
def interpolation_complex(field, xArray=None, yArray=None, mesh=None, fill_value=False):
	"""
	Interpolates a complex 2D array of any data into a function.

	Parameters:
		field: Initial complex 2D array.
		xArray: X interval (range).
		yArray: Y interval (range).
		mesh: Meshgrid.
		fill_value: Fill value for interpolation.

	Returns:
		Function for interpolated field.
	"""
	if mesh is not None:
		xArray, yArray = arrays_from_mesh(mesh)
	fieldReal = np.real(field)
	fieldImag = np.imag(field)
	real = interpolation_real(fieldReal, xArray, yArray, fill_value=fill_value)
	imag = interpolation_real(fieldImag, xArray, yArray, fill_value=fill_value)
	
	def f(x, y):
		return real(x, y) + 1j * imag(x, y)
	
	return f


# return interpolation_real(fieldReal, xArray, yArray), interpolation_real(fieldImag, xArray, yArray)


def integral_of_function_1D(integrandFunc, x1, x2, epsabs=1.e-5, maxp1=50, limit=50, **kwargs):
	"""
	Integrates a function over a 1D range, handling complex values.

	Parameters:
		integrandFunc: Function to integrate.
		x1: Lower limit of integration.
		x2: Upper limit of integration.
		epsabs: Absolute error tolerance.
		maxp1: Maximum number of subintervals.
		limit: Maximum number of evaluations.
		kwargs: Extra parameters for scipy.integrate.quad.

	Returns:
		Integral value and error estimates.
	"""
	
	def real_f(x):
		return np.real(integrandFunc(x))
	
	def imag_f(x):
		return np.imag(integrandFunc(x))
	
	real_integral = integrate.quad(real_f, x1, x2, epsabs=epsabs, maxp1=maxp1, limit=limit, **kwargs)
	imag_integral = integrate.quad(imag_f, x1, x2, epsabs=epsabs, maxp1=maxp1, limit=limit, **kwargs)
	return real_integral[0] + 1j * imag_integral[0], (real_integral[1:], imag_integral[1:])


def arrays_from_mesh(mesh, indexing='ij'):
	"""
	Returns the tuple of x1Array, x2Array, etc., from a mesh.

	Parameters:
		mesh: Meshgrid.
		indexing: Indexing style ('ij' or 'xy').

	Returns:
		Tuple of arrays representing the mesh.
	"""
	xList = []
	if indexing == 'ij':
		for i, m in enumerate(mesh):
			row = [0] * len(np.shape(m))
			row[i] = slice(None, None)
			xList.append(m[tuple(row)])
	else:
		if len(np.shape(mesh[0])) == 2:
			for i, m in enumerate(mesh):
				row = [0] * len(np.shape(m))
				row[len(np.shape(m)) - 1 - i] = slice(None, None)
				xList.append(m[tuple(row)])
		elif len(np.shape(mesh[0])) == 3:
			indexing = [1, 0, 2]
			for i, m in enumerate(mesh):
				row = [0] * len(np.shape(m))
				row[indexing[i]] = slice(None, None)
				xList.append(m[tuple(row)])
		else:
			print("'xy' cannot be recreated for 4+ dimensions")
	
	xTuple = tuple(xList)
	return xTuple


def reading_file_mat(fileName, fieldToRead="p_charges", printV=False):
	"""
	Reads a .mat file and converts one of its fields into a numpy array.

	Parameters:
		fileName: Name of the .mat file.
		fieldToRead: Field to convert (requires the name).
		printV: Flag to print the contents of the .mat file if the field name is unknown.

	Returns:
		Numpy array of the specified field.
	"""
	matFile = sio.loadmat(fileName, appendmat=False)
	if fieldToRead not in matFile:
		printV = True
	if printV:
		print(matFile)
		exit()
	return np.array(matFile[fieldToRead])


def dots3D_rescale(dots, mesh):
	"""
	Rescales dots from indices to physical coordinates based on a mesh.

	Parameters:
		dots: Array of dot indices.
		mesh: Meshgrid.

	Returns:
		Array of rescaled dot coordinates.
	"""
	xyz = arrays_from_mesh(mesh)
	dotsScaled = [[xyz[0][x], xyz[1][y], xyz[2][z]] for x, y, z in dots]
	return np.array(dotsScaled)


def random_list(values, diapason, diapason_complex=None):
	"""
	Modifies values by adding a random component within a specified range.

	Parameters:
		values: Original values.
		diapason: Range for random modification.
		diapason_complex: Range for random modification of complex values.

	Returns:
		Modified values with random components.
	"""
	import random
	if diapason_complex is not None:
		answer = [x + random.uniform(-d, +d) + 1j * random.uniform(-dC, +dC) for x, d, dC
		          in zip(values, diapason, diapason_complex)]
	else:
		answer = [x + random.uniform(-d, +d) for x, d in zip(values, diapason)]
	return answer


##############################################


def propagator_split_step_3D(E, dz=1, xArray=None, yArray=None, zSteps=1, n0=1, k0=1):
	"""
	Propagates a field in 3D using the split-step Fourier method.

	Parameters:
		E: Initial field.
		dz: Step size in z.
		xArray: Array of x coordinates.
		yArray: Array of y coordinates.
		zSteps: Number of steps in z.
		n0: Refractive index.
		k0: Wave number.

	Returns:
		3D field after propagation.
	"""
	if xArray is None:
		xArray = np.array(range(np.shape(E)[0]))
	if yArray is None:
		yArray = np.array(range(np.shape(E)[1]))
	xResolution, yResolution = len(xArray), len(yArray)
	zResolution = zSteps + 1
	intervalX = xArray[-1] - xArray[0]
	intervalY = yArray[-1] - yArray[0]
	
	# xyMesh = np.array(np.meshgrid(xArray, yArray, indexing='ij'))
	if xResolution // 2 == 1:
		kxArray = np.linspace(-1. * np.pi * (xResolution - 2) / intervalX,
		                      1. * np.pi * (xResolution - 2) / intervalX, xResolution)
		kyArray = np.linspace(-1. * np.pi * (yResolution - 2) / intervalY,
		                      1. * np.pi * (yResolution - 2) / intervalY, yResolution)
	else:
		kxArray = np.linspace(-1. * np.pi * (xResolution - 0) / intervalX,
		                      1. * np.pi * (xResolution - 2) / intervalX, xResolution)
		kyArray = np.linspace(-1. * np.pi * (yResolution - 0) / intervalY,
		                      1. * np.pi * (yResolution - 2) / intervalY, yResolution)
	
	KxyMesh = np.array(np.meshgrid(kxArray, kyArray, indexing='ij'))
	
	def nonlinearity_spec(E):
		return dz * 0
	
	# works fine!
	def linear_step(field):
		temporaryField = np.fft.fftshift(np.fft.fftn(field))
		temporaryField = (temporaryField *
		                  np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[0] ** 2) *
		                  np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[1] ** 2))  # something here in /2
		return np.fft.ifftn(np.fft.ifftshift(temporaryField))
	
	fieldReturn = np.zeros((xResolution, yResolution, zResolution), dtype=complex)
	fieldReturn[:, :, 0] = E
	for k in range(1, zResolution):
		fieldReturn[:, :, k] = linear_step(fieldReturn[:, :, k - 1])
		fieldReturn[:, :, k] = fieldReturn[:, :, k] * np.exp(nonlinearity_spec(fieldReturn[:, :, k]))
	
	return fieldReturn


def propagator_split_step_3D_linear(E, dz=1, xArray=None, yArray=None, zSteps=1, n0=1, k0=1):
	"""
	Linearly propagates a field in 3D using the split-step Fourier method.

	Parameters:
		E: Initial field.
		dz: Step size in z.
		xArray: Array of x coordinates.
		yArray: Array of y coordinates.
		zSteps: Number of steps in z.
		n0: Refractive index.
		k0: Wave number.

	Returns:
		3D field after linear propagation.
	"""
	if xArray is None:
		xArray = np.array(range(np.shape(E)[0]))
	if yArray is None:
		yArray = np.array(range(np.shape(E)[1]))
	xResolution, yResolution = len(xArray), len(yArray)
	zResolution = zSteps + 1
	intervalX = xArray[-1] - xArray[0]
	intervalY = yArray[-1] - yArray[0]
	if xResolution // 2 == 1:
		kxArray = np.linspace(-1. * np.pi * (xResolution - 2) / intervalX,
		                      1. * np.pi * (xResolution - 2) / intervalX, xResolution)
		kyArray = np.linspace(-1. * np.pi * (yResolution - 2) / intervalY,
		                      1. * np.pi * (yResolution - 2) / intervalY, yResolution)
	else:
		kxArray = np.linspace(-1. * np.pi * (xResolution - 0) / intervalX,
		                      1. * np.pi * (xResolution - 2) / intervalX, xResolution)
		kyArray = np.linspace(-1. * np.pi * (yResolution - 0) / intervalY,
		                      1. * np.pi * (yResolution - 2) / intervalY, yResolution)
	
	KxyMesh = np.array(np.meshgrid(kxArray, kyArray, indexing='ij'))
	fieldReturn = np.zeros((xResolution, yResolution, zResolution), dtype=complex)
	fieldReturn[:, :, 0] = np.fft.fftshift(np.fft.fftn(E))
	for k in range(1, zResolution):
		fieldReturn[:, :, k] = (fieldReturn[:, :, k - 1] *
		                        np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[0] ** 2) *
		                        np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[1] ** 2))
		fieldReturn[:, :, k - 1] = np.fft.ifftn(np.fft.ifftshift(fieldReturn[:, :, k - 1]))
	fieldReturn[:, :, -1] = np.fft.ifftn(np.fft.ifftshift(fieldReturn[:, :, -1]))
	return fieldReturn


def one_plane_propagator(fieldPlane, dz, stepsNumber, n0=1, k0=1):
	"""
	Propagates a field from a single plane both forward and backward in z.

	Parameters:
		fieldPlane: Initial field plane.
		dz: Step size in z.
		stepsNumber: Number of steps in z.
		n0: Refractive index.
		k0: Wave number.

	Returns:
		3D field after propagation.
	"""
	# if shapeWrong is not False:
	#     if shapeWrong is True:
	#         print(f'using the middle plane in one_plane_propagator (shapeWrong = True)')
	#         fieldPlane = fieldPlane[:, :, np.shape(fieldPlane)[2] // 2]
	#     else:
	#         fieldPlane = fieldPlane[:, :, np.shape(fieldPlane)[2] // 2 + shapeWrong]
	fieldPropMinus = propagator_split_step_3D(fieldPlane, dz=-dz, zSteps=stepsNumber, n0=n0, k0=k0)
	
	fieldPropPLus = propagator_split_step_3D(fieldPlane, dz=dz, zSteps=stepsNumber, n0=n0, k0=k0)
	fieldPropTotal = np.concatenate((np.flip(fieldPropMinus, axis=2), fieldPropPLus[:, :, 1:-1]), axis=2)
	return fieldPropTotal


def cut_filter(E, radiusPix=1, circle=True, phaseOnly=False):
	"""
	Applies a circular or rectangular filter to a field.

	Parameters:
		E: Field to filter.
		radiusPix: Radius of the filter in pixels.
		circle: Flag for circular filter.
		phaseOnly: Flag for phase-only filter.

	Returns:
		Filtered field.
	"""
	ans = np.copy(E)
	xCenter, yCenter = np.shape(ans)[0] // 2, np.shape(ans)[0] // 2
	if circle:
		for i in range(np.shape(ans)[0]):
			for j in range(np.shape(ans)[1]):
				if np.sqrt((xCenter - i) ** 2 + (yCenter - j) ** 2) > radiusPix:
					ans[i, j] = 0
	else:
		if phaseOnly:
			zeros = np.abs(np.copy(ans))
			zeros = zeros.astype(complex, copy=False)
		else:
			zeros = np.zeros(np.shape(ans), dtype=complex)
		zeros[xCenter - radiusPix:xCenter + radiusPix + 1, yCenter - radiusPix:yCenter + radiusPix + 1] \
			= ans[xCenter - radiusPix:xCenter + radiusPix + 1, yCenter - radiusPix:yCenter + radiusPix + 1]
		ans = zeros
	return ans

