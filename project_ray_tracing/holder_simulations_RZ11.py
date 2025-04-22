"""
This module simulates the integration sphere using pvtrace package.
Inside the sphere we can place different objects and structures.
Absorption is included.
"""
import logging

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)


import pickle
from scipy.ndimage import gaussian_filter
import pvtrace as pv
import numpy as np
import functools
import collections

import matplotlib.pyplot as plt
import helper_cross_sections as cs  # mine

from numpy.random import Generator, PCG64, SeedSequence

from pvtrace.geometry.cylinder import CylinderRough

rng = Generator(PCG64(SeedSequence(103)))

d_bottom_hole = 2
h_bottom_hole = 0.5
d_holder = 5.2
h_holder = 4.3 - h_bottom_hole

# here we need to have a switch between 3 different cases
# RZ11 Z7 and RZ13
# RZ11
total = 3.96
T = 0.27 / total
T_dir = 0.18 / total
R = 0.5 / total
A = 1 - T - R

# Z7
total = 8.41
T = 5.54 / total
T_dir = (3.62 - 0.03) / total
R = (1.61 - 0.03) / total
A = 1 - T - R

# RZ13
total = 5.42
T = 0.02 / total
T_dir = 0.01 / total
R = 0.2 / total
A = 1 - T - R

assert R + T + A == 1, 'WRONG T, R, A'
TF = 1 - A
length = h_holder  # in m
L_A = -length / np.log(TF)  # absorption length
L = -length / np.log(T)  # attenuation length
L_S = L * L_A / (L_A - L)  # scattering length



random_counter = 0
random_seed = 0
r = 10
r = 7.5
r_foc = d_holder / 6
dist = h_holder * 3


r = 30 / 23 / 2
r_foc = 1e-3
dist = h_holder + 4.6+23 / 6 + 1


focuses = [-h_holder - 4.6 - 23 / 6]
# focuses = [0]
# exit()
class PartialTopSurfaceMirror(pv.FresnelSurfaceDelegate):
	""" A section of the top surface is covered with a perfect mirrrors.

		All other surface obey the Fresnel equations.
	"""


	def reflected_direction_random_not_finished(self, surface, ray, geometry, container, adjacent):
		"""
		Implementation of the scattering surface. We can control the scattering angles in scattered_angles.
		Vector transformation works only for a box rn, need to implement the general case.
		:return: direction of the scattered beam
		"""

		def scattered_angles(absorption=0.9):
			"""
			Angles onf the scattered from the surface beams. (horizontal plane. Other planes
			are implemented via linear algebra transformations)
			:return: phi, theta of the scattered beam.
			"""
			roll_the_dice = np.random.uniform(0, 1, 1)
			if roll_the_dice > absorption:
				return 0, np.pi
			else:
				p1, p2 = rng.uniform(0, 1, 2)
				phi = 2 * np.pi * p1
				theta = 0.96 * 0.5 * np.pi * p2
				
				return phi, theta
		
		def perpendicular_vector_3D(X):
			"""
			add the real case, rn only for BOXes
			:param X: vector 3D.
			:return: Any perpendicular vector to X
			"""
			x, y, z = X
			if x == 0 and y == 0:
				return 1, 0, 0
			elif x == 0 and z == 0:
				return 0, 0, 1
			elif y == 0 and z == 0:
				return 0, 1, 0

		u = np.array((0, 0, -1))
		v = perpendicular_vector_3D(u)
		uv = np.cross(u, v)

		x = geometry.normal(ray.position)

		y = perpendicular_vector_3D(x)

		xy = np.cross(x, y)

		R1 = np.array([u, v, uv])
		R2 = np.array([x, y, xy])
		transform3 = np.matmul(R1.T, R2)

		phi_new, theta_new = scattered_angles(absorption=1)

		x = np.sin(theta_new) * np.cos(phi_new)
		y = np.sin(theta_new) * np.sin(phi_new)
		z = np.cos(theta_new)

		
		Xnew0 = np.array((x, y, z))
		
		Xnew = tuple(np.matmul(Xnew0, transform3))
		return Xnew

	
	def reflected_direction(self, surface, ray, geometry, container, adjacent):
		"""
		Implementation of the scattering surface. We can control the scattering angles in scattered_angles.
		Vector transformation works only for a box rn, need to implement the general case.
		:return: direction of the scattered beam
		"""
		
		normal = np.array(geometry.normal(ray.position))
		
		incident = np.array(ray.direction)
		reflected = incident - 2 * np.dot(incident, normal) * normal
		
		return tuple(reflected)

	def reflectivity(self, surface, ray, geometry, container, adjacent):
		""" Return the reflectivity of the part of the surface hit by the ray.

			Parameters
			----------
			surface: Surface
				The surface object belonging to the material.
			ray: Ray
				The ray hitting the surface in the local coordinate system of the `geometry` object.
			geometry: Geometry
				The object being hit (e.g. Sphere, Box, Cylinder, Mesh etc.)
			container: Node
				The node containing the ray.
			adjacent: Node
				The node that will contain the ray if the ray is transmitted.
		"""
		
		x, y, z = ray.position[0], ray.position[1], ray.position[2]

		if geometry.radius == d_holder / 2 and geometry.length == h_holder:  # real holder

			if np.isclose(z, h_holder / 2) and np.sqrt(x ** 2 + y ** 2) < d_holder / 2:

				
				return super(PartialTopSurfaceMirror, self).reflectivity(surface, ray, geometry, container,
				                                                         adjacent)
			elif np.isclose(z, -h_holder / 2) and np.sqrt(x ** 2 + y ** 2) < d_bottom_hole / 2:
				return super(PartialTopSurfaceMirror, self).reflectivity(surface, ray, geometry, container,
				                                                         adjacent)
			return 1
		if geometry.radius == d_bottom_hole / 2 and geometry.length == h_bottom_hole:
			if (np.isclose(z, h_bottom_hole / 2)
					and np.abs(x) < d_bottom_hole / 2 and np.abs(y) < d_bottom_hole / 2):
				return super(PartialTopSurfaceMirror, self).reflectivity(surface, ray, geometry, container,
				                                                         adjacent)
			elif (np.isclose(z, -h_bottom_hole / 2)
			      and np.abs(x) < d_bottom_hole / 2 and np.abs(y) < d_bottom_hole / 2):
				return super(PartialTopSurfaceMirror, self).reflectivity(surface, ray, geometry, container,
				                                                         adjacent)
			return 1



def cylindrical_to_cart(r, phi, z=0):
	"""
	transformation of the spherical coordinates to cartesiane coordinates
	:return: x y z
	"""
	x = r * np.cos(phi)
	y = r * np.sin(phi)
	z = z
	cart = np.column_stack((x, y, z))
	if cart.size == 3:
		return cart[0, :]
	return cart


def lines_dots(positions, x_res, y_res, z_res,
               x_max_min=(-1, 1), y_max_min=(-1, 1), z_max_min=(-1, 1),
               length_line=1, res_line=4):
	"""
	Line : (x-x1)/(x2-x1) = (y-y1)/(y2-y1) = (z-z1)/(z2-z1)
	Parametric :
		x = x1 + (x2 - x1) * a
		y = y1 + (y2 - y1) * a
		z = z1 + (z2 - z1) * a
	:param positions:
	:return:
	"""
	
	def dots_from_line(dot1, dot2):
		x1, y1, z1 = dot1
		x2, y2, z2 = dot2
		line = dot2 - dot1
		len_line = np.sqrt((line * line).sum())
		res = int(res_line * len_line / length_line)
		a = np.linspace(0, 1, res)
		x = x1 + (x2 - x1) * a
		y = y1 + (y2 - y1) * a
		z = z1 + (z2 - z1) * a
		dots = np.array((x, y, z)).T
		ind_delete = []
		for i, dot in enumerate(dots):
			if (dot[0] < x_max_min[0] or dot[0] > x_max_min[1]
					or dot[1] < y_max_min[0] or dot[1] > y_max_min[1]
					or dot[2] < z_max_min[0] or dot[2] > z_max_min[1]):
				ind_delete.append(i)
		dots_cut = np.delete(dots, ind_delete, axis=0)
		return dots_cut
	
	delta_x = (x_max_min[1] - x_max_min[0]) / (x_res - 1)
	delta_y = (y_max_min[1] - y_max_min[0]) / (y_res - 1)
	delta_z = (z_max_min[1] - z_max_min[0]) / (z_res - 1)
	scale_coeff_x = 1 / delta_x
	scale_coeff_y = 1 / delta_y
	scale_coeff_z = 1 / delta_z
	dots_all_rays = []
	for dots in positions:
		dot1 = dots[0]
		for dot2 in dots[1:]:
			dots_scaled = dots_from_line(dot1, dot2)[1:] * [scale_coeff_x, scale_coeff_y, scale_coeff_z]
			dots_round = np.rint(dots_scaled).astype(int)
			dots_unique = np.unique(dots_round, axis=0)
			dots_all_rays.append(dots_unique)
			
			dot1 = dot2
	dots = np.concatenate(np.array(dots_all_rays, dtype=object), axis=0)
	return dots


def array_3D_intensity_from_dots(dots, x_res, y_res, z_res, x_max_min=(-1, 1), y_max_min=(-1, 1), z_max_min=(-1, 1)):
	print(len(dots))
	# x_max = x_max_min[1] - x_max_min[0]
	x_cen = int((- x_max_min[0]) / (x_max_min[1] - x_max_min[0]) * x_res)
	y_cen = int((- y_max_min[0]) / (y_max_min[1] - y_max_min[0]) * y_res)
	z_cen = int((- z_max_min[0]) / (z_max_min[1] - z_max_min[0]) * z_res)
	dots_centered = dots + (x_cen, y_cen, z_cen)
	dots_new = np.zeros((x_res, y_res, z_res))
	for dot in dots_centered:
		dots_new[dot[0], dot[1], dot[2]] += 1
	return dots_new


def array_3D_intensity_from_dots_avg(dots_3D):
	x_res, y_res, z_res = np.shape(dots_3D)
	dots_ans = np.zeros((x_res, y_res, z_res))
	dots_pad = np.pad(dots_3D, 1, 'linear_ramp')  # end_values=(0, 0, 0))
	for i in range(1, x_res + 1):
		for j in range(1, y_res + 1):
			for k in range(1, z_res + 1):
				dots_ans[i - 1, j - 1, k - 1] = np.sum(dots_pad[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]) / 27
	return dots_ans


def main_create_scene_test():
	world = pv.Node(
		name="world (air)",
		geometry=pv.Sphere(
			radius=6,
			material=pv.Material(refractive_index=1.0,
 )
		)
	)
	
	import functools
	light = pv.Node(
		name="Light (555nm)",
		light=pv.Light(position=functools.partial(collimated_beam, 0.1),
		               wavelength=lambda: 455
		               ),
		parent=world
	)
	light.translate((0.0, 0.0, 3))
	light.rotate(np.pi, [1, 0, 0])
	scene = pv.Scene(world)
	return scene


def structure_sample(parent, absor=1, scat=1):
	cylinder1 = pv.Node(
		name="holder",
		geometry=CylinderRough(
			radius=d_holder / 2,
			length=h_holder,
			material=pv.Material(
				refractive_index=1.1,
				surface=pv.Surface(delegate=PartialTopSurfaceMirror()),
				components=[
					pv.Absorber(coefficient=absor),
					pv.Scatterer(coefficient=scat)
				]
			
			),
		),
		parent=parent
	)
	cylinder1.translate([0, 0, h_holder / 2])
	cylinder3 = pv.Node(
		name="bottom_hole",
		geometry=pv.Cylinder(
			radius=d_bottom_hole / 2,
			length=h_bottom_hole,
			material=pv.Material(
				refractive_index=1,
				surface=pv.Surface(delegate=PartialTopSurfaceMirror()),
			),
		),
		parent=parent
	)
	cylinder3.translate([0, 0, -h_bottom_hole / 2])


def collimated_beam_old(r):
	""" Samples directions within a cone of solid angle defined by `theta_max`.

		Notes
		-----
		Derived as follows using sympy::

			from sympy import *
			theta, theta_max, p = symbols('theta theta_max p')
			f = cos(theta) * sin(theta)
			cdf = integrate(f, (theta, 0, theta))
			pdf = cdf / cdf.subs({theta: theta_max})
			inv_pdf = solve(Eq(pdf, p), theta)[-1]
	"""
	if np.isclose(r, 0.0):
		raise ValueError("Expected 0 < r")
	r_random = np.random.normal(0, 0.5, 1)
	p2 = np.random.uniform(0, 1, 1)
	phi = np.pi * p2
	coords = cylindrical_to_cart(r_random, phi, 0)
	return coords


def collimated_beam(r):
	""" Samples directions within a cone of solid angle defined by `theta_max`.

		Notes
		-----
		Derived as follows using sympy::

			from sympy import *
			theta, theta_max, p = symbols('theta theta_max p')
			f = cos(theta) * sin(theta)
			cdf = integrate(f, (theta, 0, theta))
			pdf = cdf / cdf.subs({theta: theta_max})
			inv_pdf = solve(Eq(pdf, p), theta)[-1]
	"""
	if np.isclose(r, 0.0):
		raise ValueError("Expected 0 < r")
	x, y = np.random.multivariate_normal((0, 0), [[r ** 2, 0], [0, r ** 2]], 1)[0]
	coords = (x, y, 0)
	return coords


# r_foc = 2.5

def positions_directions(focus):
	global random_counter, random_seed, r, r_foc
	if random_counter % 2 == 0:
		random_seed += 1
	rng = np.random.default_rng(random_seed)
	r = r
	r_foc = r_foc
	z0 = focus
	x, y = rng.multivariate_normal((0, 0), [[r ** 2, 0], [0, r ** 2]], 1)[0]
	coordinates = x, y, 0
	x0, y0 = rng.multivariate_normal((0, 0), [[r_foc ** 2, 0], [0, r_foc ** 2]], 1)[0]
	z0 += 0.0 * np.random.uniform(-1, 1)
	v = np.array([x0 - x, y0 - y, z0])
	v_norm = v / np.linalg.norm(v)
	random_counter += 1
	return coordinates, v_norm


def position(focus):
	coordinates, _ = positions_directions(focus)
	return coordinates


def direction(focus):
	_, angles = positions_directions(focus)
	return angles


def light_beam(parent, focus):
	global dist
	light = pv.Node(
		name="Light (555nm)",
		
		light=pv.Light(
			direction=functools.partial(direction, focus),
			position=functools.partial(position, focus),
			wavelength=lambda: 555
		),
		parent=parent,
	)
	# light.translate([0, 0, 9.5])
	light.translate([0, 0, dist])
	light.rotate(np.pi, [1, 0, 0])


def pv_scene_real(structure=structure_sample, light=light_beam, absor=1e-10, scat=1e-10,
                  focus=1e9):
	"""

	:param structure:
	:param light:
	:return:
	"""
	world = pv.Node(
		name="world (air)",
		geometry=pv.Sphere(
			radius=17,
			material=pv.Material(refractive_index=1.0),
		)
	)
	
	structure(parent=world, absor=absor, scat=scat)
	light(parent=world, focus=focus)
	scene = pv.Scene(world)
	return scene


def plane_intensity(positions, plane_vec=(3, 2, 1), plane_dot=(1, 2, 3), x_res=21, y_res=21,
                    x_max_min=(-1, 1), y_max_min=(-1, 1)):
	"""

	:param positions:
	:param plane_vec: a, b, c in ax+by+cz+d=0
	:param plane_dot: x0,y0,z0 in d = -(ax0+by0+cz0)
	:param x_res:
	:param y_res:
	:param x_max_min:
	:param y_max_min:
	:return:
	"""
	
	def is_plane_between_points(dot1, dot2, plane):
		distance1 = np.dot(dot1, plane[:3]) + plane[3]
		distance2 = np.dot(dot2, plane[:3]) + plane[3]
		
		return (distance1 > 0 > distance2) or (distance1 < 0 < distance2)
	
	def intersection_point_with_plane(dot1, dot2, plane):
		# print(dot1, dot2, plane)
		line_direction = dot2 - dot1
		to_plane = np.dot(line_direction, plane[0:3])
		if to_plane:
			t = (-(plane[0] * dot1[0] + plane[1] * dot1[1] + plane[2] * dot1[2] + plane[3])
			     / to_plane)
			if not is_plane_between_points(dot1, dot2, plane):
				return None
			intersection_point = dot1 + t * line_direction
			return intersection_point
		else:
			return None
	
	plane = np.array(list(plane_vec) + [-np.dot(plane_vec, plane_dot)])
	dots_all_rays = []
	for dots in positions:
		# print(dots)
		dot1 = dots[0]
		for dot2 in dots[1:]:
			intersect = intersection_point_with_plane(dot1, dot2, plane)
			if intersect is not None:
				dots_all_rays.append(intersect)
			dot1 = dot2
	# return np.array(dots_all_rays)
	dots_all_rays = np.array(dots_all_rays)
	delta_x = (x_max_min[1] - x_max_min[0]) / (x_res - 1)
	delta_y = (y_max_min[1] - y_max_min[0]) / (y_res - 1)
	scale_coeff_x = 1 / delta_x
	scale_coeff_y = 1 / delta_y
	dots_scaled = dots_all_rays * [scale_coeff_x, scale_coeff_y, 1]
	dots_round = np.rint(dots_scaled).astype(int)
	dots_tuples = [tuple(point) for point in dots_round]
	dots_ind_collection = collections.Counter(dots_tuples)
	x_ind = np.arange(int(x_max_min[0] / delta_x), int(x_max_min[1] / delta_x))
	y_ind = np.arange(int(y_max_min[0] / delta_y), int(y_max_min[1] / delta_y))
	intensity = np.zeros((x_res, y_res))
	print(dots_ind_collection)
	z = np.rint(plane_dot[2]).astype(int)
	for ind_i, i in enumerate(x_ind):
		for ind_j, j in enumerate(y_ind):
			if (i, j, z) in dict(dots_ind_collection):
				intensity[ind_i, ind_j] = dots_ind_collection[(i, j, z)]
	print(intensity)
	return dots_all_rays, intensity

if __name__ == '__main__':
	number_rays = 100
	show_3d = True
	for focus in focuses:
		focus += dist
		# print('hi')
		scene = pv_scene_real(absor=1. / L_A, scat=1. / L_S, focus=focus)

		# number_rays =100
		positions = cs.scene_render_and_positions(scene, rays_number=number_rays, show_3d=show_3d, random_seed=2, )
		# continue
		x_res, y_res, z_res = 221, 221, 221
		xM = -d_holder / 2 - 0.1, d_holder / 2 + 0.1
		yM = -d_holder / 2 - 0.1, d_holder / 2 + 0.1
		zM = -h_bottom_hole * 1.000001, h_holder * 1.000001
		dots = lines_dots(positions, x_res=x_res, y_res=y_res, z_res=z_res,
			  x_max_min=xM, y_max_min=yM, z_max_min=zM,
			  res_line=int(np.sqrt(x_res ** 2 + y_res ** 2 + z_res ** 2)), length_line=1)
		# print(len(dots))
		dots_3D = array_3D_intensity_from_dots(dots, x_res, y_res, z_res, x_max_min=xM, y_max_min=yM, z_max_min=zM)
		
		# dots_3D_avg = array_3D_intensity_from_dots_avg(dots_3D)
		# exit()
		# plt.imshow(dots_3D[:, y_res // 2, :].T, cmap='nipy_spectral', interpolation='bilinear')
		# plt.show()
		# plt.imshow(dots_3D_avg[:, y_res // 2, :].T, cmap='nipy_spectral', interpolation='bilinear')
		# plt.show()
		# dots_3D_sat = np.array(dots_3D_avg)
		# max = np.max(dots_3D_sat) * 0.5
		# dots_3D_sat[dots_3D_sat > max] = max
		# plt.imshow(dots_3D_sat[:, y_res // 2, :].T, cmap='nipy_spectral', interpolation='bilinear')
		# plt.show()
		# dots_3D_sat = np.array(dots_3D_avg)
		# max = np.max(dots_3D_sat) * 0.02ё
		# dots_3D_sat[dots_3D_sat > max] = max
		
		np.save(f'Z7_dist_{round(dist, 2)}_foc_{round(focus, 2)}_rfoc_{round(r_foc, 2)}_{x_res}_{number_rays}_r{r}_15n14even', dots_3D)
		# np.save(f'Z7_positions_{number_rays}', np.array(positions))
		print(f'Z7_dist_{round(dist, 2)}_foc_{round(focus, 2)}_rfoc_{round(r_foc, 2)}_{x_res}_{number_rays}_r{r}_15n14even')
		with open(f'Z7_dist_{round(dist, 2)}_foc_{round(focus, 2)}_rfoc_{round(r_foc, 2)}_positions_{number_rays}_r{r}_15n14even.pkl', 'wb') as file:
			pickle.dump(positions, file)
		
		# plt.imshow(dots_3D[:, y_res // 2, :].T, cmap='hot', interpolation='spline36')
		# plt.tight_layout()
		# plt.show()
		# plt.imshow(dots_3D[:, :, -1].T, cmap='hot', interpolation='spline36')
		# plt.tight_layout()
		# plt.show()
		# plt.imshow(dots_3D[:, :, z_res // 2].T, cmap='hot', interpolation='spline36')
		# plt.tight_layout()
		# plt.show()
		# plt.imshow(dots_3D[:, :, 0].T, cmap='hot', interpolation='spline36')
		# plt.tight_layout()
		# plt.show()
		plt.imshow(gaussian_filter(dots_3D[:, y_res // 2, :].T, sigma=4), cmap='hot', interpolation='spline36')
		plt.tight_layout()
		plt.show()
		plt.imshow(gaussian_filter(dots_3D[:, :, -1].T, sigma=4), cmap='hot', interpolation='spline36')
		plt.tight_layout()
		plt.show()
		plt.imshow(gaussian_filter(dots_3D[:, :, z_res // 2].T, sigma=4), cmap='hot', interpolation='spline36')
		plt.tight_layout()
		plt.show()
		plt.imshow(gaussian_filter(dots_3D[:, :, 0].T, sigma=4), cmap='hot', interpolation='spline36')
		plt.tight_layout()
		plt.show()
		
		#
		#
		# dots, intensity = plane_intensity(positions, plane_vec=(0, 0, 1), plane_dot=(0, 0, 1.0001),
		#                                   # dots, intensity = plane_intensity(positions, plane_vec=(1, 0, 0.1), plane_dot=(0, 0, 0),
		#                                   x_res=51, y_res=51,
		#                                   x_max_min=(-1, 1), y_max_min=(-1, 1)
		#                                   )
		#
		# # exit()
		# x_coordinates = [point[0] for point in dots]
		# y_coordinates = [point[1] for point in dots]
		# # plt.scatter(x_coordinates, y_coordinates, color='blue', marker='o')
		# # plt.xlim(-1, 1)
		# # plt.ylim(-1, 1)
		# # plt.show()
		# plt.imshow(intensity.T, cmap='hot')  # , interpolation='bilinear')
		# plt.show()
		# plt.imshow(gaussian_filter(intensity.T, sigma=1), cmap='hot')  # , interpolation='bilinear')
		# plt.show()
		# # plt.imshow(gaussian_filter(intensity.T, sigma=4), cmap='Blues')#, interpolation='bilinear')
		# # plt.show()
		# # plt.imshow(uniform_filter(intensity.T, size=5), cmap='Blues')  # , interpolation='bilinear')
		# # plt.show()
		# # plt.imshow(uniform_filter(intensity.T, size=7), cmap='Blues')  # , interpolation='bilinear')
		# # plt.show()
		# time.sleep(2)
		# x_res, y_res, z_res = 101, 101, 51
		# xM = -5, 5
		# yM = -5, 5
		# zM = -2.5, 2.5
		# # dots = lines_dots(positions, x_res=x_res, y_res=y_res, z_res=z_res,
		# #                   x_max_min=xM, y_max_min=yM, z_max_min=zM,
		# #                   res_line=int(np.sqrt(x_res ** 2 + y_res ** 2 + z_res ** 2)), length_line=1)
