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
# from ipywidgets import interact
# from pvtrace import *

import pickle
from scipy.ndimage import gaussian_filter
import pvtrace as pv
import numpy as np
import functools
import collections
import time
import matplotlib.pyplot as plt
import cross_sections as cs  # mine
import my_functions.functions_general as fg
import my_functions.singularities as sing
import my_functions.plotings as pl
import my_functions.beams_and_pulses as bp
from dataclasses import replace
from numpy.random import Generator, PCG64, SeedSequence
from scipy import signal

from pvtrace import SurfaceDelegate

rng = Generator(PCG64(SeedSequence(102)))

d_bottom_hole = 2
h_bottom_hole = 0.5
d_holder = 5.2
h_holder = 4.3 - h_bottom_hole

total = 8.41
T = 5.54 / total
T_dir = (3.62 - 0.03) / total
R = (1.61 - 0.03) / total
A = 1 - T - R

assert R + T + A == 1, 'WRONG T, R, A'
TF = 1 - A
length = h_holder  # in m
L_A = -length / np.log(TF)  # absorption length
L = -length / np.log(T)  # attenuation length
L_S = L * L_A / (L_A - L)  # scattering length
print(T, R, A, T+R+A, L_S, L_A)
exit()
# print(L_A, L_S)
# exit()
# Some packages used by pvtrace are a little noisy

class PartialTopSurfaceMirror(pv.FresnelSurfaceDelegate):
    """ A section of the top surface is covered with a perfect mirrrors.

        All other surface obey the Fresnel equations.
    """

    # print(super(PartialTopSurfaceMirror, self).reflected_direction(surface, ray, geometry, container,
    #                                                                adjacent))
    # Xn0 = np.array((0, 0, -1))
    # # Xp0 = perpendicular_vector_3D(Xn0)
    # # Xc0 = np.cross(Xp0, Xn0)
    # # print(Xn0, Xp0, Xc0)
    # # exit()
    # Xp0 = np.array((1, 0, 0))
    # Xc0 = np.array((0, 1, 0))
    # Xn = geometry.normal(ray.position)
    # Xp = perpendicular_vector_3D(Xn)
    # Xc = np.cross(Xp, Xn)
    # dir0, up0, cross0 = Xp0, Xn0, Xc0
    # dir, up, cross = Xp, Xn, Xc
    # transform = [
    #     [np.dot(Xn, Xn0), np.dot(Xp, Xn0), np.dot(Xc, Xn0)],
    #     [np.dot(Xn, Xp0), np.dot(Xp, Xp0), np.dot(Xc, Xp0)],
    #     [np.dot(Xn, Xc0), np.dot(Xp, Xc0), np.dot(Xc, Xc0)]
    # ]
    # R1 = np.array([Xp0, Xn0, Xc0])
    # R2 = np.array([Xp, Xn, Xc])
    # transform2 = np.matmul(R1, R2.T)
    def reflected_direction_random_not_finished(self, surface, ray, geometry, container, adjacent):
        """
        Implementation of the scattering surface. We can control the scattering angles in scattered_angles.
        Vector transformation works only for a box rn, need to implement the general case.
        :return: direction of the scattered beam
        """

        # basis surface (bottom, directed upwards)
        # xn0, yn0, zy0 = 0, 0, -1

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
                # theta = np.arcsin(np.sqrt(p1) * np.sin(theta_max))
                phi = 2 * np.pi * p1
                # phi = np.pi / 5
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

        # current surface normal
        # xn, yn, zn = geometry.normal(ray.position)
        # xp, yp, zp = perpendicular_vector_3D(xn, yn, zn)
        # xc, yc, zc = np.cross([xn, yn, zn], [xp, yp, zp])
        u = np.array((0, 0, -1))
        v = perpendicular_vector_3D(u)
        uv = np.cross(u, v)
        # print(Xn0, Xp0, Xc0)
        # exit()
        x = geometry.normal(ray.position)
        print(x)
        exit()
        y = perpendicular_vector_3D(x)
        # xy = np.array((x[0]*y[2]-x[2]*y[0],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]))
        xy = np.cross(x, y)
        # print(u, v, uv)
        # print(x, y, xy)
        # transform3 = np.array([
        #     [np.dot(x, u), np.dot(x, v), np.dot(x, uv)],
        #     [np.dot(y, u), np.dot(y, v), np.dot(y, uv)],
        #     [np.dot(xy, u), np.dot(xy, v), np.dot(xy, uv)]
        # ])
        R1 = np.array([u, v, uv])
        R2 = np.array([x, y, xy])
        transform3 = np.matmul(R1.T, R2)
        # print(transform3)
        # exit()
        phi_new, theta_new = scattered_angles(absorption=1)
        # print(np.matmul(x, transform3), u)
        # print(np.matmul(u, transform3), x)
        # print(np.matmul(u, transform3.T), x)
        # print(np.matmul(x, transform3.T), u)
        # print(np.matmul(x.T, transform3), u)
        # print(np.matmul(transform3, x), u)
        # print(np.matmul(transform3, u), x)
        # print(np.matmul(transform3.T, x), u)
        # print(np.matmul(transform3.T, u), x)
        # print(np.matmul(transform3, x.T), u)
        # print('a')
        # print(np.matmul(v, transform3), y)
        # print(np.matmul(uv, transform3), xy)
        x = np.sin(theta_new) * np.cos(phi_new)
        y = np.sin(theta_new) * np.sin(phi_new)
        z = np.cos(theta_new)
        # z = np.sin(theta_new) * np.cos(phi_new)
        # y = np.sin(theta_new) * np.sin(phi_new)
        # x = np.cos(theta_new)

        Xnew0 = np.array((x, y, z))

        Xnew = tuple(np.matmul(Xnew0, transform3))
        return Xnew

        # _, phi0, theta0 = fg.spherical_coordinates(xn0, yn0, zy0)
        # _, phin, thetan = fg.spherical_coordinates(xn, yn, zn)
        #
        # phi_delta, theta_delta = phin - phi0, thetan - theta0
        #
        # phi, theta = scattered_angles()
        # phi_new, theta_new = phi + phi_delta, theta + theta_delta
        #
        # x = np.sin(theta_new) * np.cos(phi_new)
        # y = np.sin(theta_new) * np.sin(phi_new)
        # z = np.cos(theta_new)
        # print(x, y, z)
        # coords = tuple(spherical_to_cart(theta_new, phi_new))

        # return x, y, z
        # normal = geometry.normal(ray.position)
        # if np.allclose(normal, TOP_SURFACE):
        #     return 0, 0, -1
        # elif np.allclose(normal, BOT_SURFACE):
        #     return x, y, z
        # elif np.allclose(normal, LEFT_SURFACE):
        #     return 1, 0, 0
        # elif np.allclose(normal, RIGHT_SURFACE):
        #     return -1, 0, 0
        # elif np.allclose(normal, FRONT_SURFACE):
        #     return 0, 1, 0
        # elif np.allclose(normal, BACK_SURFACE):
        #     return 0, -1, 0
        # return normal
        # return x, y, z

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

        # _, phi0, theta0 = fg.spherical_coordinates(xn0, yn0, zy0)
        # _, phin, thetan = fg.spherical_coordinates(xn, yn, zn)
        #
        # phi_delta, theta_delta = phin - phi0, thetan - theta0
        #
        # phi, theta = scattered_angles()
        # phi_new, theta_new = phi + phi_delta, theta + theta_delta
        #
        # x = np.sin(theta_new) * np.cos(phi_new)
        # y = np.sin(theta_new) * np.sin(phi_new)
        # z = np.cos(theta_new)
        # print(x, y, z)
        # coords = tuple(spherical_to_cart(theta_new, phi_new))

        # return x, y, z
        # normal = geometry.normal(ray.position)
        # if np.allclose(normal, TOP_SURFACE):
        #     return 0, 0, -1
        # elif np.allclose(normal, BOT_SURFACE):
        #     return x, y, z
        # elif np.allclose(normal, LEFT_SURFACE):
        #     return 1, 0, 0
        # elif np.allclose(normal, RIGHT_SURFACE):
        #     return -1, 0, 0
        # elif np.allclose(normal, FRONT_SURFACE):
        #     return 0, 1, 0
        # elif np.allclose(normal, BACK_SURFACE):
        #     return 0, -1, 0
        # return normal
        # return x, y, z

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
        # Get the surface normal to determine which surface has been hit.
        # normal = geometry.normal(ray.position)
        # print(self, surface, ray, geometry, container, adjacent)
        # exit()
        # Normal are outward facing

        x, y, z = ray.position[0], ray.position[1], ray.position[2]
        # return super(PartialTopSurfaceMirror, self).reflectivity(surface, ray, geometry, container,
        #                                                                  adjacent)
        if geometry.radius == d_holder / 2 and geometry.length == h_holder:  # real holder
            # print(z, h_holder)
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
        # if np.allclose(normal, BOT_SURFACE):
        #     x, y = ray.position[0], ray.position[1]
        #     if x > -0.5 and y > -0.5:
        #         return 1


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
            radius=4,
            material=pv.Material(refractive_index=1.0,

                                 )
        )
    )
    # pv.Node(
    #     name="sphere1 (glass)",
    #     geometry=pv.Sphere(
    #         radius=0.8,
    #         material=pv.Material(
    #             refractive_index=1.0,
    #
    #         ),
    #     ),
    #     parent=world
    # )
    # sphere2 = pv.Node(
    #     name="sphere2 (glass)",
    #     geometry=pv.Sphere(
    #         radius=0.35,
    #         material=pv.Material(
    #             refractive_index=1.0,
    #             components=[
    #                 pv.Scatterer(coefficient=3.1)
    #             ]
    #         ),
    #     ),
    #     parent=world
    # )
    # sphere2.translate((0, 0.3, 0.3))
    # sphere3 = pv.Node(
    #     name="sphere3 (glass)",
    #     geometry=pv.Sphere(
    #         radius=0.3,
    #         material=pv.Material(
    #             refractive_index=1.0,
    #             components=[
    #                 pv.Scatterer(coefficient=3.1)
    #             ]
    #         ),
    #     ),
    #     parent=world
    # )
    #
    # sphere3.translate((0, -0.2, -0.2))

    # box = pv.Node(
    #     name="cyl1 (glass)",
    #     geometry=pv.Box(
    #         (1.0, 1.0, 2),
    #         material=pv.Material(
    #             refractive_index=1.05,
    #             surface=pv.Surface(delegate=PartialTopSurfaceMirror()),
    #             components=[
    #                 pv.Absorber(coefficient=1.1),
    #                 pv.Scatterer(coefficient=0.181)
    #             ]
    #
    #         ),
    #     ),
    #     parent=world
    # )
    # cylinder1.rotate(np.pi/2, [0, 1, 0])

    # cylinder1.translate((0.5, -0.3, 0.2))
    # cylinder1.rotate(np.pi/5, [1, 0, 0])
    import functools
    light = pv.Node(
        name="Light (555nm)",
        # light=pv.Light(direction=functools.partial(pv.cone, np.pi / 32)),
        light=pv.Light(position=functools.partial(collimated_beam, 0.1),
                       wavelength=lambda: 555
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
        geometry=pv.Cylinder(
            radius=d_holder / 2,
            length=h_holder,
            material=pv.Material(
                refractive_index=1.05,
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
    # cylinder2 = pv.Node(
    #     name="bottom_around_hole",
    #     geometry=pv.Cylinder(
    #         radius=d_holder / 2,
    #         length=h_bottom_hole,
    #         material=pv.Material(
    #             refractive_index=1,
    #             # surface=pv.Surface(delegate=PartialTopSurfaceMirror()),
    #
    #         ),
    #     ),
    #     parent=parent
    # )
    # cylinder2.translate([0, 0, -h_bottom_hole / 2])
    cylinder3 = pv.Node(
        name="bottom_hole",
        geometry=pv.Cylinder(
            radius=d_bottom_hole / 2,
            length=h_bottom_hole,
            material=pv.Material(
                refractive_index=1,
                surface=pv.Surface(delegate=PartialTopSurfaceMirror()),
                # components=[
                #     pv.Absorber(coefficient=absor),
                #     pv.Scatterer(coefficient=scat)
                # ]

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
    # r_random = np.random.normal(0, r, 1)
    r_random = np.random.normal(0, 0.5, 1)
    # r_random = np.random.uniform(0.5, 0.5 + r, 1)
    # print(r_random )
    p2 = np.random.uniform(0, 1, 1)
    # p2 = 0.75
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
    # r_random = np.random.normal(0, r, 1)
    # x = np.random.normal(0, r, 1)
    # y = np.random.normal(0, r, 1)
    # r_random = np.random.uniform(0.5, 0.5 + r, 1)
    # print(r_random )
    # p2 = np.random.uniform(0, 1, 1)
    # p2 = 0.75
    # phi = np.pi * p2
    x, y = np.random.multivariate_normal((0, 0), [[r ** 2, 0], [0, r ** 2]], 1)[0]
    coords = (x, y, 0)
    return coords


def light_beam(parent):
    r = 0.1
    light = pv.Node(
        name="Light (555nm)",
        # light=pv.Light(direction=functools.partial(pv.cone, np.pi / 128)),
        light=pv.Light(position=functools.partial(collimated_beam, r),
                       wavelength=lambda: 555
                       ),
        parent=parent,
    )
    light.translate([0, 0, 9.5])
    light.rotate(np.pi, [1, 0, 0])


def pv_scene_real(structure=structure_sample, light=light_beam, absor=1e-10, scat=1e-10):
    """

    :param structure:
    :param light:
    :return:
    """
    world = pv.Node(
        name="world (air)",
        geometry=pv.Sphere(
            radius=10.0,
            material=pv.Material(refractive_index=1.0),
        )
    )

    structure(parent=world, absor=absor, scat=scat)
    light(parent=world)
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
    # print(dots_scaled)
    dots_round = np.rint(dots_scaled).astype(int)
    dots_tuples = [tuple(point) for point in dots_round]
    #  / [scale_coeff_x, scale_coeff_y, 1]
    dots_ind_collection = collections.Counter(dots_tuples)
    x_ind = np.arange(int(x_max_min[0] / delta_x), int(x_max_min[1] / delta_x))
    y_ind = np.arange(int(y_max_min[0] / delta_y), int(y_max_min[1] / delta_y))
    intensity = np.zeros((x_res, y_res))
    print(dots_ind_collection)
    z = np.rint(plane_dot[2]).astype(int)
    for ind_i, i in enumerate(x_ind):
        for ind_j, j in enumerate(y_ind):
            # print((i, j, int(plane_dot[2])))
            if (i, j, z) in dict(dots_ind_collection):
                # print((i, j, z))
                intensity[ind_i, ind_j] = dots_ind_collection[(i, j, z)]
    # mesh = np.meshgrid(np.arange(x_res), np.arange(y_res), indexing='ij')
    print(intensity)
    return dots_all_rays, intensity

    #         dot1 = dot2
    # dots = np.concatenate(np.array(dots_all_rays, dtype=object), axis=0)


# plane_intensity()
# exit()

from scipy.ndimage import uniform_filter

if __name__ == '__main__':
    print('hi')
    scene = pv_scene_real(absor=1. / L_A, scat=1. / L_S)
    number_rays = 300000
    positions = cs.scene_render_and_positions(scene, rays_number=number_rays, show_3d=0)

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
    # max = np.max(dots_3D_sat) * 0.02
    # dots_3D_sat[dots_3D_sat > max] = max

    np.save(f'Z7_{x_res}_{number_rays}', dots_3D)
    # np.save(f'Z7_positions_{number_rays}', np.array(positions))
    with open(f'Z7_positions_{number_rays}.pkl', 'wb') as file:
        pickle.dump(positions, file)
    plt.imshow(dots_3D[:, y_res // 2, :].T, cmap='hot', interpolation='spline36')
    plt.tight_layout()
    plt.show()
    plt.imshow(dots_3D[:, :, -1].T, cmap='hot', interpolation='spline36')
    plt.tight_layout()
    plt.show()
    plt.imshow(dots_3D[:, :, z_res // 2].T, cmap='hot', interpolation='spline36')
    plt.tight_layout()
    plt.show()
    plt.imshow(dots_3D[:, :, 0].T, cmap='hot', interpolation='spline36')
    plt.tight_layout()
    plt.show()
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

    exit()

    dots, intensity = plane_intensity(positions, plane_vec=(0, 0, 1), plane_dot=(0, 0, 1.0001),
                                      # dots, intensity = plane_intensity(positions, plane_vec=(1, 0, 0.1), plane_dot=(0, 0, 0),
                                      x_res=51, y_res=51,
                                      x_max_min=(-1, 1), y_max_min=(-1, 1)
                                      )

    # exit()
    x_coordinates = [point[0] for point in dots]
    y_coordinates = [point[1] for point in dots]
    # plt.scatter(x_coordinates, y_coordinates, color='blue', marker='o')
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.show()
    plt.imshow(intensity.T, cmap='hot')  # , interpolation='bilinear')
    plt.show()
    plt.imshow(gaussian_filter(intensity.T, sigma=1), cmap='hot')  # , interpolation='bilinear')
    plt.show()
    # plt.imshow(gaussian_filter(intensity.T, sigma=4), cmap='Blues')#, interpolation='bilinear')
    # plt.show()
    # plt.imshow(uniform_filter(intensity.T, size=5), cmap='Blues')  # , interpolation='bilinear')
    # plt.show()
    # plt.imshow(uniform_filter(intensity.T, size=7), cmap='Blues')  # , interpolation='bilinear')
    # plt.show()
    time.sleep(2)
    x_res, y_res, z_res = 101, 101, 51
    xM = -5, 5
    yM = -5, 5
    zM = -2.5, 2.5
    # dots = lines_dots(positions, x_res=x_res, y_res=y_res, z_res=z_res,
    #                   x_max_min=xM, y_max_min=yM, z_max_min=zM,
    #                   res_line=int(np.sqrt(x_res ** 2 + y_res ** 2 + z_res ** 2)), length_line=1)
