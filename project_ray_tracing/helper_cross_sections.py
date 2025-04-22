"""
This module plots the cross-sections of the pvtrace ray-tracing simulations.
We can place different objects and structures.
For now spheres and cylinders are implemented.
XY, XZ, ZY implementer.

# XY
plt.scatter(crossings[:, 0], crossings[:, 1])
plt.xlabel('x')
plt.ylabel('y')
# YZ
plt.scatter(crossings[:, 1], crossings[:, 2])
plt.xlabel('x')
plt.ylabel('z')
# ZX
plt.scatter(crossings[:, 2], crossings[:, 0])
plt.xlabel('z')
plt.ylabel('x')
"""

# Some packages used by pvtrace are a little noisy
import logging

import numpy
import numpy as np
import matplotlib.pyplot as plt
import pvtrace as pv

logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)



def scene_render_and_positions(scene, rays_number=50, random_seed=0, open_browser=True, show_3d=True,):
    """
    The main purpose of this function is to propagate rays and return all there trajectories, as
    arrays of all dots they have "visited".
    Function also renders 3D picture if show_3d=True.
    :param scene: pv.Scene(world) with all Nodes (objects and light sources)
    :param random_seed: random seed for the ray-tracing
    :param open_browser: show in the browser
    :param show_3d: 3d rendering
    :return: positions of all the rays
    """
    positions = []
    np.random.seed(random_seed)
    if show_3d:
        vis = pv.MeshcatRenderer(wireframe=True, open_browser=open_browser)
        vis.render(scene)
    i = 1
    for rays in scene.emit(rays_number):
        try:
            steps = pv.photon_tracer.follow(scene, rays)
        except ValueError:
            continue
        except numpy.linalg.LinAlgError:
            print(f"skipped {i}")
            i += 1
            continue
        except pv.common.errors.GeometryError:
            print(f"skipped {i} (geometry error)")
            i += 1
            continue
        path, decisions = zip(*steps)
        positions_ray = []
        for ray in path:
            positions_ray.append(ray.position)
        positions.append(np.array(positions_ray))
        if show_3d:
            vis.add_ray_path(path)

    if show_3d:
        # vis.render(scene)
        pass
        vis = pv.MeshcatRenderer(wireframe=True)
    return positions


def dot_on_segment(dot, dot1, dot2):
    """
    Function checks if the dot is on the line in between dot1 and dot2. dot must be on
    the line.
    :param dot: (x, y, z) the do must be on the line (it's from the crossing_plane_line())
    :param dot1: (x1, y1, z1)
    :param dot2: (x2, y2, z2)
    :return: True if the line is on the segment, otherwise False
    """
    if dot is None:
        return False
    for x, x1, x2 in zip(dot, dot1, dot2):
        if x2 >= x1:
            if x1 <= x <= x2:
                pass
            else:
                return False
        else:
            if x2 <= x <= x1:
                pass
            else:
                return False
    return True


def crossing_plane_line(dot1, dot2, plane=(0, 0, 1, 0), check_segment=True):
    """
    Function finds the crossing of the line (in between dot1 and dot2) and the plane.
    Line : (x-x1)/(x2-x1) = (y-y1)/(y2-y1) = (z-z1)/(z2-z1)

    Plane: Ax + By + Cz + D = 0
    |x-x1  y-y1  z-z1 |
    |x2-x1 y2-y1 z2-z1| = 0
    |x3-x1 y3-y1 z3-z1|
    Example:
    (-5, -5, 0), (-5, 5, 0), (5, 5, 0)
    z = 0
    (https://ru.onlinemschool.com/math/assistance/cartesian_coordinate/plane/)

    Crossing: https://matworld.ru/analytic-geometry/tochka-peresechenija-prjamoj-i-ploskosti.php
    (look at equations, ignore the text)
    :param dot1: (x1, y1, z1)
    :param dot2: (x2, y2, z2)
    :param plane: (A, B, C, D) where Ax + By + Cz + D = 0
    :param check_segment: if True, checking if the don is on the segment in between dot1 and dot2
    :return: dot: (x, y ,z) or None if there is no crossing
    """
    x1, y1, z1 = dot1
    x2, y2, z2 = dot2
    m1 = x2 - x1
    p1 = y2 - y1
    l1 = z2 - z1
    # z = D plane
    A, B, C, D = plane
    matrix = [[p1, -m1, 0], [0, l1, -p1], [A, B, C]]
    vector = [[p1 * x1 - m1 * y1], [l1 * y1 - p1 * z1], [-D]]
    try:
        dot_cross = np.linalg.solve(matrix, vector)
        if not np.allclose(np.dot(matrix, dot_cross), vector):
            print(f'WRONG CROSSINGS')
        dot_ans = np.array([dot_cross[0, 0], dot_cross[1, 0], dot_cross[2, 0]])
    except np.linalg.LinAlgError:
        dot_cross = None

    if check_segment:
        check = dot_on_segment(dot_cross, dot1, dot2)
        if check:
            return dot_ans
        else:
            return None

    return dot_ans


def crossings_plane_rays(positions, plane):
    """
    Function return all crossings of rays with the plane.
    :param positions: Array [[[x1_r1, x2_r1, x3_r1], [x1_r1, x2_r1, x3_r1],...],
                            [[x1_r2, x2_r2, x3_r2], [x1_r2, x2_r2, x3_r2],...],...]
                            r1 - ray1, r2 - ray2.
                            Arrays from the photon_tracer.follow(scene, ray)
    :param plane: (A, B, C, D) where Ax + By + Cz + D = 0
    :return: [[x1, x2, x3], [x1, x2, x3],...] - all crossings
    """
    crossings = []
    for dots in positions:
        dot1 = dots[0]
        for dot2 in dots[1:]:
            dot_cross = crossing_plane_line(dot1, dot2, plane=plane)
            # print(dot_cross)
            if dot_cross is not None:
                crossings.append(dot_cross)
            dot1 = dot2
    return np.array(crossings)


def homogeneous_transform_inverse(pose):
    pose_inverse = np.zeros(np.shape(pose))
    R_AB = pose[:3, :3]
    d_AB = pose[:3, 3]
    pose_inverse[:3, :3] = R_AB.T
    pose_inverse[:3, 3] = -1 * R_AB.T@d_AB
    pose_inverse[3, 3] = 1
    return pose_inverse


def crossing_plane_cylinder(radius, plane, length, pose, circle_res=100):
    """
    This function return the circle of the plane with sphere crossing.
    For 3 different cross-section we are just renaming the axis.
    Parametric equation of the sphere is used. (check wiki: Sphere)

    ##################################################################################
    Function is not finished. Its performance is bad, in numerically finds the closest
    to the plane dots from the surface on mesh, not a real math solution.
    ##################################################################################

    3rd axis is pointing towards us. for XZ cross-section y is pointing to us,
    that means that y axis is X, and x axis is Z.

    This function uses homogeneous transformation matrices:
    http://www.ccs.neu.edu/home/rplatt/cs5335_fall2017/slides/homogeneous_transforms.pdf

    :param location: (x0, y0, z0) center of the sphere
    :param radius: radius of the sphere
    :param plane: (A, B, C, D) where Ax + By + Cz + D = 0
    :param circle_res: the number of dots for the circle
    :return: x_array, y_array, if not crossing: None, None
    """
    z_res = 300
    # number_edge_circles = 20
    phi = np.linspace(0, 2 * np.pi, circle_res)
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    z = np.linspace(-length / 2, length / 2, z_res)
    x_surf = np.array(list(x) * z_res)
    y_surf = np.array(list(y) * z_res)
    z_surf = np.array([z.T] * circle_res).T.reshape((z_res * circle_res))
    ones = np.ones(circle_res * z_res)
    p_ones = np.array([x_surf, y_surf, z_surf, ones])
    moved = pose @ p_ones
    x = moved[0]
    y = moved[1]
    z = moved[2]

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(x_surf, y_surf, z_surf)
    # ax.plot(x, y, z)
    # plt.show()
    # exit()
    # plt.plot(x, z)
    # ax = plt.subplot()
    # ax.set_aspect(1)
    # plt.show() ми фри пас
    xyz = np.array([x, y, z])
    distances_to_plane = xyz.T@plane[:3] + plane[3]
    indexes_sorted = numpy.argsort(np.abs(distances_to_plane))
    indexes = indexes_sorted[0:circle_res + z_res * 3]
    # indexes = indexes_sorted[0:circle_res + z_res * 40]
    # plt.plot(z[indexes])
    # plt.show()
    # exit()
    # plt.scatter(z[indexes], x[indexes])
    # ax = plt.subplot()
    # ax.set_aspect(1)
    # plt.show()
    # exit()
    if plane[:3] == (0, 0, 1):  # yz-cross-section
        return x[indexes], y[indexes]
    elif plane[:3] == (1, 0, 0):  # xz-cross-section
        return y[indexes], z[indexes]
    elif plane[:3] == (0, 1, 0):  # xy-cross-section
        return z[indexes], x[indexes]
    else:
        print(f'This cross-section is not implemented. Use XY, XZ, or YZ for now')
        return None, None

    # print(x_surf)
    # print(np.shape(x_surf))
    # z = np.linspace(0, 0, circle_res)
    # X, Y, Z = np.meshgrid(x, y ,z)

    # p_ones = np.array([x, y, z, ones])
    # pose_inverse = homogeneous_transform_inverse(pose)
    # print(pose_inverse)
    # print(pose)
    # moved = pose@p_ones
    # # moved = pose@p_ones
    # x = moved[0]
    # y = moved[1]
    # z = moved[2]
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(x, y, z)
    # plt.show()
    # plt.plot(x, z)
    # ax = plt.subplot()
    # ax.set_aspect(1)
    # plt.show()
    #
    # exit()
    # x = p_ones[0]
    # y = p_ones[1]
    # return x, y


def crossing_plane_sphere(location, radius, plane, circle_res=100):
    """
    This function return the circle of the plane with sphere crossing.
    For 3 different cross-section we are just renaming the axis.
    Parametric equation of the sphere is used. (check wiki: Sphere)

    3rd axis is pointing towards us. for XZ cross-section y is pointing to us,
    that means that y axis is X, and x axis is Z.

    :param location: (x0, y0, z0) center of the sphere
    :param radius: radius of the sphere
    :param plane: (A, B, C, D) where Ax + By + Cz + D = 0
    :param circle_res: the number of dots for the circle
    :return: x_array, y_array, if not crossing: None, None
    """
    if plane[:3] == (1, 0, 0):  # yz-cross-section
        z0, x0, y0 = location
    elif plane[:3] == (0, 1, 0):  # xz-cross-section
        y0, z0, x0 = location
    elif plane[:3] == (0, 0, 1):  # xy-cross-section
        x0, y0, z0 = location
    else:
        print(f'This cross-section is not implemented. Use XY, XZ, or YZ for now')
        return None, None
    z = -plane[3]
    if abs((z - z0) / radius) > 1:
        return None, None
    theta = np.arccos((z - z0) / radius)
    phi = np.linspace(0, 2 * np.pi, circle_res)
    x = x0 + radius * np.sin(theta) * np.cos(phi)
    y = y0 + radius * np.sin(theta) * np.sin(phi)
    return x, y


def plot_scene_2D(scene, plane):
    """
    Plotting the cross-section of the scene.
    :param scene: pv scene
    :param plane: (A, B, C, D) where Ax + By + Cz + D = 0 determines the cross-section
    :return: None (but it plot lines, so you need to plt.show() after)
    """
    for node in scene.root.leaves:
        if isinstance(node.geometry, pv.Sphere):
            location = node.location
            radius = node.geometry.radius
            x, y = crossing_plane_sphere(location, radius, plane)
            if x is not None:
                plt.plot(x, y, color='r')
        if isinstance(node.geometry, pv.Cylinder):
            location = node.location
            pose = node.pose
            radius = node.geometry.radius
            length = node.geometry.length
            x, y = crossing_plane_cylinder(radius, plane, length, pose, circle_res=100)
            if x is not None:
                plt.scatter(x, y, color='g')


def main_create_scene_test():
    world = pv.Node(
        name="world (air)",
        geometry=pv.Sphere(
            radius=3.0,
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
    cylinder1 = pv.Node(
        name="cyl1 (glass)",
        geometry=pv.Cylinder(
            radius=0.8,
            length=5.5,
            material=pv.Material(
                refractive_index=1.5,

            ),
        ),
        parent=world
    )
    cylinder1.rotate(np.pi/2, [0, 1, 0])
    # cylinder1.translate((0.5, -0.3, 0.2))
    # cylinder1.rotate(np.pi/5, [1, 0, 0])
    import functools
    light = pv.Node(
        name="Light (555nm)",
        light=pv.Light(direction=functools.partial(pv.cone, np.pi / 32)),
        parent=world
    )
    light.translate((0.0, 0.0, 3))
    light.rotate(np.pi, [1, 0, 0])
    scene = pv.Scene(world)
    return scene


if __name__ == '__main__':

    scene = main_create_scene_test()
    # positions = scene_render_and_positions(scene, rays_number=50, show_3d=True)
    positions = scene_render_and_positions(scene, rays_number=500, show_3d=True)
    # ax = plt.figure().add_subplot(projection='3d')
    # for position in positions:
    #     ax.plot(position[:, 0], position[:, 1], position[:, 2])
    # plt.show()
    # exit()
    plane = (0, 1, 0, 0)
    crossings = crossings_plane_rays(positions, plane)
    # plt.scatter(crossings[:, 0], crossings[:, 1])
    # plt.scatter(crossings[:, 1], crossings[:, 2])
    # plt.scatter(crossings[:, 2], crossings[:, 0])
    plot_scene_2D(scene, plane)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    ax = plt.subplot()
    ax.set_aspect(1)
    plt.tight_layout()
    plt.title(f'XY, Z={-plane[3]}')
    # XY
    plt.xlabel('x')
    plt.ylabel('y')
    # YZ
    # plt.xlabel('x')
    # plt.ylabel('z')
    # ZX
    # plt.xlabel('z')
    # plt.ylabel('x')
    plt.show()
