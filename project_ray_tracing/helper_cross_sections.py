"""
Ray‐trace a pvtrace Scene and collect all ray trajectories.

This module helps the main script called holder_diod_laser_plotting.py with
the scene creation

"""

import logging
from typing import List, Tuple

import numpy as np
import pvtrace as pv

# Suppress verbose logging from dependencies
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)


def scene_render_and_positions(
    scene: pv.Scene,
    rays_number: int = 50,
    random_seed: int = 0,
    open_browser: bool = True,
    show_3d: bool = True
) -> List[np.ndarray]:
    """
    Emit rays from the scene’s light sources, trace them through the geometry,
    and collect their positions.

    Parameters
    ----------
    scene : pvtrace.Scene
        The scene containing geometry and light sources.
    rays_number : int, optional
        Number of rays to emit (default: 50).
    random_seed : int, optional
        Seed for NumPy’s random number generator (default: 0).
    open_browser : bool, optional
        Whether to open the MeshCat browser for 3D visualization (default: True).
    show_3d : bool, optional
        Whether to render the 3D scene and ray paths (default: True).

    Returns
    -------
    positions : list of ndarray
        A list where each entry is an (N_i, 3) array of XYZ positions visited by ray i.
    """
    # Seed NumPy for reproducible ray emission
    np.random.seed(random_seed)

    # Initialize MeshCat renderer if requested
    renderer = None
    if show_3d:
        renderer = pv.MeshcatRenderer(wireframe=True, open_browser=open_browser)
        renderer.render(scene)

    positions: List[np.ndarray] = []
    ray_index = 0

    # Emit rays and trace their paths
    for ray in scene.emit(rays_number):
        ray_index += 1
        try:
            steps = pv.photon_tracer.follow(scene, ray)
        except ValueError:
            # e.g., invalid intersection
            continue
        except np.linalg.LinAlgError:
            # Numerical issues during tracing
            continue
        except pv.common.errors.GeometryError:
            # Ray got stuck or exited geometry unexpectedly
            continue

        # Unzip the (Ray, decision) tuples and collect positions
        path, _ = zip(*steps)
        coords = np.array([r.position for r in path], dtype=float)
        positions.append(coords)

        # Add the path to the 3D renderer if enabled
        if show_3d and renderer is not None:
            renderer.add_ray_path(path)

    return positions