"""
Module for simulating an integrating sphere using the pvtrace package.

This script defines a geometry representing an integrating sphere,
configures optical properties (transmission, reflection, absorption),
sets up a beam source, implements custom surface behaviors,
provides utilities for converting ray-trace output
into voxelized 3D intensity arrays and planar projections,
and includes a streamlined main driver for simulations and plotting.

Author: [Your Name]
Date: [Today's Date]
"""

import logging
import pickle
from scipy.ndimage import gaussian_filter
import numpy as np
import pvtrace as pv
import matplotlib.pyplot as plt
import functools
import collections
import helper_cross_sections as cs  # Custom cross-section helpers
from numpy.random import Generator, PCG64, SeedSequence
from pvtrace.geometry.cylinder import CylinderRough

#--------------------------------------------------------------------------------
# Logging Configuration: suppress verbose output from dependencies
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

#--------------------------------------------------------------------------------
# Random Number Generator: reproducible for scattering calculations
rng = Generator(PCG64(SeedSequence(103)))

#--------------------------------------------------------------------------------
# Geometry Parameters (cm)
d_bottom_hole = 2.0
h_bottom_hole = 0.5
d_holder = 5.2
h_holder = 4.3 - h_bottom_hole

#--------------------------------------------------------------------------------
# Optical Properties (Case Z7)
total_energy = 8.41
T = 5.54 / total_energy
T_dir = (3.62 - 0.03) / total_energy
R = (1.61 - 0.03) / total_energy
A = 1.0 - T - R
assert np.isclose(T + R + A, 1.0)
length_m = h_holder * 1e-2
TF = 1.0 - A
L_A = -length_m / np.log(TF)
L = -length_m / np.log(T)
L_S = L * L_A / (L_A - L)

#--------------------------------------------------------------------------------
# Beam Source Parameters
r_source = 30.0 / 23.0 / 2.0  # cm
r_focus = 1.0e-3             # m
dist_above_holder = h_holder + 4.6 + 23.0 / 6.0 + 1.0  # cm
focus_positions = [-h_holder - 4.6 - 23.0 / 6.0]

#--------------------------------------------------------------------------------
# Custom Surface Delegate: PartialTopSurfaceMirror
class PartialTopSurfaceMirror(pv.FresnelSurfaceDelegate):
    """Partial mirror on top/bottom of cylinder; curved sides perfect reflector."""
    def reflected_direction(self, surface, ray, geometry, container, adjacent):
        n = np.array(geometry.normal(ray.position))
        d = np.array(ray.direction)
        return tuple(d - 2.0 * np.dot(d, n) * n)
    def reflectivity(self, surface, ray, geometry, container, adjacent):
        x, y, z = ray.position
        # Top face
        if getattr(geometry, 'radius', None) == d_holder/2 and geometry.length == h_holder:
            if np.isclose(z, h_holder/2) and x**2+y**2<(d_holder/2)**2:
                return super().reflectivity(surface, ray, geometry, container, adjacent)
            if np.isclose(z, -h_holder/2) and x**2+y**2<(d_bottom_hole/2)**2:
                return super().reflectivity(surface, ray, geometry, container, adjacent)
            return 1.0
        # Bottom hole
        if getattr(geometry, 'radius', None) == d_bottom_hole/2 and geometry.length == h_bottom_hole:
            if np.isclose(z, h_bottom_hole/2) or np.isclose(z, -h_bottom_hole/2):
                return super().reflectivity(surface, ray, geometry, container, adjacent)
            return 1.0
        return 1.0

#--------------------------------------------------------------------------------
# Utility Functions

def cylindrical_to_cart(r, phi, z=0.0):
    x = r*np.cos(phi); y = r*np.sin(phi)
    return (x, y, z) if np.isscalar(x) else np.column_stack((x,y,z))

def collimated_beam(r):
    if r<=0: raise ValueError(f"Aperture radius must be positive, got {r}")
    x,y = rng.normal(0, r, 2)
    while x**2+y**2>r**2: x,y = rng.normal(0, r, 2)
    return (x, y, 0.0)

random_counter=0; random_seed=0

def positions_directions(focus_z):
    global random_counter, random_seed
    if random_counter%2==0: random_seed+=1
    gen = np.random.default_rng(random_seed)
    x0,y0 = gen.normal(0, r_source, 2)
    while x0**2+y0**2>r_source**2: x0,y0 = gen.normal(0, r_source, 2)
    xf,yf = gen.normal(0, r_focus, 2)
    zf = focus_z + gen.uniform(-1e-3, 1e-3)
    v = np.array([xf-x0, yf-y0, zf]); v/=np.linalg.norm(v)
    random_counter+=1
    return (x0,y0,0.0), tuple(v)
def position(f): return positions_directions(f)[0]
def direction(f): return positions_directions(f)[1]

#--------------------------------------------------------------------------------
# Scene & Structure Construction
def structure_sample(parent, absor=1.0, scat=1.0):
    holder = pv.Node(
        name="holder",
        geometry=CylinderRough(radius=d_holder/2, length=h_holder,
            material=pv.Material(refractive_index=1.1,
                surface=pv.Surface(delegate=PartialTopSurfaceMirror()),
                components=[pv.Absorber(coefficient=absor), pv.Scatterer(coefficient=scat)]))
        ,parent=parent)
    holder.translate([0,0,h_holder/2])
    bottom = pv.Node(name="bottom_hole",
        geometry=pv.Cylinder(radius=d_bottom_hole/2,length=h_bottom_hole,
            material=pv.Material(refractive_index=1.0,
                surface=pv.Surface(delegate=PartialTopSurfaceMirror()))),
        parent=parent)
    bottom.translate([0,0,-h_bottom_hole/2]); return holder, bottom

def light_beam(parent, focus_z):
    light = pv.Node(name="Light",
        light=pv.Light(direction=functools.partial(direction,focus_z),
                       position=functools.partial(position,focus_z), wavelength=lambda:555),
        parent=parent)
    light.translate([0,0,dist_above_holder]); light.rotate(np.pi,[1,0,0]); return light

def pv_scene_real(absor=1e-10,scat=1e-10,focus=1e9):
    world = pv.Node(name="world",geometry=pv.Sphere(radius=17,material=pv.Material(refractive_index=1.0)))
    structure_sample(world,absor,scat); light_beam(world,focus)
    return pv.Scene(world)

def plane_intensity(positions,plane_vec=(0,0,1),plane_dot=(0,0,0),x_res=21,y_res=21,
                    x_max_min=(-1,1),y_max_min=(-1,1)):
    a,b,c=plane_vec; d=-np.dot(plane_vec,plane_dot)
    plane=np.array([a,b,c,d]); pts=[]
    for path in positions:
        for p1,p2 in zip(path[:-1],path[1:]):
            n1,n2=np.dot(p1,plane[:3])+d,np.dot(p2,plane[:3])+d
            if n1*n2<0:
                t=- (np.dot(plane[:3],p1)+d)/np.dot(plane[:3],p2-p1)
                pts.append(p1+t*(p2-p1))
    pts=np.array(pts);
    dx,dy=(x_max_min[1]-x_max_min[0])/(x_res-1),(y_max_min[1]-y_max_min[0])/(y_res-1)
    idx=np.rint(pts[:,:2]*[1/dx,1/dy]).astype(int)
    cnt=collections.Counter(map(tuple,idx))
    I=np.zeros((x_res,y_res),int)
    for (i,j),v in cnt.items():
        if 0<=i<x_res and 0<=j<y_res: I[i,j]=v
    return pts, I

#--------------------------------------------------------------------------------
# Main Driver: simulation and plotting
#--------------------------------------------------------------------------------
def main():
    number_rays=100; show_3d=True
    x_res=y_res=z_res=221
    # Domain bounds
    xM=(-d_holder/2-0.1,d_holder/2+0.1)
    yM=xM; zM=(-h_bottom_hole*1.000001,h_holder*1.000001)

    for focus in focus_positions:
        scene=pv_scene_real(absor=1./L_A,scat=1./L_S,focus=focus+dist_above_holder)
        positions=cs.scene_render_and_positions(scene,rays_number=number_rays,
                                               show_3d=show_3d,random_seed=2)
        # Voxelize
        dots=cs.lines_dots(positions,x_res,y_res,z_res,xM,yM,zM,
                           res_line=int(np.sqrt(x_res**2+y_res**2+z_res**2)),length_line=1)
        dots_3D=cs.array_3D_intensity_from_dots(dots,x_res,y_res,z_res,xM,yM,zM)

        # Save data
        tag=f"Z7_f{focus:.2f}_r{r_source:.2f}_{x_res}res_{number_rays}rays"
        np.save(f"{tag}.npy",dots_3D)
        with open(f"{tag}_pos.pkl","wb") as f: pickle.dump(positions,f)

        # Plot slices
        slices=[dots_3D[:,y_res//2,:],dots_3D[:,:,z_res//2],dots_3D[:,0,:]]
        titles=["XY mid-plane (y)","XY mid-plane (z)","XZ plane (z=0)"]
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for ax,slc,title in zip(axes,slices,titles):
            im=gaussian_filter(slc.T,sigma=4)
            ax.imshow(im,cmap='hot',origin='lower')
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout(); plt.show()

if __name__=='__main__':
    main()
