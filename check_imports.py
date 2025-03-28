"""
File: check_imports.py
Description:
    This file demonstrates the proper ordering of imports and includes a routine to verify that
    all required packages/modules are installed. It is divided into two main sections:
      1. Standard library imports
      2. Third-party imports

    If any required package is missing, a message is printed listing the missing modules.
    Otherwise, it confirms that all packages are installed.
"""

### !!!!!
# add these to the list:
# import itertools
# from matplotlib.colors import LinearSegmentedColormap
# from scipy.stats import norm
# import plotly.graph_objects as go
#
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.neighbors import NearestNeighbors
#
# from scipy.interpolate import UnivariateSpline
# from scipy.interpolate import splprep, splev
# from tqdm import trange




# ---------------------------
# Standard Library Imports
# ---------------------------
import importlib
import os
import sys
from os import listdir
from os.path import isfile, join
import pickle

# ---------------------------
# Third-Party Imports
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.special import assoc_laguerre
from scipy.interpolate import CloughTocher2DInterpolator
from scipy import integrate
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import plotly.graph_objects as go
from aotools import opticalpropagation
from aotools.turbulence.phasescreen import ft_sh_phase_screen as psh

# ---------------------------
# Adding package to the path
# ---------------------------
# Define the absolute path to the directory that contains the extra_functions_package
package_dir = os.path.abspath("C:/Users/Cmex-/PycharmProjects/Dima_Codes_Optics")
# Add this directory to sys.path if it's not already there
if package_dir not in sys.path:
    print("Adding " + package_dir + " to sys.path")
    sys.path.insert(0, package_dir)
# ---------------------------
# Package Check Routine
# ---------------------------
def check_package(module_name):
    """
    Attempts to import a module by name.
    Returns True if the module is successfully imported, False otherwise.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

# List of module names to check
modules_to_check = [
    "os",
    "pickle",
    "numpy",
    "pandas",
    "matplotlib.pyplot",
    "scipy.io",
    "scipy.special",
    "scipy.interpolate",
    "scipy",
    "scipy.fftpack",
    "plotly.graph_objects",
    "aotools",
    "aotools.turbulence.phasescreen"
]

# Check for missing modules
missing_modules = [mod for mod in modules_to_check if not check_package(mod)]

if missing_modules:
    print("The following packages/modules are not installed:")
    for mod in missing_modules:
        print(" -", mod)
else:
    print("All packages are installed.")