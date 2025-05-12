"""
File: check_imports.py
Description:
    Demonstrates proper ordering of imports: standard library first, followed by third-party modules.
    Verifies that all required dependencies are installed; lists missing packages.

    To include new modules, add them to the `modules_to_check` list below and rerun.
    For pvtrace: install with `pip install pvtrace` then restart your Python environment.
"""

# ---------------------------------------------------------------------------- #
# 1. Standard Library Imports
# ---------------------------------------------------------------------------- #
import importlib
import os
import sys
import itertools
import pickle

# ---------------------------------------------------------------------------- #
# 2. Third-Party Imports
# ---------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import scipy
import scipy.io as sio
from scipy.special import assoc_laguerre
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline, splprep, splev, CloughTocher2DInterpolator
from scipy import integrate
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

import plotly.graph_objects as go
import pvtrace as pv  # pip install pvtrace

from aotools import opticalpropagation
from aotools.turbulence.phasescreen import ft_sh_phase_screen as psh

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors

from tqdm import trange

# ---------------------------------------------------------------------------- #
# 3. Add current directory to sys.path
# ---------------------------------------------------------------------------- #
package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
    print(f"Adding {package_dir} to sys.path")
    sys.path.insert(0, package_dir)

# ---------------------------------------------------------------------------- #
# 4. Dependency Check Routine
# ---------------------------------------------------------------------------- #

def check_package(module_name):
    """
    Attempt to import a module by name.
    Returns True if successful, False otherwise.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

# List all modules to verify
modules_to_check = [
    # Standard library
    "os", "sys", "itertools", "pickle",
    # Numeric & data handling
    "numpy", "pandas",
    # Plotting
    "matplotlib", "matplotlib.colors",
    # SciPy subpackages
    "scipy", "scipy.io", "scipy.special", "scipy.stats", "scipy.interpolate",
    "scipy.integrate", "scipy.fftpack", "scipy.ndimage",
    # Interactive graphics
    "plotly.graph_objects",
    # Ray tracing
    "pvtrace",
    # Atmospheric optics
    "aotools", "aotools.turbulence.phasescreen",
    # Machine learning
    "sklearn.preprocessing", "sklearn.linear_model", "sklearn.pipeline", "sklearn.neighbors",
    # Progress bars
    "tqdm"
]

missing = []
for mod in modules_to_check:
    if not check_package(mod):
        missing.append(mod)

if missing:
    print("Missing packages/modules detected:")
    for m in missing:
        print(f" - {m}")
    print("Install with pip, e.g.: pip install " + " ".join({m.split('.')[0] for m in missing}))
else:
    print("All required packages are installed.")
