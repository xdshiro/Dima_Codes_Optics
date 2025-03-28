"""
Optical Field Processing Package
================================

This package provides tools for generating, processing, and visualizing optical fields,
including laser beams, Laguerre-Gaussian (LG) modes, singularities, and turbulence effects.
It includes functions for:

- Knot field calculations
- Optical beam generation and manipulation
- Beam center detection
- Optical turbulence simulation
- Singularity analysis in optical fields
- 2D & 3D visualization

Modules:
--------
- `all_knots_functions`: Braid knot field calculations and visualization.
- `beams_and_pulses`: Optical beam shape generation and manipulation.
- `center_beam_search`: Beam center detection and correction.
- `data_generation_old`: Field processing from `.mat` files.
- `dots_processing`: Singularity dot filtering and processing.
- `functions_general`: General mathematical utilities and mesh generation.
- `functions_turbulence`: Optical turbulence simulation.
- `plotings`: 2D and 3D visualization tools.
- `singularities`: Detection and processing of singularities in optical fields.

Author:
-------
Dmitrii Tsvetkov

"""

from . import all_knots_functions
from . import beams_and_pulses
from . import center_beam_search
from . import data_generation_old
from . import dots_processing
from . import functions_general
from . import functions_turbulence
from . import plotings
from . import singularities

__all__ = [
    "all_knots_functions",
    "beams_and_pulses",
    "center_beam_search",
    "data_generation_old",
    "dots_processing",
    "functions_general",
    "functions_turbulence",
    "plotings",
    "singularities"
]
