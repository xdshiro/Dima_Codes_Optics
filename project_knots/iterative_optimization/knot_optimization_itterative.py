"""
Optimization driver for trefoil and Hopf knot coefficient tuning.
Uses iterative random perturbations combined with geometric tests
to find coefficient sets that maximize singularity‐dot separation.
"""

from helper_scripts_for_itterative_optimization.knots_optimization_helper import *


def trefoil_optimization():
    """
    Optimize coefficients for a 5-mode trefoil knot field.
    1. Builds a 3D mesh grid.
    2. (Optional) Displays 2D mid-slice magnitude & phase for debugging.
    3. Runs `check_knot_mine` with circle, boundary, and visual tests.
    """
    # --- Initial coefficient guess (5 modes) ---
    coeffTest = [1.29, -3.95, 7.49, -3.28, -3.98]
    # For a 7-mode trefoil, you could instead use:
    # coeffTest = [1.29, -3.95, 7.49, -3.28, -3.98, 0, 0]

    # --- Mesh bounds & resolution ---
    xyMinMax = 4.0
    zMinMax = 1.1
    xRes = yRes = 171
    zRes = 111
    width = 1.3

    # Create the XYZ mesh (z from 0 to zMinMax)
    xyzMesh = fg.create_mesh_XYZ(
        xyMinMax, xyMinMax, zMinMax,
        xRes, yRes, zRes,
        zMin=0
    )

    # --- Optional: quick 2D mid-slice plot for initial field ---
    plot_test = False
    if plot_test:
        xyzMesh_full = fg.create_mesh_XYZ(
            xyMinMax, xyMinMax, zMinMax,
            xRes, yRes, zRes,
            zMin=None  # symmetric about z=0
        )
        fieldTest = fOAM.trefoil_mod(
            *xyzMesh_full,
            w=1.2, width=width, k0=1, z0=0.,
            aCoeff=coeffTest, coeffPrint=False
        )
        mid = fieldTest.shape[2] // 2
        fg.plot_2D(np.abs(fieldTest[:, :, mid]), axis_equal=True)
        fg.plot_2D(np.angle(fieldTest[:, :, mid]), axis_equal=True)
        plt.show()

    # --- Run the trefoil optimization (5 modes) ---
    check_knot_mine(
        xyzMesh,
        coeffTest,
        deltaCoeff=[0.05] * 5,     # step size for each coefficient
        steps=100,                  # number of iterations
        width=width,
        six_dots=False,             # allow variable number of dots
        circletest=True,            # enforce circle shape at mid-slice
        radiustest=0.02,            # radius for circle_test
        checkboundaries=True,       # enforce minimum spacing at boundary
        boundaryValue=0.1,          # threshold for empty_space_check
        testvisual=False,           # skip the final visual sanity check
        xyzMeshPlot=fg.create_mesh_XYZ(
            xyMinMax * 1.3, xyMinMax * 1.3, zMinMax * 2.5,
            71, 71, 81,
            zMin=None
        )
    )

    # To enable 7-mode optimization, uncomment & adjust:
    # check_knot_mine_2(
    #     xyzMesh,
    #     coeffTest,
    #     deltaCoeff=[0.05] * 7,  # step size for each coefficient
    #     steps=100,  # number of iterations
    #     width=width,
    #     six_dots=False,  # allow variable number of dots
    #     circletest=True,  # enforce circle shape at mid-slice
    #     radiustest=0.02,  # radius for circle_test
    #     checkboundaries=True,  # enforce minimum spacing at boundary
    #     boundaryValue=0.1,  # threshold for empty_space_check
    #     testvisual=False,  # skip the final visual sanity check
    #     xyzMeshPlot=fg.create_mesh_XYZ(
    #         xyMinMax * 1.3, xyMinMax * 1.3, zMinMax * 2.5,
    #         71, 71, 81,
    #         zMin=None
    #     )
    # )


def hopf_optimization():
    """
    Optimize coefficients for a Hopf knot field.
    1. Builds a 3D mesh grid.
    2. Runs `check_knot_mine_hopf` with specified geometric tests.
    """
    # --- Initial coefficient guess ---
    coeff = [3.59, -6.31, 5.47, 5.0]

    # --- Mesh bounds & resolution ---
    xyMinMax = 4.0
    zMinMax = 1.1
    xRes = yRes = 121
    zRes = 71
    width = 1.3

    # Create the XYZ mesh (z from 0 to zMinMax)
    xyzMesh = fg.create_mesh_XYZ(
        xyMinMax, xyMinMax, zMinMax,
        xRes, yRes, zRes,
        zMin=0
    )

    # --- Run the Hopf optimization ---
    check_knot_mine_hopf(
        xyzMesh,
        coeff,
        deltaCoeff=[0.2] * 5,      # step size for each coefficient
        steps=100,                  # number of iterations
        width=width,
        six_dots=False,             # allow variable number of dots
        circletest=False,           # skip circle test
        radiustest=0.02,
        checkboundaries=True,       # enforce boundary spacing
        boundaryValue=0.1,          # threshold for empty_space_check
        testvisual=False,           # skip visual sanity check
        xyzMeshPlot=fg.create_mesh_XYZ(
            xyMinMax * 1.3, xyMinMax * 1.3, zMinMax * 2,
            51, 51, 141,
            zMin=None
        )
    )


if __name__ == "__main__":
    trefoil_optimization()
    # hopf_optimization()