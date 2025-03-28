"""
This module has classes of different singularities and functions for processing singularities.

The module provides various functions for finding, plotting, and manipulating singularities
in optical fields. It also includes classes for handling 3D singularities and knots.

Functions:
    - plot_knot_dots: Plots 3D or 2D scatters from the field or dictionary with dots.
    - plane_singularities_finder_9dots: Helper function to find singularities in a 2D plane using 9 dots.
    - plane_singularities_finder_4dots: Helper function to find singularities in a 2D plane using 4 dots.
    - fill_dict_as_matrix_helper: Helper function to fill a dictionary as a matrix.
    - cut_non_oam: Finds singularities and returns a 3D array with values and non-values.
    - get_singularities: Simplifies cut_non_oam by returning an array of singularities.
    - W_energy: Calculates the total power in the Oxy plane.
    - Jz_calc_no_conj: Calculates the z-component of the angular momentum without conjugation.
    - integral_number2_OAMcoefficients_FengLiPaper: Calculates the weight of OAM at a radius using FengLi paper's method.
    - integral_number3_OAMpower_FengLiPaper: Calculates the total power in the OAM with a specific charge.
    - knot_build_pyknotid: Builds a normalized pyknotid knot.
    - fill_dotsKnotList_mine: Fills a list of dots by removing charge sign and arranging them into a list.
    - dots_dens_reduction: Reduces the density of singularity lines by removing extra dots.
"""

import numpy as np
import extra_functions_package.plotings as pl
import extra_functions_package.functions_general as fg
import matplotlib.pyplot as plt
from scipy import integrate
# import functions_OAM_knots as fOAM
# import functions_general as fg
# import matplotlib.pyplot as plt
# import functions_general as fg
# import pyknotid.spacecurves as sp
# import extra_functions_package.beams_and_pulses as bp
# import timeit
# import sympy
# from python_tsp.distances import euclidean_distance_matrix
# from python_tsp.heuristics import solve_tsp_local_search
# from vispy import app

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
We can make a graph for tsp, so it is not searching for all the dots, only close z
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
trefoilW16.fill_dotsList() this function is what makes everything slow 

Stop if the distance is too big. To find a hopf 
"""


def plot_knot_dots(field, bigSingularity=False, axesAll=True,
                   size=plt.rcParams['lines.markersize'] ** 2, color=None, show=True):
    """
    Plots 3D or 2D scatters from the field or dictionary with dots.

    Parameters:
        field: Complex field or dictionary with dots to plot.
        bigSingularity: Boolean to include big singularities.
        axesAll: Boolean to include all axes.
        size: Size of the dots.
        color: Color of the dots.
        show: Boolean to show the plot.

    Returns:
        Matplotlib axis object.
    """
    if isinstance(field, dict):
        dotsOnly = field
    else:
        dotsFull, dotsOnly = cut_non_oam(np.angle(field),
                                         bigSingularity=bigSingularity, axesAll=axesAll)
    dotsPlus = np.array([list(dots) for (dots, OAM) in dotsOnly.items() if OAM == 1])
    dotsMinus = np.array([list(dots) for (dots, OAM) in dotsOnly.items() if OAM == -1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if len(np.shape(dotsPlus)) == 2:
        pl.plot_scatter_3D(dotsPlus[:, 0], dotsPlus[:, 1], dotsPlus[:, 2], ax=ax, size=size, color=color, show=False)
        if len(np.shape(dotsMinus)) == 2:
            pl.plot_scatter_3D(dotsMinus[:, 0], dotsMinus[:, 1], dotsMinus[:, 2], ax=ax, size=size, color=color
                               , show=False)
    else:
        if len(np.shape(dotsPlus)) == 2:
            pl.plot_scatter_3D(dotsMinus[:, 0], dotsMinus[:, 1], dotsMinus[:, 2], ax=ax, size=size, color=color
                               , show=False)
        else:
            print(f'no singularities to plot')
    if show:
        plt.show()
    return ax


def plane_singularities_finder_9dots(E, circle, value, nonValue, bigSingularity):
    """
    Helper function to find singularities in a 2D plane using 9 dots.

    Parameters:
        E: Complex field.
        circle: Radius of the circle around singularities.
        value: Value assigned to singularities.
        nonValue: Value assigned to non-singularities.
        bigSingularity: Boolean to include big singularities.

    Returns:
        Array with singularities and non-singularities.
    """

    def check_dot_oam_9dots_helper(E):
        flagPlus, flagMinus = True, True
        minIndex = np.argmin(E)
        for i in range(minIndex - len(E), minIndex - 1, 1):
            if E[i] >= E[i + 1]:
                flagMinus = False
                break
        maxIndex = np.argmax(E)
        for i in range(maxIndex - len(E), maxIndex - 1, 1):
            if E[i] <= E[i + 1]:
                flagPlus = False
                break
        if flagPlus:
            # print(np.arg() + np.arg() - np.arg() - np.arg())
            return True, +1
        elif flagMinus:
            return True, -1
        return False, 0

    shape = np.shape(E)
    ans = np.zeros(shape)
    for i in range(1, shape[0] - 1, 1):
        for j in range(1, shape[1] - 1, 1):
            Echeck = np.array([E[i - 1, j - 1], E[i - 1, j], E[i - 1, j + 1],
                               E[i, j + 1], E[i + 1, j + 1], E[i + 1, j],
                               E[i + 1, j - 1], E[i, j - 1]])
            oamFlag, oamValue = check_dot_oam_9dots_helper(Echeck)
            if oamFlag:
                ######
                ans[i - circle:i + 1 + circle, j - circle:j + 1 + circle] = nonValue
                #####
                ans[i, j] = oamValue * value
                if bigSingularity:
                    ans[i - 1:i + 2, j - 1:j + 2] = oamValue * value
            else:
                ans[i, j] = nonValue
    return ans


def plane_singularities_finder_4dots(E, circle, value, nonValue, bigSingularity):
    """
    Helper function to find singularities in a 2D plane using 4 dots.

    Parameters:
        E: Complex field.
        circle: Radius of the circle around singularities.
        value: Value assigned to singularities.
        nonValue: Value assigned to non-singularities.
        bigSingularity: Boolean to include big singularities.

    Returns:
        Array with singularities and non-singularities.
    """
    def check_dot_oam_4dots_helper(E):
        def arg(x):
            return np.angle(np.exp(1j * x))

        sum = arg(E[1] - E[0]) + arg(E[2] - E[3]) - arg(E[2] - E[1]) - arg(E[1] - E[0])
        if sum > 3:
            return True, +1
        if sum < -3:
            return True, -1
        return False, 0

    shape = np.shape(E)
    ans = np.zeros(shape)
    for i in range(1, shape[0] - 1, 1):
        for j in range(1, shape[1] - 1, 1):
            Echeck = np.array([E[i, j], E[i, j + 1], E[i + 1, j + 1], E[i + 1, j]])
            oamFlag, oamValue = check_dot_oam_4dots_helper(Echeck)
            if oamFlag:
                ######
                ans[i - circle:i + 1 + circle, j - circle:j + 1 + circle] = nonValue
                #####
                ans[i, j] = oamValue * value
                if bigSingularity:
                    ans[i - 1:i + 2, j - 1:j + 2] = oamValue * value
            else:
                ans[i, j] = nonValue
    return ans


def fill_dict_as_matrix_helper(E, dots=None, nonValue=0, check=False):
    """
    Helper function to fill a dictionary as a matrix.

    Parameters:
        E: Complex field.
        dots: Dictionary to fill.
        nonValue: Value assigned to non-singularities.
        check: Boolean to check existing values.

    Returns:
        Dictionary filled as a matrix.
    """
    if dots is None:
        dots = {}
    shape = np.shape(E)
    if len(shape) == 3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if E[i, j, k] != nonValue:
                        if check:
                            if dots.get((i, j, k)) is None:
                                dots[(i, j, k)] = E[i, j, k]
                        else:
                            dots[(i, j, k)] = E[i, j, k]
    else:
        for i in range(shape[0]):
            for j in range(shape[1]):
                if E[i, j] != nonValue:
                    if check:
                        if dots.get((i, j, 0)) is None:
                            dots[(i, j, 0)] = E[i, j]
                    else:
                        dots[(i, j, 0)] = E[i, j]
    return dots


def cut_non_oam(E, value=1, nonValue=0, bigSingularity=False, axesAll=False, circle=1,
                singularities_finder=plane_singularities_finder_9dots):
    """
    Finds singularities and returns a 3D array with values and non-values.

    Parameters:
        E: Complex field.
        value: Value assigned to singularities.
        nonValue: Value assigned to non-singularities.
        bigSingularity: Boolean to include big singularities.
        axesAll: Boolean to include all axes.
        circle: Radius of the circle around singularities.
        singularities_finder: Function to find singularities (9 dots or 4 dots).

    Returns:
        Tuple of 3D array with values and non-values, and dictionary of dots.
    """
    shape = np.shape(E)
    if len(shape) == 2:
        ans = singularities_finder(E, circle, value, nonValue, bigSingularity)
        ans[:1, :] = nonValue
        ans[-1:, :] = nonValue
        ans[:, :1] = nonValue
        ans[:, -1:] = nonValue
        dots = fill_dict_as_matrix_helper(ans)
    else:
        ans = np.copy(E)
        for i in range(shape[2]):
            ans[:, :, i] = cut_non_oam(ans[:, :, i], value=value, nonValue=nonValue,
                                       bigSingularity=bigSingularity)[0]
        dots = fill_dict_as_matrix_helper(ans)

        if axesAll:
            for i in range(shape[1]):
                ans[:, i, :] += cut_non_oam(E[:, i, :], value=value, nonValue=nonValue,
                                            bigSingularity=bigSingularity)[0]
            dots = fill_dict_as_matrix_helper(ans, dots, check=True)
            for i in range(shape[0]):
                ans[i, :, :] += cut_non_oam(E[i, :, :], value=value, nonValue=nonValue,
                                            bigSingularity=bigSingularity)[0]
            dots = fill_dict_as_matrix_helper(ans, dots, check=True)
    # print(ans)
    return ans, dots


def get_singularities(E, value=1, nonValue=0, bigSingularity=False, axesAll=False, circle=1,
                      singularities_finder=plane_singularities_finder_4dots, returnDict=False):
    """
    Simplifies cut_non_oam by returning an array of singularities.

    Parameters:
        E: Complex field.
        value: Value assigned to singularities.
        nonValue: Value assigned to non-singularities.
        bigSingularity: Boolean to include big singularities.
        axesAll: Boolean to include all axes.
        circle: Radius of the circle around singularities.
        singularities_finder: Function to find singularities (9 dots or 4 dots).
        returnDict: Boolean to return dictionary of dots.

    Returns:
        Array of singularities, and dictionary if returnDict is True.
    """
    if isinstance(E, dict):
        dotsOnly = E
    else:
        dotsFull, dotsOnly = cut_non_oam(E, value, nonValue, bigSingularity, axesAll, circle,
                                         singularities_finder)
    dots = np.array([list(dots) for (dots, OAM) in dotsOnly.items()])
    if returnDict:
        return dotsOnly, dots
    return dots


def W_energy(EArray, xArray=None, yArray=None):
    """
    Calculates the total power in the Oxy plane.

    Parameters:
        EArray: Complex field array.
        xArray: Array of x coordinates.
        yArray: Array of y coordinates.

    Returns:
        Total power in the Oxy plane.
    """
    if xArray is None or yArray is None:
        shape = np.shape(EArray)
        xArray = np.arange(shape[0])
        yArray = np.arange(shape[1])
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    W = np.real(np.sum(np.conj(EArray) * EArray) * dx * dy)
    return W


def Jz_calc_no_conj(EArray, xArray=None, yArray=None):
    """
    Calculates the z-component of the angular momentum without conjugation.

    Parameters:
        EArray: Complex field array.
        xArray: Array of x coordinates.
        yArray: Array of y coordinates.

    Returns:
        Z-component of the angular momentum.
    """
    EArray = np.array(EArray)
    Er, Ei = np.real(EArray), np.imag(EArray)
    if xArray is None or yArray is None:
        shape = np.shape(EArray)
        xArray = np.arange(shape[0])
        yArray = np.arange(shape[1])
    x0 = (xArray[-1] + xArray[0]) / 2
    y0 = (yArray[-1] + yArray[0]) / 2
    x = np.array(xArray) - x0 + 500
    y = np.array(yArray) - y0
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    sumJz = 0
    for i in range(1, len(xArray) - 1, 1):
        for j in range(1, len(yArray) - 1, 1):
            dErx = (Er[i + 1, j] - Er[i - 1, j]) / (2 * dx)
            dEry = (Er[i, j + 1] - Er[i, j - 1]) / (2 * dy)
            dEix = (Ei[i + 1, j] - Ei[i - 1, j]) / (2 * dx)
            dEiy = (Ei[i, j + 1] - Ei[i, j - 1]) / (2 * dy)
            # dErx = (Er[i + 1, j] - Er[i, j]) / (dx)
            # dEry = (Er[i, j + 1] - Er[i, j]) / (dy)
            # dEix = (Ei[i + 1, j] - Ei[i, j]) / (dx * 2)
            # dEiy = (Ei[i, j + 1] - Ei[i, j]) / (dy)
            # print(x[i] * Er[i, j] * dEiy, - y[j] * Er[i, j] * dEix, -
            #           x[i] * Ei[i, j] * dEry, + y[j] * Ei[i, j] * dErx)
            sumJz += (x[i] * Er[i, j] * dEiy - y[j] * Er[i, j] * dEix -
                      x[i] * Ei[i, j] * dEry + y[j] * Ei[i, j] * dErx)
            print(Er[i, j] * dEiy - Ei[i, j] * dEry)
    # Total moment
    Jz = (sumJz * dx * dy)
    W = W_energy(EArray)
    print(f'Total OAM charge = {Jz / W}\tW={W}')
    return Jz


def integral_number2_OAMcoefficients_FengLiPaper(fieldFunction, r, l):
    """
    Calculates the weight of OAM at a radius using FengLi paper's method.

    Parameters:
        fieldFunction: Function of the field.
        r: Radius at which to calculate OAM.
        l: OAM charge.

    Returns:
        Weight of OAM at the specified radius.
    """

    # function helps to get y value from x and r. Sign is used in 2 different parts of the CS.
    # helper => it is used only in other functions, you don't use it yourself
    def y_helper(x, sign, r):
        return sign * np.sqrt(r ** 2 - x ** 2)

    def f1(x):  # function f in the upper half - plane
        Y = y_helper(x, +1, r)
        return fieldFunction(x, Y) * (-1) / Y * np.exp(-1j * l * fg.phi(x, Y)) / np.sqrt(2 * np.pi)

    def f2(x):  # function f in the lower half - plane
        Y = y_helper(x, -1, r)
        return fieldFunction(x, Y) * (-1) / Y * np.exp(-1j * l * fg.phi(x, Y)) / np.sqrt(2 * np.pi)

    i1 = fg.integral_of_function_1D(f1, r, -r)  # upper half - plane integration
    i2 = fg.integral_of_function_1D(f2, -r, r)  # lower half - plane integration
    answer = i1[0] + i2[0]  # [0] - is the integral value, [1:] - errors value and other stuff, we don't need it
    return answer


def integral_number3_OAMpower_FengLiPaper(fieldFunction, rMin, rMax, rResolution, l):
    """
    Calculates the total power in the OAM with a specific charge.

    Parameters:
        fieldFunction: Function of the field.
        rMin: Minimum radius for integration.
        rMax: Maximum radius for integration.
        rResolution: Resolution for the radius.
        l: OAM charge.

    Returns:
        Total power in the OAM with the specified charge.
    """
    rArray = np.linspace(rMin, rMax, rResolution)
    aRArray = np.zeros(rResolution, dtype=complex)
    for i in range(rResolution):
        aRArray[i] = integral_number2_OAMcoefficients_FengLiPaper(fieldFunction, rArray[i], l)
    pL = integrate.simps(np.abs(aRArray) ** 2 * rArray, rArray)  # using interpolation
    return pL


def knot_build_pyknotid(dotsKnotList, **kwargs):
    """
    Builds a normalized pyknotid knot.

    Parameters:
        dotsKnotList: List of dots forming the knot.
        **kwargs: Additional parameters for the knot.

    Returns:
        Pyknotid space curve.
    """
    zMid = (max(z for x, y, z in dotsKnotList) + min(z for x, y, z in dotsKnotList)) / 2
    xMid = (max(x for x, y, z in dotsKnotList) + min(x for x, y, z in dotsKnotList)) / 2
    yMid = (max(y for x, y, z in dotsKnotList) + min(y for x, y, z in dotsKnotList)) / 2
    knotSP = sp.Knot(np.array(dotsKnotList) - [xMid, yMid, zMid], **kwargs)
    return knotSP


def fill_dotsKnotList_mine(dots):
    """
    Fills a list of dots by removing charge sign and arranging them into a list.

    Parameters:
        dots: Array of dot coordinates.

    Returns:
        List of dots arranged into a knot.
    """

    def min_dist(dot, dots):
        elements = [(list(fg.distance_between_points(dot, d)), i) for i, d in enumerate(dots)]
        minEl = min(elements, key=lambda i: i[0])
        return minEl

    dotsKnotList = []
    dotsDict = {}
    for [x, y, z] in dots:

        if not (z in dotsDict):
            dotsDict[z] = []
        dotsDict[z].append([x, y])
    print(dotsDict)
    indZ = next(iter(dotsDict))  # z coordinate
    indInZ = 0  # dot number in XY plane at z
    indexes = np.array([-1, 0, 1])  # which layers we are looking at
    currentDot = dotsDict[indZ].pop(indInZ)
    # distCheck = 20
    while dotsDict:
        # print(indZ, currentDot, dotsDict)
        minList = []  # [min, layer, position in Layer] for all indexes + indZ layers
        for i in indexes + indZ:  # searching the closest element among indexes + indZ
            if not (i in dotsDict):
                continue
            minVal, min1Ind = min_dist(currentDot, dotsDict[i])
            # if minVal <= distCheck:
            minList.append([minVal, i, min1Ind])
        if not minList:
            newPlane = 2
            while not minList:
                for i in [-newPlane, newPlane] + indZ:  # searching the closest element among indexes + indZ
                    if not (i in dotsDict):
                        continue
                    minVal, min1Ind = min_dist(currentDot, dotsDict[i])
                    # if minVal <= distCheck:
                    minList.append([minVal, i, min1Ind])
                newPlane += 1
            if newPlane > 3:
                print(f'we have some dots left. Stopped')
                print(indZ, currentDot, dotsDict)
                break
            print(f'dots are still there, the knot builred cannot use them all\nnew plane: {newPlane}')
        minFin = min(minList, key=lambda i: i[0])
        # if minFin[1] != indZ:
        dotsKnotList.append([*dotsDict[minFin[1]].pop(minFin[2]), minFin[1]])
        currentDot = dotsKnotList[-1][:-1]  # changing the current dot to a new one
        indZ = minFin[1]
        # else:
        #     dotsDict[minFin[1]].pop(minFin[2])
        #     currentDot = self.dotsList[-1][:-1]  # changing the current dot to a new one
        #     indZ = minFin[1]
        # currentDot = self.dotsList[-1][:-1][:]  # changing the current dot to a new one
        # indZ = minFin[1]
        # dotsDict[minFin[1]].pop(minFin[2])
        if not dotsDict[indZ]:  # removing the empty plane (0 dots left)
            del dotsDict[indZ]
    return dotsKnotList


def dots_dens_reduction(dots, checkValue, checkNumber=3):
    """
    Reduces the density of singularity lines by removing extra dots.

    Parameters:
        dots: Array of dot coordinates.
        checkValue: Distance threshold for checking dot density.
        checkNumber: Number of dots to check around each dot.

    Returns:
        Reduced array of dot coordinates.
    """
    dotsFinal = dots
    if checkValue == 0:
        return dotsFinal
    while True:
        distance_matrix = euclidean_distance_matrix(dotsFinal, dotsFinal)
        print(len(dotsFinal))
        for i, line in enumerate(distance_matrix):
            lineSorted = np.sort(line)
            if lineSorted[checkNumber] < checkValue:
                dotsFinal = np.delete(dotsFinal, i, 0)
                break
        else:
            break
    return dotsFinal

