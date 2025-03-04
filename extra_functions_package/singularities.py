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
# import functions_OAM_knots as fOAM
# import functions_general as fg
# import matplotlib.pyplot as plt
import numpy as np
# import functions_general as fg
# import pyknotid.spacecurves as sp
import extra_functions_package.beams_and_pulses as bp
import extra_functions_package.plotings as pl
import extra_functions_package.functions_general as fg
import matplotlib.pyplot as plt
from scipy import integrate
import timeit
import sympy
# from python_tsp.distances import euclidean_distance_matrix
# from python_tsp.heuristics import solve_tsp_local_search
from vispy import app

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


# def dots_filter(dots, checkValue, checkNumber=1):
#     """
#     remove dots which have no close enough neighbors
#     :param dots: dots array [[x, y, z],...]
#     :param checkValue: distance between the dots to check
#     :param checkNumber: how many dots should be close the the dot
#     :return: new dots array [[x, y, z],...]
#     """
#     distance_matrix = euclidean_distance_matrix(dots, dots)
#     minSum = 0
#     indStay = []
#     for i, line in enumerate(distance_matrix):
#         value = line[line > 0].min()
#         minSum += value
#         lineSorted = np.sort(line)
#         if lineSorted[checkNumber] < checkValue:
#             indStay.append(i)
#     print(f"avg_distance: {minSum / np.shape(distance_matrix)[0]}")
#     return dots[indStay]


# def knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                             checkValue2=4, checkNumber2=3,
#                             checkValue3=4, checkNumber3=3, dotsFix=True):
#     """
#     building the sequence from the dots for the knot using TSP problem.
#     :param dots: dots array [[x, y, z],...]
#     :param checkValue1: dots_filter 1
#     :param checkNumber1: dots_filter 1
#     :param checkValue2: dots_filter 2
#     :param checkNumber2: dots_filter 2
#     :param checkValue3: dots_dens_reduction
#     :param checkNumber3: dots_dens_reduction
#     :param dotsFix: if the dots are already good for the knot, it's possible to skip all step before TSP
#     :return: sequential set of dots
#     """
#     if dotsFix:
#         # filtering single separated dots
#         newDots = dots_filter(dots, checkValue=checkValue1, checkNumber=checkNumber1)
#         # filtering 2-4 dots clusters
#         newDots2 = dots_filter(newDots, checkValue=checkValue2, checkNumber=checkNumber2)
#         newDots3 = dots_dens_reduction(newDots2, checkValue=checkValue3, checkNumber=checkNumber3)
#     else:
#         newDots3 = dots
#     distance_matrix = euclidean_distance_matrix(newDots3, newDots3)
#     permutation, distance = solve_tsp_local_search(distance_matrix)
#     dotsKnotList = newDots3[permutation]
#     return dotsKnotList


# def plot_knot_pyknotid(dots, add_closure=True, clf=False, interpolation=None, per=False, **kwargs):
#     """
#     ploting the space curve from the dots-line
#     :param dots: dots array [[x, y, z],...]
#     :param add_closure: closing of the knot
#     :param clf: simplyfication of the knot
#     :param interpolation: how many dots for the interpolation
#     :return: None
#     """
#     import warnings
#     warnings.filterwarnings("ignore")
#     knotPykn = knot_build_pyknotid(dots, add_closure=add_closure)
#     # knotPykn.plot_projection()
#     if interpolation:
#         knotPykn.interpolate(interpolation, quiet=True, per=per)
#     knotPykn = sp.Knot(knotPykn.points[2:])
#     knotPykn.plot(clf=clf, **kwargs)
#     from vispy import app
#     app.run()
#     # plt.plot([0, 0], [0, 0])
#     # plt.show()
#
#
# class Singularities3D:
#     """
#     Work with singularities of any 3D complex field
#     """
#
#     def __init__(self, field3D=None):
#         """
#         :param field3D: any 3D complex field
#         """
#         self.field3D = field3D
#         self.dotsDict = None  # self.dotsXY or self.dotsAll (can switch with self.swap()
#         self.dotsXY = None  # singularities from XY planes
#         self.dotsAll = None  # singularities from XY+XZ+YZ planes
#         self.dotsList = None  # np.array [[x,y,z], [x,y,z], ...] random order
#         self.mesh = None  # np.meshgrid from LG_combination
#         self.coefficients = None  # [Cl1p1, Cl2p2...] from LG_combination
#         self.modes = None  # [(l1,p1), (l2,p2) ...] from LG_combination
#         # self.fill_dotsDict_from_field3D(_dotsXY=True)
#
#     def field_LG_combination(self, mesh, coefficients, modes, **kwargs):
#         """
#         creating the field of any combination of LG beams
#         Sum(Cl1p1 * LG_simple(*mesh, l=l1, p=p1, **kwargs))
#         :param mesh: np.meshgrid
#         :param coefficients: [Cl1p1, Cl2p2...] ...
#         :param modes: [(l1,p1), (l2,p2) ...]
#         """
#         field = 0
#         self.mesh = mesh
#         self.coefficients = coefficients
#         self.modes = modes
#         for num, coefficient in enumerate(coefficients):
#             field += coefficient * bp.LG_simple(*mesh, l=modes[num][0], p=modes[num][1], **kwargs)
#         self.field3D = field
#
#     def plot_plane_2D(self, zPlane, show=True, **kwargs):
#         """
#         Plot any z plane, both abs and angle
#         :param zPlane: number of the plane to plot (0<=z<=shape[2])
#         :return: None
#         """
#         pl.plot_2D(np.abs(self.field3D[:, :, zPlane]), **kwargs)
#         pl.plot_2D(np.angle(self.field3D[:, :, zPlane]), map='hsv', **kwargs)
#         if show:
#             plt.show()
#
#     def plot_center_2D(self, **kwargs):
#         """
#         Plot the center plane (z=0 if from z is from -l to l)
#         :return: None
#         """
#         shape = np.shape(self.field3D)
#         self.plot_plane_2D(shape[2] // 2, **kwargs)
#
#     def plotDots(self, show=True, **kwargs):
#         """
#         Plot self.dots (scatter) using fOAM.plot_knot_dots()
#         if self.dots is not initialized, initialization with self.fill_dotsDict_from_field3D()
#         :param kwargs: Everything for fOAM.plot_knot_dots()
#          also for self.fill_dotsDict_from_field3D()
#         :return: None
#         """
#         if self.dotsDict is None:
#             self.fill_dotsDict_from_field3D(**kwargs)
#         ax = plot_knot_dots(self.dotsDict, **kwargs)
#         if show:
#             plt.show()
#         return ax
#         # fg.distance_between_points()
#
#     def plot_density(self, **kwargs):
#         """
#         Plot density on the browser
#         :kwargs: Everything for fOAM.plot_knot_dots()
#         :return: None
#         """
#         pl.plot_3D_density(np.angle(self.field3D), **kwargs)
#
#     def fill_dotsDict_from_field3D(self, _dotsXY=True, **kwargs):
#         """
#         Filing self.dots with self.dotsXY. for self.dotsALL use parameter _dotsXY
#         :param kwargs: Everything for fg.cut_non_oam()
#         :param _dotsXY: if True, we are filling with self.dotsXY, otherwise with self.dotsALL
#         :return: number of dots in self.dots
#         """
#         if _dotsXY:
#             if self.dotsXY is None:
#                 self.fill_dotsXY(**kwargs)
#             self.dotsDict = self.dotsXY
#         else:
#             if self.dotsAll is None:
#                 self.fill_dotsAll(**kwargs)
#             self.dotsDict = self.dotsAll
#         return len(self.dotsDict)
#
#     def fill_dotsList(self):
#         self.dotsList = np.array([[x, y, z] for (x, y, z) in self.dotsDict])
#
#     def fill_dotsXY(self, **kwargs):
#         """
#         fill in self.dotsXY with using only XY cross-sections for singularities
#         :param kwargs: fg.cut_non_oam besides axesAll
#         :return:
#         """
#         garbage, self.dotsXY = cut_non_oam(np.angle(self.field3D), axesAll=False, **kwargs)
#         self.dotsDict = self.dotsXY
#
#     def fill_dotsAll(self, **kwargs):
#         """
#         fill in self.dotsALL with using ALL 3 cross-sections for singularities
#         :param kwargs: fg.cut_non_oam besides axesAll
#         :return:
#         """
#         garbage, self.dotsAll = cut_non_oam(np.angle(self.field3D), axesAll=True, **kwargs)
#         self.dotsDict = self.dotsAll
#
#     def dots_swap(self, **kwargs):
#         """
#         change self.dots between self.dotsXY and self.dotsAll
#         if self.dots is not either of those -> self.fill_dotsDict_from_field3D()
#         print the new self.dots
#         :return: None
#         """
#         if self.dotsDict is self.dotsXY:
#             if self.dotsAll is None:
#                 self.fill_dotsAll()
#             self.dotsDict = self.dotsAll
#             print(f'Dots are now in all 3 planes')
#         elif self.dotsDict is self.dotsAll:
#             if self.dotsXY is None:
#                 self.fill_dotsXY()
#             self.dotsDict = self.dotsXY
#             print(f'Dots are now in the XY-plane')
#         else:
#             self.fill_dotsDict_from_field3D(*kwargs)
#             print(f'Dots were not dotsXY or dotsAll. Now dots are in the XY-plane')
#
#
# class Knot(Singularities3D):
#     """
#     Knot field (unknots also are knots in that case)
#     """
#
#     def __init__(self, field3D=None):
#         """
#         :param field3D: any 3D complex field
#         """
#         Singularities3D.__init__(self, field3D)
#         self.dotsKnotList = None  # the actual knot (ordered line)
#         self.knotSP = None
#
#     def build_knot_pyknotid(self, **kwargs):
#         """
#         function build normilized pyknotid knot
#         :return:
#         """
#         if self.dotsKnotList is None:
#             self.fill_dotsKnotList()
#         zMid = (max(z for x, y, z in self.dotsKnotList) + min(z for x, y, z in self.dotsKnotList)) / 2
#         xMid = (max(x for x, y, z in self.dotsKnotList) + min(x for x, y, z in self.dotsKnotList)) / 2
#         yMid = (max(y for x, y, z in self.dotsKnotList) + min(y for x, y, z in self.dotsKnotList)) / 2
#         self.knotSP = sp.Knot(np.array(self.dotsKnotList) - [xMid, yMid, zMid], add_closure=False, **kwargs)
#
#     def plot_knot(self, **kwargs):
#         """
#         plot the knot
#         """
#         if self.dotsKnotList is None:
#             self.fill_dotsKnotList()
#         if self.knotSP is None:
#             self.build_knot_pyknotid(**kwargs)
#         plt.plot([1], [1])
#         self.knotSP.plot()
#         plt.show()
#
#     def fill_dotsKnotList(self):
#         if self.dotsList is None:
#             self.fill_dotsList()
#         distance_matrix = euclidean_distance_matrix(self.dotsList, self.dotsList)
#         permutation, distance = solve_tsp_local_search(distance_matrix)
#         # print(dots[permutation])
#         # print(permutation)
#         self.dotsKnotList = self.dotsList[permutation]
#
#     def fill_dotsKnotList_mine(self):
#         """
#         fill in self.dotsList by removing charge sign and placing everything into the list [[x, y, z], [x, y, z]...]
#         :return: None
#         """
#
#         def min_dist(dot, dots):
#             elements = [(fg.distance_between_points(dot, d), i) for i, d in enumerate(dots)]
#             minEl = min(elements, key=lambda i: i[0])
#             return minEl
#
#         self.dotsKnotList = []
#         dotsDict = {}
#         for [x, y, z] in self.dotsDict:
#             if not (z in dotsDict):
#                 dotsDict[z] = []
#             dotsDict[z].append([x, y])
#         indZ = next(iter(dotsDict))  # z coordinate
#         indInZ = 0  # dot number in XY plane at z
#         indexes = np.array([-1, 0, 1])  # which layers we are looking at
#         currentDot = dotsDict[indZ].pop(indInZ)
#         # distCheck = 20
#         while dotsDict:
#             # print(indZ, currentDot, dotsDict)
#             minList = []  # [min, layer, position in Layer] for all indexes + indZ layers
#             for i in indexes + indZ:  # searching the closest element among indexes + indZ
#                 if not (i in dotsDict):
#                     continue
#                 minVal, min1Ind = min_dist(currentDot, dotsDict[i])
#                 # if minVal <= distCheck:
#                 minList.append([minVal, i, min1Ind])
#             if not minList:
#                 newPlane = 2
#                 while not minList:
#                     for i in [-newPlane, newPlane] + indZ:  # searching the closest element among indexes + indZ
#                         if not (i in dotsDict):
#                             continue
#                         minVal, min1Ind = min_dist(currentDot, dotsDict[i])
#                         # if minVal <= distCheck:
#                         minList.append([minVal, i, min1Ind])
#                     newPlane += 1
#                 if newPlane > 3:
#                     print(f'we have some dots left. Stopped')
#                     print(indZ, currentDot, dotsDict)
#                     break
#                 print(f'dots are still there, the knot builred cannot use them all\nnew plane: {newPlane}')
#             minFin = min(minList, key=lambda i: i[0])
#             # if minFin[1] != indZ:
#             self.dotsKnotList.append([*dotsDict[minFin[1]].pop(minFin[2]), minFin[1]])
#             currentDot = self.dotsKnotList[-1][:-1]  # changing the current dot to a new one
#             indZ = minFin[1]
#             # else:
#             #     dotsDict[minFin[1]].pop(minFin[2])
#             #     currentDot = self.dotsList[-1][:-1]  # changing the current dot to a new one
#             #     indZ = minFin[1]
#             # currentDot = self.dotsList[-1][:-1][:]  # changing the current dot to a new one
#             # indZ = minFin[1]
#             # dotsDict[minFin[1]].pop(minFin[2])
#             if not dotsDict[indZ]:  # removing the empty plane (0 dots left)
#                 del dotsDict[indZ]
#
#     def check_knot_alex(self) -> bool:
#         checkVal = None
#         if self.knotSP is None:
#             self.build_knot_pyknotid()
#         t = sympy.symbols("t")
#         self.alexPol = self.knotSP.alexander_polynomial(variable=t)
#         if self.__class__.__name__ == 'Trefoil':
#             checkVal = -t ** 2 + t - 1
#         if checkVal is None:
#             print(f'There is no check value for this type of knots')
#             return False
#         if self.alexPol == checkVal:
#             return True
#         return False
#
#
# if __name__ == '__main__':
#
#     loading_field = False
#     if loading_field:
#         fileName = f'C:\\Users\\Dima\\Box\\Knots Exp\\Experimental Data\\7-13-2022\\Field SR = 0.95\\3foil_turb_25.mat'
#         field_experiment = fg.reading_file_mat(fileName=fileName, fieldToRead='U',
#                                                printV=False)
#         # pl.plot_2D(np.abs(field_experiment))
#         # pl.plot_2D(np.angle(field_experiment))
#
#         fieldAfterProp = fg.one_plane_propagator(field_experiment, dz=10.5, stepsNumber=32, n0=1, k0=1)
#         fieldAfterProp = fg.cut_filter(fieldAfterProp, radiusPix=np.shape(fieldAfterProp)[0] // 4, circle=True)
#         # fieldAfterProp = np.abs(fieldAfterProp) / np.abs(fieldAfterProp).max()
#         dots = get_singularities(np.angle(fieldAfterProp), axesAll=False)
#         # print(np.shape(dots))
#         # exit()
#         pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2], size=100)
#         # pl.plot_3D_density(np.abs(fieldAfterProp)[30:170, 30:170,:], colorscale='jet',
#         #                      opacityscale=[[0, 0], [0.1, 0], [0.2, 0], [0.3, 0.6], [1, 1]], opacity=0.2, surface_count=20,show=True)
#         exit()
#
#
#     def func_time_main():
#         xMinMax = 3
#         yMinMax = 3
#         zMinMax = 0.8
#         zRes = 70
#         xRes = yRes = 70
#         xyzMesh = fg.create_mesh_XYZ(xMinMax, yMinMax, zMinMax, xRes, yRes, zRes, zMin=None)
#         coeff = [1.715, -5.662, 6.381, -2.305, -4.356]
#         phase = [0, 0, 0, 0, 0]
#         coeff = [a * np.exp(1j * p) for a, p in zip(coeff, phase)]
#         beam = bp.LG_combination(*xyzMesh, coefficients=coeff, modes=((0, 0), (0, 1), (0, 2), (0, 3), (3, 0)))
#         # dots = get_singularities(np.angle(beam), bigSingularity=False, axesAll=False)
#         # pl.plot_2D(np.abs(beam[:,:, zRes//2]))
#         # dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#         #                   'Experimental Data\\dots\\trefoil\\Previous\\Field SR = 0.95\\3foil_turb_32.npy',  # 25
#         #                   allow_pickle=True).item()
#         # dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#         #                   'Experimental Data\\dots\\trefoil\\SR=0.9\\3foil_turb_6.npy',  # 25
#         #                   allow_pickle=True).item()
#         # dots = get_singularities(dotsExp)
#
#         pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2], size=100)
#
#         # print(indStay, print(len(indStay)))
#         #
#
#         # pl.plot_scatter_3D(newDots2[:, 0], newDots2[:, 1], newDots2[:, 2], size=100)
#         # pl.plot_scatter_3D(newDots3[:, 0], newDots3[:, 1], newDots3[:, 2], size=100)
#         # newDots3 = fill_dotsKnotList_mine(newDots2)
#         dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                            checkValue2=4, checkNumber2=3,
#                                            checkValue3=3, checkNumber3=3)
#         pl.plot_scatter_3D(dotsKnot[:, 0], dotsKnot[:, 1], dotsKnot[:, 2], size=100)
#         plot_knot_pyknotid(dotsKnot, interpolation=300, tube_radius=2.5, per=True, add_closure=False,
#                            tube_points=14, fov=0, flip=(False, False, False))
#         # plot_knot_pyknotid(dotsKnot, interpolation=250, tube)
#         # pl.plot_scatter_3D(dotsKnotList[:,0],dotsKnotList[:,1],dotsKnotList[:,2])
#         exit()
#
#
#     hopf_ploting = False
#     if hopf_ploting:
#         dotsExp1 = np.load('C:\\WORK\\CODES\\OAM_research\\hopf_exp_l2.npy',
#                            allow_pickle=True)
#         # plot_knot_pyknotid(dotsExp1, interpolation=300, tube_radius=2.5, per=True, add_closure=False,
#         #                    tube_points=14, fov=0, flip=(False, False, False))
#         dotsExp2 = np.load('C:\\WORK\\CODES\\OAM_research\\hopf_exp_l1.npy',
#                            allow_pickle=True)
#         dots = np.concatenate((dotsExp1, dotsExp2), axis=0)
#         # pl.plot_scatter_3D(dots[:, 0], dots[:, 1],dots[:, 2])
#         # plot_knot_pyknotid(dotsExp2, interpolation=300, tube_radius=2.5, per=True, add_closure=False,
#         #                    tube_points=14, fov=0, flip=(False, False, False))
#         loop1 = sp.Knot(dotsExp1, add_closure=False)
#         loop1.interpolate(250, quiet=True, per=False)
#         dotsExp1Smooth = loop1.points[2: -1]
#         loop2 = sp.Knot(dotsExp2, add_closure=False)
#         loop2.interpolate(250, quiet=True, per=False)
#         dotsExp2Smooth = loop2.points[2: -1]
#         # for i in range(len(dotsExp1Smooth) - 1):
#         #     if (fg.distance_between_points(dotsExp1Smooth[i], dotsExp1Smooth[i+1]) > 2):
#         #         print(i, fg.distance_between_points(dotsExp1Smooth[i], dotsExp1Smooth[i+1]))
#         # exit()
#         link = sp.Link([dotsExp1Smooth, dotsExp2Smooth])
#         link.plot(tube_radius=2., color='black')
#         app.run()
#     trefoil_ploting = False
#     if trefoil_ploting:
#         dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#                           'Experimental Data\\dots\\trefoil\\Previous\\Field SR = 0.95\\3foil_turb_25.npy',  # 25
#                           allow_pickle=True).item()
#         dots = get_singularities(dotsExp)
#         dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                            checkValue2=4, checkNumber2=3,
#                                            checkValue3=3, checkNumber3=3)
#         # trefoil = sp.Knot(dotsKnot, add_closure=False)
#         trefoil = knot_build_pyknotid(dotsKnot)
#         trefoil.interpolate(250, quiet=True, per=False)
#         trefoilSmooth = trefoil.points[2: -1]
#         trefoil = sp.Knot(trefoilSmooth, add_closure=False)
#         trefoil.plot(tube_radius=2.)
#         app.run()
#
#     unknot_ploting = False
#     if unknot_ploting:
#         dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#                           'Experimental Data\\dots\\trefoil\\Previous\\SR=0.9\\3foil_turb_6.npy',  # 25
#                           allow_pickle=True).item()
#         dots = get_singularities(dotsExp)
#         # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2], size=100)
#         # exit()
#         dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                            checkValue2=4, checkNumber2=3,
#                                            checkValue3=3, checkNumber3=3)
#         # trefoil = sp.Knot(dotsKnot, add_closure=False)
#         trefoil = knot_build_pyknotid(dotsKnot)
#         trefoil.interpolate(250, quiet=True, per=False)
#         trefoilSmooth = trefoil.points[2: -1]
#         trefoil = sp.Knot(trefoilSmooth, add_closure=False)
#         cmapMy = [[1, 1, 1, 0], [1, 1, 1, 1]]
#         trefoil.plot(tube_radius=2.)
#         app.run()
#     plot_table = True
#     if plot_table:
#         tr1 = False
#         if tr1:
#             dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#                               'Experimental Data\\dots\\trefoil\\Previous\\Fields SR = 0.85 (2)\\3foil_turb_1.npy',
#                               # 25
#                               allow_pickle=True).item()
#             dots = get_singularities(dotsExp)
#             dots = np.array([dot for dot in dots if dot[0] < 140 and dot[0] > 70])
#             dots = np.array([dot for dot in dots if dot[1] < 140 and dot[1] > 70])
#             # dots = np.array([dot for dot in dots if dot[2] < 140 and dot[2] > 0])
#             pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
#
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2], size=100)
#             # exit()
#             dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                                checkValue2=4, checkNumber2=3,
#                                                checkValue3=3, checkNumber3=3)
#             # trefoil = sp.Knot(dotsKnot, add_closure=False)
#             trefoil = knot_build_pyknotid(dotsKnot)
#             trefoil.interpolate(250, quiet=True, per=False)
#             trefoilSmooth = trefoil.points[2: -1]
#             trefoil = sp.Knot(trefoilSmooth, add_closure=False)
#             cmapMy = [[1, 1, 1, 0], [1, 1, 1, 1]]
#             trefoil.plot(tube_radius=2.)
#             app.run()
#         tr2 = False
#         if tr2:
#             dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#                               'Experimental Data\\dots\\trefoil\\Previous\\Fields SR = 0.85 (2)\\3foil_turb_16.npy',
#                               # 25
#                               allow_pickle=True).item()
#             dots = get_singularities(dotsExp)
#             dots = np.array([dot for dot in dots if dot[0] < 240 and dot[0] > 87])
#             dots = np.array([dot for dot in dots if dot[1] < 240 and dot[1] > 80])
#             # dots = np.array([dot for dot in dots if dot[2] < 240 and dot[2] > 0])
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
#             # exit()
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2], size=100)
#             # exit()
#             dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                                checkValue2=4, checkNumber2=3,
#                                                checkValue3=3, checkNumber3=3)
#             # trefoil = sp.Knot(dotsKnot, add_closure=False)
#             trefoil = knot_build_pyknotid(dotsKnot)
#             trefoil.interpolate(250, quiet=True, per=False)
#             trefoilSmooth = trefoil.points[2: -1]
#             trefoil = sp.Knot(trefoilSmooth, add_closure=False)
#             cmapMy = [[1, 1, 1, 0], [1, 1, 1, 1]]
#             trefoil.plot(tube_radius=2.)
#             app.run()
#         tr3 = False
#         if tr3:
#             dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#                               'Experimental Data\\dots\\trefoil\\Previous\\Fields SR = 0.85 (2)\\3foil_turb_33.npy',
#                               # 25
#                               allow_pickle=True).item()
#             dots = get_singularities(dotsExp)
#             dots = np.array([dot for dot in dots if dot[0] < 240 and dot[0] > 90])
#             dots = np.array([dot for dot in dots if dot[1] < 145 and dot[1] > 70])
#             dots = np.array([dot for dot in dots if not (dot[0] > 140 and dot[2] > 45)])
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
#             # exit()
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2], size=100)
#             # exit()
#             dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                                checkValue2=4, checkNumber2=3,
#                                                checkValue3=3, checkNumber3=3)
#             # trefoil = sp.Knot(dotsKnot, add_closure=False)
#             trefoil = knot_build_pyknotid(dotsKnot)
#             trefoil.interpolate(250, quiet=True, per=False)
#             trefoilSmooth = trefoil.points[2: -1]
#             trefoil = sp.Knot(trefoilSmooth, add_closure=False)
#             cmapMy = [[1, 1, 1, 0], [1, 1, 1, 1]]
#             trefoil.plot(tube_radius=2.)
#             app.run()
#         un1 = False
#         if un1:
#             dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#                               'Experimental Data\\dots\\trefoil\\Previous\\Fields SR = 0.85 (2)\\3foil_turb_91.npy',
#                               # 25
#                               allow_pickle=True).item()
#             dots = get_singularities(dotsExp)
#             dots = np.array([dot for dot in dots if dot[0] < 240 and dot[0] > 85])
#             dots = np.array([dot for dot in dots if dot[1] < 245 and dot[1] > 80])
#             dots = np.array([dot for dot in dots if dot[2] < 58 and dot[2] > 0])
#             dots = np.array([dot for dot in dots if not (dot[0] > 95 and dot[0] < 110 and dot[1] > 100 and dot[1] < 115
#                                                          and dot[2] > 45)])
#             # dots = np.array([dot for dot in dots if not (dot[0] > 140 and dot[2] > 45)])
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
#             # exit()
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2], size=100)
#             # exit()
#             dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                                checkValue2=4, checkNumber2=3,
#                                                checkValue3=3, checkNumber3=3)
#             # trefoil = sp.Knot(dotsKnot, add_closure=False)
#             trefoil = knot_build_pyknotid(dotsKnot)
#             trefoil.interpolate(250, quiet=True, per=False)
#             trefoilSmooth = trefoil.points[2: -1]
#             trefoil = sp.Knot(trefoilSmooth, add_closure=False)
#             cmapMy = [[1, 1, 1, 0], [1, 1, 1, 1]]
#             trefoil.plot(tube_radius=2.)
#             app.run()
#         un2 = False
#         if un2:
#             dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#                               'Experimental Data\\dots\\trefoil\\Previous\\Fields SR = 0.85 (2)\\3foil_turb_90.npy',
#                               # 25
#                               allow_pickle=True).item()
#             dots = get_singularities(dotsExp)
#             dots = np.array([dot for dot in dots if dot[0] < 150 and dot[0] > 80])
#             dots = np.array([dot for dot in dots if dot[1] < 140 and dot[1] > 70])
#             dots = np.array([dot for dot in dots if dot[2] < 200 and dot[2] > 9])
#             # dots = np.array([dot for dot in dots if not (dot[0] > 95 and dot[0] < 110 and dot[1] > 100 and dot[1] < 115
#             #                                              and dot[2] > 45)])
#             # dots = np.array([dot for dot in dots if not (dot[0] > 140 and dot[2] > 45)])
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
#             # exit()
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2], size=100)
#             # exit()
#             dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                                checkValue2=4, checkNumber2=3,
#                                                checkValue3=3, checkNumber3=3)
#             # trefoil = sp.Knot(dotsKnot, add_closure=False)
#             trefoil = knot_build_pyknotid(dotsKnot)
#             trefoil.interpolate(250, quiet=True, per=False)
#             trefoilSmooth = trefoil.points[2: -1]
#             trefoil = sp.Knot(trefoilSmooth, add_closure=False)
#             cmapMy = [[1, 1, 1, 0], [1, 1, 1, 1]]
#             trefoil.plot(tube_radius=2.)
#             app.run()
#         un3 = False
#         if un3:
#             dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#                               'Experimental Data\\dots\\trefoil\\Previous\\Fields SR = 0.85 (2)\\3foil_turb_88.npy',
#                               # 25
#                               allow_pickle=True).item()
#             dots = get_singularities(dotsExp)
#             dots = np.array([dot for dot in dots if dot[0] < 150 and dot[0] > 80])
#             dots = np.array([dot for dot in dots if dot[1] < 140 and dot[1] > 70])
#             dots = np.array([dot for dot in dots if dot[2] < 52 and dot[2] > 0])
#             dots = np.array([dot for dot in dots if not (dot[0] > 140 and dot[2] > 48)])
#             # dots = np.array([dot for dot in dots if not (dot[0] > 140 and dot[2] > 45)])
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
#             # exit()
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2], size=100)
#             # exit()
#             dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                                checkValue2=4, checkNumber2=3,
#                                                checkValue3=3, checkNumber3=3)
#             # trefoil = sp.Knot(dotsKnot, add_closure=False)
#             trefoil = knot_build_pyknotid(dotsKnot)
#             trefoil.interpolate(250, quiet=True, per=False)
#             trefoilSmooth = trefoil.points[2: -1]
#             trefoil = sp.Knot(trefoilSmooth, add_closure=False)
#             cmapMy = [[1, 1, 1, 0], [1, 1, 1, 1]]
#             trefoil.plot(tube_radius=2.)
#             app.run()
#         h1 = False
#         if h1:
#             dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#                               'Experimental Data\\dots\\trefoil\\Previous\\Fields SR = 0.85 (2)\\3foil_turb_95.npy',
#                               # 25
#                               allow_pickle=True).item()
#             dots = get_singularities(dotsExp)
#             dots = np.array([dot for dot in dots if dot[0] < 150 and dot[0] > 80])
#             dots = np.array([dot for dot in dots if dot[1] < 140 and dot[1] > 70])
#             dots = np.array([dot for dot in dots if dot[2] < 200 and dot[2] > 9])
#             # dots = np.array([dot for dot in dots if not (dot[0] > 95 and dot[0] < 110 and dot[1] > 100 and dot[1] < 115
#             #                                              and dot[2] > 45)])
#             # dots = np.array([dot for dot in dots if not (dot[0] > 140 and dot[2] > 45)])
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
#             # exit()
#
#             # dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#             #                                    checkValue2=4, checkNumber2=3,
#             #                                    checkValue3=3, checkNumber3=3)
#             # np.save('h1', dotsKnot)
#             dotsKnot = np.load('h1.npy')
#             for i in range(len(dotsKnot) - 1):
#                 print(i, fg.distance_between_points(dotsKnot[i], dotsKnot[i + 1]))
#             dot1 = 127
#             dot2 = 221
#             dotsKnot = np.roll(dotsKnot, -dot1 - 1, axis=0)
#             dotsExp1 = dotsKnot[0:dot2 - dot1 - 1, :]
#             # pl.plot_scatter_3D(dotsExp1[:, 0], dotsExp1[:, 1], dotsExp1[:, 2])
#             dotsExp2 = dotsKnot[dot2 - dot1:, :]
#             # pl.plot_scatter_3D(dotsExp2[:, 0], dotsExp2[:, 1], dotsExp2[:, 2])
#             loop1 = sp.Knot(dotsExp1, add_closure=False)
#             loop1.interpolate(250, quiet=True, per=False)
#             dotsExp1Smooth = loop1.points[2: -1]
#             loop2 = sp.Knot(dotsExp2, add_closure=False)
#             loop2.interpolate(250, quiet=True, per=False)
#             dotsExp2Smooth = loop2.points[2: -1]
#             # for i in range(len(dotsExp1Smooth) - 1):
#             #     if (fg.distance_between_points(dotsExp1Smooth[i], dotsExp1Smooth[i+1]) > 2):
#             #         print(i, fg.distance_between_points(dotsExp1Smooth[i], dotsExp1Smooth[i+1]))
#             # exit()
#             link = sp.Link([dotsExp1Smooth, dotsExp2Smooth])
#             link.plot(tube_radius=2., color='black')
#             app.run()
#             exit()
#             # trefoil = sp.Knot(dotsKnot, add_closure=False)
#             trefoil = knot_build_pyknotid(dotsKnot)
#             trefoil.interpolate(250, quiet=True, per=False)
#             trefoilSmooth = trefoil.points[2: -1]
#             trefoil = sp.Knot(trefoilSmooth, add_closure=False)
#             cmapMy = [[1, 1, 1, 0], [1, 1, 1, 1]]
#             trefoil.plot(tube_radius=2.)
#             app.run()
#         h2 = False
#         if h2:
#             dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#                               'Experimental Data\\dots\\trefoil\\Previous\\Fields SR = 0.85 (2)\\3foil_turb_11.npy',
#                               # 25
#                               allow_pickle=True).item()
#             dots = get_singularities(dotsExp)
#             dots = np.array([dot for dot in dots if dot[0] < 140 and dot[0] > 0])
#             dots = np.array([dot for dot in dots if dot[1] < 145 and dot[1] > 60])
#             dots = np.array([dot for dot in dots if dot[2] < 55 and dot[2] > 0])
#             dots = np.array([dot for dot in dots if not (dot[0] > 130
#                                                          and dot[2] > 45)])
#
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
#             # exit()
#             if 0:
#                 dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                                    checkValue2=4, checkNumber2=3,
#                                                    checkValue3=3, checkNumber3=3)
#                 np.save('h2', dotsKnot)
#                 trefoil = knot_build_pyknotid(dotsKnot)
#                 trefoil.interpolate(250, quiet=True, per=False)
#                 trefoilSmooth = trefoil.points[2: -1]
#                 trefoil = sp.Knot(trefoilSmooth, add_closure=False)
#                 cmapMy = [[1, 1, 1, 0], [1, 1, 1, 1]]
#                 trefoil.plot(tube_radius=2.)
#                 app.run()
#                 exit()
#             dotsKnot = np.load('../h2.npy')
#             for i in range(len(dotsKnot) - 1):
#                 if fg.distance_between_points(dotsKnot[i], dotsKnot[i + 1]) > 5:
#                     print(i, fg.distance_between_points(dotsKnot[i], dotsKnot[i + 1]))
#             dot1 = 45
#             dot2 = 166
#             dotsKnot = np.roll(dotsKnot, -dot1 - 1, axis=0)
#             dotsExp1 = dotsKnot[0:dot2 - dot1 - 1, :]
#             # pl.plot_scatter_3D(dotsExp1[:, 0], dotsExp1[:, 1], dotsExp1[:, 2])
#             dotsExp2 = dotsKnot[dot2 - dot1:, :]
#             # pl.plot_scatter_3D(dotsExp2[:, 0], dotsExp2[:, 1], dotsExp2[:, 2])
#             # exit()
#             loop1 = sp.Knot(dotsExp1, add_closure=False)
#             loop1.interpolate(250, quiet=True, per=False)
#             dotsExp1Smooth = loop1.points[2: -1]
#             loop2 = sp.Knot(dotsExp2, add_closure=False)
#             loop2.interpolate(250, quiet=True, per=False)
#             dotsExp2Smooth = loop2.points[2: -1]
#             # for i in range(len(dotsExp1Smooth) - 1):
#             #     if (fg.distance_between_points(dotsExp1Smooth[i], dotsExp1Smooth[i+1]) > 2):
#             #         print(i, fg.distance_between_points(dotsExp1Smooth[i], dotsExp1Smooth[i+1]))
#             # exit()
#             link = sp.Link([dotsExp1Smooth, dotsExp2Smooth])
#             link.plot(tube_radius=2., color='black')
#             app.run()
#             exit()
#             # trefoil = sp.Knot(dotsKnot, add_closure=False)
#         h3 = True
#         if h3:
#             dotsExp = np.load('C:\\Users\\Dima\\Box\\Knots Exp\\'
#                               'Experimental Data\\dots\\trefoil\\Previous\\Fields SR = 0.85 (2)\\3foil_turb_66.npy',
#                               # 25
#                               allow_pickle=True).item()
#             dots = get_singularities(dotsExp)
#             dots = np.array([dot for dot in dots if dot[0] < 150 and dot[0] > 70])
#             dots = np.array([dot for dot in dots if dot[1] < 160 and dot[1] > 65])
#             # dots = np.array([dot for dot in dots if dot[2] < 55 and dot[2] > 0])
#             dots = np.array([dot for dot in dots if not (dot[0] > 130
#                                                          and dot[2] > 50)])
#
#             # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
#             # exit()
#             if 0:
#                 dotsKnot = knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                                    checkValue2=4, checkNumber2=3,
#                                                    checkValue3=3, checkNumber3=3)
#                 np.save('h3', dotsKnot)
#                 trefoil = knot_build_pyknotid(dotsKnot)
#                 trefoil.interpolate(250, quiet=True, per=False)
#                 trefoilSmooth = trefoil.points[2: -1]
#                 trefoil = sp.Knot(trefoilSmooth, add_closure=False)
#                 cmapMy = [[1, 1, 1, 0], [1, 1, 1, 1]]
#                 trefoil.plot(tube_radius=2.)
#                 app.run()
#                 exit()
#             dotsKnot = np.load('../h3.npy')
#             for i in range(len(dotsKnot) - 1):
#                 if fg.distance_between_points(dotsKnot[i], dotsKnot[i + 1]) > 5:
#                     print(i, fg.distance_between_points(dotsKnot[i], dotsKnot[i + 1]))
#             # exit()
#             dot1 = 125
#             dot2 = 212
#             dotsKnot = np.roll(dotsKnot, -dot1 - 1, axis=0)
#             dotsExp1 = dotsKnot[0:dot2 - dot1 - 1, :]
#             # pl.plot_scatter_3D(dotsExp1[:, 0], dotsExp1[:, 1], dotsExp1[:, 2])
#             dotsExp2 = dotsKnot[dot2 - dot1:, :]
#             # pl.plot_scatter_3D(dotsExp2[:, 0], dotsExp2[:, 1], dotsExp2[:, 2])
#             # exit()
#             loop1 = sp.Knot(dotsExp1, add_closure=False)
#             loop1.interpolate(250, quiet=True, per=False)
#             dotsExp1Smooth = loop1.points[2: -1]
#             loop2 = sp.Knot(dotsExp2, add_closure=False)
#             loop2.interpolate(250, quiet=True, per=False)
#             dotsExp2Smooth = loop2.points[2: -1]
#             # for i in range(len(dotsExp1Smooth) - 1):
#             #     if (fg.distance_between_points(dotsExp1Smooth[i], dotsExp1Smooth[i+1]) > 2):
#             #         print(i, fg.distance_between_points(dotsExp1Smooth[i], dotsExp1Smooth[i+1]))
#             # exit()
#             link = sp.Link([dotsExp1Smooth, dotsExp2Smooth])
#             link.plot(tube_radius=2., color='black')
#             app.run()
#             exit()
#             # trefoil = sp.Knot(dotsKnot, add_closure=False)
#         if None:
#             if plot_hopf:
#                 # dotsExp = np.load('C:\\Users\\Cmex-\\Box\\Knots Exp\\Experimental Data\\dots\\trefoil\\Previous\\'
#                 #                   '3foil SR = 0.95 (2)\\3foil_turb_3.npy',  # 25
#                 #                   allow_pickle=True).item()
#                 # dots = sing.get_singularities(dotsExp)
#                 # # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
#                 # dotsKnot = sing.knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                 #                                         checkValue2=4, checkNumber2=3,
#                 #                                         checkValue3=3, checkNumber3=3)[::-1]
#                 # np.save('hopf_exp', dotsKnot)
#                 dotsKnot = np.load('hopf_exp.npy')
#                 # pl.plot_scatter_3D(dotsKnot[:, 0], dotsKnot[:, 1], dotsKnot[:, 2])
#                 import matplotlib.pyplot as plt
#
#                 ax = plt.axes(projection='3d')
#                 ax.plot3D(dotsKnot[:, 0], dotsKnot[:, 1], dotsKnot[:, 2], 'gray')
#                 # pl.plot_3D_dots_go(dotsKnot, show=True)
#                 # for i in range(len(dotsKnot)-1):
#                 #     print(i, fg.distance_between_points(dotsKnot[i], dotsKnot[i+1]))
#                 dotsKnot = np.roll(dotsKnot, -38, axis=0)
#                 ax.plot3D(dotsKnot[0:61, 0], dotsKnot[0:61, 1], dotsKnot[0:61, 2])
#                 np.save('hopf_exp_l1', dotsKnot[0:61])
#                 np.save('hopf_exp_l2', dotsKnot[61:])
#                 exit()
#                 ax.plot3D(dotsKnot[61:, 0], dotsKnot[61:, 1], dotsKnot[61:, 2], 'green')
#                 plt.show()
#                 exit()
#                 modesTrefoil = [(0, 0), (0, 1), (0, 2), (2, 0)]
#                 # coeffTrefoil = [2.96, -6.23, 4.75, 5.49]
#                 coeffTrefoil = [2.63, -6.32, 4.21, 5.95]
#                 xMinMax = 2.5
#                 yMinMax = 2.5
#                 zMinMax = 0.7
#                 zRes = 90
#                 xRes = yRes = 90
#                 xyzMesh = fg.create_mesh_XYZ(xMinMax, yMinMax, zMinMax, xRes, yRes, zRes, zMin=None)
#                 beam = bp.LG_combination(*xyzMesh, coefficients=coeffTrefoil, modes=modesTrefoil)
#                 # sing.plot_knot_dots(beam, show=True, axesAll=False, color='k')
#                 dots = sing.get_singularities(np.angle(beam), bigSingularity=False, axesAll=True)
#                 # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2], size=100)
#                 # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
#                 # pl.plot_3D_dots_go(dots, show=True)
#                 # pl.plot_2D((np.abs(beam[:, :, zRes//2])))
#                 dotsKnot = sing.knot_sequence_from_dots(dots, checkValue1=2, checkNumber1=1,
#                                                         checkValue2=4, checkNumber2=3,
#                                                         checkValue3=3, checkNumber3=2)[::-1]
#                 # pl.plot_3D_dots_go(dotsKnot, show=True)
#
#                 for i in range(len(dotsKnot) - 1):
#                     print(i, fg.distance_between_points(dotsKnot[i], dotsKnot[i + 1]))
#                 ax.plot3D(dotsKnot[57:114, 0], dotsKnot[57:114, 1], dotsKnot[57:114, 2])
#                 plt.show()
#                 exit()
#                 # dotsKnot = np.roll(dotsKnot, -1 * find_closet_to_point_dot(dotsKnot, [0, 0, 0]) + 40, axis=0)
#                 # pl.plot_scatter_3D(dotsKnot[:, 0], dotsKnot[:, 1], dotsKnot[:, 2], size=100)
#                 # sing.plot_knot_pyknotid(dotsKnot, interpolation=300, tube_radius=2.5, per=False, add_closure=False,
#                 #                         tube_points=14, fov=0, flip=(False, False, True))
#                 # , antialias=True,
#                 #                                 light_dir=(180,90,-50)
#                 # dots = fg.dots3D_rescale(dots, xyzMesh)
#                 exit()
#     # timeit.timeit(func_time_main, number=1)
