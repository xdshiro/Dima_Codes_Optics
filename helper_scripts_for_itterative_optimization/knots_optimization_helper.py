import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# mine
import helper_scripts_for_itterative_optimization.functions_general as fg
import helper_scripts_for_itterative_optimization.functions_OAM_knots as fOAM

import winsound

def check_knot_mine(
    xyzMesh,
    coeff,
    deltaCoeff,
    steps=1000,
    width=1.3,
    six_dots=True,
    checkboundaries=False,
    boundaryValue=0.2,
    circletest=True,
    radiustest=0.05,
    testvisual=False,
    xyzMeshPlot=None
):
    """
    Optimize trefoil‐knot coefficients by random perturbation to maximize the minimal
    distance between singularity dots, with optional circle, boundary, and visual tests.

    Parameters
    ----------
    xyzMesh : tuple of arrays
        Coordinate grids for the trefoil field evaluation (x, y, z).
    coeff : list of float
        Initial coefficients for the trefoil_mod function (aCoeff).
    deltaCoeff : list of float
        Step sizes for random perturbations of coeff.
    steps : int, optional
        Number of optimization iterations (default: 1000).
    width : float, optional
        Beam width parameter passed to trefoil_mod (default: 1.3).
    six_dots : bool, optional
        If True, require exactly six singularity dots in distance calculation.
    checkboundaries : bool, optional
        If True, perform an empty‐space check on `dotsOnly` before accepting.
    boundaryValue : float, optional
        Threshold for empty_space_check (default: 0.2).
    circletest : bool, optional
        If True, require that a central‐slice circle survives circle_test.
    radiustest : float, optional
        Radius parameter for circle_test (default: 0.05).
    testvisual : bool, optional
        If True, run a final visual() test before accepting new coeff.
    xyzMeshPlot : tuple of arrays, optional
        Mesh used for visual test (must be provided if testvisual is True).

    Returns
    -------
    list of float
        The best‐found coefficient list maximizing the minimal dot distance.
    """
    # --- INITIAL FIELD & DOT‐EXTRACTION ---
    field = fOAM.trefoil_mod(
        *xyzMesh,
        w=1.2,
        width=width,
        k0=1,
        z0=0.,
        aCoeff=coeff,
        coeffPrint=False
    )
    # extract singularity dots
    dotsOnly = fg.cut_non_oam(np.angle(field), axesAll=False)[1]
    # mesh z‐resolution for distance tests
    zRes = np.shape(xyzMesh)[2]
    # baseline minimal dot distance
    minDistance = min_distance(dotsOnly, zRes, six_dots=six_dots)

    # --- OPTIMIZATION LOOP ---
    for i in range(steps):
        # propose new coefficients
        newCoeff = fg.random_list(coeff, deltaCoeff)
        # compute new field & dots
        newField = fOAM.trefoil_mod(
            *xyzMesh,
            w=1.2,
            width=width,
            k0=1,
            z0=0.,
            aCoeff=newCoeff,
            coeffPrint=False
        )
        dotsOnly = fg.cut_non_oam(np.angle(newField), axesAll=False)[1]

        # 1) Optional circle test on the mid‐slice
        if circletest:
            mid = newField.shape[2] // 2
            if not circle_test(
                np.angle(newField)[:, :, mid],
                radius=radiustest,
                testValue=2.5
            ):
                print('circle')  # fails circle shape requirement
                continue

        # 2) Optional boundary‐empty‐space test
        if checkboundaries:
            if not empty_space_check(dotsOnly, zRes, boundaryValue):
                print('boundaries')  # fails boundary spacing
                continue

        # 3) Compute new minimal distance
        minDistanceNew = min_distance(dotsOnly, zRes, six_dots=six_dots)
        print(
            f"Iter {i:4d} | "
            f"old_dist = {minDistance:6.2f} → "
            f"new_dist = {minDistanceNew:6.2f} | "
            f"coeffs  = {newCoeff}"
        )

        # 4) Accept only if improvement found
        if minDistanceNew > minDistance:
            # 4a) Optional visual sanity check
            if testvisual:
                print('test visual')
                if not visual(dotsOnly, coeff, xyzMeshPlot):
                    print('visualy no')  # fails visual inspection
                    continue

            # 4b) Commit improvement
            ratio = minDistance / minDistanceNew
            formatted = [float(f'{a:.2f}') for a in newCoeff]
            print(f'{ratio: .3f}', formatted)
            minDistance = minDistanceNew
            coeff = newCoeff

    return coeff

def check_knot_mine_2(
    xyzMesh,
    coeff,
    deltaCoeff,
    steps=1000,
    width=1.3,
    six_dots=True,
    checkboundaries=False,
    boundaryValue=0.2,
    circletest=True,
    radiustest=0.05,
    testvisual=False,
    xyzMeshPlot=None
):
    """
    Optimize trefoil‐knot (version 2) coefficients by random perturbation to maximize
    the minimal distance between singularity dots, with optional circle, boundary,
    and visual tests.

    Parameters
    ----------
    xyzMesh : tuple of arrays
        Coordinate grids for the trefoil_mod_2 evaluation (x, y, z).
    coeff : list of float
        Initial coefficients for the trefoil_mod_2 function (aCoeff).
    deltaCoeff : list of float
        Step sizes for random perturbations of coeff.
    steps : int, optional
        Number of optimization iterations (default: 1000).
    width : float, optional
        Beam width parameter passed to trefoil_mod_2 (default: 1.3).
    six_dots : bool, optional
        If True, require exactly six singularity dots in distance calculation.
    checkboundaries : bool, optional
        If True, perform an empty‐space check on `dotsOnly` before accepting.
    boundaryValue : float, optional
        Threshold for empty_space_check (default: 0.2).
    circletest : bool, optional
        If True, require that a central‐slice circle survives circle_test.
    radiustest : float, optional
        Radius parameter for circle_test (default: 0.05).
    testvisual : bool, optional
        If True, run a final visual() test before accepting new coeff.
    xyzMeshPlot : tuple of arrays, optional
        Mesh used for visual test (must be provided if testvisual is True).

    Returns
    -------
    list of float
        The best‐found coefficient list maximizing the minimal dot distance.
    """
    # --- INITIAL FIELD & DOT EXTRACTION ---
    field = fOAM.trefoil_mod_2(
        *xyzMesh,
        w=1.2,
        width=width,
        k0=1,
        z0=0.,
        aCoeff=coeff,
        coeffPrint=False
    )
    # extract singularity‐dot positions from the phase
    dotsOnly = fg.cut_non_oam(np.angle(field), axesAll=False)[1]
    # mesh resolutions (only zRes is used in distance checks)
    xRes, yRes, zRes = np.shape(xyzMesh)[1:]
    # baseline minimal distance between dots
    minDistance = min_distance(dotsOnly, zRes, six_dots=six_dots)

    # --- OPTIMIZATION LOOP ---
    for i in range(steps):
        # propose new coefficients by random perturbation
        newCoeff = fg.random_list(coeff, deltaCoeff)

        # compute new field & extract its dots
        newField = fOAM.trefoil_mod_2(
            *xyzMesh,
            w=1.2,
            width=width,
            k0=1,
            z0=0.,
            aCoeff=newCoeff,
            coeffPrint=False
        )
        dotsOnly = fg.cut_non_oam(np.angle(newField), axesAll=False)[1]

        # 1) Optional circle‐shape test at the mid‐slice
        if circletest:
            mid = newField.shape[2] // 2
            if not circle_test(
                np.angle(newField)[:, :, mid],
                radius=radiustest,
                testValue=2.5
            ):
                print('circle')  # failed circle requirement
                continue

        # 2) Optional boundary‐empty‐space test
        if checkboundaries:
            if not empty_space_check(dotsOnly, zRes, boundaryValue):
                print('boundaries')  # failed boundary spacing
                continue

        # 3) Compute new minimal dot distance
        minDistanceNew = min_distance(dotsOnly, zRes, six_dots=six_dots)
        print(
            f"Iter {i:4d} | "
            f"old_dist = {minDistance:6.2f} → "
            f"new_dist = {minDistanceNew:6.2f} | "
            f"coeffs  = {newCoeff}"
        )

        # 4) Accept improvement only
        if minDistanceNew > minDistance:
            # 4a) Optional visual sanity check
            if testvisual:
                print('test visual')
                if not visual(dotsOnly, coeff, xyzMeshPlot):
                    print('visualy no')  # failed visual inspection
                    continue

            # 4b) Commit to the new best coefficients
            ratio = minDistance / minDistanceNew
            formatted = [float(f'{a:.2f}') for a in newCoeff]
            print(f'{ratio: .3f}', formatted)
            minDistance = minDistanceNew
            coeff = newCoeff

    return coeff


def check_knot_mine_hopf(
    xyzMesh,
    coeff,
    deltaCoeff,
    steps=1000,
    six_dots=True,
    checkboundaries=False,
    boundaryValue=0.2,
    width=1.3,
    circletest=True,
    radiustest=0.05,
    testvisual=False,
    xyzMeshPlot=None
):
    """
    Optimize Hopf‐knot coefficients by random perturbation to maximize the minimal
    distance between singularity dots, with optional circle, boundary, and visual tests.

    Parameters
    ----------
    xyzMesh : tuple of arrays
        Coordinate grids for the Hopf field evaluation (x,y,z).
    coeff : list of float
        Initial coefficients for the hopf_mod function.
    deltaCoeff : list of float
        Step sizes for random perturbations of coeff.
    steps : int, optional
        Number of iterations to attempt (default: 1000).
    six_dots : bool, optional
        If True, require exactly six singularity dots in distance calculation.
    checkboundaries : bool, optional
        If True, perform an empty‐space check on boundary mesh before accepting.
    boundaryValue : float, optional
        Threshold for empty_space_check (default: 0.2).
    width : float, optional
        Beam width parameter passed to hopf_mod (default: 1.3).
    circletest : bool, optional
        If True, require that a central‐slice circle survives circle_test.
    radiustest : float, optional
        Radius parameter for circle_test (default: 0.05).
    testvisual : bool, optional
        If True, run a final visual() test before accepting new coeff.
    xyzMeshPlot : tuple of arrays, optional
        Mesh used for boundary and visual tests (must be provided if
        checkboundaries or testvisual are True).

    Returns
    -------
    list of float
        The best‐found coefficient list maximizing the minimal dot distance.
    """
    # --- INITIAL EVALUATION ---
    # Compute field for the starting coefficients
    field = fOAM.hopf_mod(
        *xyzMesh, w=1.4, width=width, k0=1, z0=0.,
        coeff=coeff, coeffPrint=False
    )

    # Extract only the singularity‐dot locations from the phase
    dotsOnly = fg.cut_non_oam(np.angle(field), axesAll=False)[1]

    # Mesh resolution (only zRes is used below)
    xRes, yRes, zRes = np.shape(xyzMesh)[1:]

    # Baseline minimal distance between dots
    minDistance = min_distance_hopf(dotsOnly, zRes, six_dots=six_dots)

    # --- OPTIMIZATION LOOP ---
    for i in range(steps):
        # Propose new coefficients by random perturbation
        newCoeff = fg.random_list(coeff, deltaCoeff)

        # Recompute field with proposed coefficients
        newField = fOAM.hopf_mod(
            *xyzMesh, w=1.4, width=width, k0=1, z0=0.,
            coeff=newCoeff, coeffPrint=False
        )

        # Extract dots for the new field
        dotsOnly = fg.cut_non_oam(np.angle(newField), axesAll=False)[1]

        # 1) Optional circle test at the mid‐slice
        if circletest:
            mid = newField.shape[2] // 2
            if not circle_test(
                np.angle(newField)[:, :, mid],
                radius=radiustest,
                testValue=2.5
            ):
                print('circle')  # failed circle test
                continue

        # 2) Compute the new minimal dot distance
        minDistanceNew = min_distance_hopf(dotsOnly, zRes, six_dots=six_dots)
        print(
            f"Iter {i:4d} | "
            f"old_dist = {minDistance:6.2f} → "
            f"new_dist = {minDistanceNew:6.2f} | "
            f"coeffs  = {newCoeff}"
        )

        # 3) Accept the new coefficients only if they improve the distance
        if minDistanceNew > minDistance:
            # 3a) Optional boundary‐space check
            if checkboundaries:
                fieldBound = fOAM.hopf_mod(
                    *xyzMeshPlot, w=1.4, width=width, k0=1, z0=0.,
                    coeff=coeff, coeffPrint=False
                )
                dotsOnlyBound = fg.cut_non_oam(
                    np.angle(fieldBound), axesAll=False
                )[1]
                xResB, yResB, zResB = np.shape(xyzMeshPlot)[1:]
                if not empty_space_check(dotsOnlyBound, zResB, boundaryValue):
                    print('Boundary is bad')
                    continue
                else:
                    print('Boundary is good')

            # 3b) Optional visual sanity check
            if testvisual:
                print('test visual')
                if not visual(dotsOnly, coeff, xyzMeshPlot, knot='hopf'):
                    print('visualy no')
                    continue

            # 3c) Commit to the new best solution
            print(
                f'{minDistance / minDistanceNew: .3f}',
                [float(f'{a:.2f}') for a in newCoeff]
            )
            minDistance = minDistanceNew
            coeff = newCoeff

    return coeff



def cost_function_paper(field, iMin=0.01, i0=0.01, norm=1e6):
    I0 = np.max(np.abs(field)) ** 2
    I0 *= i0
    IFlat = np.ndarray.flatten(np.abs(field) ** 2)
    cutParam = I0 / i0 * iMin
    IFlat[IFlat < cutParam] = cutParam
    IMin = [1 / min(x, I0) for x in IFlat]
    # IMin = [1 / max(x, I0) for x in IFlat]
    return np.sum(IMin) / norm


def knot_permutations_all(aInitial, deltaCoeff, dotsNumber):
    aValues = []
    for i, a in enumerate(aInitial):
        aArray = np.linspace(a - deltaCoeff[i], a + deltaCoeff[i], dotsNumber[i])
        aValues.append(aArray)
    return fg.permutations_all(*aValues)


def circle_test(field, radius, testValue=1.):
    shape = np.shape(field)
    radius *= np.sqrt((shape[0] // 2 + shape[1] // 2) ** 2)
    for x in range(shape[0]):
        for y in range(shape[1]):
            if np.sqrt((x - shape[0] // 2) ** 2 + (y - shape[1] // 2) ** 2) <= radius:
                if np.abs(field[x, y]) > testValue:
                    return False
    return True


def check_knot_paper(xyzMesh, coeff, deltaCoeff, iMin, i0, radiustest=0.05, steps=1000, ):
    field = fOAM.trefoil_mod(
        *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
        aCoeff=coeff
        , coeffPrint=False,
    )
    sum = cost_function_paper(field, iMin=iMin, i0=i0)
    for i in range(steps):
        print(i)
        newCoeff = fg.random_list(coeff, deltaCoeff)
        newField = fOAM.trefoil_mod(
            *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
            aCoeff=newCoeff, coeffPrint=False)
        newSum = cost_function_paper(newField, iMin=iMin, i0=i0)
        if newSum < sum:
            if circle_test(np.angle(newField)[:, :, np.shape(newField)[2] // 2],
                           radius=radiustest, testValue=2.5):
                print(f'{sum / newSum: .3f}', [float(f'{a:.2f}') for a in newCoeff])
                fOAM.plot_knot_dots(newField, axesAll=False)
                plt.show()
                while True:
                    x = int(input())
                    if x == 9 or x == 1:
                        sum = newSum
                        coeff = newCoeff
                        break
                    if x == 0:
                        break
            else:
                print(f'It is not a knot anymore!')
    print(coeff)
    return coeff

    # check_knot_paper()
    # newCoeff = fg.random_list(initialCoeff, deltaCoeff)
    # newField = fOAM.trefoil_mod(
    #     *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
    #     coeff=newCoeff, coeffPrint=False)
    # newSum = cost_function_paper(newField, iMin=iMin, i0=i0)
    # if newSum < initialSum:
    #     if circle_test(np.angle(newField)[:, :, np.shape(newField)[2] // 2],
    #                    radius=0.04, testValue=2.5):
    #         print(f'{costMod / newSum: .3f}', [float(f'{a:.2f}') for a in newCoeff])
    #         fOAM.plot_knot_dots(newField, axesAll=False)
    #         plt.show()
    #         while True:
    #             x = int(input())
    #             if x == 9 or x == 1:
    #                 initialSum = newSum
    #                 initialCoeff = newCoeff
    #                 break
    #             if x == 0:
    #                 break
    #
    #     else:
    #         print(f'It is not a knot anymore!')


def visual(dotsOnly, coeff=None, xyzMeshVisual=None, sound=True, knot='trefoil'):
    if coeff is None:
        dots = np.array([list(dots) for (dots, OAM) in dotsOnly.items()])
        fg.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
        plt.show()
    else:
        if knot == 'trefoil':
            fieldTest = fOAM.trefoil_mod(
                *xyzMeshVisual, w=1.2, width=1.2, k0=1, z0=0.,
                aCoeff=coeff, coeffPrint=False
            )
        else:
            fieldTest = fOAM.hopf_mod(
                *xyzMeshVisual, w=1.2, width=1.2, k0=1, z0=0.,
                coeff=coeff, coeffPrint=False
            )
        if sound:
            duration = 300  # milliseconds
            freq = 440  # Hz
            winsound.Beep(freq, duration)
        fOAM.plot_knot_dots(fieldTest, axesAll=True, color='r', size=200)
        plt.show()

    while True:
        x = int(input())
        if x == 9 or x == 1:
            return True
        if x == 0:
            return False


def return_min_helper(dots, minDistance):
    for i in range(len(dots) - 1):
        for j in range(i + 1, len(dots)):
            currentDistance = fg.distance_between_points(dots[i], dots[j])
            # print(currentDistance)
            if currentDistance < minDistance:
                minDistance = currentDistance
    return minDistance


def dots12_check(dotsWithOAM, minDistance):
    dotsPlus = [dot for dot, OAM in dotsWithOAM if OAM > 0]
    dotsMinus = [dot for dot, OAM in dotsWithOAM if OAM < 0]
    minDistancePlus = return_min_helper(dotsPlus, minDistance)
    minDistanceMinus = return_min_helper(dotsMinus, minDistance)
    if minDistanceMinus < minDistancePlus:
        return minDistanceMinus
    else:
        return minDistancePlus


def min_distance(dotsOnly, zRes, six_dots=True):
    minDistance = float('inf')
    have_seen_12_dots = False
    for z in range(zRes):

        dotsInZwithOam = [(list(dots[:2]), OAM) for (dots, OAM) in dotsOnly.items()
                          if dots[2] == z]
        dotsInZ = [dot for dot, OAM in dotsInZwithOam]
        if (six_dots and len(dotsInZ) != 6) or (have_seen_12_dots and len(dotsInZ) != 12):
            break
        elif 6 < len(dotsInZ) < 12:
            continue
        elif len(dotsInZ) == 12:  # 12 dots
            if six_dots:
                return 0
            have_seen_12_dots = True
            minDistance = dots12_check(dotsInZwithOam, minDistance)

        else:  # just 6 dots
            potMinDistance = return_min_helper(dotsInZ, minDistance)
            if (not (potMinDistance < minDistance * 0.94)) or z == 0:  #######################
                minDistance = potMinDistance
            else:
                break
        # fg.plot_2D(fieldFull[:, :, z])
        # plt.show()
    return minDistance
    # print(fg.distance_between_points(dot, dotsInZ[0][:2]))


def min_distance_hopf(dotsOnly, zRes, six_dots=False):
    minDistance = float('inf')
    have_seen_8_dots = False
    for z in range(zRes):

        dotsInZwithOam = [(list(dots[:2]), OAM) for (dots, OAM) in dotsOnly.items()
                          if dots[2] == z]
        dotsInZ = [dot for dot, OAM in dotsInZwithOam if OAM > 0]
        if (six_dots and len(dotsInZ) != 4) or (have_seen_8_dots and len(dotsInZ) != 8):
            break
        elif 4 < len(dotsInZ) < 8:
            continue
        elif len(dotsInZ) == 8:  # 12 dots
            if six_dots:
                return 0
            have_seen_8_dots = True
            minDistance = dots12_check(dotsInZwithOam, minDistance)

        else:  # just 6 dots
            potMinDistance = return_min_helper(dotsInZ, minDistance)
            minDistance = potMinDistance
        # fg.plot_2D(fieldFull[:, :, z])
        # plt.show()
    return minDistance
    # print(fg.distance_between_points(dot, dotsInZ[0][:2]))


def empty_space_check(dotsOnly, zRes, valueTest):
    zArray = [dot[2] for dot in dotsOnly]
    zArray.sort()
    if max(zArray) < zRes * (1 - valueTest):
        return True
    else:
        return False
    # add here the distance in between

