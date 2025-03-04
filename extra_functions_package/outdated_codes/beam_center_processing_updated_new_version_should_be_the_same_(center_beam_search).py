"""
This Python script provides algorithms to find the beam center and beam tilt using the electrical field.
The algorithms are based on quasi-intrinsic-variance and Orbital Angular Momentum (OAM) spectrum calculations.
For references, check the following papers:
1) Yi-Dong Liu, et al., "Orbital angular momentum (OAM) spectrum correction in free space optical communication", OE, 2008.
   This paper describes the algorithm in detail. The implementation here is slightly different.
2) https://doi.org/10.1103/PhysRevLett.96.113901.
   This paper provides more theory on the quasi-intrinsic-variance and intrinsic/extrinsic parts of OAM in general.

The main function is called `beamFullCenter`, which computes the beam center and tilt. The script also includes several helper functions for field manipulation and analysis.

## Import Statements

- External Libraries:
  - numpy: For numerical operations.
  - matplotlib.pyplot: For plotting.

- Custom Modules:
  - my_functions.functions_general: General utility functions.
  - my_functions.singularities: Functions for handling singularities.
  - my_functions.plotings: Functions for plotting.
  - my_functions.beams_and_pulses: Functions related to beams and pulses.

## Functions

### LG_spectre_coeff
Calculates a single coefficient of the LG spectrum for a given field.

### displacement_lateral
Shifts the field laterally by a specified radius-vector and angle.

### displacement_deflection
Tilts the field using specified angles eta and gamma.

### removeTilt
Removes the tilt from a field and returns the corrected field.

### removeShift
Removes the shift from a mesh and returns the new mesh.

### LG_transition_matrix_real_space
Calculates the transition matrix element for a given operator in real space.

### shiftTiltCombined
Combines both shift and tilt operations on a field.

### variance_V_helper
Helper function for calculating variance.

### variance_single_transition_combined
Calculates the final variance for combined shift and tilt operations.

### LG_spectrum
Calculates the LG spectrum of a given beam.

### beamFullCenter
Finds the beam center and tilt by minimizing variance.

### find_width
Finds the approximate beam waist.

### beam_center_coordinates
Finds the beam center coordinates considering both shift and tilt.

### variance_single_transition
Calculates variance for single shift or tilt operations.

### variance_map_shift
Generates a variance map for lateral displacement.

### variance_map_tilt
Generates a variance map for tilt displacement.

### center_beam_finding
Finds the center of the beam by minimizing variance for lateral displacement.

### tilt_beam_finding
Finds the tilt of the beam by minimizing variance for tilt displacement.

The `__main__` section provides examples and is mainly used for testing.
"""

import extra_functions_package.functions_general as fg
import extra_functions_package.singularities as sing
import extra_functions_package.plotings as pl
import extra_functions_package.beams_and_pulses as bp
import numpy as np

def LG_spectre_coeff(field, l, p, xM=(-1, 1), yM=(-1, 1), width=1., k0=1., mesh=None, functions=bp.LG_simple, **kwargs):
    """
    Function calculates a single coefficient of LG_l_p in the LG spectrum of the field
    :param field: complex electric field
    :param l: azimuthal index of LG beam
    :param p: radial index of LG beam
    :param xM: x boundaries for an LG beam (if Mesh is None)
    :param yM: y boundaries for an LG beam (if Mesh is None)
    :param width: LG beam width
    :param k0: k0 in LG beam but I believe it doesn't affect anything since we are in z=0
    :param mesh: mesh for LG beam. if None, xM and yM are used
    :return: complex weight of LG_l_p in the spectrum
    """
    if mesh is None:
        shape = np.shape(field)
        mesh = fg.create_mesh_XY(xMinMax=xM, yMinMax=yM, xRes=shape[0], yRes=shape[1])
        dS = ((xM[1] - xM[0]) / (shape[0] - 1)) * ((yM[1] - yM[0]) / (shape[1] - 1))
    else:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        dS = (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
        # print(123, xArray)
    # shape = np.shape(field)
    # xyMesh = fg.create_mesh_XY_old(xMax=xM[1], yMax=yM[1], xRes=shape[0], yRes=shape[1], xMin=xM[0], yMin=yM[0])
    LGlp = functions(*mesh, l=l, p=p, width=width, k0=k0, **kwargs)
    # plt.imshow(LGlp)
    # plt.show()
    # print('hi')

    return np.sum(field * np.conj(LGlp)) * dS


def displacement_lateral(field_func, mesh, r_0, eta, **kwargs):
    """
    function shifts the field (function, not an array) to r_0 under eta angle
    :param field_func: any 2d function fo
    :param mesh: mesh for the final field
    :param r_0: radius-vector in polar coordinates
    :param eta: angle in polar coordinates
    :return: array with a shifted field
    """
    xArray, yArray = fg.arrays_from_mesh(mesh)
    xArray, yArray = xArray - r_0 * np.cos(eta), yArray - r_0 * np.sin(eta)
    mesh_new = np.meshgrid(xArray, yArray, indexing='ij')
    return field_func(*mesh_new, **kwargs)


def displacement_deflection(field_func, mesh, eta, gamma, k=1, **kwargs):
    """
    function tilt the field (function, not an array) using angles eta and gamma
    :param field_func: any 2d function fo
    :param mesh: mesh for the final field
    :param eta: check the first reference for the definition
    :param gamma: check the first reference for the definition
    :return: array with a shifted field
    """
    field_change = np.exp(1j * k * fg.rho(*mesh) * np.sin(gamma) * np.cos(fg.phi(*mesh) - eta))
    return field_func(*mesh, **kwargs) * field_change


def removeTilt(field, mesh, eta, gamma, k=1):
    """
    Function remove the tilt from the field (array) and returns new field
    :param field: initial field
    :param mesh: have to provide the mesh for the initial function
    :param k: wave number
    :return: field array
    """
    field_change = np.exp(1j * k * fg.rho(*mesh) * np.sin(gamma) * np.cos(fg.phi(*mesh) - eta))
    return field * field_change


def removeShift(mesh, x0, y0):
    """
    Function remove the shift from the mesh (array) and returns new mesh
    :param mesh: initial mesh
    :return: mesh array
    """
    xArray, yArray = fg.arrays_from_mesh(mesh)
    xArray, yArray = xArray - x0, yArray - y0
    mesh_new = np.meshgrid(xArray, yArray, indexing='ij')
    return mesh_new


def LG_transition_matrix_real_space(operator, l, p, l0, p0, xM=(-1, 1), yM=(-1, 1), shape=(100, 100),
                                    width=1., k0=1., mesh=None, **kwargs):
    """
    Transition element in equation (10) in the first reference. It works for any operator, not only misalignment
    operation T_LD.
    :param operator: T_LD, T_DD or their combination
    :param xM: x boundaries for an LG beam (if Mesh is None)
    :param yM: y boundaries for an LG beam (if Mesh is None)
    :param shape: size of the new mesh (if Mesh is None)
    :param width: LG beam width
    :param k0: k0 in LG beam but I believe it doesn't affect anything since we are in z=0
    :param mesh: mesh for LG beam. if None, xM and yM are used
    :param kwargs: into the operator
    :return: the value of the transition element
    """
    if mesh is None:
        mesh = fg.create_mesh_XY(xM, yM, xRes=shape[0], yRes=shape[1])
        dS = ((xM[1] - xM[0]) / (shape[0] - 1)) * ((yM[1] - yM[0]) / (shape[1] - 1))
    else:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        dS = (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
    operatorOnLG = operator(bp.LG_simple,
                            mesh, z=0, l=l0, p=p0, width=width, k0=k0, x0=0, y0=0, z0=0, **kwargs)
    LGlp = bp.LG_simple(*mesh, l=l, p=p, width=width, k0=k0)
    return np.sum(operatorOnLG * np.conj(LGlp)) * dS


def shiftTiltCombined(field_func, mesh, r_0, eta, eta2, gamma, k=1, **kwargs):
    """
    This function combines both shift and tilt. Basically, it's displacement_lateral() + displacement_deflection()
    See more details in these 2 functions
    """
    xArray, yArray = fg.arrays_from_mesh(mesh)
    xArray, yArray = xArray - r_0 * np.cos(eta), yArray - r_0 * np.sin(eta)
    mesh_new = np.meshgrid(xArray, yArray, indexing='ij')
    field_change = np.exp(1j * k * fg.rho(*mesh) * np.sin(gamma) * np.cos(fg.phi(*mesh) - eta2))
    return field_func(*mesh_new, **kwargs) * field_change


def variance_V_helper(Pl, lArray):
    """
    helper for the variance_single_transition: eq (23)
    Not used anywhere else
    """
    sum1, sum2 = 0, 0
    # sumTest = 0
    for p, l in zip(Pl, lArray):
        # sumTest += p
        sum1 += p * l ** 2
        sum2 += p * l
    # print(sumTest)
    return sum1 - sum2 ** 2


def variance_single_transition_combined(field, mesh, r, eta, eta2, gamma,
                                        displacement_function=shiftTiltCombined,
                                        p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                                        width=1., k0=1.):
    """
    This function is the 'heart' of the algorithm implementation.
    It calculate the final variance (eq. (23) 1st reference).
    (Not sure if it works for only tilt or only shift if you change the the displacement_function.
    It worked in the separated implementations below, should work here. You probably don't need this to separate
    them anyway, it can be used only to increase the speed for some tests)
    :param displacement_function: shiftTiltCombined for search both tilt and shift.
    :param p: lowest and highest values for the whole integration range (see eqn. (10))
    :param l: lowest and highest values for the whole integration range (see eqn. (10))
    :param p0: lowest and highest values for the whole integration range (see eqn. (10))
    :param l0: lowest and highest values for the whole integration range (see eqn. (10))
    :param width: LG width
    :param k0: LG k0, doesn't affect anything, we work in z=0
    :return: the value of the variance
    """
    p1, p2 = p
    l1, l2 = l
    p01, p02 = p0
    l01, l02 = l0
    Pl = np.zeros(l2 - l1 + 1)
    for ind_l, l in enumerate(np.arange(l1, l2 + 1)):
        sum = 0
        for p in np.arange(p1, p2 + 1):
            sum_inner = 0
            for l0 in np.arange(l01, l02 + 1):
                for p0 in np.arange(p01, p02 + 1):
                    element_ = LG_transition_matrix_real_space(displacement_function, l=l, p=p,
                                                               l0=l0, p0=p0, mesh=mesh,
                                                               r_0=r, eta=eta, eta2=eta2, gamma=gamma,
                                                               width=width, k0=k0)
                    value = LG_spectre_coeff(field, l=l0, p=p0, mesh=mesh, width=width, k0=k0)
                    # print(element_, value, (np.abs(element_ * value) ** 2))
                    sum_inner += (element_ * value)
            sum += np.abs(sum_inner) ** 2
        Pl[ind_l] = sum
        # print(Pl[ind_l])
    V = variance_V_helper(Pl, np.arange(l1, l2 + 1))
    return V

def LG_spectrum(beam, l=(-3, 3), p=(0, 5), xM=(-1, 1), yM=(-1, 1), width=1., k0=1., mesh=None, plot=True,
                functions=bp.LG_simple, cmap='jet', **kwargs):
    """

    :param beam:
    :param l:
    :param p:
    :param xM:
    :param yM:
    :param width:
    :param k0:
    :param mesh:
    :param plot:
    :return:
    """
    l1, l2 = l
    p1, p2 = p
    spectrum = np.zeros((l2 - l1 + 1, p2 - p1 + 1), dtype=complex)
    # spectrumReal = []
    # modes = []
    for l in np.arange(l1, l2 + 1):
        for p in np.arange(p1, p2 + 1):
            value = LG_spectre_coeff(beam, l=l, p=p, xM=xM, yM=yM, width=width, k0=k0, mesh=mesh,
                                     functions=functions, **kwargs)
            # print(l, p, ': ', value, np.abs(value))
            spectrum[l - l1, p] = value
            # if np.abs(value) > 0.5:
            # spectrumReal.append(value)
            # modes.append((l, p))
    # print(modes)
    if plot:
        import matplotlib.pyplot as plt
        pl.plot_2D(np.abs(spectrum), x=np.arange(l1 - 0.5, l2 + 1 + 0.5), y=np.arange(p1 - 0.5, p2 + 1 + 0.5),
                   interpolation='none', grid=True, xname='l', yname='p', map=cmap, show=False)
        plt.yticks(np.arange(p1, p2 + 1))
        plt.xticks(np.arange(l1, l2 + 1))
        plt.show()
    return spectrum


def beamFullCenter(beam, mesh, stepEG=None, stepXY=None, displacement_function=shiftTiltCombined,
                   p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                   width=1, k0=1, x=None, y=None, eta2=0., gamma=0., threshold=0.99,
                   print_info=False):
    if stepEG is None:
        stepEG = 0.1, 0.1
    if stepXY is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        # print(xArray)
        stepXY = xArray[1] - xArray[0], yArray[1] - yArray[0]
    if x is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        x = xArray[len(xArray) // 2]
    if y is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        y = yArray[len(yArray) // 2]

    def search(coordinate):
        nonlocal x, y, eta2, gamma, var0
        varIt = var0
        signX = 1
        correctWay = False
        while True:
            if coordinate == 'x':
                x = x + signX * stepXY[0]
            elif coordinate == 'y':
                y = y + signX * stepXY[1]
            elif coordinate == 'eta2':
                eta2 = eta2 + signX * stepEG[0]
            elif coordinate == 'gamma':
                gamma = gamma + signX * stepEG[1]

            var = variance_single_transition_combined(beam, mesh, r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                                      eta2=eta2, gamma=gamma,
                                                      displacement_function=displacement_function,
                                                      p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
            if print_info:
                print(f'x={round(x, 3)}, y={round(y, 3)},'
                      f' eta={round(eta2 * 180 / np.pi, 3)}*, gamma={round(gamma * 180 / np.pi, 3)}*, var={var}')
            if var < (varIt * threshold):
                varIt = var
                correctWay = True
            else:
                if coordinate == 'x':
                    x = x - signX * stepXY[0]
                elif coordinate == 'y':
                    y = y - signX * stepXY[1]
                elif coordinate == 'eta2':
                    eta2 = eta2 - signX * stepEG[0]
                elif coordinate == 'gamma':
                    gamma = gamma - signX * stepEG[1]
                if correctWay:
                    break
                else:
                    correctWay = True
                    signX *= -1

        return varIt

    var0 = variance_single_transition_combined(beam, mesh, r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                               eta2=eta2, gamma=gamma,
                                               displacement_function=displacement_function,
                                               p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)

    while True:
        search(coordinate='x')
        search(coordinate='y')
        search(coordinate='eta2')
        varEG = search(coordinate='gamma')
        if varEG == var0:
            print('FINAL COORDINATES:')
            print(f'x={round(x, 5)}, y={round(y, 5)},'
                  f' eta={round(eta2 * 180 / np.pi, 5)}*,'
                  f' gamma={round(gamma * 180 / np.pi, 5)}*, var={var0}')
            return x, y, eta2, gamma
        else:
            var0 = varEG


def find_width(beam, mesh, widthStep=0.1, l=(-8, 8), p=(0, 9), width=1., k0=1., print_steps=True):
    """
    this function finds the approximate beam waste (any position, any beam=sum(LG))
    :param mesh: mesh is required to have a scale of the beam => correct width scale
    :param widthStep: precision of the beam width search (but it's not real precision, method is not very accurate)
    :param l: we wont to cover all spectrum
    :param p: we wont to cover all spectrum
    :param width: starting beam width (you better have it as accurate as poosible, since the method is bad)
    :param k0: does nothing
    :return: ~beam width
    """
    minSpec = np.sum(np.abs(LG_spectrum(beam, l=l, p=p, mesh=mesh, width=width, k0=k0, plot=False)) ** (1 / 2))
    correctWay = False
    direction = +1
    while True:
        width += direction * widthStep
        spec = np.sum(np.abs(LG_spectrum(beam, l=l, p=p, mesh=mesh, width=width, k0=k0, plot=False)) ** (1 / 2))
        if print_steps:
            print(width, spec)
        if spec < minSpec:
            minSpec = spec
            correctWay = True
        else:
            width -= direction * widthStep
            if correctWay:
                break
            else:
                correctWay = True
                direction *= -1
    return width


def beam_center_coordinates(beam, mesh, stepXY=None, stepEG=None,
                            p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                            width=1, k0=1, x=None, y=None, eta=0., gamma=0.,
                            shift=True, tilt=True, fast=False):
    if fast or not shift or not tilt:
        print(f'Not the perfect center: shift={shift}, tilt={tilt}, fast={fast}')
        if shift:
            x, y = center_beam_finding(beam, mesh, x=x, y=y, stepXY=stepXY,
                                       p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
        if tilt:
            eta, gamma = tilt_beam_finding(beam, mesh, eta=eta, gamma=gamma, stepEG=stepEG,
                                           p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
    else:
        meshNew = mesh
        while True:
            print('New cycle')
            # print(x, y, eta, gamma)
            x, y = center_beam_finding(beam, meshNew, x=x, y=y, stepXY=stepXY,
                                       p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
            eta, gamma = tilt_beam_finding(beam, mesh, eta=eta, gamma=gamma, stepEG=stepEG,
                                           p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)

            meshNew = removeShift(meshNew, -x, -y)
            # beam = removeTilt(beam, mesh, eta=eta, gamma=-gamma)
            pl.plot_2D(np.angle(beam))
            pl.plot_2D(np.abs(beam))
            print(x, y, eta, gamma)
            if (x, y, eta, gamma) == (0, 0, 0, 0):
                break
            x, y, eta, gamma = [0] * 4
        # xBest, yBest, etaBest, gammaBest = [np.inf] * 4
        # while (xBest, yBest, etaBest, gammaBest) != (x, y, eta, gamma):
        #     print('New cycle')
        #     # print(x, y, eta, gamma)
        #     xBest, yBest, etaBest, gammaBest = (x, y, eta, gamma)
        #     print(xBest, yBest, etaBest, gammaBest)
        #     x, y = center_beam_finding(beam, mesh, x=x, y=y, stepXY=stepXY,
        #                                p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
        #     eta, gamma = tilt_beam_finding(beam, mesh, eta=eta, gamma=gamma, stepEG=stepEG,
        #                                    p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
        #
        #     mesh = removeShift(mesh, -x, -y)
        #     beam = removeTilt(beam, mesh, eta=eta, gamma=gamma)
        #     print(x, y, eta, gamma)
        #     x, y, eta, gamma = [0] * 4

    print(f'x={x}, y={y}, eta={eta / np.pi * 180}, gamma={gamma / np.pi * 180}')
    return mesh, beam


# functions below are old and may not be functional. There were used for the algorithm implementation
# for separated shift and tilt, not combined.
# Maybe useful if all the fields have only shift or only tilt
####################################################################################


def variance_single_transition(field, mesh, r, eta, displacement_function=displacement_lateral,
                               p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                               width=1., k0=1.):
    """
    basically the same as variance_single_transition_combined() but for shift or tilt
    """
    p1, p2 = p
    l1, l2 = l
    p01, p02 = p0
    l01, l02 = l0
    Pl = np.zeros(l2 - l1 + 1)
    for ind_l, l in enumerate(np.arange(l1, l2 + 1)):
        sum = 0
        for p in np.arange(p1, p2 + 1):
            sum_inner = 0
            for l0 in np.arange(l01, l02 + 1):
                for p0 in np.arange(p01, p02 + 1):
                    if displacement_function is displacement_lateral:
                        element_ = LG_transition_matrix_real_space(displacement_function, l=l, p=p,
                                                                   l0=l0, p0=p0,
                                                                   mesh=mesh, r_0=r, eta=eta,
                                                                   width=width, k0=k0)
                    else:
                        element_ = LG_transition_matrix_real_space(displacement_function, l=l, p=p,
                                                                   l0=l0, p0=p0,
                                                                   mesh=mesh, eta=r, gamma=eta,
                                                                   width=width, k0=k0)
                    value = LG_spectre_coeff(field, l=l0, p=p0, mesh=mesh, width=width, k0=k0)
                    # print(element_, value, (np.abs(element_ * value) ** 2))
                    sum_inner += (element_ * value)
            sum += np.abs(sum_inner) ** 2
        Pl[ind_l] = sum
        # print(Pl[ind_l])
    V = variance_V_helper(Pl, np.arange(l1, l2 + 1))
    return V


def variance_map_shift(beam, mesh, displacement_function=displacement_lateral,
                       resolution_V=(4, 4), xBound=(-1, 1), yBound=(-1, 1), width=1., k0=1.,
                       p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3)):
    V = np.zeros(resolution_V)

    xArray = np.linspace(*xBound, resolution_V[0])
    yArray = np.linspace(*yBound, resolution_V[1])
    for i, x in enumerate(xArray):
        print('Main Coordinate x: ', i)
        for j, y in enumerate(yArray):
            print('y: ', j)
            r = fg.rho(x, y)
            eta = np.angle(x + 1j * y)
            V[i, j] = variance_single_transition(beam, mesh, r, eta,
                                                 displacement_function=displacement_function,
                                                 width=width, k0=k0,
                                                 p=p, l=l, p0=p0, l0=l0)
    return V


def variance_map_tilt(beam, mesh, displacement_function=displacement_deflection,
                      resolution_V=(4, 4), etaBound=(-1, 1), gammaBound=(-1, 1), width=1., k0=1.,
                      p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3)):
    V = np.zeros(resolution_V)

    etaArray = np.linspace(*etaBound, resolution_V[0])
    gammaArray = np.linspace(*gammaBound, resolution_V[1])
    for i, eta in enumerate(etaArray):
        print('Main Coordinate eta: ', i)
        for j, gamma in enumerate(gammaArray):
            print('gamma: ', j)
            V[i, j] = variance_single_transition(beam, mesh, r=eta, eta=gamma,
                                                 displacement_function=displacement_function,
                                                 width=width, k0=k0,
                                                 p=p, l=l, p0=p0, l0=l0)
    return V


def center_beam_finding(beam, mesh, stepXY=None, displacement_function=displacement_lateral,
                        p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                        width=1, k0=1, x=None, y=None):
    if stepXY is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        # print(xArray)
        stepXY = xArray[1] - xArray[0], yArray[1] - yArray[0]
    if x is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        x = xArray[len(xArray) // 2]
    if y is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        y = yArray[len(yArray) // 2]

    def search(xFlag):
        nonlocal x, y, var0
        varIt = var0
        signX = 1
        correctWay = False
        while True:
            if xFlag:
                x = x + signX * stepXY[0]
            else:
                y = y + signX * stepXY[1]
            var = variance_single_transition(beam, mesh=mesh, displacement_function=displacement_function,
                                             r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                             p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
            print(f'x={x}, y={y}, var={var}')
            if var < varIt:
                varIt = var
                correctWay = True
            else:
                if xFlag:
                    x = x - signX * stepXY[0]
                else:
                    y = y - signX * stepXY[1]
                if correctWay:
                    break
                else:
                    correctWay = True
                    signX *= -1

        return varIt

    var0 = variance_single_transition(beam, mesh=mesh, r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                      p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
    # print(f'var0={var0}')

    while True:
        search(xFlag=True)
        varXY = search(xFlag=False)
        if varXY == var0:
            return x, y
        else:
            var0 = varXY


def tilt_beam_finding(beam, mesh, stepEG=None, displacement_function=displacement_deflection,
                      p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                      width=1, k0=1, eta=0., gamma=0.):
    if stepEG is None:
        stepEG = 0.1, 0.1

    def search(etaFlag):
        nonlocal eta, gamma, var0
        varIt = var0
        signX = 1
        correctWay = False
        while True:
            if etaFlag:
                eta = eta + signX * stepEG[0]
            else:
                gamma = gamma + signX * stepEG[1]
            var = variance_single_transition(beam, mesh, eta, gamma,
                                             displacement_function=displacement_function,
                                             p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)

            print(f'eta={eta}, gamma={gamma}, var={var}')
            if var < varIt:
                varIt = var
                correctWay = True
            else:
                if etaFlag:
                    eta = eta - signX * stepEG[0]
                else:
                    gamma = gamma - signX * stepEG[1]
                if correctWay:
                    break
                else:
                    correctWay = True
                    signX *= -1

        return varIt

    var0 = variance_single_transition(beam, mesh, eta, gamma,
                                      displacement_function=displacement_function,
                                      p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
    # print(f'var0={var0}')

    while True:
        search(etaFlag=True)
        varEG = search(etaFlag=False)
        if varEG == var0:
            return eta, gamma
        else:
            var0 = varEG


####################################################################################


if __name__ == '__main__':

    # test = np.load('coordinates.npy')
    # np.savetxt('centers3.txt', test)
    # np.savetxt('centers2.txt', np.round(test))
    # exit()
    import matplotlib.pyplot as plt

    xB, yB = [-4, 4], [-4, 4]
    xyMesh = fg.create_mesh_XY_old(xB[1], yB[1], 40, 40, xMin=xB[0], yMin=yB[0])

    xDis = 0.4
    yDis = -0.3
    eta = 40
    gamma = 10


    def beamF(*xyMesh, width=1, **kwargs):
        return bp.LG_combination(*xyMesh,
                                 coefficients=[1 / np.sqrt(2), 1 / np.sqrt(2)],
                                 # coefficients=[1, 0],
                                 modes=[(0, 0), (2, 1)],
                                 width=[width, width], x0=xDis, y0=yDis, **kwargs)


    beam_displaced = displacement_lateral(beamF, xyMesh, r_0=fg.rho(xDis, yDis),
                                          eta=np.angle(xDis + 1j * yDis))
    beam = beam_displaced
    ########
    # xyMesh = fg.create_mesh_XY(xRes=50, yRes=50)
    # beam = beamF(*xyMesh, width=7) * 10
    # beam = beam / np.sqrt(np.sum(np.abs(beam)**2))
    # print(np.sum(np.abs(beam)**2) )
    # pl.plot_2D(np.abs(beam))
    # exit()
    beam_tilted = displacement_deflection(beamF, xyMesh,
                                          eta=eta * np.pi / 180, gamma=gamma * np.pi / 180)
    beam = beam_tilted

    pl.plot_2D(np.abs(beam_tilted))
    pl.plot_2D(np.angle(beam_tilted))
    pl_dict = {'p': (0, 3), 'l': (-3, 4), 'p0': (0, 3), 'l0': (-3, 4)}
    # beam_center_coordinates(beam, xyMesh, stepXY=(0.1, 0.1), stepEG=(0.1, 0.1),
    #                         **pl_dict,
    #                         width=1, k0=1, x=0, y=0, eta=0., gamma=0.,
    #                         shift=True, tilt=False, fast=False)
    # new_xy_mesh = removeShift(xyMesh, 0.5, 0.5)
    # beam = removeTilt(beam, new_xy_mesh, eta=30*np.pi/180, gamma=-20*np.pi/180)
    beam = removeTilt(beam, xyMesh, eta=140 / 180 * np.pi, gamma=10 / 180 * np.pi)
    ax = pl.plot_2D(np.abs(beam), x=[-4, 4], y=[-4, 4], show=False)
    pl.plot_scatter_2D(x=0.4, y=-0.3, xlim=[-4, 4], ylim=[-4, 4], ax=ax, color='g', size=150, show=True)
    pl.plot_2D(np.angle(beam), x=[-4, 4], y=[-4, 4])
    exit()
    spec = LG_spectrum(beam, l=(-4, 5), p=(0, 4), mesh=xyMesh, plot=True, width=1, k0=1)

    beamFullCenter(beam, xyMesh, stepXY=(0.1, 0.1), stepEG=(5 / 180 * np.pi, 1 / 180 * np.pi),
                   **pl_dict, threshold=1,
                   width=1, k0=1, x=0, y=0, eta2=0., gamma=0.)
    exit()

    beam_center_coordinates(beam, xyMesh, stepXY=(0.1, 0.1), stepEG=(2.5 / 180 * np.pi, 2.5 / 180 * np.pi),
                            **pl_dict,
                            width=1, k0=1, x=0, y=0, eta=0., gamma=0.,
                            shift=True, tilt=True, fast=False)
    exit()
    width = 1.0
    k0 = 1
    # V = variance_map_tilt(beam=beam, mesh=xyMesh,
    #                       resolution_V=(5, 5), etaBound=((eta-10) * np.pi / 180, (eta + 10) * np.pi / 180),
    #                       gammaBound=((-gamma-10) * np.pi / 180, (-gamma+10) * np.pi / 180),
    #                       **pl_dict, width=width)
    # print(V)
    # pl.plot_2D(V, x=[eta-10, eta+10], y=[-gamma-10, -gamma+10])
    # V = variance_map_shift(beam=beam, mesh=xyMesh,
    #                  resolution_V=(5, 5), xBound=(-1, 1), yBound=(-1, 1),
    #                  **pl_dict, width=width)
    # print(V)
    # pl.plot_2D(V, x=[-1, 1], y=[-1, 1])
    # exit()

    check_spectrum = False
    if check_spectrum:
        l1, l2 = -3, 3
        p1, p2 = 0, 3
        spectrum = np.zeros((l2 - l1 + 1, p2 - p1 + 1))
        spectrumReal = []
        modes = []
        for l in np.arange(l1, l2 + 1):
            for p in np.arange(p1, p2 + 1):
                value = LG_spectre_coeff(beam, l=l, p=p, xM=xB, yM=yB, width=1, k0=1)
                # print(l, p, ': ', value, np.abs(value))
                spectrum[l - l1, p] = np.abs(value)
                # if np.abs(value) > 0.5:
                spectrumReal.append(value)
                modes.append((l, p))
        # print(modes)

        pl.plot_2D(spectrum, y=np.arange(l1 - 0.5, l2 + 1 + 0.5), x=np.arange(p1 - 0.5, p2 + 1 + 0.5),
                   interpolation='none', grid=True, xname='p', yname='l', show=False)
        plt.yticks(np.arange(p1, p2 + 1))
        plt.xticks(np.arange(l1, l2 + 1))
        plt.show()
        exit()
        beam = bp.LG_combination(*xyMesh, coefficients=spectrumReal, modes=modes)
        pl.plot_2D(np.abs(beam), axis_equal=True)
        pl.plot_2D(np.angle(beam), axis_equal=True, vmin=-np.pi, vmax=np.pi)
        sing.Jz_calc_no_conj(beam)

    plot_trefoil = True
    if plot_trefoil:
        xyzMesh = fg.create_mesh_XYZ(2.1, 2.1, 0.7, 50, 50, 50, zMin=None)
        beam = bp.LG_combination(*xyzMesh,
                                 coefficients=[1.71, -5.66, 6.38, -2.30, -4.36],
                                 modes=[(0, 0), (0, 1), (0, 2), (0, 3), (3, 0)],
                                 width=[1, 1, 1, 1, 1])
        dots = sing.get_singularities(np.angle(beam))
        # pl.plot_2D(np.abs(beam[:, :, zRes//2]))
        # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
        fig = pl.plot_3D_dots_go(dots)
        pl.box_set_go(fig, mesh=None, autoDots=dots, perBox=0.05)
        fig.show()
