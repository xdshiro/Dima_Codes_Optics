import numpy as np
from scipy.special import assoc_laguerre
import helper_scripts_for_itterative_optimization.functions_general as fg
import matplotlib.pyplot as plt


def laguerre_polynomial(x, l, p):
    return assoc_laguerre(x, p, l)


def LG_simple(x, y, z, l=1, p=0, width=1, k0=1, x0=0, y0=0):
    x = x - x0
    y = y - y0
    zR = k0 * width ** 2

    E = (np.sqrt(np.math.factorial(p) / (np.pi * np.math.factorial(np.abs(l) + p)))
         * fg.rho(x, y) ** np.abs(l) * np.exp(1j * l * fg.phi(x, y))
         / (width ** (np.abs(l) + 1) * (1 + 1j * z / zR) ** (np.abs(l) + 1))
         * ((1 - 1j * z / zR) / (1 + 1j * z / zR)) ** p
         * np.exp(-fg.rho(x, y) ** 2 / (2 * width ** 2 * (1 + 1j * z / zR)))
         * laguerre_polynomial(fg.rho(x, y) ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)), np.abs(l), p)
         )
    return E


def link(x, y, z, w, width=1, k0=1, z0=0.):
    a00 = -1 + w ** 2
    a01 = -w ** 2
    a10 = -2 * w
    z = z - z0

    field = (a00 * LG_simple(x, y, z, l=0, p=0, width=width, k0=k0) +
             a01 * LG_simple(x, y, z, l=0, p=1, width=width, k0=k0) +
             a10 * LG_simple(x, y, z, l=1, p=0, width=width, k0=k0))

    return field


def trefoil(x, y, z, w, width=1, k0=1, z0=0.):
    z = z - z0
    a00 = 1 - w ** 2 - 2 * w ** 4 + 6 * w ** 6
    a01 = w ** 2 * (1 + 4 * w ** 2 - 18 * w ** 4)
    a02 = - 2 * w ** 4 * (1 - 9 * w ** 2)
    a03 = -6 * w ** 6
    a30 = -8 * np.sqrt(6) * w ** 3
    field = (a00 * LG_simple(x, y, z, l=0, p=0, width=width, k0=k0) +
             a01 * LG_simple(x, y, z, l=0, p=1, width=width, k0=k0) +
             a02 * LG_simple(x, y, z, l=0, p=2, width=width, k0=k0) +
             a03 * LG_simple(x, y, z, l=0, p=3, width=width, k0=k0) +
             a30 * LG_simple(x, y, z, l=3, p=0, width=width, k0=k0)
             )

    return field


def trefoil_mod(x, y, z, w, width=1, k0=1, z0=0., aCoeff=None, coeffPrint=False):
    z = z - z0
    H = 1.0
    if aCoeff is not None or aCoeff is False:
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    else:
        a00 = 1 * (H ** 6 - H ** 4 * w ** 2 - 2 * H ** 2 * w ** 4 + 6 * w ** 6) / H ** 6
        a01 = (w ** 2 * (1 * H ** 4 + 4 * w ** 2 * H ** 2 - 18 * w ** 4)) / H ** 6
        a02 = (- 2 * w ** 4 * (H ** 2 - 9 * w ** 2)) / H ** 6
        a03 = (-6 * w ** 6) / H ** 6
        a30 = (-8 * np.sqrt(6) * w ** 3) / H ** 3
        aCoeff = [a00, a01, a02, a03, a30]
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    if coeffPrint:
        print(aCoeff)
        print(f'a00 -> a01 -> a02 ->... -> a0n -> an0:')
        for i, a in enumerate(aCoeff):
            print(f'a{i}: {a:.3f}', end=',\t')
        print()
    field = (aCoeff[0] * LG_simple(x, y, z, l=0, p=0, width=width, k0=k0) +
             aCoeff[1] * LG_simple(x, y, z, l=0, p=1, width=width, k0=k0) +
             aCoeff[2] * LG_simple(x, y, z, l=0, p=2, width=width, k0=k0) +
             aCoeff[3] * LG_simple(x, y, z, l=0, p=3, width=width, k0=k0) +
             aCoeff[4] * LG_simple(x, y, z, l=3, p=0, width=width, k0=k0)
             )
    return field

def trefoil_mod_2(x, y, z, w, width=1, k0=1, z0=0., aCoeff=None, coeffPrint=False):
    z = z - z0
    H = 1.0
    if aCoeff is not None or aCoeff is False:
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    else:
        a00 = 1 * (H ** 6 - H ** 4 * w ** 2 - 2 * H ** 2 * w ** 4 + 6 * w ** 6) / H ** 6
        a01 = (w ** 2 * (1 * H ** 4 + 4 * w ** 2 * H ** 2 - 18 * w ** 4)) / H ** 6
        a02 = (- 2 * w ** 4 * (H ** 2 - 9 * w ** 2)) / H ** 6
        a03 = (-6 * w ** 6) / H ** 6
        a30 = (-8 * np.sqrt(6) * w ** 3) / H ** 3
        aCoeff = [a00, a01, a02, a03, a30]
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    if coeffPrint:
        print(aCoeff)
        print(f'a00 -> a01 -> a02 ->... -> a0n -> an0:')
        for i, a in enumerate(aCoeff):
            print(f'a{i}: {a:.3f}', end=',\t')
        print()
    field = (aCoeff[0] * LG_simple(x, y, z, l=0, p=0, width=width, k0=k0) +
             aCoeff[1] * LG_simple(x, y, z, l=0, p=1, width=width, k0=k0) +
             aCoeff[2] * LG_simple(x, y, z, l=0, p=2, width=width, k0=k0) +
             aCoeff[3] * LG_simple(x, y, z, l=0, p=3, width=width, k0=k0) +
             aCoeff[4] * LG_simple(x, y, z, l=3, p=0, width=width, k0=k0) +
             aCoeff[5] * LG_simple(x, y, z, l=0, p=4, width=width, k0=k0) +
             aCoeff[6] * LG_simple(x, y, z, l=0, p=5, width=width, k0=k0)

             )
    return field


def hopf_mod_4(x, y, z, w, width=1, k0=1, z0=0., aCoeff=None, coeffPrint=False):
    z = z - z0
    H = 1.0
    if aCoeff is not None or aCoeff is False:
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    else:
        a00 = 1 - 2 * w ** 2 + 2 * w ** 4
        a01 = 2 * w ** 2 - 4 * w ** 4
        a02 = 2 * w ** 4
        a20 = 4 * np.sqrt(2) * w ** 2
        aCoeff = [a00, a01, a02, a20]
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    if coeffPrint:
        print(aCoeff)
        print(f'a00 -> a01 -> a02 ->... -> a0n -> an0:')
        for i, a in enumerate(aCoeff):
            print(f'a{i}: {a:.3f}', end=',\t')
        print()
    field = (aCoeff[0] * LG_simple(x, y, z, l=0, p=0, width=width, k0=k0) +
             aCoeff[1] * LG_simple(x, y, z, l=0, p=1, width=width, k0=k0) +
             aCoeff[2] * LG_simple(x, y, z, l=0, p=2, width=width, k0=k0) +
             aCoeff[3] * LG_simple(x, y, z, l=2, p=0, width=width, k0=k0)
             )
    return field


def hopf_mod(x, y, z, w, width=1, k0=1, z0=0., coeff=None, coeffPrint=False):
    z = z - z0
    H = 1.0
    if coeff is not None or coeff is True:
        paper = [2.63, -6.32, 4.21, -5.95]
        aCoeff = coeff
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    else:
        a00 = 1 - 2 * w ** 2 + 2 * w ** 4
        a01 = 2 * w ** 2 - 4 * w ** 4
        a02 = 2 * w ** 4
        a20 = 4 * np.sqrt(2) * w ** 2
        aCoeff = [a00, a01, a02, a20]
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    if coeffPrint:
        print(aCoeff)
        print(f'a00 -> a01 -> a02 ->... -> a0n -> an0:')
        for i, a in enumerate(aCoeff):
            print(f'a{i}: {a:.3f}', end=',\t')
        print()
    field = (aCoeff[0] * LG_simple(x, y, z, l=0, p=0, width=width, k0=k0) +
             aCoeff[1] * LG_simple(x, y, z, l=0, p=1, width=width, k0=k0) +
             aCoeff[2] * LG_simple(x, y, z, l=0, p=2, width=width, k0=k0) +
             aCoeff[3] * LG_simple(x, y, z, l=2, p=0, width=width, k0=k0)
             )
    return field


def knot_all(x, y, z, w, width=1, k0=1, z0=0., knot=None):
    if knot is None:
        knot = 'trefoil'
    if knot == 'trefoil':
        return trefoil(x, y, z, w, width=width, k0=k0, z0=z0)
    elif knot == 'hopf':
        print('add the HOPF!')
    elif knot == 'link':
        return link(x, y, z, w, width=width, k0=k0, z0=z0)
    elif knot == 'trefoil_mod':
        return trefoil_mod(x, y, z, w, width=width, k0=k0, z0=z0)


# # this one hasn't been checked yet
# def Jz_calc(EArray, xArray, yArray):
#     x0 = (xArray[-1] + xArray[0]) / 2
#     y0 = (yArray[-1] + yArray[0]) / 2
#     sum = 0
#     dx = xArray[1] - xArray[0]
#     dy = yArray[1] - yArray[0]
#     x = xArray - x0
#     y = yArray - y0
#     for i in range(1, len(xArray) - 1, 1):
#         for j in range(1, len(yArray) - 1, 1):
#             dEx = (EArray[i + 1, j] - EArray[i - 1, j]) / (2 * dx)
#             dEy = (EArray[i, j + 1] - EArray[i, j - 1]) / (2 * dy)
#             sum += (np.conj(EArray[i, j]) *
#                     (x[i] * dEy - y[j] * dEx))
#
#     return np.imag(sum * dx * dy)


# ploting and saving math trefoil
def knot_field_plot_save(xyMax=3, zMax=1, xyRes=50, zRes=50, w=1, width=1, k0=1,
                         knot=None, axis_equal=True,
                         save=False, saveName='rename_me', plot=True, plotLayer=None):
    xyArray = np.linspace(-xyMax, xyMax, xyRes)
    xyzMesh = fg.create_mesh_XYZ(xMin=-xyMax, xMax=xyMax,
                                 yMin=-xyMax, yMax=xyMax,
                                 zMin=-zMax, zMax=zMax, xRes=xyRes, yRes=xyRes, zRes=zRes)
    field = knot_all(xyzMesh[0], xyzMesh[1], z=xyzMesh[2], w=w, width=width, k0=k0, knot=knot)
    if plot:
        if plotLayer is None:
            plotLayer = zRes // 2
        fg.plot_2D(np.abs(field)[:, :, plotLayer], xyArray, xyArray, axis_equal=axis_equal)
        fg.plot_2D(np.angle(field)[:, :, plotLayer], xyArray, xyArray, map='hsv', axis_equal=axis_equal)
    if save:
        np.save(saveName, field)


def trefoil_test(x, y, z, *args):
    a00 = 0.61
    a01 = -2.56
    a02 = 6.15
    a03 = -6.35
    a04 = 2.92
    a05 = -0.61
    a30 = -2.45
    field = (a00 * LG_simple(x, y, z, l=0, p=0, width=1, k0=1) +
             a01 * LG_simple(x, y, z, l=0, p=1, width=1, k0=1) +
             a02 * LG_simple(x, y, z, l=0, p=2, width=1, k0=1) +
             a03 * LG_simple(x, y, z, l=0, p=3, width=1, k0=1) +
             a04 * LG_simple(x, y, z, l=0, p=4, width=1, k0=1) +
             a05 * LG_simple(x, y, z, l=0, p=5, width=1, k0=1) +
             a30 * LG_simple(x, y, z, l=3, p=0, width=1, k0=1)
             )

    return field


def milnor_Pol_testing(x, y, z, *args):
    R = fg.rho(x, y)
    f = fg.phi(x, y)
    R = fg.rho(x, y)
    f = fg.phi(x, y)
    u = (-1 ** 2 + R ** 2 + 2j * z * 1 + z ** 2) / (1 ** 2 + R ** 2 + z ** 2)
    v = (2 * R * 1 * np.exp(1j * f)) / (1 ** 2 + R ** 2 + z ** 2)
    divider = (1 + R ** 2 + z ** 2)
    return u ** 3 - v / 10  # / divider ** 2
    # return (
    #         (-2 * np.exp(1j * f) * R * (1 + R ** 2 + z ** 2) ** 2
    #          + (-1 + R ** 2 + 2 * 1j * z + z ** 2) ** 5)
    #         / (1 + R ** 2 + z ** 2) ** 5
    # )


# def milnor_Pol_testing(x, y, z):
#     R = fg.rho(x, y)
#     f = fg.phi(x, y)
#     return (
#             (-2 * np.exp(1j * f) * R * (1 + R ** 2 + z ** 2) ** 2
#              + (-1 + R ** 2 + 2 * 1j * z + z ** 2) ** 5)
#             / (1 + R ** 2 + z ** 2) ** 3
#     )


def milnor_Pol_u_v_any(x, y, z, uOrder, vOrder, H=1):
    """This function create u^a-v^b Milnor polynomial"""
    R = fg.rho(x, y)
    f = fg.phi(x, y)
    u = (-H ** 2 + R ** 2 + 2j * z * H + z ** 2) / (H ** 2 + R ** 2 + z ** 2)
    v = (2 * R * H * np.exp(1j * f)) / (H ** 2 + R ** 2 + z ** 2)
    return u ** uOrder - v ** vOrder


# plotting dot's only from the Array of +-1
def plot_knot_dots(field, bigSingularity=False, axesAll=True, cbrt=False,
                   size=plt.rcParams['lines.markersize'] ** 2, color=None):
    if isinstance(field, dict):
        dotsOnly = field
    else:
        dotsFull, dotsOnly = fg.cut_non_oam(np.angle(field),
                                            bigSingularity=bigSingularity, axesAll=axesAll, cbrt=cbrt)
    dotsPlus = np.array([list(dots) for (dots, OAM) in dotsOnly.items() if OAM == 1])
    dotsMinus = np.array([list(dots) for (dots, OAM) in dotsOnly.items() if OAM == -1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if len(np.shape(dotsPlus)) == 2:
        fg.plot_scatter_3D(dotsPlus[:, 0], dotsPlus[:, 1], dotsPlus[:, 2], ax=ax, size=size, color=color)
        if len(np.shape(dotsMinus)) == 2:
            fg.plot_scatter_3D(dotsMinus[:, 0], dotsMinus[:, 1], dotsMinus[:, 2], ax=ax, size=size, color=color)
    else:
        if len(np.shape(dotsPlus)) == 2:
            fg.plot_scatter_3D(dotsMinus[:, 0], dotsMinus[:, 1], dotsMinus[:, 2], ax=ax, size=size, color=color)
        else:
            print(f'no singularities to plot')
    return ax


def save_knot_dots(field, filename, bigSingularity=False, axesAll=True, cbrt=False):
    dotsFull, dotsOnly = fg.cut_non_oam(np.angle(field),
                                        bigSingularity=bigSingularity, axesAll=axesAll, cbrt=cbrt)

    np.save(filename, dotsOnly)
