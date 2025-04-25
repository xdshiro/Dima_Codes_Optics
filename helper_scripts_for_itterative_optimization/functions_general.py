import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy.io as sio
from scipy.interpolate import CloughTocher2DInterpolator

# common parameters
# global fig, ax
ticksFontSize = 18
xyLabelFontSize = 20
legendFontSize = 20


def create_mesh_XYZ(xMax, yMax, zMax, xRes, yRes, zRes,
                    xMin=None, yMin=None, zMin=None, indexing='ij'):
    if xMin is None:
        xMin = -xMax
    if yMin is None:
        yMin = -yMax
    if zMin is None:
        zMin = -zMax
    xArray = np.linspace(xMin, xMax, xRes)
    yArray = np.linspace(yMin, yMax, yRes)
    zArray = np.linspace(zMin, zMax, zRes)
    return np.array(np.meshgrid(xArray, yArray, zArray, indexing=indexing))


def create_mesh_XY(xMax, yMax, xRes, yRes,
                   xMin=None, yMin=None, indexing='ij'):
    if xMin is None:
        xMin = -xMax
    if yMin is None:
        yMin = -yMax
    xArray = np.linspace(xMin, xMax, xRes)
    yArray = np.linspace(yMin, yMax, yRes)
    return np.array(np.meshgrid(xArray, yArray, indexing=indexing))


def resolution_decrease(E, resDecrease=None):
    if resDecrease is None:
        resDecrease = [1, 1, 1]
    newE = E[::resDecrease[0], ::resDecrease[1], ::resDecrease[2]]
    return newE


def cut_middle_values(E, min, max, midValue=0, minValue=None, maxValue=None):
    ans = np.copy(E)
    minFlag, maxFlag = False, False
    if minValue is not None:
        minFlag = True
    if maxValue is not None:
        maxFlag = True
    shape = np.shape(ans)
    if len(shape) == 1:
        for i in range(shape[0]):
            if ans[i] <= min:
                if minFlag:
                    ans[i] = minValue
            elif ans[i] >= max:
                if maxFlag:
                    ans[i] = maxValue
            else:
                ans[i] = midValue
    else:
        for i in range(shape[0]):
            ans[i] = cut_middle_values(ans[i], min, max, midValue, minValue=minValue, maxValue=maxValue)
    return ans


def fill_dict_as_matrix_helper(E, dots=None, nonValue=0, check=False):
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


def plane_singularities_finder_9dots(E, circle, value, nonValue, bigSingularity):
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


def cut_non_oam(E, value=1, nonValue=0, bigSingularity=False, axesAll=False, cbrt=False, circle=1):
    # E = np.angle(E)
    """this function finds singularities
    returns [3D Array, dots only]
    """
    shape = np.shape(E)
    dots = {}
    if len(shape) == 2:
        ans = plane_singularities_finder_9dots(E, circle, value, nonValue, bigSingularity)
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
            if cbrt:
                ans = np.array(np.cbrt(ans), dtype=int)

    # print(ans)
    return [ans, dots]


def propagator_split_step_3D(E, dz=1, xArray=None, yArray=None, zSteps=1, n0=1, k0=1):
    if xArray is None:
        xArray = np.array(range(np.shape(E)[0]))
    if yArray is None:
        yArray = np.array(range(np.shape(E)[1]))
    xResolution, yResolution = len(xArray), len(yArray)
    zResolution = zSteps + 1
    intervalX = xArray[-1] - xArray[0]
    intervalY = yArray[-1] - yArray[0]

    # xyMesh = np.array(np.meshgrid(xArray, yArray, indexing='ij'))
    if xResolution // 2 == 1:
        kxArray = np.linspace(-1. * np.pi * (xResolution - 2) / intervalX,
                              1. * np.pi * (xResolution - 2) / intervalX, xResolution)
        kyArray = np.linspace(-1. * np.pi * (yResolution - 2) / intervalY,
                              1. * np.pi * (yResolution - 2) / intervalY, yResolution)
    else:
        kxArray = np.linspace(-1. * np.pi * (xResolution - 0) / intervalX,
                              1. * np.pi * (xResolution - 2) / intervalX, xResolution)
        kyArray = np.linspace(-1. * np.pi * (yResolution - 0) / intervalY,
                              1. * np.pi * (yResolution - 2) / intervalY, yResolution)

    KxyMesh = np.array(np.meshgrid(kxArray, kyArray, indexing='ij'))

    def nonlinearity_spec(E):
        return dz * 0

    # works fine!
    def linear_step(field):
        temporaryField = np.fft.fftshift(np.fft.fftn(field))
        temporaryField = (temporaryField *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[0] ** 2) *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[1] ** 2))  # something here in /2
        return np.fft.ifftn(np.fft.ifftshift(temporaryField))

    fieldReturn = np.zeros((xResolution, yResolution, zResolution), dtype=complex)
    fieldReturn[:, :, 0] = E
    for k in range(1, zResolution):
        fieldReturn[:, :, k] = linear_step(fieldReturn[:, :, k - 1])
        fieldReturn[:, :, k] = fieldReturn[:, :, k] * np.exp(nonlinearity_spec(fieldReturn[:, :, k]))

    return fieldReturn


def cut_filter(E, radiusPix=1, circle=True, phaseOnly=False):
    ans = np.copy(E)
    xCenter, yCenter = np.shape(ans)[0] // 2, np.shape(ans)[0] // 2
    if circle:
        for i in range(np.shape(ans)[0]):
            for j in range(np.shape(ans)[1]):
                if np.sqrt((xCenter - i) ** 2 + (yCenter - j) ** 2) > radiusPix:
                    ans[i, j] = 0
    else:
        if phaseOnly:
            zeros = np.abs(np.copy(ans))
            zeros = zeros.astype(complex, copy=False)
        else:
            zeros = np.zeros(np.shape(ans), dtype=complex)
        zeros[xCenter - radiusPix:xCenter + radiusPix + 1, yCenter - radiusPix:yCenter + radiusPix + 1] \
            = ans[xCenter - radiusPix:xCenter + radiusPix + 1, yCenter - radiusPix:yCenter + radiusPix + 1]
        ans = zeros
    return ans


# just a fourier filter in XZ cross-section
def cut_fourier_filter(E, radiusPix=1):
    ans = np.copy(E)
    ans = np.fft.fftshift(np.fft.fftn(ans))
    ans = cut_filter(ans, radiusPix=radiusPix)
    ans = np.fft.ifftn(np.fft.ifftshift(ans))
    print(np.shape(ans))
    return ans


# return the 3D array with the complex field
def one_plane_propagator(fieldPlane, dz, stepsNumber, n0=1, k0=1):  # , shapeWrong=False
    # if shapeWrong is not False:
    #     if shapeWrong is True:
    #         print(f'using the middle plane in one_plane_propagator (shapeWrong = True)')
    #         fieldPlane = fieldPlane[:, :, np.shape(fieldPlane)[2] // 2]
    #     else:
    #         fieldPlane = fieldPlane[:, :, np.shape(fieldPlane)[2] // 2 + shapeWrong]
    fieldPropMinus = propagator_split_step_3D(fieldPlane, dz=-dz, zSteps=stepsNumber, n0=n0, k0=k0)

    fieldPropPLus = propagator_split_step_3D(fieldPlane, dz=dz, zSteps=stepsNumber, n0=n0, k0=k0)
    fieldPropTotal = np.concatenate((np.flip(fieldPropMinus, axis=2), fieldPropPLus[:, :, 1:-1]), axis=2)
    return fieldPropTotal


def readingFile(fileName, fieldToRead="p_charges", printV=False):
    matFile = sio.loadmat(fileName, appendmat=False)
    if printV:
        print(matFile)
        exit()
    return np.array(matFile[fieldToRead])


def plot_3D_density(E, resDecrease=None,
                    xMinMax=None, yMinMax=None, zMinMax=None,
                    surface_count=20,
                    opacity=0.5,
                    opacityscale=None):
    if zMinMax is None:
        zMinMax = [0, 1]
    if yMinMax is None:
        yMinMax = [0, 1]
    if xMinMax is None:
        xMinMax = [0, 1]
    values = resolution_decrease(E, resDecrease)
    shape = np.array(np.shape(E))
    if resDecrease is not None:
        shape = (shape // resDecrease)
    X, Y, Z = np.mgrid[
              xMinMax[0]:xMinMax[1]:shape[0] * 1j,
              yMinMax[0]:yMinMax[1]:shape[1] * 1j,
              zMinMax[0]:zMinMax[1]:shape[2] * 1j
              ]
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),  # collapsed into 1 dimension
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=values.min(),
        isomax=values.max(),
        opacity=opacity,  # needs to be small to see through all surfaces
        opacityscale=opacityscale,
        surface_count=surface_count,  # needs to be a large number for good volume rendering
        colorscale='RdBu'
    ))
    fig.show()


def plotDots(dots):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dots[:, 0], dots[:, 1], dots[:, 2])  # plot the point (2,3,4) on the figure
    ax.view_init(70, 0)
    # plt.show()
    # plt.close()


def plot_1D(x, y, noX=False, label=None, xname='', yname='', ls='-', lw=4, color=None,
            ticksFontSize=ticksFontSize, xyLabelFontSize=xyLabelFontSize,
            legendFontSize=legendFontSize,
            marker='', markersize=12, markeredgecolor='k',
            loc='lower right',
            axis_equal=False,
            ax=None, title=None):
    if noX:
        x = range(len(y))
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, y, linestyle=ls,
            marker=marker, markersize=markersize, markeredgecolor=markeredgecolor,
            label=label, lw=lw, color=color)
    """    if dotsColor == 'rrr':
        plt.scatter(x, y, s=dotsSize, label=False)
    else:
        plt.scatter(x, y, s=dotsSize, label=False, color=color)"""
    plt.xticks(fontname='Times New Roman', fontsize=ticksFontSize)
    plt.yticks(fontname='Times New Roman', fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize, fontname='Times New Roman')
    ax.set_ylabel(yname, fontsize=xyLabelFontSize, fontname='Times New Roman')
    ax.set_title(title, fontsize=xyLabelFontSize, fontname='Times New Roman')
    if axis_equal:
        ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if label:
        legend = ax.legend(shadow=True, fontsize=legendFontSize,
                           facecolor='white', edgecolor='black', loc=loc)
        plt.setp(legend.texts, family='Times New Roman')


def plot_2D(E, x=None, y=None, xname='', yname='', map='jet', vmin=None, vmax=None, title='',
            ticksFontSize=ticksFontSize, xyLabelFontSize=xyLabelFontSize,
            axis_equal=False,
            xlim=None, ylim=None, ax=None):
    if x is None:
        x = range(np.shape(E)[0])
    if y is None:
        y = range(np.shape(E)[1])
    if ax is None:
        if axis_equal:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
    image = plt.imshow(E,
                       interpolation='bilinear', cmap=map,
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[y[0], y[-1], x[0], x[-1]],
                       vmin=vmin, vmax=vmax, label='sdfsd')
    cbr = plt.colorbar(image, shrink=0.8, pad=0.02, fraction=0.1)
    cbr.ax.tick_params(labelsize=ticksFontSize)
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize)
    ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    plt.title(title, fontweight="bold", fontsize=26)
    if axis_equal:
        ax.set_aspect('equal', adjustable='box')
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    return ax


def plot_scatter_3D(X, Y, Z, ax=None, size=plt.rcParams['lines.markersize'] ** 2, color=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=size, color=color)  # plot the point (2,3,4) on the figure
    # ax.view_init(70, 0)
    # plt.show()
    # plt.close()
    return ax


def rho(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def phi(x, y):
    return np.angle(x + 1j * y)


"""
function cropping from xPos to xPos + cropX
by default it crops middle - cropX//2 to middle + cropX//2
if percentage is not None -> cropping this percentage along Z ( cylindrical coordinates) 
"""


def crop_array_3D(field, cropX=None, cropY=None, cropZ=None, percentage=None, xPos=None, yPos=None, zPos=None):
    shape = np.shape(field)
    if cropX is None:
        cropX = shape[0]
    if cropY is None:
        cropY = shape[1]
    if cropZ is None:
        cropZ = shape[2]
    if percentage is not None:
        cropX = int(shape[0] / 100 * percentage)
        cropY = int(shape[1] / 100 * percentage)
    if xPos is None:
        xPos = shape[0] // 2 - cropX // 2
    if yPos is None:
        yPos = shape[1] // 2 - cropY // 2
    if zPos is None:
        zPos = shape[2] // 2 - cropZ // 2

    return field[xPos:xPos + cropX, yPos:yPos + cropY, zPos:zPos + cropZ]


def crop_array_Values_3D(field, cropX=None, cropY=None, cropZ=None, percentage=None, xPos=None, yPos=None, zPos=None):
    shape = np.shape(field)
    if cropX is None:
        cropX = shape[0]
    if cropY is None:
        cropY = shape[1]
    if cropZ is None:
        cropZ = shape[2]
    if percentage is not None:
        cropX = int(shape[0] / 100 * percentage)
        cropY = int(shape[1] / 100 * percentage)
    if xPos is None:
        xPos = shape[0] // 2 - cropX // 2
    if yPos is None:
        yPos = shape[1] // 2 - cropY // 2
    if zPos is None:
        zPos = shape[2] // 2 - cropZ // 2
    answer = np.zeros(shape, dtype=np.complex)
    answer[xPos:xPos + cropX, yPos:yPos + cropY, zPos:zPos + cropZ] = field[
                                                                      xPos:xPos + cropX, yPos:yPos + cropY,
                                                                      zPos:zPos + cropZ]
    return answer


def size_array_increase_3D(field, cropX=None, cropY=None, cropZ=None, percentage=None, xPos=None, yPos=None, zPos=None):
    shape = np.shape(field)
    # cropX is bigger than shape[0]
    # position is in the cropX
    if cropX is None:
        cropX = shape[0] * 2
    if cropY is None:
        cropY = shape[1] * 2
    if cropZ is None:
        cropZ = shape[2]
    if percentage is not None:
        cropX = int(shape[0] / percentage * 100)
        cropY = int(shape[1] / percentage * 100)
    if xPos is None:
        xPos = cropX // 2
    if yPos is None:
        yPos = cropY // 2
    if zPos is None:
        zPos = cropZ // 2
    answer = np.zeros((cropX, cropY, cropZ), dtype=np.complex)
    answer[
    xPos - shape[0] // 2:xPos + (shape[0] - 1) // 2 + 1,
    yPos - shape[1] // 2:yPos + (shape[1] - 1) // 2 + 1,
    zPos - shape[2] // 2:zPos + (shape[2] - 1) // 2 + 1
    ] = field
    return answer


"""
fill_valuefloat, optional
Value used to fill in for requested points outside of the convex hull of the input points. If not provided, then the default is nan."""


# function interpolate real 2D array of any data into the function(x, y)
def interpolation_real(field, xArray=None, yArray=None):
    xResolution, yResolution = np.shape(field)
    if xArray is None:
        xArray = np.linspace(0, xResolution - 1, xResolution)
    if yArray is None:
        yArray = np.linspace(0, yResolution - 1, yResolution)
    xArrayFull = np.zeros(xResolution * yResolution)
    yArrayFull = np.zeros(xResolution * yResolution)
    fArray1D = np.zeros(xResolution * yResolution)
    for i in range(xResolution * yResolution):
        xArrayFull[i] = xArray[i // yResolution]
        yArrayFull[i] = yArray[i % xResolution]
        fArray1D[i] = field[i // yResolution, i % xResolution]
    return CloughTocher2DInterpolator(list(zip(xArrayFull, yArrayFull)), fArray1D)


# function interpolate complex 2D array of any data into the function(x, y)
def interpolation_complex(field, xArray=None, yArray=None):
    fieldReal = np.real(field)
    fieldImag = np.imag(field)
    return interpolation_real(fieldReal, xArray, yArray), interpolation_real(fieldImag, xArray, yArray)


def ft_2D_helper(field, xArray, yArray, kxArray, kyArray, sign):
    """
    The function processed ordinary 2D Fourier transformation (not FFT).
    This can be helpful to get the required resolution in any window
    :param field: 2D array of any complex field
    :return: return 2D spectrum in kArray x yArray
    """

    def integrand_helper(kx, ky, sign):
        integrand = np.copy(field)
        for i, x in enumerate(xArray):
            for j, y in enumerate(yArray):
                integrand[i, j] *= np.exp(sign * 1j * x * kx) * np.exp(sign * 1j * y * ky)
        return integrand

    spectrum = np.zeros((len(kxArray), len(kyArray)), dtype=complex)
    for i, kx in enumerate(kxArray):
        print(f'row #{i} out of {len(kxArray)}, ft={sign}')
        for j, ky in enumerate(kyArray):
            spectrum[i, j] = np.sum(integrand_helper(kx, ky, sign))

    return spectrum * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0]) / 2 / np.pi


def ft_forward_2D(field, xArray, yArray, kxArray, kyArray):
    return ft_2D_helper(field, xArray, yArray, kxArray, kyArray, -1)


def ft_reverse_2D(field, xArray, yArray, kxArray, kyArray):
    xArray, yArray, kxArray, kyArray = kxArray, kyArray, xArray, yArray
    return ft_2D_helper(field, xArray, yArray, kxArray, kyArray, +1)


def permutations_all(*arrays):
    """
    This function gets arrays and return the 1D array with all possible permutations
    :param arrays: any combinations of arrays, for example: [1][2,7,13][2,3]
    :return: [[1,2,2],[1,2,3],[1,7,2]...,[1,13,3]]
    """
    N = len(arrays)
    grid = np.meshgrid(*arrays, indexing='ij')
    transposing = np.roll(np.arange(N + 1), -1)  # [1, 2, 3,... N, 0]
    permutations = np.transpose(grid, transposing)
    return permutations.reshape(-1, N)  # -1 means unspecified value (как получится)


def random_list(values, diapason):
    import random
    """
    Function returns values + random * diapason
    :param values: we are changing this values
    :param diapason: to a random value from [value - diap, value + diap]
    :return: new modified values
    """
    answer = [x + random.uniform(-d, +d) for x, d in zip(values, diapason)]
    return answer


def distance_between_points(point1, point2):
    """
    distance between 2 points in any dimensions
    :param point1: [x1, ...]
    :param point2: [x2, ...]
    :return: geometrical distance
    """
    deltas = np.array(point1) - np.array(point2)
    ans = 0
    for delta in deltas:
        ans += delta ** 2
    return np.sqrt(ans)


def field_new_resolution(field, xResNew, yResNew):
    """
    This function approximate 2D complex field and return the numpy array with a new shape
    :param field: 2D array, dtype = complex, any resolution
    :param xResNew: new resolution along x
    :param yResNew: new resolution along y, can be different from x
    :return: the same field but with a new resolution
    """
    xRes, yRes = np.shape(field)
    f = interpolation_complex(field,
                              np.linspace(-1, 1, xRes), np.linspace(-1, 1, yRes))
    x, y = create_mesh_XY(1, 1, xResNew, yResNew)
    field_inter = f[0](x, y) + 1j * f[1](x, y)
    return field_inter


def region_increase(field, xyMinMax=5, xy_increase=2, xy_res_increase=2, k_increase=6, k_res_increase=2,
                    plot_origian=True, plot_new=True, cut=True):
    """
    increase the size of the window of the field by using the discrete fourier transformation
    :param field: any 2D field
    :param xyMinMax: boundaries of the new field. doesn't affect anything. affect kArray, but I don't think it's matter
    :param xy_increase: how many times larger/smaller (0.5) should be the new screen size
    :param xy_res_increase: new xy resolution
    :param k_increase: this parameter should be played with, higher if the field is more complicated
    :param k_res_increase: I think it is =xy_res_increase or =xy_increase (not sure, need more studying)
    :param plot_origian: ploting the field
    :param plot_new: ploting the new field with the increased region
    :param cut: it cuts the square of the size of the initial field. I don't think this is useful
    :return: return the new field with a new window size and resolution
    """
    if plot_origian:
        plot_2D(np.abs(field), title='original spec')
        plot_2D(np.angle(field), title='original abs')
        plt.show()
    xRes, yRes = np.shape(field)
    xArray = np.linspace(-xyMinMax, xyMinMax, xRes)
    yArray = np.linspace(-xyMinMax, xyMinMax, yRes)
    kRes = int(xRes * k_res_increase)
    kxArray = np.linspace(-k_increase * np.pi / xyMinMax, k_increase * np.pi / xyMinMax, kRes)
    kyArray = np.linspace(-k_increase * np.pi / xyMinMax, k_increase * np.pi / xyMinMax, kRes)
    xyResNew = int(xRes * xy_res_increase)
    print(xyResNew)
    exit()
    fieldSpec = ft_forward_2D(field, xArray, yArray, kxArray, kyArray)
    radiusCut = int(xyResNew / xy_increase / k_res_increase)  # / xy_res_increase
    xArrayNew = np.linspace(-xyMinMax * xy_increase, xyMinMax * xy_increase, xyResNew)
    yArrayNew = np.linspace(-xyMinMax * xy_increase, xyMinMax * xy_increase, xyResNew)
    fieldHiger = ft_reverse_2D(fieldSpec, xArrayNew, yArrayNew, kxArray, kyArray)
    if plot_new:
        plot_2D(np.abs(fieldSpec), title='spec')
        plot_2D(np.abs(fieldHiger), title='abs')
        plot_2D(np.angle(fieldHiger), title='spec')
        plt.show()
    if cut:
        fieldHiger = cut_filter(fieldHiger, radiusCut, circle=False,
                                phaseOnly=True)  # нужно не все отрезать, а сделать там модуль. Тогда будет хорошо
        plot_2D(np.angle(fieldHiger), title='spec circled')
    return fieldHiger


def Jz_calc(EArray, xArray=None, yArray=None):
    EArray = np.array(EArray)
    if xArray is None or yArray is None:
        shape = np.shape(EArray)
        xArray = np.arange(shape[0])
        yArray = np.arange(shape[1])
    x0 = (xArray[-1] + xArray[0]) / 2
    y0 = (yArray[-1] + yArray[0]) / 2
    x = np.array(xArray) - x0
    y = np.array(yArray) - y0
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    sumJz = 0
    for i in range(1, len(xArray) - 1, 1):
        for j in range(1, len(yArray) - 1, 1):
            dEx = (EArray[i + 1, j] - EArray[i - 1, j]) / (2 * dx)
            dEy = (EArray[i, j + 1] - EArray[i, j - 1]) / (2 * dy)
            sumJz += (np.conj(EArray[i, j]) *
                      (x[i] * dEy - y[j] * dEx))
    # Total moment
    Jz = np.imag(sumJz * dx * dy)
    W = np.sum(np.conj(EArray) * EArray, axis=0) * dx * dy
    # total power
    W = np.real(np.sum(W, axis=0))
    print(f'Total OAM = {Jz / W}\tW={W}')
    return Jz / W


def W_energy(EArray, xArray=None, yArray=None):
    if xArray is None or yArray is None:
        shape = np.shape(EArray)
        xArray = np.arange(shape[0])
        yArray = np.arange(shape[1])
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    W = np.real(np.sum(np.conj(EArray) * EArray) * dx * dy)
    return W


def Jz_calc_no_conj(EArray, xArray=None, yArray=None):
    EArray = np.array(EArray)
    Er, Ei = np.real(EArray), np.imag(EArray)
    if xArray is None or yArray is None:
        shape = np.shape(EArray)
        xArray = np.arange(shape[0])
        yArray = np.arange(shape[1])
    x0 = (xArray[-1] + xArray[0]) / 2
    y0 = (yArray[-1] + yArray[0]) / 2
    x = np.array(xArray) - x0
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
    # Total moment
    Jz = (sumJz * dx * dy)
    W = W_energy(EArray)
    print(f'Total OAM = {Jz / W}\tW={W}')
    return Jz


def deltaW(EArray, dEr, dEi, xArray=None, yArray=None):
    EArray = np.array(EArray)
    dEr = np.array(dEr)
    dEi = np.array(dEi)
    Er, Ei = np.real(EArray), np.imag(EArray)
    if xArray is None or yArray is None:
        shape = np.shape(EArray)
        xArray = np.arange(shape[0])
        yArray = np.arange(shape[1])
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    dW = 2 * np.sqrt(np.sum(np.abs(Er) ** 2 * dEr ** 2 + np.abs(Ei) ** 2 * dEi ** 2)) * dx * dy
    # dW = 2 * np.sqrt(np.sum(np.abs(Er) ** 2 * dEr ** 2) + np.sum(np.abs(Ei) ** 2 * dEi ** 2)) * dx * dy
    # dW = 2 * np.sum(np.sqrt(np.abs(Er) ** 2 * dEr ** 2 + np.abs(Ei) ** 2 * dEi ** 2)) * dx * dy
    return dW


def deltaJzDemo(EArray, dEr, dEi, xArray=None, yArray=None):
    EArray = np.array(EArray)
    dEr = np.array(dEr)
    dEi = np.array(dEi)
    Er, Ei = np.real(EArray), np.imag(EArray)
    if xArray is None or yArray is None:
        shape = np.shape(EArray)
        xArray = np.arange(shape[0])
        yArray = np.arange(shape[1])
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    x0 = (xArray[-1] + xArray[0]) / 2
    y0 = (yArray[-1] + yArray[0]) / 2
    x = np.array(xArray) - x0
    y = np.array(yArray) - y0
    sumdJz = 0
    for i in range(1, len(xArray) - 2, 1):
        for j in range(1, len(yArray) - 2, 1):
            dErx = (Er[i + 1, j] - Er[i - 1, j]) / (2 * dx)
            dEry = (Er[i, j + 1] - Er[i, j - 1]) / (2 * dy)
            dEix = (Ei[i + 1, j] - Ei[i - 1, j]) / (2 * dx)
            dEiy = (Ei[i, j + 1] - Ei[i, j - 1]) / (2 * dy)
            # if Er[i, j] == 0:
            #     dErx = 0
            #     dEry = 0
            # else:
            #     dErx = (Er[i + 1, j] - Er[i, j]) / (dx)
            #     dEry = (Er[i, j + 1] - Er[i, j]) / (dy)
            # if Ei[i, j] == 0:
            #     dEix = 0
            #     dEiy = 0
            # else:
            #     dEix = (Ei[i + 1, j] - Ei[i, j]) / (dx)
            #     dEiy = (Ei[i, j + 1] - Ei[i, j]) / (dy)
            d2Erx = (Er[i + 1, j] + Er[i - 1, j] - 2 * Er[i, j]) / (dx ** 2)
            d2Ery = (Er[i, j + 1] + Er[i, j - 1] - 2 * Er[i, j]) / (dy ** 2)
            d2Eix = (Ei[i + 1, j] + Ei[i - 1, j] - 2 * Ei[i, j]) / (dx ** 2)
            d2Eiy = (Ei[i, j + 1] + Ei[i, j - 1] - 2 * Ei[i, j]) / (dy ** 2)
            dErxP1 = (Er[i + 1, j + 1] - Er[i - 1, j + 1]) / (2 * dx)
            dEryP1 = (Er[i + 1, j + 1] - Er[i + 1, j - 1]) / (2 * dy)
            dEixP1 = (Ei[i + 1, j + 1] - Ei[i - 1, j + 1]) / (2 * dx)
            dEiyP1 = (Ei[i + 1, j + 1] - Ei[i + 1, j - 1]) / (2 * dy)
            d2Erxy = (dEryP1 - dEry) / dx
            d2Eryx = (dErxP1 - dErx) / dy
            d2Eixy = (dEiyP1 - dEiy) / dx
            d2Eiyx = (dEixP1 - dEix) / dy
            if (not (dEix + dEiy) and (dEix != 0)) or (not (dErx + dEry) and (dErx != 0)):
                print(f'the same derivitivies')
                continue
            if not dErx:
                s1r = 0
            else:
                s1r = (d2Erx + d2Eryx * dy / dx) / (dErx + dEry * dy / dx)
            if not dEry:
                s2r = 0
            else:
                s2r = (d2Erxy + d2Ery * dy / dx) / (dErx + dEry * dy / dx)
            if not dEix:
                s1i = 0
            else:
                s1i = (d2Eix + d2Eiyx * dy / dx) / (dEix + dEiy * dy / dx)
            if not dEiy:
                s2i = 0
            else:
                s2i = (d2Eixy + d2Eiy * dy / dx) / (dEix + dEiy * dy / dx)

            # print(dErx, dEry, dEix, dEiy, d2Erx, d2Eryx, d2Eix, d2Eiyx)
            _x, _y = x[i], y[j]
            _Er, _Ei = Er[i, j], Ei[i, j]
            # print(_x * dEiy, - _y * dEix, - _x * _Ei * s2r, + _y * _Ei * s1r)
            a = _x * dEiy - _y * dEix - _x * _Ei * s2r + _y * _Ei * s1r
            b = _x * _Er * s2i - _y * _Er * s1i - _x * dEry + _y * dErx
            sumdJz += (np.abs(a) * dEr[i, j] +
                       np.abs(b) * dEi[i, j])
            c = _x * _Er * dEiy - _y * _Er * dEix - _x * _Ei * dEry + _y * _Ei * dErx
            if c == 0 and a != 0:
                print(a, b, c)
                print(_x * dEiy, - _y * dEix, - _x * _Ei * s2r, + _y * _Ei * s1r)
                print(_x * _Er * dEiy, - _y * _Er * dEix, - _x * _Ei * dEry, + _y * _Ei * dErx)
                exit()
            # print(a, b, c)
    dJz = (sumdJz * dx * dy)
    return dJz


def deltaJz1(EArray, dEr, dEi, xArray=None, yArray=None):
    EArray = np.array(EArray)
    dEr = np.array(dEr)
    dEi = np.array(dEi)
    Er, Ei = np.real(EArray), np.imag(EArray)
    if xArray is None or yArray is None:
        shape = np.shape(EArray)
        xArray = np.arange(shape[0])
        yArray = np.arange(shape[1])
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    x0 = (xArray[-1] + xArray[0]) / 2
    y0 = (yArray[-1] + yArray[0]) / 2
    x = np.array(xArray) - x0
    y = np.array(yArray) - y0
    sumdJz = 0
    for i in range(1, len(xArray) - 2, 1):
        for j in range(1, len(yArray) - 2, 1):
            dErx = (Er[i + 1, j] - Er[i - 1, j]) / (2 * dx)
            dEry = (Er[i, j + 1] - Er[i, j - 1]) / (2 * dy)
            dEix = (Ei[i + 1, j] - Ei[i - 1, j]) / (2 * dx)
            dEiy = (Ei[i, j + 1] - Ei[i, j - 1]) / (2 * dy)
            _x, _y = x[i], y[j]
            _Er, _Ei = Er[i, j], Ei[i, j]
            # print(_x * dEiy, - _y * dEix, - _x * _Ei * s2r, + _y * _Ei * s1r)
            sEr00 = _x * dEiy - _y * dEix + _x * _Ei / dy - _y * _Ei / dx
            sEi00 = -_x * _Er / dy + _y * _Er / dx - _x * dEry + _y * dErx
            sEi01 = _x * _Er / dy
            sEi10 = - _y * _Er / dx
            sEr01 = - _x * _Ei / dy
            sEr10 = _y * _Ei / dx
            sumdJz += (abs(sEr00) * dEr[i, j] + abs(sEi00) * dEi[i, j] + abs(sEi01) * dEi[i, j + 1]
                       + abs(sEi10) * dEi[i + 1, j] + abs(sEr01) * dEr[i, j + 1] + abs(sEr10) * dEr[i + 1, j])

            # # print(x[i] * Er[i, j] * dEiy, - y[j] * Er[i, j] * dEix, -
            # #           x[i] * Ei[i, j] * dEry, + y[j] * Ei[i, j] * dErx)
            # sumJz += (x[i] * Er[i, j] * dEiy - y[j] * Er[i, j] * dEix -
            #           x[i] * Ei[i, j] * dEry + y[j] * Ei[i, j] * dErx)

            # print(a, b, c)
    dJz = (sumdJz * dx * dy)
    return dJz


def deltaJz2(EArray, dEr, dEi, xArray=None, yArray=None):
    EArray = np.array(EArray)
    dEr = np.array(dEr)
    dEi = np.array(dEi)
    Er, Ei = np.real(EArray), np.imag(EArray)
    if xArray is None or yArray is None:
        shape = np.shape(EArray)
        xArray = np.arange(shape[0])
        yArray = np.arange(shape[1])
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    x0 = (xArray[-1] + xArray[0]) / 2
    y0 = (yArray[-1] + yArray[0]) / 2
    x = np.array(xArray) - x0
    y = np.array(yArray) - y0
    sumdJz = 0
    for i in range(1, len(xArray) - 2, 1):
        for j in range(1, len(yArray) - 2, 1):
            dErx = (Er[i + 1, j] - Er[i - 1, j]) / (2 * dx)
            dEry = (Er[i, j + 1] - Er[i, j - 1]) / (2 * dy)
            dEix = (Ei[i + 1, j] - Ei[i - 1, j]) / (2 * dx)
            dEiy = (Ei[i, j + 1] - Ei[i, j - 1]) / (2 * dy)
            _x, _y = x[i], y[j]
            _Er, _Ei = Er[i, j], Ei[i, j]
            # print(_x * dEiy, - _y * dEix, - _x * _Ei * s2r, + _y * _Ei * s1r)
            sEr00 = _x * dEiy - _y * dEix
            sEi00 = - _x * dEry + _y * dErx
            sEi01 = _x * _Er / dy / 2
            sEi10 = - _y * _Er / dx / 2
            sEr01 = - _x * _Ei / dy / 2
            sEr10 = _y * _Ei / dx / 2
            sEi0_1 = - _x * _Er / dy / 2
            sEi_10 = _y * _Er / dx / 2
            sEr0_1 = _x * _Ei / dy / 2
            sEr_10 = - _y * _Ei / dx / 2
            sArray = [sEr00, sEi00,
                      sEi01, sEi10, sEr01, sEr10,
                      sEi0_1, sEi_10, sEr0_1, sEr_10]
            dEArray = [dEr[i, j], dEi[i, j],
                       dEi[i, j + 1], dEi[i + 1, j], dEr[i, j + 1], dEr[i + 1, j],
                       dEi[i, j - 1], dEi[i - 1, j], dEr[i, j - 1], dEr[i - 1, j]]
            error = 0
            for ind, s in enumerate(sArray):
                error += s ** 2 * dEArray[ind] ** 2
            sumdJz += np.sqrt(error)
            # # print(x[i] * Er[i, j] * dEiy, - y[j] * Er[i, j] * dEix, -
            # #           x[i] * Ei[i, j] * dEry, + y[j] * Ei[i, j] * dErx)
            # sumJz += (x[i] * Er[i, j] * dEiy - y[j] * Er[i, j] * dEix -
            #           x[i] * Ei[i, j] * dEry + y[j] * Ei[i, j] * dErx)

            # print(a, b, c)
    dJz = (sumdJz * dx * dy)
    return dJz


def deltaJz(EArray, dEr, dEi, xArray=None, yArray=None):
    EArray = np.array(EArray)
    dEr = np.array(dEr)
    dEi = np.array(dEi)
    Er, Ei = np.real(EArray), np.imag(EArray)
    shape = np.shape(EArray)
    if xArray is None or yArray is None:
        xArray = np.arange(shape[0])
        yArray = np.arange(shape[1])
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    x0 = (xArray[-1] + xArray[0]) / 2
    y0 = (yArray[-1] + yArray[0]) / 2
    x = np.array(xArray) - x0
    y = np.array(yArray) - y0
    sumdJzR = np.zeros(shape)
    sumdJzI = np.zeros(shape)
    for i in range(1, len(xArray) - 2, 1):
        for j in range(1, len(yArray) - 2, 1):
            dErx = (Er[i + 1, j] - Er[i - 1, j]) / (2 * dx)
            dEry = (Er[i, j + 1] - Er[i, j - 1]) / (2 * dy)
            dEix = (Ei[i + 1, j] - Ei[i - 1, j]) / (2 * dx)
            dEiy = (Ei[i, j + 1] - Ei[i, j - 1]) / (2 * dy)
            _x, _y = x[i], y[j]
            _Er, _Ei = Er[i, j], Ei[i, j]
            # error = (_x * dEiy - _y * dEix - _x * dEry + _y * dErx
            #           + _y * Er[i + 1, j] / dx / 2 - _y * Ei[i + 1, j] / dx / 2
            #           - _x * Er[i, j + 1] / dy / 2 + _x * Ei[i, j + 1] / dy / 2
            #           + _y * Er[i - 1, j] / dx / 2 - _y * Ei[i - 1, j] / dx / 2
            #           - _x * Er[i, j - 1] / dy / 2 + _x * Ei[i, j - 1] / dy / 2
            #           )
            errorR = (- _x * dEry + _y * dErx
                      + _y * Er[i + 1, j] / dx / 2
                      - _x * Er[i, j + 1] / dy / 2
                      + _y * Er[i - 1, j] / dx / 2
                      - _x * Er[i, j - 1] / dy / 2
                      )
            errorI = (_x * dEiy - _y * dEix
                      - _y * Ei[i + 1, j] / dx / 2
                      + _x * Ei[i, j + 1] / dy / 2
                      - _y * Ei[i - 1, j] / dx / 2
                      + _x * Ei[i, j - 1] / dy / 2
                      )
            sumdJzR = errorR
            sumdJzI = errorI
            # sumdJz[i, j] = error

    # dJz = 2 * np.sqrt(np.sum(np.abs(sumdJz) ** 2 * dEr ** 2)) * dx * dy
    dJz = 2 * np.sqrt(np.sum(np.abs(sumdJzR) ** 2 * dEr ** 2 + np.abs(sumdJzI) ** 2 * dEi ** 2)) * dx * dy
    return dJz
