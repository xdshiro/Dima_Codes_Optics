import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, iv
from scipy.integrate import simps
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from hankel import HankelTransform
import plotly.graph_objects as go


# %% resolution
xResolution = 100
xStart, xFinish = -0., 3.
yResolution = 100
yStart, yFinish = -0, 6
tResolution = 100
tStart, tFinish = -0.0, 3.0
lOAM = -1
lOAMs = 1

x0 = xFinish / 2
y0 = yFinish / 2
t0 = tFinish / 2
# split-step
# temporal
k_0 = 1
k_bis = 1
n_0 = 1
dz = 0.001

def g_R(r):
    return np.exp(-r ** 2)


def Hankel_transform(rho):
    def g_R_HT(r):
        x = r * np.power(2 * np.pi * rho, -1)
        return r * g_R(x)

    # Test and Optimize
    ht = HankelTransform(
        nu=np.abs(lOAM),  # The order of the bessel function
        N=100,  # Number of steps in the integration
        h=0.01  # Proxy for "size" of steps in integration
    )
    return ht.integrate(g_R_HT)[0] * np.power(2 * np.pi * rho, -2)


def HankelArray(rho):
    rhoSize = np.shape(rho)
    tempArray = np.zeros(rhoSize)
    for i in range(rhoSize[0]):
        for j in range(rhoSize[1]):
            tempArray[i, j] = Hankel_transform(rho[i, j])
    return tempArray


def Hankel_test(rho=1):
    rArray = np.linspace(0, 30, 10000)
    HlArray = jv(np.abs(lOAM), 2 * np.pi * rho * rArray) * g_R(rArray) * rArray
    test1 = simps(HlArray, rArray)
    print(test1)
    test2 = Hankel_transform(rho)
    print(test2)
    if np.abs(test1 / test2 - 1) > 0.01:
        exit('Something is wrong with the Hankel transform')


def rho(x, t):
    return np.sqrt((x - x0) ** 2 + (t - t0) ** 2)


def phi(x, t):
    return np.angle((x - x0) + 1j * (t - t0))

def phi2(x, t):
    return (np.sign(x - x0) + 1) / 2 * np.pi
    # return np.arctan2(x, t)


def H1(rho):
    return (np.pi ** (3 / 2) * rho / 4 * np.exp(-(2 * np.pi * rho) ** 2 / 8) *
            (iv(np.abs(0), (2 * np.pi * rho) ** 2 / 8) -
             iv(np.abs(1), (2 * np.pi * rho) ** 2 / 8)))


def H2(rho):
    return (1 / 2 / (2 * np.pi * rho) ** 2 * np.exp(-(2 * np.pi * rho) ** 2 / 4) *
            (4 * np.exp((2 * np.pi * rho) ** 2 / 4) - (2 * np.pi * rho) ** 2 - 4))


def y_dependence(y):
    return np.exp(-(y - y0) ** 2)


def reference_pulse(x, y):
    alpha = 40
    #alpha = 60
    return np.exp(1j * (x - x0) * alpha - (y - y0) ** 2 - (x - x0) ** 2)


# DELETE
def grad_2D(fieldArray):
    shapeTemp = np.shape(fieldArray)
    # diffXArray = fieldArray
    diffXArray = np.zeros((shapeTemp[0], shapeTemp[1]), dtype=complex)
    diffYArray = np.zeros((shapeTemp[0], shapeTemp[1]), dtype=complex)
    for i in range(shapeTemp[0]):
        diffXArray[i, :] = np.insert(np.diff(fieldArray[i, :]), 0, 0)
    for i in range(shapeTemp[1]):
        diffYArray[:, i] = np.insert(np.diff(fieldArray[:, i]), 0, 0)
    return diffXArray, diffYArray


"""def nonlinear_step(E):
    return (E * np.exp(1j * dz * k_0 * n2 * alp * (abs(E) *  1) ** 2)  # Kerr term with Raman contribution
            * np.exp(- 1j * dz * k_0 * gam * abs(E) ** (2 * K))
            * np.exp(-dz * beta8 / (2 * np.sqrt(K)) * (abs(E) *  1) ** (2 * (K - 1))))  # plasma defocussing"""
# plasma defocussing


# works fine!
def linear_step(field):

    temporaryField = fftshift(fftn(field))
    temporaryField = (temporaryField *
                      np.exp(-1j * dz / (2 * k_0 * n_0) * KxywMesh[0] ** 2) *
                      np.exp(-1j * dz / (2 * k_0 * n_0) * KxywMesh[1] ** 2) *
                      np.exp(1j * dz * k_bis / 2 * KxywMesh[2] ** 2))

    return ifftn(ifftshift(temporaryField))

    # return ifftn(temporaryField)"""

# %% arrays creation
xArray = np.linspace(xStart, xFinish, xResolution)
yArray = np.linspace(yStart, yFinish, yResolution)
tArray = np.linspace(tStart, tFinish, tResolution)
xMesh, tMesh = np.array(np.meshgrid(xArray, tArray))
xMeshXYT, yMeshXYT, tMeshXYT = np.array(np.meshgrid(xArray, yArray, tArray))
xtArray = np.zeros((xResolution, tResolution), dtype=complex)
xytArray = np.zeros((xResolution, yResolution, tResolution), dtype=complex)
kxArray = np.linspace(-1. * np.pi * xResolution / xFinish, 1. * np.pi * xResolution / xFinish, xResolution)
kyArray = np.linspace(-1. * np.pi * yResolution / yFinish, 1. * np.pi * yResolution / yFinish, yResolution)
wArray = np.linspace(-1. * np.pi * tResolution / tFinish, 1. * np.pi * tResolution / tFinish, tResolution)
xytMesh = np.array(np.meshgrid(xArray, yArray, tArray))
KxywMesh = np.array(np.meshgrid(kxArray, kyArray, wArray))
#phiArray = phi(xMesh, tMesh)
# phi 0-2pi
if 0:
    iTemp = 0
    for phiXTemp in phiArray:
        jTemp = 0
        for phiTTemp in phiXTemp:
            if phiTTemp < 0:
                phiArray[iTemp, jTemp] = 2 * np.pi + phiTTemp
            jTemp += 1
        iTemp += 1
if __name__ == '__main__':
    Hankel_test(5)

    """
    image = plt.imshow(np.abs(2 * np.pi * (-1j) ** lOAM
                              * np.exp(-1j * lOAM * phi(xMesh, tMesh))
                              * HankelArray(rho(xMesh, tMesh))) ** 2,
                       interpolation='bilinear', cmap='magma',
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[xArray[0], xArray[-1], tArray[0], tArray[-1]])
    plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    plt.show()
    
    image = plt.imshow(np.abs(2 * np.pi * (-1j) ** lOAM
                              * np.exp(-1j * lOAM * phi(xMesh, tMesh))
                              * H2(rho(xMesh, tMesh))) ** 2,
                       interpolation='bilinear', cmap='magma',
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[xArray[0], xArray[-1], tArray[0], tArray[-1]])
    plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    plt.show()"""
    """image = plt.imshow(np.abs(xytArray[:,int(yResolution/2),:])**2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[xArray[0], xArray[-1], tArray[0], tArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()"""

    """fig = plt.figure(figsize=(8, 7))
    image = plt.imshow(np.angle(xtArray), interpolation='bilinear', cmap='jet',
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[xArray[0], xArray[-1], tArray[0], tArray[-1]])
    plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    plt.show()
    """
    """image = plt.imshow(phi2(xMesh, tMesh), interpolation='bilinear', cmap='jet',
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[xArray[0], xArray[-1], tArray[0], tArray[-1]])
    plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    plt.show()"""

    # making a Pulse
    xtArray = (2 * np.pi * (-1j) ** lOAM
               * np.exp(-1j * lOAM * phi(xMesh, tMesh))
               * H1(rho(xMesh, tMesh))).transpose()
    """xtArray = (2 * np.pi * (-1j) ** lOAM
               * np.exp(-1j * lOAM * phi(xMesh, tMesh))
               * H1(rho(xMesh, tMesh)) +
               -1j * 2 * np.pi * (-1j) ** lOAM
               * np.exp(1j * lOAM * phi(xMesh, tMesh))
               * H1(rho(xMesh, tMesh))
               ).transpose()"""
    for i in range(yResolution):
        xytArray[:, i, :] = xtArray * y_dependence(yArray[i])

    # add lOAMs
    if 1:
        for i in range(xResolution):
            for j in range(yResolution):
                xytArray[i, j, :] = xytArray[i, j, :] * ((xArray[i] - x0) + 1j * np.sign(lOAMs) *
                                                         (yArray[j] - y0)) ** np.abs(lOAMs)

        # xt cross-section
    if 1:
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.abs(xytArray[:, :, int(tResolution / 2)]) ** 2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
        plt.close()
    image = plt.imshow(np.angle(xytArray[:, int(yResolution / 2), :]), interpolation='bilinear', cmap='jet',
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[xArray[0], xArray[-1], tArray[0], tArray[-1]])
    plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    plt.show()
    # propagation
    if 0:
        #xytArray = nonlinear_step(xytArray)
        for i in range(60):
            xytArray = linear_step(xytArray)
        # xt cross-section
    if 0:
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.abs(xytArray[:, int(yResolution / 2), :]) ** 2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
        plt.close()
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.angle(xytArray[:, int(yResolution / 2), :]),
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
        plt.close()
    # with reference
    if 0:
        timeSlice = int(tResolution / 2) - 18 + 18
        xytArray_reference = reference_pulse(xMeshXYT, yMeshXYT)
        image = plt.imshow(np.abs(xytArray_reference[:, :, timeSlice] +
                                  xytArray[:, :, timeSlice]) ** 2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
        plt.close()

    # xy cross-section
    if 0:
        fig, ax = plt.subplots(figsize=(8, 7))


        image = plt.imshow(np.abs(xytArray[:, :, int(tResolution / 2)]) ** 2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
    # flow
    if 0:
        fig, ax = plt.subplots(figsize=(8, 7))
        xVT, yVT = np.imag(1 * np.conj(xtArray.transpose()) * np.gradient(xtArray.transpose()))
        q = ax.quiver(xArray, tArray, yVT, xVT, units='xy', scale=7., zorder=5, color='green',
              width=0.007, headwidth=3., headlength=6.)
        ax.quiverkey(q, X=0.3, Y=1.1, U=10,
                     label='Quiver key, length = 10', labelpos='E')
        ax.set_aspect('equal')
        #plt.xlim(-0.5, 0.5)
        #plt.ylim(-0.5, 0.5)
        plt.show()

    """plt.plot([tArray[timeSlice], tArray[timeSlice]], [xArray[0], xArray[-1]],color='lime',lw=4)
    plt.plot([tArray[timeSlice+18 -5], tArray[timeSlice+18 -5]], [xArray[0], xArray[-1]], color='lime', lw=4)
    plt.plot([tArray[timeSlice + 18], tArray[timeSlice + 18]], [xArray[0], xArray[-1]], color='lime', lw=4)
    plt.plot([tArray[timeSlice+18 +5], tArray[timeSlice+18 +5]], [xArray[0], xArray[-1]], color='lime', lw=4)
    plt.plot([tArray[timeSlice +36], tArray[timeSlice+36]], [xArray[0], xArray[-1]], color='lime', lw=4)

    plt.show()"""

    # 3D
    if 0:
        X, Y, Z = np.mgrid[xStart:xFinish:(1j * xResolution), yStart:yFinish:(1j * yResolution),
                  tStart:tFinish:1j * tResolution]
        X, Y, Z = np.mgrid[xStart:xFinish:(1j * xResolution), xStart:xFinish:(1j * yResolution),
                  xStart:xFinish:1j * tResolution]
        # print(X)

        values = abs(xytArray) ** 2
        max = values.max()
        values = values / max * 100

        fig = go.Figure(data=go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            opacity=0.6,
            isomin=30,
            isomax=30,
            surface_count=1,  # number of isosurfaces, 2 by default: only min and max
            caps=dict(x_show=False, y_show=False)
        ))
        fig.show()

    print('Thank You! :)')
