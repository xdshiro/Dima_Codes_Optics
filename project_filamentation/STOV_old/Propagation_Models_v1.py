import numpy as np
import matplotlib.pyplot as plt
from pyhank import qdht, iqdht
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.integrate import odeint

# %% Resolutions
rResolution = 17  # N_per
tResolution = 191
loopInnerResolution, loopOuterResolution = 51, 1  # M, Kmax
zResolution = loopInnerResolution * loopOuterResolution
rStart, rFinish = 0, 30  # 10
zStart, zFinish = 0, 500
tStart, tFinish = 0, 30
r0 = 0 * (rFinish - rStart) / 2
t0 = (tFinish - tStart) / 2
"""

def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt
b = 0.25
c = 5.0
y0 = [np.pi - 0.1, 0.0]
t = np.linspace(0, 10, 1)
sol = odeint(pend, y0, t, args=(b, c))
plt.scatter(t, sol[:, 0], label='theta(t)')
plt.scatter(t, sol[:, 1], label='omega(t)')
t = np.linspace(0, 10, 111)
sol = odeint(pend, y0, t, args=(b, c))
plt.scatter(t, sol[:, 0], label='theta(t)', marker='x')
plt.scatter(t, sol[:, 1], label='omega(t)', marker='x')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
def pend2(y, t, b, c):
    return - c*np.sin(t)
b = 0.25
c = 5.0
y0 = [np.pi - 0.1]  #чему в нуле равен y
t = [0, 5]
sol = odeint(pend2, y0, t, args=(b, c))
plt.scatter(t, sol, label='theta(t)')
plt.scatter(t, sol, label='omega(t)')
t = np.linspace(0, 10, 111)
sol = odeint(pend2, y0, t, args=(b, c))
plt.scatter(t, sol, label='theta(t)', marker='x')
plt.scatter(t, sol, label='omega(t)', marker='x')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()


exit()"""
# parameters
lambda0 = 3 * 1e-1

# %% temporal
rho0 = 4
# k0 = 100
f = 150000
Eps0 = 1
tp = 1.2  # pulse duration
# kDis = 0 * 0.02  # dispersion

C = 0  # chirp
cSOL = 4
# w0 = 1
n2 = 0 * 2e-2  # 3 * chi3 / (4 * eps0 * c * n0 ** 2)
K = 1
eps0 = 1
n0 = 1.1
OAM = 0

# new parameters
k0 = 2 * np.pi / lambda0
w0 = k0 * cSOL
####
wD = 5 * w0 * 1e-1 * 1e1 / 2
kDis = 2 * n0 / cSOL / wD


def Betta_func(K):
    Betta = [0, 0 * 4e-3]
    return Betta[K]


# %% Plotting parameters
ticksFontSize = 18
legendFontSize = 18
xyLabelFontSize = 18


# %% Pulse. Initial Formula only pho
def Eps_initiation(r):
    return Eps0 * np.exp(-(r - r0) ** 2 / rho0 ** 2 - 1j * k0 * r ** 2 / (2 * f))


# with time t
def Eps_initiation_with_time(r, t):
    return Eps0 * np.exp(-1 * r ** 2 / rho0 ** 2 - 1j * k0 * r ** 2 / (2 * f) -
                         (1. + 1j * C) * (t - t0) ** 2 / tp ** 2)


# %% Pulse. Analytical formula
def Eps(r, z):
    zR = k0 * rho0 ** 2 / 2
    df = f / (1. + f ** 2 / zR ** 2)

    def w(z):
        return rho0 * np.sqrt((1 - z / f) ** 2 + z ** 2 / zR ** 2)

    def R(z):
        return z - df + df * (f - df) / (z - df)

    def Psi(z):
        return np.arctan((z - df) / np.sqrt(f * df - df ** 2))

    return Eps0 * rho0 / w(z) * np.exp(- (r - r0) ** 2 / w(z) ** 2 + 1j * k0 * (r - r0) ** 2 / (2 * R(z)) - 1j * Psi(z))


# dispersion
def Eps_dispersion(t, z):
    return Eps0 / np.sqrt(1 + 2j * kDis * z / tp ** 2) * np.exp(
        -1 * (t - t0) ** 2 / tp ** 2 / (1 + 2j * kDis * z / tp ** 2))


# ??????????????????????
def Eps_dispersion2(t, z):
    zDs = tp ** 2 / (2 * kDis)

    def T2(z):
        return tp * np.sqrt((1. + C * z / zDs) ** 2 + z ** 2 / (zDs ** 2))

    def T(z):
        return tp * np.sqrt((1. + C * z / zDs) ** 2 + 1j * (z / zDs))

    def Phi(z):
        return np.arctan(((1. + C ** 2) * z + C) / zDs)

    return (Eps0 * tp / T(z) * np.exp(-1 * (t - t0) ** 2 / T(z) ** 2 * (1 + 1j * (C + (1 + C ** 2) * z / zDs)) -
                                      1j * Phi(z)))


# %% Arrays creation
rArray = np.linspace(rStart, rFinish, rResolution)
zArray = np.linspace(zStart, zFinish, zResolution)
tArray = np.linspace(tStart, tFinish, tResolution)
# rzMesh = (np.meshgrid(rArray, zArray, indexing='ij'))
rtMesh = np.array(np.meshgrid(rArray, tArray, indexing='ij'))


def phi(x, t):
    return np.angle(x + 1j * t)


# %% Crank-Nicolson Algorithm
def Crank_Nicolson_2D(E0, loopInnerM, loopOuterKmax):
    nu = 1  # cylindrical geometry 1, planar geometry 0
    uArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        uArray[i] = 1 - nu / 2 / i
    vArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        vArray[i] = 1 + nu / 2 / i

    delta = (zArray[1] - zArray[0]) / (4 * k0 * (rArray[1] - rArray[0]) ** 2)
    # L_plus
    LPlusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dPlus2Array = np.zeros(rResolution, dtype=complex)
    dPlus2Array[0], dPlus2Array[1] = 1 - 4j * delta, 4j * delta

    LPlusMatrix[0, :] = dPlus2Array
    for i in range(1, rResolution - 1):
        LPlusMatrix[i, i - 1] = 1j * delta * uArray[i]
        LPlusMatrix[i, i] = 1 - 2j * delta
        LPlusMatrix[i, i + 1] = 1j * delta * vArray[i]
    # L_minus
    LMinusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dMinus2Array = np.zeros(rResolution, dtype=complex)
    dMinus2Array[0], dMinus2Array[1] = 1 + 4j * delta, -4j * delta
    LMinusMatrix[0, :] = dMinus2Array
    LMinusMatrix[-1, -1] = 1
    for i in range(1, rResolution - 1):
        LMinusMatrix[i, i - 1] = -1j * delta * uArray[i]
        LMinusMatrix[i, i] = 1 + 2j * delta
        LMinusMatrix[i, i + 1] = -1j * delta * vArray[i]
    LMatrix = np.dot(np.linalg.inv(LMinusMatrix), LPlusMatrix)
    E = E0
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # n = k * loopInnerM + m + 1
            # Perform Costless Diagnostic
            E = np.dot(LMatrix, E)
            # print(np.sum(np.abs(E) ** 2))
    return E


# %% Alternate direction implicit (ADI) scheme + dispersion - N
def ADI_2D1(E0, loopInnerM, loopOuterKmax):
    nu = 1  # cylindrical geometry 1, planar geometry 0
    uArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        uArray[i] = 1 - nu / 2 / i
    vArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        vArray[i] = 1 + nu / 2 / i

    delta = (zArray[1] - zArray[0]) / (4 * k0 * (rArray[1] - rArray[0]) ** 2)
    # L_plus
    LPlusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dPlus2Array = np.zeros(rResolution, dtype=complex)
    dPlus2Array[0], dPlus2Array[1] = 1 - 4j * delta, 4j * delta

    LPlusMatrix[0, :] = dPlus2Array

    for i in range(1, rResolution - 1):
        LPlusMatrix[i, i - 1] = 1j * delta * uArray[i]
        LPlusMatrix[i, i] = 1 - 2j * delta
        LPlusMatrix[i, i + 1] = 1j * delta * vArray[i]
    # L_minus
    LMinusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dMinus2Array = np.zeros(rResolution, dtype=complex)
    dMinus2Array[0], dMinus2Array[1] = 1 + 4j * delta, -4j * delta
    LMinusMatrix[0, :] = dMinus2Array
    LMinusMatrix[-1, -1] = 1
    for i in range(1, rResolution - 1):
        LMinusMatrix[i, i - 1] = -1j * delta * uArray[i]
        LMinusMatrix[i, i] = 1 + 2j * delta
        LMinusMatrix[i, i + 1] = -1j * delta * vArray[i]
    # LMatrix = np.dot(np.linalg.inv(LMinusMatrix), LPlusMatrix)
    # L d plus
    deltaD = -1 * (zArray[1] - zArray[0]) * kDis / (4 * (tArray[1] - tArray[0]) ** 2)
    print(deltaD)
    LPlusMatrixD = np.zeros((tResolution, tResolution), dtype=complex)
    dPlus2ArrayD = np.zeros(tResolution, dtype=complex)
    dPlus2ArrayD[0], dPlus2ArrayD[1] = 1 - 4j * deltaD, 4j * deltaD
    # dPlus2ArrayD[0], dPlus2ArrayD[1] = 1 + 4j * deltaD, -4j * deltaD
    # dPlus2ArrayD[0], dPlus2ArrayD[1] = 0, 0
    dPlus2ArrayD[0], dPlus2ArrayD[1] = 1, 0
    # LPlusMatrixD[-1, -1] = 1
    LPlusMatrixD[0, :] = dPlus2ArrayD
    for i in range(1, tResolution - 1):
        LPlusMatrixD[i, i - 1] = 1j * deltaD
        LPlusMatrixD[i, i] = 1 - 2j * deltaD
        LPlusMatrixD[i, i + 1] = 1j * deltaD
    # L d minus
    LMinusMatrixD = np.zeros((tResolution, tResolution), dtype=complex)
    dMinus2ArrayD = np.zeros(tResolution, dtype=complex)
    dMinus2ArrayD[0], dMinus2ArrayD[1] = 1 + 4j * deltaD, -4j * deltaD
    # dMinus2ArrayD[0], dMinus2ArrayD[1] = 1 - 4j * deltaD, +4j * deltaD
    # dMinus2ArrayD[0], dMinus2ArrayD[1] = 1, 0
    # dMinus2ArrayD[0], dMinus2ArrayD[1] = 0.001, 0
    LMinusMatrixD[0, :] = dMinus2ArrayD
    LMinusMatrixD[-1, -1] = 1
    for i in range(1, tResolution - 1):
        LMinusMatrixD[i, i - 1] = -1j * deltaD
        LMinusMatrixD[i, i] = 1 + 2j * deltaD
        LMinusMatrixD[i, i + 1] = -1j * deltaD
    # print(deltaD)
    # print (LMinusMatrixD)
    # print(LPlusMatrixD)

    LMinusMatrixD = np.linalg.inv(LMinusMatrixD)
    LMinusMatrix = np.linalg.inv(LMinusMatrix)
    E = E0.transpose()
    # LMatrix = np.dot(LMinusMatrixD, LPlusMatrixD)

    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # n = k * loopInnerM + m
            # Perform Costless Diagnostic
            # E = np.dot(LMatrix, E)

            E = np.dot(LPlusMatrixD, E).transpose()
            E = np.dot(LMinusMatrix, E)
            E = np.dot(LPlusMatrix, E).transpose()
            E = np.dot(LMinusMatrixD, E)

            # print(abs(E).max())

            # print(np.sum(np.abs(E) ** 2))
    return E.transpose()


# %% Alternate direction implicit (ADI) scheme + dispersion - N
def ADI_2D1_nonlinear(E0, loopInnerM, loopOuterKmax):
    nu = 1  # cylindrical geometry 1, planar geometry 0

    uArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        uArray[i] = 1 - nu / 2 / i
    vArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        vArray[i] = 1 + nu / 2 / i

    delta = (zArray[1] - zArray[0]) / (4 * n0 * k0 * (rArray[1] - rArray[0]) ** 2)
    # L_plus
    LPlusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dPlus2Array = np.zeros(rResolution, dtype=complex)
    dPlus2Array[0], dPlus2Array[1] = 1 - 4j * delta, 4j * delta

    LPlusMatrix[0, :] = dPlus2Array

    for i in range(1, rResolution - 1):
        LPlusMatrix[i, i - 1] = 1j * delta * uArray[i]
        LPlusMatrix[i, i] = 1 - 2j * delta
        LPlusMatrix[i, i + 1] = 1j * delta * vArray[i]
    # L_minus
    LMinusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dMinus2Array = np.zeros(rResolution, dtype=complex)
    dMinus2Array[0], dMinus2Array[1] = 1 + 4j * delta, -4j * delta
    LMinusMatrix[0, :] = dMinus2Array
    LMinusMatrix[-1, -1] = 1
    for i in range(1, rResolution - 1):
        LMinusMatrix[i, i - 1] = -1j * delta * uArray[i]
        LMinusMatrix[i, i] = 1 + 2j * delta
        LMinusMatrix[i, i + 1] = -1j * delta * vArray[i]
    # LMatrix = np.dot(np.linalg.inv(LMinusMatrix), LPlusMatrix)
    # L d plus
    deltaD = -1 * (zArray[1] - zArray[0]) * kDis / (4 * (tArray[1] - tArray[0]) ** 2)
    print(deltaD)
    LPlusMatrixD = np.zeros((tResolution, tResolution), dtype=complex)
    dPlus2ArrayD = np.zeros(tResolution, dtype=complex)
    dPlus2ArrayD[0], dPlus2ArrayD[1] = 1 - 4j * deltaD, 4j * deltaD
    # dPlus2ArrayD[0], dPlus2ArrayD[1] = 1 + 4j * deltaD, -4j * deltaD
    # dPlus2ArrayD[0], dPlus2ArrayD[1] = 0, 0
    dPlus2ArrayD[0], dPlus2ArrayD[1] = 1, 0
    # LPlusMatrixD[-1, -1] = 1
    LPlusMatrixD[0, :] = dPlus2ArrayD
    for i in range(1, tResolution - 1):
        LPlusMatrixD[i, i - 1] = 1j * deltaD
        LPlusMatrixD[i, i] = 1 - 2j * deltaD
        LPlusMatrixD[i, i + 1] = 1j * deltaD
    # L d minus
    LMinusMatrixD = np.zeros((tResolution, tResolution), dtype=complex)
    dMinus2ArrayD = np.zeros(tResolution, dtype=complex)
    dMinus2ArrayD[0], dMinus2ArrayD[1] = 1 + 4j * deltaD, -4j * deltaD
    # dMinus2ArrayD[0], dMinus2ArrayD[1] = 1 - 4j * deltaD, +4j * deltaD
    # dMinus2ArrayD[0], dMinus2ArrayD[1] = 1, 0
    # dMinus2ArrayD[0], dMinus2ArrayD[1] = 0.001, 0
    LMinusMatrixD[0, :] = dMinus2ArrayD
    LMinusMatrixD[-1, -1] = 1
    for i in range(1, tResolution - 1):
        LMinusMatrixD[i, i - 1] = -1j * deltaD
        LMinusMatrixD[i, i] = 1 + 2j * deltaD
        LMinusMatrixD[i, i + 1] = -1j * deltaD
    # print(deltaD)
    # print (LMinusMatrixD)
    # print(LPlusMatrixD)

    LMinusMatrixD = np.linalg.inv(LMinusMatrixD)
    LMinusMatrix = np.linalg.inv(LMinusMatrix)


    # LMatrix = np.dot(LMinusMatrixD, LPlusMatrixD)
    def I(E):
        eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def Nonlinearity(E):
        return 1j * w0 * n2 * E * I(E) / cSOL - Betta_func(K) * E * I(E) ** (K - 1)

    E = E0  #
    Nn_2 = (zArray[1] - zArray[0]) * Nonlinearity(E)
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # n = k * loopInnerM + m + 1
            # E = np.dot(LMatrix, E)

            Nn_1 = (zArray[1] - zArray[0]) * Nonlinearity(E)

            # E = np.dot(LPlusMatrixD, E).transpose() #
            E = np.dot(LPlusMatrixD, E.transpose())  #

            Vn_1 = np.dot(LPlusMatrix, E.transpose())

            Sn_1 = Vn_1 + (3 * Nn_1 - Nn_2) / 2

            Nn_2 = Nn_1
            # Perform Costless Diagnostic
            # E = np.dot(LMinusMatrix, Vn_1)
            E = np.dot(LMinusMatrix, Sn_1)

            E = np.dot(LMinusMatrixD, E.transpose())  # ##

            E = E.transpose()  #
            # print(np.sum(np.abs(E) ** 2))
            # E[0]=0
    return E


def ADI_2D1_nonlinearTest(E0, loopInnerM, loopOuterKmax):
    nu = 1  # cylindrical geometry 1, planar geometry 0
    uArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        uArray[i] = 1 - nu / 2 / i
    vArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        vArray[i] = 1 + nu / 2 / i

    delta = (zArray[1] - zArray[0]) / (4 * k0 * (rArray[1] - rArray[0]) ** 2)
    # L_plus
    LPlusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dPlus2Array = np.zeros(rResolution, dtype=complex)
    dPlus2Array[0], dPlus2Array[1] = 1 - 4j * delta, 4j * delta

    LPlusMatrix[0, :] = dPlus2Array

    for i in range(1, rResolution - 1):
        LPlusMatrix[i, i - 1] = 1j * delta * uArray[i]
        LPlusMatrix[i, i] = 1 - 2j * delta
        LPlusMatrix[i, i + 1] = 1j * delta * vArray[i]
    # L_minus
    LMinusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dMinus2Array = np.zeros(rResolution, dtype=complex)
    dMinus2Array[0], dMinus2Array[1] = 1 + 4j * delta, -4j * delta
    LMinusMatrix[0, :] = dMinus2Array
    LMinusMatrix[-1, -1] = 1
    for i in range(1, rResolution - 1):
        LMinusMatrix[i, i - 1] = -1j * delta * uArray[i]
        LMinusMatrix[i, i] = 1 + 2j * delta
        LMinusMatrix[i, i + 1] = -1j * delta * vArray[i]
    # LMatrix = np.dot(np.linalg.inv(LMinusMatrix), LPlusMatrix)
    # L d plus
    deltaD = -1 * (zArray[1] - zArray[0]) * kDis / (4 * (tArray[1] - tArray[0]) ** 2)
    print(deltaD)
    LPlusMatrixD = np.zeros((tResolution, tResolution), dtype=complex)
    dPlus2ArrayD = np.zeros(tResolution, dtype=complex)
    dPlus2ArrayD[0], dPlus2ArrayD[1] = 1 - 4j * deltaD, 4j * deltaD
    # dPlus2ArrayD[0], dPlus2ArrayD[1] = 1 + 4j * deltaD, -4j * deltaD
    # dPlus2ArrayD[0], dPlus2ArrayD[1] = 0, 0
    dPlus2ArrayD[0], dPlus2ArrayD[1] = 1, 0
    # LPlusMatrixD[-1, -1] = 1
    LPlusMatrixD[0, :] = dPlus2ArrayD
    for i in range(1, tResolution - 1):
        LPlusMatrixD[i, i - 1] = 1j * deltaD
        LPlusMatrixD[i, i] = 1 - 2j * deltaD
        LPlusMatrixD[i, i + 1] = 1j * deltaD
    # L d minus
    LMinusMatrixD = np.zeros((tResolution, tResolution), dtype=complex)
    dMinus2ArrayD = np.zeros(tResolution, dtype=complex)
    dMinus2ArrayD[0], dMinus2ArrayD[1] = 1 + 4j * deltaD, -4j * deltaD
    # dMinus2ArrayD[0], dMinus2ArrayD[1] = 1 - 4j * deltaD, +4j * deltaD
    # dMinus2ArrayD[0], dMinus2ArrayD[1] = 1, 0
    # dMinus2ArrayD[0], dMinus2ArrayD[1] = 0.001, 0
    LMinusMatrixD[0, :] = dMinus2ArrayD
    LMinusMatrixD[-1, -1] = 1
    for i in range(1, tResolution - 1):
        LMinusMatrixD[i, i - 1] = -1j * deltaD
        LMinusMatrixD[i, i] = 1 + 2j * deltaD
        LMinusMatrixD[i, i + 1] = -1j * deltaD
    # print(deltaD)
    # print (LMinusMatrixD)
    # print(LPlusMatrixD)

    LMinusMatrixD = np.linalg.inv(LMinusMatrixD)
    LMinusMatrix = np.linalg.inv(LMinusMatrix)

    def I(E):
        eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def Nonlinearity(E):
        return 1j * w0 * n2 * E * I(E) / cSOL - Betta_func(K) * E * I(E) ** (K - 1)

    # LMatrix = np.dot(LMinusMatrixD, LPlusMatrixD)
    LMinusMatrix = np.linalg.inv(LMinusMatrix)

    E = E0
    Nn_2 = (zArray[1] - zArray[0]) * Nonlinearity(E)

    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # n = k * loopInnerM + m + 1
            # E = np.dot(LMatrix, E)
            E = np.dot(LPlusMatrixD, E.transpose()).transpose()  #

            Nn_1 = (zArray[1] - zArray[0]) * Nonlinearity(E)

            Vn_1 = np.dot(LPlusMatrix, E)  ##

            Sn_1 = Vn_1 + (3 * Nn_1 - Nn_2) / 2

            Nn_2 = Nn_1
            # Perform Costless Diagnostic
            # E = np.dot(LMinusMatrix, Vn_1)
            E = np.dot(LMinusMatrix, Sn_1).transpose()

            E = np.dot(LMinusMatrixD, E).transpose()  #

            # print(np.sum(np.abs(E) ** 2))
            # E[0]=0

    return E


# %% Crank-Nicolson Nonlinear
def Crank_Nicolson_2D_nonlinear(E0, loopInnerM, loopOuterKmax):
    # page 31 ????????????????????????????
    def I(E):
        eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def Nonlinearity(E):
        return 1j * w0 * n2 * E * I(E) / cSOL - Betta_func(K) * E * I(E) ** (K - 1)

    nu = 1  # cylindrical geometry 1, planar geometry 0
    nuPhi = 0
    uArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        uArray[i] = 1 - nu / 2 / i
    vArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        vArray[i] = 1 + nu / 2 / i

    delta = (zArray[1] - zArray[0]) / (4 * k0 * (rArray[1] - rArray[0]) ** 2)
    # L_plus
    LPlusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dPlus2Array = np.zeros(rResolution, dtype=complex)
    dPlus2Array[0], dPlus2Array[1] = 1 - 4j * delta, 4j * delta
    #
    # dPlus2Array[0], dPlus2Array[1] = 1, 0
    # LPlusMatrix[1, 0] = 0
    #
    LPlusMatrix[0, :] = dPlus2Array
    for i in range(1, rResolution - 1):
        LPlusMatrix[i, i - 1] = 1j * delta * uArray[i]
        LPlusMatrix[i, i] = 1 - (2 - nuPhi * OAM ** 2 / i ** 2) * 1j * delta  # nuPhi * OAM ** 2 / i ** 2
        LPlusMatrix[i, i + 1] = 1j * delta * vArray[i]
    # L_minus
    LMinusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dMinus2Array = np.zeros(rResolution, dtype=complex)
    dMinus2Array[0], dMinus2Array[1] = 1 + 4j * delta, -4j * delta

    #
    # dMinus2Array[0], dMinus2Array[1] = 1 + 4j * delta, -4j * delta
    # LMinusMatrix[1, 0] = 0
    #
    LMinusMatrix[0, :] = dMinus2Array
    LMinusMatrix[-1, -1] = 1
    for i in range(1, rResolution - 1):
        LMinusMatrix[i, i - 1] = -1j * delta * uArray[i]
        LMinusMatrix[i, i] = 1 + (2 - nuPhi * OAM ** 2 / i ** 2) * 1j * delta
        LMinusMatrix[i, i + 1] = -1j * delta * vArray[i]
    LMinusMatrix = np.linalg.inv(LMinusMatrix)
    E = E0
    # E[0] = 0
    Nn_2 = (zArray[1] - zArray[0]) * Nonlinearity(E)
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # n = k * loopInnerM + m + 1
            # E = np.dot(LMatrix, E)
            Nn_1 = (zArray[1] - zArray[0]) * Nonlinearity(E)
            Vn_1 = np.dot(LPlusMatrix, E)
            Sn_1 = Vn_1 + (3 * Nn_1 - Nn_2) / 2
            Nn_2 = Nn_1
            # Perform Costless Diagnostic
            # E = np.dot(LMinusMatrix, Vn_1)
            E = np.dot(LMinusMatrix, Sn_1)
            # print(np.sum(np.abs(E) ** 2))
            # E[0]=0
    return E


# %% spectral extended # Crank-Nicolson scheme
def Crank_Nicolson_2D_nonlinear_extended(E0, loopInnerM, loopOuterKmax):
    # page 31 ????????????????????????????
    def I(E):
        eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def Nonlinearity(E):
        return 1j * w0 * n2 * E * I(E) / cSOL - Betta_func(K) * E * I(E) ** (K - 1)

    nu = 1  # cylindrical geometry 1, planar geometry 0
    uArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        uArray[i] = 1 - nu / 2 / i
    vArray = np.zeros(rResolution, dtype=complex)
    for i in range(1, rResolution - 1):
        vArray[i] = 1 + nu / 2 / i

    delta = (zArray[1] - zArray[0]) / (4 * k0 * (rArray[1] - rArray[0]) ** 2)
    # L_plus
    LPlusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dPlus2Array = np.zeros(rResolution, dtype=complex)
    dPlus2Array[0], dPlus2Array[1] = 1 - 4j * delta, 4j * delta

    LPlusMatrix[0, :] = dPlus2Array
    for i in range(1, rResolution - 1):
        LPlusMatrix[i, i - 1] = 1j * delta * uArray[i]
        LPlusMatrix[i, i] = 1 - 2j * delta
        LPlusMatrix[i, i + 1] = 1j * delta * vArray[i]
    # L_minus
    LMinusMatrix = np.zeros((rResolution, rResolution), dtype=complex)
    dMinus2Array = np.zeros(rResolution, dtype=complex)
    dMinus2Array[0], dMinus2Array[1] = 1 + 4j * delta, -4j * delta
    LMinusMatrix[0, :] = dMinus2Array
    LMinusMatrix[-1, -1] = 1
    for i in range(1, rResolution - 1):
        LMinusMatrix[i, i - 1] = -1j * delta * uArray[i]
        LMinusMatrix[i, i] = 1 + 2j * delta
        LMinusMatrix[i, i + 1] = -1j * delta * vArray[i]
    LMinusMatrix = np.linalg.inv(LMinusMatrix)
    E = E0
    Nn_2 = (zArray[1] - zArray[0]) * Nonlinearity(E)
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # n = k * loopInnerM + m + 1
            # E = np.dot(LMatrix, E)
            Nn_1 = (zArray[1] - zArray[0]) * Nonlinearity(E)
            Vn_1 = np.dot(LPlusMatrix, E)
            Sn_1 = Vn_1 + (3 * Nn_1 - Nn_2) / 2
            Nn_2 = Nn_1
            # Perform Costless Diagnostic
            # E = np.dot(LMinusMatrix, Vn_1)
            E = np.dot(LMinusMatrix, Sn_1)
            # print(np.sum(np.abs(E) ** 2))
    return E


def Spectral_technique_Cylindrical(E0, loopInnerM, loopOuterKmax):
    kArray, HTArray = qdht(rArray, E0)
    delta = (zArray[1] - zArray[0]) / (4 * k0 * (rArray[1] - rArray[0]) ** 2)
    AArray = np.exp(-2j * delta * (rArray * kArray) ** 2)
    E = E0

    for k in range(loopOuterKmax):
        for m in range(loopInnerM):
            # n = k * loopInnerM + m + 1
            # Perform Costless Diagnostic
            Espec = qdht(rArray, E)[1] * AArray
            # Espec = np.dot(Espec, AArray)
            E = iqdht(kArray, Espec)[1]
            # print(np.sum(np.abs(E) ** 2))

    return E


def Spectral_technique_Dima(E0, loopInnerM, loopOuterKmax):
    rResolution2 = rResolution * 2
    rArray2 = np.linspace(0, rFinish * 2, rResolution2)
    krArray2 = np.linspace(-1. * np.pi * rResolution2 / rFinish / 2, 1. * np.pi * rResolution2 / rFinish / 2,
                           rResolution2)

    def initiation(r):
        return E0 * np.exp(- (r - rFinish) ** 2 / rho0 ** 2 - 1j * k0 * (r - rFinish) ** 2 / (2 * f))

    E = initiation(rArray2)
    dz = zArray[1] - zArray[0]
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # n = k * loopInnerM + m + 1
            # Perform Costless Diagnostic
            Espec = fftshift(fftn(E)) * np.exp(-1j * dz / (2 * k0) * (krArray2) ** 2)
            E = ifftn(ifftshift(Espec))
    return E[rResolution: rResolution2]


#  3D
def split_step_old(E0=1, loopInnerM=1, loopOuterKmax=1):
    rResolution2 = rResolution * 2 - 1
    xArray = np.linspace(0, rFinish * 2, rResolution2)

    yArray = np.linspace(0, rFinish * 2, rResolution2)
    xyMesh = np.array(np.meshgrid(xArray, yArray, indexing='ij'))
    kxArray = np.linspace(-1. * np.pi * (rResolution2 - 2) / (rFinish * 2),
                          1. * np.pi * (rResolution2 - 2) / (rFinish * 2), rResolution2)
    """print(kxArray[0],kxArray[-1],kxArray[rResolution])
    print(kxArray[1] - kxArray[0])
    delta = 2 * np.pi / (rResolution2 * (xArray[1] - xArray[0]))
    print(delta * (rResolution2 / 2))
    #print(delta * )
    exit()"""
    kyArray = np.linspace(-1. * np.pi * (rResolution2 - 2) / (rFinish * 2),
                          1. * np.pi * (rResolution2 - 2) / (rFinish * 2), rResolution2)
    KxyMesh = np.array(np.meshgrid(kxArray, kyArray, indexing='ij'))  # np.array

    def radius(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def initiation(x, y):
        return (E0 * np.exp(- (radius(x - rFinish, y - rFinish) ** 2) / rho0 ** 2 -
                            1j * k0 * radius(x - rFinish, y - rFinish) ** 2 / (2 * f))
                * np.exp(1j * OAM * phi(x - rFinish, y - rFinish)))

    def I(E):
        eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    dz = zArray[1] - zArray[0]

    def Nonlinearity_spec(E):
        return dz * 1j * w0 * n2 * I(E) / cSOL - dz * Betta_func(K) * I(E) ** (K - 1)  # & E

    E = initiation(xyMesh[0], xyMesh[1])
    print(E[rResolution - 1, rResolution - 1])

    def linear_step(field):
        temporaryField = fftshift(fftn(field))
        temporaryField = (temporaryField *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[0] ** 2) *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[1] ** 2))  # добавлено
        return ifftn(ifftshift(temporaryField))

    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            E = E * np.exp(Nonlinearity_spec(E))
            E = linear_step(E)

    return E[rResolution - 1:rResolution2, rResolution - 1], E


# 3D+1 Time # FIX LATER
def split_step_old_time(E0=1, loopInnerM=1, loopOuterKmax=1):
    rResolution2 = rResolution * 2 - 1
    xArray = np.linspace(0, rFinish * 2, rResolution2)

    yArray = np.linspace(0, rFinish * 2, rResolution2)
    xyMesh = np.array(np.meshgrid(xArray, yArray, indexing='ij'))
    kxArray = np.linspace(-1. * np.pi * (rResolution2 - 2) / (rFinish * 2),
                          1. * np.pi * (rResolution2 - 2) / (rFinish * 2), rResolution2)

    kyArray = np.linspace(-1. * np.pi * (rResolution2 - 2) / (rFinish * 2),
                          1. * np.pi * (rResolution2 - 2) / (rFinish * 2), rResolution2)
    wArray = np.linspace(-1. * np.pi * (tResolution - 2) / tFinish, 1. * np.pi * (tResolution - 2) / tFinish,
                         tResolution)

    xytMesh = np.array(np.meshgrid(xArray, yArray, tArray, indexing='ij'))
    KxywMesh = np.array(np.meshgrid(kxArray, kyArray, wArray, indexing='ij'))

    # !!!!!!!!!!!!!! delti
    """
    dx = xArray[1] - xArray[0]
    X = dx * (rResolution2)
    dw = 2 * np.pi / X
    dwMy = kxArray[1] - kxArray[0]
    print (dw, dwMy)
    exit()"""

    def radius(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def initiation(x, y, t):
        return (E0 * np.exp(- (radius(x - rFinish, y - rFinish) ** 2) / rho0 ** 2 -
                            1j * k0 * radius(x - rFinish, y - rFinish) ** 2 / (2 * f) -
                            ((t - t0) / tp) ** 2)
                * np.exp(1j * OAM * phi(x - rFinish, y - rFinish)))

    def I(E):
        eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    dz = zArray[1] - zArray[0]

    def Nonlinearity_spec(E):
        return dz * 1j * w0 * n2 * I(E) / cSOL - dz * Betta_func(K) * I(E) ** (K - 1)  # & E

    E = initiation(xytMesh[0], xytMesh[1], xytMesh[2])
    print(E[rResolution - 1, rResolution - 1, int(tResolution / 2)])

    # works fine!
    def linear_step(field):
        temporaryField = fftshift(fftn(field))
        temporaryField = (temporaryField *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxywMesh[0] ** 2) *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxywMesh[1] ** 2) *
                          np.exp(1j * dz * kDis / 2 * KxywMesh[2] ** 2))  # something here in /2
        return ifftn(ifftshift(temporaryField))

    print(KxywMesh[2][rResolution - 1, rResolution - 1, int(tResolution / 2)])
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            E = linear_step(E)
            E = E * np.exp(Nonlinearity_spec(E))

    return E[rResolution - 1, rResolution - 1, :], E  # rResolution - 1:rResolution2


# %% UPPE as a large system of ordinary differential equations
def UPPE(E0, loopInnerM, loopOuterKmax):
    rResolution2 = rResolution * 2 - 1
    xArray = np.linspace(0, rFinish * 2, rResolution2)

    yArray = np.linspace(0, rFinish * 2, rResolution2)
    xyMesh = np.array(np.meshgrid(xArray, yArray, indexing='ij'))
    kxArray = np.linspace(-1. * np.pi * (rResolution2 - 2) / (rFinish * 2),
                          1. * np.pi * (rResolution2 - 2) / (rFinish * 2), rResolution2)
    kyArray = np.linspace(-1. * np.pi * (rResolution2 - 2) / (rFinish * 2),
                          1. * np.pi * (rResolution2 - 2) / (rFinish * 2), rResolution2)
    KxyMesh = np.array(np.meshgrid(kxArray, kyArray, indexing='ij'))

    def radius(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def initiation(x, y):
        return (E0 * np.exp(- (radius(x - rFinish, y - rFinish) ** 2) / rho0 ** 2 -
                            1j * k0 * radius(x - rFinish, y - rFinish) ** 2 / (2 * f))
                * np.exp(1j * OAM * phi(x - rFinish, y - rFinish)))

    def I(E):
        eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def Nonlinearity(E):
        return 1j * w0 * n2 * E * I(E) / cSOL - Betta_func(K) * E * I(E) ** (K - 1)

    ##########
    def A_right(y, z, P, w):
        return P

    def ODE(P, A0):

        A = np.zeros(np.shape(A0), dtype=complex)
        # print(np.shape(A))
        # exit()
        # print(A[10, 10])
        for i in range(rFinish * 2):
            for j in range(rFinish * 2):
                # print(A[i,j])
                # odeint(pend2, y0, t, args=(b, c))
                A[i, j] = (odeint(A_right, np.real(A0[i, j]), [0, dz], args=(np.real(P[i, j]), 0))[1] +
                           1j * odeint(A_right, np.imag(A0[i, j]), [0, dz], args=(np.imag(P[i, j]), 0))[1])
                # print(A0[i, j])
                # print(A[i, j])
                print(A0[i, j] - A[i, j])
                # exit()
                # print()

        # print(A - A0)
        # print(A[10, 10])
        return A

    """def pend2(y, t, b, c):
        return - c * np.sin(t)

    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1]  # чему в нуле равен y
    t = [0, 5]
    sol = odeint(pend2, y0, t, args=(b, c))
    """

    #########
    E = initiation(xyMesh[0], xyMesh[1])
    Espec = fftshift(fftn(E))
    Aspec = Espec  # / np.exp(1j * np.sqrt(k0**2 - kxArray[0] ** 2 - kxArray[1] ** 2))
    dz = zArray[1] - zArray[0]
    kz = np.sqrt(k0 ** 2 - KxyMesh[0] ** 2 - KxyMesh[1] ** 2)
    P = Nonlinearity(E)
    Pspec = ifftn(ifftshift(P))
    # temporal derivative
    vPhase = w0 / k0
    # A0 = [Aspec]
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # чему в нуле равен y
            # print(Aspec[10, 10])
            Aspec = ODE(Pspec, Aspec)

            Aspec *= np.exp(1j * dz * (kz - w0 / vPhase))  # vPhase
            E = ifftn(ifftshift(Aspec))
            P = Nonlinearity(E)
            Pspec = ifftn(ifftshift(P))
            Pspec *= np.exp(-1j * dz * (kz - w0 / vPhase))
            Pspec *= 1j * w0 ** 2 / 2 / eps0 / cSOL ** 2 / kz

    E = ifftn(ifftshift(Aspec))
    return E[rResolution - 1:rResolution2, rResolution - 1], E


# %% UPPE with time
def UPPE_time(E0, loopInnerM, loopOuterKmax):
    rResolution2 = rResolution * 2 - 1
    xArray = np.linspace(0, rFinish * 2, rResolution2)

    yArray = np.linspace(0, rFinish * 2, rResolution2)
    # xyMesh = np.array(np.meshgrid(xArray, yArray, indexing='ij'))
    kxArray = np.linspace(-1. * np.pi * (rResolution2 - 2) / (rFinish * 2),
                          1. * np.pi * (rResolution2 - 2) / (rFinish * 2), rResolution2)
    kyArray = np.linspace(-1. * np.pi * (rResolution2 - 2) / (rFinish * 2),
                          1. * np.pi * (rResolution2 - 2) / (rFinish * 2), rResolution2)
    # KxyMesh = np.array(np.meshgrid(kxArray, kyArray, indexing='ij'))

    wArray = np.linspace(-1. * np.pi * (tResolution - 2) / tFinish, 1. * np.pi * (tResolution - 2) / tFinish,
                         tResolution)

    xytMesh = np.array(np.meshgrid(xArray, yArray, tArray, indexing='ij'))
    KxywMesh = np.array(np.meshgrid(kxArray, kyArray, wArray, indexing='ij'))

    def radius(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def initiation(x, y, t):
        return (E0 * np.exp(- (radius(x - rFinish, y - rFinish) ** 2) / rho0 ** 2 -
                            1j * k0 * radius(x - rFinish, y - rFinish) ** 2 / (2 * f) -
                            ((t - t0) / tp) ** 2)
                * np.exp(1j * OAM * phi(x - rFinish, y - rFinish)))

    def I(E):
        eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def Nonlinearity(E):
        return 1j * w0 * n2 * E * I(E) / cSOL - Betta_func(K) * E * I(E) ** (K - 1)

    ##########
    def A_right(y, z, P, w):
        return P

    def ODE(P, A0):

        A = np.zeros(np.shape(A0), dtype=complex)
        # print(np.shape(A))
        # exit()
        # print(A[10, 10])
        for i in range(rFinish * 2):
            for j in range(rFinish * 2):
                # print(A[i,j])
                # odeint(pend2, y0, t, args=(b, c))
                A[i, j] = (odeint(A_right, np.real(A0[i, j]), [0, dz], args=(np.real(P[i, j]), 0))[1] +
                           1j * odeint(A_right, np.imag(A0[i, j]), [0, dz], args=(np.imag(P[i, j]), 0))[1])
                # print(A0[i, j])
                # print(A[i, j])
                print(A0[i, j] - A[i, j])
                # exit()
                # print()

        # print(A - A0)
        # print(A[10, 10])
        return A

    """def pend2(y, t, b, c):
        return - c * np.sin(t)

    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1]  # чему в нуле равен y
    t = [0, 5]
    sol = odeint(pend2, y0, t, args=(b, c))
    """

    #########
    E = initiation(xytMesh[0], xytMesh[1], xytMesh[2])
    Espec = fftshift(fftn(E))
    """fig = plt.figure(figsize=(8, 7))
    plt.plot(kxArray, np.abs(Espec[:, rResolution -1, int(tResolution/2)]))
    plt.show()
    print(k0)"""

    Aspec = Espec  # / np.exp(1j * np.sqrt(k0**2 - kxArray[0] ** 2 - kxArray[1] ** 2))
    dz = zArray[1] - zArray[0]
    ############################
    n = n0 * (1. + (w0 + KxywMesh[2]) / wD)
    k = n * (w0 + KxywMesh[2]) / cSOL
    print(w0, w0 + KxywMesh[2].max())

    kz = np.sqrt(k ** 2 - KxywMesh[0] ** 2 - KxywMesh[1] ** 2)
    # print(k)
    # print(KxywMesh[0])
    # exit()
    P = Nonlinearity(E)
    Pspec = ifftn(ifftshift(P))
    # temporal derivative
    vPhase = w0 / k0 / 2
    vGroup = cSOL / (n0 + 2 * n0 * w0 / wD)
    print(vPhase)
    # exit()
    # print(vPhase)
    # exit()
    # A0 = [Aspec]
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # чему в нуле равен y
            # print(Aspec[10, 10])
            # Aspec = ODE(Pspec, Aspec)
            # Espec = fftshift(fftn(E))

            # Aspec *= np.exp(1j * dz * (kz)) #vPhase
            Aspec *= np.exp(1j * dz * (kz - (w0 + KxywMesh[2]) / vGroup))  # vPhase

            # E = ifftn(ifftshift(Aspec))
            # P = Nonlinearity(E)
            # Pspec = ifftn(ifftshift(P))
            # Pspec *= np.exp(-1j * dz * (kz - w0 / vPhase))
            # Pspec *= 1j * w0 ** 2 / 2 /eps0 / cSOL ** 2 / kz

    E = ifftn(ifftshift(Aspec))
    return E[rResolution - 1:rResolution2, rResolution - 1, int(tResolution / 2)], E
    # return E[rResolution - 1, rResolution - 1, :], E


def plot_1D(x, y, label='', xname='', yname='', ls='-', lw=4, color='rrr'):
    if color == 'rrr':
        ax.plot(x, y, ls=ls, label=label, lw=lw)
    else:
        ax.plot(x, y, ls=ls, label=label, lw=lw, color=color)
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontname='Times New Roman', fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize)
    ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    ax.legend(shadow=True, fontsize=legendFontSize, facecolor='white', edgecolor='black', loc='upper right')


if __name__ == '__main__':
    # no nonlinearity
    if 0:
        # plt.plot(rArray, np.abs(Eps_initiation(rArray)))
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(rArray, np.abs(Eps(rArray, z=zFinish)), ls='-', label='Analytical', lw=4)
        plot_1D(rArray, np.abs(Crank_Nicolson_2D(Eps_initiation(rArray), loopInnerResolution, loopOuterResolution))
                , label='Crank Nicolson', xname=r'$\rho$', yname='', ls='--', lw=8)
        fig, ax = plt.subplots(figsize=(8, 7))
        # plot_1D(rArray, np.abs(Spectral_technique_Cylindrical(Eps_initiation(rArray), loopInnerResolution, loopOuterResolution))
        #        , label='Hankel', xname=r'$\rho$', yname='')
        plot_1D(rArray,
                np.abs(Spectral_technique_Dima(Eps_initiation(rArray), loopInnerResolution, loopOuterResolution))
                , label='Fourier', xname=r'$\rho$', yname='', ls='--')
        plt.show()

    # dispersion
    if 1:
        fig, ax = plt.subplots(figsize=(8, 7))
        # ax.plot(rArray, np.abs(Eps(rArray, z=zFinish)), color='lime', ls='-', label='Analytical', lw=4)

        # ax.plot(tArray, np.abs(Eps_dispersion(tArray, z=zFinish)), ls='-', label='Analytical', color='black', lw=10)
        fieldAdi = ADI_2D1_nonlinear(Eps_initiation_with_time(rtMesh[0], rtMesh[1]), loopInnerResolution,
                                     loopOuterResolution)
        ax.plot(tArray, np.abs(fieldAdi)[0, :], ls='-', label='ADI Crank Nicolson', color='red', lw=8)
        """fieldAdiNonlinear = ADI_2D1_nonlinear(Eps_initiation_with_time(rtMesh[0], rtMesh[1]), loopInnerResolution,
                                       loopOuterResolution)
        plot_1D(tArray, np.abs(fieldAdiNonlinear)[0, :]
                , label='Crank Nicolson Dispersion', xname=r'$t$', yname='', ls='-', color='red', lw=6)"""
        fieldOLD1d, fieldOLD = split_step_old_time(1, loopInnerResolution, loopOuterResolution)
        plot_1D(tArray, np.abs(fieldOLD1d)
                , label='Fourier_everything', xname=r'$t$', yname='', color='lime', ls='-', lw=4)
        fieldUPPE1d, fieldUPPE = UPPE_time(1, loopInnerResolution, loopOuterResolution)
        plot_1D(tArray, np.abs(fieldUPPE[rResolution - 1, rResolution - 1, :])
                , label='UPPE', xname=r'$t$', yname='', color='blue', ls='-', lw=2)

        # plt.xlim(15,30.5)
        plt.show()
        fig, ax = plt.subplots(figsize=(8, 7))
        plt.plot(rArray, np.abs(fieldAdi[:, int(tResolution/2)]), color='red', lw=8, label='ADI Crank Nicolson')
        plt.plot(rArray, np.abs(fieldOLD[rResolution - 1:, rResolution - 1, int(tResolution/2)]),
                 color='lime', ls='-', lw=4, label='Fourier_everything')
        plot_1D(rArray, np.abs(fieldUPPE[rResolution - 1:, rResolution - 1, int(tResolution/2)]),
                label='UPPE', xname=r'$\rho$', yname='', color='blue', ls='-', lw=2)
        plt.show()
        print(fieldAdi[:, 0])
        if 0:
            fig, ax = plt.subplots(figsize=(8, 7))

            plot_1D(rArray, np.abs(fieldAdi)[:, int(tResolution / 2)]
                    , label='ADI Crank Nicolson', xname=r'$t$', yname='', ls='-', color='red', lw=8)
            plot_1D(rArray, np.abs(fieldOLD)[rResolution - 1:2 * rResolution - 1, rResolution - 1, int(tResolution / 2)]
                    , label='Fourier_everything', xname=r'$\rho$', yname='', color='lime', ls='-', lw=4)
            # plt.xlim(10,20.5)
            plt.show()

    # nonlinearity
    if 0:
        # plt.plot(rArray, np.abs(Eps_initiation(rArray)))
        fig, ax = plt.subplots(figsize=(8, 7))

        # ax.plot(rArray, np.abs(Eps(rArray, z=zFinish)), ls='-', label='Analytical_Linear', lw=8, color='red')

        rArray2 = np.linspace(0, rFinish * 2, rResolution * 2)
        fieldOLD1d, fieldOLD = split_step_old(1, loopInnerResolution, loopOuterResolution)
        fieldOLD1dTime, fieldOLDTime = split_step_old_time(1, loopInnerResolution, loopOuterResolution)
        plot_1D(rArray, np.abs(fieldOLD1d)
                , label='Fourier', xname=r'$\rho$', yname='', color='black', ls='-', lw=10)
        plot_1D(rArray, np.abs(fieldOLDTime)[rResolution - 1:, rResolution - 1, int(tResolution / 2)]
                , label='Fourier_time', xname=r'$\rho$', yname='', color='red', ls='-', lw=4)

        fieldCN = np.abs(Crank_Nicolson_2D_nonlinear(Eps_initiation(rArray),
                                                     loopInnerResolution, loopOuterResolution))
        plot_1D(rArray, fieldCN
                , label='Crank Nicolson Nonlinear', xname=r'$\rho$', yname='', ls='--', color='lime', lw=6)
        fieldUPPE1d, fieldUPPE = UPPE_time(1, loopInnerResolution, loopOuterResolution)
        plot_1D(rArray, np.abs(fieldUPPE1d)
                , label='UPPE', xname=r'$\rho$', yname='', color='blue', ls='-', lw=2)

        plt.show()

        # 2D
    if 0:
        fieldCN = np.abs(Crank_Nicolson_2D_nonlinear(Eps_initiation(rArray),
                                                     loopInnerResolution, loopOuterResolution))
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, polar=True)
        thetaResolution = 60
        # field = np.zeros((thetaResolution, rResolution))
        theta = np.linspace(0, 2 * np.pi, thetaResolution)
        field = np.array([fieldCN for j in range(thetaResolution)]).transpose()

        pc = ax.pcolormesh(theta, rArray, field, cmap='magma')

        fig.colorbar(pc)

        # ax.set_theta_zero_location('N')
        # ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], color='red')
        plt.show()
        fieldOLD1d, fieldOLD = split_step_old(1, loopInnerResolution, loopOuterResolution)
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.abs(fieldOLD),
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[-rFinish, rFinish, -rFinish, rFinish])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)

        plt.show()
    # everything
    if 0:
        fig = plt.figure(figsize=(8, 7))
        # fieldOLD1d, fieldOLD = split_step_old_time(1, loopInnerResolution, loopOuterResolution)
        # fieldOLD1d, fieldOLD = split_step_old(1, loopInnerResolution, loopOuterResolution)

        image = plt.imshow((np.abs(fieldOLD)[rResolution - 1:2 * rResolution - 1, rResolution - 1, :]),
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[0, tFinish, 0, rFinish])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.xlim(tFinish / 2, tFinish - 5)
        plt.ylim(0, 5)

        plt.show()
        fig = plt.figure(figsize=(8, 7))

        image = plt.imshow(np.abs(fieldAdi),
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[0, tFinish, 0, rFinish])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.xlim(tFinish / 2, tFinish - 5)
        plt.ylim(0, 5)
        plt.show()
    exit()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = plt.imshow(np.abs(Eps(rMesh, zMesh)) ** 2,
                       interpolation='bilinear', cmap='magma',
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[zArray[0], zArray[-1], rArray[0], rArray[-1]])
    plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    plt.close()
    # plt.show()
