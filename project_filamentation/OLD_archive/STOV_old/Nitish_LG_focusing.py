import numpy as np
import matplotlib.pyplot as plt
from pyhank import qdht, iqdht
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.integrate import odeint, complex_ode, ode
from scipy.special import jv, iv, assoc_laguerre
import scipy.integrate as integrate
# %% Modules
# checking spectra
module_CheckingSpectrum = 1
module_2D = 1
module_1D = 0
module_Paraxial = 0
module_NonRapaxial = 1
module_Phase = 0
module_PlottingInBetween = 0
data_save = 1





# %% Resolutions
xResolution = 17551 # N_per
yResolution = 17551
#xResolution = 851 # N_per
#yResolution = 851
fieldUPPE = np.zeros((xResolution, yResolution))

tResolution = 1
loopInnerResolution, loopOuterResolution = 2, 1  # M, Kmax
zResolution = loopInnerResolution * loopOuterResolution
xStart, xFinish = 0, 14 * 1e-3 # 10
yStart, yFinish = 0, 14 * 1e-3  # 10
zStart, zFinish = 0, 5e-2
tStart, tFinish = 0, 400 * 1e-12


"""
https://stackoverflow.com/questions/40788747/internal-working-of-scipy-integrate-ode
# ODE CHECK
def ODEs(x, t):
    a1 = 3e5
    a2 = 0.2
    a3 = 4e-7
    a4 = 0.6
    a5 = 8
    a6 = 90
    A = x[0]
    B = x[1]
    C = x[2]
    dAdt = a1 * t - a2*A - a3 * A * C
    dBdt= a3*A*C - a4*B
    dcDt = -a3 * A *C - a5* C + a6 * B
    return [dAdt, dBdt, dcDt]

x = [2e6, 0 , 90]
t = np.linspace(0,15,1000)
x = odeint(ODEs, x, t)

A=x[:, 0]
B = x[:, 1]
C = x[:, 2]
plt.semilogy(t,A)
plt.semilogy(t,B)
plt.semilogy(t,C)


def ODEsODE(t, x):
    a1 = 3e5
    a2 = 0.2
    a3 = 4e-7
    a4 = 0.6
    a5 = 8
    a6 = 90
    A = x[0]
    B = x[1]
    C = x[2]
    dAdt = a1 * t - a2 * A - a3 * A * C
    dBdt = a3 * A * C - a4 * B
    dcDt = -a3 * A * C - a5 * C + a6 * B
    return [dAdt, dBdt, dcDt]
integrator = complex_ode(ODEsODE)
x = [2e6, 0 , 90]
integrator.set_initial_value(x, 0)
t0 = 7
x = integrator.integrate(t0)
plt.scatter(t0, np.abs(x[2]))
plt.show()
exit()
"""



###
# %% Pulse parameters
rho0 = 800 * 1e-6
tp = 70 * 1e-12  # pulse duration
lambda0 = 0.571e-6
# STOV
yRadius = 80 * 1e-6
xSTOVRadius = rho0
tSTOVRadius = tp
lOAM = 0
x0 = (xFinish - xStart) / 2
y0 = (yFinish - yStart) / 2
t0 = (tFinish - tStart) / 2
f = 5e-2  # TEMPORAL ################################################################################
# linear medium parameter
wD = 5 * 1e30  # TEMPORAL ################################################################################
n0 = 1.00
# %% Nonlinear parameters
K = 1  # photons number
# %% temporal
C = 0  # chirp # TEMPORAL ################################################################################
n2 = 0*5e-1  # 3 * chi3 / (4 * eps0 * c * n0 ** 2) # TEMPORAL #######################################################
# kDis = 0 * 0.02  # dispersion
Ui = 1  # TEMPORAL ################################################################################
SigmaK = [0, 1, 2]  # TEMPORAL ################################################################################



def Betta_func(K):
    Betta = [0, 0 * 2e-0]
    return Betta[K]


# %% Constants
eps0 = 8.854187817e-12  # [F/m] - vacuum permittivity
cSOL = 2.99792458e8  # [m/s] speed of light in vacuum
# %% Plotting parameters
ticksFontSize = 18
legendFontSize = 18
xyLabelFontSize = 18
# %% parameter recalculation
k0 = 2 * np.pi / lambda0
w0 = k0 * cSOL
wD = wD * w0  # TEMPORAL ################################################################################
kDis = 2 * n0 / cSOL / wD  # TEMPORAL ################################################################################
chi3 = n2 * 4 * eps0 * cSOL * n0 ** 2 / 3
# %% Arrays creation
xArray = np.linspace(xStart, xFinish, xResolution)
yArray = np.linspace(yStart, yFinish, yResolution)
zArray = np.linspace(zStart, zFinish, zResolution)
tArray = np.linspace(tStart, tFinish, tResolution)
xtMesh = np.array(np.meshgrid(xArray, tArray, indexing='ij'))  # only ADI
kxArray = np.linspace(-1. * np.pi * (xResolution - 2) / xFinish,
                      1. * np.pi * (xResolution - 2) / xFinish, xResolution)
kyArray = np.linspace(-1. * np.pi * (yResolution - 2) / yFinish,
                      1. * np.pi * (yResolution - 2) / yFinish, yResolution)
wArray = np.linspace(-1. * np.pi * (tResolution - 2) / tFinish, 1. * np.pi * (tResolution - 2) / tFinish,
                     tResolution)
xytMesh = np.array(np.meshgrid(xArray, yArray, tArray, indexing='ij'))
KxywMesh = np.array(np.meshgrid(kxArray, kyArray, wArray, indexing='ij'))

# 5e-2
if 1:
    xArray *= 1e6
    yArray *= 1e6
    x0 *= 1e6
    y0 *= 1e6
    xy0 = 35
    #xResolution = 7551  # N_per
    #yResolution = 7551
    xResolution = 21551  # N_per
    yResolution = 21551
    data_new = np.load('dataa3.npy')
    #fig, ax = plt.subplots(figsize=(8, 7))
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_axes((0.12, 0.09, 0.85, 0.89))
    image = plt.imshow(np.abs(data_new[:, :]),
                       interpolation='bilinear', cmap='jet',
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[yArray[0] -y0, yArray[-1] -y0, xArray[0] - x0, xArray[-1] -x0])
    plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    ax.set_xlabel('y (mkm)', fontsize=xyLabelFontSize)
    ax.set_ylabel('x (mkm)', fontsize=xyLabelFontSize)
    plt.xlim(- xy0, + xy0)
    plt.ylim(- xy0, + xy0)
    plt.show()
    plt.close()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_axes((0.12, 0.09, 0.85, 0.89))
    image = plt.imshow(np.angle(data_new[:, :]),
                       interpolation='bilinear', cmap='jet',
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[yArray[0] - y0, yArray[-1] - y0, xArray[0] - x0, xArray[-1] - x0])
    plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    ax.set_xlabel('y (mkm)', fontsize=xyLabelFontSize)
    ax.set_ylabel('x (mkm)', fontsize=xyLabelFontSize)
    plt.xlim(- xy0, + xy0)
    plt.ylim(- xy0, + xy0)
    plt.show()
    plt.close()
    exit()

# initial fields
def Field_1(x, y, t):
    return (np.exp(- (radius(x - x0, y - y0) ** 2) / rho0 ** 2 -
                   1j * k0 * radius(x - x0, y - x0) ** 2 / (2 * f) -
                   ((t - t0) / tp) ** 2)
            * np.exp(1j * lOAM * phi(x - x0, y - y0)))


# STOV
def Field_STOV_1(x, y, t):
    def H1(radius):
        return (np.pi ** (3 / 2) * radius / 4 * np.exp(-(2 * np.pi * radius) ** 2 / 8) *
                (iv(np.abs(0), (2 * np.pi * radius) ** 2 / 8) -
                 iv(np.abs(1), (2 * np.pi * radius) ** 2 / 8)))

    def y_dependence(y):
        return np.exp(-(y / yRadius) ** 2)

    return (2 * np.pi * (-1j) ** lOAM
            * np.exp(-1j * lOAM * phi(x - x0, t - t0))
            * H1(radius(x - x0, t - t0))) * y_dependence(y - y0)


# STOV simple t/tw + ix/xw
def Field_STOV_simple(x, y, t):


    def y_dependence(y):
        return np.exp(-(y / yRadius) ** 2)

    def x_dependence(x):
        return np.exp(-(x / rho0) ** 2)

    def t_dependence(t):
        return np.exp(-(t / tp) ** 2)

    return (((t - t0)/tSTOVRadius + 1j * np.sign(lOAM) * (x - x0) / xSTOVRadius) ** (np.abs(lOAM)) * y_dependence(y - y0) *
            x_dependence(x - x0) * t_dependence(t - t0) *
            -
            1j * k0 * radius(x - x0, y - x0) ** 2 / (2 * f)
            )



def LG_Nitish(x, y, t):
    p = 0
    lOAM = 4

    wi = 2.5 * 1e-3
    x = x - x0
    y = y - y0
    t = t - t0

    def R(z):
        zR = f
        return f
        #return np.infty
        #return z * (1 + (zR / z) ** 2)
    #print(R(0))
    kz = np.sqrt(k0 ** 2 + KxywMesh[0] ** 2 + KxywMesh[1] ** 2)
    E=((radius(x, y) * np.sqrt(2) / wi) ** np.abs(lOAM)
         * np.exp(-1 * radius(x, y) ** 2 / wi ** 2)
         * assoc_laguerre((radius(x, y) * np.sqrt(2) / wi) ** 2, p, np.abs(lOAM))
         * np.exp(-1j * k0 * radius(x, y) ** 2 / (2 * R(0)))
         * np.exp(-1j * lOAM * phi(x, y)))
    return E



# %% general functions
def radius(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def phi(x, t):
    return np.angle(x + 1j * t)


# Couairon p57
"""def electron_density_solver():
    def Sigma(w):
        return (w0 / (n(w) * cSOL * rhoC)) * ((w0 * tauC * (1 + 1j * w * tauC)) / (1 + w ** 2 * tauC ** 2))


    def Wofi():
        SigmaK[K] * I ** (K)
        return 0

    def Wava():
        Sigma(w0) * I / Ui
        return()


    return 0"""


# %% Alternate direction implicit (ADI) scheme + dispersion - N
def ADI_2D1_nonlinear(E0, loopInnerM, loopOuterKmax):
    nu = 1  # cylindrical geometry 1, planar geometry 0

    uArray = np.zeros(xResolution, dtype=complex)
    for i in range(1, xResolution - 1):
        uArray[i] = 1 - nu / 2 / i
    vArray = np.zeros(xResolution, dtype=complex)
    for i in range(1, xResolution - 1):
        vArray[i] = 1 + nu / 2 / i

    delta = (zArray[1] - zArray[0]) / (4 * n0 * k0 * (xArray[1] - xArray[0]) ** 2)
    # L_plus
    LPlusMatrix = np.zeros((xResolution, xResolution), dtype=complex)
    dPlus2Array = np.zeros(xResolution, dtype=complex)
    dPlus2Array[0], dPlus2Array[1] = 1 - 4j * delta, 4j * delta

    LPlusMatrix[0, :] = dPlus2Array

    for i in range(1, xResolution - 1):
        LPlusMatrix[i, i - 1] = 1j * delta * uArray[i]
        LPlusMatrix[i, i] = 1 - 2j * delta
        LPlusMatrix[i, i + 1] = 1j * delta * vArray[i]
    # L_minus
    LMinusMatrix = np.zeros((xResolution, xResolution), dtype=complex)
    dMinus2Array = np.zeros(xResolution, dtype=complex)
    dMinus2Array[0], dMinus2Array[1] = 1 + 4j * delta, -4j * delta
    LMinusMatrix[0, :] = dMinus2Array
    LMinusMatrix[-1, -1] = 1
    for i in range(1, xResolution - 1):
        LMinusMatrix[i, i - 1] = -1j * delta * uArray[i]
        LMinusMatrix[i, i] = 1 + 2j * delta
        LMinusMatrix[i, i + 1] = -1j * delta * vArray[i]
    deltaD = -1 * (zArray[1] - zArray[0]) * kDis / (4 * (tArray[1] - tArray[0]) ** 2)
    print(deltaD)
    LPlusMatrixD = np.zeros((tResolution, tResolution), dtype=complex)
    dPlus2ArrayD = np.zeros(tResolution, dtype=complex)
    dPlus2ArrayD[0], dPlus2ArrayD[1] = 1 - 4j * deltaD, 4j * deltaD
    dPlus2ArrayD[0], dPlus2ArrayD[1] = 1, 0
    LPlusMatrixD[0, :] = dPlus2ArrayD
    for i in range(1, tResolution - 1):
        LPlusMatrixD[i, i - 1] = 1j * deltaD
        LPlusMatrixD[i, i] = 1 - 2j * deltaD
        LPlusMatrixD[i, i + 1] = 1j * deltaD
    # L d minus
    LMinusMatrixD = np.zeros((tResolution, tResolution), dtype=complex)
    dMinus2ArrayD = np.zeros(tResolution, dtype=complex)
    dMinus2ArrayD[0], dMinus2ArrayD[1] = 1 + 4j * deltaD, -4j * deltaD
    LMinusMatrixD[0, :] = dMinus2ArrayD
    LMinusMatrixD[-1, -1] = 1
    for i in range(1, tResolution - 1):
        LMinusMatrixD[i, i - 1] = -1j * deltaD
        LMinusMatrixD[i, i] = 1 + 2j * deltaD
        LMinusMatrixD[i, i + 1] = -1j * deltaD

    LMinusMatrixD = np.linalg.inv(LMinusMatrixD)
    LMinusMatrix = np.linalg.inv(LMinusMatrix)

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
            Nn_1 = (zArray[1] - zArray[0]) * Nonlinearity(E)
            E = np.dot(LPlusMatrixD, E.transpose())  #
            Vn_1 = np.dot(LPlusMatrix, E.transpose())
            Sn_1 = Vn_1 + (3 * Nn_1 - Nn_2) / 2
            Nn_2 = Nn_1
            # Perform Costless Diagnostic
            E = np.dot(LMinusMatrix, Sn_1)
            E = np.dot(LMinusMatrixD, E.transpose())  # ##
            E = E.transpose()  #
    return E


# 3D+1 Time # FIX LATER
def split_step_old_time(shape, loopInnerM=1, loopOuterKmax=1):
    def I(E):
        eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    dz = zArray[1] - zArray[0]

    def Nonlinearity_spec(E):
        """print(2 * k0 * n2 * np.abs(E[int(xResolution/2),int(yResolution/2),int(tResolution/2)]) ** 2)
        print((1 / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * chi3 * 1)
        print((1 / (2 * eps0)) * ((w0) ** 2 / (cSOL ** 2 * k0)) * eps0 * chi3 * 1)
        print((1 / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * chi3 * eps0 * cSOL * n0 * 1 / 2)
        print(2 * k0 * n2 * np.abs(E[int(xResolution/2),int(yResolution/2),int(tResolution/2)]) ** 2 *
              eps0 * cSOL * n0 * 1 / 2)
        print(2 * k0 * n2 * eps0 * cSOL* 3 / 4 / 2)
        exit()"""
        return dz * (1j / (2 * eps0)) * ((w0 + KxywMesh[2]) / cSOL / n0) * eps0 * chi3 * I(E) - dz * Betta_func(K) * I(E) ** (
                    K - 1)
        #return dz * 1j * w0 * n2 * I(E) / cSOL - dz * Betta_func(K) * I(E) ** (K - 1)  # & E

    E = shape(xytMesh[0], xytMesh[1], xytMesh[2])
    print(E[int(xResolution / 2), int(yResolution / 2), int(tResolution / 2)])

    # works fine!
    def linear_step(field):
        temporaryField = fftshift(fftn(field))
        temporaryField = (temporaryField *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxywMesh[0] ** 2) *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxywMesh[1] ** 2) *
                          np.exp(1j * dz * kDis / 2 * KxywMesh[2] ** 2))  # something here in /2
        return ifftn(ifftshift(temporaryField))

    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            E = linear_step(E)
            E = E * np.exp(Nonlinearity_spec(E))
        if module_PlottingInBetween:
            fig, ax = plt.subplots(figsize=(8, 7))
            image = plt.imshow(np.abs(E[:, :, int(tResolution/2)]),
                               interpolation='bilinear', cmap='magma',
                               origin='lower', aspect='auto',  # aspect ration of the axes
                               extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
            plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
            ax.set_xlabel('y', fontsize=xyLabelFontSize)
            ax.set_ylabel('x', fontsize=xyLabelFontSize)
            plt.show()


    return E


# %% UPPE with time
def UPPE_time(shape, loopInnerM, loopOuterKmax):
    def I(E):
        # return eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def Nonlinearity(E):
        # Pe = 1j * w0 * n2 * E * I(E) / cSOL - Betta_func(K) * E * I(E) ** (K - 1)
        # Pe = n2 * E * I(E) - Betta_func(K) * E * I(E) ** (K - 1)
        Pe = eps0 * chi3 * E * I(E)  # - Betta_func(K) * E * I(E) ** (K - 1)
        return Pe
        # return 0.5 * Pe + 0.5 * np.conjugate(Pe)

    #########
    E = shape(xytMesh[0], xytMesh[1], xytMesh[2])


    Espec = fftshift(fftn(E))

    """fig = plt.figure(figsize=(8, 7))
    plt.plot(kxArray, np.abs(Espec[:, xResolution -1, int(tResolution/2)]))
    plt.show()
    print(k0)"""

    Aspec = Espec  # / np.exp(1j * np.sqrt(k0**2 - kxArray[0] ** 2 - kxArray[1] ** 2))
    dz = zArray[1] - zArray[0]
    ############################
    n = n0 * (1. + (w0 + KxywMesh[2]) / wD)
    k = n * (w0 + KxywMesh[2]) / cSOL
    print(w0, w0 + KxywMesh[2].max())

    kz = np.sqrt(k ** 2 - KxywMesh[0] ** 2 - KxywMesh[1] ** 2)
    """for i in range(xResolution):
        for j in range(yResolution):
            for l in range(tResolution):
                a = (k[i, j, l] ** 2 - KxywMesh[0][i, j, l] ** 2 - KxywMesh[1][i, j, l] ** 2)
                if a>=0:
                    kz[i, j, l] = np.sqrt(a)
                else:
                    kz[i, j, l] = 0"""

    # print(k)
    # print(KxywMesh[0])
    # exit()
    # P = Nonlinearity(E)
    # Pspec = ifftn(ifftshift(P))
    # temporal derivative
    vPhase = w0 / k0 / 2
    vPhase = cSOL / (n0 + 2 * n0 * w0 / wD)  #######################

    # exit()
    # print(vPhase)
    # exit()
    # A0 = [Aspec]
    # Equation (102) models beam propagation under the effects of diffraction and the optical
    # Kerr effect, leading to beam self-focusing (for a positive n2)
    def ODEs(z, A):
        # without was better for some reason
        A *= np.exp(1j * z * (kz - (w0 + w1D) / vPhase))

        # vPhase

        A = np.reshape(A, (xResolution, yResolution, tResolution))
        E = ifftn(ifftshift(A))
        # E = 0.5 * E + 0.5 * np.conjugate(E)

        P = Nonlinearity((E))
        # P = np.real(P)
        """fig, ax = plt.subplots(figsize=(8, 7))
        # Pspec = np.reshape(np.real(P), (xResolution, yResolution, tResolution))
        plt.plot(np.imag(P[:, 4, 4]))
        plt.show()
        exit()"""
        Pspec = fftshift(fftn(P))
        Pspec = np.reshape(Pspec, (xResolution * yResolution * tResolution))
        Pspec *= np.exp(-1j * z * (kz - (w0 + w1D) / vPhase))
        Pspec *= (1j / (2 * eps0)) * ((w0 + w1D) ** 2 / (cSOL ** 2 * kz))
        # Pspec = Pspec*0 + 1j*1e6

        # print(abs(Pspec).max())
        return Pspec
    """print (k0, kz[int(xResolution/2),int(yResolution/2),int(tResolution/2)])
    print(kz[int(xResolution / 2) + 1, int(yResolution / 2), int(tResolution / 2)],
          kz[int(xResolution / 2) - 1, int(yResolution / 2), int(tResolution / 2)])
    print(kz[int(xResolution / 2), int(yResolution / 2) + 1, int(tResolution / 2)],
          kz[int(xResolution / 2), int(yResolution / 2) - 1, int(tResolution / 2)])
    print(kz[int(xResolution / 2), int(yResolution / 2), int(tResolution / 2) + 1],
          kz[int(xResolution / 2), int(yResolution / 2), int(tResolution / 2) - 1])
    exit()"""
    Aspec = np.reshape(Aspec, (xResolution * yResolution * tResolution))
    w1D = np.reshape(KxywMesh[2], (xResolution * yResolution * tResolution))
    kx1D = np.reshape(KxywMesh[0], (xResolution * yResolution * tResolution))
    ky1D = np.reshape(KxywMesh[1], (xResolution * yResolution * tResolution))
    n = n0 * (1. + (w0 + w1D) / wD)
    k = n * (w0 + w1D) / cSOL
    kz = np.sqrt(k ** 2 - kx1D ** 2 - ky1D ** 2)
    """print(n[int(tResolution / 2) - 1])
    exit()"""
    # complex_ode
    integrator = ode(ODEs).set_integrator('zvode', nsteps=1e6)
    # integrator = ode(ODEs).set_integrator('zvode', nsteps=1e7, atol=10 ** -6, rtol=10 ** -6)
    test = np.copy(Aspec)
    """ kx1D2 = np.zeros(xResolution * yResolution * tResolution)
    ky1D2 = np.zeros(xResolution * yResolution * tResolution)
    w1D2 = np.zeros(xResolution * yResolution * tResolution)
    for i in range(xResolution):
        for j in range(yResolution):
            for m in range(tResolution):
                kx1D2[m + j*tResolution + i*yResolution*tResolution] = kxArray[i]
                ky1D2[m + j * tResolution + i * yResolution * tResolution] = kyArray[j]
                w1D2[m + j * tResolution + i * yResolution * tResolution] = wArray[m]
                Aspec[m + j * tResolution + i * yResolution * tResolution] = Aspec[i, j, m]

    w1D2 = np.reshape(w1D, (xResolution, yResolution, tResolution))
    print ()
    exit()"""

    Aspec = np.reshape(Aspec, (xResolution, yResolution, tResolution))
    if module_CheckingSpectrum and 0:
        fig, ax = plt.subplots(figsize=(8, 7))
        # Pspec = np.reshape(np.real(P), (xResolution, yResolution, tResolution))
        ax.plot(np.abs(Aspec[:, int(yResolution / 2), int(tResolution / 2)]), color='b', lw=6, label='x')
        ax.plot(np.abs(Aspec[int(xResolution / 2), :, int(tResolution / 2)]), color='lime', lw=2.5, label='y')
        ax.plot(np.abs(Aspec[int(xResolution / 2), int(yResolution / 2), :]), color='r', lw=4, label='t')
        ax.legend(shadow=True, fontsize=legendFontSize, facecolor='white', edgecolor='black', loc='upper right')
        plt.show()
    for k in range(loopOuterKmax):

        Aspec = np.reshape(Aspec, (xResolution * yResolution * tResolution))
        for m in range(1, loopInnerM):
            # чему в нуле равен y
            # print(Aspec[10, 10])
            # Aspec = ODE(Pspec, Aspec)
            # Espec = fftshift(fftn(E))

            # Aspec *= np.exp(1j * dz * (kz)) #vPhase
            # Aspec *= np.exp(1j * dz * (kz - (w0 + KxywMesh[2]) / vPhase))  # vPhase
            # z = [0, dz]
            print((k) * loopInnerM + m)


            # integrator.set_initial_value(Aspec, 0)
            # Aspec = integrator.integrate(dz)
            Aspec *= np.exp(1j * dz * (kz - (w0 + w1D) / vPhase))
            # print(np.abs(Aspec - test).max())

            # print ((test - Aspec).max())

            # Aspec = odeint(ODEs,Aspec,z).set_integrator('zvode')[1]
            # E = ifftn(ifftshift(Aspec))
            # P = Nonlinearity(E)
            # Pspec = ifftn(ifftshift(P))
            # Pspec *= np.exp(-1j * dz * (kz - w0 / vPhase))
            # Pspec *= 1j * w0 ** 2 / 2 /eps0 / cSOL ** 2 / kz
        if module_CheckingSpectrum:
            print('checking spectra')
            Aspec = np.reshape(Aspec, (xResolution, yResolution, tResolution))
            fig, ax = plt.subplots(figsize=(8, 7))
            # Pspec = np.reshape(np.real(P), (xResolution, yResolution, tResolution))
            ax.plot(np.abs(Aspec[:, int(yResolution / 2), int(tResolution / 2)]), color='b', lw=6, label='x')
            ax.plot(np.abs(Aspec[int(xResolution / 2), :, int(tResolution / 2)]), color='lime', lw=2.5, label='y')
            ax.plot(np.abs(Aspec[int(xResolution / 2), int(yResolution / 2), :]), color='r', lw=4, label='t')
            ax.legend(shadow=True, fontsize=legendFontSize, facecolor='white', edgecolor='black', loc='upper right')
            plt.show()
        if module_PlottingInBetween:
            fig, ax = plt.subplots(figsize=(8, 7))
            Aspec = np.reshape(Aspec, (xResolution, yResolution, tResolution))
            E = ifftn(ifftshift(Aspec))
            image = plt.imshow(np.abs(E[:, :, int(tResolution/2)]),
                               interpolation='bilinear', cmap='magma',
                               origin='lower', aspect='auto',  # aspect ration of the axes
                               extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
            plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
            ax.set_xlabel('y', fontsize=xyLabelFontSize)
            ax.set_ylabel('x', fontsize=xyLabelFontSize)
            ax.text(0.1 * xFinish, 0.9 * yFinish, f'z={round(dz * ((k) * loopInnerM + m) * 1e3, 3)}mm', color='white',
                    fontsize=legendFontSize)
            plt.show()
            plt.close()
    Aspec = np.reshape(Aspec, (xResolution, yResolution, tResolution))
    E = ifftn(ifftshift(Aspec))

    return E


def UPPE_time_Nitish_slow_Fourier(shape, loopInnerM, loopOuterKmax):
    kxCoeff = 2
    kyCoeff = 2
    kxArray2 = kxArray * kxCoeff
    kyArray2 = kyArray * kyCoeff
    kxArray2 = np.linspace(-1.6e6, 1.6e6, xResolution)
    kyArray2 = np.linspace(-1.6e6, 1.6e6, yResolution)
    #########
    xyMesh = np.array(np.meshgrid(xArray, yArray, indexing='ij'))  # only ADI
    # KxyMesh = np.array(np.meshgrid(kxArray * kxCoeff, yArray * kyCoeff, indexing='ij'))
    E = shape(xyMesh[0], xyMesh[1], 0)
    Eshap = np.shape(E)

    plt.plot(xArray, np.abs(E[:, int(yResolution / 2)]))
    plt.show()
    Espec = fftshift(fftn(E))
    plt.plot(kxArray, np.abs(Espec[:, int(yResolution / 2)]))
    plt.show()
    E = ifftn(ifftshift(Espec))
    plt.plot(xArray, np.abs(E[:, int(yResolution / 2)]))
    plt.show()

    def fftin(E):
        # np.exp(-1j * xyMesh[0] * KxyMesh[0]) * np.exp(-1j * xyMesh[1] * KxyMesh[1])
        Espec = np.zeros((Eshap), dtype=complex)
        Espec1 = np.zeros(Eshap[1], dtype=complex)
        for i in range(Eshap[0]):
            for j in range(Eshap[1]):
                for k in range(Eshap[1]):
                    Espec1[k] = integrate.simps((E[:, k]) *
                                                np.exp(-1j * xArray[:] * kxArray2[i]), xArray)
                Espec[i, j] = integrate.simps(Espec1
                                              * np.exp(-1j * yArray[:] * kyArray2[j]), yArray)
                # print(Espec1[k])

        return Espec / (2 * np.pi)

    def fftout(E):
        # Espec * 1 / (2 * ni.pi)
        Espec = np.zeros((Eshap), dtype=complex)
        Espec1 = np.zeros(Eshap[1], dtype=complex)
        for i in range(Eshap[0]):
            for j in range(Eshap[1]):
                for k in range(Eshap[1]):
                    Espec1[k] = integrate.simps((E[:, k]) *
                                                np.exp(1j * xArray[i] * kxArray2[:]), kxArray2)
                Espec[i, j] = integrate.simps(Espec1
                                              * np.exp(1j * yArray[j] * kyArray2[:]), kyArray2)
                # print(Espec1[k])

        return Espec / (2 * np.pi)

    EspecMy = fftin(E)
    plt.plot(kxArray2, np.abs(EspecMy[:, int(yResolution / 2)]))
    plt.show()
    E = fftout(EspecMy)
    plt.plot(xArray, np.abs(E[:, int(yResolution / 2)]))
    plt.show()
    exit()
    return 0


def plot_1D(x, y, label='', xname='', yname='', ls='-', lw=4, color='rrr', leg=0):
    if color == 'rrr':
        ax.plot(x, y, ls=ls, label=label, lw=lw)
    else:
        ax.plot(x, y, ls=ls, label=label, lw=lw, color=color)
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontname='Times New Roman', fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize)
    ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    if leg:
        ax.legend(shadow=True, fontsize=legendFontSize, facecolor='white', edgecolor='black', loc='upper right')


if __name__ == '__main__':

    # gaussian
    if 0:
        fig, ax = plt.subplots(figsize=(8, 7))
        # ax.plot(rArray, np.abs(Eps(rArray, z=zFinish)), color='lime', ls='-', label='Analytical', lw=4)

        # ax.plot(tArray, np.abs(Eps_dispersion(tArray, z=zFinish)), ls='-', label='Analytical', color='black', lw=10)
        fieldAdi = ADI_2D1_nonlinear(Eps_initiation_with_time(xtMesh[0], xtMesh[1]), loopInnerResolution,
                                     loopOuterResolution)
        """ax.plot(tArray, np.abs(fieldAdi)[0, :], ls='-', label='ADI Crank Nicolson', color='red', lw=10)"""
        """fieldAdiNonlinear = ADI_2D1_nonlinear(Eps_initiation_with_time(rtMesh[0], rtMesh[1]), loopInnerResolution,
                                       loopOuterResolution)
        plot_1D(tArray, np.abs(fieldAdiNonlinear)[0, :]
                , label='Crank Nicolson Dispersion', xname=r'$t$', yname='', ls='-', color='red', lw=6)"""
        fieldOLD = split_step_old_time(Field_1, loopInnerResolution, loopOuterResolution)
        plot_1D(tArray, np.abs(fieldOLD[int(xResolution / 2), int(yResolution / 2), :])
                , label='Fourier_everything', xname=r'$t$', yname='', color='lime', ls='-', lw=6)
        fieldUPPE = UPPE_time(Field_1, loopInnerResolution, loopOuterResolution)
        plot_1D(tArray, np.abs(fieldUPPE[int(xResolution / 2), int(yResolution / 2), :])
                , label='UPPE', xname=r'$t$', yname='', color='blue', ls='-', lw=2)

        # plt.xlim(15,30.5)
        plt.show()
        fig, ax = plt.subplots(figsize=(8, 7))
        """plt.plot(xArray[:int(xResolution/2)], np.abs(fieldAdi[:int(xResolution/2), int(tResolution/2)]),
                 color='red', lw=10, ls='--')"""
        plt.plot(xArray - x0, np.abs(fieldOLD[:, int(yResolution / 2), int(tResolution / 2)]),
                 color='lime', ls='--', lw=6)
        plot_1D(xArray - x0, np.abs(fieldUPPE[:, int(yResolution / 2), int(tResolution / 2)]),
                xname=r'$\rho, t$', yname='', color='blue', ls='--', lw=2)
        # plt.xlim(xStart, xFinish)
        plt.show()


    # real parameters starting propagation:
    if 0:
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.abs(Field_STOV_simple(xytMesh[0], xytMesh[1], xytMesh[2])[:, int(yResolution / 2), :]) ** 2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.angle(Field_STOV_simple(xytMesh[0], xytMesh[1], xytMesh[2])[:, int(yResolution / 2), :]),
                           interpolation='bilinear', cmap='jet',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
        plt.close()

    # real parameters propagation:
    if 0:
        """fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.abs(Field_STOV_1(xytMesh[0], xytMesh[1], xytMesh[2])[:, int(yResolution / 2), :]) ** 2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.angle(Field_STOV_1(xytMesh[0], xytMesh[1], xytMesh[2])[:, int(yResolution / 2), :]),
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
        plt.close()
        exit()"""
        fieldOLD = split_step_old_time(Field_STOV_simple, loopInnerResolution, loopOuterResolution)
        fieldUPPE = UPPE_time(Field_STOV_simple, loopInnerResolution, loopOuterResolution)
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.abs(fieldOLD[:, int(yResolution / 2), :]) ** 2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.abs(fieldUPPE[:, int(yResolution / 2), :]) ** 2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        ax.set_xlabel('t', fontsize=xyLabelFontSize)
        ax.set_ylabel('x', fontsize=xyLabelFontSize)
        plt.show()
        plt.close()
        # phase
        if 0:
            fig, ax = plt.subplots(figsize=(8, 7))
            image = plt.imshow(np.angle(fieldOLD[:, int(yResolution / 2), :]),
                               interpolation='bilinear', cmap='jet',
                               origin='lower', aspect='auto',  # aspect ration of the axes
                               extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
            plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
            ax.set_xlabel('t', fontsize=xyLabelFontSize)
            ax.set_ylabel('x', fontsize=xyLabelFontSize)
            plt.show()
            fig, ax = plt.subplots(figsize=(8, 7))
            image = plt.imshow(np.angle(fieldUPPE[:, int(yResolution / 2), :]),
                               interpolation='bilinear', cmap='jet',
                               origin='lower', aspect='auto',  # aspect ration of the axes
                               extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
            plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
            ax.set_xlabel('t', fontsize=xyLabelFontSize)
            ax.set_ylabel('x', fontsize=xyLabelFontSize)
            plt.show()
            plt.close()
    # gauss propogation
    if 0:
        """fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.abs(Field_STOV_1(xytMesh[0], xytMesh[1], xytMesh[2])[:, int(yResolution / 2), :]) ** 2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.angle(Field_STOV_1(xytMesh[0], xytMesh[1], xytMesh[2])[:, int(yResolution / 2), :]),
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
        plt.close()
        exit()"""
        if module_Paraxial:
            fieldOLD = split_step_old_time(Field_1, loopInnerResolution, loopOuterResolution)
        if module_NonRapaxial:
            fieldUPPE = UPPE_time(Field_1, loopInnerResolution, loopOuterResolution)
        # 2D
        if module_2D:
            if module_Paraxial:
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.abs(fieldOLD[:, :, int(tResolution / 2)]) ** 2,
                                   interpolation='bilinear', cmap='magma',
                                   origin='lower', aspect='auto',  # aspect ration of the axes
                                   extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
                plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
                ax.set_xlabel('y', fontsize=xyLabelFontSize)
                ax.set_ylabel('x', fontsize=xyLabelFontSize)
                plt.show()
            if module_NonRapaxial:
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.abs(fieldUPPE[:, :, int(tResolution / 2)]) ** 2,
                                   interpolation='bilinear', cmap='magma',
                                   origin='lower', aspect='auto',  # aspect ration of the axes
                                   extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
                plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
                ax.set_xlabel('y', fontsize=xyLabelFontSize)
                ax.set_ylabel('x', fontsize=xyLabelFontSize)
                plt.show()
                plt.close()
        # 1D
        if module_1D:
            fig, ax = plt.subplots(figsize=(8, 7))
            # ax.plot(rArray, np.abs(Eps(rArray, z=zFinish)), color='lime', ls='-', label='Analytical', lw=4)

            # ax.plot(tArray, np.abs(Eps_dispersion(tArray, z=zFinish)), ls='-', label='Analytical', color='black', lw=10)
            """fieldAdi = ADI_2D1_nonlinear(Eps_initiation_with_time(xtMesh[0], xtMesh[1]), loopInnerResolution,
                                         loopOuterResolution)"""
            # ax.plot(tArray, np.abs(fieldAdi)[0, :], ls='-', label='ADI Crank Nicolson', color='red', lw=10)
            """fieldAdiNonlinear = ADI_2D1_nonlinear(Eps_initiation_with_time(rtMesh[0], rtMesh[1]), loopInnerResolution,
                                           loopOuterResolution)
            plot_1D(tArray, np.abs(fieldAdiNonlinear)[0, :]
                    , label='Crank Nicolson Dispersion', xname=r'$t$', yname='', ls='-', color='red', lw=6)"""
            #fieldOLD = split_step_old_time(Field_1, loopInnerResolution, loopOuterResolution)
            if module_Paraxial:
                plot_1D(tArray, np.abs(fieldOLD[int(xResolution / 2), int(yResolution / 2), :])
                        , label='Paraxial', xname=r'$t$', yname='', color='lime', ls='-', lw=6)
            #fieldUPPE = UPPE_time(Field_1, loopInnerResolution, loopOuterResolution)
            if module_NonRapaxial:
                plot_1D(tArray, np.abs(fieldUPPE[int(xResolution / 2), int(yResolution / 2), :])
                        , label='UPPE (non-paraxial)', xname=r'$t$', yname='', color='blue', ls='-', lw=2)
            plt.show()
            # plt.xlim(15,30.5)
            # plt.show()
            # fig, ax = plt.subplots(figsize=(8, 7))
            # plt.plot(xArray, np.abs(fieldAdi[:, int(tResolution/2)]),
            #         color='red', ls='--', lw=6)
            # E(x)
            if 1:
                fig, ax = plt.subplots(figsize=(8, 7))
                if module_Paraxial:
                    plt.plot(xArray, np.abs(fieldOLD[:, int(yResolution / 2), int(tResolution / 2)]),
                             label='Paraxial',color='lime', ls='--', lw=6)
                if module_NonRapaxial:
                    plot_1D(xArray, np.abs(fieldUPPE[:, int(yResolution / 2), int(tResolution / 2)]),
                            label='UPPE (non-paraxial)',xname=r'$x$', yname='', color='blue', ls='--', lw=2)
                #plt.xlim(xStart, xFinish)
                plt.show()
        # phase
        if module_Phase:
            if module_Paraxial:
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.angle(fieldOLD[:, int(yResolution / 2), :]),
                                   interpolation='bilinear', cmap='jet',
                                   origin='lower', aspect='auto',  # aspect ration of the axes
                                   extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
                plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
                ax.set_xlabel('t', fontsize=xyLabelFontSize)
                ax.set_ylabel('x', fontsize=xyLabelFontSize)
                plt.show()
            if module_NonRapaxial:
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.angle(fieldUPPE[:, int(yResolution / 2), :]),
                                   interpolation='bilinear', cmap='jet',
                                   origin='lower', aspect='auto',  # aspect ration of the axes
                                   extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
                plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
                ax.set_xlabel('t', fontsize=xyLabelFontSize)
                ax.set_ylabel('x', fontsize=xyLabelFontSize)
                plt.show()
                plt.close()

    # LG Nitish
    if 1:
        if module_Paraxial:
            fieldOLD = split_step_old_time(LG_Nitish, loopInnerResolution, loopOuterResolution)
        if module_NonRapaxial:
            #fieldUPPE = UPPE_time_Nitish_slow_Fourier(LG_Nitish, loopInnerResolution, loopOuterResolution)
            fieldUPPE = UPPE_time(LG_Nitish, loopInnerResolution, loopOuterResolution)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_axes((0.12, 0.09, 0.85, 0.89))
        image = plt.imshow(np.abs(LG_Nitish(xytMesh[0], xytMesh[1], xytMesh[2])[:,:,0]),
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        ax.set_xlabel('y (m)', fontsize=xyLabelFontSize)
        ax.set_ylabel('x (m)', fontsize=xyLabelFontSize)
        #plt.xlim(x0 - 5e-3, x0 + 5e-3)
        #plt.ylim(y0 - 5e-3, y0 + 5e-3)
        plt.show()
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_axes((0.12, 0.09, 0.85, 0.89))
        image = plt.imshow(np.angle(LG_Nitish(xytMesh[0], xytMesh[1], xytMesh[2])[:, :, 0]),
                           interpolation='bilinear', cmap='jet',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        ax.set_xlabel('y (m)', fontsize=xyLabelFontSize)
        ax.set_ylabel('x (m)', fontsize=xyLabelFontSize)
        # plt.xlim(x0 - 5e-3, x0 + 5e-3)
        # plt.ylim(y0 - 5e-3, y0 + 5e-3)
        #plt.show()
        #exit()
        if data_save:
            np.save('dataa3.npy', fieldUPPE)

        if module_2D:
            if module_Paraxial:
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.abs(fieldOLD[:, :, int(tResolution / 2)]),
                                   interpolation='bilinear', cmap='magma',
                                   origin='lower', aspect='auto',  # aspect ration of the axes
                                   extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
                plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
                ax.set_xlabel('y', fontsize=xyLabelFontSize)
                ax.set_ylabel('x', fontsize=xyLabelFontSize)

                plt.show()
            if module_NonRapaxial:
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.abs(fieldUPPE[:, :, int(tResolution / 2)]),
                                   interpolation='bilinear', cmap='magma',
                                   origin='lower', aspect='auto',  # aspect ration of the axes
                                   extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
                plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
                ax.set_xlabel('y', fontsize=xyLabelFontSize)
                ax.set_ylabel('x', fontsize=xyLabelFontSize)
                plt.show()
                plt.close()
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.abs(fieldUPPE[:, :, int(tResolution / 2)]),
                                   interpolation='bilinear', cmap='magma',
                                   origin='lower', aspect='auto',  # aspect ration of the axes
                                   extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
                plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
                ax.set_xlabel('y', fontsize=xyLabelFontSize)
                ax.set_ylabel('x', fontsize=xyLabelFontSize)
                #plt.xlim(x0-5e-3, x0+5e-3)
                #plt.ylim(y0 - 5e-3, y0 + 5e-3)
                plt.show()
                plt.close()
        # 1D
        if module_1D:
            fig, ax = plt.subplots(figsize=(8, 7))
            # ax.plot(rArray, np.abs(Eps(rArray, z=zFinish)), color='lime', ls='-', label='Analytical', lw=4)

            # ax.plot(tArray, np.abs(Eps_dispersion(tArray, z=zFinish)), ls='-', label='Analytical', color='black', lw=10)
            """fieldAdi = ADI_2D1_nonlinear(Eps_initiation_with_time(xtMesh[0], xtMesh[1]), loopInnerResolution,
                                         loopOuterResolution)"""
            # ax.plot(tArray, np.abs(fieldAdi)[0, :], ls='-', label='ADI Crank Nicolson', color='red', lw=10)
            """fieldAdiNonlinear = ADI_2D1_nonlinear(Eps_initiation_with_time(rtMesh[0], rtMesh[1]), loopInnerResolution,
                                           loopOuterResolution)
            plot_1D(tArray, np.abs(fieldAdiNonlinear)[0, :]
                    , label='Crank Nicolson Dispersion', xname=r'$t$', yname='', ls='-', color='red', lw=6)"""
            #fieldOLD = split_step_old_time(Field_1, loopInnerResolution, loopOuterResolution)
            if module_Paraxial:
                plot_1D(tArray, np.abs(fieldOLD[:, int(yResolution / 2), int(tResolution / 2)])
                        , label='Paraxial', xname=r'$t$', yname='', color='lime', ls='-', lw=6)
            #fieldUPPE = UPPE_time(Field_1, loopInnerResolution, loopOuterResolution)
            if module_NonRapaxial:
                plot_1D(tArray, np.abs(fieldUPPE[:, int(yResolution / 2), int(tResolution / 2)])
                        , label='UPPE (non-paraxial)', xname=r'$t$', yname='', color='blue', ls='-', lw=2)
            plt.show()
        if module_Phase:
            fig, ax = plt.subplots(figsize=(8, 7))
            image = plt.imshow(np.angle(LG_Nitish(xytMesh[0], xytMesh[1], xytMesh[2])[:, :, 0]),
                               interpolation='bilinear', cmap='jet',
                               origin='lower', aspect='auto',  # aspect ration of the axes
                               extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
            plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
            ax.set_xlabel('y', fontsize=xyLabelFontSize)
            ax.set_ylabel('x', fontsize=xyLabelFontSize)
            plt.show()
            if module_Paraxial:
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.angle(fieldOLD[:, :]),
                                   interpolation='bilinear', cmap='jet',
                                   origin='lower', aspect='auto',  # aspect ration of the axes
                                   extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
                plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
                ax.set_xlabel('y', fontsize=xyLabelFontSize)
                ax.set_ylabel('x', fontsize=xyLabelFontSize)
                plt.show()
            if module_NonRapaxial:
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.angle(fieldUPPE[:, :]),
                                   interpolation='bilinear', cmap='jet',
                                   origin='lower', aspect='auto',  # aspect ration of the axes
                                   extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
                plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
                ax.set_xlabel('y', fontsize=xyLabelFontSize)
                ax.set_ylabel('x', fontsize=xyLabelFontSize)
                #plt.xlim(x0 - 5e-3, x0 + 5e-3)
                #plt.ylim(y0 - 5e-3, y0 + 5e-3)
                plt.show()
                plt.close()
