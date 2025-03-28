import numpy as np
import matplotlib.pyplot as plt
from pyhank import qdht, iqdht
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.integrate import odeint, ode, complex_ode
from scipy.special import jv, iv
# %% Resolutions
xResolution = 37# N_per
yResolution = 21
tResolution = 97
loopInnerResolution, loopOuterResolution = 3, 1  # M, Kmax
zResolution = loopInnerResolution * loopOuterResolution
xStart, xFinish = 0, 30 # 10
yStart, yFinish = 0, 30 # 10
zStart, zFinish = 0, 10
tStart, tFinish = 0, 30
x0 = (xFinish - xStart) / 2
y0 = (yFinish - yStart) / 2
t0 = (tFinish - tStart) / 2

###

# parameters
lambda0 = 1 * 1e-2

# %% temporal
rho0 = 5
# k0 = 100
f = 150000
Eps0 = 1
tp = 1.2  # pulse duration
# kDis = 0 * 0.02  # dispersion

C = 0  # chirp
cSOL = 1
# w0 = 1
n2 =  2e-6  # 3 * chi3 / (4 * eps0 * c * n0 ** 2)
K = 1
eps0 = 1
n0 = 1.0
lOAM = 0

# new parameters
k0 = 2 * np.pi / lambda0
w0 = k0 * cSOL
####
wD = 5 * w0 * 1e-2 / 1e-2 * 10000000000
kDis = 2 * n0 / cSOL / wD


def Betta_func(K):
    Betta = [0,  34e0 * 4e-3]
    return Betta[K]


# %% Plotting parameters
ticksFontSize = 18
legendFontSize = 18
xyLabelFontSize = 18


# %% Arrays creation
xArray = np.linspace(xStart, xFinish, xResolution)
yArray = np.linspace(yStart, yFinish, yResolution)
zArray = np.linspace(zStart, zFinish, zResolution)
tArray = np.linspace(tStart, tFinish, tResolution)
xtMesh = np.array(np.meshgrid(xArray, tArray, indexing='ij')) # only ADI
kxArray = np.linspace(-1. * np.pi * (xResolution - 2) / xFinish,
                      1. * np.pi * (xResolution - 2) / xFinish, xResolution)
kyArray = np.linspace(-1. * np.pi * (yResolution - 2) / yFinish,
                      1. * np.pi * (yResolution - 2) / yFinish, yResolution)
wArray = np.linspace(-1. * np.pi * (tResolution - 2) / tFinish, 1. * np.pi * (tResolution - 2) / tFinish,
                     tResolution)
xytMesh = np.array(np.meshgrid(xArray, yArray, tArray, indexing='ij'))
KxywMesh = np.array(np.meshgrid(kxArray, kyArray, wArray, indexing='ij'))

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
        return np.exp(-y ** 2)

    #xytArray = np.zeros((xResolution, yResolution, tResolution), dtype=complex)
    #xMesh, tMesh = np.array(np.meshgrid(x, y, indexing='ij'))
    """xtArray = (2 * np.pi * (-1j) ** lOAM
               * np.exp(-1j * lOAM * radius(xMesh, tMesh))
               * H1(radius(xMesh, tMesh)))"""

    """for i in range(yResolution):
        xytArray[:, i, :] = xtArray * y_dependence(yArray[i])"""
    return (2 * np.pi * (-1j) ** lOAM
               * np.exp(-1j * lOAM * phi(x - x0, t - t0))
               * H1(radius(x - x0, t - t0))) * y_dependence(y-y0)


# with time t
def Eps_initiation_with_time(r, t):
    return Eps0 * np.exp(-1 * r ** 2 / rho0 ** 2 - 1j * k0 * r ** 2 / (2 * f) -
                         (1. + 1j * C) * (t - t0) ** 2 / tp ** 2)


# %% general functions
def radius(x, y):
    return np.sqrt(x ** 2 + y ** 2)

def phi(x, t):
    return np.angle(x + 1j * t)


#################### something with n2 #################################
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
        #eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
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



# 3D+1 Time # FIX LATER
def split_step_old_time(shape, loopInnerM=1, loopOuterKmax=1):
     # !!!!!!!!!!!!!! delti
    """
    dx = xArray[1] - xArray[0]
    X = dx * (xResolution2)
    dw = 2 * np.pi / X
    dwMy = kxArray[1] - kxArray[0]
    print (dw, dwMy)
    exit()"""

    def I(E):
        #return eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    dz = zArray[1] - zArray[0]

    def Nonlinearity_spec(E):
        return dz * 1j * w0 * n2 * I(E) / cSOL - dz * Betta_func(K) * I(E) ** (K - 1)  # & E

    E = shape(xytMesh[0], xytMesh[1], xytMesh[2])
    print(E[int(xResolution/2), int(yResolution/2), int(tResolution / 2)])

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

    return E  # xResolution - 1:xResolution2


# %% UPPE with time
def UPPE_time(shape, loopInnerM, loopOuterKmax):

    def I(E):
        eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def Nonlinearity(E):
        return 1j * w0 * n2 * E * I(E) / cSOL - Betta_func(K) * E * I(E) ** (K - 1)


    ##########



    """def pend2(y, t, b, c):
        return - c * np.sin(t)

    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1]  # чему в нуле равен y
    t = [0, 5]
    sol = odeint(pend2, y0, t, args=(b, c))
    """

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
    vPhase = cSOL / (n0 + 2 * n0 * w0 / wD)   #######################

    # exit()
    # print(vPhase)
    # exit()
    # A0 = [Aspec]
    def ODEs(z, A):
        A *= np.exp(1j * z * (kz - 0*(w0 + w1D) / vPhase))
        # vPhase

        A = np.reshape(A, (xResolution, yResolution, tResolution))
        E = ifftn(ifftshift(A))

        P = Nonlinearity(E)


        Pspec = fftshift(fftn(P))

        Pspec = np.reshape(Pspec, (xResolution * yResolution * tResolution))



        Pspec *= np.exp(-1j * z * (kz - 0*(w0 + w1D) / vPhase))
        Pspec *= 1j * (w0 + w1D) ** 2 / (2 * eps0 * cSOL ** 2 * kz)

        #Pspec = Pspec*0 + 1j*1e6
        """fig, ax = plt.subplots(figsize=(8, 7))
        Pspec = np.reshape(Pspec, (xResolution, yResolution, tResolution))
        plt.plot(abs(Pspec[18, 2, :]))
        plt.show()
        exit()"""
        #print(abs(Pspec).max())
        return Pspec

    Aspec = np.reshape(Aspec, (xResolution * yResolution * tResolution))
    w1D = np.reshape(KxywMesh[2], (xResolution * yResolution * tResolution))
    kx1D = np.reshape(KxywMesh[0], (xResolution * yResolution * tResolution))
    ky1D = np.reshape(KxywMesh[1], (xResolution * yResolution * tResolution))
    n = n0 * (1. + (w0 + w1D) / wD)
    k = n * (w0 + w1D) / cSOL
    kz = np.sqrt(k ** 2 - kx1D ** 2 - ky1D ** 2)
    #complex_ode
    integrator = ode(ODEs).set_integrator('zvode', nsteps=1e7)
    #integrator = ode(ODEs).set_integrator('zvode', nsteps=1e7, atol=10 ** -6, rtol=10 ** -6)
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
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # чему в нуле равен y
            # print(Aspec[10, 10])
            # Aspec = ODE(Pspec, Aspec)
            # Espec = fftshift(fftn(E))

            # Aspec *= np.exp(1j * dz * (kz)) #vPhase
            #Aspec *= np.exp(1j * dz * (kz - (w0 + KxywMesh[2]) / vPhase))  # vPhase
            #z = [0, dz]


            integrator.set_initial_value(Aspec, 0)
            Aspec = integrator.integrate(dz)
            #Aspec *= np.exp(1j * dz * (kz - (w0 + w1D) / vPhase))

            #print(np.abs(Aspec - test).max())


            # print ((test - Aspec).max())

            #Aspec = odeint(ODEs,Aspec,z).set_integrator('zvode')[1]
            # E = ifftn(ifftshift(Aspec))
            # P = Nonlinearity(E)
            # Pspec = ifftn(ifftshift(P))
            # Pspec *= np.exp(-1j * dz * (kz - w0 / vPhase))
            # Pspec *= 1j * w0 ** 2 / 2 /eps0 / cSOL ** 2 / kz
    Aspec = np.reshape(Aspec, (xResolution, yResolution, tResolution))
    E = ifftn(ifftshift(Aspec))

    return E

def UPPE_time_linear(shape, loopInnerM, loopOuterKmax):
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

        for i in range(xFinish * 2):
            for j in range(yFinish * 2):
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
    P = Nonlinearity(E)
    Pspec = ifftn(ifftshift(P))
    # temporal derivative
    vPhase = w0 / k0 / 2
    vGroup = cSOL / (n0 + 2 * n0 * w0 / wD)

    # exit()
    # print(vPhase)
    # exit()
    # A0 = [Aspec]
    test = np.copy(Aspec)
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # чему в нуле равен y
            # print(Aspec[10, 10])
            # Aspec = ODE(Pspec, Aspec)
            # Espec = fftshift(fftn(E))

            # Aspec *= np.exp(1j * dz * (kz)) #vPhase
            Aspec *= np.exp(1j * dz * (kz - (w0 + KxywMesh[2]) / vGroup))  # vPhase
            print(np.abs(test - Aspec).max())

            # E = ifftn(ifftshift(Aspec))
            # P = Nonlinearity(E)
            # Pspec = ifftn(ifftshift(P))
            # Pspec *= np.exp(-1j * dz * (kz - w0 / vPhase))
            # Pspec *= 1j * w0 ** 2 / 2 /eps0 / cSOL ** 2 / kz

    E = ifftn(ifftshift(Aspec))
    return E

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


    # gaussian
    if 1:
        fig, ax = plt.subplots(figsize=(8, 7))
        # ax.plot(rArray, np.abs(Eps(rArray, z=zFinish)), color='lime', ls='-', label='Analytical', lw=4)

        # ax.plot(tArray, np.abs(Eps_dispersion(tArray, z=zFinish)), ls='-', label='Analytical', color='black', lw=10)
        fieldAdi = ADI_2D1_nonlinear(Eps_initiation_with_time(xtMesh[0], xtMesh[1]), loopInnerResolution,
                                     loopOuterResolution)
        #ax.plot(tArray, np.abs(fieldAdi)[0, :], ls='-', label='ADI Crank Nicolson', color='red', lw=10)
        """fieldAdiNonlinear = ADI_2D1_nonlinear(Eps_initiation_with_time(rtMesh[0], rtMesh[1]), loopInnerResolution,
                                       loopOuterResolution)
        plot_1D(tArray, np.abs(fieldAdiNonlinear)[0, :]
                , label='Crank Nicolson Dispersion', xname=r'$t$', yname='', ls='-', color='red', lw=6)"""
        fieldOLD = split_step_old_time(Field_1, loopInnerResolution, loopOuterResolution)
        plot_1D(tArray, np.abs(fieldOLD[int(xResolution/2), int(yResolution/2), :])
                , label='Fourier_everything', xname=r'$t$', yname='', color='lime', ls='-', lw=6)
        fieldUPPE = UPPE_time(Field_1, loopInnerResolution, loopOuterResolution)
        plot_1D(tArray, np.abs(fieldUPPE[int(xResolution/2), int(yResolution/2), :])
                , label='UPPE', xname=r'$t$', yname='', color='blue', ls='-', lw=2)

        # plt.xlim(15,30.5)
        #plt.show()
        #fig, ax = plt.subplots(figsize=(8, 7))

        plt.plot(xArray - x0, np.abs(fieldOLD[:, int(yResolution/2), int(tResolution/2)]),
                 color='lime', ls='--', lw=6)
        plot_1D(xArray - x0, np.abs(fieldUPPE[:, int(yResolution/2), int(tResolution/2)]),
                xname=r'$\rho, t$', yname='', color='blue', ls='--', lw=2)
        plt.xlim(xStart,xFinish)
        plt.show()
    # 2D paraxial + nonparaxial
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
        fieldOLD = split_step_old_time(Field_STOV_1, loopInnerResolution, loopOuterResolution)
        fieldUPPE = UPPE_time(Field_STOV_1, loopInnerResolution, loopOuterResolution)
        Field_STOV_1
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.abs(fieldOLD[:, :, int(tResolution / 2)]) ** 2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()
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
        # phase
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


