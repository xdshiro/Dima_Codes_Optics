import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.integrate import odeint, complex_ode, ode
from scipy.special import erf, jv, iv, assoc_laguerre
import plotly.graph_objects as go

# %% Modules
# checking spectra
module_CheckingSpectrum = 0
module_HOBBIT = 0
module_SUM = 1
MOD_4pulses = 1
module_ADI = 0
module_Paraxial = 1
module_NonRapaxial = 0
module_Phase = 0
module_initial = 0
module_Intensity = 1  # 1-yes, 0-field abs
module_3D = 0
save = 0
save_name = 'SUMm-1B_4Hp1e5_y100_phph1_-3'

# %% Resolutions
xResolution = 171  # N_per 2
yResolution = 171
tResolution = 1   # This is a 2D version of the code
loopInnerResolution, loopOuterResolution = 501, 1  # M, Kmax
zResolution = loopInnerResolution * loopOuterResolution
xStart, xFinish = 0, 700 * 1e-6  # 10
yStart, yFinish = 0, 700 * 1e-6  # 10
zStart, zFinish = 1e-3, 0.08
tStart, tFinish = 0, 1000 * 1e-13
# #
x1, x2 = 0 * 50 * 1e-6, 0 * -50 * 1e-6
y1, y2 = 100 * 1e-6, -100 * 1e-6
t1, t2 = 0 * 100 * 1e-15, 0 * -100 * 1e-15

time = int(tResolution / 2)


################################### temp ######################################
sigma_K8 = 2.4e-42 * 1e-2 ** (2 * 4)
sigma = 4e-18 * 1e-2 ** 2  #
rho_at = 7e22 * 1e-2 ** (-3)
a = 0
tau_c = 3e-15
###
# %% Pulse parameters

rho0 = 200 * 1e-6
tp = 200 * 1e-12  # pulse duration
lambda0 = 0.517e-6
Pmax = 13.5e6  # 2.5e7
# %% Hobbit parameters
if 1:
    betaHob = 1.08
    alphaHob = 0
    ro_0Hob = 650e-6
    k = 3
    FHob = 150e-3
    wRingHob = 244e-6
    w_GHob = lambda0 * FHob / (np.pi * wRingHob)

    tau_0 = 242e-4
# STOV
yRadius = 800 * 1e-6
xSTOVRadius = rho0
tSTOVRadius = tp
lOAM = 0
phase = np.pi * 1
lOAMSUM1 = 1
lOAMSUM2 = -1
x0 = (xFinish - xStart) / 2
y0 = (yFinish - yStart) / 2
t0 = (tFinish - tStart) / 2
f = 1e5  # TEMPORAL ################################################################################
# linear medium parameter

k2Dis = 5.6e-28 / 1e-2  # ps2/m  GVD
# k2Dis = -9.443607756116762e-22  # OAM non-diffraction
n0 = 1.332
# %% Nonlinear parameters
K = 4  # photons number
# %% temporal
C = 0  # chirp # TEMPORAL ################################################################################
n2 = 2.7e-16 * 1e-2 ** 2  # 3 * chi3 / (4 * eps0 * c * n0 ** 2) # TEMPORAL #######################################################
# kDis = 0 * 0.02  # dispersion
q_e = -1.602176565e-19  # [C] electron charge
Ui = 7.1 * abs(q_e)  # TEMPORAL ################################################################################
SigmaK = [0, 1, 2]  # TEMPORAL ################################################################################


def Betta_func(K):
    Betta = [0, 0 * 2e-0, 2, 3, 2.4e-37 * 1e-2 ** (2 * K - 3), 5, 6, 7, 3.79347046850176e-121]
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
wD = 2 * n0 / (k2Dis * cSOL)
# chi3 = n2 * 4 * eps0 * cSOL * n0 ** 2 / 3

# wD = 5 * 1e20  # TEMPORAL ################################################################################
# wD = wD * w0  # TEMPORAL ################################################################################
# kDis = 2 * n0 / cSOL / wD  # TEMPORAL ################################################################################
# chi3 = n2 * 4 * eps0 * cSOL * n0 ** 2 / 3
chi3_2 = 8 * n0 * n2 / 3
epsNL = 3 * chi3_2 / 4
Int = module_Intensity + 1
Imax = 1
if n2 == 0:
    Pcrit = 1e100
else:
    Pcrit = 1.22 ** 2 * np.pi * lambda0 ** 2 / (32 * n0 * n2)
print("P crit (MW): ", Pcrit * 1e-6)


def LDF():
    return np.pi * rho0 ** 2 / lambda0


def Lcollapse():
    temp1 = 0.367 * LDF()

    if n2 == 0:
        return 0
    else:
        temp2 = (np.sqrt(Pmax / Pcrit) - 0.852) ** 2 - 0.0219
        return temp1 / np.sqrt(temp2)


print("Rayleigh length: ", LDF(), " Kerr Collapse length: ", Lcollapse())

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


def sum_fields(x, y, t):
    # field = Hobbit

    def B_m_hob(n, lHob):
        temp = betaHob * np.pi * (lHob + alphaHob - n) / 2
        return ((-1j) ** (n - 1) * 2 * np.exp(- temp ** 2) *
                np.imag(erf(1j * (1j + temp)) / 1j))

    def arg_hob(x, y):
        return np.pi * ro_0Hob * radius(x, y) / (lambda0 * FHob)

    def J_hob(arg, m):
        return jv(m, arg)

    def U_far_hob2(x, y, t, m, k):
        lHob = m
        tempMas = 0
        for i in range(m - k, m + k + 1):
            tempMas += B_m_hob(i, lHob) * np.exp(1j * i * phi(x - x0, y - y0)) * J_hob(arg_hob(x - x0, y - y0) * 2, i)
        tempMas = (tempMas * np.exp(- radius(x - x0, y - y0) ** 2 / w_GHob ** 2)) * np.exp(
            -2 * np.log(2) * ((t - t0) / tau_0) ** 2)
        return tempMas

    temp = Imax * (U_far_hob2(x - x1, y - y1, t - t1, lOAMSUM1, k) + U_far_hob2(x - x2, y - y2, t - t2, lOAMSUM2, k)
                   * np.exp(1j * phase))
    if MOD_4pulses:
        temp += Imax * (U_far_hob2(x - x2, y - y2, t - t2, -lOAMSUM2, k))

    return temp


# initial fields
def Field_1(x, y, t):
    return Imax * (np.exp(- (radius(x - x0, y - y0) ** 2) / rho0 ** 2 -
                          1j * k0 * radius(x - x0, y - x0) ** 2 / (2 * f) -
                          ((t - t0) / tp) ** 2)
                   * np.exp(1j * lOAM * phi(x - x0, y - y0)))


def Field_simple_OAM(x, y, t):
    return Imax * (np.exp(- (radius(x - x0, y - y0) ** 2) / rho0 ** 2 -
                          1j * k0 * radius(x - x0, y - x0) ** 2 / (2 * f) -
                          ((t - t0) / tp) ** 2)) * ((x - x0) / rho0 + 1j * np.sign(lOAM) * (y - y0) / rho0) ** np.abs(
        lOAM)


def Field_1_2D(r, t):
    return Imax * (np.exp(- (r ** 2) / rho0 ** 2 -
                          1j * k0 * r ** 2 / (2 * f) -
                          ((t - t0) / tp) ** 2))


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


def Hobbit(x, y, t):
    m = lOAM
    print(f"lOAM = {m}")

    def B_m_hob(n, lHob):
        temp = betaHob * np.pi * (lHob + alphaHob - n) / 2
        return ((-1j) ** (n - 1) * 2 * np.exp(- temp ** 2) *
                np.imag(erf(1j * (1j + temp)) / 1j))

    def arg_hob(x, y):
        return np.pi * ro_0Hob * radius(x, y) / (lambda0 * FHob)

    def J_hob(arg, m):
        return jv(m, arg)

    def U_far_hob2(x, y, t, m, k):
        lHob = m
        tempMas = 0
        for i in range(m - k, m + k + 1):
            tempMas += B_m_hob(i, lHob) * np.exp(1j * i * phi(x - x0, y - y0)) * J_hob(arg_hob(x - x0, y - y0) * 2, i)
        tempMas = (tempMas * np.exp(- radius(x - x0, y - y0) ** 2 / w_GHob ** 2)) * np.exp(
            -2 * np.log(2) * ((t - t0) / tau_0) ** 2)
        return tempMas

    return Imax * U_far_hob2(x, y, t, m, k)


def asymmetric_LG(x, y,t):
    z = 1e-6
    width = rho0
    p = 0
    l = lOAM
    x = x - x0
    y = y - y0  # + 1j * 1e-3 / 6

    def rayleigh_range(lambda0, rho0):
        return np.pi * rho0 ** 2 / lambda0

    zR = rayleigh_range(lambda0, width)
    print("Rayleigh Range: ", zR)

    # z = 1e-20

    def rho(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def width_z(z):
        return width * np.sqrt(1 + (z / zR) ** 2)

    def R(z):
        return z * (1 + (zR / z) ** 2)

    def ksi(z):
        return np.arctan(z / zR)

    def laguerre_polynomial(x, l, p):
        return assoc_laguerre(x, l, p)

    def nonlinearity(x, y):
        # return x + 1j * np.sign(l) * y
        return (np.sqrt((x) ** 2 + y ** 2) * np.exp(1j * np.arctan(y / (x))))

    #
    # y = y - y0 #+ 1j * 1e-3 / 6
    E = (width_z(0) / width_z(z) * (np.sqrt(2) / width_z(z)) ** (np.abs(l))
         * nonlinearity(x, y) ** (np.abs(l))
         * laguerre_polynomial(2 * rho(x, y) ** 2 / width_z(z) ** 2, np.abs(l), p)
         * np.exp(-rho(x, y) ** 2 / width_z(z) ** 2 + 1j * k0 * rho(x, y) ** 2 / (2 * R(z))
                  - 1j * (np.abs(l) + 2 * p + 1) * ksi(z)))
    return Imax * E


# STOV simple t/tw + ix/xw
def Field_STOV_simple(x, y, t):
    def y_dependence(y):
        return np.exp(-(y / yRadius) ** 2)

    def x_dependence(x):
        return np.exp(-(x / rho0) ** 2)

    def t_dependence(t):
        return np.exp(-(t / tp) ** 2)

    return Imax * (
            ((t - t0) / tSTOVRadius + 1j * np.sign(lOAM) * (x - x0) / xSTOVRadius) ** (np.abs(lOAM)) * y_dependence(
        y - y0) *
            x_dependence(x - x0) * t_dependence(t - t0) * np.exp(-1j * k0 * radius(x - x0, y - x0) ** 2 / (2 * f))
    )


def Field_STOV_simple0(x, y, t):
    def y_dependence(y):
        return np.exp(-(y / yRadius) ** 2)

    def x_dependence(x):
        return np.exp(-(x / rho0) ** 2)

    def t_dependence(t):
        return np.exp(-(t / tp) ** 2)

    return Imax * (((t - t0) / tSTOVRadius + 1j * np.sign(lOAM) * radius(x - x0, y - x0) / xSTOVRadius) ** (
        np.abs(lOAM)) * y_dependence(y - y0) *
                   x_dependence(x - x0) * t_dependence(t - t0) * np.exp(
                -1j * k0 * radius(x - x0, y - x0) ** 2 / (2 * f))
                   )


# %% general functions
def radius(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def phi(x, t):
    return np.angle(x + 1j * t)


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
    deltaD = -1 * (zArray[1] - zArray[0]) * k2Dis / (4 * (tArray[1] - tArray[0]) ** 2)
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
        # eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def plasma_density0(E):
        plasmaDensity = np.zeros((xResolution, tResolution))
        # print((abs(E)** (2 * K)).max())
        for i in range(tResolution):
            plasmaDensity[:, i] = 2 * (
                    sigma_K8 * abs(E[:, int(tResolution / 2)]) ** (2 * K) * (np.sqrt(np.pi / (8 * K)))
                    * rho_at * tFinish * 0.1)
        """for i in range(xResolution):
            for j in range(tResolution):
                if plasmaDensity[i, j]>=rho_at:
                    plasmaDensity[i, j]=0.9 * rho_at"""
        """plasmaDensity = np.zeros((xResolution, tResolution))
        #print((rho_at + plasmaDensity[:, :]).max())

        for i in range(tResolution - 1):

            plasmaDensity[:, i + 1] = (plasmaDensity[:, i] + (tArray[1] - tArray[0]) *
                                       (sigma_K8 * abs(E[:, i]) ** (2 * K) * (rho_at - plasmaDensity[:, i])
                                        + sigma / Ui * abs(E[:, i] * 1) ** 2 * plasmaDensity[:, i]
                                        - a * plasmaDensity[:, i] ** 2))"""

        return plasmaDensity

    def plasma_density(E):
        plasmaDensity = np.zeros((xResolution, tResolution))

        # TUUUUUUUUUUT SIGMAAAAAAAAAAAAA
        """def Sigma(w):
            return (w0 / (n * cSOL * rhoC)) * ((w0 * tauC * (1 + 1j * w * tauC)) / (1 + w ** 2 * tauC ** 2))

        """

        def Wofi(I):
            return sigma_K8 * I ** (K)

        def Wava(I):
            return sigma * I / Ui

        def Q_pd(I):
            return Wofi(I)

        def a_pd(I1, I2):
            tempValue = (tArray[1] - tArray[0]) * ((Wofi(I1) - Wava(I1)) + (Wofi(I2) - Wava(I2))) / 2
            return np.exp(-1 * tempValue)

        etta_pd = (tArray[1] - tArray[0]) * rho_at / 2

        for i in range(tResolution - 1):
            plasmaDensity[:, i + 1] = (a_pd(I(E[:, i]), I(E[:, i + 1])) *
                                       (plasmaDensity[:, i] + etta_pd * Q_pd(I(E[:, i])))
                                       + etta_pd * Q_pd(I(E[:, i + 1])))
        return plasmaDensity

    def Nonlinearity(E):
        # print(w0 / cSOL * n2)
        # print((1 / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * epsNL)
        # print(((1j / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * epsNL * I(E)).max(),
        #      (Betta_func(K) / 2 * I(E) ** (K - 1)).max())
        Enew = E * ((1j / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * epsNL * I(E)
                    - Betta_func(K) / 2 * I(E) ** (K - 1) * (1 - plasmaDensity / rho_at)
                    - sigma / 2 * (1 + 1j * w0 * tau_c) * plasmaDensity)  # (w0 + KxywMesh[2])
        # print(((1j / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * epsNL * I(E)).max(),(Betta_func(K) / 2 * I(E) ** (K - 1) * (1 - plasmaDensity / rho_at)).max())
        return Enew

    E = E0  #

    # plasmaDensity = np.zeros((xResolution, tResolution))
    plasmaDensity = plasma_density(E)
    Nn_2 = (zArray[1] - zArray[0]) * Nonlinearity(E)

    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            # n = k * loopInnerM + m + 1
            Nn_1 = (zArray[1] - zArray[0]) * Nonlinearity(E)
            # temp?
            plasmaDensity = plasma_density(E)
            #
            E = np.dot(LPlusMatrixD, E.transpose())  #
            Vn_1 = np.dot(LPlusMatrix, E.transpose())

            Sn_1 = Vn_1 + (3 * Nn_1 - Nn_2) / 2
            Nn_2 = Nn_1

            # Perform Costless Diagnostic
            E = np.dot(LMinusMatrix, Sn_1)

            E = np.dot(LMinusMatrixD, E.transpose())  # ##
            E = E.transpose()  #

            # print(abs(E * 1e-6).max())

    return E


def ADI_2D1_nonlinear_Z(E0, loopInnerM, loopOuterKmax):
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
    deltaD = -1 * (zArray[1] - zArray[0]) * k2Dis / (4 * (tArray[1] - tArray[0]) ** 2)
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
        # eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def plasma_density0(E):
        plasmaDensity = np.zeros((xResolution, tResolution))
        # print((abs(E)** (2 * K)).max())
        for i in range(tResolution):
            plasmaDensity[:, i] = 2 * (
                    sigma_K8 * abs(E[:, int(tResolution / 2)]) ** (2 * K) * (np.sqrt(np.pi / (8 * K)))
                    * rho_at * tFinish * 0.1)
        """for i in range(xResolution):
            for j in range(tResolution):
                if plasmaDensity[i, j]>=rho_at:
                    plasmaDensity[i, j]=0.9 * rho_at"""
        """plasmaDensity = np.zeros((xResolution, tResolution))
        #print((rho_at + plasmaDensity[:, :]).max())

        for i in range(tResolution - 1):

            plasmaDensity[:, i + 1] = (plasmaDensity[:, i] + (tArray[1] - tArray[0]) *
                                       (sigma_K8 * abs(E[:, i]) ** (2 * K) * (rho_at - plasmaDensity[:, i])
                                        + sigma / Ui * abs(E[:, i] * 1) ** 2 * plasmaDensity[:, i]
                                        - a * plasmaDensity[:, i] ** 2))"""

        return plasmaDensity

    def plasma_density(E):
        plasmaDensity = np.zeros((xResolution, tResolution))

        # TUUUUUUUUUUT SIGMAAAAAAAAAAAAA
        """def Sigma(w):
            return (w0 / (n * cSOL * rhoC)) * ((w0 * tauC * (1 + 1j * w * tauC)) / (1 + w ** 2 * tauC ** 2))

        """

        def Wofi(I):
            return sigma_K8 * I ** (K)

        def Wava(I):
            return sigma * I / Ui

        def Q_pd(I):
            return Wofi(I)

        def a_pd(I1, I2):
            tempValue = (tArray[1] - tArray[0]) * ((Wofi(I1) - Wava(I1)) + (Wofi(I2) - Wava(I2))) / 2
            return np.exp(-1 * tempValue)

        etta_pd = (tArray[1] - tArray[0]) * rho_at / 2

        for i in range(tResolution - 1):
            plasmaDensity[:, i + 1] = (a_pd(I(E[:, i]), I(E[:, i + 1])) *
                                       (plasmaDensity[:, i] + etta_pd * Q_pd(I(E[:, i])))
                                       + etta_pd * Q_pd(I(E[:, i + 1])))
        return plasmaDensity

    def Nonlinearity(E):
        # print(w0 / cSOL * n2)
        # print((1 / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * epsNL)
        # print(((1j / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * epsNL * I(E)).max(),
        #      (Betta_func(K) / 2 * I(E) ** (K - 1)).max())
        Enew = E * ((1j / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * epsNL * I(E)
                    - Betta_func(K) / 2 * I(E) ** (K - 1) * (1 - plasmaDensity / rho_at)
                    - sigma / 2 * (1 + 1j * w0 * tau_c) * plasmaDensity)  # (w0 + KxywMesh[2])
        # print(((1j / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * epsNL * I(E)).max(),(Betta_func(K) / 2 * I(E) ** (K - 1) * (1 - plasmaDensity / rho_at)).max())
        return Enew

    E = E0  #

    # plasmaDensity = np.zeros((xResolution, tResolution))
    plasmaDensity = plasma_density(E)
    Nn_2 = (zArray[1] - zArray[0]) * Nonlinearity(E)
    fieldReturn = np.zeros((xResolution, zResolution), dtype=complex)
    time = int(tResolution / 2)
    fieldReturn[:, 0] = E[:, time]
    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            n = k * loopInnerM + m + 1
            Nn_1 = (zArray[1] - zArray[0]) * Nonlinearity(E)
            # temp?
            plasmaDensity = plasma_density(E)
            #
            E = np.dot(LPlusMatrixD, E.transpose())  #
            Vn_1 = np.dot(LPlusMatrix, E.transpose())

            Sn_1 = Vn_1 + (3 * Nn_1 - Nn_2) / 2
            Nn_2 = Nn_1

            # Perform Costless Diagnostic
            E = np.dot(LMinusMatrix, Sn_1)

            E = np.dot(LMinusMatrixD, E.transpose())  # ##
            E = E.transpose()  #
            # print(abs(E * 1e-6).max())
            fieldReturn[:, (k) * loopInnerM + m] = E[:, time]

            # fieldReturn[]

    return fieldReturn


# 3D+1 Time # FIX LATER
def split_step_old_time(shape, loopInnerM=1, loopOuterKmax=1):
    def I(E):
        # page 44
        return np.abs(E) ** 2

    dz = zArray[1] - zArray[0]

    def plasma_density0(E):
        plasmaDensity = np.zeros((xResolution, yResolution, tResolution))
        # print((abs(E)** (2 * K)).max())
        for i in range(tResolution):
            plasmaDensity[:, :, i] = 2 * (
                    sigma_K8 * abs(E[:, :, int(tResolution / 2)]) ** (2 * K) * (np.sqrt(np.pi / (8 * K)))
                    * rho_at * tFinish * 0.1)

        """for i in range(tResolution - 1):
            for j in range(xResolution):
                for m in range(yResolution):
                        if plasmaDensity[j, m, i] > rho_at:
                            plasmaDensity[j, m, i] = rho_at-1
            print((a * plasmaDensity[:, :, i] ** 2).max())
            plasmaDensity[:, :, i + 1] = (plasmaDensity[:, :, i] + (tArray[1] - tArray[0]) *
                                       (sigma_K8 * abs(E[:, :, i]) ** (2 * K) * (rho_at - plasmaDensity[:, :, i])
                                        + sigma / Ui * abs(E[:, :, i]) ** 2 * plasmaDensity[:, :, i]
                                        - a * plasmaDensity[:, :, i] ** 2))"""

        # print((plasmaDensity ** 2).max())
        return plasmaDensity

    def plasma_density0(E):
        plasmaDensity = np.zeros((xResolution, yResolution, tResolution))

        for i in range(tResolution - 1):
            """for j in range(xResolution):
                for m in range(yResolution):
                        if plasmaDensity[j, m, i] > rho_at:
                            plasmaDensity[j, m, i] = rho_at-1"""
            plasmaDensity[:, :, i + 1] = (plasmaDensity[:, :, i] + (tArray[1] - tArray[0]) *
                                          (sigma_K8 * abs(E[:, :, i]) ** (2 * K) * (rho_at - plasmaDensity[:, :, i])
                                           + sigma / Ui * abs(E[:, :, i]) ** 2 * plasmaDensity[:, :, i]
                                           - a * plasmaDensity[:, :, i] ** 2))

        # print((plasmaDensity ** 2).max())
        return plasmaDensity

    def plasma_density(E):
        plasmaDensity = np.zeros((xResolution, yResolution, tResolution))

        # TUUUUUUUUUUT SIGMAAAAAAAAAAAAA
        """def Sigma(w):
            return (w0 / (n * cSOL * rhoC)) * ((w0 * tauC * (1 + 1j * w * tauC)) / (1 + w ** 2 * tauC ** 2))

        """

        def Wofi(I):
            return sigma_K8 * I ** (K)

        def Wava(I):
            return sigma * I / Ui

        def Q_pd(I):
            return Wofi(I)

        def a_pd(I1, I2):
            tempValue = (tArray[1] - tArray[0]) * ((Wofi(I1) - Wava(I1)) + (Wofi(I2) - Wava(I2))) / 2
            return np.exp(-1 * tempValue)

        etta_pd = (tArray[1] - tArray[0]) * rho_at / 2

        for i in range(tResolution - 1):
            plasmaDensity[:, :, i + 1] = (a_pd(I(E[:, :, i]), I(E[:, :, i + 1])) *
                                          (plasmaDensity[:, :, i] + etta_pd * Q_pd(I(E[:, :, i])))
                                          + etta_pd * Q_pd(I(E[:, :, i + 1])))
            # - 0 * (tArray[1] - tArray[0]) * a * plasmaDensity[:, :, i] ** 2
        """
                                          + (tArray[1] - tArray[0]) *
         (sigma_K8 * abs(E[:, :, i]) ** (2 * K) * (rho_at - plasmaDensity[:, :, i])
          + sigma / Ui * abs(E[:, :, i]) ** 2 * plasmaDensity[:, :, i]
          - a * plasmaDensity[:, :, i] ** 2))
        """

        return plasmaDensity

    def Nonlinearity_spec(E):

        # print((w0 * n2 / cSOL * I(E)).max(), (sigma / 2 * (1) * plasmaDensity).max(),
        #      (sigma / 2 * (1j * w0 * tau_c) * plasmaDensity).max())
        # print((plasmaDensity).max())
        return dz * ((1j / (2 * eps0)) * ((w0 + KxywMesh[2]) / cSOL / n0) * eps0 * epsNL * I(E)
                     - Betta_func(K) / 2 * I(E) ** (K - 1) * (1 - plasmaDensity / rho_at)
                     - sigma / 2 * (1 + 1j * w0 * tau_c) * plasmaDensity)
        # return dz * 1j * w0 * n2 * I(E) / cSOL - dz * Betta_func(K) * I(E) ** (K - 1)  # & E

    E = shape(xytMesh[0], xytMesh[1], xytMesh[2])
    print(E[int(xResolution / 2), int(yResolution / 2), int(tResolution / 2)])
    """print('look here', rho0**2 / (2 / (k0 * n0)))
    print(2 * tp ** 2 / 4 / (rho0**2 / (2 / (k0 * n0))))
    exit()"""

    # works fine!
    def linear_step(field):
        temporaryField = fftshift(fftn(field))
        temporaryField = (temporaryField *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxywMesh[0] ** 2) *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxywMesh[1] ** 2) *
                          np.exp(1j * dz * k2Dis / 2 * KxywMesh[2] ** 2))  # something here in /2
        return ifftn(ifftshift(temporaryField))

    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            plasmaDensity = plasma_density(E)
            E = linear_step(E)
            E = E * np.exp(Nonlinearity_spec(E))

    # plt.plot(tArray, abs(plasmaDensity[int(xResolution/2), int(xResolution/2), :]))
    # plt.show()
    # exit()
    return E


def split_step_old_time_Z(shape, loopInnerM=1, loopOuterKmax=1):
    def I(E):
        # page 44
        return np.abs(E) ** 2

    dz = zArray[1] - zArray[0]

    def plasma_density0(E):
        plasmaDensity = np.zeros((xResolution, yResolution, tResolution))
        # print((abs(E)** (2 * K)).max())
        for i in range(tResolution):
            plasmaDensity[:, :, i] = 2 * (
                    sigma_K8 * abs(E[:, :, int(tResolution / 2)]) ** (2 * K) * (np.sqrt(np.pi / (8 * K)))
                    * rho_at * tFinish * 0.1)

        """for i in range(tResolution - 1):
            for j in range(xResolution):
                for m in range(yResolution):
                        if plasmaDensity[j, m, i] > rho_at:
                            plasmaDensity[j, m, i] = rho_at-1
            print((a * plasmaDensity[:, :, i] ** 2).max())
            plasmaDensity[:, :, i + 1] = (plasmaDensity[:, :, i] + (tArray[1] - tArray[0]) *
                                       (sigma_K8 * abs(E[:, :, i]) ** (2 * K) * (rho_at - plasmaDensity[:, :, i])
                                        + sigma / Ui * abs(E[:, :, i]) ** 2 * plasmaDensity[:, :, i]
                                        - a * plasmaDensity[:, :, i] ** 2))"""

        # print((plasmaDensity ** 2).max())
        return plasmaDensity

    def plasma_density0(E):
        plasmaDensity = np.zeros((xResolution, yResolution, tResolution))

        for i in range(tResolution - 1):
            """for j in range(xResolution):
                for m in range(yResolution):
                        if plasmaDensity[j, m, i] > rho_at:
                            plasmaDensity[j, m, i] = rho_at-1"""
            plasmaDensity[:, :, i + 1] = (plasmaDensity[:, :, i] + (tArray[1] - tArray[0]) *
                                          (sigma_K8 * abs(E[:, :, i]) ** (2 * K) * (rho_at - plasmaDensity[:, :, i])
                                           + sigma / Ui * abs(E[:, :, i]) ** 2 * plasmaDensity[:, :, i]
                                           - a * plasmaDensity[:, :, i] ** 2))

        # print((plasmaDensity ** 2).max())
        return plasmaDensity

    def plasma_density(E):
        plasmaDensity = np.zeros((xResolution, yResolution, tResolution))
        if (tResolution == 1):
            plasmaDensity[:, :, 0] = (tFinish *
                                      (sigma_K8 * abs(E[:, :, 0]) ** (2 * K) * (rho_at)
                                       + sigma / Ui * abs(E[:, :, 0]) ** 2 * plasmaDensity[:, :, 0]
                                       ))
            return plasmaDensity
            # TUUUUUUUUUUT SIGMAAAAAAAAAAAAA
            """def Sigma(w):
                return (w0 / (n * cSOL * rhoC)) * ((w0 * tauC * (1 + 1j * w * tauC)) / (1 + w ** 2 * tauC ** 2))
    
            """
        else:

            def Wofi(I):
                return sigma_K8 * I ** (K)

            def Wava(I):
                return sigma * I / Ui

            def Q_pd(I):
                return Wofi(I)

            def a_pd(I1, I2):
                tempValue = (tArray[1] - tArray[0]) * ((Wofi(I1) - Wava(I1)) + (Wofi(I2) - Wava(I2))) / 2
                return np.exp(-1 * tempValue)

            etta_pd = (tArray[1] - tArray[0]) * rho_at / 2

            for i in range(tResolution - 1):
                plasmaDensity[:, :, i + 1] = (a_pd(I(E[:, :, i]), I(E[:, :, i + 1])) *
                                              (plasmaDensity[:, :, i] + etta_pd * Q_pd(I(E[:, :, i])))
                                              + etta_pd * Q_pd(I(E[:, :, i + 1])))
                # - 0 * (tArray[1] - tArray[0]) * a * plasmaDensity[:, :, i] ** 2
            """
                                              + (tArray[1] - tArray[0]) *
             (sigma_K8 * abs(E[:, :, i]) ** (2 * K) * (rho_at - plasmaDensity[:, :, i])
              + sigma / Ui * abs(E[:, :, i]) ** 2 * plasmaDensity[:, :, i]
              - a * plasmaDensity[:, :, i] ** 2))
            """

            return plasmaDensity

    def Nonlinearity_spec(E):

        # print((w0 * n2 / cSOL * I(E)).max(), (sigma / 2 * (1) * plasmaDensity).max(),
        #      (sigma / 2 * (1j * w0 * tau_c) * plasmaDensity).max())
        # print((plasmaDensity).max())
        return dz * ((1j / (2 * eps0)) * ((w0 + KxywMesh[2]) / cSOL / n0) * eps0 * epsNL * I(E)
                     - Betta_func(K) / 2 * I(E) ** (K - 1) * (1 - plasmaDensity / rho_at)
                     - sigma / 2 * (1 + 1j * w0 * tau_c) * plasmaDensity)
        # return dz * 1j * w0 * n2 * I(E) / cSOL - dz * Betta_func(K) * I(E) ** (K - 1)  # & E

    E = shape(xytMesh[0], xytMesh[1], xytMesh[2])
    print(E[int(xResolution / 2), int(yResolution / 2), int(tResolution / 2)])
    """print('look here', rho0**2 / (2 / (k0 * n0)))
    print(2 * tp ** 2 / 4 / (rho0**2 / (2 / (k0 * n0))))
    exit()"""

    # works fine!
    def linear_step(field):
        temporaryField = fftshift(fftn(field))
        temporaryField = (temporaryField *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxywMesh[0] ** 2) *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxywMesh[1] ** 2) *
                          np.exp(1j * dz * k2Dis / 2 * KxywMesh[2] ** 2))  # something here in /2
        return ifftn(ifftshift(temporaryField))

    fieldReturn = np.zeros((xResolution, yResolution, zResolution), dtype=complex)

    # y = int(yResolution / 2)
    # tArray[int(tResolution/2)]
    # tArray[time]
    time = int(tResolution / 2)
    fieldReturn[:, :, 0] = E[:, :, time]

    for k in range(loopOuterKmax):
        if module_CheckingSpectrum:
            Etest = abs((E))
            plt.plot(xArray, Etest[:, int(yResolution / 2), int(tResolution / 2)])
            plt.show()
            plt.close()
            plt.plot(yArray, Etest[int(xResolution / 2), :, int(tResolution / 2)])
            plt.show()
            plt.close()
            plt.plot(tArray, Etest[int(xResolution / 2), int(yResolution / 2), :])
            plt.show()
            plt.close()
            Etest = abs(fftshift(fftn(E)))
            plt.plot(kxArray, Etest[:, int(yResolution / 2), int(tResolution / 2)])
            plt.show()
            plt.close()
            plt.plot(kyArray, Etest[int(xResolution / 2), :, int(tResolution / 2)])
            plt.show()
            plt.close()
            plt.plot(wArray, Etest[int(xResolution / 2), int(yResolution / 2), :])
            plt.show()
            plt.close()

        for m in range(1, loopInnerM):
            zInd = (k) * loopInnerM + m
            plasmaDensity = plasma_density(E)
            E = linear_step(E)
            E = E * np.exp(Nonlinearity_spec(E))
            fieldReturn[:, :, zInd] = E[:, :, time]
        if module_CheckingSpectrum:
            Etest = abs(fftshift(fftn(E)))
            plt.plot(kxArray, Etest[:, int(yResolution / 2), int(tResolution / 2)])
            plt.show()
            plt.close()
            plt.plot(kyArray, Etest[int(xResolution / 2), :, int(tResolution / 2)])
            plt.show()
            plt.close()
            plt.plot(wArray, Etest[int(xResolution / 2), int(yResolution / 2), :])
            plt.show()
            plt.close()

    # plt.plot(tArray, abs(plasmaDensity[int(xResolution/2), int(xResolution/2), :]))
    # plt.show()
    # exit()
    return fieldReturn


# %% UPPE with time
def UPPE_time(shape, loopInnerM, loopOuterKmax):
    def I(E):
        # return eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def Nonlinearity(E):
        # Pe = 1j * w0 * n2 * E * I(E) / cSOL - Betta_func(K) * E * I(E) ** (K - 1)
        # Pe = n2 * E * I(E) - Betta_func(K) * E * I(E) ** (K - 1)
        Pe = eps0 * epsNL * E * I(E)  # - Betta_func(K) * E * I(E) ** (K - 1)
        """print(abs(Pe).max())

        print(w0 * n2 / cSOL)
        print((1j / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * epsNL)
        print(eps0 * epsNL)
        exit()"""
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

    if module_CheckingSpectrum:
        Aspec = np.reshape(Aspec, (xResolution, yResolution, tResolution))
        fig, ax = plt.subplots(figsize=(8, 7))
        # Pspec = np.reshape(np.real(P), (xResolution, yResolution, tResolution))
        ax.plot(np.abs(Aspec[:, int(yResolution / 2), int(tResolution / 2)]), color='b', lw=6, label='x')
        ax.plot(np.abs(Aspec[int(xResolution / 2), :, int(tResolution / 2)]), color='lime', lw=2.5, label='y')
        ax.plot(np.abs(Aspec[int(xResolution / 2), int(yResolution / 2), :]), color='r', lw=4, label='t')
        ax.legend(shadow=True, fontsize=legendFontSize, facecolor='white', edgecolor='black', loc='upper right')
        plt.show()
    for k in range(loopOuterKmax):
        if module_CheckingSpectrum:
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

            integrator.set_initial_value(Aspec, 0)
            Aspec = integrator.integrate(dz)
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
            fig, ax = plt.subplots(figsize=(8, 7))
            Aspec = np.reshape(Aspec, (xResolution, yResolution, tResolution))
            ax.plot(np.abs(Aspec[:, int(yResolution / 2), int(tResolution / 2)]), color='b', lw=6, label='x')
            ax.plot(np.abs(Aspec[int(xResolution / 2), :, int(tResolution / 2)]), color='lime', lw=2.5, label='y')
            ax.plot(np.abs(Aspec[int(xResolution / 2), int(yResolution / 2), :]), color='r', lw=4, label='t')
            ax.legend(shadow=True, fontsize=legendFontSize, facecolor='white', edgecolor='black', loc='upper right')
            plt.show()
    Aspec = np.reshape(Aspec, (xResolution, yResolution, tResolution))
    E = ifftn(ifftshift(Aspec))

    return E


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


def plot_2D(E, x, y, xname='', yname='', map='jet', vmin=0.13, vmax=1.14):
    if vmin == 0.13 and vmax == 1.14:
        vmin = E.min()
        vmax = E.max()

    image = plt.imshow(E,
                       interpolation='bilinear', cmap=map,
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[y[0], y[-1], x[0], x[-1]],
                       vmin=vmin, vmax=vmax, label='sdfsd')
    cbr = plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    cbr.ax.tick_params(labelsize=ticksFontSize)
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize)
    ax.set_ylabel(yname, fontsize=xyLabelFontSize)


def plot_3D(field3D):
    X, Y, Z = np.mgrid[xStart:xFinish:(1j * xResolution), yStart:yFinish:(1j * yResolution),
              tStart:tFinish:1j * tResolution]
    """X, Y, Z = np.mgrid[xStart:xFinish:(1j * xResolution), xStart:xFinish:(1j * yResolution),
              xStart:xFinish:1j * tResolution]"""
    # print(X)

    values = abs(field3D) ** 2
    max = values.max()
    values = values / max * 100

    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        opacity=0.6,
        isomin=40,
        isomax=40,
        surface_count=1,  # number of isosurfaces, 2 by default: only min and max
        caps=dict(x_show=False, y_show=False)
    ))
    fig.show()


if __name__ == '__main__':
    if module_HOBBIT:
        fieldTEMP = Field_1
        E = fieldTEMP(xytMesh[0], xytMesh[1], xytMesh[2])[:, :, int(tResolution / 2)]

        # E = Hobbit(1, 2, 3, lOAM, k)
        Sq = np.sum(np.abs(E) ** 2) * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
        # P_cr = P_critical_initialization(wavelength)  # дж / с
        Imax = np.sqrt(Pmax / Sq)
        print(Imax)

        fig, ax = plt.subplots(figsize=(8, 7))
        """fieldAdiZ = ADI_2D1_nonlinear_Z(Field_1_2D(xtMesh[0], xtMesh[1]),
                                     loopInnerResolution, loopOuterResolution)"""
        fieldOLD = split_step_old_time_Z(fieldTEMP, loopInnerResolution, loopOuterResolution)
        print(fieldOLD)
        plot_2D(np.abs(fieldOLD[:, int(yResolution / 2), :]) ** Int, xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
                map='magma')
        # plt.xlim(2.25 - 1, 2.25 + 1)
        # plt.ylim(2.25 - 1, 2.25 + 1)
        # plt.title(f't={round(time * 1e15, 0)}fs', fontweight="bold",
        #           fontsize=26)
        plt.show()
        plt.close()
        if save:
            np.save(save_name, fieldOLD)
        exit()
    if module_SUM:
        """fig, ax = plt.subplots(figsize=(8, 7))
        fieldOLD = (sum_fields(xytMesh[0], xytMesh[1], xytMesh[2]))
        plot_2D(np.abs(fieldOLD[:, :, int(tResolution / 2)]) ** Int, xArray * 1e3, yArray * 1e3, 'y (mm)', 'x (mm)',
                map='nipy_spectral') # gist_ncar
        plt.show()
        exit()"""
        fieldTEMP = asymmetric_LG
        E = fieldTEMP(xytMesh[0], xytMesh[1], xytMesh[2])[:, :, int(tResolution / 2)]

        # E = Hobbit(1, 2, 3, lOAM, k)
        Sq = np.sum(np.abs(E) ** 2) * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
        # P_cr = P_critical_initialization(wavelength)  # дж / с
        Imax = np.sqrt(Pmax / Sq)
        print(Imax)

        fig, ax = plt.subplots(figsize=(8, 7))
        """fieldAdiZ = ADI_2D1_nonlinear_Z(Field_1_2D(xtMesh[0], xtMesh[1]),
                                     loopInnerResolution, loopOuterResolution)"""
        fieldOLD = split_step_old_time_Z(sum_fields, loopInnerResolution, loopOuterResolution)
        plot_2D(np.abs(fieldOLD[:, int(yResolution / 2), :]) ** Int, xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
                map='magma')
        # plt.xlim(2.25 - 1, 2.25 + 1)
        # plt.ylim(2.25 - 1, 2.25 + 1)
        plt.title(f't={round(time * 1e15, 0)}fs', fontweight="bold",
                  fontsize=26)
        plt.show()
        plt.close()
        if save:
            np.save(save_name, fieldOLD)
        plt.show()
        exit()
