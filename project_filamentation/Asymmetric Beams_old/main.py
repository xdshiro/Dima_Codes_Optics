import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.integrate import odeint, complex_ode, ode
from scipy.special import erf, jv, iv, assoc_laguerre, eval_genlaguerre
from functions import *

module_CheckingSpectrum = 0

module_Paraxial = 0
module_Analit = 1
module_Phase = 1
module_Intensity = 1  # 1-yes, 0-field abs


# %% Resolutions
xResolution = 356  # N_per 2 661
yResolution = 356
tResolution = 1
loopInnerResolution, loopOuterResolution = 21, 1  # M, Kmax
zResolution = loopInnerResolution * loopOuterResolution
xStart, xFinish = 0, 10 * 1e-3  # 10
yStart, yFinish = 0, 10 * 1e-3  # 10
x0 = (xStart + xFinish) / 2
y0 = (yStart + yFinish) / 2 #+ 0j * 1e-6
zStart, zFinish = 1e-5, 4.0
tStart, tFinish = 0, 1000 * 1e-13
# #
t1, t2 = 0 * 100 * 1e-15, 0 * -100 * 1e-15

################################### temp ######################################
sigma_K8 = 2.4e-42 * 1e-2 ** (2 * 4)
sigma = 4e-18 * 1e-2 ** 2  #
rho_at = 7e22 * 1e-2 ** (-3)
a = 0
tau_c = 3e-15
###
# %% Pulse parameters
# 70 * 1e-6
rho0 = 0.5 * 1e-3
c = 5e5 * rho0 / 40.
alpha = 2. / rho0
tp = 200 * 1e-4  # pulse duration
lambda0 = 0.532e-6
Pmax = 0e6  # 3e6 / 1.8**2  # 2.5e7

# %% Hobbit parameters

# STOV
yRadius = 800 * 1e-6
xSTOVRadius = rho0
tSTOVRadius = tp
lOAM = 2
p = 0

t0 = (tFinish - tStart) / 2
f = 1e10 * 0.025e0
# linear medium parameter

k2Dis = 1e-50  # k2Dis = 5.6e-28 / 1e-2  # ps2/m  GVD ????????????????????????????????????????????????????????????????
# k2Dis = -9.443607756116762e-22  # OAM non-diffraction
n0 = 1  # n0 = 1.332 ????????????????????????????????????????????????????????????????
# %% Nonlinear parameters
K = 4  # photons number
# %% temporal
C = 0  # chirp
n2 = 0 * 2.7e-16 * 1e-2 ** 2  # 3 * chi3 / (4 * eps0 * c * n0 ** 2)
# kDis = 0 * 0.02  # dispersion
q_e = -1.602176565e-19  # [C] electron charge
Ui = 7.1 * abs(q_e)
SigmaK = [0, 1, 2]


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

chi3_2 = 8 * n0 * n2 / 3
epsNL = 3 * chi3_2 / 4
Int = module_Intensity + 1
Imax = 1
if n2 == 0:
    Pcrit = 1e100
else:
    Pcrit = 1.22 ** 2 * np.pi * lambda0 ** 2 / (32 * n0 * n2)
print("P crit (MW): ", Pcrit * 1e-6)


def asymmetric_Gauss(x, y, z, l=lOAM, p=0, width=rho0):
    zR = rayleigh_range(lambda0, width)
    x = x - x0 + 1j * 1e-2
    y = y - y0
    width = 1e-2
    def rho(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def width_z(z):
        return width * np.sqrt(1 + (z / zR) ** 2)

    def E0():
        return 1

    def R(z):
        return z * (1 + (zR / z) ** 2)

    def ksi(z):
        return np.angle(z / zR)

    return (E0() * (width / width_z(z)) * np.exp(-rho(x, y) / (width_z(z)) ** 2)
            * np.exp(-1j * (k0 * z + k0 * rho(x, y) ** 2 / 2 / R(z) - ksi(z))))

def asymmetric_LG(x, y, z, l=lOAM, p=0, width=rho0):
    x = x - x0
    y = y - y0 + 1j * 1e-3 / 6
    zR = rayleigh_range(lambda0, width)
    print("Rayleigh Range: ", zR)
    #z = 1e-20

    def rho(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def width_z(z):
        return width * np.sqrt(1 + (z / zR) ** 2)

    def R(z):
        return z * (1 + (zR / z) ** 2)

    def ksi(z):
        return np.arctan(z / zR)

    def laguerre_polynomial(x, l, p):
        return eval_genlaguerre(p, l, x)
        # return assoc_laguerre(x, l, p)

    def nonlinearity(x, y):
        # y = y - 1j * 1e-3 / 6
        # return x + 1j * np.sign(l) * y
        return (np.sqrt((x) ** 2 + y ** 2) * np.exp(1j * np.arctan(y / (x))))

    #
    #y = y - y0 #+ 1j * 1e-3 / 6
    E = (width_z(0) / width_z(z) * (np.sqrt(2) / width_z(z)) ** (np.abs(l))
         * nonlinearity(x, y) ** (np.abs(l))
         * laguerre_polynomial(2 * rho(x, y) ** 2 / width_z(z) ** 2, np.abs(l), p)
         * np.exp(-rho(x, y) ** 2 / width_z(z) ** 2 + 1j * k0 * rho(x, y) ** 2/ (2 * R(z))
                  - 1j * (np.abs(l) + 2 * p + 1) * ksi(z)))
    #E=np.exp(-rho(x, y) ** 2 / width_z(z) ** 2) * np.exp( 1j * k0 * rho(x, y) ** 2 / (2 * R(z)))
    return E

def asymmetric_BG(x, y, z, c, l=lOAM, width=rho0, alphaPar=-13.):
    x = x - x0
    y = y - y0
    theta0 = 0  # ???????????????????????????????????????????
    print(x0, y0)
    #z = 1e-20
    if alphaPar == -13:
        alpha = k0 * np.sin(theta0)
    else:
        alpha = alphaPar
    print(alphaPar)
    def phi(x, y):
        return np.angle(x + 1j * y)

    def rho(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def rayleigh_range(lambda0, rho0):
        return np.pi * rho0 ** 2 / lambda0

    def q(z):
        z0 = rayleigh_range(lambda0, width)
        print("Rayleigh Range: ", z0)
        return 1 + 1j * z / z0

    def J(arg, l):
        return jv(l, arg)

    def asymmetry(x, y, z, c):
        return (2 * c * q(z) * np.exp(1j * (phi(x, y))))

    def extra_phase_Dima(x, y):
        return np.exp(0j* (phi(x, y) - 4 * rho(x, y) / rho0))

    print(c, alpha, l, width)
    E = ((1 / q(z))
         * np.exp(1j * k0 * z
                  - 1j * alpha ** 2 * z / (2 * k0 * q(z))
                  - rho(x, y) ** 2 / (q(z) * width ** 2)
                  + 1j * l * phi(x, y))
         * (alpha * rho(x, y)
            / (alpha * rho(x, y) - (2 * c * q(z) * np.exp(1j * (0*phi(x, y)))))) ** (l / 2)
         * J(np.sqrt(alpha * rho(x, y) * (alpha * rho(x, y) - asymmetry(x, y, z, c))) / q(z), l)
         * extra_phase_Dima(x, y))
    """E = ((1 / q(z))
         * np.exp(1j * k0 * z
                  - 1j * alpha ** 2 * z / (2 * k0 * q(0))
                  - rho(x, y) ** 2 / (q(z) * width ** 2)
                  + 1j * l * phi(x, y))
                  * J(alpha * rho(x, y)/ q(z), l))"""
    return np.sqrt(alpha * rho(x, y) * (alpha * rho(x, y) - asymmetry(x, y, z, c))) / q(z)

# print("Rayleigh length: ", zR, " Kerr Collapse length: ", Lcollapse())

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
xyzMesh = np.array(np.meshgrid(xArray, yArray, zArray, indexing='ij'))
KxywMesh = np.array(np.meshgrid(kxArray, kyArray, wArray, indexing='ij'))

def Jz_calc(EArray, xArray, yArray):
    x0 = (xArray[-1] + xArray[0]) / 2
    y0 = (yArray[-1] + yArray[0]) / 2
    sum = 0
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    x = xArray - x0
    y = yArray - y0
    for i in range(1,len(xArray)-1,1):
        for j in range(1,len(yArray)-1,1):
            dEx = (EArray[i + 1, j] - EArray[i -1, j]) / (2 * dx)
            dEy = (EArray[i, j + 1] - EArray[i, j - 1]) / (2 * dy)
            sum += (np.conj(EArray[i, j]) *
                    (x[i] * dEy - y[j] * dEx))

    return np.imag(sum * dx * dy)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if module_Analit:
        fieldOLD = asymmetric_LG(xyzMesh[0], xyzMesh[1], xyzMesh[2], l=lOAM, width=rho0)#, c=c, alphaPar=alpha)
        zparM2 = [int((len(zArray) - 1) / 6 * 0), int((len(zArray) - 1) / 6), int((len(zArray) - 1) / 6 * 2),
                  int((len(zArray) - 1) / 6 * 3),
                  int((len(zArray) - 1) / 6 * 4),
                  int((len(zArray) - 1) / 6 * 5), int((len(zArray) - 1) / 6 * 6)]
        zparM2 = [int((len(zArray) - 1) / 6 * 0), int((len(zArray) - 1) / 6 * 1), int((len(zArray) - 1) * 4 / 6), int((len(zArray) - 1) * 6 / 6)]
        zparM2 = [int((len(zArray) - 1) / 6 * 0),
                  int((len(zArray) - 1) * 6 / 6)]

        W = (np.sum(np.conj(fieldOLD) * fieldOLD, axis=0)
             * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0]))
        W = np.sum(W, axis=0)
        for i in zparM2:
            Jz = Jz_calc(fieldOLD[:, :, i], xArray, yArray)
            print(W[i], Jz / W[i])

        for zpar in zparM2:

            plot_2D(np.abs(fieldOLD[:, :, zpar]), xArray * 1e3, yArray * 1e3, 'y (mm)', 'x (mm)',
                    map='jet', title=f'z={round(zArray[zpar], 1)}m')
            plot_2D(np.real(fieldOLD[:, :, zpar]), xArray * 1e3, yArray * 1e3, 'y (mm)', 'x (mm)',
                    map='jet', title=f'z={round(zArray[zpar], 1)}m')
            if module_Phase:
                plot_2D(np.angle(fieldOLD[:, :, zpar]), xArray * 1e3, yArray * 1e3, 'y (mm)', 'x (mm)',
                        map='binary', vmax=np.pi, vmin=-np.pi, title=f'z={round(zArray[zpar], 1)}m')
    if module_Paraxial:
        fieldTEMP = asymmetric_LG_z0
        fieldOLD = split_step_old_time_Z(fieldTEMP, loopInnerResolution, loopOuterResolution)
        zparM2 = [int((len(zArray) - 1) / 6 * 0), int((len(zArray) - 1) / 6), int((len(zArray) - 1) / 6 * 2),
                  int((len(zArray) - 1) / 6 * 3),
                  int((len(zArray) - 1) / 6 * 4),
                  int((len(zArray) - 1) / 6 * 5), int((len(zArray) - 1) / 6 * 6)]
        for zpar in zparM2:

            plot_2D(np.abs(fieldOLD[:, :, zpar]) ** Int, xArray * 1e3, yArray * 1e3, 'y (mm)', 'x (mm)',
                    map='nipy_spectral', title=f'z={round(zArray[zpar] * 1e3, 2)}mm')

            # plt.title(f'z={round(zArray[zpar] * 1e3, 0)}mm', fontweight="bold", fontsize=26)
            # plt.xlim(0.4, 0.8)
            # plt.ylim(0.4, 0.8)
            if module_Phase:
                plot_2D(np.angle(fieldOLD[:, :, zpar]), xArray * 1e3, yArray * 1e3, 'y (mm)', 'x (mm)',
                        map='hsv', vmax=np.pi, vmin=-np.pi, title=f'z={round(zArray[zpar] * 1e3, 2)}mm')
            # plt.title(f'z={round(zArray[zpar] * 1e3, 0)}mm', fontweight="bold",
            #          fontsize=26)
            # plt.xlim(0.4, 0.8)
            # plt.ylim(0.4, 0.8)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
