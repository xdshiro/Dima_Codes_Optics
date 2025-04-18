import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.integrate import odeint, complex_ode, ode
from scipy.special import jv, iv
import plotly.graph_objects as go
# %% Modules
# checking spectra
module_checking_spectrum = 1  # Flag for checking spectral properties
module_Paraxial = 1  # you can choose either paraxial or non paraxial
module_NonRapaxial = 0
module_Intensity = 1  # 1-yes, 0-field abs
module_3D = 1



# %% Resolutions
xResolution = 71 # N_per
yResolution = 71
tResolution = 71
loopInnerResolution, loopOuterResolution = 51, 1  # M, Kmax
zResolution = loopInnerResolution * loopOuterResolution
xStart, xFinish = 0, 4500 * 1e-6  # 10
yStart, yFinish = 0, 4500 * 1e-6  # 10
zStart, zFinish = 0, 1.5
tStart, tFinish = 0, 400 * 1e-12


###
# %% Pulse parameters
rho0 = 800 * 1e-6
tp = 70 * 1e-12  # pulse duration
lambda0 = 0.775e-6
Pmax = 4500e6 #2.5e7
# STOV
yRadius = 800 * 1e-6
xSTOVRadius = rho0
tSTOVRadius = tp
lOAM = 1

x0 = (xFinish - xStart) / 2
y0 = (yFinish - yStart) / 2
t0 = (tFinish - tStart) / 2
f = 1e20  # focusing. for now it's infinite focus
# linear medium parameter

k2Dis = 2.0 * 1e-15 ** 2 / 1e-2  # ps2/m  GVD
n0 = 1.00
# %% Nonlinear parameters
K = 1  # photons number
# %% temporal
C = 0  # chirp
# set to 0 to look at stove
n2 = 0 * 5.57e-19 * 1e-2 ** 2  # 3 * chi3 / (4 * eps0 * c * n0 ** 2)
Ui = 1



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
wD = 2 * n0 / (k2Dis * cSOL)

chi3_2 = 8 * n0 * n2 / 3
epsNL = 3 * chi3_2 / 4
Int = module_Intensity + 1
Imax = 1
if n2==0:
    Pcrit=1e100
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

print("Rayleigh length: ", LDF()," Kerr Collapse length: ", Lcollapse())

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
            x_dependence(x - x0) * t_dependence(t - t0) *np.exp( -1j * k0 * radius(x - x0, y - x0) ** 2 / (2 * f))
            )


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



# 3D+1 Time # FIX LATER
def split_step_old_time(shape, loopInnerM=1, loopOuterKmax=1):
    def I(E):
        eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    dz = zArray[1] - zArray[0]

    def Nonlinearity_spec(E):

        #print((1j / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * chi3)
        """print(1j * w0 * n2 / cSOL)
        print((1j / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * chi3_2)
        print((1j / (2 * eps0)) * ((w0) / cSOL / n0) * eps0 * epsNL)
        exit()"""

        return dz * (1j / (2 * eps0)) * ((w0 + KxywMesh[2]) / cSOL / n0) * eps0 * epsNL * I(E) - dz * Betta_func(K) * I(E) ** (
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
                          np.exp(1j * dz * k2Dis / 2 * KxywMesh[2] ** 2))  # something here in /2
        return ifftn(ifftshift(temporaryField))

    for k in range(loopOuterKmax):
        for m in range(1, loopInnerM):
            E = linear_step(E)
            E = E * np.exp(Nonlinearity_spec(E))

    return E


# %% UPPE with time
def UPPE_time(shape, loopInnerM, loopOuterKmax):
    def I(E):
        # return eps0 * cSOL * n0 * np.abs(E) ** 2 / 2
        # page 44
        return np.abs(E) ** 2

    def Nonlinearity(E):
        # Pe = 1j * w0 * n2 * E * I(E) / cSOL - Betta_func(K) * E * I(E) ** (K - 1)
        #Pe = n2 * E * I(E) - Betta_func(K) * E * I(E) ** (K - 1)
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


    if module_checking_spectrum:
        Aspec = np.reshape(Aspec, (xResolution, yResolution, tResolution))
        fig, ax = plt.subplots(figsize=(8, 7))
        # Pspec = np.reshape(np.real(P), (xResolution, yResolution, tResolution))
        ax.plot(np.abs(Aspec[:, int(yResolution / 2), int(tResolution / 2)]), color='b', lw=6, label='x')
        ax.plot(np.abs(Aspec[int(xResolution / 2), :, int(tResolution / 2)]), color='lime', lw=2.5, label='y')
        ax.plot(np.abs(Aspec[int(xResolution / 2), int(yResolution / 2), :]), color='r', lw=4, label='t')
        ax.legend(shadow=True, fontsize=legendFontSize, facecolor='white', edgecolor='black', loc='upper right')
        plt.show()
    for k in range(loopOuterKmax):
        if module_checking_spectrum:
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
        if module_checking_spectrum:
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
        isomin=30,
        isomax=30,
        surface_count=1,  # number of isosurfaces, 2 by default: only min and max
        caps=dict(x_show=False, y_show=False)
    ))
    fig.show()
if __name__ == '__main__':
    # real parameters propagation:
    if 1:
        fig, ax = plt.subplots(figsize=(8, 7))
        image = plt.imshow(np.abs(Field_STOV_simple(xytMesh[0], xytMesh[1], xytMesh[2])[:, int(yResolution / 2), :]) ** 2,
                           interpolation='bilinear', cmap='magma',
                           origin='lower', aspect='auto',  # aspect ration of the axes
                           extent=[yArray[0], yArray[-1], xArray[0], xArray[-1]])
        plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
        plt.show()

        if module_Paraxial:
            fieldOLD = split_step_old_time(Field_STOV_simple, loopInnerResolution, loopOuterResolution)
        if module_NonRapaxial:
            fieldUPPE = UPPE_time(Field_STOV_simple, loopInnerResolution, loopOuterResolution)

        if module_Paraxial:
            fig, ax = plt.subplots(figsize=(8, 7))
            image = plt.imshow(np.abs(fieldOLD[:, int(yResolution / 2), :]) ** 2,
                               interpolation='bilinear', cmap='magma',
                               origin='lower', aspect='auto',  # aspect ration of the axes
                               extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
            plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
            plt.show()
        if module_NonRapaxial:
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

    # STOV propogation
    if 0:
        E = Field_1(xytMesh[0], xytMesh[1], xytMesh[2])[:, :, int(tResolution / 2)]
        if module_initial:
            fig, ax = plt.subplots(figsize=(8, 7))
            plt.plot(tArray, np.abs(Field_1(xytMesh[0], xytMesh[1], xytMesh[2])
                                    [int(xResolution / 2), int(yResolution / 2), :]) ** Int)
            # plt.xlim(t0-tFWHM/2,t0+tFWHM/2)
            plt.show()
            fig, ax = plt.subplots(figsize=(8, 7))
            plt.plot(xArray, np.abs(Field_1(xytMesh[0], xytMesh[1], xytMesh[2])
                                    [:, int(yResolution / 2), int(tResolution / 2)]) ** Int)
            # plt.xlim(x0 - rFWHM / 2, x0 + rFWHM/2)
            plt.show()
        Sq = np.sum(E ** 2) * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
        # P_cr = P_critical_initialization(wavelength)  # дж / с
        Imax = np.sqrt(Pmax / Sq)
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
                image = plt.imshow(np.abs(fieldOLD[:, :, int(tResolution / 2)])  ** Int,
                                   interpolation='bilinear', cmap='magma',
                                   origin='lower', aspect='auto',  # aspect ration of the axes
                                   extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
                plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
                ax.set_xlabel('y', fontsize=xyLabelFontSize)
                ax.set_ylabel('x', fontsize=xyLabelFontSize)
                plt.show()
            if module_NonRapaxial:
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.abs(fieldUPPE[:, :, int(tResolution / 2)])  ** Int,
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
                plot_1D(tArray, np.abs(fieldOLD[int(xResolution / 2), int(yResolution / 2), :]  ** Int)
                        , label='Paraxial', xname=r'$t$', yname='', color='lime', ls='-', lw=6)
            #fieldUPPE = UPPE_time(Field_1, loopInnerResolution, loopOuterResolution)
            if module_NonRapaxial:
                plot_1D(tArray, np.abs(fieldUPPE[int(xResolution / 2), int(yResolution / 2), :]  ** Int)
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
                    plt.plot(xArray, np.abs(fieldOLD[:, int(yResolution / 2), int(tResolution / 2)]) ** Int,
                             label='Paraxial',color='lime', ls='--', lw=6)
                    plt.text(0, 0, f'z={zFinish * 1e3}mm', color='black', fontsize=26)
                if module_NonRapaxial:
                    plot_1D(xArray, np.abs(fieldUPPE[:, int(yResolution / 2), int(tResolution / 2)]) ** Int,
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
        # gauss propogation
    if 0:
        E = Field_1(xytMesh[0], xytMesh[1], xytMesh[2])[:, :, int(tResolution / 2)]
        if module_initial:
            fig, ax = plt.subplots(figsize=(8, 7))
            plt.plot(tArray, np.abs(Field_STOV_simple(xytMesh[0], xytMesh[1], xytMesh[2])
                                    [int(xResolution / 2), int(yResolution / 2), :]) ** Int)
            # plt.xlim(t0-tFWHM/2,t0+tFWHM/2)
            plt.show()
            fig, ax = plt.subplots(figsize=(8, 7))
            plt.plot(xArray, np.abs(Field_STOV_simple(xytMesh[0], xytMesh[1], xytMesh[2])
                                    [:, int(yResolution / 2), int(tResolution / 2)]) ** Int)
            # plt.xlim(x0 - rFWHM / 2, x0 + rFWHM/2)
            plt.show()
        Sq = np.sum(E ** 2) * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
        # P_cr = P_critical_initialization(wavelength)  # дж / с
        Imax = np.sqrt(Pmax / Sq)
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
            fieldOLD = split_step_old_time(Field_STOV_simple, loopInnerResolution, loopOuterResolution)
        if module_NonRapaxial:
            fieldUPPE = UPPE_time(Field_STOV_simple, loopInnerResolution, loopOuterResolution)
        # 2D
        if module_2D:
            if module_Paraxial:
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.abs(fieldOLD[:, :, int(tResolution / 2)]) ** Int,
                                   interpolation='bilinear', cmap='magma',
                                   origin='lower', aspect='auto',  # aspect ration of the axes
                                   extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
                plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
                ax.set_xlabel('y', fontsize=xyLabelFontSize)
                ax.set_ylabel('x', fontsize=xyLabelFontSize)
                plt.show()
            if module_NonRapaxial:
                fig, ax = plt.subplots(figsize=(8, 7))
                image = plt.imshow(np.abs(fieldUPPE[:, :, int(tResolution / 2)]) ** Int,
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
            # fieldOLD = split_step_old_time(Field_1, loopInnerResolution, loopOuterResolution)
            if module_Paraxial:
                plot_1D(tArray, np.abs(fieldOLD[int(xResolution / 2), int(yResolution / 2), :] ** Int)
                        , label='Paraxial', xname=r'$t$', yname='', color='lime', ls='-', lw=6)
            # fieldUPPE = UPPE_time(Field_1, loopInnerResolution, loopOuterResolution)
            if module_NonRapaxial:
                plot_1D(tArray, np.abs(fieldUPPE[int(xResolution / 2), int(yResolution / 2), :] ** Int)
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
                    plt.plot(xArray, np.abs(fieldOLD[:, int(yResolution / 2), int(tResolution / 2)]) ** Int,
                             label='Paraxial', color='lime', ls='--', lw=6)
                    plt.text(0, 0, f'z={zFinish * 1e3}mm', color='black', fontsize=26)
                if module_NonRapaxial:
                    plot_1D(xArray, np.abs(fieldUPPE[:, int(yResolution / 2), int(tResolution / 2)]) ** Int,
                            label='UPPE (non-paraxial)', xname=r'$x$', yname='', color='blue', ls='--', lw=2)
                # plt.xlim(xStart, xFinish)
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