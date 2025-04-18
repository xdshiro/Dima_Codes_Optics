from main import *


def radius(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def phi(x, t):
    return np.angle(x + 1j * t)


def theta(x, y):
    return np.angle(x + 1j * y)


def plot_2D(E, x, y, xname='', yname='', map='jet', vmin=0.13, vmax=1.14, title=''):
    fig, ax = plt.subplots(figsize=(8, 7))
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
    plt.title(title, fontweight="bold", fontsize=26)
    plt.show()
    plt.close()


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


def rayleigh_range(lambda0, rho0):
    return np.pi * rho0 ** 2 / lambda0


def Lcollapse():
    temp1 = 0.367 * rayleigh_range(lambda0, rho0)

    if n2 == 0:
        return 0
    else:
        temp2 = (np.sqrt(Pmax / Pcrit) - 0.852) ** 2 - 0.0219
        return temp1 / np.sqrt(temp2)


def asymmetric_LG_z0(x, y, t):
    l = lOAM
    p = 0
    width = rho0
    x = x - x0
    y = y - y0
    zR = rayleigh_range(lambda0, width)
    z = 1e-20

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

    E = ((np.sqrt(2) / width) ** (np.abs(l))
         * (x + 1j * np.sign(l) * y) ** (np.abs(l))
         * laguerre_polynomial(2 * rho(x, y) ** 2 / width ** 2, np.abs(l), p)
         * np.exp(-rho(x, y) ** 2 / width ** 2))
    return np.abs(E)