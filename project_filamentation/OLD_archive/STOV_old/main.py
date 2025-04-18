import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# %% temporal values
k_z0 = 1
k_normal0 = 1
w_0 = 1

# %% resolution
xResolution = 128
xStart, xFinish = 0, 10
yResolution = 64
yStart, yFinish = 0, 9
lOAM = 2


# %% functions
def wave_function_kr_less_1(x, y, z, t, l_oam):
    return ((x + 1j * np.sign(l_oam) * y) ** (np.abs(l_oam))
            * np.exp(1j * (k_z0 * z - w_0 * t)))


def wave_function(x, y, z, t, l_oam):
    return (jv(np.abs(lOAM), k_normal0 * np.sqrt(x ** 2 + y ** 2))
            * np.exp(1j * (l_oam * np.angle(x + 1j * y) + k_z0 * z - w_0 * t)))


# %% arrays creation
xArray = np.linspace(xStart, xFinish, xResolution)
yArray = np.linspace(yStart, yFinish, yResolution)
xMesh, yMesh = np.array(np.meshgrid(xArray, yArray))
if __name__ == '__main__':
    # print(np.abs(wave_function(xMesh, yMesh, 0, 0, lOAM)))

    image = plt.imshow(np.abs(wave_function(xMesh, yMesh, 0, 0, lOAM)) ** 2, interpolation='bilinear', cmap='jet',
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[xArray[0], xArray[-1], yArray[0], yArray[-1]])
    plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    plt.show()
    image = plt.imshow(np.angle(wave_function(xMesh, yMesh, 0, 0, lOAM)), interpolation='bilinear', cmap='jet',
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[xArray[0], xArray[-1], yArray[0], yArray[-1]])
    plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    plt.show()
    print('Thank You! :)')
