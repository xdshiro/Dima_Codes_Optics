from UPPE_and_Fourier_Nonlinear_real import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"
rcParams['mathtext.fontset'] = 'stix'







def plot_2D(E, x, y, xname='', yname='', map='jet', vmin=0.13, vmax=1.14):
    if vmin == 0.13 and vmax == 1.14:
        vmin = E.min()
        vmax = E.max()

    fig, ax = plt.subplots(figsize=(8, 7))
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
# xy phase
if 0:
    xArray = np.array([0, 4500 * 1e-6])
    yArray = np.linspace(0, 4500 * 1e-6, 141)
    tArray = np.linspace(0, 400 * 1e-12, 141)
    TIME = 73
    data_new = np.load('10.npy')
    plot_2D(np.angle(data_new[:, :, TIME]), xArray * 1e3, yArray * 1e3, 'x (mm)', 'y (mm)',
            map='viridis', vmin=-np.pi, vmax=np.pi)
    plt.xlim(2.25 - 1, 2.25 + 1)
    plt.ylim(2.25 - 1, 2.25 + 1)
    plt.title(f'z=1m, t={round((tArray[TIME]  + (tArray[0] - tArray[-1])/2)* 1e12,1)}ps', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xy A
if 0:
    xArray = np.array([0, 4500 * 1e-6])
    yArray = np.linspace(0, 4500 * 1e-6, 141)
    tArray = np.linspace(0, 400 * 1e-12, 141)
    TIME = 63
    data_new = np.load('10.npy')
    plot_2D(np.abs(data_new[:, :, TIME]), xArray * 1e3, yArray * 1e3, 'x (mm)', 'y (mm)',
            map='viridis')
    plt.xlim(2.25 - 1, 2.25 + 1)
    plt.ylim(2.25 - 1, 2.25 + 1)
    plt.title(f'z=1m, t={round((tArray[TIME]  + (tArray[0] - tArray[-1])/2)* 1e12,1)}ps', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()

    exit()
# xt phase
if 0:
    xArray = np.array([0, 4500 * 1e-6])
    yArray = np.linspace(0, 4500 * 1e-6, 141)
    tArray = np.linspace(0, 400 * 1e-12, 141)
    TIME = 70
    data_new = np.load('10.npy')
    plot_2D(np.angle(data_new[:, 70, :]), tArray * 1e12, xArray * 1e3, 'x (mm)', 't (ps)',
            vmin=-np.pi, vmax=np.pi, map='hsv')
    plt.xlim(2.25 - 1, 2.25 + 1)
    plt.ylim(100, 300)
    #plt.title(f'z=1m, t={round((tArray[TIME]  + (tArray[0] - tArray[-1])/2)* 1e12,1)}ps', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()

    exit()
# xt A
if 0:
    xArray = np.array([0, 4500 * 1e-6])
    yArray = np.linspace(0, 4500 * 1e-6, 141)
    tArray = np.linspace(0, 400 * 1e-12, 141)
    TIME = 70
    data_new = np.load('10.npy')
    plot_2D(np.abs(data_new[:, 70, :]), tArray * 1e12, xArray * 1e3, 'x (mm)', 't (ps)',
            map='magma')
    plt.xlim(2.25 - 1, 2.25 + 1)
    plt.ylim(100, 300)
    #plt.title(f'z=1m, t={round((tArray[TIME]  + (tArray[0] - tArray[-1])/2)* 1e12,1)}ps', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()

    exit()

# xy A
if 0:
    xArray = np.linspace(0, 4000 * 1e-6, 151)
    yArray = np.linspace(0, 4000 * 1e-6, 151)
    tArray = np.linspace(0, 300 * 1e-12, 71)
    zArray = np.linspace(0, 1.9, 701)
    z = 540
    data_new = np.load('filam2.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, :, z]), xArray * 1e3, yArray * 1e3, 'x (mm)', 'y (mm)',
            map='magma')
    plt.xlim(2 - 0.3, 2 + 0.3)
    plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'z={round(zArray[z],3)}m', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xy Phase
if 0:
    xArray = np.linspace(0, 4000 * 1e-6, 151)
    yArray = np.linspace(0, 4000 * 1e-6, 151)
    tArray = np.linspace(0, 300 * 1e-12, 71)
    zArray = np.linspace(0, 1.9, 701)
    z = 545
    data_new = np.load('filam2.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, :, z]), xArray * 1e3, yArray * 1e3, 'x (mm)', 'y (mm)',
            map='hsv')
    plt.xlim(2 - 0.3, 2 + 0.3)
    plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'z={round(zArray[z],3)}m, Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xz A
if 0:
    xArray = np.linspace(0, 4000 * 1e-6, 151)
    yArray = np.linspace(0, 4000 * 1e-6, 151)
    tArray = np.linspace(0, 300 * 1e-12, 71)
    zArray = np.linspace(0, 1.9, 701)
    z = 0
    data_new = np.load('filam2.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(151/2), :]), xArray * 1e3, zArray, 'z (m)', 'x (mm)',
            map='magma')
    plt.xlim(1.4, 1.6)
    plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xz Phase
if 0:
    xArray = np.linspace(0, 4000 * 1e-6, 151)
    yArray = np.linspace(0, 4000 * 1e-6, 151)
    tArray = np.linspace(0, 300 * 1e-12, 71)
    zArray = np.linspace(0, 1.9, 701)
    z = 0
    data_new = np.load('filam2.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(151/2), :]), xArray * 1e3, zArray, 'z (m)', 'x (mm)',
            map='hsv')
    plt.xlim(1.45, 1.5)
    plt.ylim(2 - 0.1, 2 + 0.1)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()

# 3
# xy A
if 0:
    xArray = np.linspace(0, 4000 * 1e-6, 151)
    yArray = np.linspace(0, 4000 * 1e-6, 151)
    tArray = np.linspace(0, 300 * 1e-12, 71)
    zArray = np.linspace(0, 1.9, 701)
    z = 540
    data_new = np.load('filam2.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, :, z]), xArray * 1e3, yArray * 1e3, 'x (mm)', 'y (mm)',
            map='magma')
    plt.xlim(2 - 0.3, 2 + 0.3)
    plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'z={round(zArray[z],3)}m', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xy Phase
if 0:
    xArray = np.linspace(0, 4000 * 1e-6, 151)
    yArray = np.linspace(0, 4000 * 1e-6, 151)
    tArray = np.linspace(0, 300 * 1e-12, 71)
    zArray = np.linspace(0, 1.9, 701)
    z = 545
    data_new = np.load('filam2.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, :, z]), xArray * 1e3, yArray * 1e3, 'x (mm)', 'y (mm)',
            map='hsv')
    plt.xlim(2 - 0.3, 2 + 0.3)
    plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'z={round(zArray[z],3)}m, Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xz A
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 171)
    yArray = np.linspace(0, 400 * 1e-6, 171)
    tArray = np.linspace(0, 300 * 1e-12, 121)
    zArray = np.linspace(0, 0.02, 191)
    z = 0
    data_new = np.load('filam3.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(171/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(1.4, 1.6)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xz Phase
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 171)
    yArray = np.linspace(0, 400 * 1e-6, 171)
    tArray = np.linspace(0, 300 * 1e-12, 121)
    zArray = np.linspace(0, 0.02, 191)
    z = 0
    data_new = np.load('filam3.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(151/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='hsv')
    #plt.xlim(1.45, 1.5)
    #plt.ylim(2 - 0.1, 2 + 0.1)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()

# 4
if 0:
    xArray = np.linspace(0, 600 * 1e-6, 181)
    yArray = np.linspace(0, 600 * 1e-6, 181)
    tArray = np.linspace(0, 300 * 1e-12, 91)
    zArray = np.linspace(0, 0.12, 451)
    z = 0
    data_new = np.load('filam4.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(181/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(1.4, 1.6)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xz Phase
if 0:
    xArray = np.linspace(0, 600 * 1e-6, 181)
    yArray = np.linspace(0, 600 * 1e-6, 181)
    tArray = np.linspace(0, 300 * 1e-12, 91)
    zArray = np.linspace(0, 0.12, 451)
    z = 0
    data_new = np.load('filam4.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(181/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='hsv')
    #plt.xlim(1.45, 1.5)
    #plt.ylim(2 - 0.1, 2 + 0.1)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xy A
if 0:
    xArray = np.linspace(0, 600 * 1e-6, 181)
    yArray = np.linspace(0, 600 * 1e-6, 181)
    tArray = np.linspace(0, 300 * 1e-12, 91)
    zArray = np.linspace(0, 1.2, 451)
    z = 241
    data_new = np.load('filam4.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, :, z]) ** 2, xArray * 1e3, yArray * 1e3, 'x (mm)', 'y (mm)',
            map='magma')
    # plt.xlim(2 - 0.3, 2 + 0.3)
    # plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'z={round(zArray[z],3)}m', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
# xy Phase
if 0:
    xArray = np.linspace(0, 600 * 1e-6, 181)
    yArray = np.linspace(0, 600 * 1e-6, 181)
    tArray = np.linspace(0, 300 * 1e-12, 91)
    zArray = np.linspace(0, 0.12, 451)
    #z = 230
    data_new = np.load('filam4.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, :, z]), xArray * 1e3, yArray * 1e3, 'x (mm)', 'y (mm)',
            map='hsv', vmin=-np.pi, vmax=np.pi)
    #plt.xlim(2 - 0.3, 2 + 0.3)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    #plt.title(f'z={round(zArray[z],3)}m, Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xt phase
if 0:
    xArray = np.linspace(0, 600 * 1e-6, 181)
    yArray = np.linspace(0, 600 * 1e-6, 181)
    tArray = np.linspace(0, 300 * 1e-12, 91)
    zArray = np.linspace(0, 0.12, 451)
    TIME = 70
    data_new = np.load('filam4_t.npy')
    print(np.shape(data_new))
    plot_2D(np.angle(data_new[:, int(181/2), :]), xArray * 1e3, tArray * 1e12, 'x (mm)', 't (ps)',
             map='hsv') #vmin=-np.pi, vmax=np.pi,
    #plt.xlim(2.25 - 1, 2.25 + 1)
    #plt.ylim(100, 300)
    #plt.title(f'z=1m, t={round((tArray[TIME]  + (tArray[0] - tArray[-1])/2)* 1e12,1)}ps', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()

    exit()

# fs
# xz A
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 131)
    yArray = np.linspace(0, 400 * 1e-6, 131)
    tArray = np.linspace(0, 300 * 1e-12, 191)
    zArray = np.linspace(0, 0.05, 601)
    z = 0
    data_new = np.load('filam_fs.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(131/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    plt.xlim(32.5, 33.5)
    plt.ylim(0.15, 0.25)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xz Phase
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 131)
    yArray = np.linspace(0, 400 * 1e-6, 131)
    tArray = np.linspace(0, 300 * 1e-12, 191)
    zArray = np.linspace(0, 0.05, 601)
    z = 0
    data_new = np.load('filam_fs.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(131/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='hsv')
    plt.xlim(32.5, 33.5)
    plt.ylim(0.15, 0.25)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xz Phase_viridis
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 131)
    yArray = np.linspace(0, 400 * 1e-6, 131)
    tArray = np.linspace(0, 300 * 1e-12, 191)
    zArray = np.linspace(0, 0.05, 601)
    z = 0
    data_new = np.load('filam_fs.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(131/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='viridis')
    #plt.xlim(32.5, 33.5)
    #plt.ylim(0.15, 0.25)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xy A
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 131)
    yArray = np.linspace(0, 400 * 1e-6, 131)
    tArray = np.linspace(0, 300 * 1e-12, 191)
    zArray = np.linspace(0, 0.05, 601)
    z = 396
    data_new = np.load('filam_fs.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, :, z]), xArray * 1e3, yArray * 1e3, 'x (mm)', 'y (mm)',
            map='magma')
    plt.xlim(0.15, 0.25)
    plt.ylim(0.15, 0.25)
    plt.title(f'z={round(zArray[z],5)}m, |E|', fontweight="bold", fontsize=26) #$|E|^2$
    plt.show()
    plt.close()
# xy Phase
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 131)
    yArray = np.linspace(0, 400 * 1e-6, 131)
    tArray = np.linspace(0, 300 * 1e-12, 191)
    zArray = np.linspace(0, 0.05, 601)
    z = 396
    data_new = np.load('filam_fs.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, :, z]), xArray * 1e3, yArray * 1e3, 'x (mm)', 'y (mm)',
            map='hsv')
    plt.xlim(0.15, 0.25)
    plt.ylim(0.15, 0.25)
    plt.title(f'z={round(zArray[z],5)}m, Phase', fontweight="bold", fontsize=26) #$|E|^2$
    plt.show()
    plt.close()


#xz_real 33
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.032946769357046254, 0.03305379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_xz.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(13, 13.4)
    plt.ylim(0.15, 0.25)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
#xz_real Phase
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.032946769357046254, 0.03305379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_xz.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='hsv')
    # plt.xlim(13, 13.4)
    plt.ylim(0.15, 0.25)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
#xz_real Phase_viridis
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.032946769357046254, 0.03305379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_xz.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='viridis')
    plt.xlim(32.97, 33.03)
    plt.ylim(0.15, 0.25)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()

#xz_real 35
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.034946769357046254, 0.03505379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_xz_35.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(13, 13.4)
    plt.ylim(0.15, 0.25)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
#xz_real Phase 35
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.032946769357046254, 0.03305379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_xz_35.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='hsv')
    # plt.xlim(13, 13.4)
    plt.ylim(0.15, 0.25)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
#xz_real Phase_viridis 35
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.034946769357046254, 0.03505379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_xz_35.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='viridis')
    plt.xlim(34.97, 35.03)
    plt.ylim(0.15, 0.25)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()

#xz_real 34
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.033946769357046254, 0.03405379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_xz_34.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(13, 13.4)
    plt.ylim(0.15, 0.25)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
#xz_real Phase_viridis 34
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.033946769357046254, 0.03405379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_xz_34.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='viridis')
    #plt.xlim(33.97, 34.03)
    #plt.ylim(0.15, 0.25)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()

#xz_real 335
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.033946769357046254 - 0.0005, 0.03405379692638943 - 0.0005, 191)
    z = 0
    data_new = np.load('filam_fs_xz_335.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(13, 13.4)
    plt.ylim(0.15, 0.25)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
#xz_real Phase_viridis 335
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.033946769357046254 - 0.0005, 0.03405379692638943 - 0.0005, 191)
    z = 0
    data_new = np.load('filam_fs_xz_335.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='viridis')
    plt.xlim(33.97-0.5, 34.03-0.5)
    plt.ylim(0.15, 0.25)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()

#xz_real 345
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.033946769357046254 + 0.0005, 0.03405379692638943 + 0.0005, 191)
    z = 0
    data_new = np.load('filam_fs_xz_345.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(13, 13.4)
    plt.ylim(0.15, 0.25)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
#xz_real Phase_viridis 335
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.033946769357046254 + 0.0005, 0.03405379692638943 + 0.0005, 191)
    z = 0
    data_new = np.load('filam_fs_xz_345.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='viridis')
    #plt.xlim(33.97+0.5, 34.03+0.5)
    #plt.ylim(0.15, 0.25)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()

#xt_real
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.032946769357046254, 0.03305379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_time.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(291/2), :]), xArray * 1e3, tArray * 1e15, 't (fs)', 'x (mm)',
            map='magma')
    #plt.xlim(13, 13.4)
    plt.ylim(0.15, 0.25)
    plt.title(f'z=33mm, |E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
#xt_real_Phase
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.032946769357046254, 0.03305379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_time.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(291/2), :]), xArray * 1e3, tArray * 1e15, 't (fs)', 'x (mm)',
            map='hsv')
    #plt.xlim(13, 13.4)
    plt.ylim(0.15, 0.25)
    plt.title(f'z=33mm, Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
#xt_real_Phase_viridis
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.032946769357046254, 0.03305379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_time.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(291/2), :]), xArray * 1e3, tArray * 1e15, 't (fs)', 'x (mm)',
            map='viridis')
    #plt.xlim(13, 13.4)
    plt.ylim(0.15, 0.25)
    plt.title(f'z=33mm, Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()

# OAM=1
# xz A
if 0:
    xArray = np.linspace(0, 600 * 1e-6, 291)
    yArray = np.linspace(0, 600 * 1e-6, 291)
    tArray = np.linspace(0, 400 * 1e-12, 171)
    zArray = np.linspace(0, 0.05, 401)
    z = 000
    data_new = np.load('test_simpOAM.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(35, 36)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xy A
if 0:
    xArray = np.linspace(0, 600 * 1e-6, 291)
    yArray = np.linspace(0, 600 * 1e-6, 291)
    tArray = np.linspace(0, 400 * 1e-12, 171)
    zArray = np.linspace(0, 0.1, 401)
    z = int(0.75 * 400)
    data_new = np.load('test_simpOAM.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, :, z]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(35, 36)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xz Phase
if 0:
    xArray = np.linspace(0, 600 * 1e-6, 291)
    yArray = np.linspace(0, 600 * 1e-6, 291)
    tArray = np.linspace(0, 400 * 1e-12, 171)
    zArray = np.linspace(0, 0.05, 401)
    z = 000
    data_new = np.load('test_simpOAM.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(291/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='viridis')
    plt.xlim(35, 37)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'|E|', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()

# xz_real OAM 1
if 0:
    xArray = np.linspace(0, 900 * 1e-6, 331)
    yArray = np.linspace(0, 900 * 1e-6, 331)
    tArray = np.linspace(0, 400 * 1e-12, 171)
    zArray = np.linspace(0.07492552114413198 , 0.07507531569694519, 171)
    z = 000
    data_new = np.load('filam_simpleOAM_xz_75_25.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(331/2), :]) ** 2 , xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(35, 37)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'|E|$^2$', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xz_real OAM 1 PHase
if 0:
    xArray = np.linspace(0, 900 * 1e-6, 331)
    yArray = np.linspace(0, 900 * 1e-6, 331)
    tArray = np.linspace(0, 400 * 1e-12, 171)
    zArray = np.linspace(0.07492552114413198 , 0.07507531569694519, 171)
    z = 000
    data_new = np.load('filam_simpleOAM_xz_75_25.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(331/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='viridis')
    #plt.xlim(35, 37)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xy_real OAM 1
z = 110
if 1:
    xArray = np.linspace(0, 900 * 1e-6, 331)
    yArray = np.linspace(0, 900 * 1e-6, 331)
    tArray = np.linspace(0, 400 * 1e-12, 171)
    zArray = np.linspace(0.07492552114413198 , 0.07507531569694519, 171)

    data_new = np.load('filam_simpleOAM_xz_75_25.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, :, z]) ** 2, xArray * 1e3, yArray * 1e3, 'y (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(35, 37)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'|E|$^2$, z={round(zArray[z], 6)}mm', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()

# xy_real OAM 1 PHase
if 1:
    xArray = np.linspace(0, 900 * 1e-6, 331)
    yArray = np.linspace(0, 900 * 1e-6, 331)
    tArray = np.linspace(0, 400 * 1e-12, 171)
    zArray = np.linspace(0.07492552114413198 , 0.07507531569694519, 171)

    data_new = np.load('filam_simpleOAM_xz_75_25.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, :, z]), xArray * 1e3, yArray * 1e3, 'y (mm)', 'x (mm)',
            map='viridis')
    #plt.xlim(35, 37)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'Phase, z={round(zArray[z], 6)}mm', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()

#xt_real OAM
if 0:
    xArray = np.linspace(0, 900 * 1e-6, 331)
    yArray = np.linspace(0, 900 * 1e-6, 331)
    tArray = np.linspace(0, 400 * 1e-12, 171)

    z = 0
    data_new = np.load('OAM_at_75_vs_time.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(331/2), :]) ** 2, xArray * 1e3, tArray * 1e15, 't (fs)', 'x (mm)',
            map='magma')
    #plt.xlim(13, 13.4)
    #plt.ylim(0.15, 0.25)
    plt.title(f'z=75mm, |E|$^2$', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()

#xt_real_Phase_viridis OAM
if 0:
    xArray = np.linspace(0, 400 * 1e-6, 291)
    yArray = np.linspace(0, 400 * 1e-6, 291)
    tArray = np.linspace(0, 500 * 1e-15, 191)
    zArray = np.linspace(0.032946769357046254, 0.03305379692638943, 191)
    z = 0
    data_new = np.load('filam_fs_time.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(291/2), :]), xArray * 1e3, tArray * 1e15, 't (fs)', 'x (mm)',
            map='viridis')
    #plt.xlim(13, 13.4)
    plt.ylim(0.15, 0.25)
    plt.title(f'z=33mm, Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()


# xz_filament OAM 1
if 0:
    xArray = np.linspace(0, 900 * 1e-6, 331)
    yArray = np.linspace(0, 900 * 1e-6, 331)
    tArray = np.linspace(0, 700 * 1e-12, 171)
    zArray = np.linspace(0, 0.1, 401)
    z = 000
    data_new = np.load('test_OAM_abs.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, int(331/2), :]) ** 2, xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(35, 37)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'|E|$^2$', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xy_filament OAM 1
if 0:
    xArray = np.linspace(0, 900 * 1e-6, 331)
    yArray = np.linspace(0, 900 * 1e-6, 331)
    tArray = np.linspace(0, 700 * 1e-12, 171)
    zArray = np.linspace(0, 0.1, 401)
    z = 350
    data_new = np.load('test_OAM_abs.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.abs(data_new[:, :, z]) ** 2, xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='magma')
    #plt.xlim(35, 37)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'|E|$^2$' + f' z={round(zArray[z] * 1e3,1)}mm', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()
# xz_filament OAM 1 Phase
if 0:
    xArray = np.linspace(0, 900 * 1e-6, 331)
    yArray = np.linspace(0, 900 * 1e-6, 331)
    tArray = np.linspace(0, 700 * 1e-12, 171)
    zArray = np.linspace(0, 0.1, 401)
    z = 000
    data_new = np.load('test_OAM_abs.npy')  #, vmin=-np.pi, vmax=np.pi
    plot_2D(np.angle(data_new[:, int(331/2), :]), xArray * 1e3, zArray * 1e3, 'z (mm)', 'x (mm)',
            map='viridis')
    #plt.xlim(72, 78)
    #plt.ylim(2 - 0.3, 2 + 0.3)
    plt.title(f'Phase', fontweight="bold", fontsize=26)
    plt.show()
    plt.close()
    exit()