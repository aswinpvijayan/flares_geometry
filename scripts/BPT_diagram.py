"""
Figure 10 in paper: BPT diagram of FLARES galaxies and the 
bagpipes toy galaxies
"""

import numpy as np
from functools import partial
import h5py, schwimmbad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D      
import cmasher as cmr

import sys
sys.path.append('../src')

#[Oiii] 5007/Hbeta vs NII 6583/Halpha

def get_data(ii, tag, limit=1000, bins = np.arange(0,5.25,0.25)):

    num = str(ii)

    if len(num) == 1:
        num =  '0'+num

    sim = rF"../../flares_pipeline/data/flares.hdf5"
    num = num+'/'

    with h5py.File(sim, 'r') as hf:
        S_len       = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)

        mstar   = np.array(hf[num+tag+'/Galaxy/Mstar_aperture/'].get('30'), dtype = np.float32)*1e10
        SFR10   = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('30/10Myr'), dtype = np.float32)
        SFR100  = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('30/100Myr'), dtype = np.float32)

        S_ap    = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype = bool)

        Halpha = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI/HI6563'].get('Luminosity'), dtype = np.float64)
        Hbeta = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI/HI4861'].get('Luminosity'), dtype = np.float64)

        HalphaBC = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/No_ISM/HI6563'].get('Luminosity'), dtype = np.float64)
        HbetaBC = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/No_ISM/HI4861'].get('Luminosity'), dtype = np.float64)

        Halphaint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/Intrinsic/HI6563'].get('Luminosity'), dtype = np.float64)
        Hbetaint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/Intrinsic/HI4861'].get('Luminosity'), dtype = np.float64)

        OIII5007 = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI/OIII5007'].get('Luminosity'), dtype = np.float64)
        NII6583 = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI/NII6583'].get('Luminosity'), dtype = np.float64)

        OIII5007BC = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/No_ISM/OIII5007'].get('Luminosity'), dtype = np.float64)
        NII6583BC = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/No_ISM/NII6583'].get('Luminosity'), dtype = np.float64)

        OIII5007int = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/Intrinsic/OIII5007'].get('Luminosity'), dtype = np.float64)
        NII6583int = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/Intrinsic/NII6583'].get('Luminosity'), dtype = np.float64)



    n = len(S_len)
    begin = np.zeros(n, dtype=np.int32)
    end = np.zeros(n, dtype=np.int32)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)
    begin = begin.astype(np.int32)
    end = end.astype(np.int32)

    S_len30 = np.zeros(n, dtype=np.int32)
    for kk in range(n):
        S_len30[kk] = np.sum(S_ap[begin[kk]:end[kk]])

    ok = np.where(S_len30>limit)
    mstar, SFR10, SFR100, Halpha, HalphaBC, Halphaint, Hbeta, HbetaBC, Hbetaint, OIII5007, OIII5007BC, OIII5007int, NII6583, NII6583BC, NII6583int = mstar[ok], SFR10[ok], SFR100[ok], Halpha[ok], HalphaBC[ok], Halphaint[ok], Hbeta[ok], HbetaBC[ok], Hbetaint[ok], OIII5007[ok], OIII5007BC[ok], OIII5007int[ok], NII6583[ok], NII6583BC[ok], NII6583int[ok]

    BPT_x = np.log10(NII6583/Halpha)
    BPT_y = np.log10(OIII5007/Hbeta)

    BPTBC_x = np.log10(NII6583BC/HalphaBC)
    BPTBC_y = np.log10(OIII5007BC/HbetaBC)

    BPTint_x = np.log10(NII6583int/Halphaint)
    BPTint_y = np.log10(OIII5007int/Hbetaint)

    return mstar, SFR10, SFR100, BPT_x, BPT_y, BPTBC_x, BPTBC_y, BPTint_x, BPTint_y

def BPT_K03(x):
    #BPT from Kauffman+2003 https://arxiv.org/abs/astro-ph/0304239

    return 0.61/(x-0.05) + 1.3

def BPT_Strom_2017(x):

    return 0.61/(x-0.22) + 1.12

def get_contours(x, y, axs, colour):

    n = len(x)
    x_edges = np.arange(min(x)-0.05, max(x)+0.05, 0.03)
    y_edges = np.arange(min(y)-0.05, max(y)+0.05, 0.03)
    hist, xedges, yedges = np.histogram2d(x, y, bins=(x_edges, y_edges))

    xidx = np.digitize(x[np.isfinite(x)], x_edges)-1
    yidx = np.digitize(y[np.isfinite(y)], y_edges)-1
    num_pts = (hist[xidx, yidx])
    srt = np.sort(num_pts)
    perc = np.array([0.05, 0.32, 0.5])*n
    perc = perc.astype(int)
    levels = srt[perc]

    axs.contour(hist.T, extent=[xedges.min(),xedges.max(),yedges.min(),y_edges.max()], levels = levels, colors = colour, linestyles = ['dotted', 'dashed', 'solid'])

    return axs

if __name__ == "__main__":

   quantiles = [0.84,0.50,0.16]
   limit=500

   tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

   fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5), sharex=True, sharey=True, facecolor='w', edgecolor='k')
   axs = axs.ravel()

   num = np.array([6692, 2913, 1264, 540, 236, 99], dtype=np.int32)
   BPT_xs, BPT_ys, BPTBC_xs, BPTBC_ys, BPTint_xs, BPTint_ys = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

   # choose a colormap
   cmap = cmap = cmr.pepper_r                     # CMasher
   norm = matplotlib.colors.Normalize(1,15)
   # create a ScalarMappable and initialize a data structure
   s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
   s_m.set_array([])
   nsigma = np.arange(2,15,2)
   markersize=60

   ##Bagpipes values for toy galaxy with same metallicity
   Halpha_int      = 3.814790954309774e-17
   Hbeta_int       = 1.3727318926805297e-17
   OIII_int        = 2.6165967202406073e-17
   NII_int         = 2.329891830159413e-19
   Halpha_dusts    = np.array([9.55936879e-18, 1.19942593e-17, 1.59819450e-17, 1.61244860e-17,
      1.61797160e-17, 1.62160042e-17, 1.28686864e-17])
   Hbeta_dusts     = np.array([1.74736949e-18, 2.91804097e-18, 4.77206051e-18, 5.00629968e-18,
      5.00485836e-18, 5.23443666e-18, 3.86012921e-18])
   OIII_dusts      = np.array([3.59516659e-18, 5.78066219e-18, 9.25411524e-18, 9.66405185e-18,
      9.66725274e-18, 1.00745428e-17, 7.47485193e-18])
   NII_dusts       = np.array([5.86751052e-20, 7.34971007e-20, 9.77778733e-20, 9.86239543e-20,
      9.89620607e-20, 9.91444236e-20, 7.87359798e-20])

   axs[0].scatter(np.log10(NII_int/Halpha_int), np.log10(OIII_int/Hbeta_int), color='red', marker='*', zorder=10, s=markersize)
   for ii, ns in enumerate(nsigma):
      axs[1].scatter(np.log10(NII_dusts[ii]/Halpha_dusts[ii]), np.log10(OIII_dusts[ii]/Hbeta_dusts[ii]), marker='*', color=s_m.to_rgba(ns), zorder=10, s=markersize)

   ##Bagpipes values for toy galaxy with varying metallicity
   Halpha_int      = 1.8520472010484847e-17
   Hbeta_int       = 6.654083690680629e-18
   OIII_int        = 2.113145946041032e-17
   NII_int         = 5.256158853629885e-19
   Halpha_dusts    = np.array([4.55926007e-18, 6.38962480e-18, 5.99209331e-18, 6.52013576e-18,
      1.09164371e-17, 7.02551803e-18, 7.14772401e-18])
   Hbeta_dusts     = np.array([8.27392621e-19, 1.63126855e-18, 1.70140726e-18, 1.85413342e-18,
      3.56640032e-18, 2.16086749e-18, 2.18829608e-18])
   OIII_dusts      = np.array([2.80218924e-18, 4.99099859e-18, 6.16141060e-18, 5.99808004e-18,
      1.01687029e-17, 7.55367480e-18, 7.28680442e-18])
   NII_dusts       = np.array([1.31995604e-19, 1.66287397e-19, 2.06095304e-19, 1.73510369e-19,
      2.38123425e-19, 2.26167372e-19, 2.41468765e-19])

   axs[0].scatter(np.log10(NII_int/Halpha_int), np.log10(OIII_int/Hbeta_int), color='red', marker='h', zorder=10, s=markersize)
   for ii, ns in enumerate(nsigma):
      axs[1].scatter(np.log10(NII_dusts[ii]/Halpha_dusts[ii]), np.log10(OIII_dusts[ii]/Hbeta_dusts[ii]), marker='h', color=s_m.to_rgba(ns), zorder=10, s=markersize)

   # H  1  6562.81A = 4.761775776718768e-17
   # H  1  4861.33A = 1.7021359308151464e-17
   # H  1  4340.46A = 8.119082391384192e-18
   # O  3  5006.84A = 4.052214103124736e-17
   # N  2  6583.45A = 3.1447631607462463e-19
   ##Bagpipes values for toy galaxy with varying ages
   Halpha_int      = 4.761775776718768e-17
   Hbeta_int       = 1.7021359308151464e-17
   OIII_int        = 4.052214103124736e-17
   NII_int         = 3.1447631607462463e-19
   Halpha_dusts    = np.array([1.24391177e-17, 1.59170501e-17, 2.03469979e-17, 1.84036827e-17,
      1.55487035e-17, 2.22280838e-17, 2.19749692e-17])
   Hbeta_dusts     = np.array([2.38911613e-18, 4.38285003e-18, 6.04943822e-18, 5.73974606e-18,
      4.46791623e-18, 6.93861507e-18, 6.99724843e-18])
   OIII_dusts      = np.array([6.13608189e-18, 1.06130282e-17, 1.51995119e-17, 1.37587515e-17,
      1.03921337e-17, 1.79331638e-17, 1.83562855e-17])
   NII_dusts       = np.array([8.20238573e-20, 1.05370173e-19, 1.30734201e-19, 1.21532324e-19,
      1.05929327e-19, 1.42819711e-19, 1.40777126e-19])

   axs[0].scatter(np.log10(NII_int/Halpha_int), np.log10(OIII_int/Hbeta_int), color='red', marker='s', zorder=10, s=markersize)
   for ii, ns in enumerate(nsigma):
      axs[1].scatter(np.log10(NII_dusts[ii]/Halpha_dusts[ii]), np.log10(OIII_dusts[ii]/Hbeta_dusts[ii]), marker='s', color=s_m.to_rgba(ns), zorder=10, s=markersize)


   # choose a colormap
   c_m =plt.get_cmap('cmr.bubblegum_r')
   norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5,6,1), c_m.N)
   # create a ScalarMappable and initialize a data structure
   s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
   s_m.set_array([])

   for ii, tag in enumerate(tags):

      z = float(tag[5:].replace('p','.'))

      func    = partial(get_data, tag=tag, limit=limit)
      pool    = schwimmbad.MultiPool(processes=8)
      dat     = np.array(list(pool.map(func, np.arange(0,40))))
      pool.close()

      mstar   = np.concatenate(dat[:,0])
      SFR10   = np.concatenate(dat[:,1])
      SFR100  = np.concatenate(dat[:,2])

      BPT_x       = np.concatenate(dat[:,3])
      BPT_y       = np.concatenate(dat[:,4])
      BPTBC_x     = np.concatenate(dat[:,5])
      BPTBC_y     = np.concatenate(dat[:,6])
      BPTint_x    = np.concatenate(dat[:,7])
      BPTint_y    = np.concatenate(dat[:,8])


      axs[0].scatter(BPTint_x, BPTint_y, s=4, alpha=0.5, color=s_m.to_rgba(ii), label = rF'z={int(z)}')
      axs[1].scatter(BPT_x, BPT_y, s=4, alpha=0.5, color=s_m.to_rgba(ii))
      # axs[1].scatter(BPTBC_x, BPTBC_y, s=2, alpha=0.6, color=s_m.to_rgba(ii))

      BPT_xs      = np.append(BPT_xs, BPT_x)
      BPT_ys      = np.append(BPT_ys, BPT_y)
      BPTBC_xs    = np.append(BPTBC_xs, BPTBC_x)
      BPTBC_ys    = np.append(BPTBC_ys, BPTBC_y)
      BPTint_xs   = np.append(BPTint_xs, BPTint_x)
      BPTint_ys   = np.append(BPTint_ys, BPTint_y)

   print (np.nanstd(np.abs(BPT_xs-BPTint_xs)), np.nanmedian(np.abs(BPT_xs-BPTint_xs)))
   print (np.nanstd(np.abs(BPT_ys-BPTint_ys)), np.nanmedian(np.abs(BPT_ys-BPTint_ys)))

   x = np.arange(-3., -0.25, 0.2)
   axs[0].plot(x, BPT_Strom_2017(x), color='orange', label='Strom+2017 z=2')
   axs[1].plot(x, BPT_Strom_2017(x), color='orange')
   get_contours(BPTint_xs, BPTint_ys, axs[0], 'black')
   get_contours(BPT_xs, BPT_ys, axs[1], 'black')
   # axs[2].plot(x, BPT_Strom_2017(x), color='orange')

   for ax in axs:
      ax.grid()
      ax.set_xlim(-2.5, -0.5)
      ax.set_ylim(-1, 1.)

   custom = [Line2D([], [], marker='*', markersize=8, color='black', linestyle='None'),
            Line2D([], [], marker='h', markersize=8, color='black', linestyle='None'),
            Line2D([], [], marker='s', markersize=8, color='black', linestyle='None')]
   
   axs[0].legend(frameon=False, scatterpoints=1, markerscale=4)
   axs[1].legend(custom, [r'Varying only A$_{\mathrm{V}}$', 'Varying A$_{\mathrm{V}}$ and metallicity', 'Varying A$_{\mathrm{V}}$ and ages'], frameon=False, fontsize=10, loc=4, title=r'$\mathbf{Toy\, Galaxies}$')

   axs[0].set_xlabel(r"log$_{10}$([NII]/H$_{\alpha})_{int}$", fontsize=12)
   axs[0].set_ylabel(r"log$_{10}$([OIII]/H$_{\beta})_{int}$", fontsize=12)

   axs[1].set_xlabel(r"log$_{10}$([NII]/H$_{\alpha})_{att}$", fontsize=12)
   axs[1].set_ylabel(r"log$_{10}$([OIII]/H$_{\beta})_{att}$", fontsize=12)


   plt.savefig('../results/BPT_z5_10_Zvary_Agevary.png', bbox_inches='tight', dpi=400)
   plt.show()
