"""
Figure 8 in paper Balmer decrement inferred attenuation
for different attenuation curves and FLARES, bagpipes 
toy galaxies
"""

import numpy as np
import pandas as pd
from functools import partial
import h5py, schwimmbad, matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cmasher as cmr
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.stats import spearmanr, pearsonr

from interrogator.sed import dust_curves

import sys
sys.path.append('../src')

from helpers import get_slen30

def get_data(ii, tag, limit=500):

    num = str(ii)

    if len(num) == 1:
        num =  '0'+num

    sim = rF"../../flares_pipeline/data/flares.hdf5"
    num = num+'/'

    with h5py.File(sim, 'r') as hf:
        S_len       = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)

        SFR10   = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('30/10Myr'), dtype = np.float32)
        SFR100  = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('30/100Myr'), dtype = np.float32)
        Mstar   = np.array(hf[num+tag+'/Galaxy/Mstar_aperture'].get('30'), dtype = np.float32)*1e10


        LFUVint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float64)
        LFUVatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float64)


        LVint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('V'), dtype = np.float64)
        LVatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('V'), dtype = np.float64)

        LBint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('B'), dtype = np.float64)
        LBatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('B'), dtype = np.float64)


        Halphaint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/Intrinsic/HI6563'].get('Luminosity'), dtype = np.float64)
        Halpha = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI/HI6563'].get('Luminosity'), dtype = np.float64)


        Hbetaint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/Intrinsic/HI4861'].get('Luminosity'), dtype = np.float64)
        Hbeta = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI/HI4861'].get('Luminosity'), dtype = np.float64)

        Hgammaint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/Intrinsic/HI4340'].get('Luminosity'), dtype = np.float64)
        Hgamma = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI/HI4340'].get('Luminosity'), dtype = np.float64)


        S_ap    = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype = bool)

        beta = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Indices/beta'].get('DustModelI'), dtype = np.float32)

        ####Adding beta and beta0 to see their effect on these curve shape

    z = float(tag[5:].replace('p','.'))

    S_len, begin, end, S_ap = get_slen30(S_len, S_ap, limit)
    ok = np.where(S_len>limit)[0]

    Mstar, SFR10, SFR100, LFUVint, LFUVatt, LVint, LVatt, LBint, LBatt, Halpha, Hbeta, Halphaint, Hbetaint, Hgammaint, Hgamma, beta = Mstar[ok], SFR10[ok], SFR100[ok], LFUVint[ok], LFUVatt[ok], LVint[ok], LVatt[ok], LBint[ok], LBatt[ok], Halpha[ok], Hbeta[ok], Halphaint[ok], Hbetaint[ok], Hgammaint[ok], Hgamma[ok], beta[ok]

    Av          = -2.5 * np.log10(LVatt/LVint)
    Ab          = -2.5 * np.log10(LBatt/LBint)
    Auv         = -2.5 * np.log10(LFUVatt/LFUVint)
    balmer1     = Halpha/Hbeta
    balmer1int  = Halphaint/Hbetaint
    balmer2     = Hbeta/Hgamma
    balmer2int  = Hbetaint/Hgammaint
    AHalpha     = -2.5 * np.log10(Halpha/Halphaint)
    AHbeta      = -2.5 * np.log10(Hbeta/Hbetaint)

    return Mstar, SFR10, SFR100, LFUVatt, Auv, Av, Ab, balmer1, balmer1int, balmer2, balmer2int, beta, AHalpha, AHbeta


def smc_gordon(wavs):
    """ Calculate the ratio A(lambda)/A(V) for the Gordon et al.
    (2003) Small Magellanic Cloud extinction curve. Warning: this
    currently diverges at small wavelengths, probably some sort of
    power law interpolation at the blue end should be added. """

    A_lambda = np.zeros_like(wavs)

    inv_mic = 1./(wavs*10.**-4.)

    c1 = -4.959
    c2 = 2.264
    c3 = 0.389
    c4 = 0.461
    x0 = 4.6
    gamma = 1.0
    Rv = 2.74

    D = inv_mic**2/((inv_mic**2 - x0**2)**2 + inv_mic**2*gamma**2)
    F = 0.5392*(inv_mic - 5.9)**2 + 0.05644*(inv_mic - 5.9)**3
    F[inv_mic < 5.9] = 0.
    # values at 2.198 and 1.25 changed to provide smooth interpolation
    # as noted in Gordon et al. (2016, ApJ, 826, 104)

    A_lambda = (c1 + c2*inv_mic + c3*D + c4*F)/Rv + 1.

    # Generate region redder than 2760AA by interpolation
    ref_wavs = np.array([0.276, 0.296, 0.37, 0.44, 0.55,
                         0.65, 0.81, 1.25, 1.65, 2.198, 3.1])*10**4

    ref_ext = np.array([2.220, 2.000, 1.672, 1.374, 1.00,
                        0.801, 0.567, 0.25, 0.169, 0.11, 0.])

    if np.max(wavs) > 2760.:
        A_lambda[wavs > 2760.] = np.interp(wavs[wavs > 2760.],
                                           ref_wavs, ref_ext, right=0.)

    return A_lambda


def calc_AHalpha(x, kHalpha, kHbeta, delta=0.):

    """
    Get the attenuation from the Balmer decrement

    Args:
        x = log10(Halpha/Hbeta)
        kHalpha = Value of the extinction curve at Halpha wavelength
        kHbeta = Value of the extinction curve at Hbeta wavelength

    Returns:
        y = Returns the Halpha attenuation

    """

    Halpha_lam  = 6562.81
    Hbeta_lam   = 4861.33
    V_lam       = 5500.
    out         = x#(x - np.log10(2.86))
    fac         = (Hbeta_lam/Halpha_lam)**(delta)

    y = (1./(fac * (kHbeta/kHalpha) - 1.)) * out 

    return y

def calc_AHbeta(x, kHbeta, kHgamma, delta=0.):

    """
    Get the attenuation from the Balmer decrement

    Args:
        x = log10(Hbeta/Hgamma)
        kHbeta = Value of the extinction curve at Hbeta wavelength
        kHgamma = Value of the extinction curve at Hgamma wavelength

    Returns:
        y = Returns the Hbeta attenuation

    """

    Hbeta_lam   = 4861.33
    Hgamma_lam  = 4340.46

    V_lam       = 5500.
    out         = x#(x + np.log10(0.47))
    fac         = (Hgamma_lam/Hbeta_lam)**(delta)

    y = (1./(fac * (kHgamma/kHbeta) - 1.)) * out 

    return y

def plot_median(x, xx, yy, ax, label='_nolegend_'):

    bincen = (x[1:]+x[:-1])/2.
    x_med, y_med, y_up, y_lo = np.array([]), np.array([]), np.array([]), np.array([])
    for jj in range(len(bincen)):
        ok = np.logical_and(xx>=x[jj], xx<x[jj+1])
        if np.sum(ok)>=5:
            x_med = np.append(x_med, bincen[jj])
            y_med = np.append(y_med, np.median(yy[ok]))
            y_up  = np.append(y_up, np.percentile(yy[ok], 84))
            y_lo  = np.append(y_lo, np.percentile(yy[ok], 16))



    ax.plot(x_med, y_med, color='black', label=label, lw=3)
    ax.fill_between(x_med, y_up, y_lo, color='black', alpha=0.2)
    # axins.plot(x_med, y_med, color='black', lw=3)
    # axins.fill_between(x_med, y_up, y_lo, color='black', alpha=0.1)
    return ax

def get_balmer_ratio(x, ratio=2.79):

    return ratio * 10**(x/2.5)

def get_balmer_decrement(x, ratio=2.79):
    
    return 2.5 * np.log10(x/ratio) 

if __name__ == "__main__":

    quantiles = [0.84,0.50,0.16]
    limit=500
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    Halpha_lam  = 6562.81
    Hbeta_lam   = 4861.33
    Hgamma_lam  = 4340.46

    intr_ratio1 = 2.79
    intr_ratio2 = 0.47

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 6), sharex=False, sharey=False, facecolor='w', edgecolor='k')
    axs = axs.ravel()

    # choose a colormap
    cz_m = plt.get_cmap('cmr.bubblegum_r')
    norm = matplotlib.colors.BoundaryNorm(np.arange(4.5,11,1), cz_m.N)
    # create a ScalarMappable and initialize a data structure
    z_m = matplotlib.cm.ScalarMappable(cmap=cz_m, norm=norm)
    z_m.set_array([])


    for ii, tag in enumerate(tags):

        z = float(tag[5:].replace('p','.'))

        func    = partial(get_data, tag=tag, limit=limit)
        pool    = schwimmbad.MultiPool(processes=8)
        dat     = np.array(list(pool.map(func, np.arange(0,40))), dtype='object')
        pool.close()

        Mstar       = np.concatenate(dat[:,0])
        SFR10       = np.concatenate(dat[:,1])
        SFR100      = np.concatenate(dat[:,2])
        LFUVatt     = np.concatenate(dat[:,3])
        Auv         = np.concatenate(dat[:,4])
        Av          = np.concatenate(dat[:,5])
        Ab          = np.concatenate(dat[:,6])
        balmer1     = np.concatenate(dat[:,7])
        balmer1int  = np.concatenate(dat[:,8])
        balmer2     = np.concatenate(dat[:,9])
        balmer2int  = np.concatenate(dat[:,10])
        beta        = np.concatenate(dat[:,11])
        AHalpha     = np.concatenate(dat[:,12])
        AHbeta      = np.concatenate(dat[:,13])

        ws = np.zeros(len(Mstar))
        n = 0
        for jj in range(40):
            if jj==0:
                ws[0:len(dat[jj][0])] = weights[jj]
            else:
                ws[n:n+len(dat[jj][0])] = weights[jj]

            n+=len(dat[jj][0])


        y, x = AHalpha, 2.5*np.log10(balmer1/intr_ratio1)#, np.log10(SFR100)
        ok = (np.log10(SFR100)>=-2.)
        if ii == 0:
            xx1, yy1, zz1 = x, y, z*np.ones(len(x))
        else:
            xx1 = np.append(xx1, x)
            yy1 = np.append(yy1, y)
            zz1 = np.append(zz1, z*np.ones(len(x)))

        y, x = AHbeta, 2.5*np.log10(balmer2/(1/intr_ratio2))#, np.log10(SFR100)
        ok = (np.log10(SFR100)>=-2.)
        if ii == 0:
            xx2, yy2, zz2 = x, y, z*np.ones(len(x))
        else:
            xx2 = np.append(xx2, x)
            yy2 = np.append(yy2, y)
            zz2 = np.append(zz2, z*np.ones(len(x)))

    print ("-----------------alpha-beta----------------------\n")
    ok = np.isfinite(xx1)*np.isfinite(yy1)
    print ("pearsonr: ", pearsonr(xx1[ok], yy1[ok]), "\n")
    print ("spearmanr: ", spearmanr(xx1[ok], yy1[ok]))
    print ("---------------------------------------")

    print ("------------------beta-gamma---------------------\n")
    ok = np.isfinite(xx1)*np.isfinite(yy1)
    print ("pearsonr: ", pearsonr(xx2[ok], yy2[ok]), "\n")
    print ("spearmanr: ", spearmanr(xx2[ok], yy2[ok]))
    print ("---------------------------------------")

    vmin, vmax = 4.5, 10.5
    extent1 = [*[np.min(xx1[np.isfinite(xx1)])-0.1,np.max(xx1[np.isfinite(xx1)])+0.1], *[np.min(yy1[np.isfinite(yy1)])-0.1,np.max(yy1[np.isfinite(yy1)])+0.1]]
    extent1 = [*[np.min(xx2[np.isfinite(xx2)])-0.1,np.max(xx2[np.isfinite(xx2)])+0.1], *[np.min(yy2[np.isfinite(yy2)])-0.1,np.max(yy2[np.isfinite(yy2)])+0.1]]
    gridsize = (90,50)

    # hb = axs.hexbin(xx, yy, C=zz, reduce_C_function=np.median, cmap=cmap, vmin=vmin, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=0)
    
    hb = axs[0].scatter(xx1, yy1, c=zz1, cmap=cz_m, s=2, vmin=vmin, vmax=vmax, alpha=0.4)
    axs[1].scatter(xx2, yy2, c=zz2, cmap=cz_m, s=2, vmin=vmin, vmax=vmax, alpha=0.4)

    # axins = zoomed_inset_axes(axs, zoom=3.5, loc=4)
    # axins.scatter(xx, yy, c=zz, cmap=cmap, s=2, vmin=vmin, vmax=vmax, alpha=0.4)

    # axs.set_title(r'$z \in [5,10]$')
    x = np.arange(0.0, 1.7, 0.05)
    # bincen = (x[1:]+x[:-1])/2.
    # x_med, y_med, y_up, y_lo = np.array([]), np.array([]), np.array([]), np.array([])
    # for jj in range(len(bincen)):
    #     ok = np.logical_and(xx>=x[jj], xx<x[jj+1])
    #     if np.sum(ok)>=5:
    #         x_med = np.append(x_med, bincen[jj])
    #         y_med = np.append(y_med, np.median(yy[ok]))
    #         y_up  = np.append(y_up, np.percentile(yy[ok], 84))
    #         y_lo  = np.append(y_lo, np.percentile(yy[ok], 16))
    #
    #
    #
    # axs.plot(x_med, y_med, color='black', label='Median', lw=3)
    # axs.fill_between(x_med, y_up, y_lo, color='black', alpha=0.2)
    # axins.plot(x_med, y_med, color='black', lw=3)
    # axins.fill_between(x_med, y_up, y_lo, color='black', alpha=0.1)

    plot_median(x, xx1, yy1, axs[0], label='Median')
    plot_median(x, xx2, yy2, axs[1])

    x = np.arange(0.0, 2, 0.1)



    #SMC
    axs[0].plot(x, calc_AHalpha(x, dust_curves.SMC_Pei92().tau(Halpha_lam), dust_curves.SMC_Pei92().tau(Hbeta_lam)), label = 'SMC', color = 'green', ls = 'dashed')
    # axs[1].plot(x, calc_AHbeta(x, dust_curves.SMC_Pei92().tau(Hbeta_lam), dust_curves.SMC_Pei92().tau(Hgamma_lam)), label = 'SMC', color = 'green', ls = 'dashed')
    axs[1].plot(x, calc_AHbeta(x, smc_gordon(np.array([Hbeta_lam])), smc_gordon(np.array([Hgamma_lam]))), color = 'green', ls = 'dashed')
    # axins.plot(x, calc_AHalpha(x, dust_curves.SMC_Pei92().tau(Halpha_lam), dust_curves.SMC_Pei92().tau(Hbeta_lam)), color = 'green', ls = 'dashed')

    #Calzetti
    # axs.plot(x, calc_AHalpha(x, dust_curves.Starburst_Calzetti2000().tau(np.array([Halpha_lam])), dust_curves.Starburst_Calzetti2000().tau(np.array([Hbeta_lam]))), label = 'Calzetti', color = 'yellow', ls = 'dashed')
    # axins.plot(x, calc_AHalpha(x, dust_curves.Starburst_Calzetti2000().tau(np.array([Halpha_lam])), dust_curves.Starburst_Calzetti2000().tau(np.array([Hbeta_lam]))), color = 'yellow', ls = 'dashed')

    #Input
    axs[0].plot(x, calc_AHalpha(x, dust_curves.simple().tau(Halpha_lam), dust_curves.simple().tau(Hbeta_lam)), label = 'Input', color = 'brown', ls = 'dashed')
    axs[1].plot(x, calc_AHbeta(x, dust_curves.simple().tau(Hbeta_lam), dust_curves.simple().tau(Hgamma_lam)), color = 'brown', ls = 'dashed')
    # axins.plot(x, calc_AHalpha(x, dust_curves.simple().tau(Halpha_lam), dust_curves.simple().tau(Hbeta_lam)), color = 'grey', ls = 'dashed')

    #Calzetti-delta
    axs[0].plot(x, calc_AHalpha(x, dust_curves.Starburst_Calzetti2000().tau(np.array([Halpha_lam])), dust_curves.Starburst_Calzetti2000().tau(np.array([Hbeta_lam])), delta=-1), label = r'Calzetti: $\delta=(-1, 0, 0.5)$', color = 'magenta', ls = 'dashed')
    axs[1].plot(x, calc_AHbeta(x, dust_curves.Starburst_Calzetti2000().tau(np.array([Hbeta_lam])), dust_curves.Starburst_Calzetti2000().tau(np.array([Hgamma_lam])), delta=-1), color = 'magenta', ls = 'dashed')
    # axins.plot(x, calc_AHalpha(x, dust_curves.Starburst_Calzetti2000().tau(np.array([Halpha_lam])), dust_curves.Starburst_Calzetti2000().tau(np.array([Hbeta_lam])), delta=-1), color = 'magenta', ls = 'dashed')

    axs[0].plot(x, calc_AHalpha(x, dust_curves.Starburst_Calzetti2000().tau(np.array([Halpha_lam])), dust_curves.Starburst_Calzetti2000().tau(np.array([Hbeta_lam])), delta=0), color = 'magenta', ls = 'dashed')
    axs[1].plot(x, calc_AHbeta(x, dust_curves.Starburst_Calzetti2000().tau(np.array([Hbeta_lam])), dust_curves.Starburst_Calzetti2000().tau(np.array([Hgamma_lam])), delta=0), color = 'magenta', ls = 'dashed')
    # axins.plot(x, calc_AHalpha(x, dust_curves.Starburst_Calzetti2000().tau(np.array([Halpha_lam])), dust_curves.Starburst_Calzetti2000().tau(np.array([Hbeta_lam])), delta=0), color = 'magenta', ls = 'dashed')


    axs[0].plot(x, calc_AHalpha(x, dust_curves.Starburst_Calzetti2000().tau(np.array([Halpha_lam])), dust_curves.Starburst_Calzetti2000().tau(np.array([Hbeta_lam])), delta=0.5), color = 'magenta', ls = 'dashed')
    axs[1].plot(x, calc_AHbeta(x, dust_curves.Starburst_Calzetti2000().tau(np.array([Hbeta_lam])), dust_curves.Starburst_Calzetti2000().tau(np.array([Hgamma_lam])), delta=0.5), color = 'magenta', ls = 'dashed')
    # axins.plot(x, calc_AHalpha(x, dust_curves.Starburst_Calzetti2000().tau(np.array([Halpha_lam])), dust_curves.Starburst_Calzetti2000().tau(np.array([Hbeta_lam])), delta=1), color = 'magenta', ls = 'dashed')


    # choose a colormap
    cmap = cmr.pepper_r                   # CMasher
    norm = matplotlib.colors.Normalize(1,15)
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_m.set_array([])
    nsigma = np.arange(2,15,2)

    #Values from bagpipes toy galaxy with same metallicity
    Halpha_int      = 3.814790954309774e-17
    Hbeta_int       = 1.3727318926805297e-17
    Hgamma_int      = 6.5689082680727794e-18
    Halpha_dusts    = np.array([8.55297193e-18, 1.32262399e-17, 1.47995803e-17, 1.65490399e-17,
       1.70950515e-17, 1.78348431e-17, 1.77434853e-17])
    Hbeta_dusts     = np.array([1.47228589e-18, 3.26737312e-18, 4.28066054e-18, 5.01546820e-18,
       5.35011298e-18, 5.92935855e-18, 5.81508480e-18])
    Hgamma_dusts    = np.array([5.20698213e-19, 1.36993033e-18, 1.91943525e-18, 2.28466001e-18,
       2.46536278e-18, 2.78221828e-18, 2.71322783e-18])

    markersize=60

    for ii, ns in enumerate(nsigma):
        axs[0].scatter(2.5*np.log10((Halpha_dusts[ii]/Hbeta_dusts[ii])/intr_ratio1), -2.5*np.log10(Halpha_dusts[ii]/Halpha_int), marker='*', color=s_m.to_rgba(ns), zorder=15, s=markersize)
        # axins.scatter(2.5*np.log10((Halpha_dusts[ii]/Hbeta_dusts[ii])/2.86), -2.5*np.log10(Halpha_dusts[ii]/Halpha_int), marker='*', color=s_m.to_rgba(ns))

        axs[1].scatter(2.5*np.log10((Hbeta_dusts[ii]/Hgamma_dusts[ii])/(1/intr_ratio2)), -2.5*np.log10(Hbeta_dusts[ii]/Hbeta_int), marker='*', color=s_m.to_rgba(ns), zorder=15, s=markersize)


    # H  1  6562.81A = 1.8520472010484847e-17
    # H  1  4861.33A  = 6.654083690680629e-18
    # H  1  4340.46A = 3.184825603994815e-18
    # O  3  5006.84A = 2.113145946041032e-17
    # N  2  6583.45A = 5.256158853629885e-19

    # #Values from bagpipes toy galaxy with varying metallicity
    Halpha_int      = 1.8520472010484847e-17
    Hbeta_int       = 6.654083690680629e-18
    Hgamma_int      = 3.184825603994815e-18
    Halpha_dusts    = np.array([4.55926007e-18, 6.38962480e-18, 5.99209331e-18, 6.52013576e-18,
       1.09164371e-17, 7.02551803e-18, 7.14772401e-18])
    Hbeta_dusts     = np.array([8.27392621e-19, 1.63126855e-18, 1.70140726e-18, 1.85413342e-18,
       3.56640032e-18, 2.16086749e-18, 2.18829608e-18])
    Hgamma_dusts    = np.array([2.99849920e-19, 6.96334904e-19, 7.61699283e-19, 8.28499922e-19,
       1.65924209e-18, 9.93849993e-19, 9.99389336e-19])


    for ii, ns in enumerate(nsigma):
        axs[0].scatter(2.5*np.log10((Halpha_dusts[ii]/Hbeta_dusts[ii])/intr_ratio1), -2.5*np.log10(Halpha_dusts[ii]/Halpha_int), marker='h', color=s_m.to_rgba(ns), zorder=10, s=markersize)
        # axins.scatter(2.5*np.log10((Halpha_dusts[ii]/Hbeta_dusts[ii])/2.86), -2.5*np.log10(Halpha_dusts[ii]/Halpha_int), marker='*', color=s_m.to_rgba(ns))

        axs[1].scatter(2.5*np.log10((Hbeta_dusts[ii]/Hgamma_dusts[ii])/(1/intr_ratio2)), -2.5*np.log10(Hbeta_dusts[ii]/Hbeta_int), marker='h', color=s_m.to_rgba(ns), zorder=10, s=markersize)

    #Values from bagpipes toy galaxy with varying ages
    Halpha_int      = 4.761775776718768e-17
    Hbeta_int       = 1.7021359308151464e-17
    Hgamma_int      = 8.119082391384192e-18
    Halpha_dusts    = np.array([1.24391177e-17, 1.59170501e-17, 2.03469979e-17, 1.84036827e-17,
       1.55487035e-17, 2.22280838e-17, 2.19749692e-17])
    Hbeta_dusts     = np.array([2.38911613e-18, 4.38285003e-18, 6.04943822e-18, 5.73974606e-18,
       4.46791623e-18, 6.93861507e-18, 6.99724843e-18])
    Hgamma_dusts    = np.array([8.90132006e-19, 1.95126013e-18, 2.72380809e-18, 2.64457305e-18,
       1.99739058e-18, 3.17982731e-18, 3.24075415e-18])


    for ii, ns in enumerate(nsigma):
        axs[0].scatter(2.5*np.log10((Halpha_dusts[ii]/Hbeta_dusts[ii])/intr_ratio1), -2.5*np.log10(Halpha_dusts[ii]/Halpha_int), marker='s', color=s_m.to_rgba(ns), zorder=10, s=markersize)
        # axins.scatter(2.5*np.log10((Halpha_dusts[ii]/Hbeta_dusts[ii])/2.86), -2.5*np.log10(Halpha_dusts[ii]/Halpha_int), marker='*', color=s_m.to_rgba(ns))

        axs[1].scatter(2.5*np.log10((Hbeta_dusts[ii]/Hgamma_dusts[ii])/(1/intr_ratio2)), -2.5*np.log10(Hbeta_dusts[ii]/Hbeta_int), marker='s', color=s_m.to_rgba(ns), zorder=10, s=markersize)


    for ax in axs:
        # ax.legend(frameon=False, fontsize=10, loc=4)
        ax.grid(True, alpha = 0.4, ls='dashed')
        ax.minorticks_on()
        # axs.xaxis.set_tick_params(labeltop=True, labelbottom=False)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.set_ylim(0,5)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)

        

    custom = [Line2D([], [], marker='*', markersize=8, color='black', linestyle='None'),
              Line2D([], [], marker='h', markersize=8, color='black', linestyle='None'),
              Line2D([], [], marker='s', markersize=8, color='black', linestyle='None')]

    axs[0].set_xlim(0,0.81)
    axs[1].set_xlim(0,0.35)

    axs[0].legend(frameon=False, fontsize=10, loc="upper left")
    axs[1].legend(custom, [r'Varying only A$_{\mathrm{V}}$', 'Varying A$_{\mathrm{V}}$ and metallicity', 'Varying A$_{\mathrm{V}}$ and ages'], frameon=False, fontsize=10, loc="upper left", title=r'$\mathbf{Toy\, Galaxies}$')

    # x1, x2, y1, y2 = 0., 0.25, 0.05, 1.2
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.xaxis.set_tick_params(labeltop=True, labelbottom=False)
    # axins.tick_params(top=True)
    # # axins.set_xticklabels('')
    # # axins.set_yticklabels('')
    # axins.grid(True)
    # mark_inset(axs, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    fig.subplots_adjust(wspace=0.15, hspace=0, bottom=0.15, right=0.9)
    axs[0].set_ylabel(r'$A_{H\alpha}$', fontsize=14)
    axs[0].set_xlabel(r'2.5$\times$log$_{10}(\frac{\mathrm{H}_{\alpha}/\mathrm{H}_{\beta}}{2.79})$', fontsize=14)

    axs[1].set_ylabel(r'$A_{H\beta}$', fontsize=14)
    axs[1].set_xlabel(r'2.5$\times$log$_{10}(\frac{\mathrm{H}_{\beta}/\mathrm{H}_{\gamma}}{1/0.47})$', fontsize=14)

    secax1 = axs[0].secondary_xaxis('top', functions=(partial(get_balmer_ratio, ratio=intr_ratio1), partial(get_balmer_decrement, ratio=intr_ratio1)))
    secax1.set_xlabel(r'$\mathrm{H}_{\alpha}/\mathrm{H}_{\beta}$', fontsize=15)
    secax1.set_xlim(get_balmer_ratio(0), get_balmer_ratio(0.81))

    secax2 = axs[1].secondary_xaxis('top', functions=(partial(get_balmer_ratio, ratio=(1/intr_ratio2)), partial(get_balmer_decrement, ratio=(1/intr_ratio2))))
    secax2.set_xlabel(r'$\mathrm{H}_{\beta}/\mathrm{H}_{\gamma}$', fontsize=15)
    secax2.set_xlim(get_balmer_ratio(0, ratio=intr_ratio2), get_balmer_ratio(0.35, ratio=intr_ratio2))

    cbaxes = fig.add_axes([0.93, 0.25, 0.01, 0.5])
    fig.colorbar(z_m, cax=cbaxes)

    cbaxes.set_ylabel(r'$z$', fontsize = 18)
    cbaxes.set_yticks(np.arange(5,11,1))
    for label in cbaxes.get_yticklabels():
        label.set_fontsize(13)

    plt.savefig(F"../results/balmer_Halpha_Hbeta_z_zin5_10.pdf", bbox_inches='tight', dpi=300)

    plt.show()
