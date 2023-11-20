"""
Figure 6 in paper: inp=0; Galaxy attenuation curve for the FLARES
galaxies in different SFR bins.
Figure 7 in paper: inp=1; Galaxy attenuation curve with similar 
selection as the Scoville2015 galaxies of I_{AB}<25
"""

import numpy as np
import pandas as pd
from functools import partial
import h5py, schwimmbad, matplotlib
import matplotlib.pyplot as plt
import cmasher as cmr
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from uncertainties import unumpy
from scipy.interpolate import splrep, BSpline

import sys
sys.path.append('../src')

from interrogator.sed import dust_curves
from helpers import get_slen30

def get_data(ii, tag, inp=0, limit=1000, dl=1e30):

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

        LFUVatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float32)
        LFUVint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float32)

        LVatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('V'), dtype = np.float32)
        LVint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('V'), dtype = np.float32)

        LIatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('I'), dtype = np.float32)

        S_ap    = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype = bool)

        SED_int = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/SED'].get('Intrinsic'), dtype = np.float32)
        SED_att = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/SED'].get('DustModelI'), dtype = np.float32)
        lam = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/SED'].get('Wavelength'), dtype = np.float32)*1e-4 #in microns

        BB = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Indices/BB_Wilkins'].get('DustModelI'), dtype = np.float32)


    S_len, begin, end, S_ap = get_slen30(S_len, S_ap, limit)
    ok = np.where(S_len>limit)[0]

    if inp==1:
        z = float(tag[5:].replace('p','.'))

        I_AB = -2.5 * np.log10(LIatt/(4*np.pi*(dl**2))) - 48.60

        ok = np.where((S_len>limit) & (I_AB<25.))[0]

    Mstar, SFR10, SFR100, LFUVint, LFUVatt, LVint, LVatt, SED_int, SED_att = Mstar[ok], SFR10[ok], SFR100[ok], LFUVint[ok], LFUVatt[ok], LVint[ok], LVatt[ok], SED_int[ok], SED_att[ok]

    ok = np.where((lam>=0.1) & (lam<=1.))[0]
    lam = lam[ok]
    SED_int = SED_int[:,ok]
    SED_att = SED_att[:,ok]

    Av = -2.5 * np.log10(LVatt/LVint)
    Alam = -2.5*np.log10(SED_att/SED_int)


    scoville_x = np.array([950., 1250., 1500., 1800., 2250., 2800., 3900., 6350., 1e4])/1e4
    A_lamsmooth = np.zeros((len(Mstar), len(scoville_x)))

    for ii, jj in enumerate(scoville_x):
        ok = np.where((lam>jj-0.02) & (lam<jj+0.02))[0]
        filt    = np.zeros(len(lam))
        filt[ok] = 1.
        A_lamsmooth[:,ii]  = -2.5 * np.log10((np.trapz(SED_att * filt, lam) / np.trapz(filt, lam)) / (np.trapz(SED_int * filt, lam) / np.trapz(filt, lam)))


    ok = np.where((lam>0.11) & (lam<0.15))[0]
    filt    = np.zeros(len(lam))
    filt[ok] = 1.
    A1300    = -2.5 * np.log10((np.trapz(SED_att * filt, lam) / np.trapz(filt, lam)) / (np.trapz(SED_int * filt, lam) / np.trapz(filt, lam)))

    ok = np.where((lam>0.28) & (lam<0.32))[0]
    filt    = np.zeros(len(lam))
    filt[ok] = 1.
    A3000    = -2.5 * np.log10((np.trapz(SED_att * filt, lam) / np.trapz(filt, lam)) / (np.trapz(SED_int * filt, lam) / np.trapz(filt, lam)))

    return Mstar, SFR10, SFR100 , LFUVint, LFUVatt, Av, lam, Alam, A_lamsmooth, A1300, A3000


def Calzetti2021(lam):

    # lam is the wavelength in units of micron

    k_lam, k_lamerr = np.zeros(len(lam)), np.zeros(len(lam))
    ok = np.where((lam>=0.63) & (lam<=2.2))
    x = unumpy.uarray([1.00], [0.08])*(-2.365 + 1.345/lam[ok]) + unumpy.uarray([1.97], [0.15])
    k_lam[ok], k_lamerr[ok] = unumpy.nominal_values(x), unumpy.std_devs(x)
    ok = np.where((lam>=0.12) & (lam<0.63))
    x = unumpy.uarray([1.00], [0.08])*(-2.450 + 1.809/lam[ok] - 0.293/(lam[ok]**2) + 0.0216/(lam[ok]**3)) + unumpy.uarray([1.97], [0.15])
    k_lam[ok], k_lamerr[ok] = unumpy.nominal_values(x), unumpy.std_devs(x)

    return unumpy.uarray(k_lam, k_lamerr)


def cubicspline(x, y, interp_x):

    cs = CubicSpline(x, y)
    interp_y = cs(interp_x)

    return interp_y

def plot_th_curves(axs, x, lam_norm, lw=3, alpha=1):

    # x in units of micron

    axs.plot(x, dust_curves.SMC_Pei92().tau(x*1e4)/(dust_curves.SMC_Pei92().tau(lam_norm)), label='SMC', color = 'green', ls='dashed', lw=lw, alpha=alpha)
    axs.plot(x, dust_curves.Starburst_Calzetti2000().tau(x*1e4)/dust_curves.Starburst_Calzetti2000().tau(np.array([lam_norm])), label='Calzetti-Starburst', color = 'magenta', ls='dashed', lw=lw, alpha=alpha)
    axs.plot(x, dust_curves.simple().tau(x*1e4)/dust_curves.simple().tau(lam_norm), label='Input', color = 'brown', ls='dashed', lw=lw, alpha=alpha)
    axs.plot(x, dust_curves.MW_N18().tau(x*1e4)/dust_curves.MW_N18().tau(lam_norm), label='MW N18', color = 'orange', ls='dashed', lw=lw, alpha=alpha)

    return axs

def plot_Scoville15(axs, color='blue', ls='dotted', lw=2):

    scoville_x = np.array([950., 1250., 1500., 1800., 2250., 2800., 3900., 6350., 1e4])/1e4
    scoville_y = np.array([1.431, 0.999, 1.003, 0.924, 0.865, 0.716, 0.509, 0.255, 0.127])
    scoville_yerr = np.array([0.026, 0.024, 0.025, 0.030, 0.043, 0.061, 0.073, 0.092, 0.096])

    axs.errorbar(scoville_x, scoville_y, yerr=scoville_yerr, label='Scoville+2015 ($4<z<6.5$)', color = color, ls=ls, lw=lw)

    return axs

def plot_Calzetti2021(axs, x, lam_norm):

    calzetti_y = Calzetti2021(x)/Calzetti2021(np.array([lam_norm/1e4]))
    axs.plot(x, unumpy.nominal_values(calzetti_y), label='Calzetti+2021 (NGC 3351)', color='brown', ls='dotted', lw=2)
    axs.fill_between(x, unumpy.nominal_values(calzetti_y)+unumpy.std_devs(calzetti_y), unumpy.nominal_values(calzetti_y)-unumpy.std_devs(calzetti_y), color='brown', alpha=0.2)

    return axs

if __name__ == "__main__":

    inp = int(sys.argv[1])

    quantiles = [0.84,0.50,0.16]
    limit = 500
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    smooth_x = np.array([950., 1250., 1500., 1800., 2250., 2800., 3900., 6350., 1e4])/1e4

    if inp==0:
        fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(14, 10), sharex=True, sharey=True, facecolor='w', edgecolor='k')
        axs=axs.ravel()
        cbins = np.arange(-1, 3.1, 0.5)

        # choose a colormap
        cmap = cmr.rainforest                   # CMasher
        c_m = plt.get_cmap('cmr.cosmic')   # MPL
        # c_m = matplotlib.cm.plasma
        norm = matplotlib.colors.BoundaryNorm(cbins, c_m.N)
        # create a ScalarMappable and initialize a data structure
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])

    else:
        fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(6, 6), sharex=True, sharey=True, facecolor='w', edgecolor='k')
        tags = ['011_z004p770', '010_z005p000', '009_z006p000']
        axs.set_title(r'$z \in [4.77,6]$', fontsize = 16)
        y = np.array([])
        lam_norm = 1300.



    for ii, tag in enumerate(tags[::-1]):

        z = float(tag[5:].replace('p','.'))

        dist = cosmo.luminosity_distance(z).to(u.cm).value

        func    = partial(get_data, tag=tag, inp=inp, limit=limit, dl=dist)
        pool    = schwimmbad.MultiPool(processes=8)
        dat     = np.array(list(pool.map(func, np.arange(0,40))), dtype=object)
        pool.close()

        Mstar       = np.concatenate(dat[:,0])
        SFR10       = np.concatenate(dat[:,1])
        SFR100      = np.concatenate(dat[:,2])
        LFUVint     = np.concatenate(dat[:,3])
        LFUVatt     = np.concatenate(dat[:,4])
        Av          = np.concatenate(dat[:,5])
        lam         = dat[0][6]

        A1300       = np.concatenate(dat[:,9])
        A3000       = np.concatenate(dat[:,10])

        if inp==0:
            Alam     = np.zeros((len(SFR10), len(lam)))
        elif inp==1:
            Alam     = np.zeros((len(SFR10), len(smooth_x)))
        else:
            print ("Option not specified")
            sys.exit()


        inicount, fincount = 0, 0

        for jj in range(40):
            fincount+=len(dat[jj][0])

            if inp==0:
                Alam[inicount:fincount] = dat[jj][7]
            elif inp==1:
                Alam[inicount:fincount] = dat[jj][8]
            else:
                print ("Option not specified")

            inicount=fincount

        if inp==0:
            c = np.log10(SFR100)

            for jj in range(len(cbins)-1):
                color=s_m.to_rgba((cbins[jj]+cbins[jj+1])/2)

                ok = np.where((c>=cbins[jj]) & (c<cbins[jj+1]))[0]
                if len(ok)>=5:

                    y = Alam[ok]/np.vstack(Av[ok])
                    yy = np.nanmedian(y, axis=0)
                    tck = splrep(lam, yy, s=1000)
                    yy84 = np.nanpercentile(y, 84, axis=0)
                    yy16 = np.nanpercentile(y, 16, axis=0)
                    tmp = np.linspace(np.min(lam), np.max(lam), 10)
                    axs[ii].plot(tmp, BSpline(*tck)(tmp), color=color, lw=2)

            axs[ii].text(0.2, 0.5, r'$z = {}$'.format(z), fontsize = 14)

            plot_th_curves(axs[ii], lam, 5500., lw=3, alpha=0.6)

        if inp==1:
            if ii==0:
                y = Alam/np.vstack(A1300)
            else:
                y = np.append(y, Alam/np.vstack(A1300), axis=0)

    if inp==0:

        axs[0].legend(frameon=False, fontsize=14)

        for ax in axs:
            ax.grid(True, alpha = 0.4)
            ax.minorticks_on()
            ax.tick_params(axis='x', which='minor', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.set_ylim(0.3,4.7)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(14)

        fig.subplots_adjust(wspace=0, hspace=0, bottom=0.1)

        fig.text(0.07, 0.5, r'A$_{\lambda}$/A$_{\mathrm{V}}$', va='center', rotation='vertical', fontsize=18)
        fig.text(0.48, 0.05, r'$\lambda$/$\mu$m', va='center', fontsize=18)

        fig.subplots_adjust(right = 0.91, wspace=0, hspace=0)
        cbaxes = fig.add_axes([0.925, 0.35, 0.007, 0.3])
        fig.colorbar(s_m, cax=cbaxes)
        cbaxes.set_ylabel(r'log$_{10}$(SFR/(M$_{\odot}$yr$^{-1}$))', fontsize = 12)
        for label in cbaxes.get_yticklabels():
            label.set_fontsize(13)

        plt.savefig(F"../results/att_curve_bins_sfr100.pdf", bbox_inches='tight', dpi=300)
        plt.show()


    if inp==1:
        yy = np.nanmean(y, axis=0)
        yy84 = np.nanpercentile(y, 84, axis=0)
        yy16 = np.nanpercentile(y, 16, axis=0)

        axs.plot(smooth_x, yy, color='grey', lw=1, label=r'FLARES mean')
        axs.fill_between(smooth_x, yy84, yy16, color='grey', alpha=0.2)

        plot_th_curves(axs, smooth_x, lam_norm)
        plot_Scoville15(axs)

        axs.legend(frameon=False, fontsize=12)
        
        axs.set_xlabel(r'$\lambda$/$\mu$m', fontsize=16)
        axs.set_ylabel(r'A$_{\lambda}$/A$_{\mathrm{1300}}$', fontsize=16)

        axs.grid(True, alpha = 0.4)
        axs.minorticks_on()
        axs.tick_params(axis='x', which='minor', direction='in')
        axs.tick_params(axis='y', which='minor', direction='in')
        axs.set_ylim(0,2.)
        for label in (axs.get_xticklabels() + axs.get_yticklabels()):
            label.set_fontsize(13)

        plt.savefig(F"../results/att_curve_A1300_z477_5_6_IAB25.pdf", bbox_inches='tight', dpi=300)
        plt.show()
