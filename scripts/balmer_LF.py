"""
Figure 9 in paper: Halpha luminosity function inferred 
from Balmer decrement
"""

import numpy as np
import pandas as pd
from functools import partial
import h5py, schwimmbad
import matplotlib.pyplot as plt
import cmasher as cmr

from interrogator.sed import dust_curves
from uvlf import get_flares_LF

import sys
sys.path.append('../src')

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


        # S_ap    = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype = bool)

        beta = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Indices/beta'].get('DustModelI'), dtype = np.float32)

        ####Adding beta and beta0 to see their effect on these curve shape

    # z = float(tag[5:].replace('p','.'))

    # S_len, begin, end, S_ap = get_slen30(S_len, S_ap, limit)
    # ok = np.where(S_len>limit)[0]

    # Mstar, SFR10, SFR100, LFUVint, LFUVatt, LVint, LVatt, LBint, LBatt, Halpha, Hbeta, Halphaint, Hbetaint, Hgammaint, Hgamma, beta = Mstar[ok], SFR10[ok], SFR100[ok], LFUVint[ok], LFUVatt[ok], LVint[ok], LVatt[ok], LBint[ok], LBatt[ok], Halpha[ok], Hbeta[ok], Halphaint[ok], Hbetaint[ok], Hgammaint[ok], Hgamma[ok], beta[ok]

    # balmer1     = Halpha/Hbeta
    # balmer1int  = Halphaint/Hbetaint
    # balmer2     = Hbeta/Hgamma
    # balmer2int  = Hbetaint/Hgammaint
    
    Halpha_lam  = 6562.81
    Hbeta_lam   = 4861.33
    Hgamma_lam  = 4340.46

    #Simple attenuation curve, powerlaw
    EBV_balmer = get_EBV(Halpha/Hbeta, Halpha_lam, Hbeta_lam, 2.79, dust_curves.simple())
    Halpha_bcorr_pl = calc_line_corr(Halpha, Halpha_lam, EBV_balmer, dust_curves.simple()) 
    Hbeta_bcorr_pl = calc_line_corr(Hbeta, Hbeta_lam, EBV_balmer, dust_curves.simple()) 

    #Calzetti attenuation curve delta=0
    EBV_balmer = get_EBV(Halpha/Hbeta, np.array([Halpha_lam]), np.array([Hbeta_lam]), 2.79, dust_curves.Starburst_Calzetti2000())
    Halpha_bcorr_cal_delta_0 = calc_line_corr(Halpha, np.array([Halpha_lam]), EBV_balmer, dust_curves.Starburst_Calzetti2000()) 
    Hbeta_bcorr_cal_delta_0 = calc_line_corr(Hbeta, np.array([Halpha_lam]), EBV_balmer, dust_curves.Starburst_Calzetti2000()) 

    #Calzetti attenuation curve delta=0.5
    EBV_balmer = get_EBV(Halpha/Hbeta, np.array([Halpha_lam]), np.array([Hbeta_lam]), 2.79, dust_curves.Starburst_Calzetti2000(), delta=0.5)
    Halpha_bcorr_cal_delta_0p5 = calc_line_corr(Halpha, np.array([Halpha_lam]), EBV_balmer, dust_curves.Starburst_Calzetti2000(), delta=0.5) 
    Hbeta_bcorr_cal_delta_0p5 = calc_line_corr(Hbeta, np.array([Halpha_lam]), EBV_balmer, dust_curves.Starburst_Calzetti2000(), delta=0.5) 

    return np.log10(Halpha), np.log10(Halphaint), np.log10(Halpha_bcorr_pl), np.log10(Halpha_bcorr_cal_delta_0), np.log10(Halpha_bcorr_cal_delta_0p5)  

def calc_line_corr(line_lum, line_lam, EBV, curve, delta=0):
    
    k_line = curve.tau(line_lam) * (line_lam**delta)

    A_line = EBV * k_line

    y = line_lum * 10**(A_line/2.5)
    
    return y

def get_EBV(Balmer_obs, lam1, lam2, intr_ratio, curve, delta=0):

    k_1 = curve.tau(lam1) * (lam1**delta)
    k_2 = curve.tau(lam2) * (lam2**delta)

    EBV = (2.5/(k_2-k_1)) * np.log10(Balmer_obs/intr_ratio)

    return EBV

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

    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(12, 8), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()

    bins = np.arange(39, 47, 0.4)
    bincen = (bins[1:]+bins[:-1])/2.
    binwidth = bins[1:] - bins[:-1]
    h = 0.6777
    vol = (4/3)*np.pi*(14/h)**3

    for ii, tag in enumerate(tags[::-1]):

        z = float(tag[5:].replace('p','.'))

        # axs[ii].text(0.5, 0.3, r'$z = {}$'.format(z), fontsize = 12)

        func    = partial(get_data, tag=tag, limit=limit)
        pool    = schwimmbad.MultiPool(processes=8)
        dat     = np.array(list(pool.map(func, np.arange(0,40))), dtype='object')
        pool.close()


        Halpha      = np.concatenate(dat[:,0])
        Halphaint   = np.concatenate(dat[:,1])
        
        Halpha_bcorr_pl             = np.concatenate(dat[:,2])
        Halpha_bcorr_cal_delta_0    = np.concatenate(dat[:,3])
        Halpha_bcorr_cal_delta_0p5  = np.concatenate(dat[:,4])

        
        n = np.array([])
        for jj in range(40):
            n = np.append(n, len(dat[jj][0]))


        hist, M, err = get_flares_LF(Halphaint, weights, bins, n)
        ok = np.where(hist>=5)[0]
        phi, phierr = M/(binwidth*vol), err/(binwidth*vol)
        axs[ii].plot(bincen, np.log10(phi), lw=4, alpha=0.4, ls='solid', label='Intrinsic', color='blue')
        axs[ii].fill_between(bincen[ok], np.log10(phi[ok]-phierr[ok]), np.log10(phi[ok]+phierr[ok]), alpha=0.25, color='blue')

        hist, M, err = get_flares_LF(Halpha_bcorr_pl, weights, bins, n)
        ok = np.where(hist>=5)[0]
        phi, phierr = M/(binwidth*vol), err/(binwidth*vol)
        axs[ii].plot(bincen, np.log10(phi), lw=4, alpha=0.4, ls='solid', label='Balmer corrected (Input)', color='grey')
        # axs[ii].fill_between(bincen[ok], np.log10(phi[ok]-phierr[ok]), np.log10(phi[ok]+phierr[ok]), alpha=0.25, color='grey')

        hist, M, err = get_flares_LF(Halpha_bcorr_cal_delta_0, weights, bins, n)
        ok = np.where(hist>=5)[0]
        phi, phierr = M/(binwidth*vol), err/(binwidth*vol)
        axs[ii].plot(bincen, np.log10(phi), lw=4, alpha=0.4, ls='dotted', label=r'Balmer corrected (Calzetti; $\delta=0$)', color='magenta')
        # axs[ii].fill_between(bincen[ok], np.log10(phi[ok]-phierr[ok]), np.log10(phi[ok]+phierr[ok]), alpha=0.25, color='magenta')

        hist, M, err = get_flares_LF(Halpha_bcorr_cal_delta_0p5, weights, bins, n)
        ok = np.where(hist>=5)[0]
        phi, phierr = M/(binwidth*vol), err/(binwidth*vol)
        axs[ii].plot(bincen, np.log10(phi), lw=4, alpha=0.2, ls='solid', color='magenta', label=r'Balmer corrected (Calzetti; $\delta=0.5$)')
        axs[ii].fill_between(bincen[ok], np.log10(phi[ok]-phierr[ok]), np.log10(phi[ok]+phierr[ok]), alpha=0.15, color='magenta')

        hist, M, err = get_flares_LF(Halpha, weights, bins, n)
        ok = np.where(hist>=5)[0]
        phi, phierr = M/(binwidth*vol), err/(binwidth*vol)
        axs[ii].plot(bincen, np.log10(phi), lw=4, alpha=0.4, ls='solid', label='Observed', color='orange')
        axs[ii].fill_between(bincen[ok], np.log10(phi[ok]-phierr[ok]), np.log10(phi[ok]+phierr[ok]), alpha=0.25, color='orange')

        axs[ii].text(42.4, -6.5, r'$z = {}$'.format(z), fontsize = 12)

    for ax in axs:
        ax.grid(True, alpha=0.6)
        ax.set_xticks(np.arange(42.5, 44.2, 0.5))
        ax.set_xlim(42.2, 44.2)
        ax.set_ylim(-8.4, -1.7)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)

    axs[0].legend(frameon=False, fontsize=10)
    axs[4].set_xlabel(r'log$_{10}$($\mathrm{H}_{\alpha}$/(erg/s))', fontsize=15)

    fig.subplots_adjust(bottom=0.12, wspace=0, hspace=0)

    fig.text(0.065, 0.5, r'log$_{10}$($\Phi$/($\mathrm{cMpc}^{-3}\mathrm{dex}^{-1}$))', va='center', rotation='vertical', fontsize=15)


    plt.savefig(F"../results/Balmer_corr_Halpha_z5_10.pdf", bbox_inches='tight', dpi=300)
    plt.show()