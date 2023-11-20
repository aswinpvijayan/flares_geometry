"""
Figure 3 in paper: The intrinsic UV LF and the observed UV LF 
for the line-of-sight model and the one obtained assuming a 
linear relationship between intrinsic UV and AFUV
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['legend.fancybox'] = False
import matplotlib.pyplot as plt
from FLARE.photom import lum_to_M
from functools import partial
import schwimmbad

import seaborn as sns
sns.set_context("paper")

def get_beta(Muv, z):

    zs = np.array([4.0, 5.0, 5.9, 7.0, 8.0])
    beta_Muvs = np.array([-2.00, -2.08, -2.20, -2.27, -2.34])
    dbeta_dMuvs = np.array([-0.13, -0.16, -0.17, -0.21, -0.25])

    M0          = -19.5
    c           = -2.33
    dbeta_dMuv  = np.interp(z, zs, dbeta_dMuvs)
    beta_Muv    = np.interp(z, zs, beta_Muvs)


    # ok = (Muv >= M0)
    beta_mean = np.zeros(len(Muv))
    # beta_mean[ok]   =  (beta_Muv - c) * np.exp(-(dbeta_dMuv * (Muv[ok] - M0)/(beta_Muv - c))) + c
    beta_mean  =  dbeta_dMuv * (Muv - M0) + beta_Muv

    return beta_mean

def mean_Afuv(Muv, z):

    sigma_beta      = 0.34

    mu, sigma = 0, sigma_beta
    s = np.random.normal(mu, sigma, len(Muv))


    out = 4.43 + 0.79 * np.log(10) * (sigma**2) + 1.99 * get_beta(Muv, z)

    out[out<0] = 0

    return out

def get_flares_LF(dat, weights, bins, n):

    sims = np.arange(0,len(weights))

    hist = np.zeros(len(bins)-1)
    out = np.zeros(len(bins)-1)
    err = np.zeros(len(bins)-1)

    nsum = np.cumsum(n)
    nsum = np.append([0], nsum)
    nsum = nsum.astype(np.int32)

    for ii, sim in enumerate(sims):
        h, edges = np.histogram(dat[nsum[ii]:nsum[ii+1]], bins = bins)
        hist+=h
        out+=h*weights[ii]
        err+=np.square(np.sqrt(h)*weights[ii])

    return hist, out, np.sqrt(err)



def get_data(ii, tag, limit=500, bins = np.arange(0,5.25,0.25)):

    z = float(tag[5:].replace('p','.'))

    num = str(ii)

    if len(num) == 1:
        num =  '0'+num

    sim = rF"../../flares_pipeline/data/flares.hdf5"
    num = num+'/'

    with h5py.File(sim, 'r') as hf:
        S_len       = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)
        S_ap        = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype = bool)

        mstar   = np.array(hf[num+tag+'/Galaxy/Mstar_aperture/'].get('30'), dtype = np.float32)*1e10
        SFR10   = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('30/10Myr'), dtype = np.float32)
        SFR100  = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('30/100Myr'), dtype = np.float32)
        LFUVint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float64)
        LFUVatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float64)


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

    AFUV = -2.5*np.log10(LFUVatt/LFUVint)

    MFUVint     = lum_to_M(LFUVint)
    MFUVatt     = lum_to_M(LFUVatt)
    AFUVbeta    = mean_Afuv(MFUVint, z)


    return mstar, SFR10, SFR100, MFUVint, MFUVatt, AFUV, AFUVbeta


if __name__ == "__main__":

    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    zs = [5, 6, 7, 8, 9, 10]
    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(12, 7), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()

    bins = -np.arange(16, 25, 0.5)[::-1]
    bincen = (bins[1:]+bins[:-1])/2.
    binwidth = bins[1:] - bins[:-1]
    h = 0.6777
    vol = (4/3)*np.pi*(14/h)**3


    for ii, tag in enumerate(tags[::-1]):

        z = float(tag[5:].replace('p','.'))

        func    = partial(get_data, tag=tag)
        pool    = schwimmbad.MultiPool(processes=8)
        dat     = np.array(list(pool.map(func, np.arange(0,40))), dtype='object')
        pool.close()

        Mstar       = np.concatenate(dat[:,0])
        SFR10       = np.concatenate(dat[:,1])
        SFR100      = np.concatenate(dat[:,2])
        MFUVint     = np.concatenate(dat[:,3])
        MFUVatt     = np.concatenate(dat[:,4])
        AFUV        = np.concatenate(dat[:,5])
        AFUVbeta    = np.concatenate(dat[:,6])

        n = np.array([])
        for jj in range(40):
            n = np.append(n, len(dat[jj][0]))


        MFUV_obs = MFUVint + AFUVbeta

        hist, M, err = get_flares_LF(MFUVint, weights, bins, n)
        ok = np.where(hist>=5)[0]
        phi, phierr = M/(binwidth*vol), err/(binwidth*vol)
        axs[ii].plot(bincen, np.log10(phi), lw=2, alpha=0.8, ls='dashed', label='Intrinsic', color='magenta')
        axs[ii].fill_between(bincen[ok], np.log10(phi[ok]-phierr[ok]), np.log10(phi[ok]+phierr[ok]), alpha=0.4, color='magenta')



        hist, M, err = get_flares_LF(MFUV_obs, weights, bins, n)
        ok = np.where(hist>=5)[0]
        phi, phierr = M/(binwidth*vol), err/(binwidth*vol)
        axs[ii].plot(bincen, np.log10(phi), lw=2, alpha=0.7, ls='dashed', label='Scaled relation', color='orange')
        axs[ii].fill_between(bincen[ok], np.log10(phi[ok]-phierr[ok]), np.log10(phi[ok]+phierr[ok]), alpha=0.7, color='orange')



        hist, M, err = get_flares_LF(MFUVatt, weights, bins, n)
        ok = np.where(hist>=5)[0]
        phi, phierr = M/(binwidth*vol), err/(binwidth*vol)
        axs[ii].plot(bincen, np.log10(phi), lw=2, alpha=0.7, ls='solid', label='LOS model', color='green')
        axs[ii].fill_between(bincen[ok], np.log10(phi[ok]-phierr[ok]), np.log10(phi[ok]+phierr[ok]), alpha=0.7, color='green')

        axs[ii].text(-18.6, -6.5, r'$z = {}$'.format(z), fontsize = 14)

    for ax in axs:
        ax.grid(True, alpha=0.6)
        ax.set_xlim(-17.9, -24.7)
        ax.set_ylim(-9.4, -1.7)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)

    axs[0].legend(frameon=False, fontsize=12)

    fig.subplots_adjust(bottom=0.09, left=0.08, wspace=0, hspace=0)

    fig.text(0.025, 0.5, r'$\mathrm{log}_{10}(\Phi/(\mathrm{cMpc}^{-3}\mathrm{Mag}^{-1}))$', va='center', rotation='vertical', fontsize=15)
    fig.text(0.48, 0.03, r'$\mathrm{M}_{\mathrm{FUV}}$', va='center', fontsize=15)

    plt.savefig(F"../results/LF_FUV_z5_10.pdf", bbox_inches='tight', dpi=300)
    plt.show()
