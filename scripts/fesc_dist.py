"""
Figure 4 in paper: Here inp=sys.argv[1]=1 for FUV plot.
variation of the dust obscuration in the UV within galaxies
as a function of the observed UV luminosity
"""

import numpy as np
import pandas as pd
from functools import partial
import h5py, schwimmbad
import matplotlib
import matplotlib.pyplot as plt
import cmasher as cmr
from scipy.stats import spearmanr, pearsonr

from FLARE.photom import lum_to_M
import sys
sys.path.append('../src')

from helpers import get_slen30
import flares

def get_data(ii, tag, limit=500, aperture='30', percentile=68):

    num = str(ii)

    if len(num) == 1:
        num =  '0'+num

    sim_orient = rF"../data/FLARES_{num}_data.hdf5"
    sim = rF"../../flares_pipeline/data/flares.hdf5"
    num = num+'/'

    with h5py.File(sim, 'r') as hf:
        S_len       = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)

        mstar   = np.array(hf[num+tag+'/Galaxy/Mstar_aperture'].get('30'), dtype = np.float32)*1e10
        SFR10   = np.array(hf[num+tag+'/Galaxy/SFR_aperture/30'].get('10Myr'), dtype = np.float32)
        SFR100  = np.array(hf[num+tag+'/Galaxy/SFR_aperture/30'].get('100Myr'), dtype = np.float32)
        wAge    = np.array(hf[num+tag+'/Galaxy/StellarAges'].get('MassWeightedStellarAge'), dtype = np.float32)
        wZ   = np.array(hf[num+tag+'/Galaxy/Metallicity'].get('MassWeightedGasZ'), dtype = np.float32)

        LFUVint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float32)
        LFUVatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float32)
        LFUVbc  = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/No_ISM'].get('FUV'), dtype = np.float32)

        S_ap    = np.array(hf[num+tag+'/Particle/Apertures/Star'].get(aperture), dtype = bool)

    begin = np.zeros(len(S_len), dtype = np.int32)
    end = np.zeros(len(S_len), dtype = np.int32)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    S_len30, begin30, end30, S_ap30 = get_slen30(S_len, S_ap, limit)

    del S_len
    ok = np.where(S_len30>limit)[0]

    mstar, SFR10, SFR100, S_len30, wAge, wZ, a, b = mstar[ok], SFR10[ok], SFR100[ok], S_len30[ok], wAge[ok], wZ[ok], LFUVint[ok], LFUVatt[ok]


    begin30 = np.zeros(len(S_len30), dtype = np.int32)
    end30 = np.zeros(len(S_len30), dtype = np.int32)
    begin30[1:] = np.cumsum(S_len30)[:-1]
    end30 = np.cumsum(S_len30)


    with h5py.File(sim_orient, 'r') as hf:
        LFUVint_part = np.array(hf[tag+'/Particle/BPASS_2.2.1/Chabrier300/zaxis/Luminosity/Intrinsic'].get('FUV'), dtype = np.float64)
        LFUVatt_part = np.array(hf[tag+'/Particle/BPASS_2.2.1/Chabrier300/zaxis/Luminosity/DustModelI'].get('FUV'), dtype = np.float64)
        LFUVbc_part = np.array(hf[tag+'/Particle/BPASS_2.2.1/Chabrier300/zaxis/Luminosity/No_ISM'].get('FUV'), dtype = np.float64)

    fbc     = LFUVbc_part/LFUVint_part
    fdust   = LFUVatt_part/LFUVint_part
    LFUVonlyism_part = LFUVint_part * (fdust/fbc)


    LFUVint = np.zeros(len(begin30))
    LFUVatt = np.zeros(len(begin30))

    fesc_mean    = np.zeros(len(begin30))
    fesc_med     = np.zeros(len(begin30))
    fesc_16perc  = np.zeros(len(begin30))
    fesc_84perc  = np.zeros(len(begin30))

    att_mean    = np.zeros(len(begin30))
    att_med     = np.zeros(len(begin30))
    att_16perc  = np.zeros(len(begin30))
    att_84perc  = np.zeros(len(begin30))

    Agesplit    = np.zeros(len(begin30), dtype=bool)

    if len(ok)>0:
        for ii, jj in enumerate(begin30):

            int = LFUVint_part[begin30[ii]:end30[ii]]#[S_ap30[begin30[ii]:end30[ii]]]
            att = LFUVatt_part[begin30[ii]:end30[ii]]#[S_ap30[begin30[ii]:end30[ii]]]
            ism = LFUVonlyism_part[begin30[ii]:end30[ii]]

            LFUVint[ii] = np.sum(int)
            LFUVatt[ii] = np.nansum(att)

            x           = att/int
            quantiles   = [0.84,0.50,0.16]
            out         = flares.binned_weighted_quantile(np.zeros_like(x), x, np.ones_like(x), [0.,1.], quantiles)

            fesc_mean[ii]     = np.nansum(x*int) / np.nansum(int)
            fesc_med[ii]      = out[1]
            fesc_16perc[ii]   = out[2]
            fesc_84perc[ii]   = out[0]

            x           = -2.5*np.log10(x)
            quantiles   = [0.84,0.50,0.16]
            out         = flares.binned_weighted_quantile(np.zeros_like(x), x, np.ones_like(x), [0.,100.], quantiles)

            att_mean[ii]     = np.nansum(x*int) / np.nansum(int)
            att_med[ii]      = out[1]
            att_16perc[ii]   = out[2]
            att_84perc[ii]   = out[0]


        Agesplit = (SFR100<np.percentile(SFR100, percentile))


    return mstar, SFR10, SFR100, LFUVint, LFUVatt, LFUVint_part, LFUVatt_part, fesc_mean, fesc_med, fesc_16perc, fesc_84perc, att_mean, att_med, att_16perc, att_84perc, Agesplit, wAge, wZ

def plt_histogram(data, bins, norm, axs, **kwargs):

    bin_width = np.diff(bins)
    bincen = (bins[1:]+bins[:-1])/2.
    counts = np.histogram(data, bins=bins)[0]
    y = counts/(norm * bin_width)

    axs.hist(bincen, bins, **kwargs, weights=y)

if __name__ == "__main__":

    inp = int(sys.argv[1])

    quantiles = [0.84,0.50,0.16]
    limit=500
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    if inp==0:
        title = 'mstar'
        xlabel = r'log$_{10}$(M$_{\star}$/M$_{\odot}$)'
        bins = np.arange(8.5, 11.75, 0.5)
        xlim = [8.7,11.4]
    elif inp==1:
        title = 'fuv'
        xlabel = r'M$_{\mathrm{FUV}}$'
        bins = -np.arange(18, 25, 1)[::-1]
        xlim = [-17.8,-23.8]

    vmin, vmax = -1, 3

    bins_label = np.array([F'${bins[ii]}$ - ${bins[ii+1]}$' for ii in range(len(bins)-1)])
    afuvbins = np.arange(0,4.25,0.25)

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # choose a colormap
    c_m = plt.get_cmap('cmr.cosmic')#matplotlib.cm.coolwarm
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)

    aperture = '30' #30kpc
    percentile = 50

    # fig = plt.figure(figsize=(13, 8))
    fig = plt.figure(figsize=(13, 8))
    axs = fig.subplot_mosaic(
    """
    AABCCDEEF
    GGHIIJKKL
    """
    )
    plt.subplots_adjust(hspace=0, wspace=0)
    
    axs_num = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F',
               6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L'}

    for ii, tag in enumerate(tags[::-1]):

        z = float(tag[5:].replace('p','.'))

        func    = partial(get_data, tag=tag, limit=limit, aperture=aperture, percentile=percentile)
        pool    = schwimmbad.MultiPool(processes=1)
        dat     = np.array(list(pool.map(func, np.arange(0,40))), dtype=object)
        pool.close()

        mstar   = np.concatenate(dat[:,0])
        SFR10   = np.concatenate(dat[:,1])
        SFR100  = np.concatenate(dat[:,2])
        wAge    = np.concatenate(dat[:,-2])
        wZ      = np.concatenate(dat[:,-1])

        Agesplit    = np.concatenate(dat[:,-3])

        LFUVint     = np.zeros(len(mstar))
        LFUVatt     = np.zeros(len(mstar))

        fesc_mean    = np.zeros(len(mstar))
        fesc_med     = np.zeros(len(mstar))
        fesc_16perc  = np.zeros(len(mstar))
        fesc_84perc  = np.zeros(len(mstar))

        ws = np.zeros(len(mstar))

        inicount, fincount = 0, 0

        for jj in range(40):
            fincount+=len(dat[jj][0])

            ws[inicount:fincount] = np.ones(len(dat[jj][0]))*weights[jj]


            LFUVint[inicount:fincount] = dat[:,3][jj]
            LFUVatt[inicount:fincount] = dat[:,4][jj]

            fesc_mean[inicount:fincount]     = dat[:,7][jj]
            fesc_med[inicount:fincount]      = dat[:,8][jj]
            fesc_16perc[inicount:fincount]   = dat[:,9][jj]
            fesc_84perc[inicount:fincount]   = dat[:,10][jj]

            inicount=fincount

        LFUVint_part = np.concatenate(dat[:,5])
        LFUVatt_part = np.concatenate(dat[:,6])

        del dat

        sSFR = SFR100/mstar
        AFUV = -2.5*np.log10(LFUVatt/LFUVint)
        c = np.log10(SFR100)

        if inp==0:
            x=np.log10(mstar)
        elif inp==1:
            x = lum_to_M(LFUVatt)
            print (np.max(x), np.min(x))

        print (F"SFR {percentile} percentile: ", np.percentile(c, percentile))

        print ("-----------------fesc_perc vs Auv----------------------\n")
        ok = np.isfinite(AFUV)*np.isfinite(fesc_84perc - fesc_16perc)
        print ("pearsonr: ", pearsonr(AFUV[ok], fesc_84perc[ok] - fesc_16perc[ok]), "\n")
        print ("spearmanr: ", spearmanr(AFUV[ok], fesc_84perc[ok] - fesc_16perc[ok]))
        print ("---------------------------------------")

        print ("-----------------SFR100 vs fesc_perc----------------------\n")
        ok = np.isfinite(SFR100)*np.isfinite(fesc_84perc - fesc_16perc)
        print ("pearsonr: ", pearsonr(SFR100[ok], fesc_84perc[ok] - fesc_16perc[ok]), "\n")
        print ("spearmanr: ", spearmanr(SFR100[ok], fesc_84perc[ok] - fesc_16perc[ok]))
        print ("---------------------------------------")

        axs[axs_num[2*ii]].hexbin(x, fesc_84perc - fesc_16perc, C=c, reduce_C_function=np.median, cmap=c_m, vmin=vmin, vmax=vmax, gridsize=(30,20), extent=(*xlim, *[0,1]))

        counts, bins = np.histogram(fesc_84perc - fesc_16perc, bins=np.arange(0,1.1,0.1))
        total_count = np.sum(counts)
        plt_histogram(fesc_84perc - fesc_16perc, bins, total_count, axs[axs_num[2*ii+1]], color='blue', orientation='horizontal', histtype='stepfilled', alpha=0.2)
        plt_histogram(fesc_84perc[Agesplit] - fesc_16perc[Agesplit], bins, total_count, axs[axs_num[2*ii+1]], color='black', lw=1, orientation='horizontal', ls='dotted', histtype='step') # less than the specified percentile
        plt_histogram(fesc_84perc[~Agesplit] - fesc_16perc[~Agesplit], bins, total_count, axs[axs_num[2*ii+1]], color='black', lw=1, orientation='horizontal', ls='dashed', histtype='step') # greater than the specified percentile

        axs[axs_num[2*ii+1]].text(0.6, 0.5, r'$z = {}$'.format(z), horizontalalignment='center', verticalalignment='center', transform=axs[axs_num[2*ii+1]].transAxes, fontsize = 14)

        axs[axs_num[2*ii]].set_xlim(xlim)
        axs[axs_num[2*ii+1]].set_xlim(0,3.6)

   

    for ax in axs:
        axs[ax].set_ylim(-0.05,1.05)
        axs[ax].grid(True, linestyle=(0, (0.5, 3)))
        axs[ax].minorticks_on()
        axs[ax].tick_params(axis='x', which='minor', direction='in')
        axs[ax].tick_params(axis='y', which='minor', direction='in')

        for label in (axs[ax].get_xticklabels() + axs[ax].get_yticklabels()):
            label.set_fontsize(12)


    for kk in range(1,12,2):
        axs[axs_num[kk]].set_xticks([0, 1, 2, 3])

    for kk in range(3):
        axs[axs_num[6+2*kk]].set_xlabel(xlabel, fontsize=12)
        axs[axs_num[6+2*kk+1]].set_xlabel('PDF', fontsize=12)

    for kk in range(6):
        axs[axs_num[kk]].set_xticklabels([])
        axs[axs_num[2*kk]].set_xlim(xlim)

    for kk in range(12):
        if (kk==0) or (kk==6):
            axs[axs_num[kk]].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axs[axs_num[kk]].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            axs[axs_num[kk]].set_yticklabels([])


    fig.subplots_adjust(right=0.95, wspace=0, hspace=0)

    fig.text(0.07, 0.5, r"f$_{\mathrm{esc,84}}$ $-$ f$_{\mathrm{esc,16}}$", va='center', rotation='vertical', fontsize=16)

    cbaxes = fig.add_axes([0.97, 0.3, 0.01, 0.4])
    fig.colorbar(s_m, cax=cbaxes, orientation='vertical')
    cbaxes.set_ylabel(r'log$_{10}$(SFR/(M$_{\odot}$yr$^{-1}$))', fontsize=12)
    for label in cbaxes.get_yticklabels():
        label.set_fontsize(12)

    plt.savefig(F'../results/AFUV_percentile_diff_{title}_{aperture}ap_z5_10_SFR_wSFR.pdf', bbox_inches='tight', dpi=300)
    plt.show()
