import numpy as np
import pandas as pd
from functools import partial
import h5py, schwimmbad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('ggplot')

from FLARE.photom import lum_to_M, M_to_lum
from interrogator.sed import dust_curves

import sys
sys.path.append('../src')

from helpers import get_slen30
import flares

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

        LFUVint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float32)
        LFUVatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float32)

        LVint   = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('V'), dtype = np.float32)
        LVatt   = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('V'), dtype = np.float32)

        beta    = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Indices/beta/'].get('DustModelI'), dtype = np.float32)
        beta0   = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Indices/beta/'].get('Intrinsic'), dtype = np.float32)


        S_ap    = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype = np.bool)


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
    mstar, SFR10, SFR100, LFUVint, LFUVatt, LVint, LVatt, beta, beta0 = mstar[ok], SFR10[ok], SFR100[ok], LFUVint[ok], LFUVatt[ok], LVint[ok], LVatt[ok], beta[ok], beta0[ok]

    AFUV    = -2.5*np.log10(LFUVatt/LFUVint)
    AV      = -2.5*np.log10(LVatt/LVint)
    delta   = np.log10(AFUV/AV)/np.log10(1500/5500)

    return mstar, SFR10, SFR100, LFUVatt, beta, beta0, AFUV, AV, delta


if __name__ == "__main__":

    quantiles = [0.84,0.50,0.16]
    limit=500
    df = pd.read_csv('../data/weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    # choose a colormap
    cmap = matplotlib.cm.coolwarm
    vmin, vmax = 0.,4

    fig, axs = plt.subplots(nrows = 1, ncols = 6, figsize=(12, 2.5), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()

    for ii, tag in enumerate(tags):
        z = float(tag[5:].replace('p','.'))


        func    = partial(get_data, tag=tag, limit=limit)
        pool    = schwimmbad.MultiPool(processes=8)
        dat     = np.array(list(pool.map(func, np.arange(0,40))))
        pool.close()

        mstar       = np.concatenate(dat[:,0])
        SFR10       = np.concatenate(dat[:,1])
        SFR100      = np.concatenate(dat[:,2])
        LFUVatt     = np.concatenate(dat[:,3])
        beta        = np.concatenate(dat[:,4])
        beta0       = np.concatenate(dat[:,5])
        AFUV        = np.concatenate(dat[:,6])
        AV          = np.concatenate(dat[:,7])
        delta       = np.concatenate(dat[:,8])

        x, y = np.log10(SFR100/mstar)+9., AFUV/AV
        z = AFUV

        if ii==0:

            cmap = matplotlib.cm.coolwarm
            vmin, vmax = np.min(z[np.isfinite(z)]), np.max(z[np.isfinite(z)])
            extent = [*[np.min(x[np.isfinite(x)])-0.1,np.max(x[np.isfinite(x)])+0.1], *[np.min(y[np.isfinite(y)])-0.1,np.max(y[np.isfinite(y)])+0.1]]
            gridsize = (50,90)

            hb = axs[ii].hexbin(x, y, C=z, reduce_C_function=np.median, cmap=cmap, vmin=vmin, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=1)

        else:
            axs[ii].hexbin(x, y, C=z, reduce_C_function=np.median, cmap=cmap, vmin=vmin, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=1)

    for ax in axs:
        ax.set_ylim(0,15)
        ax.grid(True, linestyle=(0, (0.5, 3)))
        ax.tick_params(axis="y",direction="in")
        ax.tick_params(axis="x",direction="in")

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(11)

    axs[0].legend(fontsize=12, frameon=False)

    axs[0].set_ylabel(r'A$_{\mathrm{FUV}}$/A$_{\mathrm{V}}$', fontsize=14)
    fig.text(0.5, 0.02, r'sSFR/Gyr$^{-1}$', va='center', fontsize=14)

    fig.subplots_adjust(right=0.9, wspace=0, hspace=0,bottom=0.15)
    cbaxes = fig.add_axes([0.93, 0.3, 0.01, 0.4])
    fig.colorbar(hb, cax=cbaxes, label=r'A$_{\mathrm{FUV}}$')

    # plt.savefig(F'../results/AUV_slope.pdf', bbox_inches='tight', dpi=300)

    plt.show()
