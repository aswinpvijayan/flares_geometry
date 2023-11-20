"""
Figure 1 in paper: Relationship between galaxy stellar mass 
(left) and the galaxy UV luminosity (right) against their 
star formation rate (SFR)
"""

import numpy as np
import pandas as pd
from functools import partial
import schwimmbad, h5py, matplotlib
import cmasher as cmr
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")
import flares
from FLARE.photom import lum_to_M

def get_data(ii, tag, limit=1000):

    region = str(ii)
    if len(region) == 1:
        region = '0'+region

    with h5py.File(F'../../flares_pipeline/data/flares.hdf5', 'r') as hf:
        mstar       = np.array(hf[region+'/'+tag+'/Galaxy/Mstar_aperture'].get('30'), dtype=np.float64)*1e10
        MFUV        = lum_to_M(np.array(hf[region+'/'+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float64))
        SFR         = np.array(hf[region+'/'+tag+'/Galaxy/SFR_aperture/30'].get('100Myr'), dtype=np.float64)
        S_ap        = np.array(hf[region+'/'+tag+'/Particle/Apertures/Star'].get('30'), dtype=bool)
        S_len       = np.array(hf[region+'/'+tag+'/Galaxy'].get('S_Length'), dtype=np.int32)


    begin = np.zeros(len(S_len), dtype=np.int32)
    end = np.zeros(len(S_len), dtype=np.int32)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)
    begin = begin.astype(np.int32)
    end = end.astype(np.int32)

    n = len(S_len)
    S_len30 = np.zeros(n, dtype=np.int32)
    for kk in range(n):
        S_len30[kk] = np.sum(S_ap[begin[kk]:end[kk]])

    ok = np.where(S_len30>limit)[0]

    return mstar[ok], MFUV[ok], SFR[ok]

def get_hist(x, bins):

    hist, edges = np.histogram(x, bins = bins)
    left,right = edges[:-1],edges[1:]
    X = np.array([left,right]).T.flatten()
    Y = np.log10(np.array([hist,hist]).T.flatten())
    Y[Y<0] = -1

    return X, Y


if __name__ == "__main__":

    limit   = 500
    quantiles = [0.84, 0.50, 0.16]

    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    tags = ['010_z005p000','009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    fig     = plt.figure(figsize = (10, 5))
    # choose a colormap
    c_m =plt.get_cmap('cmr.bubblegum_r')
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5,6,1), c_m.N)
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])


    axM     = fig.add_axes((0.10, 0.10, 0.30, 0.50))
    axM_T   = fig.add_axes((0.10, 0.60, 0.30, 0.35))
    axS     = fig.add_axes((0.40, 0.10, 0.30, 0.50))
    axS_T   = fig.add_axes((0.40, 0.60, 0.30, 0.35))
    ax_R    = fig.add_axes((0.70, 0.10, 0.20, 0.50))


    x1bins      = np.arange(8.8, 12.5, 0.5)
    x1bincen    = (x1bins[1:]+x1bins[:-1])/2.
    x1label     = r'log$_{10}$(M$_{\star}$/M$_{\odot}$)'
    x1lim       = [8.8, 11.9]

    x2bins      = -np.arange(16.8, 24.5, 0.5)[::-1]
    x2bincen    = (x2bins[1:]+x2bins[:-1])/2.
    x2label     = r'M$_{\mathrm{FUV}}$'
    x2lim       = [-16.8,-24.8]

    savename    = '../results/Mstar_Mfuv_sfr.pdf'

    ylabel = r'log$_{10}$(SFR/(M$_{\odot}$yr$^{-1}$))'

    num = []
    x1tot = np.array([])
    x2tot = np.array([])
    ytot = np.array([])


    for ii, tag in enumerate(tags):

        z = float(tag[5:].replace('p','.'))

        func = partial(get_data, tag=tag, limit=limit)
        pool = schwimmbad.MultiPool(processes=8)
        dat = np.array(list(pool.map(func, np.arange(0,40))), dtype='object')
        pool.close()

        mstar   = np.concatenate(dat[:,0])
        mfuv    = np.concatenate(dat[:,1])
        y       = np.concatenate(dat[:,2])

        x1          = np.log10(mstar)
        x2          = mfuv

        x1tot       = np.append(x1tot, x1)
        x2tot       = np.append(x2tot, x2)

        ws = np.zeros(len(mstar))
        n = 0
        for jj in range(40):
            if jj==0:
                ws[0:len(dat[jj][0])] = weights[jj]
            else:
                ws[n:n+len(dat[jj][0])] = weights[jj]

            n+=len(dat[jj][0])


        y = np.log10(y)
        ytot = np.append(ytot, y)

        num.append(len(x1))
        out = flares.binned_weighted_quantile(x1, y, ws, x1bins, quantiles)
        hist, edges = np.histogram(x1, x1bins)
        ok = np.where(hist>5)[0]
        axM.errorbar(x1bincen[ok], out[:,1][ok], yerr = [out[:,0][ok] - out[:,1][ok], out[:,1][ok] - out[:,2][ok]], color=s_m.to_rgba(ii), zorder=10, lw=2, label = rF'z={z} ({len(x1)})')

        out = flares.binned_weighted_quantile(x2, y, ws, x2bins, quantiles)
        hist, edges = np.histogram(x2, x2bins)
        ok = np.where(hist>5)[0]
        axS.errorbar(x2bincen[ok], out[:,1][ok], yerr = [out[:,0][ok] - out[:,1][ok], out[:,1][ok] - out[:,2][ok]], color=s_m.to_rgba(ii), zorder=10, lw=2)


        X, Y = get_hist(x1, bins = x1bins)
        axM_T.plot(X, Y, lw = 2, color=s_m.to_rgba(ii))

        X, Y = get_hist(x2, bins = x2bins)
        axS_T.plot(X, Y, lw = 2, color=s_m.to_rgba(ii))

        ybins = np.arange(-1, 4, 0.5)
        X, Y = get_hist(y, bins = ybins)
        ax_R.plot(Y, X, lw = 2, color=s_m.to_rgba(ii))


    hb1 = axM.hexbin(x1tot, ytot, cmap=plt.get_cmap('cmr.horizon'), gridsize=[32,16], extent=[8.8,11.5, -1,3.], linewidths=0., mincnt=1, alpha=0.5, bins='log', vmin=1, vmax=430)
    hb2 = axS.hexbin(x2tot, ytot, cmap=plt.get_cmap('cmr.horizon'), gridsize=[37,16], extent=[-24.8,-16.8, -1,3.], linewidths=0., mincnt=1, alpha=0.5, bins='log', vmin=1, vmax=430)


    for axs in [axM, axS, ax_R, axM_T, axS_T]:
        axs.grid(True, ls='dotted')
        for label in (axs.get_xticklabels() + axs.get_yticklabels()):
            label.set_fontsize(11)


    axM.set_xlim(x1lim)
    axM_T.set_xlim(x1lim)
    axM_T.set_xticklabels([])
    axM_T.set_ylim(0,4)

    axS.set_xlim(x2lim)
    axS.set_yticklabels([])
    axS_T.set_xlim(x2lim)
    axS_T.set_xticklabels([])
    axS_T.set_ylim(0,4)
    axS_T.set_yticklabels([])

    axM.set_ylim([-1,3.4])
    axS.set_ylim([-1,3.4])
    ax_R.set_ylim([-1,3.4])
    ax_R.set_yticklabels([])

    ax_R.set_xlim(left=0.)
    axM_T.set_ylim(bottom=0.)
    axS_T.set_ylim(bottom=0.)

    ax_R.set_xticks(np.arange(0.,4.,1.))
    axM_T.set_yticks(np.arange(0.,4.,1.))
    axS_T.set_yticks(np.arange(0.,4.,1.))

    ax_R.set_xlabel(r'log$_{10}$(N)', fontsize = 12)
    fig.text(0.035, 0.80, r'log$_{10}$(N)', va='center', rotation='vertical', fontsize=12)

    axM.set_xlabel(x1label, fontsize = 12)
    axS.set_xlabel(x2label, fontsize = 12)
    axM.set_ylabel(ylabel, fontsize = 12)
    axM.legend(frameon=False, fontsize=9, scatterpoints=1, loc=4, markerscale=3)

    cbaxes = fig.add_axes([0.72, 0.8, 0.17, 0.025])
    fig.colorbar(hb1, cax=cbaxes, orientation="horizontal")
    cbaxes.set_xlabel(r'N', fontsize = 12)
    for label in cbaxes.get_xticklabels():
        label.set_fontsize(13)

    plt.savefig(savename, bbox_inches='tight', dpi=300)

    plt.show()
