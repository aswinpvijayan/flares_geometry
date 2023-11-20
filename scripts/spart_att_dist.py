"""
Figure 
"""

import numpy as np
import pandas as pd
from functools import partial
import h5py, schwimmbad
import matplotlib.pyplot as plt
from flare.photom import lum_to_M, M_to_lum

import sys
sys.path.append('../src')

from helpers import get_slen30, compute_kappa
import flares

def get_data(ii, tag, limit=1000):

    num = str(ii)

    if len(num) == 1:
        num =  '0'+num

    sim_orient = rF"../data1/FLARES_{num}_data.hdf5"
    sim = rF"../../flares_pipeline/data/flares.hdf5"
    num = num+'/'

    with h5py.File(sim, 'r') as hf:
        cop         = np.array(hf[num+tag+'/Galaxy'].get('COP'), dtype = np.float64)
        cop_vel     = np.array(hf[num+tag+'/Galaxy'].get('Velocity'), dtype = np.float64)
        S_len       = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)

        mstar   = np.array(hf[num+tag+'/Galaxy/Mstar_aperture'].get('Mstar_30'), dtype = np.float32)*1e10
        SFR10   = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('SFR_30/SFR_10_Myr'), dtype = np.float32)
        SFR100  = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('SFR_30/SFR_100_Myr'), dtype = np.float32)
        wAge    = np.array(hf[num+tag+'/Galaxy/StellarAges'].get('MassWeightedStellarAge'), dtype = np.float32)

        S_ap    = np.array(hf[num+tag+'/Particle/Apertures'].get('Star'), dtype = np.bool)
        S_mass  = np.array(hf[num+tag+'/Particle'].get('S_Mass'), dtype = np.float64) * 1e10
        S_coord = np.array(hf[num+tag+'/Particle'].get('S_Coordinates'), dtype = np.float64)
        S_vel   = np.array(hf[num+tag+'/Particle'].get('S_Vel'), dtype = np.float64)


    begin = np.zeros(len(S_len), dtype = np.int32)
    end = np.zeros(len(S_len), dtype = np.int32)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    z = float(tag[5:].replace('p','.'))
    cop/=(1+z)
    S_coord/=(1+z)

    S_len30, begin30, end30, S_ap30 = get_slen30(S_len, S_ap, limit)
    del S_len
    ok = np.where(S_len30>limit)[0]

    cop, cop_vel, mstar, SFR10, SFR100, S_len30, wAge = cop[:,ok], cop_vel[:,ok], mstar[ok], SFR10[ok], SFR100[ok], S_len30[ok], wAge[ok]


    with h5py.File(sim_orient, 'r') as hf:
        LFUVint_part = np.array(hf[tag+'/Particle/BPASS_2.2.1/Chabrier300/zaxis/Luminosity/Intrinsic'].get('FUV'), dtype = np.float32)
        LFUVatt_part = np.array(hf[tag+'/Particle/BPASS_2.2.1/Chabrier300/zaxis/Luminosity/DustModelI'].get('FUV'), dtype = np.float32)

    LFUVint = np.zeros((6,len(begin30)))
    LFUVatt = np.zeros((6,len(begin30)))

    att_mean    = np.zeros((6,len(begin30)))
    att_med     = np.zeros((6,len(begin30)))
    att_16perc  = np.zeros((6,len(begin30)))
    att_84perc  = np.zeros((6,len(begin30)))
    kappa_star  = np.zeros((6,len(begin30)))

    if len(ok)>0:
        for ii, jj in enumerate(begin30):

            for kk in range(6):
                int = LFUVint_part[begin30[ii]:end30[ii]][S_ap30[kk,begin30[ii]:end30[ii]]]
                att = LFUVatt_part[begin30[ii]:end30[ii]][S_ap30[kk,begin30[ii]:end30[ii]]]

                LFUVint[kk,ii] = np.sum(int)
                LFUVatt[kk,ii] = np.sum(att)

                x           = att/int
                quantiles   = [0.84,0.50,0.16]
                out         = flares.binned_weighted_quantile(np.zeros_like(x), x, int, [-0.5,0.5], quantiles)

                att_mean[kk,ii]     = 1./(np.nansum((1/x)*int)/np.nansum(int))
                att_med[kk,ii]      = out[1]
                att_16perc[kk,ii]   = out[2]
                att_84perc[kk,ii]   = out[0]

                this_smass   = S_mass[begin[ok][ii]:end[ok][ii]][S_ap[kk,begin[ok][ii]:end[ok][ii]]]
                this_scoord = (S_coord[:, begin[ok][ii]:end[ok][ii]]).T[S_ap[kk,begin[ok][ii]:end[ok][ii]]] - cop[:,ii]
                this_svel   = (S_vel[:, begin[ok][ii]:end[ok][ii]]).T[S_ap[kk,begin[ok][ii]:end[ok][ii]]] - cop_vel[:,ii]

                kappa_star[kk,ii] = compute_kappa(this_smass, this_scoord, this_svel)[0]



    return mstar, SFR10, SFR100, LFUVint, LFUVatt, LFUVint_part, LFUVatt_part, att_mean, att_med, att_16perc, att_84perc, kappa_star, wAge


def plot_fig(x, y, z, ws, axs, bins, v, extent):

    hb = axs.hexbin(x, y, C = z, gridsize=(43,25), cmap=plt.cm.get_cmap('nipy_spectral'), reduce_C_function=np.median, linewidths=0., mincnt=0., extent=extent, vmin=v[0], vmax=v[1])

    bincen = (bins[1:]+bins[:-1])/2.
    out = flares.binned_weighted_quantile(x, y, ws, bins, quantiles)
    hist, binedges = np.histogram(x, bins)
    ok = np.where(hist>0)[0]
    ok1 = np.where(hist[ok]>3)[0][0]

    # axs.fill_between(bincen[ok][ok1:], out[:,2][ok][ok1:], out[:,0][ok][ok1:], color='black', alpha=0.25)
    # axs.plot(bincen[ok], out[:,1][ok], ls='dashed', color='grey', alpha=.5, lw=1)
    # axs.plot(bincen[ok][ok1:], out[:,1][ok][ok1:], ls='-', color='black', alpha=.5, lw=1)

    return hb


if __name__ == "__main__":

    inp = int(sys.argv[1])

    quantiles = [0.84,0.50,0.16]
    limit=300
    df = pd.read_csv('../data/weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(13, 8), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()
    clabel = r'log$_{10}$(Stellar Age/Gyr)'#r'log$_{10}$(SFR$_{100}$/M$_{\odot}$yr$^{-1}$)'#r'A$_{\mathrm{FUV}}$'

    if inp == 0:
        xlabel = r'log$_{10}$(sSFR/Gyr$^{-1}$)'
        lx, ly = -2, -0.5
        title = 'sSFR'
    elif inp==1:
        xlabel = r'log$_{10}$(M$_{\star}$/M$^{\odot}$)'
        lx, ly = 10, -0.5
        title = 'mstar'
    elif inp==2:
        xlabel = r'M$_{1500}$'
        lx, ly = -17.5, -0.75
        title = 'mean_FUV_Age'
    elif inp==3:
        xlabel = r'$\kappa_{\star}$'
        lx, ly = 0.5, -0.5
        title = 'kappa'


    for ii, tag in enumerate(tags):

        z = float(tag[5:].replace('p','.'))

        func    = partial(get_data, tag=tag, limit=limit)
        pool    = schwimmbad.MultiPool(processes=8)
        dat     = np.array(list(pool.map(func, np.arange(0,40))))
        pool.close()

        mstar   = np.concatenate(dat[:,0])
        SFR10   = np.concatenate(dat[:,1])
        SFR100  = np.concatenate(dat[:,2])
        wAge    = np.concatenate(dat[:,-1])

        LFUVint     = np.zeros((6, len(mstar)))
        LFUVatt     = np.zeros((6, len(mstar)))
        att_mean    = np.zeros((6, len(mstar)))
        att_med     = np.zeros((6, len(mstar)))
        att_16perc  = np.zeros((6, len(mstar)))
        att_84perc  = np.zeros((6, len(mstar)))
        kappa_star  = np.zeros((6, len(mstar)))

        ws = np.zeros(len(mstar))

        inicount, fincount = 0, 0

        for jj in range(40):
            fincount+=len(dat[jj][0])

            ws[inicount:fincount] = np.ones(len(dat[jj][0]))*weights[jj]

            for kk in range(6):
                LFUVint[kk][inicount:fincount] = dat[:,3][jj][kk]
                LFUVatt[kk][inicount:fincount] = dat[:,4][jj][kk]

                att_mean[kk][inicount:fincount]     = dat[:,7][jj][kk]
                att_med[kk][inicount:fincount]      = dat[:,8][jj][kk]
                att_16perc[kk][inicount:fincount]   = dat[:,9][jj][kk]
                att_84perc[kk][inicount:fincount]   = dat[:,10][jj][kk]
                kappa_star[kk][inicount:fincount]   = dat[:,11][jj][kk]

            inicount=fincount

        LFUVint_part = np.concatenate(dat[:,5])
        LFUVatt_part = np.concatenate(dat[:,6])

        del dat

        x = LFUVatt[5]/LFUVint[5]
        y = (x - att_mean[5])/x
        c = np.log10(wAge)#np.log10(SFR100)#-2.5*np.log10(x)


        if inp==0:
            x = np.log10(SFR10/mstar * 1e9)
        elif inp==1:
            x = np.log10(mstar)
        elif inp==2:
            x = lum_to_M(LFUVatt[5])
        elif inp==3:
            x = kappa_star[5]

        bins = np.arange(np.min(x[np.isfinite(x)]), np.max(x[np.isfinite(x)])+0.1, 0.25)

        if ii == 0:
            v = [np.min(c[np.isfinite(c)])-0.1, np.max(c[np.isfinite(c)])+0.1]

            extent = [*[np.min(x[np.isfinite(x)])-0.1,np.max(x[np.isfinite(x)])+0.1], *[np.min(y[np.isfinite(y)])-0.1,np.max(y[np.isfinite(y)])+0.1]]

        hb = plot_fig(x, y, c, ws, axs[ii], bins, v, extent)


        axs[ii].text(lx, ly, r'$z = {}$'.format(z), fontsize = 12)



    for ax in axs:
        ax.grid(True, linestyle=(0, (0.5, 3)))
        ax.legend(fontsize=8, frameon=False, loc=2, numpoints=1, markerscale=1)
        ax.tick_params(axis="y",direction="in")
        ax.tick_params(axis="x",direction="in")

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)

    #sSFR/Gyr$^{-1}$ log$_{10}$(sSFR/Gyr$^{-1}$)
    axs[-2].set_xlabel(xlabel, fontsize = 13)
    if inp==2: axs[0].invert_xaxis()


    # axs.legend(fontsize=7, frameon=False)
    fig.text(0.05, 0.5, r"1-f$_{\mathrm{FUV,mean}}$/f$_{\mathrm{FUV,tot}}$", va='center', rotation='vertical', fontsize=15)

    fig.subplots_adjust(right = 0.91, wspace=0, hspace=0)
    cbaxes = fig.add_axes([0.92, 0.25, 0.005, 0.5])
    fig.colorbar(hb, cax=cbaxes)
    cbaxes.set_ylabel(clabel, fontsize = 15)
    for label in cbaxes.get_yticklabels():
        label.set_fontsize(10)

    plt.savefig(F"../results/att_spread_{title}_z5_10.pdf", bbox_inches='tight', dpi=300)
    plt.show()
