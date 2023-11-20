"""
Figure 12 in paper: Use inp=int(sys.argv[1])=1
Percentile difference with different LOS for
UV attenuation, beta, balmer decrement
Figure 13 in paper: Use inp=int(sys.argv[1])=4
"""

import numpy as np
import pandas as pd
from functools import partial
import h5py, schwimmbad, matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import cmasher as cmr

import sys
sys.path.append('../src')

from helpers import get_slen30
import flares
from mollview import get_kappa
from calc_rhalf import calc_rhalf

def get_data(ii, tag, limit=500, intr_ratio=2.79, dl=1e30):

    num = str(ii)

    if len(num) == 1:
        num =  '0'+num

    sim = rF"../../flares_pipeline/data/flares.hdf5"

    with h5py.File(F'../data/FLARES_{num}_data.hdf5', 'r') as hf:
        beta = np.array(hf[tag+'/Galaxy/BPASS_2.2.1/Chabrier300/los/Indices/DustModelI'].get('beta'), dtype=np.float64)
        LFUV = np.array(hf[tag+'/Galaxy/BPASS_2.2.1/Chabrier300/los/Luminosity/DustModelI'].get('FUV'), dtype=np.float64)
        HI6563_lum = np.array(hf[tag+'/Galaxy/BPASS_2.2.1/Chabrier300/los/Lines/DustModelI/HI6563'].get('Luminosity'), dtype=np.float64)
        HI4861_lum = np.array(hf[tag+'/Galaxy/BPASS_2.2.1/Chabrier300/los/Lines/DustModelI/HI4861'].get('Luminosity'), dtype=np.float64)

    num = num+'/'
    with h5py.File(sim, 'r') as hf:
        S_len   = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)
        S_ap    = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype = bool)

        Mstar   = np.array(hf[num+tag+'/Galaxy/Mstar_aperture'].get('30'), dtype = np.float32)*1e10
        SFR10   = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('30/10Myr'), dtype = np.float32)
        SFR100  = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('30/100Myr'), dtype = np.float32)

        LFUVint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float32)



    S_len, begin, end, S_ap = get_slen30(S_len, S_ap, limit)
    ok = np.where(S_len>limit)[0]

    if len(ok)!=0:

        SFR10, SFR100, Mstar, LFUVint = SFR10[ok], SFR100[ok], Mstar[ok], LFUVint[ok]

        AFUV = -2.5 * np.log10(LFUV/np.vstack(LFUVint))

        balmer_decrement = 2.5 * np.log10((HI6563_lum/HI4861_lum)/intr_ratio)

    else:
        Mstar, SFR10, SFR100, AFUV, beta, balmer_decrement = np.array([]), np.array([]), np.array([]), np.zeros((0,192)), np.zeros((0,192)), np.zeros((0,192))

    return Mstar, SFR10, SFR100, AFUV, beta, balmer_decrement


def get_Sigmadust(ii, tag, limit=500):


    num = str(ii)
    redshift = float(tag[5:].replace('p','.'))

    if len(num) == 1:
        num =  '0'+num

    sim = rF"../../flares_pipeline/data/flares.hdf5"
    num = num+'/'

    with h5py.File(sim, 'r') as hf:
        S_len       = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)
        G_len       = np.array(hf[num+tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        S_ap        = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype=bool)
        G_ap        = np.array(hf[num+tag+'/Particle/Apertures/Gas'].get('30'), dtype=bool)

        cop = np.array(hf[num+tag+'/Galaxy'].get('COP'), dtype = np.float64)
        DTM = np.array(hf[num+tag+'/Galaxy'].get('DTM'), dtype = np.float64)

        S_mass = np.array(hf[num+tag+'/Particle'].get('S_Mass'), dtype = np.float64)[S_ap]*1e10
        S_coords = np.array(hf[num+tag+'/Particle'].get('S_Coordinates'), dtype = np.float64).T[S_ap]

        G_mass = np.array(hf[num+tag+'/Particle'].get('G_Mass'), dtype = np.float64)[G_ap]*1e10
        G_coords = np.array(hf[num+tag+'/Particle'].get('G_Coordinates'), dtype = np.float64).T[G_ap]
        G_Z      = np.array(hf[num+tag+'/Particle'].get('G_Z_smooth'), dtype = np.float64)

    n = len(S_len)
    begin = np.zeros(n, dtype=np.int32)
    end = np.zeros(n, dtype=np.int32)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)
    begin = begin.astype(np.int32)
    end = end.astype(np.int32)
    S_len30 = np.zeros(n, dtype=np.int32)

    gbegin = np.zeros(n, dtype=np.int32)
    gend = np.zeros(n, dtype=np.int32)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)
    gbegin = gbegin.astype(np.int32)
    gend = gend.astype(np.int32)
    G_len30 = np.zeros(n, dtype=np.int32)

    for kk in range(n):
        S_len30[kk] = np.sum(S_ap[begin[kk]:end[kk]])
        G_len30[kk] = np.sum(G_ap[gbegin[kk]:gend[kk]])

    ok = np.where(S_len30>limit)[0]

    begin[1:]  = np.cumsum(S_len30)[:-1]
    end        = np.cumsum(S_len30)
    begin      = begin.astype(np.int32)
    end        = end.astype(np.int32)


    sigma_dust  = np.zeros(len(ok))
    gbegin[1:]  = np.cumsum(G_len30)[:-1]
    gend        = np.cumsum(G_len30)
    gbegin      = gbegin.astype(np.int32)
    gend        = gend.astype(np.int32)



    for jj, kk in enumerate(ok):

        smass    = S_mass[begin[kk]:end[kk]]
        scoords  = 1e3*(S_coords[begin[kk]:end[kk]] - cop[:,kk])/(1+redshift)

        hmsr = calc_rhalf(scoords, smass)

        mass    = G_mass[gbegin[kk]:gend[kk]]
        gZ      = G_Z[gbegin[kk]:gend[kk]]
        coords  = 1e3*(G_coords[gbegin[kk]:gend[kk]] - cop[:,kk])/(1+redshift)

        dist_g = np.linalg.norm(coords, axis=1)

        this_ok = np.where(dist_g<=2*hmsr)[0]
        if hmsr>30: print (hmsr)
        sigma_dust[jj] = DTM[kk] * np.sum(mass[this_ok]*gZ[this_ok])/(2 * np.pi * (2*hmsr)**2)


    return sigma_dust


def plot_bins(ax, x, z, bins, s_m, xlim, ylim, y=False, binsize=0.5, gridsize=(20,15), plot_percentile=True, inp=1, tag='010_z005p000'):

    vmax=len(z)

    spread_y = np.nanpercentile(x, 84, axis=1) - np.nanpercentile(x, 16, axis=1)
    print ("tag=", tag)
    print ("median spread", np.nanmedian(spread_y))
    tmpxbin = np.arange(np.nanmin(x)-binsize, np.nanmax(x)+binsize, binsize)
    print (tmpxbin)
    if len(tmpxbin)>2:
        med_y, a, b = stats.binned_statistic(np.nanmedian(x, axis=1), spread_y, statistic="median", bins=tmpxbin)
        std_y, a, b = stats.binned_statistic(np.nanmedian(x, axis=1), spread_y, statistic="std", bins=tmpxbin)

        print (tmpxbin, med_y, std_y)
        print ("max", np.nanmax(std_y))

    print("---------------------------------\n")

    extent = [*xlim, *ylim]

    if inp in [0,1]:
        median_x = np.nanmedian(x, axis=1)
        hb = ax.hexbin(median_x, spread_y, bins='log', cmap=matplotlib.cm.Greys_r, vmin=1, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=1, alpha=0.4)
        for kk in range(len(bins)-1):
            # print (bins)

            bincen = (bins[kk]+bins[kk+1])/2

            ok = np.logical_and(z>=bins[kk], z<bins[kk+1])
            if inp in [0,1]:
                xx, yy = np.median(x[ok], axis=1), np.nanpercentile(x[ok], 84, axis=1) - np.nanpercentile(x[ok], 16, axis=1)
            else:
                xx, yy = np.median(x[ok], axis=1), np.nanpercentile(y[ok], 84, axis=1) - np.nanpercentile(y[ok], 16, axis=1)

            if np.sum(ok)>=5:
                this_bins = np.arange(np.nanmin(xx)-binsize, np.nanmax(xx)+binsize, binsize)

                this_xx, this_yy, this_yy84, this_yy16 = np.array([]), np.array([]), np.array([]), np.array([])
                # print (this_bins)

                for jj in range(len(this_bins)-1):

                    this_ok = np.logical_and(xx>=this_bins[jj], xx<this_bins[jj+1])
                    if np.sum(this_ok)>=5:

                        this_xx = np.append(this_xx, (this_bins[jj]+this_bins[jj+1])/2.)
                        this_yy = np.append(this_yy, np.nanmedian(yy[this_ok]))
                        if plot_percentile:
                            this_yy84 = np.append(this_yy84, np.nanpercentile(yy[this_ok], 84))
                            this_yy16 = np.append(this_yy16, np.nanpercentile(yy[this_ok], 16))

                ax.plot(this_xx, this_yy, color=s_m.to_rgba(bincen))
                # print (bincen,this_xx,this_yy)

                if plot_percentile:
                    ax.fill_between(this_xx, this_yy16, this_yy84, color=s_m.to_rgba(bincen), alpha=0.2)

    else:
        hb = ax.hexbin(x, spread_y, C=z, cmap=matplotlib.cm.coolwarm, vmin=vmin, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=1)

    return ax


if __name__ == "__main__":

    inp = int(sys.argv[1])

    quantiles = [0.84,0.50,0.16]
    limit=500
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    Halpha_lam = 6563.
    Hbeta_lam = 4861.
    intr_ratio = '2.79'

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    if inp in [0,1]:
        fig, axs = plt.subplots(nrows = 3, ncols = 6, figsize=(10, 7), sharex=False, sharey=False, facecolor='w', edgecolor='k')
        # axs = axs.ravel()
    elif inp in [2]:
        fig, axs = plt.subplots(nrows = 1, ncols = 6, figsize=(10, 3), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    elif inp in [3]:
        fig, axs = plt.subplots(nrows = 2, ncols = 6, figsize=(12, 5), sharex=True, sharey=False, facecolor='w', edgecolor='k')
    else:
        fig, axs = plt.subplots(nrows = 2, ncols = 6, figsize=(12, 5), sharex=True, sharey=False, facecolor='w', edgecolor='k')

    mbins = np.linspace(8.8, 11.5, 6, endpoint=True)
    sbins = np.arange(-1, 3.1, 0.5)
    fuvbins = np.linspace(0, 4.1, 6, endpoint=True)

    if inp==0:
        bins = mbins
        clabel = r'log$_{10}$(M$_{\star}$/M$_{\odot}$)'
        title = 'mstar'
    elif inp==1:
        bins = sbins
        clabel = r'log$_{10}$(SFR/(M$_{\odot}$/yr))'
        title = 'SFR'
    elif inp==2:
        bins = fuvbins
        clabel = 'A$_{FUV}$'
        title = 'beta_vs_afuv'
    elif inp==3:
        bins = np.linspace(0,0.65,6,endpoint=True)
        clabel = r'$\kappa_{co}$'
        title = 'afuv_beta_Kco'
    elif inp==4:
        bins = np.linspace(0,0.65,6,endpoint=True)
        clabel = r'$\kappa_{\mathrm{co},\star}$'
        title = 'Sigma_Kco'
    else:
        print ("Not a valid argument")
        sys.exit()

    xlim_afuv = [0,4.5]
    ylim_afuv = [0,0.8]

    xlim_beta = [-2.6,-0.5]
    ylim_beta = [0,0.5]

    xlim_balmer = [0,1.9]
    ylim_balmer = [0,0.5]


    bincen = (bins[1:]+bins[:-1])/2
    labels = [f'${np.round(bins[ii],1)}-{np.round(bins[ii+1],1)}$' for ii in range(len(bins)-1)]
    # labels[-1] = f'>{np.round(bins[-1],1)}'

    # choose a colormap
    c_m = plt.get_cmap('cmr.cosmic')  #matplotlib.cm.plasma
    norm = matplotlib.colors.BoundaryNorm(bins, c_m.N)
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    for ii, tag in enumerate(tags[::-1]):

        redshift = float(tag[5:].replace('p','.'))

        # axs[ii].text(0.5, 0.3, r'$z = {}$'.format(z), fontsize = 12)

        func    = partial(get_data, tag=tag, limit=limit)
        pool    = schwimmbad.MultiPool(processes=8)
        dat     = np.array(list(pool.map(func, np.arange(0,40))), dtype=object)
        pool.close()


        Mstar       = np.concatenate(dat[:,0], dtype=np.float64)
        # print (tag, np.min(Mstar))
        SFR10       = np.concatenate(dat[:,1], dtype=np.float64)
        SFR100      = np.concatenate(dat[:,2], dtype=np.float64)

        ws = np.zeros(len(Mstar))
        n = 0
        for jj in range(40):
            if jj==0:
                ws[0:len(dat[jj][0])] = weights[jj]
            else:
                ws[n:n+len(dat[jj][0])] = weights[jj]

            n+=len(dat[jj][0])


        for jj, kk in enumerate(['AFUV', 'beta', 'balmer']):

            tmp = dat[:,jj+3]
            tmp = [d for d in tmp if d.shape[0] != 0]
            vars()[kk] = np.asarray(np.vstack(tmp), dtype=np.float64)

        if inp==0:
            z = np.log10(Mstar)
        elif inp==1:
            z = np.log10(SFR100)
        elif inp==2:
            z = np.nanmedian(AFUV, axis=1)
        elif inp==3:
            func    = partial(get_kappa, tag=tag, z=redshift, limit=limit)
            pool    = schwimmbad.MultiPool(processes=8)
            dat     = np.array(list(pool.map(func, np.arange(0,40))), dtype=object)
            pool.close()
            z = np.concatenate(dat[:,0], dtype=np.float64)
        elif inp==4:
            func    = partial(get_kappa, tag=tag, z=redshift, limit=limit)
            pool    = schwimmbad.MultiPool(processes=8)
            dat     = np.array(list(pool.map(func, np.arange(0,40))), dtype=object)
            z = np.concatenate(dat[:,0], dtype=np.float64)

            func    = partial(get_Sigmadust, tag=tag, limit=limit)
            pool    = schwimmbad.MultiPool(processes=8)
            dat     = np.array(list(pool.map(func, np.arange(0,40))), dtype=object)

            sigma_dust = np.concatenate(dat, dtype=np.float64)

            pool.close()

        if inp in [0,1]:

            plot_bins(axs[0,ii], AFUV, z, bins, s_m, xlim_afuv, ylim_afuv, inp=inp, tag=tag)
            plot_bins(axs[1,ii], beta, z, bins, s_m, xlim_beta, ylim_beta, binsize=0.25, inp=inp, tag=tag)
            plot_bins(axs[2,ii], balmer, z, bins, s_m, xlim_balmer, ylim_balmer, binsize=0.25, inp=inp, tag=tag)

        elif inp==2:

            vmin=0.
            vmax=4.

            median_x = np.nanmedian(beta, axis=1)
            spread_y = np.nanpercentile(AFUV, 84, axis=1) - np.nanpercentile(AFUV, 16, axis=1)

            extent = [*xlim_beta, *ylim_afuv]
            gridsize=(20,15)

            hb = axs[ii].hexbin(median_x, spread_y, C=z, cmap=matplotlib.cm.coolwarm, vmin=vmin, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=1)
            axs[ii].set_title(F'$z={redshift}$', fontsize=12)

        elif inp==3:

            vmin=0.
            vmax=0.65

            median_x0 = np.nanmedian(AFUV, axis=1)
            spread_y0 = np.nanpercentile(AFUV, 84, axis=1) - np.nanpercentile(AFUV, 16, axis=1)

            median_x1 = np.nanmedian(beta, axis=1)
            spread_y1 = np.nanpercentile(beta, 84, axis=1) - np.nanpercentile(beta, 16, axis=1)
            print ('-----------------------------------------')
            print (tag)
            extent = [*xlim_afuv, *ylim_afuv]
            gridsize=(20,15)
            hb = axs[0,ii].hexbin(median_x0, spread_y0, C=z, cmap=matplotlib.cm.coolwarm, vmin=vmin, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=1)
            axs[0,ii].set_title(F'$z={redshift}$', fontsize=12)
            print ('AFUV spearman correlation:', stats.spearmanr(median_x0, spread_y0))
            print ('AFUV spearman correlation, kappa:', stats.spearmanr(z, spread_y0))

            extent = [*xlim_beta, *ylim_beta]
            gridsize=(20,15)
            hb = axs[1,ii].hexbin(median_x1, spread_y1, C=z, cmap=matplotlib.cm.coolwarm, vmin=vmin, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=1)
            print ('beta spearman correlation:', stats.spearmanr(median_x1, spread_y1))
            print ('beta spearman correlation, kappa:', stats.spearmanr(z, spread_y1))

        elif inp==4:

            vmin=0.
            vmax=0.65

            x = np.log10(sigma_dust)
            bins = np.arange(np.min(x[np.isfinite(x)]), np.max(x[np.isfinite(x)])+0.5, 0.5)
            bincen = (bins[1:]+bins[:-1])/2.
            binwidth = bins[1:] - bins[:-1]
            hist, edges = np.histogram(x, bins)
            ok = np.where(hist>=5)[0]

            median_x0 = np.nanmedian(AFUV, axis=1)
            spread_y0 = np.nanpercentile(AFUV, 84, axis=1) - np.nanpercentile(AFUV, 16, axis=1)
            out0 = flares.binned_weighted_quantile(x, spread_y0, ws, bins, quantiles)

            median_x1 = np.nanmedian(beta, axis=1)
            spread_y1 = np.nanpercentile(beta, 84, axis=1) - np.nanpercentile(beta, 16, axis=1)
            out1 = flares.binned_weighted_quantile(x, spread_y1, ws, bins, quantiles)

            print ('-----------------------------------------')
            print (tag)
            extent = [*[4,7.6], *ylim_afuv]
            gridsize=(20,15)
            hb = axs[0,ii].hexbin(x, spread_y0, C=z, cmap=plt.get_cmap('cmr.heat_r') , vmin=vmin, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=1)
            axs[0,ii].plot(bincen[ok], out0[:,1][ok], color='black', alpha=0.5)
            axs[0,ii].fill_between(bincen[ok], out0[:,2][ok], out0[:,0][ok], color='black', alpha=0.25)
            axs[0,ii].set_title(F'$z={redshift}$', fontsize=12)
            print ('AFUV spearman correlation:', stats.spearmanr(np.log10(sigma_dust), spread_y0))
            print ('AFUV spearman correlation, kappa:', stats.spearmanr(z, spread_y0))

            extent = [*[4,7.6], *ylim_beta]
            gridsize=(20,15)
            hb = axs[1,ii].hexbin(x, spread_y1, C=z, cmap=plt.get_cmap('cmr.heat_r') , vmin=vmin, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=1)
            axs[1,ii].plot(bincen[ok], out1[:,1][ok], color='black', alpha=0.5)
            axs[1,ii].fill_between(bincen[ok], out1[:,2][ok], out1[:,0][ok], color='black', alpha=0.25)
            print ('beta spearman correlation:', stats.spearmanr(np.log10(sigma_dust), spread_y1))
            print ('beta spearman correlation, kappa:', stats.spearmanr(z, spread_y1))



    if inp in [0,1]:
        for ii in range(6):
            axs[0,ii].set_xlim(xlim_afuv)
            axs[0,ii].set_ylim(ylim_afuv)

            axs[1,ii].set_xlim(xlim_beta)
            axs[1,ii].set_ylim(ylim_beta)

            axs[2,ii].set_xlim(xlim_balmer)
            axs[2,ii].set_ylim(ylim_balmer)

            if ii!=0:
                axs[0,ii].set_yticklabels([])
                axs[1,ii].set_yticklabels([])
                axs[2,ii].set_yticklabels([])


        fig.subplots_adjust(wspace=0, hspace=0.35, right=0.95)
        cbaxes = fig.add_axes([0.98, 0.35, 0.007, 0.3])
        fig.colorbar(s_m, cax=cbaxes)

        for ii, tag in enumerate(tags[::-1]):
            redshift = float(tag[5:].replace('p','.'))
            axs[0,ii].set_title(F'$z={redshift}$', fontsize=12)

        fig.text(0.05, 0.5, r'$84^{th}-16^{th}$', va='center', rotation='vertical', fontsize=12)

        fig.text(0.48, 0.62, r'Median(A$_{FUV}$)', va='center', fontsize=12)
        fig.text(0.49, 0.34, r'Median($\beta$)', va='center', fontsize=12)
        fig.text(0.42, 0.04, r'Median(2.5$\times$log$_{10}(\frac{\mathrm{H}_{\alpha}/\mathrm{H}_{\beta}}{2.79})$)', va='center', fontsize=12)

    elif inp==3:
        fig.subplots_adjust(wspace=0, hspace=0.3, right=0.95, bottom=0.1)
        axs[0,0].set_ylabel(r'A$_{FUV,84}$ - A$_{FUV,16}$', fontsize=12)
        axs[1,0].set_ylabel(r'$\beta_{84}$ - $\beta_{16}$', fontsize=12)
        fig.text(0.48, 0.45, r'Median(A$_{FUV}$)', va='center', fontsize=12)
        fig.text(0.49, 0.04, r'Median($\beta$)', va='center', fontsize=12)

        cbaxes = fig.add_axes([0.98, 0.35, 0.007, 0.3])
        fig.colorbar(hb, cax=cbaxes)

        for ii in range(6):
            if ii!=0:
                axs[0,ii].set_yticklabels([])
                axs[1,ii].set_yticklabels([])

    elif inp==4:
        for ii in range(6):
            axs[0,ii].set_xlim((4,7.8))
            axs[1,ii].set_xlim((4,7.8))

        fig.subplots_adjust(wspace=0, hspace=0, right=0.95, bottom=0.1)
        axs[0,0].set_ylabel(r'A$_{FUV,84}$ - A$_{FUV,16}$', fontsize=12)
        axs[1,0].set_ylabel(r'$\beta_{84}$ - $\beta_{16}$', fontsize=12)
        fig.text(0.45, 0.01, r"log$_{10}$($\Sigma_{\mathrm{dust}}$($\leq 2$r$_{1/2}$)/M$_{\odot}$kpc$^{-2}$)", va='center', fontsize=12)

        cbaxes = fig.add_axes([0.98, 0.35, 0.007, 0.3])
        fig.colorbar(hb, cax=cbaxes)

        for ii in range(6):
            if ii!=0:
                axs[0,ii].set_yticklabels([])
                axs[1,ii].set_yticklabels([])


    else:
        fig.subplots_adjust(wspace=0, right=0.95, bottom=0.15)
        axs[0].set_ylabel(r'A$_{FUV,84}$ - A$_{FUV,16}$', fontsize=12)
        fig.text(0.49, 0.01, r'Median($\beta$)', va='center', fontsize=12)

        cbaxes = fig.add_axes([0.98, 0.35, 0.007, 0.3])
        fig.colorbar(hb, cax=cbaxes)


    for ax in axs.ravel():
        ax.grid(ls='dotted')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(13)


    cbaxes.set_ylabel(clabel, fontsize=11)
    cbaxes.yaxis.set_label_position("left")

    plt.savefig(F'../results/los_spread_{title}_kappa.pdf', bbox_inches='tight', dpi=300)

    plt.show()
