"""
Figure 2 in paper: Variation in the UV star formation 
surface rate density and the UV-attenuation with 
distance from the galaxy centre
"""

import numpy as np
import pandas as pd
from functools import partial
import h5py, schwimmbad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from FLARE.photom import lum_to_M

import sys
sys.path.append('../src')

norm = np.linalg.norm

def get_data(ii, tag, limit=500, bins = np.arange(0,5.25,0.25)):

    num = str(ii)

    if len(num) == 1:
        num =  '0'+num

    sim_orient = rF"../data/FLARES_{num}_data.hdf5"
    sim_orient1 = rF"../data1/FLARES_{num}_data.hdf5"
    sim = rF"../../flares_pipeline/data/flares.hdf5"
    num = num+'/'

    with h5py.File(sim, 'r') as hf:
        S_len       = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)

        mstar   = np.array(hf[num+tag+'/Galaxy/Mstar_aperture/'].get('30'), dtype = np.float32)*1e10
        SFR10   = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('30/10Myr'), dtype = np.float32)
        SFR100  = np.array(hf[num+tag+'/Galaxy/SFR_aperture'].get('30/100Myr'), dtype = np.float32)
        LFUVint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float32)
        LFUVatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float32)

        cop     = np.array(hf[num+'/'+tag+'/Galaxy'].get('COP'), dtype=np.float64).T/(1.+z)

        S_ap        = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype = bool)
        S_coord     = np.array(hf[num+'/'+tag+'/Particle'].get('S_Coordinates'), dtype=np.float64).T/(1.+z)
        S_mass      = np.array(hf[num+'/'+tag+'/Particle'].get('S_MassInitial'), dtype=np.float64)*1e10
        S_age       = np.array(hf[num+'/'+tag+'/Particle'].get('S_Age'), dtype=np.float64)*1e3


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

    ok = np.where(S_len30>limit)[0]
    mstar, SFR10, SFR100, LFUVatt, Afuv, cop = mstar[ok], SFR10[ok], SFR100[ok],  LFUVatt[ok], -2.5*np.log10(LFUVatt[ok]/LFUVint[ok]), cop[ok]

    this_scoord     = np.zeros((n,3), dtype=np.float32)
    this_smass      = np.zeros(n, dtype=np.float32)

    with h5py.File(sim_orient, 'r') as hf:
        LFUVint_part = np.array(hf[tag+'/Particle/BPASS_2.2.1/Chabrier300/zaxis/Luminosity/Intrinsic'].get('FUV'), dtype = np.float32)
        LFUVatt_part = np.array(hf[tag+'/Particle/BPASS_2.2.1/Chabrier300/zaxis/Luminosity/DustModelI'].get('FUV'), dtype = np.float32)

    with h5py.File(sim_orient1, 'r') as hf:

        stellar_rhalf   = np.array(hf[tag+'/Galaxy/HalfMassRadii'].get('Star'), dtype = np.float64)
        dust_rhalf      = np.array(hf[tag+'/Galaxy/HalfMassRadii'].get('Dust'), dtype = np.float64)


    sfr10_sbins      = np.zeros((len(ok), len(bins)-1))
    sfr100_sbins     = np.zeros((len(ok), len(bins)-1))
    Afuv_sbins       = np.zeros((len(ok), len(bins)-1))

    sfr10_dbins      = np.zeros((len(ok), len(bins)-1))
    sfr100_dbins     = np.zeros((len(ok), len(bins)-1))
    Afuv_dbins       = np.zeros((len(ok), len(bins)-1))

    inicount = 0
    fincount = 0

    for jj, kk in enumerate(ok):
        fincount+=S_len30[kk]

        this_lfuvint    = LFUVint_part[inicount:fincount]
        this_lfuvatt    = LFUVatt_part[inicount:fincount]

        this_scoord     = 1e3*(S_coord[begin[kk]:end[kk]][S_ap[begin[kk]:end[kk]]] - cop[jj])
        this_smass      = S_mass[begin[kk]:end[kk]][S_ap[begin[kk]:end[kk]]]
        this_sage       = S_age[begin[kk]:end[kk]][S_ap[begin[kk]:end[kk]]]

        dist = norm(this_scoord, axis=1)
        sdist = dist/stellar_rhalf[jj]
        ddist = dist/dust_rhalf[jj]


        for ii in range(len(bins)-1):

            sok = np.logical_and(sdist>=bins[ii], sdist<bins[ii+1])
            dok = np.logical_and(ddist>=bins[ii], ddist<bins[ii+1])

            Afuv_sbins[jj,ii] = np.sum(this_lfuvatt[sok])/np.sum(this_lfuvint[sok])
            Afuv_dbins[jj,ii] = np.sum(this_lfuvatt[dok])/np.sum(this_lfuvint[dok])


            sfr10_sbins[jj,ii]  = np.sum(this_smass[sok][this_sage[sok]<10.])/(10*1e6)
            sfr100_sbins[jj,ii] = np.sum(this_smass[sok][this_sage[sok]<100.])/(100*1e6)

            sfr10_dbins[jj,ii]  = np.sum(this_smass[dok][this_sage[dok]<10.])/(10*1e6)
            sfr100_dbins[jj,ii] = np.sum(this_smass[dok][this_sage[dok]<100.])/(100*1e6)

        inicount=fincount

    return mstar, SFR10, SFR100, LFUVatt, Afuv, sfr10_sbins, sfr100_sbins, Afuv_sbins, sfr10_dbins, sfr100_dbins, Afuv_dbins, stellar_rhalf, dust_rhalf


if __name__ == "__main__":

    quantiles = [0.84,0.50,0.16]
    limit=500
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']


    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(12, 7), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()
    twinaxs = np.ones_like(axs)

    rbins = np.arange(0,5.5,0.5)
    rbinscen = (rbins[1:]+rbins[:-1])/2.
    lbins = np.arange(-24.,-17.5,1.)

    c_m = matplotlib.cm.plasma
    NORM = matplotlib.colors.BoundaryNorm(lbins, c_m.N)
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=NORM)
    s_m.set_array([])

    lines = [Line2D([0], [0], color='black', linewidth=2, linestyle=ls) for ls in ['-', ':']]
    labels = [r'$\Sigma_{\mathrm{SFR,UV}}$', r'A$_{\mathrm{FUV}}$']


    for ii, tag in enumerate(tags[::-1]):

        z = float(tag[5:].replace('p','.'))

        func    = partial(get_data, tag=tag, limit=limit, bins=rbins)
        pool    = schwimmbad.MultiPool(processes=8)
        dat     = np.array(list(pool.map(func, np.arange(0,40))), dtype='object')
        pool.close()

        mstar       = np.concatenate(dat[:,0])
        SFR10       = np.concatenate(dat[:,1])
        SFR100      = np.concatenate(dat[:,2])
        LFUVatt     = lum_to_M(np.concatenate(dat[:,3]))
        Afuv        = np.concatenate(dat[:,4])


        stellar_rhalf   = np.concatenate(dat[:,11])
        dust_rhalf      = np.concatenate(dat[:,12])
        tmp             = np.ones((len(mstar),len(rbins)))

        stellar_rhalfbins   = (tmp.T*stellar_rhalf).T*rbins
        dust_rhalfbins      = (tmp.T*dust_rhalf).T*rbins
        sbins_area          = np.pi*(stellar_rhalfbins[:,1:]**2 - stellar_rhalfbins[:,:-1]**2)
        dbins_area          = np.pi*(dust_rhalfbins[:,1:]**2 - dust_rhalfbins[:,:-1]**2)

        sfr10_sbins     = np.zeros((len(mstar), len(rbinscen)))
        sfr100_sbins    = np.zeros((len(mstar), len(rbinscen)))
        Afuv_sbins      = np.zeros((len(mstar), len(rbinscen)))
        sfr10_dbins     = np.zeros((len(mstar), len(rbinscen)))
        sfr100_dbins    = np.zeros((len(mstar), len(rbinscen)))
        Afuv_dbins      = np.zeros((len(mstar), len(rbinscen)))


        inicount, fincount = 0, 0

        for jj in range(40):

            fincount+=len(dat[jj][0])

            sfr10_sbins[inicount:fincount]  = dat[:,5][jj]
            sfr100_sbins[inicount:fincount] = dat[:,6][jj]
            Afuv_sbins[inicount:fincount]   = dat[:,7][jj]
            sfr10_dbins[inicount:fincount]  = dat[:,8][jj]
            sfr100_dbins[inicount:fincount] = dat[:,9][jj]
            Afuv_dbins[inicount:fincount]   = dat[:,10][jj]

            inicount=fincount


        axs[ii].text(2.5, 1.5, r'$z = {}$'.format(z), fontsize = 14)
        twinaxs[ii] = axs[ii].twinx()

        print ('------------------------------------------------')
        print ('tag:', tag)

        for jj in range(len(lbins)-1):

            ok = np.logical_and(LFUVatt>lbins[jj], LFUVatt<=lbins[jj+1])
            if np.sum(ok)>=10:

                axs[ii].plot(rbinscen, np.log10(np.nanmedian(Afuv_sbins[ok]*sfr100_sbins[ok]/sbins_area[ok], axis=0)), color=s_m.to_rgba((lbins[jj]+lbins[jj+1])/2), lw=3)

                twinaxs[ii].plot(rbinscen, -2.5*np.log10(np.nanmedian(Afuv_sbins[ok], axis=0)), color=s_m.to_rgba((lbins[jj]+lbins[jj+1])/2), ls='dotted', lw=2)

        print ('------------------------------------------------')

        twinaxs[ii].grid(False)
        twinaxs[ii].set_ylim(0,5.5)
        twinaxs[ii].set_yticks(np.arange(0,5.5,1))

        for label in (twinaxs[ii].get_yticklabels()):
            label.set_fontsize(13)

    for ii, ax in enumerate(axs):
        ax.grid(True, linestyle=(0, (0.5, 3)))
        ax.set_xlim(0,4.75)
        ax.set_xticks(np.arange(0,5,1))
        ax.set_ylim(-2.5,2.25)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)

        twinaxs[ii].set_yticklabels([])
        if (ii==2) or (ii==5):
            twinaxs[ii].set_yticklabels(np.arange(0,5.5,1))


    axs[-2].set_xlabel(r'$\mathrm{r/r}_{1/2,\star}$', fontsize=15)
    axs[0].legend(lines, labels, frameon=False, fontsize=12, loc=2)

    fig.subplots_adjust(right=0.9, wspace=0, hspace=0)

    fig.text(0.07, 0.5, r'log$_{10}$($\Sigma_{\mathrm{SFR,UV}}$/(M$_{\odot}$yr$^{-1}$kpc$^{-2}$))', va='center', rotation='vertical', fontsize=15)
    fig.text(0.935, 0.5, r'A$_{\mathrm{FUV}}$', va='center', rotation='vertical', fontsize=15)

    cbaxes = fig.add_axes([0.97, 0.3, 0.01, 0.4])
    cbaxes.set_ylabel(r'M$_{\mathrm{FUV}}$', fontsize=15)
    fig.colorbar(s_m, cax=cbaxes,label=r'M$_{\mathrm{FUV}}$')

    plt.savefig(F'../results/SFR100unobsc_radial_variation_z5_10.pdf', bbox_inches='tight', dpi=300)
    plt.show()
