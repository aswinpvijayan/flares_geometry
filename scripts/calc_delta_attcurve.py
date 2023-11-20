import numpy as np
import pandas as pd
from functools import partial
import h5py, schwimmbad, matplotlib
import matplotlib.pyplot as plt
from interrogator.sed import dust_curves

import sys
sys.path.append('../src')

from helpers import get_slen30

def get_data(ii, tag, limit=1000):

    num = str(ii)

    if len(num) == 1:
        num =  '0'+num

    sim = rF"../../flares_pipeline/data/flares.hdf5"
    num = num+'/'

    with h5py.File(sim, 'r') as hf:
        S_len       = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)

        Mstar   = np.array(hf[num+tag+'/Galaxy/Mstar_aperture'].get('30'), dtype = np.float32)*1e10

        LFUVatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float32)
        LFUVint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float32)

        LVatt = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('V'), dtype = np.float32)
        LVint = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('V'), dtype = np.float32)

        S_ap    = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype = np.bool)

    ####Adding beta and beta0 to see their effect on these curve shape

    z = float(tag[5:].replace('p','.'))

    S_len, begin, end, S_ap = get_slen30(S_len, S_ap, limit)
    ok = np.where(S_len>limit)[0]

    Mstar, SFR10, SFR100, LFUVint, LFUVatt, LVint, LVatt = Mstar[ok], SFR10[ok], SFR100[ok], LFUVint[ok], LFUVatt[ok], LVint[ok], LVatt[ok]


    Av = -2.5 * np.log10(LVatt/LVint)
    Afuv = -2.5*np.log10(LFUVatt/LFUVint)

    return Mstar, SFR10, SFR100, Av, Afuv, Alam


if __name__ == "__main__":

    quantiles = [0.84,0.50,0.16]
    limit = 500
    df = pd.read_csv('../data/weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(13, 8), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()

    cbins = np.arange(-2,2.5,0.5)#np.arange(0,4.5,0.5)#np.array([-24, -23, -22, -21, -20, -19, -18.5, -18])#np.arange(-24, -17.5, 1)
    # choose a colormap
    c_m = matplotlib.cm.plasma
    norm = matplotlib.colors.BoundaryNorm(cbins, c_m.N)
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])


    for ii, tag in enumerate(tags):
        func    = partial(get_data, tag=tag, limit=limit)
        pool    = schwimmbad.MultiPool(processes=8)
        dat     = np.array(list(pool.map(func, np.arange(0,40))))
        pool.close()

        Mstar       = np.concatenate(dat[:,0])
        SFR10       = np.concatenate(dat[:,1])
        SFR100      = np.concatenate(dat[:,2])
        LFUVint     = np.concatenate(dat[:,3])
        LFUVatt     = np.concatenate(dat[:,4])
        Av          = np.concatenate(dat[:,5])
        lam         = dat[0][6]


        Alam     = np.zeros((len(SFR10), len(lam)))

        inicount, fincount = 0, 0

        for jj in range(40):
            fincount+=len(dat[jj][0])

            Alam[inicount:fincount] = dat[jj][-1]

            inicount=fincount


        # SFR10, SFR100 , LFUVint, LFUVatt, Av, lam, SED_int, SED_att = func(0)
        # Alam = -2.5*np.log10(SED_att/SED_int)
        # c = lum_to_M(LFUVatt)
        c = 9+np.log10(SFR10/Mstar)#-2.5*np.log10(LFUVatt/LFUVint)
        x = np.log10(lam)


        for jj in range(len(cbins)-1):

            ok = np.where((c>=cbins[jj]) & (c<cbins[jj+1]))[0]
            if len(ok)>0:

                y = Alam[ok]/np.vstack(Av[ok])
                yy = np.median(y, axis=0)
                yy84 = np.percentile(y, 84, axis=0)
                yy16 = np.percentile(y, 16, axis=0)

                axs[ii].plot(lam, yy, lw=1, color=s_m.to_rgba((cbins[jj]+cbins[jj+1])/2))
                # axs.fill_between(lam, yy84, yy16, color=s_m.to_rgba((MFUVbins[ii]+MFUVbins[ii+1])/2), alpha=0.5)

        axs[ii].plot(lam, dust_curves.SMC_Pei92().tau(lam*1e4), label='SMC', color = 'red', ls='dotted', lw=4)
        axs[ii].plot(lam, dust_curves.Starburst_Calzetti2000().tau(lam*1e4), label='Calzetti-Starburst', color = 'pink', ls='dotted', lw=4)
        axs[ii].plot(lam, dust_curves.simple().tau(lam*1e4), label='Default', color = 'green', ls='dotted', lw=4)


    axs[-1].legend(frameon=False)
    fig.text(0.05, 0.5, r'A$_{\lambda}$/A$_{\mathrm{V}}$', va='center', rotation='vertical', fontsize=15)
    axs[-2].set_xlabel(r'log$_{10}$($\lambda$/$\mu$m)', fontsize=12)

    # axs.set_ylabel(r'A$_{\lambda}$/A$_{\mathrm{V}}$', fontsize=12)


    for ax in axs:
        ax.grid(True, alpha = 0.4)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.set_ylim(0,10)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)

    fig.subplots_adjust(right = 0.91, wspace=0, hspace=0)
    cbaxes = fig.add_axes([0.925, 0.35, 0.007, 0.3])
    fig.colorbar(s_m, cax=cbaxes)
    cbaxes.set_ylabel(r'A$_{\mathrm{FUV}}$', fontsize = 12)
    cbaxes.set_ylabel(r'sSFR/Gyr$^{-1}$', fontsize = 12)
    for label in cbaxes.get_yticklabels():
        label.set_fontsize(13)

    plt.savefig(F"../results/att_curve_z5_10_sSFR.pdf", bbox_inches='tight', dpi=300)
    plt.show()
