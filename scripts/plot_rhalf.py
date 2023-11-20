import numpy as np
import pandas as pd
from functools import partial
import matplotlib, h5py, schwimmbad
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['legend.fancybox'] = False
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")


def get_rhalf(ii, tag, dataset='Star'):

    num = str(ii)

    if len(num) == 1:
        num =  '0'+num

    sim_orient = rF"../data1/FLARES_{num}_data.hdf5"
    sim = rF"../../flares_pipeline/data/flares.hdf5"
    num = num+'/'

    with h5py.File(sim, 'r') as hf:
        S_len       = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)
        mstar   = np.array(hf[num+tag+'/Galaxy/Mstar_aperture/'].get('30'), dtype = np.float32)*1e10

        S_ap        = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype = np.bool)

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
    mstar = mstar[ok]

    with h5py.File(sim_orient, 'r') as hf:

        rhalf   = np.array(hf[tag+'/Galaxy/HalfMassRadii'].get(dataset), dtype = np.float32)


    return mstar, rhalf


if __name__ == "__main__":

    quantiles = [0.84,0.50,0.16]
    limit=500
    df = pd.read_csv('../data/weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']


    fig, axs = plt.subplots(nrows = 1, ncols = 6, figsize=(13, 3), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()

    for ii in range(6):
        tag = tags[ii]
        z = float(tag[5:].replace('p','.'))


        func    = partial(get_rhalf, tag=tag)
        pool    = schwimmbad.MultiPool(processes=8)
        dat     = np.array(list(pool.map(func, np.arange(0,40))))
        pool.close()

        x       = np.log10(np.concatenate(dat[:,0]))
        y       = np.log10(np.concatenate(dat[:,1]))


        if ii==0:

            vmax=len(mstar)
            extent = [*[np.min(x[np.isfinite(x)])-0.1,np.max(x[np.isfinite(x)])+0.1], *[np.min(y[np.isfinite(y)])-0.1,np.max(y[np.isfinite(y)])+0.1]]
            gridsize = (20,20)

            hb = axs[ii].hexbin(x, y, bins='log', cmap=matplotlib.cm.copper_r, vmin=1, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=1)

        else:

            axs[ii].hexbin(x, y, bins='log', cmap=matplotlib.cm.copper_r, vmin=1, vmax=vmax, gridsize=gridsize, extent=extent, linewidths=0., mincnt=1)


        axs[ii].text(10.25, 0.5, r'$z = {}$'.format(z), fontsize = 12)


    for ax in axs:
        ax.grid(True, linestyle=(0, (0.5, 3)))
        ax.tick_params(axis="y",direction="in")
        ax.tick_params(axis="x",direction="in")

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)

    axs[0].set_ylabel(r'R$_{1/2,\star}$', fontsize=14)
    fig.text(0.47, 0.02, r'log$_{10}$(M$_{\star}$/M$_{\odot}$)', va='center', fontsize=14)


    fig.subplots_adjust(right=0.9, wspace=0, hspace=0)
    cbaxes = fig.add_axes([0.93, 0.3, 0.01, 0.4])
    cbaxes.set_ylabel(r'$\mathrm{N}$', fontsize=11)
    # cbaxes.set_xlabel(r'f$_{\mathrm{esc}}$', fontsize=11)
    fig.colorbar(hb, cax=cbaxes, label=r'$\mathrm{N}$')

    plt.savefig(F'../results/rhalf_stellar.pdf', bbox_inches='tight', dpi=300)

    plt.show()
