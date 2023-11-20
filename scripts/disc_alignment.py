import sys
sys.path.append('../src')
import numpy as np
import h5py
from functools import partial
import schwimmbad
import matplotlib
import matplotlib.pyplot as plt

from helpers import get_slen30
from mollview import get_kappa
from Mstar_sfr import get_hist

def get_data(ii, tag, z):

    num = str(ii)
    if len(num) == 1:
        num = '0'+num

    with h5py.File(F'../data1/FLARES_{num}_data.hdf5', 'r') as hf:
        LFUV = np.array(hf[tag+'/Galaxy/BPASS_2.2.1/Chabrier300/los/Luminosity/DustModelI'].get('FUV'), dtype=np.float64)

    sim = rF"../../flares_pipeline/data/flares.hdf5"
    with h5py.File(sim, 'r') as hf:
        S_len   = np.array(hf[num+'/'+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)
        S_ap    = np.array(hf[num+'/'+tag+'/Particle/Apertures/Star'].get('30'), dtype = bool)
        LFUVint = np.array(hf[num+'/'+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float64)
        SFR100  = np.array(hf[num+'/'+tag+'/Galaxy/SFR_aperture/30'].get('100Myr'), dtype = np.float64)

    S_len, begin, end, S_ap = get_slen30(S_len, S_ap, limit)
    ok = np.where(S_len>limit)[0]
    if len(ok)>0:
        AFUV = -2.5 * np.log10(LFUV/np.vstack(LFUVint[ok]))
        SFR100 = SFR100[ok]
        AFUV_spread = np.percentile(AFUV, 84, axis=1) - np.percentile(AFUV, 16, axis=1)

        Kco_s, Krot_s, ang_vector_star, ang_vector_star_s, Kco_gas_s, Krot_gas_s, ang_vector_gas, ang_vector_gas_s = get_kappa(num, tag, z)

        a_b             = np.linalg.norm(ang_vector_star) * np.linalg.norm(ang_vector_gas)
        gs_cos_angle    = np.absolute(np.diag(ang_vector_star @ ang_vector_gas.T))/a_b


    else:
        AFUV_spread, gs_cos_angle = np.array([]), np.array([])

    return AFUV_spread, gs_cos_angle



if __name__ == "__main__":
    # inp = int(sys.argv[1])
    limit=500

    # choose a colormap
    cmap = cmr.rainforest                   # CMasher
    c_m = plt.get_cmap('cmr.bubblegum_r')   # MPL
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5,6,1), c_m.N)
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    tags = ['010_z005p000','009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), sharex=False, sharey=False, facecolor='w', edgecolor='k')

    fig     = plt.figure(figsize = (5, 4))
    ax      = fig.add_axes((0.1, 0.1, 0.6, 0.6))
    ax_R    = fig.add_axes((0.7, 0.1, 0.3, 0.6))
    ax_T    = fig.add_axes((0.1, 0.7, 0.6, 0.35))

    num = []

    for ii, tag in enumerate(tags):

        z = float(tag[5:].replace('p','.'))

        func = partial(get_data, tag=tag, z=z)
        pool = schwimmbad.MultiPool(processes=8)
        dat = np.array(list(pool.map(func, np.arange(0,40))), dtype='object')
        pool.close()

        AFUV_spread     = np.concatenate(dat[:,0])
        gs_cos_angle    = np.concatenate(dat[:,1])


        x = AFUV_spread
        y = gs_cos_angle
        ylabel = r'cos$(\theta)$'
        xlabel = r'A$_{FUV,84}$ - A$_{FUV,16}$'
        savename = '../results/AFUV_spread_gsangle.pdf'

        ax.scatter(x, y, s=3, alpha=0.25, color=s_m.to_rgba(ii), label = rF'z={z} ({len(x)})')
        num.append(len(x))

        bins = np.arange(0, 1.1, 0.1)
        X, Y = get_hist(x, bins = bins)
        ax_T.plot(X, Y, lw = 2, color=s_m.to_rgba(ii))

        bins = np.arange(0, 1, 0.1)
        X, Y = get_hist(y, bins = bins)
        ax_R.plot(Y, X, lw = 2, color=s_m.to_rgba(ii))


    for axs in [ax, ax_R, ax_T]:
        axs.grid(True, ls='dotted')
        for label in (axs.get_xticklabels() + axs.get_yticklabels()):
            label.set_fontsize(11)


    ax.set_xlim([0, 1.1])
    ax_T.set_xlim([0, 1.1])
    ax_T.set_xticklabels([])

    ax.set_ylim([0,1.1])
    ax_R.set_ylim([0,1.1])
    ax_R.set_yticklabels([])

    ax_R.set_xlim(left=0)
    ax_T.set_ylim(bottom=0)

    # ax_R.set_xticks(np.arange(0,4,1))
    # ax_T.set_yticks(np.arange(0,4,1))

    ax_R.set_xlabel(r'log$_{10}$(N)')
    ax_T.set_ylabel(r'log$_{10}$(N)')


    ax.set_xlabel(xlabel, fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)
    ax.legend(frameon=False, fontsize=9, scatterpoints=1, loc=4, markerscale=3)

    plt.savefig(savename, bbox_inches='tight', dpi=300)

    plt.show()
