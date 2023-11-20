import timeit
import sys
sys.path.append('../src')
import numpy as np
import healpy as hp
import h5py
from functools import partial
import schwimmbad
import matplotlib
import matplotlib.pyplot as plt
from FLARE.photom import lum_to_M, M_to_lum

from helpers import get_slen30
from mollview import get_kappa


def get_theta(phi_vals, normal):

    x, y, z = normal

    theta_vals = - np.arctan2(z, x*np.cos(phi_vals) + y*np.sin(phi_vals))
    theta_vals[theta_vals<0]+=np.pi

    return theta_vals

def create_mollview_plot(ii, att_s, SFR, Kco_s, Krot_s, ang_vector_star, ang_vector_star_s, Kco_gas_s, Krot_gas_s, ang_vector_gas, ang_vector_gas_s):

    #viewing angles
    nside = 8
    hp_theta, hp_phi = hp.pix2ang(nside, range(hp.nside2npix(nside)))
    angles = np.vstack([hp_theta, hp_phi]).T.astype(np.float32)

    ## Define pixels on sky
    pixel_indices = hp.ang2pix(nside, hp_theta, hp_phi)
    m = np.zeros(hp.nside2npix(nside))

    m[pixel_indices] = att_s[ii]

    hp.mollview(m, cbar=None,  cmap=matplotlib.cm.coolwarm, title=r"SFR$=%0.3f$M$_{\odot}/$yr, $\kappa_{\mathrm{co},\star}=%0.3f$, $\kappa_{\mathrm{co, gas}}=%0.3f$"%(SFR[ii], Kco_s[ii], Kco_gas_s[ii]))


    phi = np.linspace(0, 2*np.pi,10000)

    theta = get_theta(phi, ang_vector_star[ii])
    hp.projplot(theta, phi, ls='solid', color='orange')
    hp.projscatter(ang_vector_star_s[ii][1], ang_vector_star_s[ii][2], marker='X', color='orange')


    theta = get_theta(phi, ang_vector_gas[ii])
    hp.projplot(theta, phi, ls='solid', color='blue')
    hp.projscatter(ang_vector_gas_s[ii][1], ang_vector_gas_s[ii][2], marker='X', color='blue')

    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    # cbaxes = fig.colorbar(image, ax=ax, cmap=matplotlib.cm.coolwarm)
    cbaxes = fig.add_axes([0.3, 0.05, 0.4, 0.05])
    fig.colorbar(image, cax=cbaxes, orientation="horizontal")
    cbaxes.set_xlabel(r'A$_{\mathrm{FUV}}$', fontsize=15)
    c_xlims = cbaxes.get_xlim()
    cbaxes.set_xticks(np.round(np.linspace(np.round(c_xlims[0]-0.01, 3), np.round(c_xlims[1]+0.01, 3), num=2, endpoint=True), 3))
    


if __name__ == "__main__":

    total=0
    disc=0
    ii = sys.argv[1]

    tag, limit = '010_z005p000', 500
    num = str(ii)
    if len(num) == 1:
        num =  '0'+num

    z = float(tag[5:].replace('p','.'))


    Kco_s, Krot_s, ang_vector_star, ang_vector_star_s, Kco_gas_s, Krot_gas_s, ang_vector_gas, ang_vector_gas_s = get_kappa(num, tag, z)

    # total+=len(Kco_s)

    with h5py.File(F'../data/FLARES_{num}_data.hdf5', 'r') as hf:
        LFUV = np.array(hf[tag+'/Galaxy/BPASS_2.2.1/Chabrier300/los/Luminosity/DustModelI'].get('FUV'), dtype=np.float64)

    sim = rF"../../flares_pipeline/data/flares.hdf5"
    with h5py.File(sim, 'r') as hf:
        S_len   = np.array(hf[num+'/'+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)
        S_ap    = np.array(hf[num+'/'+tag+'/Particle/Apertures/Star'].get('30'), dtype = bool)
        LFUVint = np.array(hf[num+'/'+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float32)
        SFR100  = np.array(hf[num+'/'+tag+'/Galaxy/SFR_aperture/30'].get('100Myr'), dtype = np.float64)

    S_len, begin, end, S_ap = get_slen30(S_len, S_ap, limit)
    ok = np.where(S_len>limit)[0]
    AFUV = -2.5 * np.log10(LFUV/np.vstack(LFUVint[ok]))
    SFR100 = SFR100[ok]

    for ii, jj in enumerate(ok):

        create_mollview_plot(ii, AFUV, SFR100, Kco_s, Krot_s, ang_vector_star, ang_vector_star_s, Kco_gas_s, Krot_gas_s, ang_vector_gas, ang_vector_gas_s)

        plt.savefig(F'../results/mollview_plots/{num}/{num}_{jj}.pdf', dpi = 300, bbox_inches='tight')
        plt.close()

