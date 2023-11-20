"""
Figure 5 in paper: Attenuation curve variation with different
amount of dust along different lines-of-sight, 
choose inp=sys.argv[1]=None
"""

import sys
import bagpipes as pipes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cmasher as cmr

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(650010)))

def calc_AHalpha(x, kHalpha, kHbeta, delta=0.):

    """
    Get the attenuation from the Balmer decrement

    Args:
        x = 2.5 x log10((Halpha/Hbeta)/2.86)
        kHalpha = Value of the extinction curve at Halpha wavelength
        kHbeta = Value of the extinction curve at Hbeta wavelength

    Returns:
        y = Returns the Halpha attenuation

    """

    Halpha_lam  = 6562.81
    V_lam       = 5500.

    y = 2.5 * (kHalpha/(kHbeta - kHalpha)) * x * (Halpha_lam/V_lam)**delta

    return y
if __name__ == "__main__":

    inp = sys.argv[1]

    model = 'bc03_miles'
    # pipes.config.set_model_type(model)

    obs_wavs = np.arange(1290., 10000., 5.)*(7)

    z=5
    up=8.5

    colours = ['green', 'blue', 'orange', 'magenta', 'violet', 'olive', 'red', 'yellow']

    # choose a colormap
    cmap = cmr.pepper                   # CMasher
    norm = matplotlib.colors.Normalize(1,15)
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_m.set_array([])


    n=50#6

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5), sharex=True, sharey=False, facecolor='w', edgecolor='k')
    axs = axs.ravel()


    mu, sigma = 2., 0.3
    nsigma = np.arange(2,15,2)

    age = 0.005
    Z = 0.1
    if inp == "None":
        ages=[age]*n
        metallicities=[Z]*n
    elif inp == 'agevary':
        ages=np.random.uniform(0.001,0.01,n)
        metallicities=[Z]*n
    elif inp == 'Zvary':
        ages=[age]*n
        metallicities=np.random.uniform(0.01,1.,n)
    else:
        print("Not a valid argument!")
        sys.exit()

    Halpha_dusts    = np.zeros(len(nsigma))
    Hbeta_dusts     = np.zeros(len(nsigma))
    Hgamma_dusts    = np.zeros(len(nsigma))
    OIII_dusts      = np.zeros(len(nsigma))
    NII_dusts       = np.zeros(len(nsigma))
    # age=[0.009]*n
    # metallicity=[0.1]*n
    for kk, ns in enumerate(nsigma):
        star_cluster_props = {'age':ages, 'metallicity':metallicities, 'Av':mu + np.random.normal(0, ns*sigma, n), 'logU':[-2.]*n, 'mass': [10**7]*n}

        print ('sigma=', ns)
        print (star_cluster_props)

        star_cluster_models_intrinsic, star_cluster_models_dust, star_cluster_models_noneb_intrinsic, star_cluster_models_noneb_dust = {}, {}, {}, {}
        total_int = 0.
        total_dust = 0.
        total_noneb_int = 0.
        total_noneb_dust = 0.
        dust_type='SMC'

        Halpha  = 'H  1  6562.81A'
        Hbeta   = 'H  1  4861.33A'
        Hgamma  = 'H  1  4340.46A'
        OIII    = 'O  3  5006.84A'
        NII     = 'N  2  6583.45A'

        Halpha_int  = 0
        Halpha_dust = 0
        Hbeta_int   = 0
        Hbeta_dust  = 0
        Hgamma_int  = 0
        Hgamma_dust = 0
        OIII_int    = 0
        OIII_dust   = 0
        NII_int     = 0
        NII_dust    = 0

        for ii in range(n):

            for jj in range(4):

                model_components = {}                   # The model components dictionary
                model_components["redshift"] = z

                sfh = {}                          # Tau model star formation history component
                # sfh["age"] = star_cluster_props['age'][ii]                   # Gyr
                # sfh["tau"] = 0.9                 # Gyr

                sfh['age'] = star_cluster_props['age'][ii]
                # sfh['tanefolds'] = 0.5
                sfh["massformed"] = np.log10(star_cluster_props['mass'][ii])            # log_10(M*/M_solar)
                sfh["metallicity"] = star_cluster_props['metallicity'][ii]          # Z/Z_oldsolar

                dust = {}                         # Dust component
                dust["type"] = dust_type        # Define the shape of the attenuation curve
                if (jj==0) or (jj==2):
                    dust["Av"] = 0 # no dust
                else:
                    av = star_cluster_props['Av'][ii]
                    if av<=0: av=0.01
                    dust["Av"] = av                 # magnitudes

                dust["eta"] = 1.                  # Extra dust for young stars: multiplies Av


                nebular = {}                      # Nebular emission component
                nebular["logU"] = star_cluster_props['logU'][ii]             # log_10(ionization parameter)


                model_components["t_bc"] = 0.01        # Lifetime of birth clouds (Gyr)
                # model_components["veldisp"] = 200.      # km/s
                if jj<2:
                    model_components["t_bc"] = 0.01
                    model_components["nebular"] = nebular
                model_components["burst"] = sfh
                model_components["dust"] = dust



                if jj==0:
                    star_cluster_models_intrinsic[F'{ii}'] = pipes.model_galaxy(model_components, spec_wavs=obs_wavs, spec_units='mujy')

                    total_int+=star_cluster_models_intrinsic[F'{ii}'].spectrum[:,1]
                elif jj==1:
                    star_cluster_models_dust[F'{ii}'] = pipes.model_galaxy(model_components, spec_wavs=obs_wavs, spec_units='mujy')

                    total_dust+=star_cluster_models_dust[F'{ii}'].spectrum[:,1]

                elif jj==2:
                    star_cluster_models_noneb_intrinsic[F'{ii}'] = pipes.model_galaxy(model_components, spec_wavs=obs_wavs, spec_units='mujy')
                    total_noneb_int+=star_cluster_models_noneb_intrinsic[F'{ii}'].spectrum[:,1]
                elif jj==3:
                    star_cluster_models_noneb_dust[F'{ii}'] = pipes.model_galaxy(model_components, spec_wavs=obs_wavs, spec_units='mujy')
                    total_noneb_dust+=star_cluster_models_noneb_dust[F'{ii}'].spectrum[:,1]


        lam = star_cluster_models_dust['0'].spectrum[:,0]/(1+z)
        tmp = np.abs(lam-5500)
        lam = lam/1e4
        ok = np.argmin(tmp)

        for ii in range(n):

            Halpha_int+=star_cluster_models_intrinsic[F'{ii}'].line_fluxes[Halpha]
            Halpha_dust+=star_cluster_models_dust[F'{ii}'].line_fluxes[Halpha]

            Hbeta_int+=star_cluster_models_intrinsic[F'{ii}'].line_fluxes[Hbeta]
            Hbeta_dust+=star_cluster_models_dust[F'{ii}'].line_fluxes[Hbeta]

            Hgamma_int+=star_cluster_models_intrinsic[F'{ii}'].line_fluxes[Hgamma]
            Hgamma_dust+=star_cluster_models_dust[F'{ii}'].line_fluxes[Hgamma]

            OIII_int+=star_cluster_models_intrinsic[F'{ii}'].line_fluxes[OIII]
            OIII_dust+=star_cluster_models_dust[F'{ii}'].line_fluxes[OIII]

            NII_int+=star_cluster_models_intrinsic[F'{ii}'].line_fluxes[NII]
            NII_dust+=star_cluster_models_dust[F'{ii}'].line_fluxes[NII]


        print (F'{Halpha} =', Halpha_int)
        print (F'{Hbeta}  =', Hbeta_int)
        print (F'{Hgamma} =', Hgamma_int)
        print (F'{OIII} =', OIII_int)
        print (F'{NII} =', NII_int)

        Halpha_dusts[kk]    = Halpha_dust
        Hbeta_dusts[kk]     = Hbeta_dust
        Hgamma_dusts[kk]    = Hgamma_dust
        OIII_dusts[kk]      = OIII_dust
        NII_dusts[kk]       = NII_dust

        A_lam = -2.5*np.log10(total_dust/total_int)
        A_lam_V = A_lam/A_lam[ok]
        if kk==1:
            axs[0].plot(lam, np.log10(total_int), color='blue', lw=2, ls='dotted', label='Intrinsic')

        axs[0].plot(lam, np.log10(total_dust), color=s_m.to_rgba(ns), lw=2)
        axs[1].plot(lam, A_lam_V, color=s_m.to_rgba(ns), label=Fr'Av = {np.around(A_lam[ok],2)}, $\sigma={int(ns)}$', lw=2)

    A_curve = star_cluster_models_dust['0'].dust_atten.A_cont
    lam = star_cluster_models_dust['0'].dust_atten.wavelengths
    tmp = np.abs(lam-5500)
    lam = lam/1e4
    ok = np.argmin(tmp)
    A_curve_V = A_curve/A_curve[ok]

    ok = np.logical_and(lam>0.1, lam<1.)

    axs[1].plot(lam[ok], A_curve_V[ok], label=F'{dust_type}', color = 'blue', ls='dotted', lw=4)

    for ax in axs:
        ax.set_xlim(0.1,1)
        ax.set_xlabel(r'$\lambda$/$\mu$m', fontsize=14)
        ax.grid(True, ls='dotted')
        ax.legend(frameon=False, fontsize=10)

    axs[0].set_ylim(-4,1)
    axs[0].set_ylabel(r'log$_{10}$(F$_{\lambda}/\mu$jy)', fontsize=14)

    axs[1].set_ylim(0.,6)
    axs[1].set_ylabel(r'A$_{\lambda}$/A$_{\mathrm{V}}$', fontsize=14)

    plt.savefig(F'../results/att_curve_{dust_type}_{model}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    np.savez(F'../results/{dust_type}_{age}Myr_{Z}Zsun', Halpha_int = Halpha_int, Halpha_dusts = Halpha_dusts, Hbeta_int = Hbeta_int, Hbeta_dusts = Hbeta_dusts, Hgamma_int = Hgamma_int, Hgamma_dusts = Hgamma_dusts, OIII_int = OIII_int, OIII_dusts = OIII_dusts, NII_int = NII_int, NII_dusts = NII_dusts)