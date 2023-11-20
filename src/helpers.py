import numpy as np
from numba import njit, float64, float32, int32, prange, types, config, threading_layer
config.THREADING_LAYER = 'threadsafe'
import healpy as hlp
import astropy.units as u
from astropy_healpix import HEALPix
from photutils import CircularAperture, aperture_photometry
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from astropy.cosmology import Planck13 as cosmo
import sympy

import SynthObs
from SynthObs.SED import models
import flare
import flare.filters



def get_slen30(S_len, S_ap, limit):

    begin = np.zeros(len(S_len), dtype = np.int32)
    end = np.zeros(len(S_len), dtype = np.int32)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    n = len(S_len)
    S_len_this = np.zeros(n, dtype=np.int32)
    for kk in range(n):
        k = np.sum(S_ap[begin[kk]:end[kk]])
        S_len_this[kk] = k

    begin_this, end_this, S_ap_this = get_aperture30(begin, end, S_len_this, S_ap, limit)

    return S_len_this, begin_this, end_this, S_ap_this

def get_aperture30(begin, end, S_len_this, S_ap, limit):

    ok = np.where(S_len_this>limit)[0]
    n = np.sum(S_len_this[ok])

    S_ap_this = np.zeros(n, dtype=np.bool)
    inicount = 0
    fincount = 0
    if len(ok)>0:
        for ii in ok:
            fincount+=S_len_this[ii]
            # print (S_len30[ii], end[ii]-begin[ii])

            # print (S_ap30[kk,inicount:fincount])
            # print (S_ap[kk,begin[ii]:end[ii]][inicount:fincount])
            S_ap_this[inicount:fincount] = S_ap[begin[ii]:end[ii]][0:S_len_this[ii]]

            inicount=fincount

        begin_this = np.zeros(len(ok), dtype=np.int32)
        end_this = np.zeros(len(ok), dtype=np.int32)
        begin_this[1:] = np.cumsum(S_len_this[ok])[:-1]
        end_this = np.cumsum(S_len_this[ok])
        begin_this = begin_this.astype(np.int32)
        end_this = end_this.astype(np.int32)

    else:
        begin_this, end_this = np.array([]), np.array([])

    return begin_this, end_this, S_ap_this



@njit((types.float32[:,:], types.float32[:,:], types.float32[:], types.float32[:], types.float32[:], types.float32[:], types.int32), parallel=True, nogil=True)
def get_Z_LOS(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins):

    """

    Compute the los metal surface density (in Msun/Mpc^2) for star particles inside the galaxy taking
    the z-axis as the los.
    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length

    """
    n           = len(s_cood)
    Z_los_SD    = np.zeros(n, dtype=np.float32)

    #Fixing the observer direction as z-axis. Use make_faceon() for changing the
    #particle orientation to face-on
    xdir, ydir, zdir = 0, 1, 2

    for ii in prange(n):

        thisspos    = s_cood[ii]
        ok          = g_cood[:,zdir] > thisspos[zdir]
        thisgpos    = g_cood[ok]
        thisgsml    = g_sml[ok]
        thisgZ      = g_Z[ok]
        thisgmass   = g_mass[ok]

        x = thisgpos[:,xdir] - thisspos[xdir]
        y = thisgpos[:,ydir] - thisspos[ydir]

        boverh = np.sqrt(x*x + y*y) / thisgsml

        ok          = boverh <= 1.
        kernel_vals = np.array([lkernel[int(kbins*ll)] for ll in boverh[ok]])


        Z_los_SD[ii] = np.sum((thisgmass[ok]*thisgZ[ok]/(thisgsml[ok]*thisgsml[ok]))*kernel_vals) #in units of Msun/Mpc^2


    return Z_los_SD



def get_Z_LOS_kd(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins,
                 dimens=(0, 1, 2)):
    """

    Compute the los metal surface density (in Msun/Mpc^2) for star
    particles inside the galaxy taking the z-axis as the los.

    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length
        dimens (tuple: int): tuple of xyz coordinates

    """

    # Generalise dimensions (function assume LOS along z-axis)
    xdir, ydir, zdir = dimens

    # Get how many stars
    nstar = s_cood.shape[0]

    # Lets build the kd tree from star positions
    tree = cKDTree(s_cood[:, (xdir, ydir)])

    # Query the tree for all gas particles (can now supply multiple rs!)
    query = tree.query_ball_point(g_cood[:, (xdir, ydir)], r=g_sml, p=1)

    # Now we just need to collect each stars neighbours
    star_gas_nbours = {s: [] for s in range(nstar)}
    for g_ind, sparts in enumerate(query):
        for s_ind in sparts:
            star_gas_nbours[s_ind].append(g_ind)

    # Initialise line of sight metal density
    Z_los_SD = np.zeros(nstar)

    # Loop over stars
    for s_ind in range(nstar):

        # Extract gas particles to consider
        g_inds = star_gas_nbours.pop(s_ind)

        # Extract data for these particles
        thisspos = s_cood[s_ind]
        thisgpos = g_cood[g_inds]
        thisgsml = g_sml[g_inds]
        thisgZ = g_Z[g_inds]
        thisgmass = g_mass[g_inds]

        # We only want to consider particles "in-front" of the star
        ok = np.where(thisgpos[:, zdir] > thisspos[zdir])[0]
        thisgpos = thisgpos[ok]
        thisgsml = thisgsml[ok]
        thisgZ = thisgZ[ok]
        thisgmass = thisgmass[ok]

        # Get radii and divide by smooting length
        b = np.linalg.norm(thisgpos[:, (xdir, ydir)]
                           - thisspos[((xdir, ydir), )],
                           axis=-1)
        boverh = b / thisgsml

        # Apply kernel
        kernel_vals = np.array([lkernel[int(kbins * ll)] for ll in boverh])

        # Finally get LOS metal surface density in units of Msun/pc^2
        Z_los_SD[s_ind] = np.sum((thisgmass * thisgZ
                                  / (thisgsml * thisgsml))
                                 * kernel_vals)

    return Z_los_SD

def get_spherical_from_cartesian(coords):

    x, y, z = coords

    xy = x**2 + y**2
    r = np.sqrt(xy + z**2)
    theta = np.arctan2(np.sqrt(xy), z) # for elevation angle defined from Z-axis down
    phi = np.arctan2(y, x)


    return r, theta, phi


def get_cartesian_from_spherical(t_angles):

    x = np.sin(t_angles[0])*np.cos(t_angles[1])
    y = np.sin(t_angles[0])*np.sin(t_angles[1])
    z = np.cos(t_angles[0])

    return np.asarray([x, y, z], dtype=np.float64)


def get_rotation_matrix(i_v, unit=None):
    # This solution is from ---
    # https://stackoverflow.com/questions/43507491/imprecision-with-rotation-matrix-to-align-a-vector-to-an-axis

    # This uses the Rodrigues' rotation formula for the re-projection

    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unit is None:
        unit = [0.0, 0.0, 1.0]
    # Normalize vector length
    i_v /= np.linalg.norm(i_v)

    # Get axis
    uvw = np.cross(i_v, unit)

    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)

    #normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    matrix = (
        rcos * np.eye(3) +
        rsin * np.array([
            [ 0, -w,  v],
            [ w,  0, -u],
            [-v,  u,  0]
        ]) +
        (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    )

    return matrix.astype(np.float64)



def ang_mom_vector(this_mass, this_cood, this_vel, z):


    hubble_flow = cosmo.H(z).value * this_cood * 1e-3 # Hubble flow is km/(Mpc s) and dist in kpc

    this_hvel = this_vel + hubble_flow

    #Get the angular momentum unit vector
    L_tot = np.array([this_mass]).T * np.cross(this_cood, this_hvel)
    L_tot_mag = np.sqrt(np.sum(np.nansum(L_tot, axis = 0)**2))
    L_unit = np.sum(L_tot, axis = 0)/L_tot_mag

    return L_unit


def compute_kappa(this_mass, this_coord, this_vel, z):


    hubble_flow = cosmo.H(z).value * this_coord * 1e-3 # Hubble flow is km/(Mpc s) and dist in kpc

    this_hvel = this_vel + hubble_flow

    L_tot = np.array([this_mass]).T*np.cross(this_coord, this_hvel)
    L_tot_mag = np.sqrt(np.sum(np.nansum(L_tot, axis = 0)**2))

    L_unit = np.sum(L_tot, axis = 0)/L_tot_mag

    R_z = np.cross(this_coord,L_unit)
    absR_z = np.sqrt(np.sum(R_z**2, axis = 1))
    mR = this_mass*absR_z
    K = np.nansum(this_mass*np.sum(this_hvel**2, axis = 1))

    L = np.sum(L_tot*L_unit, axis = 1)
    L_co = np.copy(L)
    co = np.where(L_co > 0.)
    L_co = L_co[co]

    L_mR = (L/mR)**2
    L_co_mR = (L_co/mR[co])**2
    Krot = np.nansum(this_mass*L_mR)/K


    Kco = np.nansum(this_mass[co]*L_co_mR)/K


    return Kco, Krot


def lum(DustSurfaceDensities, Masses, Ages, Metallicities, kappa=0.0795, BC_fac=1.0, IMF = 'Chabrier_300', filters = ['FAKE.TH.FUV'], Type = 'Total', log10t_BC = 7., extinction = 'default'):

    model = models.define_model(F'BPASSv2.2.1.binary/{IMF}') # DEFINE SED GRID -
    if extinction == 'default':
        model.dust_ISM  = ('simple', {'slope': -1.})    #Define dust curve for ISM
        model.dust_BC   = ('simple', {'slope': -1.})     #Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM  = ('Starburst_Calzetti2000', {''})
        model.dust_BC   = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM  = ('SMC_Pei92', {''})
        model.dust_BC   = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM  = ('MW_Pei92', {''})
        model.dust_BC   = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM  = ('MW_N18', {''})
        model.dust_BC   = ('MW_N18', {''})
    else: ValueError("Extinction type not recognised")

    # --- create rest-frame luminosities
    F = flare.filters.add_filters(filters, new_lam = model.lam)
    model.create_Lnu_grid(F) # --- create new L grid for each filter. In units of erg/s/Hz


    if Type == 'Total':
        tauVs_ISM   = kappa * DustSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle
        tauVs_BC    = BC_fac * (Metallicities/0.01)
        fesc        = 0.0

    elif Type == 'Pure-stellar':
        tauVs_ISM   = np.zeros(len(Masses))
        tauVs_BC    = np.zeros(len(Masses))
        fesc        = 1.0

    elif Type == 'Intrinsic':
        tauVs_ISM   = np.zeros(len(Masses))
        tauVs_BC    = np.zeros(len(Masses))
        fesc        = 0.0

    elif Type == 'Only-BC':
        tauVs_ISM   = np.zeros(len(Masses))
        tauVs_BC    = BC_fac * (Metallicities/0.01)
        fesc        = 0.0

    else:
        ValueError(F"Undefined Type {Type}")

    # Lnu = models.generate_Lnu(model, Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC, F, fesc = fesc, log10t_BC = log10t_BC)

    # --- calculate rest-frame Luminosity. In units of erg/s/Hz
    Lnu         = models.generate_Lnu(model = model, F = F, Masses = Masses, Ages = Ages,
                                      Metallicities = Metallicities, tauVs_ISM = tauVs_ISM, tauVs_BC = tauVs_BC, fesc = fesc, log10t_BC = log10t_BC)

    Lnu = list(Lnu.values())

    return Lnu


def lum_from_stars(DustSurfaceDensities, Masses, Ages, Metallicities, kappa=0.0795, BC_fac=1.0, IMF = 'Chabrier_300', filters = ['FAKE.TH.FUV'], Type = 'Total', log10t_BC = 7., extinction = 'default'):

    model = models.define_model(F'BPASSv2.2.1.binary/{IMF}') # DEFINE SED GRID -
    if extinction == 'default':
        model.dust_ISM  = ('simple', {'slope': -1.})    #Define dust curve for ISM
        model.dust_BC   = ('simple', {'slope': -1.})     #Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM  = ('Starburst_Calzetti2000', {''})
        model.dust_BC   = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM  = ('SMC_Pei92', {''})
        model.dust_BC   = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM  = ('MW_Pei92', {''})
        model.dust_BC   = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM  = ('MW_N18', {''})
        model.dust_BC   = ('MW_N18', {''})
    else: ValueError("Extinction type not recognised")

    # --- create rest-frame luminosities
    F = flare.filters.add_filters(filters, new_lam = model.lam)
    model.create_Lnu_grid(F) # --- create new L grid for each filter. In units of erg/s/Hz

    if Type == 'Total':
        tauVs_ISM   = kappa * DustSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle
        tauVs_BC    = BC_fac * (Metallicities/0.01)
        fesc        = 0.0

    elif Type == 'Pure-stellar':
        tauVs_ISM   = np.zeros(len(Masses))
        tauVs_BC    = np.zeros(len(Masses))
        fesc        = 1.0

    elif Type == 'Intrinsic':
        tauVs_ISM   = np.zeros(len(Masses))
        tauVs_BC    = np.zeros(len(Masses))
        fesc        = 0.0

    elif Type == 'Only-BC':
        tauVs_ISM   = np.zeros(len(Masses))
        tauVs_BC    = BC_fac * (Metallicities/0.01)
        fesc        = 0.0

    else:
        ValueError(F"Undefined Type {Type}")

    # --- calculate rest-frame Luminosity. In units of erg/s/Hz
    Lnu = {f: models.generate_Lnu_array(model = model, F = F, f=f, Masses = Masses, Ages = Ages, Metallicities = Metallicities, tauVs_ISM = tauVs_ISM, tauVs_BC = tauVs_BC, fesc = fesc, log10t_BC = log10t_BC) for f in filters}

    Lnu = list(Lnu.values())

    return Lnu


def get_lines(DustSurfaceDensities, Masses, Ages, Metallicities, line='HI6563', kappa=0.0795, BC_fac=1.0, IMF = 'Chabrier_300', LF = False, Type = 'Total', log10t_BC = 7., extinction = 'default', verbose=False):

    # --- calculate intrinsic quantities
    if extinction == 'default':
        dust_ISM  = ('simple', {'slope': -1.})    #Define dust curve for ISM
        dust_BC   = ('simple', {'slope': -1.})     #Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        dust_ISM  = ('Starburst_Calzetti2000', {''})
        dust_BC   = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        dust_ISM  = ('SMC_Pei92', {''})
        dust_BC   = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        dust_ISM  = ('MW_Pei92', {''})
        dust_BC   = ('MW_Pei92', {''})
    elif extinction == 'N18':
        dust_ISM  = ('MW_N18', {''})
        dust_BC   = ('MW_N18', {''})
    else: ValueError("Extinction type not recognised")


    # --- initialise model with SPS model and IMF. Set verbose = True to see a list of available lines.
    m = models.EmissionLines(F'BPASSv2.2.1.binary/{IMF}', dust_BC = dust_BC, dust_ISM = dust_ISM, verbose = False)

    if Type == 'Total':
        tauVs_ISM   = kappa * DustSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle
        tauVs_BC    = BC_fac * (Metallicities/0.01)
        fesc        = 0.0

    elif Type == 'Pure-stellar':
        tauVs_ISM   = np.zeros(len(Masses))
        tauVs_BC    = np.zeros(len(Masses))
        fesc        = 1.0

    elif Type == 'Intrinsic':
        tauVs_ISM   = np.zeros(len(Masses))
        tauVs_BC    = np.zeros(len(Masses))
        fesc        = 0.0

    elif Type == 'Only-BC':
        tauVs_ISM   = np.zeros(len(Masses))
        tauVs_BC    = BC_fac * (Metallicities/0.01)
        fesc        = 0.0

    else:
        ValueError(F"Undefined Type {Type}")


    o = m.get_line_luminosity(line, Masses, Ages, Metallicities, tauVs_BC = tauVs_BC, tauVs_ISM = tauVs_ISM, verbose = False, log10t_BC = log10t_BC)

    lum = o['luminosity']
    EW = o['EW']

    return lum, EW


def calc_axes(coods):
    """
    Args:
        coods - normed coordinates

    Returns:
        [a, b, c]:
        e_vectors:
    """

    I = np.zeros((3, 3))

    I[0,0] = np.sum(coods[:,1]**2 + coods[:,2]**2)
    I[1,1] = np.sum(coods[:,0]**2 + coods[:,2]**2)
    I[2,2] = np.sum(coods[:,1]**2 + coods[:,0]**2)

    I[0,1] = I[1,0] = - np.sum(coods[:,0] * coods[:,1])
    I[1,2] = I[2,1] = - np.sum(coods[:,2] * coods[:,1])
    I[0,2] = I[2,0] = - np.sum(coods[:,2] * coods[:,0])

    e_values, e_vectors = np.linalg.eig(I)

    sort_idx = np.argsort(e_values)

    e_values = e_values[sort_idx]
    e_vectors = e_vectors[sort_idx,:]

    a = ((5. / (2 * len(coods))) * (e_values[1] + e_values[2] - e_values[0]))**0.5
    b = ((5. / (2 * len(coods))) * (e_values[0] + e_values[2] - e_values[1]))**0.5
    c = ((5. / (2 * len(coods))) * (e_values[0] + e_values[1] - e_values[2]))**0.5

#     print a, b, c

    return [a,b,c], e_vectors


def create_image(pos, Ndim, i, j, imgrange, ls, smooth):
    # Define x and y positions for the gaussians
    Gx, Gy = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1], Ndim),
                         np.linspace(imgrange[1][0], imgrange[1][1], Ndim))
    # Initialise the image array
    gsmooth_img = np.zeros((Ndim, Ndim))
    # Loop over each star computing the smoothed gaussian distribution for this particle
    for x, y, l, sml in zip(pos[:, i], pos[:, j], ls, smooth):
        # Compute the image
        g = np.exp(-(((Gx - x) ** 2 + (Gy - y) ** 2) / (2.0 * sml ** 2)))
        # Get the sum of the gaussian
        gsum = np.sum(g)
        # If there are stars within the image in this gaussian add it to the image array
        if gsum > 0:
            gsmooth_img += g * l / gsum
    # img, xedges, yedges = np.histogram2d(pos[:, i], pos[:, j], bins=nbin, range=imgrange, weights=ls)
    return gsmooth_img

def calc_halflightradius(coords, L, sml, z):

    # Define comoving softening length in pMpc
    csoft = 0.001802390 / (0.6777 * (1 + z)) * 1e3
    # Define width (in Mpc)
    ini_width = 62
    # Compute the resolution
    ini_res = ini_width / csoft
    res = int(np.ceil(ini_res))
    # Compute the new width
    width = csoft * res

    # Define range and extent for the images
    imgrange = np.array([[-width / 2, width / 2], [-width / 2, width / 2]])
    imgextent = np.array([-width / 2, width / 2, -width / 2, width / 2])
    # Set up aperture objects
    positions = np.array([res / 2, res / 2])
    app_radii = np.linspace(0.001, res / 2, 500)  # 500 apertures out to 1/4 the image width
    # app_radii *= csoft

    tot_l = np.sum(L)
    img = create_image(coords, res, 0, 1, imgrange, L, sml)

    hlr = get_img_hlr(img, positions, tot_l, app_radii, res, csoft)

    return hlr

def get_img_hlr(img, positions, tot_l, app_rs, res, csoft):

    apertures = [CircularAperture(positions, r=r) for r in app_rs]
    # Apply the apertures
    phot_table = aperture_photometry(img, apertures, method='subpixel', subpixels=5)
    # Extract the aperture luminosities
    row = np.lib.recfunctions.structured_to_unstructured(np.array(phot_table[0]))
    lumins = row[3:]
    # Get half the total luminosity
    half_l = tot_l / 2
    # Interpolate to increase resolution
    func = interp1d(app_rs, lumins, kind="linear")
    interp_rs = np.linspace(0.001, res / 2, 1000)
    interp_lumins = func(interp_rs)
    # Get the half mass radius particle
    hlr_ind = np.argmin(np.abs(interp_lumins - half_l))
    hlr = interp_rs[hlr_ind] * csoft
    return hlr


# Disc bulge decomposition from Irodotou and Thomas 2020 obtained at https://github.com/DimitriosIrodotou/Irodotou-Thomas-2021/blob/master/IT21.py
def decomposition_IT20(sp_mass, sp_am_unit_vector):
    """
    Perform disc/spheroid decomposition of simulated galaxies based on the method introduced in Irodotou and Thomas 2020 (hereafter IT20)
    (https://ui.adsabs.harvard.edu/abs/2020arXiv200908483I/abstract).
    This function takes as arguments the mass and angular momentum of stellar particles and returns two masks that contain disc and spheroid particles
    and the disc-to-total stellar mass ratio of the galaxy.
    # Step (i) in Section 2.2 Decomposition of IT20 #
    :param sp_mass: list of stellar particles's (sp) masses.
    :param sp_am_unit_vector: list of stellar particles's (sp) normalised angular momenta (am) unit vectors.
    :return: disc_mask_IT20, spheroid_mask_IT20, disc_fraction_IT20
    """

    # Step (ii) in Section 2.2 Decomposition of IT20 #
    # Calculate the azimuth (alpha) and elevation (delta) angle of the angular momentum of all stellar particles #
    alpha = np.degrees(np.arctan2(sp_am_unit_vector[:, 1], sp_am_unit_vector[:, 0]))  # In degrees.
    delta = np.degrees(np.arcsin(sp_am_unit_vector[:, 2]))  # In degrees.

    # Step (ii) in Section 2.2 Decomposition of IT20 #
    # Generate the pixelisation of the angular momentum map #
    nside = 2 ** 4  # Define the resolution of the grid (number of divisions along the side of a base-resolution grid cell).
    hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
    indices = hp.lonlat_to_healpix(alpha * u.deg, delta * u.deg)  # Create a list of HEALPix indices from particles's alpha and delta.
    densities = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix grid cell.

    # Step (iii) in Section 2.2 Decomposition of IT20 #
    # Smooth the angular momentum map with a top-hat filter of angular radius 30 degrees #
    smoothed_densities = np.zeros(hp.npix)
    # Loop over all grid cells #
    for i in range(hp.npix):
        mask = hlp.query_disc(nside, hlp.pix2vec(nside, i), np.pi / 6.0)  # Do a 30 degree cone search around each grid cell.
        smoothed_densities[i] = np.mean(densities[mask])  # Average the densities of the ones inside and assign this value to the grid cell.

    # Step (iii) in Section 2.2 Decomposition of IT20 #
    # Find the location of the density maximum #
    index_densest = np.argmax(smoothed_densities)
    alpha_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi  # In radians.
    delta_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2  # In radians.

    # Step (iv) in Section 2.2 Decomposition of IT20 #
    # Calculate the angular separation of each stellar particle from the centre of the densest grid cell #
    Delta_theta = np.arccos(np.sin(delta_densest) * np.sin(np.radians(delta)) + np.cos(delta_densest) * np.cos(np.radians(delta)) * np.cos(
        alpha_densest - np.radians(alpha)))  # In radians.

    # Step (v) in Section 2.2 Decomposition of IT20 #
    # Calculate the disc mass fraction as the mass within 30 degrees from the densest grid cell #
    disc_mask_IT20, = np.where(Delta_theta < (np.pi / 6.0))
    spheroid_mask_IT20, = np.where(Delta_theta >= (np.pi / 6.0))
    disc_fraction_IT20 = np.sum(sp_mass[disc_mask_IT20]) / np.sum(sp_mass)

    # Step (vi) in Section 2.2 Decomposition of IT20 #
    # Normalise the disc fractions #
    chi = 0.5 * (1 - np.cos(np.pi / 6))
    disc_fraction_IT20 = np.divide(1, 1 - chi) * (disc_fraction_IT20 - chi)

    return disc_mask_IT20, spheroid_mask_IT20, disc_fraction_IT20
