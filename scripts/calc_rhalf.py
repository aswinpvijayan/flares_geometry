# Calculates the luminosity for individual star particles along the z los (0) and also the galaxy luminosity for different orientations

import timeit, sys, gc
import numpy as np
import h5py
import sys
sys.path.append('../src')

from scipy.interpolate import CubicSpline, interp1d
import flares

norm = np.linalg.norm

def get_data(ii, tag, limit=1000):

    z = float(tag[5:].replace('p','.'))

    region = str(ii)
    if len(region) == 1:
        region = '0'+region

    with h5py.File(F'../../flares_pipeline/data/flares.hdf5', 'r') as hf:

        cop         = np.array(hf[region+'/'+tag+'/Galaxy'].get('COP'), dtype=np.float64).T/(1.+z)
        S_len       = np.array(hf[region+'/'+tag+'/Galaxy'].get('S_Length'), dtype=np.int32)
        G_len       = np.array(hf[region+'/'+tag+'/Galaxy'].get('G_Length'), dtype=np.int32)
        DTM         = np.array(hf[region+'/'+tag+'/Galaxy'].get('DTM'), dtype=np.float32)

        S_ap        = np.array(hf[region+'/'+tag+'/Particle/Apertures/Star'].get('30'), dtype=bool)
        S_coord     = np.array(hf[region+'/'+tag+'/Particle'].get('S_Coordinates'), dtype=np.float64).T[S_ap]/(1.+z)
        S_mass      = np.array(hf[region+'/'+tag+'/Particle'].get('S_Mass'), dtype=np.float64)[S_ap]*1e10

        G_ap        = np.array(hf[region+'/'+tag+'/Particle/Apertures/Gas'].get('30'), dtype=bool)
        G_coord     = np.array(hf[region+'/'+tag+'/Particle'].get('G_Coordinates'), dtype=np.float64).T[G_ap]/(1.+z)
        G_mass      = np.array(hf[region+'/'+tag+'/Particle'].get('G_Mass'), dtype=np.float64)[G_ap]*1e10
        G_Z         = np.array(hf[region+'/'+tag+'/Particle'].get('G_Z_smooth'), dtype=np.float64)[G_ap]




    n = len(S_len)
    begin = np.zeros(n, dtype=np.int32)
    end = np.zeros(n, dtype=np.int32)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)
    begin = begin.astype(np.int32)
    end = end.astype(np.int32)

    gbegin = np.zeros(n, dtype=np.int32)
    gend = np.zeros(n, dtype=np.int32)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)
    gbegin = gbegin.astype(np.int32)
    gend = gend.astype(np.int32)


    S_len30 = np.zeros(n, dtype=np.int32)
    G_len30 = np.zeros(n, dtype=np.int32)
    for kk in range(n):
        S_len30[kk] = np.sum(S_ap[begin[kk]:end[kk]])
        G_len30[kk] = np.sum(G_ap[gbegin[kk]:gend[kk]])


    ok = np.where(S_len30>limit)
    DTM = DTM[ok]
    cop = cop[ok]
    n = np.sum(S_len30[ok])
    m = np.sum(G_len30[ok])

    this_scoord     = np.zeros((n,3), dtype=np.float32)
    this_smass      = np.zeros(n, dtype=np.float32)

    this_gcoord     = np.zeros((m,3), dtype=np.float32)
    this_gmass      = np.zeros(m, dtype=np.float32)
    this_gZ         = np.zeros(m, dtype=np.float32)


    begin[1:] = np.cumsum(S_len30)[:-1]
    end = np.cumsum(S_len30)
    begin = begin.astype(np.int32)
    end = end.astype(np.int32)

    gbegin[1:] = np.cumsum(G_len30)[:-1]
    gend = np.cumsum(G_len30)
    gbegin = gbegin.astype(np.int32)
    gend = gend.astype(np.int32)

    inicount = 0
    fincount = 0

    ginicount = 0
    gfincount = 0

    for jj, kk in enumerate(ok[0]):
        fincount+=S_len30[kk]
        this_scoord[inicount:fincount]      = S_coord[begin[kk]:end[kk]] - cop[jj]
        this_smass[inicount:fincount]       = S_mass[begin[kk]:end[kk]]
        inicount=fincount

        gfincount+=G_len30[kk]
        this_gcoord[ginicount:gfincount]    = G_coord[gbegin[kk]:gend[kk]] - cop[jj]
        this_gmass[ginicount:gfincount]     = G_mass[gbegin[kk]:gend[kk]]
        this_gZ[ginicount:gfincount]        = G_Z[gbegin[kk]:gend[kk]]
        ginicount=gfincount



    begin = np.zeros(len(ok[0]), dtype=np.int32)
    end = np.zeros(len(ok[0]), dtype=np.int32)
    begin[1:] = np.cumsum(S_len30[ok])[:-1]
    end = np.cumsum(S_len30[ok])
    begin = begin.astype(np.int32)
    end = end.astype(np.int32)

    gbegin = np.zeros(len(ok[0]), dtype=np.int32)
    gend = np.zeros(len(ok[0]), dtype=np.int32)
    gbegin[1:] = np.cumsum(G_len30[ok])[:-1]
    gend = np.cumsum(G_len30[ok])
    gbegin = gbegin.astype(np.int32)
    gend = gend.astype(np.int32)


    return DTM, this_scoord, this_smass, this_gcoord, this_gmass, this_gZ, begin, end, gbegin, gend, S_len, S_len30, G_len, G_len30, ok


def calc_rhalf(coords, mass, Z=np.ones(1), DTM=1.):

    dist = norm(coords, axis=1)
    sort_order = np.argsort(dist)
    sorted_dist = dist[sort_order]

    if len(Z)>1:
        sorted_mass = mass[sort_order]*Z[sort_order]*DTM
    else:
        sorted_mass = mass[sort_order]

    fracmass = np.cumsum(sorted_mass)/np.sum(sorted_mass)

    try:
        interp_func = interp1d(fracmass, sorted_dist)
        rhalf = interp_func(0.5)

    except:
        rhalf = 0

    return rhalf


if __name__ == "__main__":

    start = timeit.default_timer()

    ii, tag = sys.argv[1], sys.argv[2]
    num = str(ii)
    if len(num) == 1:
        num = '0'+num

    limit=500
    DTM, this_scoord, this_smass, this_gcoord, this_gmass, this_gZ, begin, end, gbegin, gend, S_len, S_len30, G_len, G_len30, ok = get_data(num, tag, limit)

    filename = F'../data1/FLARES_{num}_data.hdf5'
    fl = flares.flares(fname = filename, sim_type = 'FLARES')

    fl.create_group(F'{tag}/Galaxy')

    #Calculating half-mass radii
    stellar_rhalf = np.zeros(len(begin))
    dust_rhalf = np.zeros(len(begin))

    for ii, jj in enumerate(begin):

        #Stellar half-mass radii
        stellar_rhalf[ii] = calc_rhalf(this_scoord[begin[ii]:end[ii]], this_smass[begin[ii]:end[ii]])*1e3

        #Dust half-mass radii
        dust_rhalf[ii] = calc_rhalf(this_gcoord[gbegin[ii]:gend[ii]], this_gmass[gbegin[ii]:gend[ii]], this_gZ[gbegin[ii]:gend[ii]], DTM[ii])*1e3


    fl.create_group(F'{tag}/Galaxy/HalfMassRadii')

    fl.create_dataset(stellar_rhalf, F'Star', F'{tag}/Galaxy/HalfMassRadii',
        desc = F'Stellar half-mass radius within 30 pkpc aperture',
        unit = 'pkpc', dtype=np.float32, overwrite=True)

    fl.create_dataset(dust_rhalf, F'Dust', F'{tag}/Galaxy/HalfMassRadii',
        desc = F'Dust half-mass radius within 30 pkpc aperture',
        unit = 'pkpc', dtype=np.float32, overwrite=True)
