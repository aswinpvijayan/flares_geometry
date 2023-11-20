import sys
sys.path.append('../src')
import numpy as np
import h5py

from helpers import ang_mom_vector, get_spherical_from_cartesian, compute_kappa

def get_kappa(ii, tag, z, limit=500, dl=1e30):

    num = str(ii)
    z = float(tag[5:].replace('p','.'))

    if len(num) == 1:
        num =  '0'+num

    sim = rF"../../flares_pipeline/data/flares.hdf5"
    num = num+'/'

    with h5py.File(sim, 'r') as hf:
        S_len       = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)
        G_len       = np.array(hf[num+tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        S_ap        = np.array(hf[num+'/'+tag+'/Particle/Apertures/Star'].get('30'), dtype=bool)
        G_ap        = np.array(hf[num+'/'+tag+'/Particle/Apertures/Gas'].get('30'), dtype=bool)

        cop = np.array(hf[num+tag+'/Galaxy'].get('COP'), dtype = np.float64)
        cop_vel = np.array(hf[num+tag+'/Galaxy'].get('Velocity'), dtype = np.float64)

        S_mass_curr = np.array(hf[num+tag+'/Particle'].get('S_Mass'), dtype = np.float64)*1e10
        G_mass = np.array(hf[num+tag+'/Particle'].get('G_Mass'), dtype = np.float64)*1e10
        S_coords = np.array(hf[num+tag+'/Particle'].get('S_Coordinates'), dtype = np.float64)
        G_coords = np.array(hf[num+tag+'/Particle'].get('G_Coordinates'), dtype = np.float64)
        S_vel = np.array(hf[num+tag+'/Particle'].get('S_Vel'), dtype = np.float64)
        G_vel = np.array(hf[num+tag+'/Particle'].get('G_Vel'), dtype = np.float64)



    cop = cop/(1+z)
    S_coords/=(1+z)
    G_coords/=(1+z)

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

    Kco     = np.zeros(len(ok))
    Krot    = np.zeros(len(ok))

    ang_vector_stars    = np.zeros((len(ok),3))
    ang_vector_sp_stars = np.zeros((len(ok),3))


    inicount = 0
    fincount = 0

    for jj, kk in enumerate(ok):
        # fincount+=S_len30[kk]

        mass    = S_mass_curr[begin[kk]:end[kk]][S_ap[begin[kk]:end[kk]]]
        coords  = 1e3*(S_coords[:, begin[kk]:end[kk]].T[S_ap[begin[kk]:end[kk]]] - cop[:,kk])
        vel     = S_vel[:, begin[kk]:end[kk]].T[S_ap[begin[kk]:end[kk]]] - cop_vel[:,kk]

        Kco[jj], Krot[jj]       = compute_kappa(mass, coords, vel, z)
        ang_vector_stars[jj]    = ang_mom_vector(mass, coords, vel, z)
        ang_vector_sp_stars[jj] = get_spherical_from_cartesian(ang_vector_stars[jj])

        # inicount=fincount

    Kco_gas     = np.zeros(len(ok))
    Krot_gas    = np.zeros(len(ok))

    ang_vector_gas    = np.zeros((len(ok),3))
    ang_vector_sp_gas = np.zeros((len(ok),3))


    inicount = 0
    fincount = 0

    for jj, kk in enumerate(ok):
        # fincount+=G_len30[kk]

        mass    = G_mass[gbegin[kk]:gend[kk]][G_ap[gbegin[kk]:gend[kk]]]
        coords  = 1e3*(G_coords[:, gbegin[kk]:gend[kk]].T[G_ap[gbegin[kk]:gend[kk]]] - cop[:,kk])
        vel     = G_vel[:, gbegin[kk]:gend[kk]].T[G_ap[gbegin[kk]:gend[kk]]] - cop_vel[:,kk]

        Kco_gas[jj], Krot_gas[jj]   = compute_kappa(mass, coords, vel, z)
        ang_vector_gas[jj]          = ang_mom_vector(mass, coords, vel, z)
        ang_vector_sp_gas[jj]       = get_spherical_from_cartesian(ang_vector_gas[jj])

        # inicount=fincount

    return Kco, Krot, ang_vector_stars, ang_vector_sp_stars, Kco_gas, Krot_gas, ang_vector_gas, ang_vector_sp_gas
