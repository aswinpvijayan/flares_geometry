import timeit, sys, gc
import numpy as np
import healpy as hp
import h5py
from functools import partial
from mpi4py import MPI
import schwimmbad
from astropy import units as u

sys.path.append('/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline')
sys.path.append('../src')

from helpers import get_Z_LOS, get_rotation_matrix, get_cartesian_from_spherical
import flares

conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)

def get_data(ii, tag, inp = 'FLARES', limit=500):

    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num

        sim = rF"../../flares_pipeline/data/flares.hdf5"
        num = num+'/'

    else:
        sim = rF"../../flares_pipeline/data/EAGLE_{inp}_sp_info.hdf5"
        num=''

    with h5py.File(sim, 'r') as hf:
        cop = np.array(hf[num+tag+'/Galaxy'].get('COP'), dtype = np.float32)

        S_len = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int32)
        G_len = np.array(hf[num+tag+'/Galaxy'].get('G_Length'), dtype = np.int32)

        S_coords = np.array(hf[num+tag+'/Particle'].get('S_Coordinates'), dtype = np.float32)
        G_coords = np.array(hf[num+tag+'/Particle'].get('G_Coordinates'), dtype = np.float32)
        G_mass = np.array(hf[num+tag+'/Particle'].get('G_Mass'), dtype = np.float32)*1e10
        G_sml = np.array(hf[num+tag+'/Particle'].get('G_sml'), dtype = np.float32)
        G_Z = np.array(hf[num+tag+'/Particle'].get('G_Z_smooth'), dtype = np.float32)

        S_ap = np.array(hf[num+tag+'/Particle/Apertures/Star'].get('30'), dtype = bool)
        G_ap = np.array(hf[num+tag+'/Particle/Apertures/Gas'].get('30'), dtype = bool)



    begin = np.zeros(len(S_len), dtype = np.int32)
    end = np.zeros(len(S_len), dtype = np.int32)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    gbegin = np.zeros(len(G_len), dtype = np.int32)
    gend = np.zeros(len(G_len), dtype = np.int32)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)

    n = len(S_len)
    S_len30 = np.zeros(n, dtype=np.int32)
    ptotal = 0
    for kk in range(n):
        n = np.sum(S_ap[begin[kk]:end[kk]])
        S_len30[kk] = n
        if n>=limit:
            ptotal+=n


    return cop, S_coords, G_coords, G_mass, G_sml, G_Z, begin, end, gbegin, gend, S_len, S_len30, S_ap, G_ap, ptotal



def get_ZLOS(angle, scoords, gcoords, this_gmass, this_gZ, this_gsml, lkernel, kbins):

    vector = get_cartesian_from_spherical(angle)
    rot = get_rotation_matrix(vector)
    this_scoords = (rot @ scoords.T).T
    this_gcoords = (rot @ gcoords.T).T

    Z_los_SD = get_Z_LOS(this_scoords, this_gcoords, this_gmass, this_gZ, this_gsml, lkernel, kbins) * conv #in units id Msun/pc^2

    return Z_los_SD



if __name__ == "__main__":

    ## MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start = timeit.default_timer()

    limit = 500

    ii, tag, sim_type = sys.argv[1], sys.argv[2], sys.argv[3]

    # tag='010_z005p000'
    # sim_type='FLARES'

    if rank == 0:
        print (F"Calculating line-of-sight metal densities for region {ii} for tag {tag}")

    #sph kernel approximations
    kinp = np.load('../kernel_files/kernel_sph-anarchy.npz', allow_pickle=True)
    lkernel = kinp['kernel'].astype(np.float32)
    header = kinp['header']
    kbins = header.item()['bins']

    # Generate different viewing angles
    nside=8
    hp_theta, hp_phi = hp.pix2ang(nside, range(hp.nside2npix(nside)))
    angles = np.vstack([hp_theta, hp_phi]).T.astype(np.float32)

    #For galaxies in region `num`
    num = str(ii)
    if len(num) == 1:
        num = '0'+num

    cop, S_coords, G_coords, G_mass, \
    G_sml, G_Z, begin, end, gbegin, gend, \
    S_len, S_len30, S_ap, G_ap, ptotal = get_data(num, tag, inp = 'FLARES')

    z = float(tag[5:].replace('p','.'))

    # Convert coordinates into pMpc
    cop = cop/(1+z)
    S_coords/=(1+z)
    G_coords/=(1+z)

    req_ind = np.where(S_len30>limit)[0]
    if rank == 0:
        print ("Number of selected galaxies =", len(req_ind))

        filename = F'../data1/FLARES_{num}_data.hdf5'
        print (F"Writing to {filename}")

        fl = flares.flares(fname = filename, sim_type = sim_type)
        fl.create_group(F'{tag}/Particle')

    # jj = 0
    pool = schwimmbad.MPIPool()

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    Zlos_s = np.zeros((ptotal, len(angles)), dtype=np.float64)
    icount = 0
    fcount = 0

    for kk, jj in enumerate(req_ind):

        # pool = schwimmbad.SerialPool()
        # Zlos = np.array(list(pool.map(calc_Zlos, angles)))
        # pool.close()



        start = timeit.default_timer()
        #Coordinates and attributes for the jj galaxy in ii region
        scoords = (S_coords[:, begin[jj]:end[jj]]).T[S_ap[begin[jj]:end[jj]]] - cop[:,jj]
        gcoords = (G_coords[:, gbegin[jj]:gend[jj]]).T[G_ap[gbegin[jj]:gend[jj]]] - cop[:,jj]

        this_gmass = G_mass[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
        this_gZ = G_Z[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
        this_gsml = G_sml[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]

        if rank == 0:
            print (F"Computing Zlos's for task {kk+1}/{len(req_ind)} for region {num}.....")

        calc_Zlos = partial(get_ZLOS, scoords=scoords, gcoords=gcoords, this_gmass=this_gmass, this_gZ=this_gZ, this_gsml=this_gsml, lkernel=lkernel, kbins=kbins)

        Zlos = np.array(list(pool.map(calc_Zlos, angles)))


        gc.collect()

        stop = timeit.default_timer()
        if rank == 0:
            print (F"Task {kk+1}/{len(req_ind)} for region {num} took {np.round((stop - start)/60, 3)} minutes")

            fcount+=S_len30[jj]
            Zlos_s[icount:fcount] = Zlos.T
            icount=fcount

    if rank==0:

        fl.create_dataset(Zlos_s, F'S_los', F'{tag}/Particle',
                desc = F'Star particle line-of-sight metal column density along the z-axis for different viewing angles within 30 pkpc aperture, healpy nside={nside}',
                unit = 'Msun/pc^2', dtype=np.float64)

    pool.close()
