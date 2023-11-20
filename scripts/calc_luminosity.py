# Calculates the luminosity for individual star particles along the z los (0) and also the galaxy luminosity for different orientations

import timeit, sys, gc
from functools import partial
import numpy as np
import h5py
import schwimmbad
from mpi4py import MPI
sys.path.append('../src')
import flares
from helpers import lum, lum_from_stars, get_lines


def get_data(ii, tag, inp=0, limit=500):

    region = str(ii)
    if len(region) == 1:
        region = '0'+region

    with h5py.File(F'../../flares_pipeline/data/flares.hdf5', 'r') as hf:

        S_len       = np.array(hf[region+'/'+tag+'/Galaxy'].get('S_Length'), dtype=np.int32)
        DTM         = np.array(hf[region+'/'+tag+'/Galaxy'].get('DTM'), dtype=np.float32)

        S_ap        = np.array(hf[region+'/'+tag+'/Particle/Apertures/Star'].get('30'), dtype=bool)
        S_mass      = np.array(hf[region+'/'+tag+'/Particle'].get('S_MassInitial'), dtype=np.float64)*1e10
        S_Z         = np.array(hf[region+'/'+tag+'/Particle'].get('S_Z_smooth'), dtype=np.float64)
        S_age       = np.array(hf[region+'/'+tag+'/Particle'].get('S_Age'), dtype=np.float64)*1e3

        if inp==0:
            S_los = np.array(hf[region+'/'+tag+'/Particle'].get('S_los'), dtype=np.float64)
        else:
            with h5py.File(F'../data1/FLARES_{region}_data.hdf5', 'r') as hf1:
                S_los = np.array(hf1[tag+'/Particle'].get('S_los'), dtype=np.float64)




    n = len(S_len)
    begin = np.zeros(len(S_len), dtype=np.int32)
    end = np.zeros(len(S_len), dtype=np.int32)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)
    begin = begin.astype(np.int32)
    end = end.astype(np.int32)
    S_len30 = np.zeros(n, dtype=np.int32)
    for kk in range(n):
        S_len30[kk] = np.sum(S_ap[begin[kk]:end[kk]])

    ok = np.where(S_len30>limit)[0]


    m = np.sum(S_len30[ok])
    this_smass = np.zeros(m, dtype=np.float64)
    this_sz = np.zeros(m, dtype=np.float64)
    this_sage = np.zeros(m, dtype=np.float64)
    if inp==0:
        this_slos = np.zeros(m, dtype=np.float64)
    else:
        this_slos = S_los
    # begin[1:] = np.cumsum(S_len30)[:-1]
    # end = np.cumsum(S_len30)
    # begin = begin.astype(np.int32)
    # end = end.astype(np.int32)

    inicount = 0
    fincount = 0

    for kk in ok:
        fincount+=S_len30[kk]
        this_smass[inicount:fincount]   = S_mass[begin[kk]:end[kk]][S_ap[begin[kk]:end[kk]]]
        this_sz[inicount:fincount]      = S_Z[begin[kk]:end[kk]][S_ap[begin[kk]:end[kk]]]
        this_sage[inicount:fincount]    = S_age[begin[kk]:end[kk]][S_ap[begin[kk]:end[kk]]]
        if inp==0:
            this_slos[inicount:fincount]    = S_los[begin[kk]:end[kk]][S_ap[begin[kk]:end[kk]]]
        inicount=fincount

    begin = np.zeros(len(ok), dtype=np.int32)
    end = np.zeros(len(ok), dtype=np.int32)
    begin[1:] = np.cumsum(S_len30[ok])[:-1]
    end = np.cumsum(S_len30[ok])
    begin = begin.astype(np.int32)
    end = end.astype(np.int32)


    return DTM[ok], this_smass, this_sz, this_sage, this_slos, begin, end, ok



if __name__ == "__main__":

    ## MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    start = timeit.default_timer()

    ii, tag, inp = sys.argv[1], sys.argv[2], int(sys.argv[3])
    num = str(ii)
    if len(num) == 1:
        num = '0'+num

    model = {'Intrinsic':'Intrinsic', 'DustModelI':'Total'}
    # model = {'No_ISM':'Only-BC'}
    limit=500
    DTM, S_mass, S_Z, S_age, S_los, begin, end, ok = get_data(num, tag, inp, limit)



    filename = F'../data/FLARES_{num}_data.hdf5'
    fl = flares.flares(fname = filename, sim_type = 'FLARES')



    if inp==0:

        # with h5py.File(F'../../flares_pipeline/data/flares.hdf5', 'r') as hf:
        #
        #     LFUVint = np.array(hf[num+'/'+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float32)[ok]
        #     LFUVatt = np.array(hf[num+'/'+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float32)[ok]

        # LFUVint1 = np.zeros(len(begin))
        # LFUVatt1 = np.zeros(len(begin))

        print (F'Calculating star particle photometry along z-axis for {num} and {tag}')

        DustSurfaceDensities = np.zeros(len(S_mass))

        for ii, jj in enumerate(begin):
            DustSurfaceDensities[jj:end[ii]] = DTM[ii] * S_los[jj:end[ii]]

        for ii in model:

            lum = lum_from_stars(DustSurfaceDensities, S_mass, S_age, S_Z, filters = ['FAKE.TH.U', 'FAKE.TH.V', 'FAKE.TH.J'], Type = model[ii])

            fl.create_group(F'{tag}/Particle/BPASS_2.2.1/Chabrier300/zaxis/Luminosity/{ii}')

            for jj, kk in enumerate(['U', 'V', 'J']):
                print (ii, model[ii], jj, kk)
                fl.create_dataset(lum[jj], F'{kk}', F'{tag}/Particle/BPASS_2.2.1/Chabrier300/zaxis/Luminosity/{ii}',
                    desc = F'Star particle {kk} luminosity along the z-axis within 30 pkpc aperture',
                    unit = 'erg/s/Hz', dtype=np.float64, overwrite=True)

        # for ii, jj in enumerate(begin):
        #     LFUVint1[ii] = np.sum(lum['Intrinsic'][0][begin[ii]:end[ii]])
        #     LFUVatt1[ii] = np.sum(lum['DustModelI'][0][begin[ii]:end[ii]])

    elif inp==1:

        if rank==0:
            print (F'Calculating select photometry for different lines of sight for {num} and {tag}')


        # pool = schwimmbad.MPIPool()
        #
        # if not pool.is_master():
        #     pool.wait()
        #     sys.exit(0)
        #

        lines = ['HI6563', 'HI4861', 'OIII4959', 'OIII5007']


        nn      = np.shape(S_los)[1]
        part    = int(np.ceil(nn/size))

        comm.Barrier()
        if rank!=size-1:
            this_slos   = [rank*part,(rank+1)*part]
            los_nn      = part
        else:
            this_slos   = [rank*part,nn]
            los_nn      = nn - rank*part


        #Only need to calculate the dust attenuated values
        FUV        = np.zeros((len(DTM), los_nn))
        NUV        = np.zeros((len(DTM), los_nn))

        HI6563_lum      = np.zeros((len(DTM), los_nn))
        HI6563_EW       = np.zeros((len(DTM), los_nn))
        HI4861_lum      = np.zeros((len(DTM), los_nn))
        HI4861_EW       = np.zeros((len(DTM), los_nn))
        OIII4959_lum    = np.zeros((len(DTM), los_nn))
        OIII4959_EW     = np.zeros((len(DTM), los_nn))
        OIII5007_lum    = np.zeros((len(DTM), los_nn))
        OIII5007_EW     = np.zeros((len(DTM), los_nn))


        pool    = schwimmbad.MultiPool(processes=2)

        for ii, jj in enumerate(begin):

            DustSurfaceDensities = DTM[ii] * S_los[jj:end[ii]][:,this_slos[0]:this_slos[1]]

            Masses              = S_mass[jj:end[ii]]
            Ages                = S_age[jj:end[ii]]
            Metallicities       = S_Z[jj:end[ii]]

            func_lum    = partial(lum, Masses=Masses, Ages=Ages, Metallicities=Metallicities, filters=['FAKE.TH.FUV', 'FAKE.TH.NUV'])

            dat_lum     = np.array(list(pool.map(func_lum, DustSurfaceDensities.T)))

            FUV[ii]    = dat_lum[:,0]
            NUV[ii]    = dat_lum[:,1]

            for line in lines:
                func_line   = partial(get_lines, line=line, Masses=Masses, Ages=Ages, Metallicities=Metallicities)
                dat_line    = np.array(list(pool.map(func_line, DustSurfaceDensities.T)))

                vars()[line+'_lum'][ii] = dat_line[:,0]
                vars()[line+'_EW'][ii]  = dat_line[:,1]


        pool.close()

        comm.Barrier()
        gc.collect()

        if rank == 0:
            print ("Gathering data from different processes")

        FUVs = comm.gather(FUV, root=0)
        del FUV
        NUVs = comm.gather(NUV, root=0)
        del NUV
        for line in lines:
            vars()[line+'_lums'] = comm.gather(vars()[line+'_lum'], root=0)
            vars()[line+'_EWs']  = comm.gather(vars()[line+'_EW'], root=0)

        gc.collect()


        if rank == 0:

            print ("Gathering completed")

            FUVs = np.hstack(FUVs)
            NUVs = np.hstack(NUVs)

            for line in lines:
                vars()[line+'_lums'] = np.hstack(vars()[line+'_lums'])
                vars()[line+'_EWs']  = np.hstack(vars()[line+'_EWs'])


        if rank==0:

            # np.savez(F'test_{tag}', FUVs=FUVs, HI6563_lums=HI6563_lums, HI6563_EWs=HI6563_EWs)
            # import pickle
            # data = {'FUVs':FUVs, 'HI6563_lums':HI6563_lums, 'HI6563_EWs':HI6563_EWs}
            # a_file = open("data1.pkl", "wb")
            # pickle.dump(data, a_file)
            # a_file.close()

            fl.create_group(F'{tag}/Galaxy/BPASS_2.2.1/Chabrier300/los/Luminosity/DustModelI')
            fl.create_dataset(FUVs, 'FUV', F'{tag}/Galaxy/BPASS_2.2.1/Chabrier300/los/Luminosity/DustModelI',
                desc = F'Galaxy dust attenuated FUV luminosity along different los within 30 pkpc aperture',
                unit = 'erg/s/Hz', dtype=np.float64, overwrite=True)
            fl.create_dataset(NUVs, 'NUV', F'{tag}/Galaxy/BPASS_2.2.1/Chabrier300/los/Luminosity/DustModelI',
                desc = F'Galaxy dust attenuated FUV luminosity along different los within 30 pkpc aperture',
                unit = 'erg/s/Hz', dtype=np.float64, overwrite=True)


            fl.create_group(F'{tag}/Galaxy/BPASS_2.2.1/Chabrier300/los/Lines/DustModelI')
            for line in lines:
                fl.create_dataset(vars()[line+'_lums'], f'{line}/Luminosity', F'{tag}/Galaxy/BPASS_2.2.1/Chabrier300/los/Lines/DustModelI',
                    desc = F'Galaxy dust attenuated {line} luminosity along different los within 30 pkpc aperture',
                    unit = 'erg/s/Hz', dtype=np.float64, overwrite=True)

                fl.create_dataset(vars()[line+'_EWs'], f'{line}/EW', F'{tag}/Galaxy/BPASS_2.2.1/Chabrier300/los/Lines/DustModelI',
                    desc = F'Galaxy dust attenuated {line} EWs along different los within 30 pkpc aperture',
                    unit = 'Angstrom', dtype=np.float64, overwrite=True)

            fl.create_group(F'{tag}/Galaxy/BPASS_2.2.1/Chabrier300/los/Indices/DustModelI')
            beta = (np.log10(FUVs/NUVs)/np.log10(1500/2500))-2.0
            fl.create_dataset(beta, 'beta', F'{tag}/Galaxy/BPASS_2.2.1/Chabrier300/los/Indices/DustModelI',
                desc = F'Galaxy dust attenuated UV-continuum slope along different los within 30 pkpc aperture',
                unit = 'No unit', dtype=np.float32, overwrite=True)
