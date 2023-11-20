import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def getLOS(np.float32_t[:,:] s_cood, np.float32_t[:,:] g_cood, np.float32_t[:] g_mass, np.float32_t[:] g_Z, np.float32_t[:] g_sml, np.float32_t[:] lkernel, np.int_t kbins):

    cdef int n = len(s_cood)
    cdef int xdir = 0
    cdef int ydir = 1
    cdef int zdir = 2
    cdef int ii, kk
    cdef float thisgsml, thisgZ, thisgmass, x, y, boverh, xx, yy, zz

    cdef np.ndarray[np.float32_t, ndim=1] Z_los_SD, thisspos
    Z_los_SD = np.zeros(n, dtype=np.float32)


    for ii in range(n):

        xx = s_cood[ii,xdir]
        yy = s_cood[ii,ydir]
        zz = s_cood[ii,zdir]
        for kk in range(len(g_cood)):

            if g_cood[kk,zdir] > zz:

                x = g_cood[kk,xdir] - xx
                y = g_cood[kk,ydir] - yy
                thisgsml = g_sml[kk]
                thisgZ = g_Z[kk]
                thisgmass = g_mass[kk]

                boverh = (x*x + y*y)**0.5 / thisgsml

                Z_los_SD[ii]+=(thisgmass*thisgZ)/(thisgsml*thisgsml) * lkernel[int(kbins*boverh)] #in units of Msun/Mpc^2

            else:
                Z_los_SD[ii]+=0.

    return Z_los_SD
