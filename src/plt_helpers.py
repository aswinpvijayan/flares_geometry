import numpy as np
import pandas as import pd
import matplotlib.pyplot as plt


def plt_hexbin(axs, x, y, z='None', cmap= , color= , extent='None', gridsize='None', func=np.median, mincnt=0.):

    if extent=='None':
        extent = [*[np.min(x[np.isfinite(x)])-0.1,np.max(x[np.isfinite(x)])+0.1], *[np.min(y[np.isfinite(y)])-0.1,np.max(y[np.isfinite(y)])+0.1]]

    if gridsize=='None':
        gridsize  = ( (np.min(x[np.isfinite(x)])-0.1+np.max(x[np.isfinite(x)])+0.1)/0.25, (np.min(y[np.isfinite(y)])-0.1,np.max(y[np.isfinite(y)])+0.1)/0.25 )

    if z!='None':
        v = [np.min(z[np.isfinite(z)])-0.1, np.max(z[np.isfinite(z)])+0.1]
        axs.hexbin(x, y, C = z, gridsize=gridsize, cmap=cmap, reduce_C_function=func, linewidths=0., mincnt=mincnt, extent=extent, vmin=v[0], vmax=v[1])

    else:
        axs.hexbin(x, y, gridsize=gridsize, cmap=cmap, linewidths=0., mincnt=mincnt, extent=extent)

    return axs



def plt_weighted(axs):

    return axs
