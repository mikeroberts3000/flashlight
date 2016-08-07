from pylab import *

import scipy.interpolate

def interp1d_vector_wrt_scalar(t,x,kind="linear"):

    interp1d_funcs = []

    for d in range(x.shape[1]):
        interp1d_funcs.append(scipy.interpolate.interp1d(t,matrix(x[:,d]).A1,kind=kind))

    def interp1d_vector_wrt_scalar_func(t):
        t    = matrix(t).astype(float64).A1
        vals = zeros((t.shape[0],len(interp1d_funcs)))
        for d in range(len(interp1d_funcs)):
            vals[:,d] = interp1d_funcs[d](t)
        if t.shape[0] == 1:
            return matrix(vals).T
        else:
            return vals

    return interp1d_vector_wrt_scalar_func

def resample_scalar_wrt_scalar(t,x,t_new,kind="linear"):

    x_interp_func = scipy.interpolate.interp1d(t,x,kind=kind)
    x_new         = x_interp_func(t_new)

    return x_new

def resample_vector_wrt_scalar(t,x,t_new,kind="linear"):

    x_interp_func = interp1d_vector_wrt_scalar(t,x,kind=kind)
    x_new         = x_interp_func(t_new)

    return x_new
