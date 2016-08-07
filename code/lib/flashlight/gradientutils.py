from pylab import *

def gradient_scalar_wrt_scalar(x,dt):

    return gradient(x,dt)

def gradient_scalar_wrt_scalar_nonconst_dt(x,t): 

    x            = x.astype(float64).squeeze()
    t            = t.astype(float64).squeeze()
    x_grad       = zeros_like(x)
    x_grad[0]    = (x[1]  - x[0])   / (t[1]  - t[0])
    x_grad[-1]   = (x[-1] - x[-2])  / (t[-1] - t[-2])
    x_grad[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])

    return x_grad

def gradients_scalar_wrt_scalar_smooth_boundaries(x,dt,max_gradient,poly_deg):

    num_ext_points = max_gradient
    t              = arange(x.shape[0])*dt

    x_poly_l     = polyfit(t[:(poly_deg+1)],  x[:(poly_deg+1)],  deg=poly_deg)
    x_poly_r     = polyfit(t[-(poly_deg+1):], x[-(poly_deg+1):], deg=poly_deg)
    t_ext_only_l = (t[0]  - cumsum([dt]*num_ext_points))[::-1]
    t_ext_only_r = (t[-1] + cumsum([dt]*num_ext_points))
    x_ext_only_l = polyval(x_poly_l, t_ext_only_l)
    x_ext_only_r = polyval(x_poly_r, t_ext_only_r)

    te = hstack([t_ext_only_l,t,t_ext_only_r])
    xe = hstack([x_ext_only_l,x,x_ext_only_r])

    xe_grads      = zeros((max_gradient+1,xe.shape[0]))
    xe_grads[0,:] = xe

    for i in range(1,max_gradient+1):
        xe_grads[i,:] = gradient(xe_grads[i-1,:],dt)

    return xe_grads[:,num_ext_points:-num_ext_points]

def gradients_scalar_wrt_scalar_smooth_boundaries_nonconst_dt(x,t,max_gradient,poly_deg):

    num_ext_points = max_gradient

    dt_l         = diff(t)[0]
    dt_r         = diff(t)[-1]
    x_poly_l     = polyfit(t[:(poly_deg+1)],  x[:(poly_deg+1)],  deg=poly_deg)
    x_poly_r     = polyfit(t[-(poly_deg+1):], x[-(poly_deg+1):], deg=poly_deg)
    t_ext_only_l = (t[0]  - cumsum([dt_l]*num_ext_points))[::-1]
    t_ext_only_r = (t[-1] + cumsum([dt_r]*num_ext_points))
    x_ext_only_l = polyval(x_poly_l, t_ext_only_l)
    x_ext_only_r = polyval(x_poly_r, t_ext_only_r)

    te = hstack([t_ext_only_l,t,t_ext_only_r])
    xe = hstack([x_ext_only_l,x,x_ext_only_r])

    xe_grads      = zeros((max_gradient+1,xe.shape[0]))
    xe_grads[0,:] = xe

    for i in range(1,max_gradient+1):
        xe_grads[i,:] = gradient_scalar_wrt_scalar_nonconst_dt(xe_grads[i-1,:],te)

    return xe_grads[:,num_ext_points:-num_ext_points]

def gradients_scalar_wrt_scalar_smooth_boundaries_forward_diffs(x,dt,max_gradient,poly_deg):

    num_ext_points = max_gradient
    t              = arange(x.shape[0])*dt

    x_poly_r     = polyfit(t[-(poly_deg+1):], x[-(poly_deg+1):], deg=poly_deg)
    t_ext_only_r = (t[-1] + cumsum([dt]*num_ext_points))
    x_ext_only_r = polyval(x_poly_r, t_ext_only_r)

    te = hstack([t,t_ext_only_r])
    xe = hstack([x,x_ext_only_r])

    xe_grads      = zeros((max_gradient+1,xe.shape[0]))
    xe_grads[0,:] = xe

    for i in range(1,max_gradient+1):
        xe_grads[i,:-1] = diff(xe_grads[i-1,:])/dt

    return xe_grads[:,:-num_ext_points]

def gradient_vector_wrt_scalar(x,dt):

    x_grad = zeros_like(x)

    for d in range(x.shape[1]):
        x_grad[:,d] = gradient(x[:,d],dt)

    return x_grad

def gradient_vector_wrt_scalar_nonconst_dt(x,t):

    x_grad = zeros_like(x)

    for d in range(x.shape[1]):
        x_grad[:,d] = gradient_scalar_wrt_scalar_nonconst_dt(x[:,d],t)

    return x_grad

def gradients_vector_wrt_scalar_smooth_boundaries(x,dt,max_gradient,poly_deg):

    x_grads = zeros((max_gradient+1,x.shape[0],x.shape[1]))

    for d in range(x.shape[1]):
        x_grads[:,:,d] = gradients_scalar_wrt_scalar_smooth_boundaries(x[:,d],dt,max_gradient,poly_deg)

    return x_grads

def gradients_vector_wrt_scalar_smooth_boundaries_nonconst_dt(x,t,max_gradient,poly_deg):

    x_grads = zeros((max_gradient+1,x.shape[0],x.shape[1]))

    for d in range(x.shape[1]):
        x_grads[:,:,d] = gradients_scalar_wrt_scalar_smooth_boundaries_nonconst_dt(x[:,d],t,max_gradient,poly_deg)

    return x_grads

def gradients_vector_wrt_scalar_smooth_boundaries_forward_diffs(x,dt,max_gradient,poly_deg):

    x_grads = zeros((max_gradient+1,x.shape[0],x.shape[1]))

    for d in range(x.shape[1]):
        x_grads[:,:,d] = gradients_scalar_wrt_scalar_smooth_boundaries_forward_diffs(x[:,d],dt,max_gradient,poly_deg)

    return x_grads
