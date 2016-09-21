from pylab import *

import scipy.interpolate
import sklearn.metrics

import interpolate_utils

def reparameterize_curve(p,user_progress,verbose=False):

    if len(p.shape) == 1:
        p = matrix(p).T
    else:
        p = matrix(p)

    num_samples_p             = p.shape[0]
    num_samples_user_progress = user_progress.shape[0]
    num_dimensions            = p.shape[1]

    if allclose(p,0.0):
        return zeros((num_samples_user_progress,num_dimensions)), linspace(0.0,1.0,num_samples_p), 0.0, linspace(0.0,1.0,num_samples_p)

    t_p_linspace_norm = linspace(0.0,1.0,num_samples_p)
    D                 = sklearn.metrics.pairwise_distances(p,p)
    l                 = diag(D,k=1)
    l_cum             = r_[0.0,cumsum(l)]
    l_cum_norm        = l_cum / l_cum[-1]

    try:
        t_user_progress_cubic = interpolate_utils.resample_scalar_wrt_scalar(l_cum_norm,t_p_linspace_norm,user_progress,kind="cubic")
    except LinAlgError:
        t_user_progress_cubic = -1.0*ones(num_samples_p)

    # since l_cum_norm, t_p_linspace_norm, and user_progress all go from 0 to 1,
    # we are justified in setting the endpoints of t_user_progress to be 0 and 1
    # which isn't guaranteed due to numerical artifacts

    t_user_progress_linear = interpolate_utils.resample_scalar_wrt_scalar(l_cum_norm,t_p_linspace_norm,user_progress,kind="linear")

    t_user_progress             = t_user_progress_cubic
    use_linear                  = logical_or(t_user_progress_cubic < 0.0, t_user_progress_cubic > 1.0)
    t_user_progress[use_linear] = t_user_progress_linear[use_linear]

    t_user_progress[0]  = 0.0
    t_user_progress[-1] = 1.0

    if any(use_linear) and verbose:
        print "flashlight.curve: WARNING: using linear interpolation to reparameterize curve, for progress curve indices %s." % str(nonzero(use_linear)[0])

    assert allclose(min(t_user_progress),0.0) and allclose(max(t_user_progress),1.0)

    t_user_progress[0]  = 0.0
    t_user_progress[-1] = 1.0

    t_user_progress               = clip(t_user_progress,0.0,1.0)
    p_user_progress               = interpolate_utils.resample_vector_wrt_scalar(t_p_linspace_norm,p,t_user_progress,kind="cubic")
    t_user_progress_linspace_norm = linspace(0.0,1.0,num_samples_user_progress)

    return p_user_progress, t_user_progress, l_cum, t_user_progress_linspace_norm
