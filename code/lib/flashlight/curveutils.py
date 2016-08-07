from pylab import *

import scipy.interpolate
import sklearn.metrics

import interpolateutils

def reparameterize_curve(p,user_progress):

    """

Reparameterizes the curve p, using the progress curve specified in
user_progress. This function assumes that the values in user_progress
are in the range [0,1].

Returns the reparameterized curve, the normalized progress values that
induce the reparameterized curve, the total length of the curve, and an
array filled with linearly spaced samples in the range [0,1] that is
same length as the reparameterized curve.

    """

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
        t_user_progress_cubic = interpolateutils.resample_scalar_wrt_scalar(l_cum_norm,t_p_linspace_norm,user_progress,kind="cubic")
    except LinAlgError:
        t_user_progress_cubic = -1.0*ones(num_samples_p)

    #
    # since l_cum_norm, t_p_linspace_norm, and user_progress all go from 0 to 1,
    # we are justified in setting the endpoints of t_user_progress to be 0 and 1
    # which isn't guaranteed due to numerical artifacts
    #
    t_user_progress_linear = interpolateutils.resample_scalar_wrt_scalar(l_cum_norm,t_p_linspace_norm,user_progress,kind="linear")

    t_user_progress             = t_user_progress_cubic
    use_linear                  = logical_or(t_user_progress_cubic < 0.0, t_user_progress_cubic > 1.0)
    t_user_progress[use_linear] = t_user_progress_linear[use_linear]

    t_user_progress[0]  = 0.0
    t_user_progress[-1] = 1.0

    if any(use_linear):
        print "flashlight.curveutils: WARNING: Using linear interpolation to reparameterize curve, for progress curve indices use_linear = %s." % str(nonzero(use_linear)[0])

    assert allclose(min(t_user_progress),0.0) and allclose(max(t_user_progress),1.0)

    t_user_progress[0]  = 0.0
    t_user_progress[-1] = 1.0

    t_user_progress               = clip(t_user_progress,0.0,1.0)
    p_user_progress               = interpolateutils.resample_vector_wrt_scalar(t_p_linspace_norm,p,t_user_progress,kind="cubic")
    t_user_progress_linspace_norm = linspace(0.0,1.0,num_samples_user_progress)

    return p_user_progress, t_user_progress, l_cum, t_user_progress_linspace_norm
