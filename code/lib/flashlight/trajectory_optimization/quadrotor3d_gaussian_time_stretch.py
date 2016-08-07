from pylab import *

import os, sys, time
import pdb
import scipy.stats
import sklearn.metrics

import pathutils
pathutils.add_relative_to_current_source_file_path_to_sys_path("..")

import flashlight.splineutils      as splineutils
import flashlight.curveutils       as curveutils
import flashlight.gradientutils    as gradientutils
import flashlight.interpolateutils as interpolateutils
import flashlight.sympyutils       as sympyutils
import flashlight.quadrotor3d      as quadrotor3d

def _get_current_vals(p_eval,psi_eval,t_current,user_progress_current,dt_current):

    p_eval_user_progress,   t_user_progress, p_eval_cumulative_length,   t_user_progress_linspace_norm = curveutils.reparameterize_curve( p_eval,   user_progress_current )
    psi_eval_user_progress, t_user_progress, psi_eval_cumulative_length, t_user_progress_linspace_norm = curveutils.reparameterize_curve( psi_eval, user_progress_current )

    p_current   = p_eval_user_progress
    psi_current = psi_eval_user_progress[:,0]

    q_q_dot_q_dot_dot_current = quadrotor3d.compute_state_space_trajectory_and_derivatives(p_current,psi_current,dt_current,check_angles=False)
    u_current                 = quadrotor3d.compute_control_trajectory(q_q_dot_q_dot_dot_current)

    p_current, p_dot_current, p_dot_dot_current, theta_current, theta_dot_current, theta_dot_dot_current, psi_current, psi_dot_current, psi_dot_dot_current, phi_current, phi_dot_current, phi_dot_dot_current = q_q_dot_q_dot_dot_current

    q_current    = c_[ p_current,     theta_current,     psi_current,     phi_current ]
    qdot_current = c_[ p_dot_current, theta_dot_current, psi_dot_current, phi_dot_current ]
    x_current    = c_[ q_current, qdot_current ]

    return p_current,psi_current,x_current,u_current

def _update_easing_curve(t_current,user_progress_current,dt_current,num_timesteps,t_begin,gauss_mean_in_terms_of_t,gauss_width_in_terms_of_dt,gauss_max_in_terms_of_dt):

    gauss_func   = scipy.stats.norm(loc=gauss_mean_in_terms_of_t,scale=dt_current*gauss_width_in_terms_of_dt)
    gauss        = gauss_func.pdf(t_current)
    gauss_scaled = (gauss / max(gauss))*dt_current*gauss_max_in_terms_of_dt

    dt_nonconst_stretched = dt_current*ones(num_timesteps) + gauss_scaled
    t_stretched           = hstack( [ t_begin, t_begin + cumsum(dt_nonconst_stretched)[:-1] ] )
    t_end_stretched       = t_stretched[-1]

    t_current             = linspace(t_begin,t_end_stretched,num_timesteps)
    user_progress_current = interpolateutils.resample_scalar_wrt_scalar(t_stretched,user_progress_current,t_current)
    dt_current            = (t_end_stretched - t_begin ) / num_timesteps

    return t_current,user_progress_current,dt_current

def optimize_feasible(p_eval,psi_eval,t_nominal,user_progress_nominal,dt_nominal,x_min_ti,x_max_ti,u_min_ti,u_max_ti,max_stretch_iters,gauss_width_in_terms_of_dt,gauss_max_in_terms_of_dt,extra_iters):

    print "flashlight.trajectory_optimization.quadrotor3d_gaussian_time_stretch: Attempting to find feasible trajectory with max_stretch_iters = %d." % max_stretch_iters

    num_timesteps         = t_nominal.shape[0]
    t_begin               = t_nominal[0]
    t_current             = t_nominal
    user_progress_current = user_progress_nominal
    dt_current            = dt_nominal
    found                 = False

    for i in range(max_stretch_iters):

        p_current,psi_current,x_current,u_current = _get_current_vals(p_eval,psi_eval,t_current,user_progress_current,dt_current)

        if all(x_current >= x_min_ti.T) and all(x_current <= x_max_ti.T) and all(u_current >= u_min_ti.T) and all(u_current <= u_max_ti.T):
            print "flashlight.trajectory_optimization.quadrotor3d_gaussian_time_stretch: Found feasible trajectory after %d smoothing iterations." % i
            found = True
            break
        else:
            x_min_constraint_violations = zeros_like(x_current)
            x_max_constraint_violations = zeros_like(x_current)
            u_min_constraint_violations = zeros_like(u_current)
            u_max_constraint_violations = zeros_like(u_current)

            x_min_constraint_violations[ x_current < x_min_ti.T ] = ( abs( x_current - x_min_ti.T.A ) / ( x_max_ti.T.A - x_min_ti.T.A ) )[ x_current < x_min_ti.T ]
            x_max_constraint_violations[ x_current > x_max_ti.T ] = ( abs( x_current - x_max_ti.T.A ) / ( x_max_ti.T.A - x_min_ti.T.A ) )[ x_current > x_max_ti.T ]
            u_min_constraint_violations[ u_current < u_min_ti.T ] = ( abs( u_current - u_min_ti.T.A ) / ( u_max_ti.T.A - u_min_ti.T.A ) )[ u_current < u_min_ti.T ]
            u_max_constraint_violations[ u_current > u_max_ti.T ] = ( abs( u_current - u_max_ti.T.A ) / ( u_max_ti.T.A - u_min_ti.T.A ) )[ u_current > u_max_ti.T ]

            x_constraint_violations = concatenate(( x_min_constraint_violations[newaxis], x_max_constraint_violations[newaxis] ))
            u_constraint_violations = concatenate(( u_min_constraint_violations[newaxis], u_max_constraint_violations[newaxis] ))

            if amax(x_constraint_violations) > amax(u_constraint_violations):
                constraint_violations = x_constraint_violations
            else:
                constraint_violations = u_constraint_violations

            t_constraint_violation_max = t_current[ unravel_index( argmax(abs(constraint_violations)), constraint_violations.shape )[1] ]
            gauss_mean_in_terms_of_t   = t_constraint_violation_max

            t_current,user_progress_current,dt_current = _update_easing_curve(t_current,user_progress_current,dt_current,num_timesteps,t_begin,gauss_mean_in_terms_of_t,gauss_width_in_terms_of_dt,gauss_max_in_terms_of_dt)

    if found:

        print "flashlight.trajectory_optimization.quadrotor3d_gaussian_time_stretch: Applying %d extra smoothing iterations to feasible trajectory." % extra_iters

        for i in range(extra_iters):
            p_current,psi_current,x_current,u_current  = _get_current_vals(p_eval,psi_eval,t_current,user_progress_current,dt_current)
            t_umax                                     = t_current[ unravel_index( argmax(abs(u_current)), u_current.shape )[0] ]
            gauss_mean_in_terms_of_t                   = t_umax
            t_current,user_progress_current,dt_current = _update_easing_curve(t_current,user_progress_current,dt_current,num_timesteps,t_begin,gauss_mean_in_terms_of_t,gauss_width_in_terms_of_dt,gauss_max_in_terms_of_dt)

        p_current,psi_current,x_current,u_current = _get_current_vals(p_eval,psi_eval,t_current,user_progress_current,dt_current)
        return x_current,u_current,t_current,user_progress_current,dt_current

    else:
        print "flashlight.trajectory_optimization.quadrotor3d_gaussian_time_stretch: Could not find feasible trajectory."
        return None



def optimize_numerically_stable_infeasible(p_eval,psi_eval,t_nominal,user_progress_nominal,dt_nominal,x_min_ti,x_max_ti,u_min_ti,u_max_ti,max_stretch_iters,gauss_width_in_terms_of_dt,gauss_max_in_terms_of_dt,max_extra_iters):

    print "flashlight.trajectory_optimization.quadrotor3d_gaussian_time_stretch: Attempting to find numerically stable trajectory with max_stretch_iters = %d." % max_stretch_iters

    num_timesteps         = t_nominal.shape[0]
    t_begin               = t_nominal[0]
    t_current             = t_nominal
    user_progress_current = user_progress_nominal
    dt_current            = dt_nominal
    found                 = False

    for i in range(max_stretch_iters):

        p_current,psi_current,x_current,u_current = _get_current_vals(p_eval,psi_eval,t_current,user_progress_current,dt_current)

        if quadrotor3d.check_differentially_flat_trajectory_numerically_stable(p_current,psi_current,dt_current):
            print "flashlight.trajectory_optimization.quadrotor3d_gaussian_time_stretch: Found numerically stable, but not neccesarily feasible, trajectory after %d smoothing iterations." % i
            found = True
            break
        else:

            udotN_current                              = gradientutils.gradients_vector_wrt_scalar_smooth_boundaries(u_current,dt=dt_current,max_gradient=2,poly_deg=5)
            udot2_current                              = udotN_current[2]
            t_udot2max                                 = t_current[ unravel_index( argmax(abs(udot2_current)), udot2_current.shape )[0] ]
            gauss_mean_in_terms_of_t                   = t_udot2max
            t_current,user_progress_current,dt_current = _update_easing_curve(t_current,user_progress_current,dt_current,num_timesteps,t_begin,gauss_mean_in_terms_of_t,gauss_width_in_terms_of_dt,gauss_max_in_terms_of_dt)

    if found:

        print "flashlight.trajectory_optimization.quadrotor3d_gaussian_time_stretch: Applying at most %d extra smoothing iterations to numerically stable trajectory." % max_extra_iters

        for i in range(max_extra_iters):

            p_current,psi_current,x_current,u_current = _get_current_vals(p_eval,psi_eval,t_current,user_progress_current,dt_current)

            if all(x_current >= x_min_ti.T) and all(x_current <= x_max_ti.T) and all(u_current >= u_min_ti.T) and all(u_current <= u_max_ti.T):
                print "flashlight.trajectory_optimization.quadrotor3d_gaussian_time_stretch: Found feasible trajectory after %d extra smoothing iterations to numerically stable trajectory." % i
                break
            else:
                t_umax                                     = t_current[ unravel_index( argmax(abs(u_current)), u_current.shape )[0] ]
                gauss_mean_in_terms_of_t                   = t_umax
                t_current,user_progress_current,dt_current = _update_easing_curve(t_current,user_progress_current,dt_current,num_timesteps,t_begin,gauss_mean_in_terms_of_t,gauss_width_in_terms_of_dt,gauss_max_in_terms_of_dt)

        p_current,psi_current,x_current,u_current = _get_current_vals(p_eval,psi_eval,t_current,user_progress_current,dt_current)
        return x_current,u_current,t_current,user_progress_current,dt_current

    else:
        print "flashlight.trajectory_optimization.quadrotor3d_gaussian_time_stretch: Could not find numerically stable trajectory."
        return None
