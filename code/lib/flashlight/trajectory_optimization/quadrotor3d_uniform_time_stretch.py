from pylab import *

import os, sys, time
import pdb
import sklearn.metrics

import pathutils
pathutils.add_relative_to_current_source_file_path_to_sys_path("..")

import flashlight.splineutils      as splineutils
import flashlight.curveutils       as curveutils
import flashlight.gradientutils    as gradientutils
import flashlight.interpolateutils as interpolateutils
import flashlight.sympyutils       as sympyutils
import flashlight.quadrotor3d      as quadrotor3d

def optimize_feasible(p_nominal,psi_nominal,dt_nominal,x_min_ti,x_max_ti,u_min_ti,u_max_ti,max_bin_search_iters,dt_upper_init):

    print "flashlight.trajectory_optimization.quadrotor3d_uniform_time_stretch: Attempting to find feasible trajectory with max_bin_search_iters = %d, dt_upper_init = %f." % (max_bin_search_iters,dt_upper_init)

    psi_nominal = matrix(psi_nominal).A1

    dt_lower   = dt_nominal
    dt_upper   = dt_upper_init
    dt_current = dt_nominal

    # if lower bound satisfies constraints, return
    q_q_dot_q_dot_dot_lower = quadrotor3d.compute_state_space_trajectory_and_derivatives(p_nominal,psi_nominal,dt_lower,check_angles=False)
    u_lower                 = quadrotor3d.compute_control_trajectory(q_q_dot_q_dot_dot_lower)

    p_lower, p_dot_lower, p_dot_dot_lower, theta_lower, theta_dot_lower, theta_dot_dot_lower, psi_lower, psi_dot_lower, psi_dot_dot_lower, phi_lower, phi_dot_lower, phi_dot_dot_lower = q_q_dot_q_dot_dot_lower

    q_lower    = c_[ p_lower,     theta_lower,     psi_lower,     phi_lower ]
    qdot_lower = c_[ p_dot_lower, theta_dot_lower, psi_dot_lower, phi_dot_lower ]
    x_lower    = c_[ q_lower, qdot_lower ]

    if all(x_lower >= x_min_ti.T) and all(x_lower <= x_max_ti.T) and all(u_lower >= u_min_ti.T) and all(u_lower <= u_max_ti.T):
        print "flashlight.trajectory_optimization.quadrotor3d_uniform_time_stretch: Input trajectory is already feasible."
        return x_lower,u_lower,1.0

    # assert upper bound satisfies constraints
    q_q_dot_q_dot_dot_upper = quadrotor3d.compute_state_space_trajectory_and_derivatives(p_nominal,psi_nominal,dt_upper,check_angles=False)
    u_upper                 = quadrotor3d.compute_control_trajectory(q_q_dot_q_dot_dot_upper)

    p_upper, p_dot_upper, p_dot_dot_upper, theta_upper, theta_dot_upper, theta_dot_dot_upper, psi_upper, psi_dot_upper, psi_dot_dot_upper, phi_upper, phi_dot_upper, phi_dot_dot_upper = q_q_dot_q_dot_dot_upper

    q_upper    = c_[ p_upper,     theta_upper,     psi_upper,     phi_upper ]
    qdot_upper = c_[ p_dot_upper, theta_dot_upper, psi_dot_upper, phi_dot_upper ]
    x_upper    = c_[ q_upper, qdot_upper ]

    assert all(x_upper >= x_min_ti.T) and all(x_upper <= x_max_ti.T) and all(u_upper >= u_min_ti.T) and all(u_upper <= u_max_ti.T)

    # binary search
    for i in range(max_bin_search_iters):

        dt_current = (dt_lower+dt_upper)/2.0

        q_q_dot_q_dot_dot_current = quadrotor3d.compute_state_space_trajectory_and_derivatives(p_nominal,psi_nominal,dt_current,check_angles=False)
        u_current                 = quadrotor3d.compute_control_trajectory(q_q_dot_q_dot_dot_current)

        p_current, p_dot_current, p_dot_dot_current, theta_current, theta_dot_current, theta_dot_dot_current, psi_current, psi_dot_current, psi_dot_dot_current, phi_current, phi_dot_current, phi_dot_dot_current = q_q_dot_q_dot_dot_current

        q_current    = c_[ p_current,     theta_current,     psi_current,     phi_current ]
        qdot_current = c_[ p_dot_current, theta_dot_current, psi_dot_current, phi_dot_current ]
        x_current    = c_[ q_current, qdot_current ]

        if all(x_current >= x_min_ti.T) and all(x_current <= x_max_ti.T) and all(u_current >= u_min_ti.T) and all(u_current <= u_max_ti.T):
            dt_upper = dt_current
        else:
            dt_lower = dt_current

    # return smallest feasible dt from binary search
    dt_current                = dt_upper
    q_q_dot_q_dot_dot_current = quadrotor3d.compute_state_space_trajectory_and_derivatives(p_nominal,psi_nominal,dt_current,check_angles=False)
    u_current                 = quadrotor3d.compute_control_trajectory(q_q_dot_q_dot_dot_current)

    p_current, p_dot_current, p_dot_dot_current, theta_current, theta_dot_current, theta_dot_dot_current, psi_current, psi_dot_current, psi_dot_dot_current, phi_current, phi_dot_current, phi_dot_dot_current = q_q_dot_q_dot_dot_current

    q_current        = c_[ p_current,     theta_current,     psi_current,     phi_current ]
    qdot_current     = c_[ p_dot_current, theta_dot_current, psi_dot_current, phi_dot_current ]
    x_current        = c_[ q_current, qdot_current ]
    dt_scale_current = dt_current/dt_nominal

    assert all(x_current >= x_min_ti.T) and all(x_current <= x_max_ti.T) and all(u_current >= u_min_ti.T) and all(u_current <= u_max_ti.T)

    return x_current,u_current,dt_scale_current



def optimize_numerically_stable_infeasible(p_nominal,psi_nominal,dt_nominal,max_bin_search_iters,dt_upper_init):

    print "flashlight.trajectory_optimization.quadrotor3d_uniform_time_stretch: Attempting to find numerically stable trajectory with max_bin_search_iters = %d, dt_upper_init = %f." % (max_bin_search_iters,dt_upper_init)

    psi_nominal = matrix(psi_nominal).A1

    dt_lower   = dt_nominal
    dt_upper   = dt_upper_init
    dt_current = dt_nominal

    # if lower bound satisfies constraints, return
    if quadrotor3d.check_differentially_flat_trajectory_numerically_stable(p_nominal,psi_nominal,dt_lower):
        print "flashlight.trajectory_optimization.quadrotor3d_uniform_time_stretch: Input trajectory is already numerically stable."
        return 1.0

    # assert upper bound satisfies constraints
    assert quadrotor3d.check_differentially_flat_trajectory_numerically_stable(p_nominal,psi_nominal,dt_upper)

    # binary search
    for i in range(max_bin_search_iters):

        dt_current = (dt_lower+dt_upper)/2.0

        if quadrotor3d.check_differentially_flat_trajectory_numerically_stable(p_nominal,psi_nominal,dt_current):
            dt_upper = dt_current
        else:
            dt_lower = dt_current

    # return smallest numerically stable dt from binary search
    dt_current       = dt_upper
    dt_scale_current = dt_current/dt_nominal

    return dt_scale_current
