from pylab import *

import os, sys, time
import pdb
import scipy.interpolate
import sklearn.metrics
import sympy

import pathutils
pathutils.add_relative_to_current_source_file_path_to_sys_path("..")

import flashlight.splineutils      as splineutils
import flashlight.curveutils       as curveutils
import flashlight.gradientutils    as gradientutils
import flashlight.interpolateutils as interpolateutils
import flashlight.sympyutils       as sympyutils
import flashlight.quadrotor3d      as quadrotor3d

import flashlight.trajectory_optimization.quadrotor3d_uniform_time_stretch  as quadrotor3d_uniform_time_stretch
import flashlight.trajectory_optimization.quadrotor3d_gaussian_time_stretch as quadrotor3d_gaussian_time_stretch

from optimize.snopt7 import SNOPT_solver



build_sympy_modules = False



max_stretch_iters_numerically_stable          = 100
gauss_width_in_terms_of_dt_numerically_stable = 200.0
gauss_max_in_terms_of_dt_numerically_stable   = 0.2
min_extra_iters_numerically_stable            = 1 # infeasible trajectory dataset
max_extra_iters_numerically_stable            = 10

max_stretch_iters_feasible          = 100
gauss_width_in_terms_of_dt_feasible = 200.0
gauss_max_in_terms_of_dt_feasible   = 0.2
extra_iters_feasible                = 1

max_bin_search_iters_feasible   = 10
dt_upper_init_feasible          = 4.0
dt_scale_extra_stretch_feasible = 1.3

use_gaussian_time_stretching_for_feasible = False

snopt_max_major_iterations             = 50
snopt_max_update_dt_current_iterations = 10
snopt_major_iter_count                 = 0
snopt_current_best_obj_val             = 10e9
snopt_Alpha_1d_current_best            = None
snopt_obj_vals                         = None

dt_current_vals       = None
dt_prev_feasible_vals = None



if build_sympy_modules:

    print "flashlight.trajectory_optimization.quadrotor3d_direct_transcription_nonconst_dt: Constructing sympy symbols..."
    sys_time_begin = time.time()

    x_ti_expr, x_ti_expr_entries = sympyutils.construct_matrix_and_entries("x_ti", (num_x_dims,1))
    u_ti_expr, u_ti_expr_entries = sympyutils.construct_matrix_and_entries("u_ti", (num_u_dims,1))
    dt_ti_expr                  = sympy.Symbol("delta_t_ti")

    lamb_J_control_effort_ti_expr = sympy.Symbol("lamb_J_control_effort_ti")

    lamb_J_x_p_waypoint_ti_expr                                    = sympy.Symbol("lamb_J_x_p_waypoint_ti")
    J_x_p_waypoint_ref_ti_expr, J_x_p_waypoint_ref_ti_expr_entries = sympyutils.construct_matrix_and_entries("J_x_p_waypoint_ref_ti", (num_dims_J_x_p_waypoint_ref_ti,1))

    lamb_J_dt_ti_expr = sympy.Symbol("lamb_J_dt_ti")
    J_dt_ref_ti_expr  = sympy.Symbol("J_dt_ref_ti")

    lamb_g_x_waypoint_ti_expr                              = sympy.Symbol("lamb_g_x_waypoint_ti")
    x_waypoint_ref_ti_expr, x_waypoint_ref_ti_expr_entries = sympyutils.construct_matrix_and_entries("x_waypoint_ref_ti", (num_dims_x_waypoint_ref_ti,1))

    lamb_g_x_p_waypoint_ti_expr                                = sympy.Symbol("lamb_g_x_p_waypoint_ti")
    x_p_waypoint_ref_ti_expr, x_p_waypoint_ref_ti_expr_entries = sympyutils.construct_matrix_and_entries("x_p_waypoint_ref_ti", (num_dims_x_p_waypoint_ref_ti,1))

    lamb_g_dt_ti_expr = sympy.Symbol("lamb_g_dt_ti")
    dt_ref_ti_expr    = sympy.Symbol("dt_ref_ti")

    lamb_syms   = hstack( [ lamb_J_control_effort_ti_expr, lamb_J_x_p_waypoint_ti_expr, lamb_J_dt_ti_expr, lamb_g_x_waypoint_ti_expr, lamb_g_x_p_waypoint_ti_expr, lamb_g_dt_ti_expr ] )
    ref_syms    = hstack( [ matrix(J_x_p_waypoint_ref_ti_expr).A1, J_dt_ref_ti_expr, matrix(x_waypoint_ref_ti_expr).A1, matrix(x_p_waypoint_ref_ti_expr).A1, dt_ref_ti_expr ] )
    var_syms    = hstack( [ matrix(x_ti_expr).A1, matrix(u_ti_expr).A1, dt_ti_expr ] )
    common_syms = hstack( [ lamb_syms, ref_syms, var_syms ] )

    sys_time_end = time.time()
    print "flashlight.trajectory_optimization.quadrotor3d_direct_transcription_nonconst_dt: Finished constructing sympy symbols (%.03f seconds)." % (sys_time_end - sys_time_begin)

    print "flashlight.trajectory_optimization.quadrotor3d_direct_transcription_nonconst_dt: Constructing sympy expressions..."
    sys_time_begin = time.time()

    J_ti_expr = \
        lamb_J_control_effort_ti_expr * ( sympyutils.square(sympyutils.norm(u_ti_expr)) * dt_ti_expr )                             + \
        lamb_J_x_p_waypoint_ti_expr   * ( sympyutils.square(sympyutils.norm(x_ti_expr[x_p_inds,:] - J_x_p_waypoint_ref_ti_expr)) ) + \
        lamb_J_dt_ti_expr             * ( sympyutils.square((dt_ti_expr - J_dt_ref_ti_expr)) )

    dJti_dxti_expr  = sympyutils.diff_scalar_wrt_vector(J_ti_expr,x_ti_expr)
    dJti_duti_expr  = sympyutils.diff_scalar_wrt_vector(J_ti_expr,u_ti_expr)
    dJti_ddtti_expr = J_ti_expr.diff(dt_ti_expr)

    g_x_waypoint_ti_expr    = lamb_g_x_waypoint_ti_expr * ( x_ti_expr - x_waypoint_ref_ti_expr )
    dgxwaypointti_dxti_expr = g_x_waypoint_ti_expr.jacobian(x_ti_expr)

    g_x_p_waypoint_ti_expr   = lamb_g_x_p_waypoint_ti_expr * ( x_ti_expr[x_p_inds,:] - x_p_waypoint_ref_ti_expr )
    dgxpwaypointti_dxti_expr = g_x_p_waypoint_ti_expr.jacobian(x_ti_expr)

    g_dt_ti_expr      = lamb_g_dt_ti_expr * ( dt_ti_expr - dt_ref_ti_expr )
    dgdtti_ddtti_expr = g_dt_ti_expr.diff(dt_ti_expr)

    sys_time_end = time.time()
    print "flashlight.trajectory_optimization.quadrotor3d_direct_transcription_nonconst_dt: Finished constructing sympy expressions (%.03f seconds)." % (sys_time_end - sys_time_begin)

    print "flashlight.trajectory_optimization.quadrotor3d_direct_transcription_nonconst_dt: Building sympy modules..."
    sys_time_begin = time.time()

    current_source_file_path = pathutils.get_current_source_file_path()

    sympyutils.build_module_autowrap( expr=J_ti_expr,                syms=common_syms, module_name="J_ti",                tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dJti_dxti_expr,           syms=common_syms, module_name="dJti_dxti",           tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dJti_duti_expr,           syms=common_syms, module_name="dJti_duti",           tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dJti_ddtti_expr,          syms=common_syms, module_name="dJti_ddtti",          tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=g_x_waypoint_ti_expr,     syms=common_syms, module_name="g_x_waypoint_ti",     tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dgxwaypointti_dxti_expr,  syms=common_syms, module_name="dgxwaypointti_dxti",  tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=g_x_p_waypoint_ti_expr,   syms=common_syms, module_name="g_x_p_waypoint_ti",   tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dgxpwaypointti_dxti_expr, syms=common_syms, module_name="dgxpwaypointti_dxti", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=g_dt_ti_expr,             syms=common_syms, module_name="g_dt_ti",             tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dgdtti_ddtti_expr,        syms=common_syms, module_name="dgdtti_ddtti",        tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )

    sys_time_end = time.time()
    print "flashlight.trajectory_optimization.quadrotor3d_direct_transcription_nonconst_dt: Finished building sympy modules (%.03f seconds)." % (sys_time_end - sys_time_begin)

print "flashlight.trajectory_optimization.quadrotor3d_direct_transcription_nonconst_dt: Loading sympy modules..."
sys_time_begin = time.time()

current_source_file_path = pathutils.get_current_source_file_path()

J_ti_autowrap                = sympyutils.import_anon_func_from_from_module_autowrap( module_name="J_ti",                path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dJti_dxti_autowrap           = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dJti_dxti",           path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dJti_duti_autowrap           = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dJti_duti",           path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dJti_ddtti_autowrap          = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dJti_ddtti",          path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
g_x_waypoint_ti_autowrap     = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_x_waypoint_ti",     path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dgxwaypointti_dxti_autowrap  = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgxwaypointti_dxti",  path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
g_x_p_waypoint_ti_autowrap   = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_x_p_waypoint_ti",   path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dgxpwaypointti_dxti_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgxpwaypointti_dxti", path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
g_dt_ti_autowrap             = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_dt_ti",             path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dgdtti_ddtti_autowrap        = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgdtti_ddtti",        path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )

J_ti_vectorized_autowrap                = sympyutils.import_anon_func_from_from_module_autowrap( module_name="J_ti_vectorized",                path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dJti_dxti_vectorized_autowrap           = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dJti_dxti_vectorized",           path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dJti_duti_vectorized_autowrap           = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dJti_duti_vectorized",           path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dJti_ddtti_vectorized_autowrap          = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dJti_ddtti_vectorized",          path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
g_x_waypoint_ti_vectorized_autowrap     = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_x_waypoint_ti_vectorized",     path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dgxwaypointti_dxti_vectorized_autowrap  = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgxwaypointti_dxti_vectorized",  path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
g_x_p_waypoint_ti_vectorized_autowrap   = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_x_p_waypoint_ti_vectorized",   path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dgxpwaypointti_dxti_vectorized_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgxpwaypointti_dxti_vectorized", path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
g_dt_ti_vectorized_autowrap             = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_dt_ti_vectorized",             path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )
dgdtti_ddtti_vectorized_autowrap        = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgdtti_ddtti_vectorized",        path=current_source_file_path+"/data/quadrotor3d_direct_transcription_nonconst_dt" )

sys_time_end = time.time()
print "flashlight.trajectory_optimization.quadrotor3d_direct_transcription_nonconst_dt: Finished loading sympy modules (%.03f seconds)." % (sys_time_end - sys_time_begin)



def optimize(p_eval,psi_eval,                            \
             t_nominal,user_progress_nominal,dt_nominal, \
             const_vals_ti,                              \
             x_min_ti,x_max_ti,                          \
             u_min_ti,u_max_ti):

    assert allclose(psi_eval,0.0)

    print "flashlight.trajectory_optimization.quadrotor3d_direct_transcription_nonconst_dt: Initializing optimization problem..."
    sys_time_begin    = time.time()
    solver_time_begin = sys_time_begin

    #
    # find numerically stable and feasible trajectories to initialize the solver
    #
    numerically_stable_infeasible_trajectory = quadrotor3d_gaussian_time_stretch.optimize_numerically_stable_infeasible( p_eval,psi_eval,                               \
                                                                                                                         t_nominal,user_progress_nominal,dt_nominal,    \
                                                                                                                         x_min_ti,x_max_ti,                             \
                                                                                                                         u_min_ti,u_max_ti,                             \
                                                                                                                         max_stretch_iters_numerically_stable,          \
                                                                                                                         gauss_width_in_terms_of_dt_numerically_stable, \
                                                                                                                         gauss_max_in_terms_of_dt_numerically_stable,   \
                                                                                                                         0 )

    x_numerically_stable,u_numerically_stable,t_numerically_stable,user_progress_numerically_stable,dt_numerically_stable = numerically_stable_infeasible_trajectory

    if use_gaussian_time_stretching_for_feasible:

        # use gaussian time stretching to find a feasible trajectory
        feasible_trajectory = quadrotor3d_gaussian_time_stretch.optimize_feasible( p_eval,psi_eval,                                                             \
                                                                                   t_numerically_stable,user_progress_numerically_stable,dt_numerically_stable, \
                                                                                   x_min_ti,x_max_ti,                                                           \
                                                                                   u_min_ti,u_max_ti,                                                           \
                                                                                   max_stretch_iters_feasible,                                                  \
                                                                                   gauss_width_in_terms_of_dt_feasible,                                         \
                                                                                   gauss_max_in_terms_of_dt_feasible,                                           \
                                                                                   extra_iters_feasible )

        x_feasible,u_feasible,t_feasible,user_progress_feasible,dt_feasible = feasible_trajectory

    else:

        # use uniform time stretching to find a feasible trajectory
        p_nominal, _, _, _   = curveutils.reparameterize_curve( p_eval, user_progress_nominal )
        psi_nominal, _, _, _ = curveutils.reparameterize_curve( psi_eval, user_progress_nominal )

        feasible_trajectory = quadrotor3d_uniform_time_stretch.optimize_feasible( p_nominal,psi_nominal,dt_nominal, \
                                                                                  x_min_ti,x_max_ti,                \
                                                                                  u_min_ti,u_max_ti,                \
                                                                                  max_bin_search_iters_feasible,    \
                                                                                  dt_upper_init_feasible )

        x_feasible,u_feasible,dt_scale_feasible = feasible_trajectory
        t_feasible                              = t_nominal*dt_scale_feasible*dt_scale_extra_stretch_feasible
        user_progress_feasible                  = user_progress_nominal
        dt_feasible                             = dt_nominal*dt_scale_feasible*dt_scale_extra_stretch_feasible

    # return user_progress_numerically_stable,None,None,None,None,t_numerically_stable,t_numerically_stable[-1]
    # return user_progress_feasible,None,None,None,None,t_feasible,t_feasible[-1]

    sys_time_end = time.time()
    print "flashlight.optimize.quadrotor3d_fixed_path: Finished initializing optimization problem (%.03f seconds)." % (sys_time_end - sys_time_begin)

    #
    # set up optimization problem constants
    #
    num_trajectory_samples = p_eval.shape[0]
    num_x_dims             = quadrotor3d.num_x_dims
    num_u_dims             = quadrotor3d.num_u_dims
    num_dt_dims            = 1
    num_alpha_dims         = num_x_dims + num_u_dims + num_dt_dims
    x_p_inds               = arange(0,3)
    x_e_inds               = arange(3,6)
    num_x_p_inds           = x_p_inds.size

    # soft control effort constraints
    lamb_J_control_effort = 0.0*ones(num_trajectory_samples)

    # soft position waypoint constraints
    num_dims_J_x_p_waypoint_ref_ti = 3
    lamb_J_x_p_waypoint            = 0.01*ones(num_trajectory_samples)
    J_x_p_waypoint_ref             = x_numerically_stable[:,0:3]

    # soft dt constraints
    num_dims_J_dt_ref_ti = 1
    lamb_J_dt            = 0.0001*ones(num_trajectory_samples)
    J_dt_ref             = dt_numerically_stable*ones(num_trajectory_samples)

    # hard dynamics constraints
    num_dims_g_dynamics_ti = num_x_dims

    # hard state space waypoint constraints
    num_dims_g_x_waypoint_ti   = num_x_dims
    num_dims_x_waypoint_ref_ti = num_x_dims
    lamb_g_x_waypoint          = zeros(num_trajectory_samples)
    lamb_g_x_waypoint[[0,-1]]  = 1
    X_waypoint_ref             = zeros((num_trajectory_samples,num_dims_x_waypoint_ref_ti))
    X_waypoint_ref[0]          = array([ p_eval[0,0],  p_eval[0,1],  p_eval[0,2],  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
    X_waypoint_ref[-1]         = array([ p_eval[-1,0], p_eval[-1,1], p_eval[-1,2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])

    lamb_g_x_waypoint_nonzero                                     = nonzero(lamb_g_x_waypoint)[0]
    num_lamb_g_x_waypoint_nonzero                                 = len(lamb_g_x_waypoint_nonzero)
    lamb_g_x_waypoint_ti_to_ti_sparse                             = -1*ones_like(lamb_g_x_waypoint,dtype=int32)
    lamb_g_x_waypoint_ti_to_ti_sparse[lamb_g_x_waypoint_nonzero]  = arange(num_lamb_g_x_waypoint_nonzero)

    # hard position waypoint constraints
    num_dims_g_x_p_waypoint_ti   = num_x_p_inds
    num_dims_x_p_waypoint_ref_ti = num_x_p_inds
    lamb_g_x_p_waypoint          = zeros(num_trajectory_samples)
    X_p_waypoint_ref             = zeros((num_trajectory_samples,num_dims_x_p_waypoint_ref_ti))

    lamb_g_x_p_waypoint_nonzero                                      = nonzero(lamb_g_x_p_waypoint)[0]
    num_lamb_g_x_p_waypoint_nonzero                                  = len(lamb_g_x_p_waypoint_nonzero)
    lamb_g_x_p_waypoint_ti_to_ti_sparse                              = -1*ones_like(lamb_g_x_p_waypoint,dtype=int32)
    lamb_g_x_p_waypoint_ti_to_ti_sparse[lamb_g_x_p_waypoint_nonzero] = arange(num_lamb_g_x_p_waypoint_nonzero)

    # hard dt constraints
    num_dims_g_dt_ti   = 1
    num_dims_dt_ref_ti = 1
    lamb_g_dt          = zeros(num_trajectory_samples)
    dt_ref             = zeros(num_trajectory_samples)

    lamb_g_dt_nonzero                            = nonzero(lamb_g_dt)[0]
    num_lamb_g_dt_nonzero                        = len(lamb_g_dt_nonzero)
    lamb_g_dt_ti_to_ti_sparse                    = -1*ones_like(lamb_g_dt,dtype=int32)
    lamb_g_dt_ti_to_ti_sparse[lamb_g_dt_nonzero] = arange(num_lamb_g_dt_nonzero)

    dt_min_ti = dt_numerically_stable*0.45
    dt_max_ti = dt_feasible*1.55

    # stack all the const, lamb, and ref values
    const_vals    = tile(const_vals_ti,(num_trajectory_samples,1))
    lamb_vals     = c_[ lamb_J_control_effort, lamb_J_x_p_waypoint, lamb_J_dt, lamb_g_x_waypoint, lamb_g_x_p_waypoint, lamb_g_dt ]
    ref_vals      = c_[ J_x_p_waypoint_ref, J_dt_ref, X_waypoint_ref, X_p_waypoint_ref, dt_ref ]

    # number of constraints and decision variables
    num_constraints_g_dynamics = num_trajectory_samples-1
    num_dims_g_dynamics_ti     = num_x_dims

    num_constraints_1d_g_dynamics     = num_constraints_g_dynamics*num_dims_g_dynamics_ti
    num_constraints_1d_g_x_waypoint   = num_lamb_g_x_waypoint_nonzero*num_dims_g_x_waypoint_ti
    num_constraints_1d_g_x_p_waypoint = num_lamb_g_x_p_waypoint_nonzero*num_dims_g_x_p_waypoint_ti
    num_constraints_1d_g_dt           = num_lamb_g_dt_nonzero*num_dims_g_dt_ti

    num_decision_vars_1d_X  = num_trajectory_samples*num_x_dims
    num_decision_vars_1d_U  = num_trajectory_samples*num_u_dims
    num_decision_vars_1d_DT = num_trajectory_samples*num_dt_dims

    def _unpack_Alpha_1d(Alpha_1d):

        X_1d_begin,X_1d_end   = 0,        0        + num_trajectory_samples*num_x_dims
        U_1d_begin,U_1d_end   = X_1d_end, X_1d_end + num_trajectory_samples*num_u_dims
        DT_1d_begin,DT_1d_end = U_1d_end, U_1d_end + num_trajectory_samples*num_dt_dims

        X_1d  = Alpha_1d[X_1d_begin:X_1d_end]
        U_1d  = Alpha_1d[U_1d_begin:U_1d_end]
        DT_1d = Alpha_1d[DT_1d_begin:DT_1d_end]
        X     = X_1d.reshape((num_trajectory_samples,num_x_dims))
        U     = U_1d.reshape((num_trajectory_samples,num_u_dims))
        DT    = DT_1d.reshape((num_trajectory_samples,num_dt_dims))

        return X,U,DT

    def _compute_common_vals(ti,X,U,DT):

        lamb_J_control_effort_ti = lamb_J_control_effort[ti]
        lamb_J_x_p_waypoint_ti   = lamb_J_x_p_waypoint[ti]
        lamb_J_dt_ti             = lamb_J_dt[ti]
        lamb_g_x_waypoint_ti     = lamb_g_x_waypoint[ti]
        lamb_g_x_p_waypoint_ti   = lamb_g_x_p_waypoint[ti]
        lamb_g_dt_ti             = lamb_g_dt[ti]
        J_x_p_waypoint_ref_ti    = matrix(J_x_p_waypoint_ref[ti]).T
        J_dt_ref_ti              = J_dt_ref[ti]
        x_waypoint_ref_ti        = matrix(X_waypoint_ref[ti]).T
        x_p_waypoint_ref_ti      = matrix(X_p_waypoint_ref[ti]).T
        dt_ref_ti                = dt_ref[ti]
        x_ti                     = matrix(X[ti]).T
        u_ti                     = matrix(U[ti]).T
        dt_ti                    = DT[ti]

        lamb_vals_ti   = hstack( [ lamb_J_control_effort_ti, lamb_J_x_p_waypoint_ti, lamb_J_dt_ti, lamb_g_x_waypoint_ti, lamb_g_x_p_waypoint_ti, lamb_g_dt_ti ] )
        ref_vals_ti    = hstack( [ matrix(J_x_p_waypoint_ref_ti).A1, J_dt_ref_ti, matrix(x_waypoint_ref_ti).A1, matrix(x_p_waypoint_ref_ti).A1, dt_ref_ti ] )
        var_vals_ti    = hstack( [ x_ti.A1, u_ti.A1, dt_ti ] )
        common_vals_ti = hstack( [ lamb_vals_ti, ref_vals_ti, var_vals_ti ] )

        return common_vals_ti

    def _compute_sparse_jacobian_indices(ti,ti_to_ti_sparse,num_dims_gi):

        ti_sparse = ti_to_ti_sparse[ti]
        gi_begin  = (ti_sparse+0)*num_dims_gi
        gi_end    = (ti_sparse+1)*num_dims_gi
        xi_begin  = (ti+0)*num_x_dims
        xi_end    = (ti+1)*num_x_dims
        ui_begin  = (ti+0)*num_u_dims
        ui_end    = (ti+1)*num_u_dims
        dti_begin = (ti+0)*num_dt_dims
        dti_end   = (ti+1)*num_dt_dims

        return gi_begin,gi_end,xi_begin,xi_end,ui_begin,ui_end,dti_begin,dti_end

    # Define objective function
    def _obj_func(Alpha_1d):

        global snopt_major_iter_count
        global snopt_obj_vals

        X,U,DT = _unpack_Alpha_1d(Alpha_1d)

        common_vals                                                  = c_[ lamb_vals, ref_vals, X, U, DT ]
        const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals = c_[ const_vals[:-1], X[:-1], X[1:], U[:-1], DT[:-1] ]

        J_ti       = J_ti_vectorized_autowrap(common_vals)
        g_dynamics = quadrotor3d.g_dynamics_ti_vectorized_autowrap(const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals)

        g_x_waypoint   = zeros((num_lamb_g_x_waypoint_nonzero,   num_dims_g_x_waypoint_ti))
        g_x_p_waypoint = zeros((num_lamb_g_x_p_waypoint_nonzero, num_dims_g_x_p_waypoint_ti))
        g_dt           = zeros((num_lamb_g_dt_nonzero,           num_dims_g_dt_ti))

        for ti in range(num_trajectory_samples):

            common_vals_ti = _compute_common_vals(ti,X,U,DT)

            lamb_g_x_waypoint_ti   = lamb_g_x_waypoint[ti]
            lamb_g_x_p_waypoint_ti = lamb_g_x_p_waypoint[ti]
            lamb_g_dt_ti           = lamb_g_dt[ti]

            if lamb_g_x_waypoint_ti   != 0: g_x_waypoint[lamb_g_x_waypoint_ti_to_ti_sparse[ti]]     = sympyutils.evaluate_anon_func( g_x_waypoint_ti_autowrap,   common_vals_ti ).T
            if lamb_g_x_p_waypoint_ti != 0: g_x_p_waypoint[lamb_g_x_p_waypoint_ti_to_ti_sparse[ti]] = sympyutils.evaluate_anon_func( g_x_p_waypoint_ti_autowrap, common_vals_ti ).T
            if lamb_g_dt_ti           != 0: g_dt[lamb_g_dt_ti_to_ti_sparse[ti]]                     = sympyutils.evaluate_anon_func( g_dt_ti_autowrap,           common_vals_ti )

        J = sum(J_ti)

        g_1d = hstack( [ matrix(g_dynamics).A1, matrix(g_x_waypoint).A1, matrix(g_x_p_waypoint).A1, matrix(g_dt).A1 ] )

        snopt_obj_vals[snopt_major_iter_count,0] = J
        snopt_obj_vals[snopt_major_iter_count,1] = sum(norm(g_dynamics,axis=1))
        snopt_major_iter_count                   = snopt_major_iter_count+1

        set_printoptions(suppress=True)
        print "SNOPT major iteration: %d, Objective value: %f, Total g_dynamics error: %f" % (snopt_major_iter_count,J,sum(square(g_dynamics)))

        fail = 0
        return J, g_1d, fail

    # Define gradient function
    def _grad_func(Alpha_1d, J, g_1d, compute_nonzero_only=False):

        X,U,DT = _unpack_Alpha_1d(Alpha_1d)

        dJ_dX  = zeros((num_trajectory_samples,num_x_dims))
        dJ_dU  = zeros((num_trajectory_samples,num_u_dims))
        dJ_dDT = zeros((num_trajectory_samples,num_dt_dims))

        dgdynamics_dX  = zeros((num_constraints_1d_g_dynamics,num_decision_vars_1d_X))
        dgdynamics_dU  = zeros((num_constraints_1d_g_dynamics,num_decision_vars_1d_U))
        dgdynamics_dDT = zeros((num_constraints_1d_g_dynamics,num_decision_vars_1d_DT))

        dgxwaypoint_dX  = zeros((num_constraints_1d_g_x_waypoint,num_decision_vars_1d_X))
        dgxwaypoint_dU  = zeros((num_constraints_1d_g_x_waypoint,num_decision_vars_1d_U))
        dgxwaypoint_dDT = zeros((num_constraints_1d_g_x_waypoint,num_decision_vars_1d_DT))

        dgxpwaypoint_dX  = zeros((num_constraints_1d_g_x_p_waypoint,num_decision_vars_1d_X))
        dgxpwaypoint_dU  = zeros((num_constraints_1d_g_x_p_waypoint,num_decision_vars_1d_U))
        dgxpwaypoint_dDT = zeros((num_constraints_1d_g_x_p_waypoint,num_decision_vars_1d_DT))

        dgdt_dX  = zeros((num_constraints_1d_g_dt,num_decision_vars_1d_X))
        dgdt_dU  = zeros((num_constraints_1d_g_dt,num_decision_vars_1d_U))
        dgdt_dDT = zeros((num_constraints_1d_g_dt,num_decision_vars_1d_DT))

        if not compute_nonzero_only:
            
            common_vals                                                  = c_[ lamb_vals, ref_vals, X, U, DT ]
            const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals = c_[ const_vals[:-1], X[:-1], X[1:], U[:-1], DT[:-1] ]

            dJ_dX  = dJti_dxti_vectorized_autowrap(common_vals)
            dJ_dU  = dJti_duti_vectorized_autowrap(common_vals)
            dJ_dDT = dJti_ddtti_vectorized_autowrap(common_vals)

            dgdynamics_dX_current_block  = quadrotor3d.dgdynamicsti_dxcurrent_vectorized_autowrap(const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals)
            dgdynamics_dX_next_block     = quadrotor3d.dgdynamicsti_dxnext_vectorized_autowrap(const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals)
            dgdynamics_dU_current_block  = quadrotor3d.dgdynamicsti_ducurrent_vectorized_autowrap(const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals)
            dgdynamics_dDT_current_block = quadrotor3d.dgdynamicsti_ddtcurrent_vectorized_autowrap(const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals)

        for ti in range(num_trajectory_samples):

            if compute_nonzero_only:
                dJ_dX[ti]  = 1
                dJ_dU[ti]  = 1
                dJ_dDT[ti] = 1

        for ti in range(num_constraints_g_dynamics):

            gi_begin = (ti+0)*num_dims_g_dynamics_ti
            gi_end   = (ti+1)*num_dims_g_dynamics_ti

            ai_x_current_begin  = (ti+0)*num_x_dims
            ai_x_current_end    = (ti+1)*num_x_dims
            ai_x_next_begin     = (ti+1)*num_x_dims
            ai_x_next_end       = (ti+2)*num_x_dims
            ai_u_current_begin  = (ti+0)*num_u_dims
            ai_u_current_end    = (ti+1)*num_u_dims
            ai_dt_current_begin = (ti+0)*num_dt_dims
            ai_dt_current_end   = (ti+1)*num_dt_dims

            if compute_nonzero_only:
                dgdynamics_dX[gi_begin:gi_end,ai_x_current_begin:ai_x_current_end]    = 1
                dgdynamics_dX[gi_begin:gi_end,ai_x_next_begin:ai_x_next_end]          = 1
                dgdynamics_dU[gi_begin:gi_end,ai_u_current_begin:ai_u_current_end]    = 1
                dgdynamics_dDT[gi_begin:gi_end,ai_dt_current_begin:ai_dt_current_end] = 1
            else:
                dgdynamics_dX[gi_begin:gi_end,ai_x_current_begin:ai_x_current_end]    = dgdynamics_dX_current_block[ti]
                dgdynamics_dX[gi_begin:gi_end,ai_x_next_begin:ai_x_next_end]          = dgdynamics_dX_next_block[ti]
                dgdynamics_dU[gi_begin:gi_end,ai_u_current_begin:ai_u_current_end]    = dgdynamics_dU_current_block[ti]
                dgdynamics_dDT[gi_begin:gi_end,ai_dt_current_begin:ai_dt_current_end] = matrix(dgdynamics_dDT_current_block[ti]).T

        for ti in range(num_trajectory_samples):

            common_vals_ti = _compute_common_vals(ti,X,U,DT)

            lamb_g_x_waypoint_ti   = lamb_g_x_waypoint[ti]
            lamb_g_x_p_waypoint_ti = lamb_g_x_p_waypoint[ti]
            lamb_g_dt_ti           = lamb_g_dt[ti]

            if lamb_g_x_waypoint_ti != 0:

                gi_begin,gi_end,xi_begin,xi_end,ui_begin,ui_end,li_begin,li_end = _compute_sparse_jacobian_indices(ti,lamb_g_x_waypoint_ti_to_ti_sparse,num_dims_g_x_waypoint_ti)
                dgxwaypoint_dX[gi_begin:gi_end,xi_begin:xi_end]                 = sympyutils.evaluate_anon_func( dgxwaypointti_dxti_autowrap, common_vals_ti )

            if lamb_g_x_p_waypoint_ti != 0:

                gi_begin,gi_end,xi_begin,xi_end,ui_begin,ui_end,li_begin,li_end = _compute_sparse_jacobian_indices(ti,lamb_g_x_p_waypoint_ti_to_ti_sparse,num_dims_g_x_p_waypoint_ti)
                dgxpwaypoint_dX[gi_begin:gi_end,xi_begin:xi_end]                = sympyutils.evaluate_anon_func( dgxpwaypointti_dxti_autowrap, common_vals_ti )

            if lamb_g_dt_ti != 0:

                gi_begin,gi_end,xi_begin,xi_end,ui_begin,ui_end,dti_begin,dti_end = _compute_sparse_jacobian_indices(ti,lamb_g_dt_ti_to_ti_sparse,num_dims_g_dt_ti)
                dgdt_dDT[gi_begin:gi_end,dti_begin:dti_end]                       = sympyutils.evaluate_anon_func( dgdtti_ddtti_autowrap, common_vals_ti )

        dJ_dAlpha_1d = hstack( [ matrix(dJ_dX).A1, matrix(dJ_dU).A1, matrix(dJ_dDT).A1 ] )

        dgdynamics_dAlpha   = c_[ dgdynamics_dX,   dgdynamics_dU,   dgdynamics_dDT   ]
        dgxwaypoint_dAlpha  = c_[ dgxwaypoint_dX,  dgxwaypoint_dU,  dgxwaypoint_dDT  ]
        dgxpwaypoint_dAlpha = c_[ dgxpwaypoint_dX, dgxpwaypoint_dU, dgxpwaypoint_dDT ]
        dgdt_dAlpha         = c_[ dgdt_dX,         dgdt_dU,         dgdt_dDT ]

        dg_dAlpha = r_[ dgdynamics_dAlpha, dgxwaypoint_dAlpha, dgxpwaypoint_dAlpha, dgdt_dAlpha ]

        fail = 0
        return matrix(dJ_dAlpha_1d).A, dg_dAlpha, fail

    def _obj_grad_func(status,Alpha_1d,needF,needG,cu,iu,ru):

        J, g_1d, fail                    = _obj_func(Alpha_1d)
        dJ_dAlpha_1d, dg_dAlpha, fail    = _grad_func(Alpha_1d,J,g_1d)
        J_g_1d                           = hstack( [ J, g_1d, snopt_dummy_val ] )
        dJ_dAlpha_dg_dAlpha              = r_[ dJ_dAlpha_1d, dg_dAlpha ]
        dJ_dAlpha_dg_dAlpha_nonzero_vals = dJ_dAlpha_dg_dAlpha[dJ_dAlpha_dg_dAlpha_nonzero_inds]

        return status, J_g_1d, dJ_dAlpha_dg_dAlpha_nonzero_vals

    inf   = 1.0e20
    snopt = SNOPT_solver()

    snopt.setOption('Verbose',False)
    snopt.setOption('Solution print',False)
    snopt.setOption('Major print level',0)
    snopt.setOption('Print level',0)

    snopt_obj_row      = 1
    snopt_num_funcs_1d = num_constraints_1d_g_dynamics + num_constraints_1d_g_x_waypoint + num_constraints_1d_g_x_p_waypoint + num_constraints_1d_g_dt + 1
    snopt_num_vars_1d  = num_decision_vars_1d_X + num_decision_vars_1d_U + num_decision_vars_1d_DT
    snopt_dummy_val    = 0.0
    snopt_dummy_array  = zeros((1,snopt_num_vars_1d))

    global snopt_major_iter_count
    global snopt_obj_vals

    snopt_major_iter_count = 0
    snopt_obj_vals         = -1*ones((10000,2))

    X_min  = tile(x_min_ti.A1,(1,num_trajectory_samples))
    X_max  = tile(x_max_ti.A1,(1,num_trajectory_samples))
    U_min  = tile(u_min_ti.A1,(1,num_trajectory_samples))
    U_max  = tile(u_max_ti.A1,(1,num_trajectory_samples))
    DT_min = tile(dt_min_ti,(1,num_trajectory_samples))
    DT_max = tile(dt_max_ti,(1,num_trajectory_samples))

    Alpha_min = hstack( [ matrix(X_min).A1, matrix(U_min).A1, matrix(DT_min).A1 ] )
    Alpha_max = hstack( [ matrix(X_max).A1, matrix(U_max).A1, matrix(DT_max).A1 ] )

    X_0     = x_feasible
    U_0     = u_feasible
    DT_0    = dt_feasible*ones(num_trajectory_samples)
    Alpha_0 = hstack( [ matrix(X_0).A1, matrix(U_0).A1, matrix(DT_0).A1 ] )

    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Calculating objective value on initial guess..."        
    _obj_func(Alpha_0)

    J_g_1d_min = hstack( [ -inf, zeros(num_constraints_1d_g_dynamics), zeros(num_constraints_1d_g_x_waypoint), zeros(num_constraints_1d_g_x_p_waypoint), zeros(num_constraints_1d_g_dt), snopt_dummy_val ] )
    J_g_1d_max = hstack( [  inf, zeros(num_constraints_1d_g_dynamics), zeros(num_constraints_1d_g_x_waypoint), zeros(num_constraints_1d_g_x_p_waypoint), zeros(num_constraints_1d_g_dt), snopt_dummy_val ] )

    dJ_dAlpha_dg_dAlpha_const       = r_[ zeros((snopt_num_funcs_1d,snopt_num_vars_1d)), snopt_dummy_array ]
    dJ_dAlpha_dg_dAlpha_const[-1,0] = 10e-9

    dJ_dAlpha_nonzero, dg_dAlpha_nonzero, fail = _grad_func(Alpha_0, J=None, g_1d=None, compute_nonzero_only=True)

    dJ_dAlpha_dg_dAlpha_nonzero      = r_[ dJ_dAlpha_nonzero, dg_dAlpha_nonzero, snopt_dummy_array ]
    dJ_dAlpha_dg_dAlpha_nonzero_inds = dJ_dAlpha_dg_dAlpha_nonzero.nonzero()

    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Solving optimization problem..."
    sys_time_begin = time.time()

    snopt.snopta(
        name="quadrotor_3d_fixed_path_optimization_test",
        usrfun=_obj_grad_func,
        x0=Alpha_0,
        xlow=Alpha_min,
        xupp=Alpha_max,
        Flow=J_g_1d_min,
        Fupp=J_g_1d_max,
        ObjRow=snopt_obj_row,
        A=dJ_dAlpha_dg_dAlpha_const,
        G=dJ_dAlpha_dg_dAlpha_nonzero,
        xnames=None,
        Fnames=None)

    sys_time_end = time.time()
    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished solving optimization problem (%.03f seconds)." % (sys_time_end - sys_time_begin)

    solver_time_end = sys_time_end
    solver_time     = solver_time_end - solver_time_begin
    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Total solver time was %.03f seconds." % solver_time

    Alpha_opt_1d       = snopt.x
    X_opt,U_opt,DT_opt = _unpack_Alpha_1d(Alpha_opt_1d)

    dt_opt_cumsum = cumsum(DT_opt[:-1])
    t_opt         = hstack( [ t_numerically_stable[0], t_numerically_stable[0] + dt_opt_cumsum ] )
    T_final_opt   = dt_opt_cumsum[-1]

    return X_opt,U_opt,DT_opt,t_opt,T_final_opt,solver_time,snopt_obj_vals
