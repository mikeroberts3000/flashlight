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



# infeasible trajectory dataset, also works for teaser
max_stretch_iters_numerically_stable          = 100
gauss_width_in_terms_of_dt_numerically_stable = 200.0
gauss_max_in_terms_of_dt_numerically_stable   = 0.2
min_extra_iters_numerically_stable            = 1
max_extra_iters_numerically_stable            = 10

max_stretch_iters_feasible          = 100
gauss_width_in_terms_of_dt_feasible = 200.0
gauss_max_in_terms_of_dt_feasible   = 0.2
extra_iters_feasible                = 1

max_bin_search_iters_feasible   = 10
dt_upper_init_feasible          = 4.0
dt_scale_extra_stretch_feasible = 1.3

use_gaussian_time_stretching_for_feasible = False

Beta_0_min_stretch = 0.45
Beta_0_max_stretch = 1.55
Beta_14_stretch    = 0.3
Beta_14_stretch    = 0.3
Gamma_stretch      = 0.3
Gamma_stretch      = 0.3



snopt_max_major_iterations             = 50
snopt_max_update_dt_current_iterations = 10
snopt_major_iter_count                 = 0
snopt_current_best_obj_val             = 10e9
snopt_Alpha_1d_current_best            = None
snopt_obj_vals                         = None

dt_current_vals       = None
dt_prev_feasible_vals = None



if build_sympy_modules:

    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Constructing sympy symbols..."
    sys_time_begin = time.time()

    const_syms = quadrotor3d.const_syms

    # path constants for each dimension
    sigma_0_z_expr      = sympy.Symbol("sigma_0_z",      real=True)
    d1sigmads1_0_z_expr = sympy.Symbol("d1sigmads1_0_z", real=True)
    d2sigmads2_0_z_expr = sympy.Symbol("d2sigmads2_0_z", real=True)
    d3sigmads3_0_z_expr = sympy.Symbol("d3sigmads3_0_z", real=True)
    d4sigmads4_0_z_expr = sympy.Symbol("d4sigmads4_0_z", real=True)

    sigma_0_y_expr      = sympy.Symbol("sigma_0_y",      real=True)
    d1sigmads1_0_y_expr = sympy.Symbol("d1sigmads1_0_y", real=True)
    d2sigmads2_0_y_expr = sympy.Symbol("d2sigmads2_0_y", real=True)
    d3sigmads3_0_y_expr = sympy.Symbol("d3sigmads3_0_y", real=True)
    d4sigmads4_0_y_expr = sympy.Symbol("d4sigmads4_0_y", real=True)

    sigma_0_x_expr      = sympy.Symbol("sigma_0_x",      real=True)
    d1sigmads1_0_x_expr = sympy.Symbol("d1sigmads1_0_x", real=True)
    d2sigmads2_0_x_expr = sympy.Symbol("d2sigmads2_0_x", real=True)
    d3sigmads3_0_x_expr = sympy.Symbol("d3sigmads3_0_x", real=True)
    d4sigmads4_0_x_expr = sympy.Symbol("d4sigmads4_0_x", real=True)

    sigma_0_psi_expr      = sympy.Symbol("sigma_0_psi",      real=True)
    d1sigmads1_0_psi_expr = sympy.Symbol("d1sigmads1_0_psi", real=True)
    d2sigmads2_0_psi_expr = sympy.Symbol("d2sigmads2_0_psi", real=True)
    d3sigmads3_0_psi_expr = sympy.Symbol("d3sigmads3_0_psi", real=True)
    d4sigmads4_0_psi_expr = sympy.Symbol("d4sigmads4_0_psi", real=True)

    ds_expr = sympy.Symbol("ds", real=True)

    dNsigma_dsN_0_z_expr   = sympy.Matrix( [ sigma_0_z_expr,   d1sigmads1_0_z_expr,   d2sigmads2_0_z_expr,   d3sigmads3_0_z_expr,   d4sigmads4_0_z_expr   ] )
    dNsigma_dsN_0_y_expr   = sympy.Matrix( [ sigma_0_y_expr,   d1sigmads1_0_y_expr,   d2sigmads2_0_y_expr,   d3sigmads3_0_y_expr,   d4sigmads4_0_y_expr   ] )
    dNsigma_dsN_0_x_expr   = sympy.Matrix( [ sigma_0_x_expr,   d1sigmads1_0_x_expr,   d2sigmads2_0_x_expr,   d3sigmads3_0_x_expr,   d4sigmads4_0_x_expr   ] )
    dNsigma_dsN_0_psi_expr = sympy.Matrix( [ sigma_0_psi_expr, d1sigmads1_0_psi_expr, d2sigmads2_0_psi_expr, d3sigmads3_0_psi_expr, d4sigmads4_0_psi_expr ] )

    path_syms = hstack( [ matrix(dNsigma_dsN_0_z_expr).A1, matrix(dNsigma_dsN_0_y_expr).A1, matrix(dNsigma_dsN_0_x_expr).A1, matrix(dNsigma_dsN_0_psi_expr).A1, ds_expr ] )

    # path constants
    sigma_0_expr       = sympy.Symbol("sigma_0",      real=True)
    d1sigma_ds1_0_expr = sympy.Symbol("d1sigmads1_0", real=True)
    d2sigma_ds2_0_expr = sympy.Symbol("d2sigmads1_0", real=True)
    d3sigma_ds3_0_expr = sympy.Symbol("d3sigmads1_0", real=True)
    d4sigma_ds4_0_expr = sympy.Symbol("d4sigmads4_0", real=True)

    dNsigma_dsN_0_expr = sympy.Matrix( [ sigma_0_expr, d1sigma_ds1_0_expr, d2sigma_ds2_0_expr, d3sigma_ds3_0_expr, d4sigma_ds4_0_expr ] )

    # decision variables
    s_0_expr      = sympy.Symbol("s_0",     real=True)
    sdot1_0_expr  = sympy.Symbol("sdot1_0", real=True)
    sdot2_0_expr  = sympy.Symbol("sdot2_0", real=True)
    sdot3_0_expr  = sympy.Symbol("sdot3_0", real=True)
    sdot4_0_expr  = sympy.Symbol("sdot4_0", real=True)

    sdotN_0_expr = sympy.Matrix( [ s_0_expr, sdot1_0_expr, sdot2_0_expr, sdot3_0_expr, sdot4_0_expr ] )
    beta_expr    = sympy.Matrix( [ sdot1_0_expr, sdot2_0_expr, sdot3_0_expr, sdot4_0_expr ] )

    beta_syms = hstack( [ matrix(beta_expr).A1 ] )

    sys_time_end = time.time()
    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished constructing sympy symbols (%.03f seconds)." % (sys_time_end - sys_time_begin)

    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Constructing sympy expressions..."
    sys_time_begin = time.time()

    t_expr = sympy.Symbol("t", real=True)

    # q,qdot,qdotdot,x in terms of full state variables
    p_z_expr   = sympy.Symbol("p_z",   real=True)(t_expr)
    p_y_expr   = sympy.Symbol("p_y",   real=True)(t_expr)
    p_x_expr   = sympy.Symbol("p_x",   real=True)(t_expr)
    theta_expr = sympy.Symbol("theta", real=True)(t_expr)
    psi_expr   = sympy.Symbol("psi",   real=True)(t_expr)
    phi_expr   = sympy.Symbol("phi",   real=True)(t_expr)

    q_expr     = sympy.Matrix([p_z_expr,p_y_expr,p_x_expr,theta_expr,psi_expr,phi_expr])
    qdot1_expr = q_expr.diff(t_expr)
    qdot2_expr = qdot1_expr.diff(t_expr)
    qdotN_expr = sympy.Matrix([q_expr,qdot1_expr,qdot2_expr])

    x_expr = sympy.Matrix([q_expr,qdot1_expr])

    H_expr,C_expr,G_expr,B_expr = quadrotor3d.compute_manipulator_matrices_symbolic(x_expr,t_expr)

    # treat Binv as a constant
    B_pinv_expr, B_pinv_entries = sympyutils.construct_matrix_and_entries("B_pinv", (4,6), real=True)
    Binv_syms                   = hstack( [ matrix(B_pinv_expr).A1 ] )

    u_expr = B_pinv_expr*(H_expr*qdot2_expr + C_expr*qdot1_expr + G_expr)

    # q,qdot,qdotdot in terms of flat outputs and time derivatives pdotN
    p_expr         = sympy.Matrix([p_z_expr,p_y_expr,p_x_expr])
    p_dot_expr     = p_expr.diff(t_expr)
    p_dot_dot_expr = p_dot_expr.diff(t_expr)

    f_t_expr            = quadrotor3d.m_expr*p_dot_dot_expr - quadrotor3d.f_external_expr
    f_t_normalized_expr = sympyutils.normalize(f_t_expr)

    z_axis_intermediate_expr = sympy.Matrix( [ sympy.cos(psi_expr), 0, -sympy.sin(psi_expr) ])
    y_axis_expr              = f_t_normalized_expr
    x_axis_expr              = sympyutils.normalize(z_axis_intermediate_expr.cross(y_axis_expr))
    z_axis_expr              = sympyutils.normalize(y_axis_expr.cross(x_axis_expr))

    R_world_from_body_expr = sympyutils.construct_matrix_from_block_matrix( sympy.Matrix( [[z_axis_expr, y_axis_expr, x_axis_expr]] ) )    
    psi_recomputed_in_terms_of_sigmadotN_expr,theta_in_terms_of_sigmadotN_expr,phi_in_terms_of_sigmadotN_expr = sympyutils.euler_from_matrix(R_world_from_body_expr,"ryxz")

    theta_phi_in_terms_of_sigmadotN_subs = zip([theta_expr,phi_expr],[theta_in_terms_of_sigmadotN_expr,phi_in_terms_of_sigmadotN_expr])

    q_in_terms_of_sigmadotN_expr     = q_expr.subs(theta_phi_in_terms_of_sigmadotN_subs)
    qdot1_in_terms_of_sigmadotN_expr = q_in_terms_of_sigmadotN_expr.diff(t_expr)
    qdot2_in_terms_of_sigmadotN_expr = qdot1_in_terms_of_sigmadotN_expr.diff(t_expr)
    qdotN_in_terms_of_sigmadotN_expr = sympy.Matrix([q_in_terms_of_sigmadotN_expr,qdot1_in_terms_of_sigmadotN_expr,qdot2_in_terms_of_sigmadotN_expr])

    # q,qdot,qdotdot in terms of dNp/dsN and sdotN
    s_expr                   = sympy.Symbol("s",     real=True)(t_expr)
    sigma_in_terms_of_s_expr = sympy.Symbol("sigma", real=True)(s_expr)

    sigmadotN_in_terms_of_dNsigmadsN0_and_sdotN0_expr = sympyutils.diff_scalar_using_chain_rule(sigma_in_terms_of_s_expr,s_expr,t_expr,ord=4,dNf_dgN_subs=list(dNsigma_dsN_0_expr),dNg_dhN_subs=list(sdotN_0_expr))

    sigmadotNz_in_terms_of_dNsigmadsN0z_and_sdotN0_expr     = sigmadotN_in_terms_of_dNsigmadsN0_and_sdotN0_expr.subs(zip(dNsigma_dsN_0_expr,dNsigma_dsN_0_z_expr),   simultaneous=True)
    sigmadotNy_in_terms_of_dNsigmadsN0y_and_sdotN0_expr     = sigmadotN_in_terms_of_dNsigmadsN0_and_sdotN0_expr.subs(zip(dNsigma_dsN_0_expr,dNsigma_dsN_0_y_expr),   simultaneous=True)
    sigmadotNx_in_terms_of_dNsigmadsN0x_and_sdotN0_expr     = sigmadotN_in_terms_of_dNsigmadsN0_and_sdotN0_expr.subs(zip(dNsigma_dsN_0_expr,dNsigma_dsN_0_x_expr),   simultaneous=True)
    sigmadotNpsi_in_terms_of_dNsigmadsN0psi_and_sdotN0_expr = sigmadotN_in_terms_of_dNsigmadsN0_and_sdotN0_expr.subs(zip(dNsigma_dsN_0_expr,dNsigma_dsN_0_psi_expr), simultaneous=True)

    sigmadotNz_expr   = sympy.Matrix( [ p_z_expr, sympy.Derivative(p_z_expr,t_expr), sympy.Derivative(p_z_expr,t_expr,t_expr), sympy.Derivative(p_z_expr,t_expr,t_expr,t_expr), sympy.Derivative(p_z_expr,t_expr,t_expr,t_expr,t_expr) ] )
    sigmadotNy_expr   = sympy.Matrix( [ p_y_expr, sympy.Derivative(p_y_expr,t_expr), sympy.Derivative(p_y_expr,t_expr,t_expr), sympy.Derivative(p_y_expr,t_expr,t_expr,t_expr), sympy.Derivative(p_y_expr,t_expr,t_expr,t_expr,t_expr) ] )
    sigmadotNx_expr   = sympy.Matrix( [ p_x_expr, sympy.Derivative(p_x_expr,t_expr), sympy.Derivative(p_x_expr,t_expr,t_expr), sympy.Derivative(p_x_expr,t_expr,t_expr,t_expr), sympy.Derivative(p_x_expr,t_expr,t_expr,t_expr,t_expr) ] )
    sigmadotNpsi_expr = sympy.Matrix( [ psi_expr, sympy.Derivative(psi_expr,t_expr), sympy.Derivative(psi_expr,t_expr,t_expr), sympy.Derivative(psi_expr,t_expr,t_expr,t_expr), sympy.Derivative(psi_expr,t_expr,t_expr,t_expr,t_expr) ] )

    sigmadotN_in_terms_of_dNsigmadsN0_and_sdotN0_subs = zip( list(sigmadotNz_expr[::-1])  + \
                                                             list(sigmadotNy_expr[::-1])  + \
                                                             list(sigmadotNx_expr[::-1])  + \
                                                             list(sigmadotNpsi_expr[::-1]), \
                                                             list(sigmadotNz_in_terms_of_dNsigmadsN0z_and_sdotN0_expr[::-1]) + \
                                                             list(sigmadotNy_in_terms_of_dNsigmadsN0y_and_sdotN0_expr[::-1]) + \
                                                             list(sigmadotNx_in_terms_of_dNsigmadsN0x_and_sdotN0_expr[::-1]) + \
                                                             list(sigmadotNpsi_in_terms_of_dNsigmadsN0psi_and_sdotN0_expr[::-1]) )

    qdotN_in_terms_of_dNsigmadsN0_and_sdotN0_expr = qdotN_in_terms_of_sigmadotN_expr.subs(sigmadotN_in_terms_of_dNsigmadsN0_and_sdotN0_subs)
    qdotN_in_terms_of_dNsigmadsN0_and_sdotN0_subs = zip(qdotN_expr[::-1],qdotN_in_terms_of_dNsigmadsN0_and_sdotN0_expr[::-1])

    # x and u in terms of dNp/dsN and sdotN
    x_in_terms_of_dNsigmadsN0_and_sdotN0_expr           = sympyutils.subs_matrix_verbose(x_expr,qdotN_in_terms_of_dNsigmadsN0_and_sdotN0_subs,simultaneous=True)
    u_in_terms_of_Bpinv_and_dNsigmadsN0_and_sdotN0_expr = sympyutils.subs_matrix_verbose(u_expr,qdotN_in_terms_of_dNsigmadsN0_and_sdotN0_subs,simultaneous=True)

    # J (minimum time) in terms of dNp/dsN and sdotN
    const_and_path_and_beta_syms = hstack( [ const_syms, path_syms, beta_syms ] )

    J_mintime_ti_expr      = ds_expr / sdot1_0_expr
    dJmintimeti_dbeta_expr = sympyutils.diff_scalar_wrt_vector(J_mintime_ti_expr,beta_expr)

    # J (tracking) in terms of dNp/dsN and sdotN
    sdot1_ref_expr                       = sympy.Symbol("sdot1_ref")
    const_and_path_and_ref_and_beta_syms = hstack( [ const_syms, path_syms, sdot1_ref_expr, beta_syms ] )

    J_tracking_ti_expr      = (sdot1_0_expr - sdot1_ref_expr)**2
    dJtrackingti_dbeta_expr = sympyutils.diff_scalar_wrt_vector(J_tracking_ti_expr,beta_expr)

    # g_x_ti,g_u_ti in terms of dNp/dsN and sdotN
    const_and_path_and_Binv_and_beta_syms = hstack( [ const_syms, path_syms, Binv_syms, beta_syms ] )

    g_x_ti_expr = x_in_terms_of_dNsigmadsN0_and_sdotN0_expr
    g_u_ti_expr = u_in_terms_of_Bpinv_and_dNsigmadsN0_and_sdotN0_expr

    dgxti_dbeta_expr = g_x_ti_expr.jacobian(beta_expr)
    dguti_dbeta_expr = g_u_ti_expr.jacobian(beta_expr)

    # g_dynamics in terms of beta and gamma
    beta_current_expr, beta_current_entries   = sympyutils.construct_matrix_and_entries("beta_current", (4,1), real=True)
    beta_next_expr,    beta_next_expr_entries = sympyutils.construct_matrix_and_entries("beta_next",    (4,1), real=True)
    gamma_current_expr                        = sympy.Symbol("gamma_current")
    dt_current_expr                           = sympy.Symbol("dt_fixed")

    beta_gamma_dt_current_next_syms                            = hstack( [ matrix(beta_current_expr).A1, matrix(beta_next_expr).A1, gamma_current_expr, dt_current_expr ] )
    const_and_path_and_beta_and_gamma_and_dt_current_next_syms = hstack( [ const_syms, path_syms, beta_gamma_dt_current_next_syms ] )

    A_dynamics_expr = sympy.Matrix([ [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0] ])
    B_dynamics_expr = sympy.Matrix( [0,0,0,1] )

    # g_dynamics_ti_expr              = beta_next_expr - ( beta_current_expr + (A_dynamics_expr*beta_current_expr + B_dynamics_expr*gamma_current_expr)*(ds_expr/beta_current_expr[0]) )
    g_dynamics_ti_expr              = beta_next_expr - ( beta_current_expr + (A_dynamics_expr*beta_current_expr + B_dynamics_expr*gamma_current_expr)*dt_current_expr )
    dgdynamicsti_dbetacurrent_expr  = g_dynamics_ti_expr.jacobian(beta_current_expr)
    dgdynamicsti_dbetanext_expr     = g_dynamics_ti_expr.jacobian(beta_next_expr)
    dgdynamicsti_dgammacurrent_expr = g_dynamics_ti_expr.diff(gamma_current_expr)

    sys_time_end = time.time()
    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished constructing sympy expressions (%.03f seconds)." % (sys_time_end - sys_time_begin)

    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Building sympy modules..."
    sys_time_begin = time.time()

    current_source_file_path = pathutils.get_current_source_file_path()

    sympyutils.build_module_autowrap( expr=J_mintime_ti_expr,               syms=const_and_path_and_beta_syms,                               module_name="J_mintime_ti",               cse=False,                      build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dJmintimeti_dbeta_expr,          syms=const_and_path_and_beta_syms,                               module_name="dJmintimeti_dbeta",          cse=False,                      build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=J_tracking_ti_expr,              syms=const_and_path_and_ref_and_beta_syms,                       module_name="J_tracking_ti",              cse=False,                      build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dJtrackingti_dbeta_expr,         syms=const_and_path_and_ref_and_beta_syms,                       module_name="dJtrackingti_dbeta",         cse=False,                      build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=g_x_ti_expr,                     syms=const_and_path_and_beta_syms,                               module_name="g_x_ti",                     cse=True,  cse_ordering="none", build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dgxti_dbeta_expr,                syms=const_and_path_and_beta_syms,                               module_name="dgxti_dbeta",                cse=True,  cse_ordering="none", build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=g_u_ti_expr,                     syms=const_and_path_and_Binv_and_beta_syms,                      module_name="g_u_ti",                     cse=True,  cse_ordering="none", build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dguti_dbeta_expr,                syms=const_and_path_and_Binv_and_beta_syms,                      module_name="dguti_dbeta",                cse=True,  cse_ordering="none", build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=g_dynamics_ti_expr,              syms=const_and_path_and_beta_and_gamma_and_dt_current_next_syms, module_name="g_dynamics_ti",              cse=False,                      build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dgdynamicsti_dbetacurrent_expr,  syms=const_and_path_and_beta_and_gamma_and_dt_current_next_syms, module_name="dgdynamicsti_dbetacurrent",  cse=False,                      build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dgdynamicsti_dbetanext_expr,     syms=const_and_path_and_beta_and_gamma_and_dt_current_next_syms, module_name="dgdynamicsti_dbetanext",     cse=False,                      build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dgdynamicsti_dgammacurrent_expr, syms=const_and_path_and_beta_and_gamma_and_dt_current_next_syms, module_name="dgdynamicsti_dgammacurrent", cse=False,                      build_vectorized=True, tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d_fixed_path", verbose=True, request_delete_tmp_dir=True )

    sys_time_end = time.time()
    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished building sympy modules (%.03f seconds)." % (sys_time_end - sys_time_begin)

print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Loading sympy modules..."
sys_time_begin = time.time()

current_source_file_path = pathutils.get_current_source_file_path()

J_mintime_ti_autowrap               = sympyutils.import_anon_func_from_from_module_autowrap( module_name="J_mintime_ti",               path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dJmintimeti_dbeta_autowrap          = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dJmintimeti_dbeta",          path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
J_tracking_ti_autowrap              = sympyutils.import_anon_func_from_from_module_autowrap( module_name="J_tracking_ti",              path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dJtrackingti_dbeta_autowrap         = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dJtrackingti_dbeta",         path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
g_x_ti_autowrap                     = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_x_ti",                     path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dgxti_dbeta_autowrap                = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgxti_dbeta",                path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
g_u_ti_autowrap                     = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_u_ti",                     path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dguti_dbeta_autowrap                = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dguti_dbeta",                path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
g_dynamics_ti_autowrap              = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_dynamics_ti",              path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dgdynamicsti_dbetacurrent_autowrap  = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgdynamicsti_dbetacurrent",  path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dgdynamicsti_dbetanext_autowrap     = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgdynamicsti_dbetanext",     path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dgdynamicsti_dgammacurrent_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgdynamicsti_dgammacurrent", path=current_source_file_path+"/data/quadrotor3d_fixed_path" )

J_mintime_ti_vectorized_autowrap               = sympyutils.import_anon_func_from_from_module_autowrap( module_name="J_mintime_ti_vectorized",               path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dJmintimeti_dbeta_vectorized_autowrap          = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dJmintimeti_dbeta_vectorized",          path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
J_tracking_ti_vectorized_autowrap              = sympyutils.import_anon_func_from_from_module_autowrap( module_name="J_tracking_ti_vectorized",              path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dJtrackingti_dbeta_vectorized_autowrap         = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dJtrackingti_dbeta_vectorized",         path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
g_x_ti_vectorized_autowrap                     = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_x_ti_vectorized",                     path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dgxti_dbeta_vectorized_autowrap                = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgxti_dbeta_vectorized",                path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
g_u_ti_vectorized_autowrap                     = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_u_ti_vectorized",                     path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dguti_dbeta_vectorized_autowrap                = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dguti_dbeta_vectorized",                path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
g_dynamics_ti_vectorized_autowrap              = sympyutils.import_anon_func_from_from_module_autowrap( module_name="g_dynamics_ti_vectorized",              path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dgdynamicsti_dbetacurrent_vectorized_autowrap  = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgdynamicsti_dbetacurrent_vectorized",  path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dgdynamicsti_dbetanext_vectorized_autowrap     = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgdynamicsti_dbetanext_vectorized",     path=current_source_file_path+"/data/quadrotor3d_fixed_path" )
dgdynamicsti_dgammacurrent_vectorized_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="dgdynamicsti_dgammacurrent_vectorized", path=current_source_file_path+"/data/quadrotor3d_fixed_path" )

sys_time_end = time.time()
print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished loading sympy modules (%.03f seconds)." % (sys_time_end - sys_time_begin)



def optimize(p_eval,psi_eval,                            \
             t_nominal,user_progress_nominal,dt_nominal, \
             const_vals_ti,                              \
             x_min_ti,x_max_ti,                          \
             u_min_ti,u_max_ti,                          \
             opt_problem_type):

    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Initializing optimization problem..."
    sys_time_begin    = time.time()
    solver_time_begin = sys_time_begin


    #
    # perturb easing curve to be strictly positive
    #
    make_positive_max_iters                  = 10
    make_positive_gauss_width_in_terms_of_dt = 100.0
    make_positive_gauss_max_in_terms_of_dt   = 0.02

    diff_user_progress_current = diff(user_progress_nominal)
    found                      = False

    for i in range(make_positive_max_iters):

        if all(diff_user_progress_current >= 0.00001):
            print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Found positive easing curve after %d smoothing iterations." % i
            found = True
            break
        else:
            t_diffmin                = t_nominal[ argmin(diff_user_progress_current) ]
            gauss_mean_in_terms_of_t = t_diffmin

            gauss_func   = scipy.stats.norm(loc=gauss_mean_in_terms_of_t,scale=dt_nominal*make_positive_gauss_width_in_terms_of_dt)
            gauss        = gauss_func.pdf(t_nominal[:-1])
            gauss_scaled = (gauss / max(gauss))*dt_nominal*make_positive_gauss_max_in_terms_of_dt

            diff_user_progress_current = diff_user_progress_current + gauss_scaled

    if not found:
        print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: WARNING: Could not find positive easing curve."

    #
    # find numerically stable and feasible trajectories to initialize the solver
    #
    user_progress_nominal_positive_denorm = hstack( [ user_progress_nominal[0], user_progress_nominal[0] + cumsum(diff_user_progress_current) ] )
    user_progress_nominal_positive        = user_progress_nominal_positive_denorm / user_progress_nominal_positive_denorm[-1]

    numerically_stable_infeasible_trajectory = quadrotor3d_gaussian_time_stretch.optimize_numerically_stable_infeasible( p_eval,psi_eval,                                     \
                                                                                                                         t_nominal,user_progress_nominal_positive,dt_nominal, \
                                                                                                                         x_min_ti,x_max_ti,                                   \
                                                                                                                         u_min_ti,u_max_ti,                                   \
                                                                                                                         max_stretch_iters_numerically_stable,                \
                                                                                                                         gauss_width_in_terms_of_dt_numerically_stable,       \
                                                                                                                         gauss_max_in_terms_of_dt_numerically_stable,         \
                                                                                                                         0 )

    x_numerically_stable,u_numerically_stable,t_numerically_stable,user_progress_numerically_stable,dt_numerically_stable = numerically_stable_infeasible_trajectory

    if use_gaussian_time_stretching_for_feasible:

        # use uniform time stretching to find a feasible trajectory
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
    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished initializing optimization problem (%.03f seconds)." % (sys_time_end - sys_time_begin)

    Alpha_opt_1d = None

    for extra_iters_numerically_stable in range(min_extra_iters_numerically_stable,max_extra_iters_numerically_stable):

        print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Attempting to solve problem with extra_iters_numerically_stable = %d." % extra_iters_numerically_stable
        print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Computing numerically stable goal trajectory..."
        sys_time_begin = time.time()

        numerically_stable_infeasible_trajectory = quadrotor3d_gaussian_time_stretch.optimize_numerically_stable_infeasible( p_eval,psi_eval,                                                             \
                                                                                                                             t_numerically_stable,user_progress_numerically_stable,dt_numerically_stable, \
                                                                                                                             x_min_ti,x_max_ti,                                                           \
                                                                                                                             u_min_ti,u_max_ti,                                                           \
                                                                                                                             max_stretch_iters_numerically_stable,                                        \
                                                                                                                             gauss_width_in_terms_of_dt_numerically_stable,                               \
                                                                                                                             gauss_max_in_terms_of_dt_numerically_stable,                                 \
                                                                                                                             extra_iters_numerically_stable )

        x_numerically_stable,u_numerically_stable,t_numerically_stable,user_progress_numerically_stable,dt_numerically_stable = numerically_stable_infeasible_trajectory

        sys_time_end = time.time()
        print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished computing numerically stable goal trajectory (%.03f seconds)." % (sys_time_end - sys_time_begin)

        num_trajectory_samples = t_nominal.shape[0]

        num_beta_dims  = 4
        num_gamma_dims = 1

        num_dims_g_dynamics_ti = 4
        num_dims_g_x_ti        = 12
        num_dims_g_u_ti        = 4

        num_constraints_g_dynamics = num_trajectory_samples-1
        num_constraints_g_x        = num_trajectory_samples
        num_constraints_g_u        = num_trajectory_samples

        num_constraints_1d_g_dynamics = num_constraints_g_dynamics*num_dims_g_dynamics_ti
        num_constraints_1d_g_x        = num_constraints_g_x*num_dims_g_x_ti
        num_constraints_1d_g_u        = num_constraints_g_u*num_dims_g_u_ti
        num_decision_vars_1d_Beta     = num_trajectory_samples*num_beta_dims
        num_decision_vars_1d_Gamma    = num_trajectory_samples*num_gamma_dims

        s_numerically_stable  = user_progress_numerically_stable
        ds_numerically_stable = gradientutils.gradients_scalar_wrt_scalar_smooth_boundaries(s_numerically_stable,dt=1,max_gradient=1,poly_deg=5)[1]

        s_feasible  = user_progress_feasible
        ds_feasible = gradientutils.gradients_scalar_wrt_scalar_smooth_boundaries(s_feasible,dt=1,max_gradient=1,poly_deg=5)[1]

        # compute path derivatives using numerically stable trajectory as a reference
        sigma_numerically_stable = x_numerically_stable[:,[0,1,2,4]]

        dNsigma_dsN_numerically_stable = gradientutils.gradients_vector_wrt_scalar_smooth_boundaries_nonconst_dt(sigma_numerically_stable,s_numerically_stable,max_gradient=4,poly_deg=5)
        d1sigma_ds1_numerically_stable = dNsigma_dsN_numerically_stable[1]
        d2sigma_ds2_numerically_stable = dNsigma_dsN_numerically_stable[2]
        d3sigma_ds3_numerically_stable = dNsigma_dsN_numerically_stable[3]
        d4sigma_ds4_numerically_stable = dNsigma_dsN_numerically_stable[4]

        # set limits on beta using numerically stable and feasible as upper and lower bound
        sdotN_numerically_stable = gradientutils.gradients_vector_wrt_scalar_smooth_boundaries_forward_diffs(s_numerically_stable[:,newaxis],dt_numerically_stable,max_gradient=5,poly_deg=5)
        sdotN_feasible           = gradientutils.gradients_vector_wrt_scalar_smooth_boundaries_forward_diffs(s_feasible[:,newaxis],dt_feasible,max_gradient=5,poly_deg=5)

        sdotN_min = minimum(sdotN_numerically_stable,sdotN_feasible)
        sdotN_max = maximum(sdotN_numerically_stable,sdotN_feasible)

        Beta_min  = c_[ sdotN_min[1], sdotN_min[2], sdotN_min[3], sdotN_min[4] ]
        Beta_max  = c_[ sdotN_max[1], sdotN_max[2], sdotN_max[3], sdotN_max[4] ]
        Gamma_min = sdotN_min[5]
        Gamma_max = sdotN_max[5]

        Beta_min[:,0]   = clip(Beta_min[:,0]*Beta_0_min_stretch, 0.00001,         10.0)
        Beta_max[:,0]   = clip(Beta_max[:,0]*Beta_0_max_stretch, 0.00001+0.00001, 10.0)
        Beta_min[:,1:4] = Beta_min[:,1:4] - (Beta_max[:,1:4]-Beta_min[:,1:4])*Beta_14_stretch - 0.001
        Beta_max[:,1:4] = Beta_max[:,1:4] + (Beta_max[:,1:4]-Beta_min[:,1:4])*Beta_14_stretch + 0.001
        Gamma_min       = Gamma_min - (Gamma_max-Gamma_min)*Gamma_stretch - 0.001
        Gamma_max       = Gamma_max + (Gamma_max-Gamma_min)*Gamma_stretch + 0.001

        # set up optimization problem
        const_vals = tile(const_vals_ti,(num_trajectory_samples,1))
        path_vals  = c_[ sigma_numerically_stable[:,0], d1sigma_ds1_numerically_stable[:,0], d2sigma_ds2_numerically_stable[:,0], d3sigma_ds3_numerically_stable[:,0], d4sigma_ds4_numerically_stable[:,0], \
                         sigma_numerically_stable[:,1], d1sigma_ds1_numerically_stable[:,1], d2sigma_ds2_numerically_stable[:,1], d3sigma_ds3_numerically_stable[:,1], d4sigma_ds4_numerically_stable[:,1], \
                         sigma_numerically_stable[:,2], d1sigma_ds1_numerically_stable[:,2], d2sigma_ds2_numerically_stable[:,2], d3sigma_ds3_numerically_stable[:,2], d4sigma_ds4_numerically_stable[:,2], \
                         sigma_numerically_stable[:,3], d1sigma_ds1_numerically_stable[:,3], d2sigma_ds2_numerically_stable[:,3], d3sigma_ds3_numerically_stable[:,3], d4sigma_ds4_numerically_stable[:,3], \
                         ds_numerically_stable ]
        ref_vals = sdotN_numerically_stable[1]

        def _unpack_Alpha_1d(Alpha_1d):

            Beta_1d_begin,Beta_1d_end   = 0,           0           + num_trajectory_samples*num_beta_dims
            Gamma_1d_begin,Gamma_1d_end = Beta_1d_end, Beta_1d_end + num_trajectory_samples*num_gamma_dims

            Beta_1d  = Alpha_1d[Beta_1d_begin:Beta_1d_end]
            Gamma_1d = Alpha_1d[Gamma_1d_begin:Gamma_1d_end]
            Beta     = Beta_1d.reshape((num_trajectory_samples,num_beta_dims))
            Gamma    = Gamma_1d.reshape((num_trajectory_samples,num_gamma_dims))

            return Beta,Gamma

        def _refine_dt(dt):

            dt_raw                 = dt
            dt_raw_threshold       = 0.0001
            num_trajectory_samples = dt.shape[0]
            t_tmp                  = arange(num_trajectory_samples)

            assert nonzero(dt_raw >= dt_raw_threshold)[0].shape[0] >= 2

            if dt_raw[0] < dt_raw_threshold:
                dt_good_ind_1 = nonzero(dt_raw >= dt_raw_threshold)[0][0]
                dt_good_ind_2 = nonzero(dt_raw >= dt_raw_threshold)[0][1]
                di_01         = dt_good_ind_1-0
                ddt_di_12     = (dt_raw[dt_good_ind_2]-dt_raw[dt_good_ind_1]) / (dt_good_ind_2 - dt_good_ind_1)
                dt_raw[0]     = max( dt_raw_threshold, dt_raw[dt_good_ind_1] - ddt_di_12*di_01 )
            if dt_raw[-1] < dt_raw_threshold:
                dt_good_ind_n2 = nonzero(dt_raw >= dt_raw_threshold)[0][-1]
                dt_good_ind_n3 = nonzero(dt_raw >= dt_raw_threshold)[0][-2]
                di_n1n2        = num_trajectory_samples-1-dt_good_ind_n2
                ddt_di_n2n3    = (dt_raw[dt_good_ind_n2]-dt_raw[dt_good_ind_n3]) / (dt_good_ind_n2 - dt_good_ind_n3)
                dt_raw[-1]     = max( dt_raw_threshold, dt_raw[dt_good_ind_n2] - ddt_di_n2n3*di_n1n2 )

            dt_func = scipy.interpolate.interp1d(t_tmp[dt_raw >= dt_raw_threshold], dt_raw[dt_raw >= dt_raw_threshold], kind="linear")
            dt      = dt_func(t_tmp)

            return dt

        def _obj_func(Alpha_1d):

            global snopt_max_major_iterations
            global snopt_max_update_dt_current_iterations
            global snopt_major_iter_count
            global snopt_Alpha_opt_1d
            global snopt_current_best_obj_val
            global snopt_Alpha_1d_current_best
            global snopt_obj_vals
            global dt_current_vals

            Beta,Gamma = _unpack_Alpha_1d(Alpha_1d)

            const_and_path_and_beta_vals         = c_[ const_vals, path_vals, Beta ]
            const_and_path_and_ref_and_beta_vals = c_[ const_vals, path_vals, ref_vals, Beta ]

            if snopt_major_iter_count < snopt_max_update_dt_current_iterations:
                dt_current_vals = ds_numerically_stable[:-1] / matrix(Beta[:-1,0]).A1
                # dt_current_vals = _refine_dt(dt_current_vals)

            const_and_path_and_beta_and_gamma_and_dt_current_next_vals = c_[ const_vals[:-1], path_vals[:-1], Beta[:-1], Beta[1:], Gamma[:-1], dt_current_vals ]

            assert opt_problem_type == "mintime" or opt_problem_type == "track"

            if opt_problem_type == "mintime":
                J_ti = J_mintime_ti_vectorized_autowrap(const_and_path_and_beta_vals)
            if opt_problem_type == "track":
                J_ti = J_tracking_ti_vectorized_autowrap(const_and_path_and_ref_and_beta_vals)

            g_x        = g_x_ti_vectorized_autowrap(const_and_path_and_beta_vals)
            g_dynamics = g_dynamics_ti_vectorized_autowrap(const_and_path_and_beta_and_gamma_and_dt_current_next_vals)

            const_and_x_vals = c_[ const_vals, g_x ]

            B     = quadrotor3d.B_vectorized_autowrap(const_and_x_vals)
            Bpinv = zeros((num_trajectory_samples,4,6))

            for ti in range(num_trajectory_samples):
                Bpinv[ti] = linalg.pinv(B[ti])

            const_and_path_and_Bpinv_and_beta_vals = c_[ const_vals, path_vals, Bpinv.reshape((num_trajectory_samples,-1)), Beta ]

            g_u = g_u_ti_vectorized_autowrap(const_and_path_and_Bpinv_and_beta_vals)

            J    = sum(J_ti)
            g_1d = hstack( [ matrix(g_dynamics).A1, matrix(g_x).A1, matrix(g_u).A1 ] )

            Beta_min_constraint_violations  = zeros_like(Beta)
            Beta_max_constraint_violations  = zeros_like(Beta)
            Gamma_min_constraint_violations = zeros_like(Gamma)
            Gamma_max_constraint_violations = zeros_like(Gamma)

            x_min_constraint_violations = zeros_like(g_x)
            x_max_constraint_violations = zeros_like(g_x)
            u_min_constraint_violations = zeros_like(g_u)
            u_max_constraint_violations = zeros_like(g_u)

            Beta_min_constraint_violations[ Beta < Beta_min ]    = ( abs( Beta - Beta_min )   / ( Beta_max - Beta_min ) )[ Beta < Beta_min ]
            Beta_max_constraint_violations[ Beta > Beta_max ]    = ( abs( Beta - Beta_min )   / ( Beta_max - Beta_min ) )[ Beta > Beta_max ]
            Gamma_min_constraint_violations[ Gamma < Gamma_min ] = ( abs( Gamma - Gamma_min ) / ( Gamma_max - Gamma_min ) )[ Gamma < Gamma_min ]
            Gamma_max_constraint_violations[ Gamma > Gamma_max ] = ( abs( Gamma - Gamma_min ) / ( Gamma_max - Gamma_min ) )[ Gamma > Gamma_max ]

            x_min_constraint_violations[ g_x < x_min_ti.T ] = ( abs( g_x - x_min_ti.T.A ) / ( x_max_ti.T.A - x_min_ti.T.A ) )[ g_x < x_min_ti.T ]
            x_max_constraint_violations[ g_x > x_max_ti.T ] = ( abs( g_x - x_max_ti.T.A ) / ( x_max_ti.T.A - x_min_ti.T.A ) )[ g_x > x_max_ti.T ]
            u_min_constraint_violations[ g_u < u_min_ti.T ] = ( abs( g_u - u_min_ti.T.A ) / ( u_max_ti.T.A - u_min_ti.T.A ) )[ g_u < u_min_ti.T ]
            u_max_constraint_violations[ g_u > u_max_ti.T ] = ( abs( g_u - u_max_ti.T.A ) / ( u_max_ti.T.A - u_min_ti.T.A ) )[ g_u > u_max_ti.T ]

            snopt_obj_vals[snopt_major_iter_count,0] = J
            snopt_obj_vals[snopt_major_iter_count,1] = sum(norm(g_dynamics,axis=1))
            snopt_major_iter_count                   = snopt_major_iter_count+1

            set_printoptions(suppress=True)
            print "SNOPT major iter: %d, Obj: %f, equalities: %f, inequalities: %f %f %f %f, var limits: %f %f %f %f" % \
                ( snopt_major_iter_count,                       \
                  J,                                            \
                  sum(square(g_dynamics)),                      \
                  sum(square(x_min_constraint_violations)),     \
                  sum(square(x_max_constraint_violations)),     \
                  sum(square(u_min_constraint_violations)),     \
                  sum(square(u_max_constraint_violations)),     \
                  sum(square(Beta_min_constraint_violations)),  \
                  sum(square(Beta_max_constraint_violations)),  \
                  sum(square(Gamma_min_constraint_violations)), \
                  sum(square(Gamma_max_constraint_violations)) )

            feasible = False

            if allclose(g_dynamics,0.0)                      and \
               allclose(Beta_min_constraint_violations,0.0)  and \
               allclose(Beta_max_constraint_violations,0.0)  and \
               allclose(Gamma_min_constraint_violations,0.0) and \
               allclose(Gamma_max_constraint_violations,0.0) and \
               allclose(x_min_constraint_violations,0.0)     and \
               allclose(x_max_constraint_violations,0.0)     and \
               allclose(u_min_constraint_violations,0.0)     and \
               allclose(u_max_constraint_violations,0.0):
                feasible = True

            if feasible and J < snopt_current_best_obj_val:
                snopt_Alpha_1d_current_best = Alpha_1d

            if feasible and snopt_major_iter_count >= snopt_max_major_iterations:
                snopt_Alpha_opt_1d = snopt_Alpha_1d_current_best
                fail               = 1
            else:
                fail = 0

            return J, g_1d, fail

        def _grad_func(Alpha_1d, J, g_1d, compute_nonzero_only=False):

            global snopt_max_major_iterations
            global snopt_max_update_dt_current_iterations
            global snopt_major_iter_count
            global snopt_Alpha_opt_1d
            global snopt_current_best_obj_val
            global snopt_Alpha_1d_current_best
            global dt_current_vals

            dJ_dBeta          = zeros((num_trajectory_samples,num_beta_dims))
            dJ_dGamma         = zeros((num_trajectory_samples,num_gamma_dims))
            dgx_dBeta         = zeros((num_constraints_1d_g_x,num_decision_vars_1d_Beta))
            dgx_dGamma        = zeros((num_constraints_1d_g_x,num_decision_vars_1d_Gamma))
            dgu_dBeta         = zeros((num_constraints_1d_g_u,num_decision_vars_1d_Beta))
            dgu_dGamma        = zeros((num_constraints_1d_g_u,num_decision_vars_1d_Gamma))
            dgdynamics_dBeta  = zeros((num_constraints_1d_g_dynamics,num_decision_vars_1d_Beta))
            dgdynamics_dGamma = zeros((num_constraints_1d_g_dynamics,num_decision_vars_1d_Gamma))

            if not compute_nonzero_only:

                Beta,Gamma = _unpack_Alpha_1d(Alpha_1d)

                const_and_path_and_beta_vals         = c_[ const_vals, path_vals, Beta ]
                const_and_path_and_ref_and_beta_vals = c_[ const_vals, path_vals, ref_vals, Beta ]

                if snopt_major_iter_count < snopt_max_update_dt_current_iterations:
                    dt_current_vals = ds_numerically_stable[:-1] / matrix(Beta[:-1,0]).A1
                    # dt_current_vals = _refine_dt(dt_current_vals)

                const_and_path_and_beta_and_gamma_and_dt_current_next_vals = c_[ const_vals[:-1], path_vals[:-1], Beta[:-1], Beta[1:], Gamma[:-1], dt_current_vals ]

                g_x = g_x_ti_vectorized_autowrap(const_and_path_and_beta_vals)

                const_and_x_vals = c_[ const_vals, g_x ]

                B     = quadrotor3d.B_vectorized_autowrap(const_and_x_vals)
                Bpinv = zeros((num_trajectory_samples,4,6))

                for ti in range(num_trajectory_samples): Bpinv[ti] = linalg.pinv(B[ti])

                const_and_path_and_Bpinv_and_beta_vals = c_[ const_vals, path_vals, Bpinv.reshape((num_trajectory_samples,-1)), Beta ]

                assert opt_problem_type == "mintime" or opt_problem_type == "track"

                if opt_problem_type == "mintime":
                    dJ_dBeta = dJmintimeti_dbeta_vectorized_autowrap(const_and_path_and_beta_vals)
                if opt_problem_type == "track":
                    dJ_dBeta = dJtrackingti_dbeta_vectorized_autowrap(const_and_path_and_ref_and_beta_vals)

                dgx_dBeta_block                 = dgxti_dbeta_vectorized_autowrap(const_and_path_and_beta_vals)
                dgdynamics_dBeta_current_block  = dgdynamicsti_dbetacurrent_vectorized_autowrap(const_and_path_and_beta_and_gamma_and_dt_current_next_vals)
                dgdynamics_dBeta_next_block     = dgdynamicsti_dbetanext_vectorized_autowrap(const_and_path_and_beta_and_gamma_and_dt_current_next_vals)
                dgdynamics_dGamma_current_block = dgdynamicsti_dgammacurrent_vectorized_autowrap(const_and_path_and_beta_and_gamma_and_dt_current_next_vals)
                dgu_dBeta_block                 = dguti_dbeta_vectorized_autowrap(const_and_path_and_Bpinv_and_beta_vals)

            for ti in range(num_trajectory_samples):

                gi_x_begin = (ti+0)*num_dims_g_x_ti
                gi_x_end   = (ti+1)*num_dims_g_x_ti
                gi_u_begin = (ti+0)*num_dims_g_u_ti
                gi_u_end   = (ti+1)*num_dims_g_u_ti

                bi_begin = (ti+0)*num_beta_dims
                bi_end   = (ti+1)*num_beta_dims

                if compute_nonzero_only:
                    dJ_dBeta[ti]                                   = 1
                    dgx_dBeta[gi_x_begin:gi_x_end,bi_begin:bi_end] = 1
                    dgu_dBeta[gi_u_begin:gi_u_end,bi_begin:bi_end] = 1
                else:
                    dgx_dBeta[gi_x_begin:gi_x_end,bi_begin:bi_end] = dgx_dBeta_block[ti]
                    dgu_dBeta[gi_u_begin:gi_u_end,bi_begin:bi_end] = dgu_dBeta_block[ti]

            for ti in range(num_constraints_g_dynamics):

                gi_begin = (ti+0)*num_dims_g_dynamics_ti
                gi_end   = (ti+1)*num_dims_g_dynamics_ti

                beta_current_begin  = (ti+0)*num_beta_dims
                beta_current_end    = (ti+1)*num_beta_dims
                beta_next_begin     = (ti+1)*num_beta_dims
                beta_next_end       = (ti+2)*num_beta_dims
                gamma_current_begin = (ti+0)*num_gamma_dims
                gamma_current_end   = (ti+1)*num_gamma_dims

                if compute_nonzero_only:
                    dgdynamics_dBeta[gi_begin:gi_end,beta_current_begin:beta_current_end]    = 1
                    dgdynamics_dBeta[gi_begin:gi_end,beta_next_begin:beta_next_end]          = 1
                    dgdynamics_dGamma[gi_begin:gi_end,gamma_current_begin:gamma_current_end] = 1
                else:
                    dgdynamics_dBeta[gi_begin:gi_end,beta_current_begin:beta_current_end]    = dgdynamics_dBeta_current_block[ti]
                    dgdynamics_dBeta[gi_begin:gi_end,beta_next_begin:beta_next_end]          = dgdynamics_dBeta_next_block[ti]
                    dgdynamics_dGamma[gi_begin:gi_end,gamma_current_begin:gamma_current_end] = matrix(dgdynamics_dGamma_current_block[ti]).T

            dJ_dAlpha_1d = hstack( [ matrix(dJ_dBeta).A1, matrix(dJ_dGamma).A1 ] )

            dgdynamics_dAlpha = c_[ dgdynamics_dBeta, dgdynamics_dGamma ]
            dgx_dAlpha        = c_[ dgx_dBeta,        dgx_dGamma ]
            dgu_dAlpha        = c_[ dgu_dBeta,        dgu_dGamma ]

            dg_dAlpha = r_[ dgdynamics_dAlpha, dgx_dAlpha, dgu_dAlpha ]

            fail = 0
            return matrix(dJ_dAlpha_1d).A, dg_dAlpha, fail

        def _obj_grad_func(status,Alpha_1d,needF,needG,cu,iu,ru):

            J, g_1d, obj_fail                  = _obj_func(Alpha_1d)
            dJ_dAlpha_1d, dg_dAlpha, grad_fail = _grad_func(Alpha_1d,J,g_1d)
            J_g_1d                             = hstack( [ J, g_1d, snopt_dummy_val ] )
            dJ_dAlpha_dg_dAlpha                = r_[ dJ_dAlpha_1d, dg_dAlpha ]
            dJ_dAlpha_dg_dAlpha_nonzero_vals   = dJ_dAlpha_dg_dAlpha[dJ_dAlpha_dg_dAlpha_nonzero_inds]

            if obj_fail == 1 or grad_fail == 1:
                J_g_1d = None
                status = -1

            return status, J_g_1d, dJ_dAlpha_dg_dAlpha_nonzero_vals

        inf   = 1.0e20
        snopt = SNOPT_solver()

        snopt.setOption('Verbose',False)
        snopt.setOption('Solution print',False)
        snopt.setOption('Major print level',0)
        snopt.setOption('Print level',0)

        snopt_obj_row      = 1
        snopt_num_funcs_1d = num_constraints_1d_g_dynamics + num_constraints_1d_g_x + num_constraints_1d_g_u + 1
        snopt_num_vars_1d  = num_decision_vars_1d_Beta + num_decision_vars_1d_Gamma
        snopt_dummy_val    = 0.0
        snopt_dummy_array  = zeros((1,snopt_num_vars_1d))

        global snopt_max_major_iterations
        global snopt_max_update_dt_current_iterations
        global snopt_major_iter_count
        global snopt_Alpha_opt_1d
        global snopt_current_best_obj_val
        global snopt_Alpha_1d_current_best
        global snopt_obj_vals
        global dt_current_vals

        snopt_major_iter_count      = 0
        snopt_obj_vals              = -1*ones((10000,2))
        snopt_Alpha_opt_1d          = None
        snopt_current_best_obj_val  = 10e9
        snopt_Alpha_1d_current_best = None
        dt_current_vals             = None

        Alpha_min = hstack( [ matrix(Beta_min).A1, matrix(Gamma_min).A1 ] )
        Alpha_max = hstack( [ matrix(Beta_max).A1, matrix(Gamma_max).A1 ] )

        Beta_0  = matrix( c_[ matrix(sdotN_feasible[1]).A1, matrix(sdotN_feasible[2]).A1, matrix(sdotN_feasible[3]).A1, matrix(sdotN_feasible[4]).A1 ] ).A1
        Gamma_0 = matrix( c_[ matrix(sdotN_feasible[5]).A1 ] ).A1
        Alpha_0 = hstack( [ Beta_0, Gamma_0 ] )

        print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Calculating objective value on initial guess..."        
        _obj_func(Alpha_0)

        snopt_major_iter_count = 0

        J_g_1d_min = hstack( [ -inf, zeros(num_constraints_1d_g_dynamics), matrix(tile(x_min_ti.A1,(1,num_constraints_g_x))).A1, matrix(tile(u_min_ti.A1,(1,num_constraints_g_u))).A1, snopt_dummy_val ] )
        J_g_1d_max = hstack( [  inf, zeros(num_constraints_1d_g_dynamics), matrix(tile(x_max_ti.A1,(1,num_constraints_g_x))).A1, matrix(tile(u_max_ti.A1,(1,num_constraints_g_u))).A1, snopt_dummy_val ] )

        dJ_dAlpha_dg_dAlpha_const       = r_[ zeros((snopt_num_funcs_1d,snopt_num_vars_1d)), snopt_dummy_array ]
        dJ_dAlpha_dg_dAlpha_const[-1,0] = 10e-9

        dJ_dAlpha_nonzero, dg_dAlpha_nonzero, fail = _grad_func(Alpha_0, J=None, g_1d=None, compute_nonzero_only=True)

        dJ_dAlpha_dg_dAlpha_nonzero      = r_[ dJ_dAlpha_nonzero, dg_dAlpha_nonzero, snopt_dummy_array ]
        dJ_dAlpha_dg_dAlpha_nonzero_inds = dJ_dAlpha_dg_dAlpha_nonzero.nonzero()

        try:
            print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Solving optimization problem..."
            sys_time_begin = time.time()

            snopt.snopta(
                name="flashlight.trajectory_optimization.quadrotor3d_fixed_path",
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

            if snopt.exit == 0 and snopt.info == 1:
                Alpha_opt_1d = snopt.x
                print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished successfully with extra_iters_numerically_stable = %d." % extra_iters_numerically_stable
                sys_time_end = time.time()
                print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished solving optimization problem (%.03f seconds)." % (sys_time_end - sys_time_begin)
                break
            else:
                print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Encountered numerical difficulties, no feasibe trajectory found, trying again with more conservative goal trajectory."
                sys_time_end = time.time()
                print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished solving optimization problem (%.03f seconds)." % (sys_time_end - sys_time_begin)
        except (SystemError):
            if snopt_Alpha_opt_1d is not None:
                Alpha_opt_1d = snopt_Alpha_1d_current_best
                print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Maximum number of iterations reached, returning current best trajectory with extra_iters_numerically_stable = %d." % extra_iters_numerically_stable
                sys_time_end = time.time()
                print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished solving optimization problem (%.03f seconds)." % (sys_time_end - sys_time_begin)
                break
            else:
                print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Encountered numerical difficulties, no feasibe trajectory found, trying again with more conservative goal trajectory."
                sys_time_end = time.time()
                print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Finished solving optimization problem (%.03f seconds)." % (sys_time_end - sys_time_begin)

    if Alpha_opt_1d is None:
        print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: No feasibe trajectory found, returning initial feasible guess trajectory."
        Alpha_opt_1d = Alpha_0

    solver_time_end = sys_time_end
    solver_time     = solver_time_end - solver_time_begin
    print "flashlight.trajectory_optimization.quadrotor3d_fixed_path: Total solver time was %.03f seconds." % solver_time

    Beta_opt,Gamma_opt = _unpack_Alpha_1d(Alpha_opt_1d)

    const_and_path_and_beta_vals = c_[ const_vals, path_vals, Beta_opt ]
    X_opt                        = g_x_ti_vectorized_autowrap(const_and_path_and_beta_vals)
    const_and_x_vals             = c_[ const_vals, X_opt ]

    B     = quadrotor3d.B_vectorized_autowrap(const_and_x_vals)
    Bpinv = zeros((num_trajectory_samples,4,6))

    for ti in range(num_trajectory_samples): Bpinv[ti] = linalg.pinv(B[ti])

    const_and_path_and_Bpinv_and_beta_vals = c_[ const_vals, path_vals, Bpinv.reshape((num_trajectory_samples,-1)), Beta_opt ]
    U_opt                                  = g_u_ti_vectorized_autowrap(const_and_path_and_Bpinv_and_beta_vals)

    dt_opt        = ds_numerically_stable / Beta_opt[:,0]
    dt_opt        = _refine_dt(dt_opt)
    dt_opt_cumsum = cumsum(dt_opt[:-1])
    t_opt         = hstack( [ t_numerically_stable[0], t_numerically_stable[0] + dt_opt_cumsum ] )
    T_final_opt   = dt_opt_cumsum[-1]

    return s_numerically_stable,Beta_opt,Gamma_opt,X_opt,U_opt,t_opt,T_final_opt,solver_time,snopt_obj_vals
