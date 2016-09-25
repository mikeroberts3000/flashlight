from pylab import *

import matplotlib.cm
import os
import sklearn.preprocessing
import subprocess
import sympy
import time
import transformations

import gradient_utils
import ipython_display_utils
import path_utils
import sympy_utils
import trig_utils



front_prop_and_quad_positive_x_axis_angle = pi/4
rear_prop_and_quad_negative_x_axis_angle  = pi/4
y_axis_torque_per_newton_per_prop         = 1

alpha = front_prop_and_quad_positive_x_axis_angle
beta  = rear_prop_and_quad_negative_x_axis_angle
gamma = y_axis_torque_per_newton_per_prop

d = 1.0                          # distance from arm to center
m = 1.0                          # mass
g = 9.8                          # gravity
I = m*d**2.0*matrix(identity(3)) # moment of intertia for body

f_gravity_world  = matrix([0,-m*g,0]).T
f_external_world = f_gravity_world

num_q_dims  = 6
num_x_dims  = 2*num_q_dims
num_u_dims  = 4
num_dt_dims = 1

build_sympy_modules_on_import = False
# build_sympy_modules_on_import = False

const_vals = hstack( [ alpha, beta, gamma, d, m, I.A1, f_external_world.A1 ] )



print "flashlight.quadrotor_3d: Constructing sympy symbols..."
sys_time_begin = time.time()

# constants
alpha_expr = sympy.Symbol("alpha", real=True)
beta_expr  = sympy.Symbol("beta",  real=True)
gamma_expr = sympy.Symbol("gamma", real=True)
d_expr     = sympy.Symbol("d",     real=True)
m_expr     = sympy.Symbol("m",     real=True)

I_expr, I_expr_entries                   = sympy_utils.construct_matrix_and_entries("I",(3,3),   real=True)
f_external_expr, f_external_expr_entries = sympy_utils.construct_matrix_and_entries("f_e",(3,1), real=True)

# variables
t_expr = sympy.Symbol("t", real=True)

p_z_expr   = sympy.Symbol("p_z",   real=True)(t_expr)
p_y_expr   = sympy.Symbol("p_y",   real=True)(t_expr)
p_x_expr   = sympy.Symbol("p_x",   real=True)(t_expr)
theta_expr = sympy.Symbol("theta", real=True)(t_expr)
psi_expr   = sympy.Symbol("psi",   real=True)(t_expr)
phi_expr   = sympy.Symbol("phi",   real=True)(t_expr)

# state and control vectors
q_expr                 = sympy.Matrix([p_z_expr,p_y_expr,p_x_expr,theta_expr,psi_expr,phi_expr])
q_dot_expr             = q_expr.diff(t_expr)
x_expr                 = sympy_utils.construct_matrix_from_block_matrix( sympy.Matrix( [ [q_expr], [q_dot_expr] ] ) )    
u_expr, u_expr_entries = sympy_utils.construct_matrix_and_entries("u",(num_u_dims,1))

# symbols to solve for g_dynamics, given x_current x_next u_current
# x_current_expr, x_current_expr_entries = sympy_utils.construct_matrix_and_entries("x_current", (num_x_dims,1), real=True)
# x_next_expr,    x_next_expr_entries    = sympy_utils.construct_matrix_and_entries("x_next",    (num_x_dims,1), real=True)
# u_current_expr, u_current_expr_entries = sympy_utils.construct_matrix_and_entries("u_current", (num_u_dims,1), real=True)
# dt_current_expr                        = sympy.Symbol("delta_t_current", real=True)

# symbol collections
const_syms                                                   = hstack( [ alpha_expr, beta_expr, gamma_expr, d_expr, m_expr, matrix(I_expr).A1, matrix(f_external_expr).A1 ] )
const_and_x_syms                                             = hstack( [ const_syms, matrix(x_expr).A1 ] )
const_and_x_and_u_syms                                       = hstack( [ const_syms, matrix(x_expr).A1, matrix(u_expr).A1 ] )
# const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms = hstack( [ const_syms, matrix(x_current_expr).A1, matrix(x_next_expr).A1, matrix(u_current_expr).A1, dt_current_expr ] )

sys_time_end = time.time()
print "flashlight.quadrotor_3d: Finished constructing sympy symbols (%.03f seconds)." % (sys_time_end - sys_time_begin)



def construct_manipulator_matrix_expressions(x_expr,t_expr):

    print "flashlight.quadrotor_3d: Constructing manipulator matrix expressions..."

    theta_expr     = x_expr[3,0]
    psi_expr       = x_expr[4,0]
    phi_expr       = x_expr[5,0]
    theta_dot_expr = x_expr[9,0]
    psi_dot_expr   = x_expr[10,0]
    phi_dot_expr   = x_expr[11,0]

    R_z_theta_expr = sympy_utils.construct_axis_aligned_rotation_matrix_right_handed(theta_expr,0)
    R_y_psi_expr   = sympy_utils.construct_axis_aligned_rotation_matrix_right_handed(psi_expr,1)
    R_x_phi_expr   = sympy_utils.construct_axis_aligned_rotation_matrix_right_handed(phi_expr,2)

    R_expr         = sympy.trigsimp(R_y_psi_expr*R_z_theta_expr*R_x_phi_expr)
    R_dot_expr     = sympy.trigsimp(R_expr.diff(t_expr))
    R_dot_R_T_expr = sympy.trigsimp(R_dot_expr*R_expr.T)

    omega_z_terms_expr = sympy_utils.collect_into_dict_include_zero_and_constant_terms( R_dot_R_T_expr[2,1],  [theta_dot_expr,psi_dot_expr,phi_dot_expr] )
    omega_y_terms_expr = sympy_utils.collect_into_dict_include_zero_and_constant_terms( -R_dot_R_T_expr[2,0], [theta_dot_expr,psi_dot_expr,phi_dot_expr] )
    omega_x_terms_expr = sympy_utils.collect_into_dict_include_zero_and_constant_terms( R_dot_R_T_expr[1,0],  [theta_dot_expr,psi_dot_expr,phi_dot_expr] )

    A_expr = sympy.Matrix( [ \
        [ omega_z_terms_expr[theta_dot_expr], omega_z_terms_expr[psi_dot_expr], omega_z_terms_expr[phi_dot_expr] ], \
        [ omega_y_terms_expr[theta_dot_expr], omega_y_terms_expr[psi_dot_expr], omega_y_terms_expr[phi_dot_expr] ], \
        [ omega_x_terms_expr[theta_dot_expr], omega_x_terms_expr[psi_dot_expr], omega_x_terms_expr[phi_dot_expr] ] ] )

    A_dot_expr = sympy.trigsimp(A_expr.diff(t_expr))

    R_world_from_body_expr = R_expr
    R_body_from_world_expr = R_world_from_body_expr.T

    euler_dot_expr         = sympy.Matrix([theta_dot_expr,psi_dot_expr,phi_dot_expr])
    omega_in_body_expr     = sympy.trigsimp(R_body_from_world_expr*A_expr*euler_dot_expr)
    I_omega_in_body_X_expr = sympy_utils.construct_cross_product_left_term_matrix_from_vector(I_expr*omega_in_body_expr)

    M_thrust_body_from_control_expr = sympy.Matrix([[0,0,0,0],[1,1,1,1],[0,0,0,0]])

    M_torque_body_from_control_expr = sympy.Matrix(([ \
        [-d_expr*sympy.cos(alpha_expr),d_expr*sympy.cos(beta_expr),d_expr*sympy.cos(beta_expr),-d_expr*sympy.cos(alpha_expr)], \
        [gamma_expr,-gamma_expr,gamma_expr,-gamma_expr], \
        [d_expr*sympy.sin(alpha_expr),d_expr*sympy.sin(beta_expr),-d_expr*sympy.sin(beta_expr),-d_expr*sympy.sin(alpha_expr)]]))

    H_00_expr    = sympy.Matrix(m_expr*sympy.eye(3))
    H_11_expr    = sympy.trigsimp(I_expr*R_body_from_world_expr*A_expr)
    H_zeros_expr = sympy.Matrix.zeros(3,3)

    C_11_expr    = I_expr*R_body_from_world_expr*A_dot_expr - I_omega_in_body_X_expr*R_body_from_world_expr*A_expr
    C_zeros_expr = sympy.Matrix.zeros(3,3)

    G_0_expr = -f_external_expr
    G_1_expr = sympy.Matrix.zeros(3,1)

    B_0_expr = R_world_from_body_expr*M_thrust_body_from_control_expr
    B_1_expr = M_torque_body_from_control_expr

    H_expr = sympy_utils.construct_matrix_from_block_matrix( sympy.Matrix( [ [H_00_expr,    H_zeros_expr], [H_zeros_expr, H_11_expr] ] ) )
    C_expr = sympy_utils.construct_matrix_from_block_matrix( sympy.Matrix( [ [C_zeros_expr, C_zeros_expr], [C_zeros_expr, C_11_expr] ] ) )
    G_expr = sympy_utils.construct_matrix_from_block_matrix( sympy.Matrix( [ [G_0_expr],                   [G_1_expr] ] ) )
    B_expr = sympy_utils.construct_matrix_from_block_matrix( sympy.Matrix( [ [B_0_expr],                   [B_1_expr] ] ) )

    print "flashlight.quadrotor_3d: Finished constructing manipulator matrix expressions."

    return H_expr,C_expr,G_expr,B_expr



def build_sympy_modules():

    print "flashlight.quadrotor_3d: Constructing sympy expressions..."
    sys_time_begin = time.time()

    # manipulator matrices
    H_expr,C_expr,G_expr,B_expr = construct_manipulator_matrix_expressions()

    # expressions to solve for df_dx and df_du
    q_dot_dot_expr = H_expr.inv()*(B_expr*u_expr - (C_expr*q_dot_expr + G_expr))
    x_dot_expr     = sympy_utils.construct_matrix_from_block_matrix( sympy.Matrix( [ [q_dot_expr], [q_dot_dot_expr] ] ) )
    df_dx_expr     = x_dot_expr.jacobian(x_expr)
    df_du_expr     = x_dot_expr.jacobian(u_expr)

    # expressions to solve for g_dynamics, given x_current x_next u_current dt_current
    # x_dot_current_expr           = x_dot_expr.subs( dict( zip(x_expr,x_current_expr) + zip(u_expr,u_current_expr) ) )
    # g_dynamics_ti_expr           = x_next_expr - (x_current_expr + x_dot_current_expr*dt_current_expr)
    # dgdynamicsti_dxcurrent_expr  = g_dynamics_ti_expr.jacobian(x_current_expr)
    # dgdynamicsti_dxnext_expr     = g_dynamics_ti_expr.jacobian(x_next_expr)
    # dgdynamicsti_ducurrent_expr  = g_dynamics_ti_expr.jacobian(u_current_expr)
    # dgdynamicsti_ddtcurrent_expr = g_dynamics_ti_expr.diff(dt_current_expr)

    sys_time_end = time.time()
    print "flashlight.quadrotor_3d: Finished constructing sympy expressions (%.03f seconds)." % (sys_time_end - sys_time_begin)

    print "flashlight.quadrotor_3d: Building sympy modules..."
    sys_time_begin = time.time()

    current_source_file_path = path_utils.get_current_source_file_path()

    sympy_utils.build_module_autowrap( expr=H_expr, syms=const_and_x_syms, module_name="quadrotor_3d_H", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympy_utils.build_module_autowrap( expr=C_expr, syms=const_and_x_syms, module_name="quadrotor_3d_C", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympy_utils.build_module_autowrap( expr=G_expr, syms=const_and_x_syms, module_name="quadrotor_3d_G", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympy_utils.build_module_autowrap( expr=B_expr, syms=const_and_x_syms, module_name="quadrotor_3d_B", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )

    sympy_utils.build_module_autowrap( expr=x_dot_expr, syms=const_and_x_and_u_syms, module_name="quadrotor_3d_x_dot", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympy_utils.build_module_autowrap( expr=df_dx_expr, syms=const_and_x_and_u_syms, module_name="quadrotor_3d_df_dx", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympy_utils.build_module_autowrap( expr=df_du_expr, syms=const_and_x_and_u_syms, module_name="quadrotor_3d_df_du", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )

    # sympy_utils.build_module_autowrap( expr=g_dynamics_ti_expr,           syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotor3d_g_dynamics_ti",           tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    # sympy_utils.build_module_autowrap( expr=dgdynamicsti_dxcurrent_expr,  syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotor3d_dgdynamicsti_dxcurrent",  tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    # sympy_utils.build_module_autowrap( expr=dgdynamicsti_dxnext_expr,     syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotor3d_dgdynamicsti_dxnext",     tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    # sympy_utils.build_module_autowrap( expr=dgdynamicsti_ducurrent_expr,  syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotor3d_dgdynamicsti_ducurrent",  tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    # sympy_utils.build_module_autowrap( expr=dgdynamicsti_ddtcurrent_expr, syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotor3d_dgdynamicsti_ddtcurrent", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor3d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )

    sys_time_end = time.time()
    print "flashlight.quadrotor_3d: Finished building sympy modules (%.03f seconds)." % (sys_time_end - sys_time_begin)



if build_sympy_modules_on_import:
    build_sympy_modules()

print "flashlight.quadrotor_3d: Loading sympy modules..."
sys_time_begin = time.time()

current_source_file_path = path_utils.get_current_source_file_path()

H_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_3d_H", path=current_source_file_path+"/data/quadrotor_3d" )
C_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_3d_C", path=current_source_file_path+"/data/quadrotor_3d" )
G_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_3d_G", path=current_source_file_path+"/data/quadrotor_3d" )
B_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_3d_B", path=current_source_file_path+"/data/quadrotor_3d" )

x_dot_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_3d_x_dot", path=current_source_file_path+"/data/quadrotor_3d" )
df_dx_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_3d_df_dx", path=current_source_file_path+"/data/quadrotor_3d" )
df_du_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_3d_df_du", path=current_source_file_path+"/data/quadrotor_3d" )

# g_dynamics_ti_autowrap           = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_g_dynamics_ti",           path=current_source_file_path+"/data/quadrotor3d" )
# dgdynamicsti_dxcurrent_autowrap  = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_dgdynamicsti_dxcurrent",  path=current_source_file_path+"/data/quadrotor3d" )
# dgdynamicsti_dxnext_autowrap     = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_dgdynamicsti_dxnext",     path=current_source_file_path+"/data/quadrotor3d" )
# dgdynamicsti_ducurrent_autowrap  = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_dgdynamicsti_ducurrent",  path=current_source_file_path+"/data/quadrotor3d" )
# dgdynamicsti_ddtcurrent_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_dgdynamicsti_ddtcurrent", path=current_source_file_path+"/data/quadrotor3d" )

H_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_3d_H_vectorized", path=current_source_file_path+"/data/quadrotor_3d" )
C_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_3d_C_vectorized", path=current_source_file_path+"/data/quadrotor_3d" )
G_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_3d_G_vectorized", path=current_source_file_path+"/data/quadrotor_3d" )
B_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_3d_B_vectorized", path=current_source_file_path+"/data/quadrotor_3d" )

# x_dot_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_x_dot_vectorized", path=current_source_file_path+"/data/quadrotor3d" )
# df_dx_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_df_dx_vectorized", path=current_source_file_path+"/data/quadrotor3d" )
# df_du_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_df_du_vectorized", path=current_source_file_path+"/data/quadrotor3d" )

# g_dynamics_ti_vectorized_autowrap           = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_g_dynamics_ti_vectorized",           path=current_source_file_path+"/data/quadrotor3d" )
# dgdynamicsti_dxcurrent_vectorized_autowrap  = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_dgdynamicsti_dxcurrent_vectorized",  path=current_source_file_path+"/data/quadrotor3d" )
# dgdynamicsti_dxnext_vectorized_autowrap     = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_dgdynamicsti_dxnext_vectorized",     path=current_source_file_path+"/data/quadrotor3d" )
# dgdynamicsti_ducurrent_vectorized_autowrap  = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_dgdynamicsti_ducurrent_vectorized",  path=current_source_file_path+"/data/quadrotor3d" )
# dgdynamicsti_ddtcurrent_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor3d_dgdynamicsti_ddtcurrent_vectorized", path=current_source_file_path+"/data/quadrotor3d" )

sys_time_end = time.time()
print "flashlight.quadrotor_3d: Finished loading sympy modules (%.03f seconds)." % (sys_time_end - sys_time_begin)



def pack_state(p, theta, psi, phi, p_dot, theta_dot, psi_dot, phi_dot):

    q     = matrix( [ p.item(0),     p.item(1),     p.item(2),     theta,     psi,     phi ] ).T
    q_dot = matrix( [ p_dot.item(0), p_dot.item(1), p_dot.item(2), theta_dot, psi_dot, phi_dot ] ).T
    x     = matrix( r_[ q.A1, q_dot.A1 ] ).T

    return x, q, q_dot

def unpack_state(x):

    p         = matrix( [ x.item(0), x.item(1), x.item(2) ] ).T
    theta     = x.item(3)
    psi       = x.item(4)
    phi       = x.item(5)
    p_dot     = matrix( [ x.item(6), x.item(7), x.item(8) ] ).T
    theta_dot = x.item(9)
    psi_dot   = x.item(10)
    phi_dot   = x.item(11)
    q         = matrix( [ p.item(0),     p.item(1),     p.item(2),     theta,     psi,     phi,    ] ).T
    q_dot     = matrix( [ p_dot.item(0), p_dot.item(1), p_dot.item(2), theta_dot, psi_dot, phi_dot ] ).T

    return p, theta, psi, phi, p_dot, theta_dot, psi_dot, phi_dot, q, q_dot

def pack_state_space_trajectory(p, theta, psi, phi, p_dot, theta_dot, psi_dot, phi_dot):

    q     = c_[ p,     theta,     psi,     phi ]
    q_dot = c_[ p_dot, theta_dot, psi_dot, phi_dot ]
    x     = c_[ q, q_dot ]

    return x, q, q_dot

def unpack_state_space_trajectory(x):

    p         = x[:,0:3]
    theta     = x[:,3]
    psi       = x[:,4]
    phi       = x[:,5]
    p_dot     = x[:,6:9]
    theta_dot = x[:,9]
    psi_dot   = x[:,10]
    phi_dot   = x[:,11]
    q         = x[:,0:6]
    q_dot     = x[:,6:12]

    return p, theta, psi, phi, p_dot, theta_dot, psi_dot, phi_dot, q, q_dot

def pack_state_space_trajectory_and_derivatives(q_qdot_qdotdot):

    p, p_dot, p_dot_dot, theta, theta_dot, theta_dot_dot, psi, psi_dot, psi_dot_dot, phi, phi_dot, phi_dot_dot = q_qdot_qdotdot

    q         = c_[ p,         theta,         psi,         phi ]
    q_dot     = c_[ p_dot,     theta_dot,     psi_dot,     phi_dot ]
    q_dot_dot = c_[ p_dot_dot, theta_dot_dot, psi_dot_dot, phi_dot_dot ]
    x         = c_[ q, q_dot ]

    return x, q, q_dot, q_dot_dot



def compute_manipulator_matrices(x):

    const_and_x_vals = hstack( [ const_vals, matrix(x).A1 ] )
    H                = matrix( sympy_utils.evaluate_anon_func( H_autowrap, const_and_x_vals ) )
    C                = matrix( sympy_utils.evaluate_anon_func( C_autowrap, const_and_x_vals ) )
    G                = matrix( sympy_utils.evaluate_anon_func( G_autowrap, const_and_x_vals ) )
    B                = matrix( sympy_utils.evaluate_anon_func( B_autowrap, const_and_x_vals ) )

    return H, C, G, B



def compute_x_dot(x,u):

    const_and_x_and_u_vals = hstack( [ const_vals, matrix(x).A1, matrix(u).A1 ] )
    x_dot                  = matrix( sympy_utils.evaluate_anon_func( x_dot_autowrap, const_and_x_and_u_vals ) )

    return x_dot



def compute_df_dx_and_df_du(x,u):

    const_and_x_and_u_vals = hstack( [ const_vals, matrix(x).A1, matrix(u).A1 ] )
    df_dx                  = matrix( sympy_utils.evaluate_anon_func( df_dx_autowrap, const_and_x_and_u_vals ) )
    df_du                  = matrix( sympy_utils.evaluate_anon_func( df_du_autowrap, const_and_x_and_u_vals ) )

    return df_dx, df_du



# def compute_dynamics_constraints_direct_transcription(X,U,dt):

#     g_dynamics = zeros((X.shape[0]-1,X.shape[1]))

#     for ti in range(X.shape[0]-1):

#         x_current = matrix(X[ti]).T
#         u_current = matrix(U[ti]).T
#         x_next    = matrix(X[ti+1]).T

#         if isinstance(dt, ndarray) or isinstance(dt, matrix):
#             dt_current = dt[ti]
#         else:
#             dt_current = dt

#         const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals = hstack( [ const_vals, x_current.A1, x_next.A1, u_current.A1, dt_current ] )

#         g_dynamics[ti] = matrix( sympy_utils.evaluate_anon_func( g_dynamics_ti_autowrap, const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals ) ).A1

#     return g_dynamics



# def compute_dynamics_constraints_jacobian_direct_transcription(X,U,dt):

#     num_trajectory_samples      = X.shape[0]
#     num_dynamics_constraints_1d = (X.shape[0]-1) * X.shape[1]
#     num_decision_vars_X         = X.shape[0] * X.shape[1]
#     num_decision_vars_U         = U.shape[0] * U.shape[1]

#     if isinstance(dt, ndarray) or isinstance(dt, matrix):
#         num_decision_vars_dt = dt.shape[0] * dt.shape[1]
#     else:
#         num_decision_vars_dt = 1

#     dgdynamics_dX  = zeros((num_dynamics_constraints_1d,num_decision_vars_X))
#     dgdynamics_dU  = zeros((num_dynamics_constraints_1d,num_decision_vars_U))
#     dgdynamics_ddt = zeros((num_dynamics_constraints_1d,num_decision_vars_dt))

#     for ti in range(num_trajectory_samples-1):

#         x_current = matrix(X[ti]).T
#         u_current = matrix(U[ti]).T
#         x_next    = matrix(X[ti+1]).T

#         if isinstance(dt, ndarray) or isinstance(dt, matrix):
#             dt_current = dt[ti]
#         else:
#             dt_current = dt

#         const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals = hstack( [ const_vals, x_current.A1, x_next.A1, u_current.A1, dt_current ] )

#         gi_begin           = (ti+0)*num_dims_g_dynamics_ti
#         gi_end             = (ti+1)*num_dims_g_dynamics_ti
#         ai_x_current_begin = (ti+0)*num_x_dims
#         ai_x_current_end   = (ti+1)*num_x_dims
#         ai_x_next_begin    = (ti+1)*num_x_dims
#         ai_x_next_end      = (ti+2)*num_x_dims
#         ai_u_current_begin = (ti+0)*num_u_dims
#         ai_u_current_end   = (ti+1)*num_u_dims

#         if isinstance(dt, ndarray) or isinstance(dt, matrix):
#             ai_dt_current_begin = (ti+0)*num_dt_dims
#             ai_dt_current_end   = (ti+1)*num_dt_dims
#         else:
#             ai_dt_current_begin = 0*num_dt_dims
#             ai_dt_current_end   = 1*num_dt_dims

#         dgdynamicsti_dxcurrent  = sympy_utils.evaluate_anon_func( dgdynamicsti_dxcurrent_autowrap,  const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals )
#         dgdynamicsti_dxnext     = sympy_utils.evaluate_anon_func( dgdynamicsti_dxnext_autowrap,     const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals )
#         dgdynamicsti_ducurrent  = sympy_utils.evaluate_anon_func( dgdynamicsti_ducurrent_autowrap,  const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals )
#         dgdynamicsti_ddtcurrent = sympy_utils.evaluate_anon_func( dgdynamicsti_ddtcurrent_autowrap, const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals )

#         dgdynamics_dX[gi_begin:gi_end,ai_x_current_begin:ai_x_current_end]    = dgdynamicsti_dxcurrent
#         dgdynamics_dX[gi_begin:gi_end,ai_x_next_begin:ai_x_next_end]          = dgdynamicsti_dxnext
#         dgdynamics_dU[gi_begin:gi_end,ai_u_current_begin:ai_u_current_end]    = dgdynamicsti_ducurrent
#         dgdynamics_ddt[gi_begin:gi_end,ai_dt_current_begin:ai_dt_current_end] = dgdynamicsti_ddtcurrent

#     return dgdynamics_dX, dgdynamics_dU, dgdynamics_ddt



def compute_state_space_trajectory_and_derivatives(p,psi,dt,check_angles=True):

    num_timesteps = p.shape[0]

    psi = trig_utils.compute_continuous_angle_array(psi)

    p_dotN    = gradient_utils.gradients_vector_wrt_scalar_smooth_boundaries(p,dt,max_gradient=2,poly_deg=5)
    p_dot     = p_dotN[1]
    p_dot_dot = p_dotN[2]

    f_thrust_world            = m*p_dot_dot - f_external_world.T.A
    f_thrust_world_normalized = sklearn.preprocessing.normalize(f_thrust_world)

    z_axis_intermediate = c_[ cos(psi), zeros_like(psi), -sin(psi) ]
    y_axis              = f_thrust_world_normalized
    x_axis              = sklearn.preprocessing.normalize(cross(z_axis_intermediate, y_axis))
    z_axis              = sklearn.preprocessing.normalize(cross(y_axis, x_axis))

    theta          = zeros(num_timesteps)
    psi_recomputed = zeros(num_timesteps)
    phi            = zeros(num_timesteps)

    for ti in range(num_timesteps):

        z_axis_ti = c_[matrix(z_axis[ti]),0].T
        y_axis_ti = c_[matrix(y_axis[ti]),0].T
        x_axis_ti = c_[matrix(x_axis[ti]),0].T

        R_world_from_body_ti              = c_[z_axis_ti,y_axis_ti,x_axis_ti,[0,0,0,1]]
        psi_recomputed_ti,theta_ti,phi_ti = transformations.euler_from_matrix(R_world_from_body_ti,"ryxz")

        assert allclose(R_world_from_body_ti, transformations.euler_matrix(psi_recomputed_ti,theta_ti,phi_ti,"ryxz"))

        theta[ti]          = theta_ti
        psi_recomputed[ti] = psi_recomputed_ti
        phi[ti]            = phi_ti

    theta          = trig_utils.compute_continuous_angle_array(theta)
    psi_recomputed = trig_utils.compute_continuous_angle_array(psi_recomputed)
    phi            = trig_utils.compute_continuous_angle_array(phi)

    if check_angles:
        assert allclose(psi_recomputed, psi)

    psi = psi_recomputed

    theta_dotN = gradient_utils.gradients_scalar_wrt_scalar_smooth_boundaries(theta,dt,max_gradient=2,poly_deg=5)
    psi_dotN   = gradient_utils.gradients_scalar_wrt_scalar_smooth_boundaries(psi,dt,max_gradient=2,poly_deg=5)
    phi_dotN   = gradient_utils.gradients_scalar_wrt_scalar_smooth_boundaries(phi,dt,max_gradient=2,poly_deg=5)

    theta_dot = theta_dotN[1]
    psi_dot   = psi_dotN[1]
    phi_dot   = phi_dotN[1]

    theta_dot_dot = theta_dotN[2]
    psi_dot_dot   = psi_dotN[2]
    phi_dot_dot   = phi_dotN[2]

    return p, p_dot, p_dot_dot, theta, theta_dot, theta_dot_dot, psi, psi_dot, psi_dot_dot, phi, phi_dot, phi_dot_dot



def compute_control_trajectory(q_qdot_qdotdot):

    x, q, q_dot, q_dot_dot = pack_state_space_trajectory_and_derivatives(q_qdot_qdotdot)
    num_timesteps          = q.shape[0]
    u                      = zeros((num_timesteps,num_u_dims))

    const_vals_array = tile(const_vals,(num_timesteps,1))
    const_and_x_vals = c_[ const_vals_array, x ]

    H = H_vectorized_autowrap(const_and_x_vals)
    C = C_vectorized_autowrap(const_and_x_vals)
    G = G_vectorized_autowrap(const_and_x_vals)
    B = B_vectorized_autowrap(const_and_x_vals)

    for ti in range(num_timesteps):
        u[ti] = matrix( linalg.pinv(matrix(B[ti]))*( matrix(H[ti])*matrix(q_dot_dot[ti]).T + matrix(C[ti])*matrix(q_dot[ti]).T + matrix(G[ti]).T ) ).T

    return u



def check_differentially_flat_trajectory_numerically_stable(p,psi,dt):

    num_timesteps = len(p)

    psi = trig_utils.compute_continuous_angle_array(psi)

    p_dotN    = gradient_utils.gradients_vector_wrt_scalar_smooth_boundaries(p,dt,max_gradient=2,poly_deg=5)
    p_dot     = p_dotN[1]
    p_dot_dot = p_dotN[2]

    f_thrust_world            = m*p_dot_dot - f_external_world.T.A
    f_thrust_world_normalized = sklearn.preprocessing.normalize(f_thrust_world)

    z_axis_intermediate = c_[ cos(psi), zeros_like(psi), -sin(psi) ]
    y_axis              = f_thrust_world_normalized
    x_axis              = sklearn.preprocessing.normalize(cross(z_axis_intermediate, y_axis))
    z_axis              = sklearn.preprocessing.normalize(cross(y_axis, x_axis))

    theta          = zeros(num_timesteps)
    psi_recomputed = zeros(num_timesteps)
    phi            = zeros(num_timesteps)

    for ti in range(num_timesteps):

        z_axis_ti = c_[matrix(z_axis[ti]),0].T
        y_axis_ti = c_[matrix(y_axis[ti]),0].T
        x_axis_ti = c_[matrix(x_axis[ti]),0].T

        R_world_from_body_ti              = c_[z_axis_ti,y_axis_ti,x_axis_ti,[0,0,0,1]]
        psi_recomputed_ti,theta_ti,phi_ti = transformations.euler_from_matrix(R_world_from_body_ti,"ryxz")

        assert allclose(R_world_from_body_ti, transformations.euler_matrix(psi_recomputed_ti,theta_ti,phi_ti,"ryxz"))

        theta[ti]          = theta_ti
        psi_recomputed[ti] = psi_recomputed_ti
        phi[ti]            = phi_ti

    theta          = trig_utils.compute_continuous_angle_array(theta)
    psi_recomputed = trig_utils.compute_continuous_angle_array(psi_recomputed)
    phi            = trig_utils.compute_continuous_angle_array(phi)

    return allclose(psi_recomputed, psi)



def compute_normalized_constraint_violation_score(x,u,x_min_ti,x_max_ti,u_min_ti,u_max_ti):

    eps = 0.0005
    num_timesteps = x.shape[0]

    x_min_constraint_violation_score = abs( 2 * ( ( x - x_min_ti.T.A ) / ( x_max_ti.T.A - x_min_ti.T.A ) ) - 1 )
    x_max_constraint_violation_score = abs( 2 * ( ( x - x_min_ti.T.A ) / ( x_max_ti.T.A - x_min_ti.T.A ) ) - 1 )
    u_min_constraint_violation_score = abs( 2 * ( ( u - u_min_ti.T.A ) / ( u_max_ti.T.A - u_min_ti.T.A ) ) - 1 )
    u_max_constraint_violation_score = abs( 2 * ( ( u - u_min_ti.T.A ) / ( u_max_ti.T.A - u_min_ti.T.A ) ) - 1 )

    x_violations    = logical_or( x_min_constraint_violation_score > 1.0 + eps, x_max_constraint_violation_score > 1.0 + eps )
    u_violations    = logical_or( u_min_constraint_violation_score > 1.0 + eps, u_max_constraint_violation_score > 1.0 + eps )
    x_no_violations = logical_not(x_violations)
    u_no_violations = logical_not(u_violations)

    some_constraint_violated     = logical_or( any(x_violations,axis=1), any(u_violations,axis=1))
    all_constraints_not_violated = logical_not(some_constraint_violated)

    # sum up and normalize violations
    x_min_constraint_violation_score_violations_only = zeros_like(x_min_constraint_violation_score)
    x_max_constraint_violation_score_violations_only = zeros_like(x_max_constraint_violation_score)
    u_min_constraint_violation_score_violations_only = zeros_like(u_min_constraint_violation_score)
    u_max_constraint_violation_score_violations_only = zeros_like(u_max_constraint_violation_score)

    x_min_constraint_violation_score_violations_only[x_violations] = x_min_constraint_violation_score[x_violations]
    x_max_constraint_violation_score_violations_only[x_violations] = x_max_constraint_violation_score[x_violations]
    u_min_constraint_violation_score_violations_only[u_violations] = u_min_constraint_violation_score[u_violations]
    u_max_constraint_violation_score_violations_only[u_violations] = u_max_constraint_violation_score[u_violations]

    x_min_constraint_violation_score_violations_only_sum = zeros(num_timesteps)
    x_max_constraint_violation_score_violations_only_sum = zeros(num_timesteps)
    u_min_constraint_violation_score_violations_only_sum = zeros(num_timesteps)
    u_max_constraint_violation_score_violations_only_sum = zeros(num_timesteps)

    x_min_constraint_violation_score_violations_only_sum[some_constraint_violated] = sum(x_min_constraint_violation_score_violations_only,axis=1)[some_constraint_violated]
    x_max_constraint_violation_score_violations_only_sum[some_constraint_violated] = sum(x_max_constraint_violation_score_violations_only,axis=1)[some_constraint_violated]
    u_min_constraint_violation_score_violations_only_sum[some_constraint_violated] = sum(u_min_constraint_violation_score_violations_only,axis=1)[some_constraint_violated]
    u_max_constraint_violation_score_violations_only_sum[some_constraint_violated] = sum(u_max_constraint_violation_score_violations_only,axis=1)[some_constraint_violated]

    constraint_violation_score_violations_only_denorm = zeros(num_timesteps)
    constraint_violation_score_violations_only        = zeros(num_timesteps)

    if any(some_constraint_violated):

        constraint_violation_score_violations_only_denorm[some_constraint_violated] = \
            x_min_constraint_violation_score_violations_only_sum[some_constraint_violated] + \
            x_max_constraint_violation_score_violations_only_sum[some_constraint_violated] + \
            u_min_constraint_violation_score_violations_only_sum[some_constraint_violated] + \
            u_max_constraint_violation_score_violations_only_sum[some_constraint_violated]

        constraint_violation_score_violations_only[some_constraint_violated] = \
            constraint_violation_score_violations_only_denorm[some_constraint_violated] / max(constraint_violation_score_violations_only_denorm[some_constraint_violated])

    # sum up and normalize non-violations
    x_min_constraint_violation_score_no_violations = zeros_like(x_min_constraint_violation_score)
    x_max_constraint_violation_score_no_violations = zeros_like(x_max_constraint_violation_score)
    u_min_constraint_violation_score_no_violations = zeros_like(u_min_constraint_violation_score)
    u_max_constraint_violation_score_no_violations = zeros_like(u_max_constraint_violation_score)

    x_min_constraint_violation_score_no_violations[x_no_violations] = x_min_constraint_violation_score[x_no_violations]
    x_max_constraint_violation_score_no_violations[x_no_violations] = x_max_constraint_violation_score[x_no_violations]
    u_min_constraint_violation_score_no_violations[u_no_violations] = u_min_constraint_violation_score[u_no_violations]
    u_max_constraint_violation_score_no_violations[u_no_violations] = u_max_constraint_violation_score[u_no_violations]

    x_min_constraint_violation_score_no_violations_sum = zeros(num_timesteps)
    x_max_constraint_violation_score_no_violations_sum = zeros(num_timesteps)
    u_min_constraint_violation_score_no_violations_sum = zeros(num_timesteps)
    u_max_constraint_violation_score_no_violations_sum = zeros(num_timesteps)

    x_min_constraint_violation_score_no_violations_sum[all_constraints_not_violated] = sum(x_min_constraint_violation_score_no_violations,axis=1)[all_constraints_not_violated]
    x_max_constraint_violation_score_no_violations_sum[all_constraints_not_violated] = sum(x_max_constraint_violation_score_no_violations,axis=1)[all_constraints_not_violated]
    u_min_constraint_violation_score_no_violations_sum[all_constraints_not_violated] = sum(u_min_constraint_violation_score_no_violations,axis=1)[all_constraints_not_violated]
    u_max_constraint_violation_score_no_violations_sum[all_constraints_not_violated] = sum(u_max_constraint_violation_score_no_violations,axis=1)[all_constraints_not_violated]

    constraint_violation_score_no_violations_denorm = zeros(num_timesteps)
    constraint_violation_score_no_violations = zeros(num_timesteps)

    if any(all_constraints_not_violated):

        constraint_violation_score_no_violations_denorm[all_constraints_not_violated] = \
            x_min_constraint_violation_score_no_violations_sum[all_constraints_not_violated] + \
            x_max_constraint_violation_score_no_violations_sum[all_constraints_not_violated] + \
            u_min_constraint_violation_score_no_violations_sum[all_constraints_not_violated] + \
            u_max_constraint_violation_score_no_violations_sum[all_constraints_not_violated]

        constraint_violation_score_no_violations_denorm[all_constraints_not_violated] = \
            ( ( constraint_violation_score_no_violations_denorm[all_constraints_not_violated] / max(constraint_violation_score_no_violations_denorm[all_constraints_not_violated]) ) - 1 )

        constraint_violation_score_no_violations = \
            -1*abs(constraint_violation_score_no_violations_denorm) / max(abs(constraint_violation_score_no_violations_denorm))

    # combine them into a single array
    constraint_violation_score_norm_n1p1                               = zeros(num_timesteps)
    constraint_violation_score_norm_n1p1[some_constraint_violated]     = constraint_violation_score_violations_only[some_constraint_violated]
    constraint_violation_score_norm_n1p1[all_constraints_not_violated] = constraint_violation_score_no_violations[all_constraints_not_violated]

    constraint_violation_score_norm_01 = 0.5*constraint_violation_score_norm_n1p1 + 0.5

    color_map_scale                                                             = 0.7
    constraint_violation_score_norm_n01p01_scaled                               = color_map_scale*constraint_violation_score_norm_n1p1
    constraint_violation_score_norm_n01p01_scaled[some_constraint_violated]     = (1-color_map_scale) + constraint_violation_score_norm_n01p01_scaled[some_constraint_violated]
    constraint_violation_score_norm_n01p01_scaled[all_constraints_not_violated] = constraint_violation_score_norm_n01p01_scaled[all_constraints_not_violated] - (1-color_map_scale)

    constraint_violation_score_norm_01_scaled = 0.5*constraint_violation_score_norm_n01p01_scaled + 0.5
    constraint_violation_colors               = array([cm.RdBu(s) for s in 1-constraint_violation_score_norm_01_scaled])

    return constraint_violation_score_norm_n1p1, constraint_violation_score_norm_01, constraint_violation_colors



def draw(t, x, t_nominal=None, x_nominal=None, tmp_dir="tmp", request_delete_tmp_dir=True, verbose=False, savefig=False, out_dir="", out_file="", inline=False):

    import mayavi
    import mayavi.mlab
    import vtk

    if t_nominal is not None:
        assert x_nominal is not None

    if x_nominal is not None:
        assert t_nominal is not None

    if t_nominal is not None and x_nominal is not None:
        p_nominal, theta_nominal, psi_nominal, phi_nominal, p_dot_nominal, theta_dot_nominal, psi_dot_nominal, phi_dot_nominal, q_nominal, q_dot_nominal = unpack_state_space_trajectory(x_nominal)

    p, theta, psi, phi, p_dot, theta_dot, psi_dot, phi_dot, q, q_dot = unpack_state_space_trajectory(x)

    d_quad_center_to_cross_beam_end                        = 1.0
    d_quad_center_to_cross_beam_end_xsquared_plus_zsquared = d_quad_center_to_cross_beam_end**2
    d_quad_center_to_cross_beam_end_x                      = sqrt(d_quad_center_to_cross_beam_end_xsquared_plus_zsquared/2.0)
    d_quad_center_to_cross_beam_end_z                      = sqrt(d_quad_center_to_cross_beam_end_xsquared_plus_zsquared/2.0)
    d_quad_cross_beam_end_to_vert_beam_end                 = 0.4
    d_quad_axis_length                                     = 1.0

    s_quad_center        = 0.5
    s_quad_cross_beam    = 0.05
    s_quad_vert_beam     = 0.05
    s_quad_prop          = 0.25
    s_quad_axis          = 0.05
    s_world_axis         = 0.025
    s_trajectory         = 0.075

    p_body_center           = matrix([ 0.0,                                0.0,                                    0.0,                                1.0 ]).T
    p_body_cross_beam_end_0 = matrix([ d_quad_center_to_cross_beam_end_z,  0.0,                                    d_quad_center_to_cross_beam_end_x,  1.0 ]).T
    p_body_cross_beam_end_1 = matrix([ d_quad_center_to_cross_beam_end_z,  0.0,                                    -d_quad_center_to_cross_beam_end_x, 1.0 ]).T
    p_body_cross_beam_end_2 = matrix([ -d_quad_center_to_cross_beam_end_z, 0.0,                                    -d_quad_center_to_cross_beam_end_x, 1.0 ]).T
    p_body_cross_beam_end_3 = matrix([ -d_quad_center_to_cross_beam_end_z, 0.0,                                    d_quad_center_to_cross_beam_end_x,  1.0 ]).T
    p_body_vert_beam_end_0  = matrix([ d_quad_center_to_cross_beam_end_z,  d_quad_cross_beam_end_to_vert_beam_end, d_quad_center_to_cross_beam_end_x,  1.0 ]).T
    p_body_vert_beam_end_1  = matrix([ d_quad_center_to_cross_beam_end_z,  d_quad_cross_beam_end_to_vert_beam_end, -d_quad_center_to_cross_beam_end_x, 1.0 ]).T
    p_body_vert_beam_end_2  = matrix([ -d_quad_center_to_cross_beam_end_z, d_quad_cross_beam_end_to_vert_beam_end, -d_quad_center_to_cross_beam_end_x, 1.0 ]).T
    p_body_vert_beam_end_3  = matrix([ -d_quad_center_to_cross_beam_end_z, d_quad_cross_beam_end_to_vert_beam_end, d_quad_center_to_cross_beam_end_x,  1.0 ]).T
    p_body_axis_end_z       = matrix([ d_quad_axis_length,                 0.0,                                    0.0,                                1.0 ]).T
    p_body_axis_end_y       = matrix([ 0.0,                                d_quad_axis_length,                     0.0,                                1.0 ]).T
    p_body_axis_end_x       = matrix([ 0.0,                                0.0,                                    d_quad_axis_length,                 1.0 ]).T

    c_quad_center     = (1.0,1.0,1.0)
    c_quad_cross_beam = (1.0,1.0,1.0)
    c_quad_vert_beam  = (1.0,1.0,1.0)
    c_quad_prop_0     = (0.25,0.25,0.25)
    c_quad_prop_1     = (0.5,0.5,0.5)
    c_quad_prop_2     = (0.75,0.75,0.75)
    c_quad_prop_3     = (1.0,1.0,1.0)
    c_quad_axis_z     = (0.0,0.0,1.0)
    c_quad_axis_y     = (0.0,1.0,0.0)
    c_quad_axis_x     = (1.0,0.0,0.0)
    c_world_axis_z    = (0.0,0.0,1.0)
    c_world_axis_y    = (0.0,1.0,0.0)
    c_world_axis_x    = (1.0,0.0,0.0)

    img_width,img_height = 800,600

    mayavi.mlab.figure(size=(img_width,img_height))

    # invisible trajectory to set the appropriate camera scaling
    mayavi.mlab.plot3d(p[:,2], p[:,1], p[:,0], t, tube_radius=s_trajectory, opacity=0.0)

    # center
    points3d_quad_center = mayavi.mlab.points3d(0.0, 0.0, 0.0, scale_factor=s_quad_center, color=c_quad_center)

    # cross beams
    plot3d_quad_cross_beam_0 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_cross_beam, color=c_quad_cross_beam)
    plot3d_quad_cross_beam_1 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_cross_beam, color=c_quad_cross_beam)

    # vertical beams
    plot3d_quad_vert_beam_0 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_vert_beam, color=c_quad_vert_beam)
    plot3d_quad_vert_beam_1 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_vert_beam, color=c_quad_vert_beam)
    plot3d_quad_vert_beam_2 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_vert_beam, color=c_quad_vert_beam)
    plot3d_quad_vert_beam_3 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_vert_beam, color=c_quad_vert_beam)

    # props
    points3d_quad_prop_0 = mayavi.mlab.points3d(0.0, 0.0, 0.0, scale_factor=s_quad_prop, color=c_quad_prop_0)
    points3d_quad_prop_1 = mayavi.mlab.points3d(0.0, 0.0, 0.0, scale_factor=s_quad_prop, color=c_quad_prop_1)
    points3d_quad_prop_2 = mayavi.mlab.points3d(0.0, 0.0, 0.0, scale_factor=s_quad_prop, color=c_quad_prop_2)
    points3d_quad_prop_3 = mayavi.mlab.points3d(0.0, 0.0, 0.0, scale_factor=s_quad_prop, color=c_quad_prop_3)

    # local reference frame
    plot3d_quad_axis_z = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_axis, color=c_quad_axis_z)
    plot3d_quad_axis_y = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_axis, color=c_quad_axis_y)
    plot3d_quad_axis_x = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_axis, color=c_quad_axis_x)

    # global reference frame
    mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,5.0], tube_radius=s_world_axis, color=c_world_axis_z)
    mayavi.mlab.plot3d([0.0,0.0], [0.0,5.0], [0.0,0.0], tube_radius=s_world_axis, color=c_world_axis_y)
    mayavi.mlab.plot3d([0.0,5.0], [0.0,0.0], [0.0,0.0], tube_radius=s_world_axis, color=c_world_axis_x)

    if t_nominal is not None and x_nominal is not None:

        pts = mayavi.mlab.quiver3d(p_nominal[:,2], p_nominal[:,1], p_nominal[:,0], ones(p_nominal.shape[0]), ones(p_nominal.shape[0]), ones(p_nominal.shape[0]), scalars=t_nominal, mode="sphere", scale_factor=s_trajectory)
        pts.glyph.color_mode = "color_by_scalar"
        pts.glyph.glyph_source.glyph_source.center = [0,0,0]

    mayavi.mlab.view(0,0)

    def draw_quad(x_ti):

        p_ti, theta_ti, psi_ti, phi_ti, p_dot_ti, theta_dot_ti, psi_dot_ti, phi_dot_ti, q_ti, q_dot_ti = unpack_state(x_ti)

        R = matrix(transformations.euler_matrix(psi_ti,theta_ti,phi_ti,"ryxz"))
        T = matrix(transformations.translation_matrix(p_ti.A1))
        M = T * R

        p_world_center           = M * p_body_center
        p_world_cross_beam_end_0 = M * p_body_cross_beam_end_0
        p_world_cross_beam_end_1 = M * p_body_cross_beam_end_1
        p_world_cross_beam_end_2 = M * p_body_cross_beam_end_2
        p_world_cross_beam_end_3 = M * p_body_cross_beam_end_3
        p_world_vert_beam_end_0  = M * p_body_vert_beam_end_0
        p_world_vert_beam_end_1  = M * p_body_vert_beam_end_1
        p_world_vert_beam_end_2  = M * p_body_vert_beam_end_2
        p_world_vert_beam_end_3  = M * p_body_vert_beam_end_3
        p_world_axis_end_z       = M * p_body_axis_end_z
        p_world_axis_end_y       = M * p_body_axis_end_y
        p_world_axis_end_x       = M * p_body_axis_end_x

        p_world_center           = ( p_world_center           / p_world_center[3]           ).A1
        p_world_cross_beam_end_0 = ( p_world_cross_beam_end_0 / p_world_cross_beam_end_0[3] ).A1
        p_world_cross_beam_end_1 = ( p_world_cross_beam_end_1 / p_world_cross_beam_end_1[3] ).A1
        p_world_cross_beam_end_2 = ( p_world_cross_beam_end_2 / p_world_cross_beam_end_2[3] ).A1
        p_world_cross_beam_end_3 = ( p_world_cross_beam_end_3 / p_world_cross_beam_end_3[3] ).A1
        p_world_vert_beam_end_0  = ( p_world_vert_beam_end_0  / p_world_vert_beam_end_0[3]  ).A1
        p_world_vert_beam_end_1  = ( p_world_vert_beam_end_1  / p_world_vert_beam_end_1[3]  ).A1
        p_world_vert_beam_end_2  = ( p_world_vert_beam_end_2  / p_world_vert_beam_end_2[3]  ).A1
        p_world_vert_beam_end_3  = ( p_world_vert_beam_end_3  / p_world_vert_beam_end_3[3]  ).A1
        p_world_axis_end_z       = ( p_world_axis_end_z       / p_world_axis_end_z[3]       ).A1
        p_world_axis_end_y       = ( p_world_axis_end_y       / p_world_axis_end_y[3]       ).A1
        p_world_axis_end_x       = ( p_world_axis_end_x       / p_world_axis_end_x[3]       ).A1

        # center
        points3d_quad_center.mlab_source.set( x=p_world_center[2], y=p_world_center[1], z=p_world_center[0] )

        # cross beams
        plot3d_quad_cross_beam_0.mlab_source.set( x=[p_world_cross_beam_end_0[2],p_world_cross_beam_end_2[2]], y=[p_world_cross_beam_end_0[1],p_world_cross_beam_end_2[1]], z=[p_world_cross_beam_end_0[0],p_world_cross_beam_end_2[0]] )
        plot3d_quad_cross_beam_1.mlab_source.set( x=[p_world_cross_beam_end_1[2],p_world_cross_beam_end_3[2]], y=[p_world_cross_beam_end_1[1],p_world_cross_beam_end_3[1]], z=[p_world_cross_beam_end_1[0],p_world_cross_beam_end_3[0]] )

        # vertical beams
        plot3d_quad_vert_beam_0.mlab_source.set( x=[p_world_cross_beam_end_0[2],p_world_vert_beam_end_0[2]], y=[p_world_cross_beam_end_0[1],p_world_vert_beam_end_0[1]], z=[p_world_cross_beam_end_0[0],p_world_vert_beam_end_0[0]] )
        plot3d_quad_vert_beam_1.mlab_source.set( x=[p_world_cross_beam_end_1[2],p_world_vert_beam_end_1[2]], y=[p_world_cross_beam_end_1[1],p_world_vert_beam_end_1[1]], z=[p_world_cross_beam_end_1[0],p_world_vert_beam_end_1[0]] )
        plot3d_quad_vert_beam_2.mlab_source.set( x=[p_world_cross_beam_end_2[2],p_world_vert_beam_end_2[2]], y=[p_world_cross_beam_end_2[1],p_world_vert_beam_end_2[1]], z=[p_world_cross_beam_end_2[0],p_world_vert_beam_end_2[0]] )
        plot3d_quad_vert_beam_3.mlab_source.set( x=[p_world_cross_beam_end_3[2],p_world_vert_beam_end_3[2]], y=[p_world_cross_beam_end_3[1],p_world_vert_beam_end_3[1]], z=[p_world_cross_beam_end_3[0],p_world_vert_beam_end_3[0]] )

        # props
        points3d_quad_prop_0.mlab_source.set( x=p_world_vert_beam_end_0[2], y=p_world_vert_beam_end_0[1], z=p_world_vert_beam_end_0[0] )
        points3d_quad_prop_1.mlab_source.set( x=p_world_vert_beam_end_1[2], y=p_world_vert_beam_end_1[1], z=p_world_vert_beam_end_1[0] )
        points3d_quad_prop_2.mlab_source.set( x=p_world_vert_beam_end_2[2], y=p_world_vert_beam_end_2[1], z=p_world_vert_beam_end_2[0] )
        points3d_quad_prop_3.mlab_source.set( x=p_world_vert_beam_end_3[2], y=p_world_vert_beam_end_3[1], z=p_world_vert_beam_end_3[0] )

        # local reference frame
        plot3d_quad_axis_z.mlab_source.set( x=[p_world_center[2],p_world_axis_end_z[2]], y=[p_world_center[1],p_world_axis_end_z[1]], z=[p_world_center[0],p_world_axis_end_z[0]] )
        plot3d_quad_axis_y.mlab_source.set( x=[p_world_center[2],p_world_axis_end_y[2]], y=[p_world_center[1],p_world_axis_end_y[1]], z=[p_world_center[0],p_world_axis_end_y[0]] )
        plot3d_quad_axis_x.mlab_source.set( x=[p_world_center[2],p_world_axis_end_x[2]], y=[p_world_center[1],p_world_axis_end_x[1]], z=[p_world_center[0],p_world_axis_end_x[0]] )

    @mayavi.mlab.animate
    def mlab_animate():

        ti = 0
        while 1:
            x_ti = matrix(x[ti]).T
            draw_quad(x_ti)
            ti = (ti+1)%x.shape[0]
            yield

    def render_images():

        for ti in range(x.shape[0]):
            x_ti = matrix(x[ti]).T
            draw_quad(x_ti)
            mayavi.mlab.savefig(tmp_dir+"/"+"%04d.png" % ti)
        mayavi.mlab.close()

    def render_video():

        import cv2

        img_0        = cv2.imread(tmp_dir+"/"+"0000.png")
        video_writer = cv2.VideoWriter(tmp_dir+"/"+"tmp.mp4", fromstring("avc1", dtype=int8).view(int32), fps=25.0, frameSize=(img_0.shape[1],img_0.shape[0]))
        for ti in range(x.shape[0]):
            img = cv2.imread(tmp_dir+"/"+"%04d.png" % ti)
            video_writer.write(img)
        video_writer.release()

    tmp_dir_exists_before_call = os.path.exists(tmp_dir)

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    vtk_debug_output = vtk.vtkFileOutputWindow()
    vtk_debug_output.SetFileName(tmp_dir+"/"+"vtk_debug_output.txt")
    vtk.vtkOutputWindow().SetInstance(vtk_debug_output)

    if savefig or inline:
        render_images()
        render_video()
    else:
        mlab_animate()
        mayavi.mlab.show()

    if savefig:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cmd = "cp %s/tmp.mp4 %s/%s" % (tmp_dir,out_dir,out_file)
        if verbose: print cmd
        output = subprocess.check_output(cmd, shell=True)
        if verbose and len(output) > 0: print output

    if inline:
        inline_video = ipython_display_utils.get_inline_video("%s/tmp.mp4" % tmp_dir)

    if not tmp_dir_exists_before_call and request_delete_tmp_dir:
        cmd = "rm -rf %s" % tmp_dir
        if verbose: print cmd
        output = subprocess.check_output(cmd, shell=True)
        if verbose and len(output) > 0: print output

    if inline:
        return inline_video
    else:
        return None
