from pylab import *

import sys
import time
import sklearn
import sklearn.preprocessing
import transformations
import linalgutils
import trigutils
import sympy
import sympy.matrices
import sympy.physics
import sympy.physics.mechanics
import sympy.physics.mechanics.functions

import gradient
import path
import trig
import sympy



front_prop_and_quad_positive_x_axis_angle = pi/4
rear_prop_and_quad_negative_x_axis_angle  = pi/4
y_axis_torque_per_newton_per_prop         = 1

alpha = front_prop_and_quad_positive_x_axis_angle
beta  = rear_prop_and_quad_negative_x_axis_angle
gamma = y_axis_torque_per_newton_per_prop

d      = 1.0                          # distance from arm to center
m      = 1.0                          # mass
g      = 9.8                          # gravity
I_body = m*d**2.0*matrix(identity(3)) # moment of intertia for body

f_gravity_world  = matrix([0,-m*g,0]).T
f_external_world = f_gravity_world

num_q_dims  = 9
num_x_dims  = 2*num_q_dims
num_u_dims  = 7
num_dt_dims = 1

num_dims_g_dynamics_ti = num_x_dims

build_sympy_modules = False
# build_sympy_modules = False



print "flashlight.quadrotorcamera3d: Constructing sympy symbols..."
sys_time_begin = time.time()

# constants
alpha_expr = sympy.Symbol("alpha")
beta_expr  = sympy.Symbol("beta")
gamma_expr = sympy.Symbol("gamma")
d_expr     = sympy.Symbol("d")
m_expr     = sympy.Symbol("m")

I_body_expr,     I_body_expr_entries     = sympyutils.construct_matrix_and_entries("I_body",(3,3))
f_external_expr, f_external_expr_entries = sympyutils.construct_matrix_and_entries("f_e",(3,1))

#variables
t_expr  = sympy.Symbol("t")

p_z_expr        = sympy.physics.mechanics.dynamicsymbols("p_z")
p_y_expr        = sympy.physics.mechanics.dynamicsymbols("p_y")
p_x_expr        = sympy.physics.mechanics.dynamicsymbols("p_x")
theta_body_expr = sympy.physics.mechanics.dynamicsymbols("theta_body")
psi_body_expr   = sympy.physics.mechanics.dynamicsymbols("psi_body")
phi_body_expr   = sympy.physics.mechanics.dynamicsymbols("phi_body")
theta_cam_expr  = sympy.physics.mechanics.dynamicsymbols("theta_cam")
psi_cam_expr    = sympy.physics.mechanics.dynamicsymbols("psi_cam")
phi_cam_expr    = sympy.physics.mechanics.dynamicsymbols("phi_cam")

# state and control vectors
q_expr                 = sympy.Matrix([p_z_expr,p_y_expr,p_x_expr,theta_body_expr,psi_body_expr,phi_body_expr,theta_cam_expr,psi_cam_expr,phi_cam_expr])
q_dot_expr             = q_expr.diff(t_expr)
x_expr                 = sympyutils.construct_matrix_from_block_matrix( sympy.Matrix( [ [q_expr], [q_dot_expr] ] ) )
u_expr, u_expr_entries = sympyutils.construct_matrix_and_entries("u",(num_u_dims,1))

# symbols to solve for g_dynamics, given x_current x_next u_current dt_current
x_current_expr, x_current_expr_entries = sympyutils.construct_matrix_and_entries("x_current", (num_x_dims,1))
x_next_expr,    x_next_expr_entries    = sympyutils.construct_matrix_and_entries("x_next",    (num_x_dims,1))
u_current_expr, u_current_expr_entries = sympyutils.construct_matrix_and_entries("u_current", (num_u_dims,1))
dt_current_expr                        = sympy.Symbol("delta_t_current")

# symbol collections
const_vals                                                   = hstack( [ alpha, beta, gamma, d, m, I_body.A1, f_external_world.A1 ] )
const_syms                                                   = hstack( [ alpha_expr, beta_expr, gamma_expr, d_expr, m_expr, matrix(I_body_expr).A1, matrix(f_external_expr).A1 ] )
const_and_x_syms                                             = hstack( [ const_syms, matrix(x_expr).A1 ] )
const_and_x_and_u_syms                                       = hstack( [ const_syms, matrix(x_expr).A1, matrix(u_expr).A1 ] )
const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms = hstack( [ const_syms, matrix(x_current_expr).A1, matrix(x_next_expr).A1, matrix(u_current_expr).A1, dt_current_expr ] )

sys_time_end = time.time()
print "flashlight.quadrotorcamera3d: Finished constructing sympy symbols (%.03f seconds)." % (sys_time_end - sys_time_begin)

if build_sympy_modules:

    print "flashlight.quadrotorcamera3d: Constructing sympy expressions..."

    theta_expr     = sympy.physics.mechanics.dynamicsymbols("theta")
    psi_expr       = sympy.physics.mechanics.dynamicsymbols("psi")
    phi_expr       = sympy.physics.mechanics.dynamicsymbols("phi")
    theta_dot_expr = theta_expr.diff(t_expr)
    psi_dot_expr   = psi_expr.diff(t_expr)
    phi_dot_expr   = phi_expr.diff(t_expr)

    R_z_theta_expr = sympyutils.construct_axis_aligned_rotation_matrix_right_handed(theta_expr,0)
    R_y_psi_expr   = sympyutils.construct_axis_aligned_rotation_matrix_right_handed(psi_expr,1)
    R_x_phi_expr   = sympyutils.construct_axis_aligned_rotation_matrix_right_handed(phi_expr,2)

    R_expr         = sympy.trigsimp(R_y_psi_expr*R_z_theta_expr*R_x_phi_expr)
    R_dot_expr     = sympy.trigsimp(R_expr.diff(t_expr))
    R_dot_R_T_expr = sympy.trigsimp(R_dot_expr*R_expr.T)

    omega_z_terms_expr = sympyutils.collect_into_dict_include_zero_and_constant_terms( R_dot_R_T_expr[2,1],  [theta_dot_expr,psi_dot_expr,phi_dot_expr] )
    omega_y_terms_expr = sympyutils.collect_into_dict_include_zero_and_constant_terms( -R_dot_R_T_expr[2,0], [theta_dot_expr,psi_dot_expr,phi_dot_expr] )
    omega_x_terms_expr = sympyutils.collect_into_dict_include_zero_and_constant_terms( R_dot_R_T_expr[1,0],  [theta_dot_expr,psi_dot_expr,phi_dot_expr] )

    A_expr = sympy.Matrix( [ \
        [ omega_z_terms_expr[theta_dot_expr], omega_z_terms_expr[psi_dot_expr], omega_z_terms_expr[phi_dot_expr] ], \
        [ omega_y_terms_expr[theta_dot_expr], omega_y_terms_expr[psi_dot_expr], omega_y_terms_expr[phi_dot_expr] ], \
        [ omega_x_terms_expr[theta_dot_expr], omega_x_terms_expr[psi_dot_expr], omega_x_terms_expr[phi_dot_expr] ] ] )

    A_dot_expr = sympy.trigsimp(A_expr.diff(t_expr))

    theta_body_dot_expr = theta_body_expr.diff(t_expr)
    psi_body_dot_expr   = psi_body_expr.diff(t_expr)
    phi_body_dot_expr   = phi_body_expr.diff(t_expr)
    theta_cam_dot_expr  = theta_cam_expr.diff(t_expr)
    psi_cam_dot_expr    = psi_cam_expr.diff(t_expr)
    phi_cam_dot_expr    = phi_cam_expr.diff(t_expr)

    e_expr          = sympy.Matrix([ theta_expr,      psi_expr,      phi_expr ])
    e_body_expr     = sympy.Matrix([ theta_body_expr, psi_body_expr, phi_body_expr ])
    e_cam_expr      = sympy.Matrix([ theta_cam_expr,  psi_cam_expr,  phi_cam_expr ])
    e_dot_expr      = e_expr.diff(t_expr)
    e_body_dot_expr = e_body_expr.diff(t_expr)
    e_cam_dot_expr  = e_cam_expr.diff(t_expr)

    A_body_expr     = A_expr.subs(     dict( zip(e_expr,e_body_expr) + zip(e_dot_expr,e_body_dot_expr) ) )
    A_body_dot_expr = A_dot_expr.subs( dict( zip(e_expr,e_body_expr) + zip(e_dot_expr,e_body_dot_expr) ) )
    A_cam_expr      = A_expr.subs(     dict( zip(e_expr,e_cam_expr)  + zip(e_dot_expr,e_cam_dot_expr) ) )
    A_cam_dot_expr  = A_dot_expr.subs( dict( zip(e_expr,e_cam_expr)  + zip(e_dot_expr,e_cam_dot_expr) ) )

    R_world_from_body_expr = R_expr.subs( dict( zip(e_expr,e_body_expr) ) )
    R_body_from_world_expr = R_world_from_body_expr.T

    omega_body_in_body_expr     = sympy.trigsimp(R_body_from_world_expr*A_body_expr*e_body_dot_expr)
    I_omega_body_in_body_X_expr = sympyutils.construct_cross_product_left_term_matrix_from_vector(I_body_expr*omega_body_in_body_expr)

    M_thrust_body_from_control_expr = sympy.Matrix([[0,0,0,0],[1,1,1,1],[0,0,0,0]])

    M_torque_body_from_control_expr = sympy.Matrix(([ \
        [-d_expr*sympy.cos(alpha_expr),d_expr*sympy.cos(beta_expr),d_expr*sympy.cos(beta_expr),-d_expr*sympy.cos(alpha_expr)], \
        [gamma_expr,-gamma_expr,gamma_expr,-gamma_expr], \
        [d_expr*sympy.sin(alpha_expr),d_expr*sympy.sin(beta_expr),-d_expr*sympy.sin(beta_expr),-d_expr*sympy.sin(alpha_expr)]]))

    H_00_expr    = sympy.Matrix(m_expr*sympy.eye(3))
    H_11_expr    = sympy.trigsimp(I_body_expr*R_body_from_world_expr*A_body_expr)
    H_22_expr    = A_cam_expr
    H_zeros_expr = sympy.Matrix.zeros(3,3)

    C_11_expr    = I_body_expr*R_body_from_world_expr*A_body_dot_expr - I_omega_body_in_body_X_expr*R_body_from_world_expr*A_body_expr
    C_22_expr    = A_cam_dot_expr
    C_zeros_expr = sympy.Matrix.zeros(3,3)

    G_0_expr = -f_external_expr
    G_1_expr = sympy.Matrix.zeros(3,1)
    G_2_expr = sympy.Matrix.zeros(3,1)

    B_00_expr = R_world_from_body_expr*M_thrust_body_from_control_expr
    B_01_expr = sympy.Matrix.zeros(3,3)
    B_10_expr = M_torque_body_from_control_expr
    B_11_expr = sympy.Matrix.zeros(3,3)
    B_20_expr = sympy.Matrix.zeros(3,4)
    B_21_expr = sympy.eye(3)

    H_expr = sympyutils.construct_matrix_from_block_matrix( sympy.Matrix( [ [H_00_expr,    H_zeros_expr, H_zeros_expr], [H_zeros_expr, H_11_expr, H_zeros_expr], [H_zeros_expr, H_zeros_expr, H_22_expr] ] ) )
    C_expr = sympyutils.construct_matrix_from_block_matrix( sympy.Matrix( [ [C_zeros_expr, C_zeros_expr, C_zeros_expr], [C_zeros_expr, C_11_expr, C_zeros_expr], [C_zeros_expr, C_zeros_expr, C_22_expr] ] ) )
    G_expr = sympyutils.construct_matrix_from_block_matrix( sympy.Matrix( [ [G_0_expr],                                 [G_1_expr],                              [G_2_expr] ] ) )
    B_expr = sympyutils.construct_matrix_from_block_matrix( sympy.Matrix( [ [B_00_expr, B_01_expr],                     [B_10_expr, B_11_expr],                  [B_20_expr, B_21_expr] ] ) )

    # expressions to solve for df_dx and df_du
    q_dot_dot_expr = H_expr.inv()*(B_expr*u_expr - (C_expr*q_dot_expr + G_expr))
    x_dot_expr     = sympyutils.construct_matrix_from_block_matrix( sympy.Matrix( [ [q_dot_expr], [q_dot_dot_expr] ] ) )
    df_dx_expr     = x_dot_expr.jacobian(x_expr)
    df_du_expr     = x_dot_expr.jacobian(u_expr)

    x_dot_current_expr           = x_dot_expr.subs( dict( zip(x_expr,x_current_expr) + zip(u_expr,u_current_expr) ) )
    g_dynamics_ti_expr           = x_next_expr - (x_current_expr + x_dot_current_expr*dt_current_expr)
    dgdynamicsti_dxcurrent_expr  = g_dynamics_ti_expr.jacobian(x_current_expr)
    dgdynamicsti_dxnext_expr     = g_dynamics_ti_expr.jacobian(x_next_expr)
    dgdynamicsti_ducurrent_expr  = g_dynamics_ti_expr.jacobian(u_current_expr)
    dgdynamicsti_ddtcurrent_expr = g_dynamics_ti_expr.diff(dt_current_expr)

    sys_time_end = time.time()
    print "flashlight.quadrotorcamera3d: Finished constructing sympy expressions (%.03f seconds)." % (sys_time_end - sys_time_begin)

    print "flashlight.quadrotorcamera3d: Building sympy modules..."
    sys_time_begin = time.time()

    current_source_file_path = pathutils.get_current_source_file_path()

    sympyutils.build_module_autowrap( expr=H_expr, syms=const_and_x_syms, module_name="quadrotorcamera3d_H", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=C_expr, syms=const_and_x_syms, module_name="quadrotorcamera3d_C", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=G_expr, syms=const_and_x_syms, module_name="quadrotorcamera3d_G", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=B_expr, syms=const_and_x_syms, module_name="quadrotorcamera3d_B", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )

    sympyutils.build_module_autowrap( expr=x_dot_expr, syms=const_and_x_and_u_syms, module_name="quadrotorcamera3d_x_dot", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=df_dx_expr, syms=const_and_x_and_u_syms, module_name="quadrotorcamera3d_df_dx", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=df_du_expr, syms=const_and_x_and_u_syms, module_name="quadrotorcamera3d_df_du", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )

    sympyutils.build_module_autowrap( expr=g_dynamics_ti_expr,           syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotorcamera3d_g_dynamics_ti",           tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dgdynamicsti_dxcurrent_expr,  syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotorcamera3d_dgdynamicsti_dxcurrent",  tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dgdynamicsti_dxnext_expr,     syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotorcamera3d_dgdynamicsti_dxnext",     tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dgdynamicsti_ducurrent_expr,  syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotorcamera3d_dgdynamicsti_ducurrent",  tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympyutils.build_module_autowrap( expr=dgdynamicsti_ddtcurrent_expr, syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotorcamera3d_dgdynamicsti_ddtcurrent", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotorcamera3d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )

    sys_time_end = time.time()
    print "flashlight.quadrotorcamera3d: Finished building sympy modules (%.03f seconds)." % (sys_time_end - sys_time_begin)

print "flashlight.quadrotorcamera3d: Loading sympy modules..."
sys_time_begin = time.time()

current_source_file_path = pathutils.get_current_source_file_path()

H_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_H", path=current_source_file_path+"/data/quadrotorcamera3d" )
C_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_C", path=current_source_file_path+"/data/quadrotorcamera3d" )
G_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_G", path=current_source_file_path+"/data/quadrotorcamera3d" )
B_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_B", path=current_source_file_path+"/data/quadrotorcamera3d" )

x_dot_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_x_dot", path=current_source_file_path+"/data/quadrotorcamera3d" )
df_dx_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_df_dx", path=current_source_file_path+"/data/quadrotorcamera3d" )
df_du_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_df_du", path=current_source_file_path+"/data/quadrotorcamera3d" )

g_dynamics_ti_autowrap           = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_g_dynamics_ti",           path=current_source_file_path+"/data/quadrotorcamera3d" )
dgdynamicsti_dxcurrent_autowrap  = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_dgdynamicsti_dxcurrent",  path=current_source_file_path+"/data/quadrotorcamera3d" )
dgdynamicsti_dxnext_autowrap     = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_dgdynamicsti_dxnext",     path=current_source_file_path+"/data/quadrotorcamera3d" )
dgdynamicsti_ducurrent_autowrap  = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_dgdynamicsti_ducurrent",  path=current_source_file_path+"/data/quadrotorcamera3d" )
dgdynamicsti_ddtcurrent_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_dgdynamicsti_ddtcurrent", path=current_source_file_path+"/data/quadrotorcamera3d" )

H_vectorized_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_H_vectorized", path=current_source_file_path+"/data/quadrotorcamera3d" )
C_vectorized_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_C_vectorized", path=current_source_file_path+"/data/quadrotorcamera3d" )
G_vectorized_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_G_vectorized", path=current_source_file_path+"/data/quadrotorcamera3d" )
B_vectorized_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_B_vectorized", path=current_source_file_path+"/data/quadrotorcamera3d" )

x_dot_vectorized_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_x_dot_vectorized", path=current_source_file_path+"/data/quadrotorcamera3d" )
df_dx_vectorized_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_df_dx_vectorized", path=current_source_file_path+"/data/quadrotorcamera3d" )
df_du_vectorized_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_df_du_vectorized", path=current_source_file_path+"/data/quadrotorcamera3d" )

g_dynamics_ti_vectorized_autowrap           = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_g_dynamics_ti_vectorized",           path=current_source_file_path+"/data/quadrotorcamera3d" )
dgdynamicsti_dxcurrent_vectorized_autowrap  = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_dgdynamicsti_dxcurrent_vectorized",  path=current_source_file_path+"/data/quadrotorcamera3d" )
dgdynamicsti_dxnext_vectorized_autowrap     = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_dgdynamicsti_dxnext_vectorized",     path=current_source_file_path+"/data/quadrotorcamera3d" )
dgdynamicsti_ducurrent_vectorized_autowrap  = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_dgdynamicsti_ducurrent_vectorized",  path=current_source_file_path+"/data/quadrotorcamera3d" )
dgdynamicsti_ddtcurrent_vectorized_autowrap = sympyutils.import_anon_func_from_from_module_autowrap( module_name="quadrotorcamera3d_dgdynamicsti_ddtcurrent_vectorized", path=current_source_file_path+"/data/quadrotorcamera3d" )

sys_time_end = time.time()
print "flashlight.quadrotorcamera3d: Finished loading sympy modules (%.03f seconds)." % (sys_time_end - sys_time_begin)



def unpack_state(x):

    p_body         = matrix( [ x.item(0), x.item(1), x.item(2) ] ).T
    theta_body     = x.item(3)
    psi_body       = x.item(4)
    phi_body       = x.item(5)
    theta_cam      = x.item(6)
    psi_cam        = x.item(7)
    phi_cam        = x.item(8)
    p_body_dot     = matrix( [ x.item(9), x.item(10), x.item(11) ] ).T
    theta_body_dot = x.item(12)
    psi_body_dot   = x.item(13)
    phi_body_dot   = x.item(14)
    theta_cam_dot  = x.item(15)
    psi_cam_dot    = x.item(16)
    phi_cam_dot    = x.item(17)
    q              = matrix( [ p_body.item(0),     p_body.item(1),     p_body.item(2),     theta_body,     psi_body,     phi_body,     theta_cam,     psi_cam,     phi_cam,    ] ).T
    q_dot          = matrix( [ p_body_dot.item(0), p_body_dot.item(1), p_body_dot.item(2), theta_body_dot, psi_body_dot, phi_body_dot, theta_cam_dot, psi_cam_dot, phi_cam_dot ] ).T

    return p_body, theta_body, psi_body, phi_body, theta_cam, psi_cam, phi_cam, p_body_dot, theta_body_dot, psi_body_dot, phi_body_dot, theta_cam_dot, psi_cam_dot, phi_cam_dot, q, q_dot



def compute_manipulator_matrices(x):

    const_and_x_vals = hstack( [ const_vals, matrix(x).A1 ] )
    H                = matrix( sympyutils.evaluate_anon_func( H_autowrap, const_and_x_vals ) )
    C                = matrix( sympyutils.evaluate_anon_func( C_autowrap, const_and_x_vals ) )
    G                = matrix( sympyutils.evaluate_anon_func( G_autowrap, const_and_x_vals ) )
    B                = matrix( sympyutils.evaluate_anon_func( B_autowrap, const_and_x_vals ) )

    return H, C, G, B



def compute_x_dot(x,u):

    const_and_x_and_u_vals = hstack( [ const_vals, matrix(x).A1, matrix(u).A1 ] )
    x_dot                  = matrix( sympyutils.evaluate_anon_func( x_dot_autowrap, const_and_x_and_u_vals ) )

    return x_dot



def compute_df_dx_and_df_du(x,u):

    const_and_x_and_u_vals = hstack( [ const_vals, matrix(x).A1, matrix(u).A1 ] )
    df_dx                  = matrix( sympyutils.evaluate_anon_func( df_dx_autowrap, const_and_x_and_u_vals ) )
    df_du                  = matrix( sympyutils.evaluate_anon_func( df_du_autowrap, const_and_x_and_u_vals ) )

    return df_dx, df_du



def compute_dynamics_constraints_direct_transcription(X,U,dt):

    g_dynamics = zeros((X.shape[0]-1,X.shape[1]))

    for ti in range(X.shape[0]-1):

        x_current = matrix(X[ti]).T
        u_current = matrix(U[ti]).T
        x_next    = matrix(X[ti+1]).T

        if isinstance(dt, ndarray) or isinstance(dt, matrix):
            dt_current = dt[ti]
        else:
            dt_current = dt

        const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals = hstack( [ const_vals, x_current.A1, x_next.A1, u_current.A1, dt_current ] )

        g_dynamics[ti] = matrix( sympyutils.evaluate_anon_func( g_dynamics_ti_autowrap, const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals ) ).A1

    return g_dynamics



def compute_dynamics_constraints_jacobian_direct_transcription(X,U,dt):

    num_trajectory_samples      = X.shape[0]
    num_dynamics_constraints_1d = (X.shape[0]-1) * X.shape[1]
    num_decision_vars_X         = X.shape[0] * X.shape[1]
    num_decision_vars_U         = U.shape[0] * U.shape[1]

    if isinstance(dt, ndarray) or isinstance(dt, matrix):
        num_decision_vars_dt = dt.shape[0] * dt.shape[1]
    else:
        num_decision_vars_dt = 1

    dgdynamics_dX = zeros((num_dynamics_constraints_1d,num_decision_vars_X))
    dgdynamics_dU = zeros((num_dynamics_constraints_1d,num_decision_vars_U))
    dgdynamics_ddt = zeros((num_dynamics_constraints_1d,num_decision_vars_dt))

    for ti in range(num_trajectory_samples-1):

        x_current = matrix(X[ti]).T
        u_current = matrix(U[ti]).T
        x_next    = matrix(X[ti+1]).T

        if isinstance(dt, ndarray) or isinstance(dt, matrix):
            dt_current = dt[ti]
        else:
            dt_current = dt

        const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals = hstack( [ const_vals, x_current.A1, x_next.A1, u_current.A1, dt_current ] )

        gi_begin           = (ti+0)*num_dims_g_dynamics_ti
        gi_end             = (ti+1)*num_dims_g_dynamics_ti
        ai_x_current_begin = (ti+0)*num_x_dims
        ai_x_current_end   = (ti+1)*num_x_dims
        ai_x_next_begin    = (ti+1)*num_x_dims
        ai_x_next_end      = (ti+2)*num_x_dims
        ai_u_current_begin = (ti+0)*num_u_dims
        ai_u_current_end   = (ti+1)*num_u_dims

        if isinstance(dt, ndarray) or isinstance(dt, matrix):
            ai_dt_current_begin = (ti+0)*num_dt_dims
            ai_dt_current_end   = (ti+1)*num_dt_dims
        else:
            ai_dt_current_begin = 0*num_dt_dims
            ai_dt_current_end   = 1*num_dt_dims

        dgdynamicsti_dxcurrent  = sympyutils.evaluate_anon_func( dgdynamicsti_dxcurrent_autowrap,  const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals )
        dgdynamicsti_dxnext     = sympyutils.evaluate_anon_func( dgdynamicsti_dxnext_autowrap,     const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals )
        dgdynamicsti_ducurrent  = sympyutils.evaluate_anon_func( dgdynamicsti_ducurrent_autowrap,  const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals )
        dgdynamicsti_ddtcurrent = sympyutils.evaluate_anon_func( dgdynamicsti_ddtcurrent_autowrap, const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_vals )

        dgdynamics_dX[gi_begin:gi_end,ai_x_current_begin:ai_x_current_end]    = dgdynamicsti_dxcurrent
        dgdynamics_dX[gi_begin:gi_end,ai_x_next_begin:ai_x_next_end]          = dgdynamicsti_dxnext
        dgdynamics_dU[gi_begin:gi_end,ai_u_current_begin:ai_u_current_end]    = dgdynamicsti_ducurrent
        dgdynamics_ddt[gi_begin:gi_end,ai_dt_current_begin:ai_dt_current_end] = dgdynamicsti_ddtcurrent

    return dgdynamics_dX, dgdynamics_dU, dgdynamics_ddt



def compute_state_space_trajectory_and_derivatives(p_body,p_look_at,y_axis_cam_hint,dt):

    num_timesteps = len(p_body)

    #
    # compute the yaw, roll, and pitch of the quad using differential flatness
    #
    p_body_dotN    = gradientutils.gradients_vector_wrt_scalar_smooth_boundaries(p_body,dt,max_gradient=2,poly_deg=5)
    p_body_dot     = p_body_dotN[1]
    p_body_dot_dot = p_body_dotN[2]

    f_thrust_world            = m*p_body_dot_dot - f_external_world.T.A
    f_thrust_world_normalized = sklearn.preprocessing.normalize(f_thrust_world)

    y_axis_body = f_thrust_world_normalized
    z_axis_body = sklearn.preprocessing.normalize(cross(y_axis_body, p_look_at - p_body))
    x_axis_body = sklearn.preprocessing.normalize(cross(z_axis_body, y_axis_body))

    R_world_from_body = zeros((num_timesteps,4,4))
    theta_body        = zeros(num_timesteps)
    psi_body          = zeros(num_timesteps)
    phi_body          = zeros(num_timesteps)

    for ti in range(num_timesteps):

        z_axis_body_ti = c_[matrix(z_axis_body[ti]),0].T
        y_axis_body_ti = c_[matrix(y_axis_body[ti]),0].T
        x_axis_body_ti = c_[matrix(x_axis_body[ti]),0].T

        R_world_from_body_ti                  = c_[z_axis_body_ti,y_axis_body_ti,x_axis_body_ti,[0,0,0,1]]
        psi_body_ti,theta_body_ti,phi_body_ti = transformations.euler_from_matrix(R_world_from_body_ti,"ryxz")

        assert allclose(R_world_from_body_ti, transformations.euler_matrix(psi_body_ti,theta_body_ti,phi_body_ti,"ryxz"))

        R_world_from_body[ti] = R_world_from_body_ti
        theta_body[ti]        = theta_body_ti
        psi_body[ti]          = psi_body_ti
        phi_body[ti]          = phi_body_ti

    psi_body   = trigutils.compute_continuous_angle_array(psi_body)
    theta_body = trigutils.compute_continuous_angle_array(theta_body)
    phi_body   = trigutils.compute_continuous_angle_array(phi_body)

    #
    # now that we have the full orientation of the quad, compute the full orientation of the camera relative to the quad
    #
    x_axis_cam = sklearn.preprocessing.normalize( p_look_at - p_body )
    z_axis_cam = sklearn.preprocessing.normalize( cross(y_axis_cam_hint, x_axis_cam) )
    y_axis_cam = sklearn.preprocessing.normalize( cross(x_axis_cam,      z_axis_cam) )

    theta_cam  = zeros(num_timesteps)
    psi_cam    = zeros(num_timesteps)
    phi_cam    = zeros(num_timesteps)

    for ti in range(num_timesteps):

        z_axis_cam_ti = c_[matrix(z_axis_cam[ti]),0].T
        y_axis_cam_ti = c_[matrix(y_axis_cam[ti]),0].T
        x_axis_cam_ti = c_[matrix(x_axis_cam[ti]),0].T

        R_world_from_cam_ti  = matrix(c_[z_axis_cam_ti, y_axis_cam_ti, x_axis_cam_ti, [0,0,0,1]])
        R_world_from_body_ti = matrix(R_world_from_body[ti])
        R_body_from_cam_ti   = R_world_from_body_ti.T*R_world_from_cam_ti

        psi_cam_ti, theta_cam_ti, phi_cam_ti = transformations.euler_from_matrix(R_body_from_cam_ti,"ryxz")

        #
        # these two sets of Euler angles aren't used anywhere, but can be used to verify that
        # the R_world_from_body Euler angles will not be the same as the R_world_from_cam Euler
        # angles in general (the three asserts immediately below will fail)
        #
        # psi_world_from_cam_ti,  theta_world_from_cam_ti,  phi_world_from_cam_ti  = transformations.euler_from_matrix(R_world_from_cam_ti,"ryxz")
        # psi_world_from_body_ti, theta_world_from_body_ti, phi_world_from_body_ti = transformations.euler_from_matrix(R_world_from_body_ti,"ryxz")
        #        
        # assert allclose(psi_world_from_cam_ti,   psi_world_from_body_ti)
        # assert allclose(theta_world_from_cam_ti, theta_world_from_body_ti)
        # assert allclose(phi_world_from_cam_ti,   phi_world_from_body_ti)
        #

        #
        # sanity check that world-from-body rotation matrix we compute actually
        # maps the vector [0,0,1] to the quadrotor x axis
        #
        assert allclose(c_[matrix(x_axis_body[ti]),1].T, R_world_from_body_ti*matrix([0,0,1,1]).T)

        #
        # sanity check that the world-from-camera rotation matrix we compute actually
        # maps the vector [0,0,1] to the camera x axis
        #
        assert allclose(c_[matrix(x_axis_cam[ti]),1].T, R_world_from_cam_ti*matrix([0,0,1,1]).T)

        #
        # sanity check that the world-from-body and body-from-camera rotation matrices
        # we compute actually maps the vector [0,0,1] to the camera x axis
        #        
        assert allclose(c_[matrix(x_axis_cam[ti]),1].T, R_world_from_body_ti*R_body_from_cam_ti*matrix([0,0,1,1]).T)

        theta_cam[ti] = theta_cam_ti
        psi_cam[ti]   = psi_cam_ti
        phi_cam[ti]   = phi_cam_ti

    theta_cam = trigutils.compute_continuous_angle_array(theta_cam)
    psi_cam   = trigutils.compute_continuous_angle_array(psi_cam)
    phi_cam   = trigutils.compute_continuous_angle_array(phi_cam)

    #
    # assert that we never need any camera yaw in the body frame of the quad
    #
    assert allclose(psi_cam, 0)

    #
    # compute derivatives
    #
    theta_body_dotN = gradientutils.gradients_scalar_wrt_scalar_smooth_boundaries(theta_body,dt,max_gradient=2,poly_deg=5)
    psi_body_dotN   = gradientutils.gradients_scalar_wrt_scalar_smooth_boundaries(psi_body,dt,max_gradient=2,poly_deg=5)
    phi_body_dotN   = gradientutils.gradients_scalar_wrt_scalar_smooth_boundaries(phi_body,dt,max_gradient=2,poly_deg=5)

    theta_cam_dotN = gradientutils.gradients_scalar_wrt_scalar_smooth_boundaries(theta_cam,dt,max_gradient=2,poly_deg=5)
    psi_cam_dotN   = gradientutils.gradients_scalar_wrt_scalar_smooth_boundaries(psi_cam,dt,max_gradient=2,poly_deg=5)
    phi_cam_dotN   = gradientutils.gradients_scalar_wrt_scalar_smooth_boundaries(phi_cam,dt,max_gradient=2,poly_deg=5)

    theta_body_dot     = theta_body_dotN[1]
    psi_body_dot       = psi_body_dotN[1]
    phi_body_dot       = phi_body_dotN[1]

    theta_cam_dot      = theta_cam_dotN[1]
    psi_cam_dot        = psi_cam_dotN[1]
    phi_cam_dot        = phi_cam_dotN[1]

    theta_body_dot_dot = theta_body_dotN[2]
    psi_body_dot_dot   = psi_body_dotN[2]
    phi_body_dot_dot   = phi_body_dotN[2]

    theta_cam_dot_dot  = theta_cam_dotN[2]
    psi_cam_dot_dot    = psi_cam_dotN[2]
    phi_cam_dot_dot    = phi_cam_dotN[2]

    return p_body, p_body_dot, p_body_dot_dot, theta_body, theta_body_dot, theta_body_dot_dot, psi_body, psi_body_dot, psi_body_dot_dot, phi_body, phi_body_dot, phi_body_dot_dot, theta_cam, theta_cam_dot, theta_cam_dot_dot, psi_cam, psi_cam_dot, psi_cam_dot_dot, phi_cam, phi_cam_dot, phi_cam_dot_dot



def compute_control_trajectory(q_q_dot_q_dot_dot):

    p_body, p_body_dot, p_body_dot_dot, theta_body, theta_body_dot, theta_body_dot_dot, psi_body, psi_body_dot, psi_body_dot_dot, phi_body, phi_body_dot, phi_body_dot_dot, theta_cam, theta_cam_dot, theta_cam_dot_dot, psi_cam, psi_cam_dot, psi_cam_dot_dot, phi_cam, phi_cam_dot, phi_cam_dot_dot = q_q_dot_q_dot_dot

    num_timesteps = len(p_body)
    
    u = zeros((num_timesteps,7))

    for ti in range(num_timesteps):

        q_ti         = matrix( [ p_body[ti,0],         p_body[ti,1],         p_body[ti,2],         theta_body[ti],         psi_body[ti],         phi_body[ti],         theta_cam[ti],         psi_cam[ti],         phi_cam[ti]         ] ).T
        q_dot_ti     = matrix( [ p_body_dot[ti,0],     p_body_dot[ti,1],     p_body_dot[ti,2],     theta_body_dot[ti],     psi_body_dot[ti],     phi_body_dot[ti],     theta_cam_dot[ti],     psi_cam_dot[ti],     phi_cam_dot[ti]     ] ).T
        q_dot_dot_ti = matrix( [ p_body_dot_dot[ti,0], p_body_dot_dot[ti,1], p_body_dot_dot[ti,2], theta_body_dot_dot[ti], psi_body_dot_dot[ti], phi_body_dot_dot[ti], theta_cam_dot_dot[ti], psi_cam_dot_dot[ti], phi_cam_dot_dot[ti] ] ).T
        x_ti         = bmat( [[q_ti],[q_dot_ti]] )
        
        H, C, G, B = compute_manipulator_matrices(x_ti)

        u_ti    = linalg.pinv(B)*(H*q_dot_dot_ti + C*q_dot_ti + G)
        u[ti,:] = matrix(u_ti).A1

    return u



def draw(x, q_q_dot_q_dot_dot_nominal=None, p_look_at_nominal=None, t=None):

    import mayavi
    import mayavi.mlab
    import vtk

    if q_q_dot_q_dot_dot_nominal is not None:
        assert t is not None

    if q_q_dot_q_dot_dot_nominal is not None:
        p_body, p_body_dot, p_body_dot_dot, theta_body, theta_body_dot, theta_body_dot_dot, psi_body, psi_body_dot, psi_body_dot_dot, phi_body, phi_body_dot, phi_body_dot_dot, theta_cam, theta_cam_dot, theta_cam_dot_dot, psi_cam, psi_cam_dot, psi_cam_dot_dot, phi_cam, phi_cam_dot, phi_cam_dot_dot = q_q_dot_q_dot_dot_nominal

    d_quad_center_to_cross_beam_end                        = 1.0
    d_quad_center_to_cross_beam_end_xsquared_plus_zsquared = d_quad_center_to_cross_beam_end**2
    d_quad_center_to_cross_beam_end_x                      = sqrt(d_quad_center_to_cross_beam_end_xsquared_plus_zsquared/2.0)
    d_quad_center_to_cross_beam_end_z                      = sqrt(d_quad_center_to_cross_beam_end_xsquared_plus_zsquared/2.0)
    d_quad_cross_beam_end_to_vert_beam_end                 = 0.4
    d_quad_axis_length                                     = 1.0
    d_cam_axis_length                                      = 10.0

    s_quad_center        = 0.5
    s_quad_cross_beam    = 0.05
    s_quad_vert_beam     = 0.05
    s_quad_prop          = 0.25
    s_quad_axis          = 0.1
    s_world_axis         = 0.025
    s_body_trajectory    = 0.075
    s_look_at_center     = 0.5
    s_look_at_trajectory = 0.075
    s_cam_axis           = 0.05

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
    p_look_at_center        = matrix([ 0.0,                                0.0,                                    0.0,                                1.0 ]).T
    p_cam_center            = matrix([ 0.0,                                0.0,                                    0.0,                                1.0 ]).T
    p_cam_axis_end_z        = matrix([ d_cam_axis_length,                  0.0,                                    0.0,                                1.0 ]).T
    p_cam_axis_end_y        = matrix([ 0.0,                                d_cam_axis_length,                      0.0,                                1.0 ]).T
    p_cam_axis_end_x        = matrix([ 0.0,                                0.0,                                    d_cam_axis_length,                  1.0 ]).T

    c_quad_center     = (1.0,1.0,1.0)
    c_quad_cross_beam = (1.0,1.0,1.0)
    c_quad_vert_beam  = (1.0,1.0,1.0)
    c_quad_prop_0     = (0.25,0.25,0.25)
    c_quad_prop_1     = (0.5,0.5,0.5)
    c_quad_prop_2     = (0.75,0.75,0.75)
    c_quad_prop_3     = (1.0,1.0,1.0)
    c_quad_axis_z     = (0.0,0.0,0.5)
    c_quad_axis_y     = (0.0,0.5,0.0)
    c_quad_axis_x     = (0.5,0.0,0.0)
    c_world_axis_z    = (0.0,0.0,1.0)
    c_world_axis_y    = (0.0,1.0,0.0)
    c_world_axis_x    = (1.0,0.0,0.0)
    c_look_at_center  = (1.0,1.0,1.0)
    c_cam_axis_z      = (0.0,0.0,1.0)
    c_cam_axis_y      = (0.0,1.0,0.0)
    c_cam_axis_x      = (1.0,0.0,0.0)

    # quadrotor center
    points3d_quad_center = mayavi.mlab.points3d(0.0, 0.0, 0.0, scale_factor=s_quad_center, color=c_quad_center)

    # quadrotor cross beams
    plot3d_quad_cross_beam_0 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_cross_beam, color=c_quad_cross_beam)
    plot3d_quad_cross_beam_1 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_cross_beam, color=c_quad_cross_beam)

    # quadrotor vertical beams
    plot3d_quad_vert_beam_0 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_vert_beam, color=c_quad_vert_beam)
    plot3d_quad_vert_beam_1 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_vert_beam, color=c_quad_vert_beam)
    plot3d_quad_vert_beam_2 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_vert_beam, color=c_quad_vert_beam)
    plot3d_quad_vert_beam_3 = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_vert_beam, color=c_quad_vert_beam)

    # quadrotor props
    points3d_quad_prop_0 = mayavi.mlab.points3d(0.0, 0.0, 0.0, scale_factor=s_quad_prop, color=c_quad_prop_0)
    points3d_quad_prop_1 = mayavi.mlab.points3d(0.0, 0.0, 0.0, scale_factor=s_quad_prop, color=c_quad_prop_1)
    points3d_quad_prop_2 = mayavi.mlab.points3d(0.0, 0.0, 0.0, scale_factor=s_quad_prop, color=c_quad_prop_2)
    points3d_quad_prop_3 = mayavi.mlab.points3d(0.0, 0.0, 0.0, scale_factor=s_quad_prop, color=c_quad_prop_3)

    # quadrotor reference frame
    plot3d_quad_axis_z = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_axis, color=c_quad_axis_z)
    plot3d_quad_axis_y = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_axis, color=c_quad_axis_y)
    plot3d_quad_axis_x = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_quad_axis, color=c_quad_axis_x)

    # look at center
    points3d_look_at_center = mayavi.mlab.points3d(0.0, 0.0, 0.0, scale_factor=s_look_at_center, color=c_look_at_center)

    # camera reference frame
    plot3d_cam_axis_z = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_cam_axis, color=c_cam_axis_z)
    plot3d_cam_axis_y = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_cam_axis, color=c_cam_axis_y)
    plot3d_cam_axis_x = mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,1.0], tube_radius=s_cam_axis, color=c_cam_axis_x)

    # global reference frame
    mayavi.mlab.plot3d([0.0,0.0], [0.0,0.0], [0.0,5.0], tube_radius=s_world_axis, color=c_world_axis_z, transparent=True)
    mayavi.mlab.plot3d([0.0,0.0], [0.0,5.0], [0.0,0.0], tube_radius=s_world_axis, color=c_world_axis_y, transparent=True)
    mayavi.mlab.plot3d([0.0,5.0], [0.0,0.0], [0.0,0.0], tube_radius=s_world_axis, color=c_world_axis_x, transparent=True)

    if q_q_dot_q_dot_dot_nominal is not None:

        # body trajectory
        pts = mayavi.mlab.quiver3d(p_body[:,2], p_body[:,1], p_body[:,0], ones(p_body.shape[0]), ones(p_body.shape[0]), ones(p_body.shape[0]), scalars=t, mode="sphere", scale_factor=s_body_trajectory)
        pts.glyph.color_mode = "color_by_scalar"
        pts.glyph.glyph_source.glyph_source.center = [0,0,0]

    if p_look_at_nominal is not None:

        # look at trajectory
        mayavi.mlab.plot3d(p_look_at_nominal[:,2], p_look_at_nominal[:,1], p_look_at_nominal[:,0], t, tube_radius=s_look_at_trajectory)

    mayavi.mlab.view(0,0)

    def draw_quad(x_ti, p_look_at_ti=None):

        p_body_ti, theta_body_ti, psi_body_ti, phi_body_ti, theta_cam_ti, psi_cam_ti, phi_cam_ti, p_body_dot_ti, theta_body_dot_ti, psi_body_dot_ti, phi_body_dot_ti, theta_cam_dot_ti, psi_cam_dot_ti, phi_cam_dot_ti, q_ti, q_dot_ti = unpack_state(x_ti)

        R_world_from_body = matrix(transformations.euler_matrix(psi_body_ti,theta_body_ti,phi_body_ti,"ryxz"))
        T                 = matrix(transformations.translation_matrix(p_body_ti.A1))
        M                 = T * R_world_from_body

        p_world_body_center      = M * p_body_center
        p_world_cross_beam_end_0 = M * p_body_cross_beam_end_0
        p_world_cross_beam_end_1 = M * p_body_cross_beam_end_1
        p_world_cross_beam_end_2 = M * p_body_cross_beam_end_2
        p_world_cross_beam_end_3 = M * p_body_cross_beam_end_3
        p_world_vert_beam_end_0  = M * p_body_vert_beam_end_0
        p_world_vert_beam_end_1  = M * p_body_vert_beam_end_1
        p_world_vert_beam_end_2  = M * p_body_vert_beam_end_2
        p_world_vert_beam_end_3  = M * p_body_vert_beam_end_3
        p_world_body_axis_end_z  = M * p_body_axis_end_z
        p_world_body_axis_end_y  = M * p_body_axis_end_y
        p_world_body_axis_end_x  = M * p_body_axis_end_x

        p_world_body_center      = ( p_world_body_center      / p_world_body_center[3]      ).A1
        p_world_cross_beam_end_0 = ( p_world_cross_beam_end_0 / p_world_cross_beam_end_0[3] ).A1
        p_world_cross_beam_end_1 = ( p_world_cross_beam_end_1 / p_world_cross_beam_end_1[3] ).A1
        p_world_cross_beam_end_2 = ( p_world_cross_beam_end_2 / p_world_cross_beam_end_2[3] ).A1
        p_world_cross_beam_end_3 = ( p_world_cross_beam_end_3 / p_world_cross_beam_end_3[3] ).A1
        p_world_vert_beam_end_0  = ( p_world_vert_beam_end_0  / p_world_vert_beam_end_0[3]  ).A1
        p_world_vert_beam_end_1  = ( p_world_vert_beam_end_1  / p_world_vert_beam_end_1[3]  ).A1
        p_world_vert_beam_end_2  = ( p_world_vert_beam_end_2  / p_world_vert_beam_end_2[3]  ).A1
        p_world_vert_beam_end_3  = ( p_world_vert_beam_end_3  / p_world_vert_beam_end_3[3]  ).A1
        p_world_body_axis_end_z  = ( p_world_body_axis_end_z  / p_world_body_axis_end_z[3]  ).A1
        p_world_body_axis_end_y  = ( p_world_body_axis_end_y  / p_world_body_axis_end_y[3]  ).A1
        p_world_body_axis_end_x  = ( p_world_body_axis_end_x  / p_world_body_axis_end_x[3]  ).A1

        R_body_from_cam = matrix(transformations.euler_matrix(psi_cam_ti,theta_cam_ti,phi_cam_ti,"ryxz"))
        M               = T * R_world_from_body * R_body_from_cam

        p_world_cam_center       = M * p_cam_center
        p_world_cam_axis_end_z   = M * p_cam_axis_end_z
        p_world_cam_axis_end_y   = M * p_cam_axis_end_y
        p_world_cam_axis_end_x   = M * p_cam_axis_end_x

        p_world_cam_center       = ( p_world_cam_center     / p_world_cam_center[3]     ).A1
        p_world_cam_axis_end_z   = ( p_world_cam_axis_end_z / p_world_cam_axis_end_z[3] ).A1
        p_world_cam_axis_end_y   = ( p_world_cam_axis_end_y / p_world_cam_axis_end_y[3] ).A1
        p_world_cam_axis_end_x   = ( p_world_cam_axis_end_x / p_world_cam_axis_end_x[3] ).A1

        # quadrotor center
        points3d_quad_center.mlab_source.set( x=p_world_body_center[2], y=p_world_body_center[1], z=p_world_body_center[0] )

        # quadrotor cross beams
        plot3d_quad_cross_beam_0.mlab_source.set( x=[p_world_cross_beam_end_0[2],p_world_cross_beam_end_2[2]], y=[p_world_cross_beam_end_0[1],p_world_cross_beam_end_2[1]], z=[p_world_cross_beam_end_0[0],p_world_cross_beam_end_2[0]] )
        plot3d_quad_cross_beam_1.mlab_source.set( x=[p_world_cross_beam_end_1[2],p_world_cross_beam_end_3[2]], y=[p_world_cross_beam_end_1[1],p_world_cross_beam_end_3[1]], z=[p_world_cross_beam_end_1[0],p_world_cross_beam_end_3[0]] )

        # quadrotor vertical beams
        plot3d_quad_vert_beam_0.mlab_source.set( x=[p_world_cross_beam_end_0[2],p_world_vert_beam_end_0[2]], y=[p_world_cross_beam_end_0[1],p_world_vert_beam_end_0[1]], z=[p_world_cross_beam_end_0[0],p_world_vert_beam_end_0[0]] )
        plot3d_quad_vert_beam_1.mlab_source.set( x=[p_world_cross_beam_end_1[2],p_world_vert_beam_end_1[2]], y=[p_world_cross_beam_end_1[1],p_world_vert_beam_end_1[1]], z=[p_world_cross_beam_end_1[0],p_world_vert_beam_end_1[0]] )
        plot3d_quad_vert_beam_2.mlab_source.set( x=[p_world_cross_beam_end_2[2],p_world_vert_beam_end_2[2]], y=[p_world_cross_beam_end_2[1],p_world_vert_beam_end_2[1]], z=[p_world_cross_beam_end_2[0],p_world_vert_beam_end_2[0]] )
        plot3d_quad_vert_beam_3.mlab_source.set( x=[p_world_cross_beam_end_3[2],p_world_vert_beam_end_3[2]], y=[p_world_cross_beam_end_3[1],p_world_vert_beam_end_3[1]], z=[p_world_cross_beam_end_3[0],p_world_vert_beam_end_3[0]] )

        # quadrotor props
        points3d_quad_prop_0.mlab_source.set( x=p_world_vert_beam_end_0[2], y=p_world_vert_beam_end_0[1], z=p_world_vert_beam_end_0[0] )
        points3d_quad_prop_1.mlab_source.set( x=p_world_vert_beam_end_1[2], y=p_world_vert_beam_end_1[1], z=p_world_vert_beam_end_1[0] )
        points3d_quad_prop_2.mlab_source.set( x=p_world_vert_beam_end_2[2], y=p_world_vert_beam_end_2[1], z=p_world_vert_beam_end_2[0] )
        points3d_quad_prop_3.mlab_source.set( x=p_world_vert_beam_end_3[2], y=p_world_vert_beam_end_3[1], z=p_world_vert_beam_end_3[0] )

        # quadrotor reference frame
        plot3d_quad_axis_z.mlab_source.set( x=[p_world_body_center[2],p_world_body_axis_end_z[2]], y=[p_world_body_center[1],p_world_body_axis_end_z[1]], z=[p_world_body_center[0],p_world_body_axis_end_z[0]] )
        plot3d_quad_axis_y.mlab_source.set( x=[p_world_body_center[2],p_world_body_axis_end_y[2]], y=[p_world_body_center[1],p_world_body_axis_end_y[1]], z=[p_world_body_center[0],p_world_body_axis_end_y[0]] )
        plot3d_quad_axis_x.mlab_source.set( x=[p_world_body_center[2],p_world_body_axis_end_x[2]], y=[p_world_body_center[1],p_world_body_axis_end_x[1]], z=[p_world_body_center[0],p_world_body_axis_end_x[0]] )

        if p_look_at_nominal is not None:

            # look at center
            points3d_look_at_center.mlab_source.set( x=p_look_at_ti[2], y=p_look_at_ti[1], z=p_look_at_ti[0] )

        # camera reference frame
        plot3d_cam_axis_z.mlab_source.set( x=[p_world_cam_center[2],p_world_cam_axis_end_z[2]], y=[p_world_cam_center[1],p_world_cam_axis_end_z[1]], z=[p_world_cam_center[0],p_world_cam_axis_end_z[0]] )
        plot3d_cam_axis_y.mlab_source.set( x=[p_world_cam_center[2],p_world_cam_axis_end_y[2]], y=[p_world_cam_center[1],p_world_cam_axis_end_y[1]], z=[p_world_cam_center[0],p_world_cam_axis_end_y[0]] )
        plot3d_cam_axis_x.mlab_source.set( x=[p_world_cam_center[2],p_world_cam_axis_end_x[2]], y=[p_world_cam_center[1],p_world_cam_axis_end_x[1]], z=[p_world_cam_center[0],p_world_cam_axis_end_x[0]] )

    @mayavi.mlab.animate
    def mlab_animate():

        ti = 0

        while 1:

            ti   = (ti+1)%len(x)
            x_ti = matrix(x[ti]).T

            if p_look_at_nominal is not None:
                p_look_at_nominal_ti = matrix(p_look_at_nominal[ti]).T
            else:
                p_look_at_nominal_ti = None

            draw_quad(x_ti, p_look_at_nominal_ti)

            yield

    mlab_animate()

    vtk_debug_output = vtk.vtkFileOutputWindow()
    vtk_debug_output.SetFileName("vtk_debug_output.txt")
    vtk.vtkOutputWindow().SetInstance(vtk_debug_output)

    mayavi.mlab.show()