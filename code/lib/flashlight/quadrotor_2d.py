from pylab import *

import matplotlib.animation
import scipy.interpolate
import sklearn.preprocessing
import sympy
import sympy.matrices
import sympy.physics
import sympy.physics.mechanics
import sympy.physics.mechanics.functions
import time
import transformations

import gradient_utils
import interpolate_utils
import path_utils
import sympy_utils
import trig_utils



m = 1.0      # mass
g = 9.8      # gravity
d = 1.0      # distance from arm to center
I = m*d**2.0 # moment of intertia

f_gravity_world  = matrix([-m*g,0]).T
f_external_world = f_gravity_world

num_q_dims  = 3
num_x_dims  = 2*num_q_dims
num_u_dims  = 2
num_dt_dims = 1

num_dims_g_dynamics_ti = num_x_dims

build_sympy_modules_on_import = False
# build_sympy_modules_on_import = False

const_vals = hstack( [ d, m, I, f_external_world.A1 ] )



print "flashlight.quadrotor_2d: Constructing sympy symbols..."
sys_time_begin = time.time()

# constants
d_expr = sympy.Symbol("d", real=True)
m_expr = sympy.Symbol("m", real=True)
I_expr = sympy.Symbol("I", real=True)

f_external_expr, f_external_expr_entries = sympy_utils.construct_matrix_and_entries("f_external",(2,1), real=True)

# variables
t_expr = sympy.Symbol("t", real=True)

p_y_expr   = sympy.Symbol("p_y",   real=True)(t_expr)
p_x_expr   = sympy.Symbol("p_x",   real=True)(t_expr)
theta_expr = sympy.Symbol("theta", real=True)(t_expr)

# state and control vectors
q_expr                 = sympy.Matrix([p_y_expr,p_x_expr,theta_expr])
q_dot_expr             = q_expr.diff(t_expr)
x_expr                 = sympy_utils.construct_matrix_from_block_matrix( sympy.Matrix( [ [q_expr], [q_dot_expr] ] ) )    
u_expr, u_expr_entries = sympy_utils.construct_matrix_and_entries("u",(num_u_dims,1), real=True)

# symbols to solve for g_dynamics, given x_current x_next u_current
x_current_expr, x_current_expr_entries = sympy_utils.construct_matrix_and_entries("x_current", (num_x_dims,1), real=True)
x_next_expr,    x_next_expr_entries    = sympy_utils.construct_matrix_and_entries("x_next",    (num_x_dims,1), real=True)
u_current_expr, u_current_expr_entries = sympy_utils.construct_matrix_and_entries("u_current", (num_u_dims,1), real=True)
dt_current_expr                        = sympy.Symbol("delta_t_current", real=True)

# symbol collections
const_syms                                                   = hstack( [ d_expr, m_expr, I_expr, matrix(f_external_expr).A1 ] )
const_and_x_syms                                             = hstack( [ const_syms, matrix(x_expr).A1 ] )
const_and_x_and_u_syms                                       = hstack( [ const_syms, matrix(x_expr).A1, matrix(u_expr).A1 ] )
const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms = hstack( [ const_syms, matrix(x_current_expr).A1, matrix(x_next_expr).A1, matrix(u_current_expr).A1, dt_current_expr ] )

sys_time_end = time.time()
print "flashlight.quadrotor_2d: Finished constructing sympy symbols (%.03f seconds)." % (sys_time_end - sys_time_begin)



def construct_manipulator_matrix_expressions(x_expr,t_expr):

    print "flashlight.quadrotor_2d: Constructing manipulator matrix expressions..."

    theta_expr = x_expr[2,0]

    H_00_expr = sympy.Matrix(m_expr*sympy.eye(2))
    H_01_expr = sympy.Matrix.zeros(2,1)
    H_10_expr = sympy.Matrix.zeros(1,2)
    H_11_expr = sympy.Matrix([I_expr])

    C_00_expr = sympy.Matrix.zeros(2,2)
    C_01_expr = sympy.Matrix.zeros(2,1)
    C_10_expr = sympy.Matrix.zeros(1,2)
    C_11_expr = sympy.Matrix.zeros(1,1)

    G_0_expr = -f_external_expr
    G_1_expr = sympy.Matrix.zeros(1,1)

    B_0_expr  = sympy.Matrix( [ [sympy.cos(theta_expr),sympy.cos(theta_expr)], [-sympy.sin(theta_expr),-sympy.sin(theta_expr)] ] )
    B_1_expr  = sympy.Matrix( [ [-d_expr,d_expr] ] )

    H_expr = sympy_utils.construct_matrix_from_block_matrix( sympy.Matrix( [ [H_00_expr, H_01_expr], [H_10_expr, H_11_expr] ] ) )
    C_expr = sympy_utils.construct_matrix_from_block_matrix( sympy.Matrix( [ [C_00_expr, C_01_expr], [C_10_expr, C_11_expr] ] ) )
    G_expr = sympy_utils.construct_matrix_from_block_matrix( sympy.Matrix( [ [G_0_expr],             [G_1_expr] ] ) )
    B_expr = sympy_utils.construct_matrix_from_block_matrix( sympy.Matrix( [ [B_0_expr],             [B_1_expr] ] ) )

    print "flashlight.quadrotor_2d: Finished constructing manipulator matrix expressions..."

    return H_expr,C_expr,G_expr,B_expr



def build_sympy_modules():

    print "flashlight.quadrotor_2d: Constructing sympy expressions..."
    sys_time_begin = time.time()

    # manipulator matrices
    H_expr,C_expr,G_expr,B_expr = construct_manipulator_matrix_expressions(x_expr,t_expr)

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
    print "flashlight.quadrotor_2d: Finished constructing sympy expressions (%.03f seconds)." % (sys_time_end - sys_time_begin)

    print "flashlight.quadrotor_2d: Building sympy modules..."
    sys_time_begin = time.time()

    current_source_file_path = path_utils.get_current_source_file_path()

    sympy_utils.build_module_autowrap( expr=H_expr, syms=const_and_x_syms, module_name="quadrotor_2d_H", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_2d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympy_utils.build_module_autowrap( expr=C_expr, syms=const_and_x_syms, module_name="quadrotor_2d_C", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_2d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympy_utils.build_module_autowrap( expr=G_expr, syms=const_and_x_syms, module_name="quadrotor_2d_G", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_2d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympy_utils.build_module_autowrap( expr=B_expr, syms=const_and_x_syms, module_name="quadrotor_2d_B", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_2d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )

    sympy_utils.build_module_autowrap( expr=x_dot_expr, syms=const_and_x_and_u_syms, module_name="quadrotor_2d_x_dot", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_2d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympy_utils.build_module_autowrap( expr=df_dx_expr, syms=const_and_x_and_u_syms, module_name="quadrotor_2d_df_dx", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_2d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    sympy_utils.build_module_autowrap( expr=df_du_expr, syms=const_and_x_and_u_syms, module_name="quadrotor_2d_df_du", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor_2d", dummify=True, build_vectorized=True, verbose=True, request_delete_tmp_dir=True )

    # sympy_utils.build_module_autowrap( expr=g_dynamics_ti_expr,           syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotor2d_g_dynamics_ti",           tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor2d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    # sympy_utils.build_module_autowrap( expr=dgdynamicsti_dxcurrent_expr,  syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotor2d_dgdynamicsti_dxcurrent",  tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor2d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    # sympy_utils.build_module_autowrap( expr=dgdynamicsti_dxnext_expr,     syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotor2d_dgdynamicsti_dxnext",     tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor2d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    # sympy_utils.build_module_autowrap( expr=dgdynamicsti_ducurrent_expr,  syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotor2d_dgdynamicsti_ducurrent",  tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor2d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )
    # sympy_utils.build_module_autowrap( expr=dgdynamicsti_ddtcurrent_expr, syms=const_and_xcurrent_and_xnext_and_ucurrent_and_dtcurrent_syms, module_name="quadrotor2d_dgdynamicsti_ddtcurrent", tmp_dir="tmp", out_dir=current_source_file_path+"/data/quadrotor2d", build_vectorized=True, verbose=True, request_delete_tmp_dir=True )

    sys_time_end = time.time()
    print "flashlight.quadrotor_2d: Finished building sympy modules (%.03f seconds)." % (sys_time_end - sys_time_begin)



if build_sympy_modules_on_import:
    build_sympy_modules()

print "flashlight.quadrotor_2d: Loading sympy modules..."
sys_time_begin = time.time()

current_source_file_path = path_utils.get_current_source_file_path()

H_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_2d_H", path=current_source_file_path+"/data/quadrotor_2d" )
C_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_2d_C", path=current_source_file_path+"/data/quadrotor_2d" )
G_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_2d_G", path=current_source_file_path+"/data/quadrotor_2d" )
B_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_2d_B", path=current_source_file_path+"/data/quadrotor_2d" )

x_dot_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_2d_x_dot", path=current_source_file_path+"/data/quadrotor_2d" )
df_dx_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_2d_df_dx", path=current_source_file_path+"/data/quadrotor_2d" )
df_du_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_2d_df_du", path=current_source_file_path+"/data/quadrotor_2d" )

# g_dynamics_ti_autowrap           = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_g_dynamics_ti",           path=current_source_file_path+"/data/quadrotor2d" )
# dgdynamicsti_dxcurrent_autowrap  = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_dgdynamicsti_dxcurrent",  path=current_source_file_path+"/data/quadrotor2d" )
# dgdynamicsti_dxnext_autowrap     = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_dgdynamicsti_dxnext",     path=current_source_file_path+"/data/quadrotor2d" )
# dgdynamicsti_ducurrent_autowrap  = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_dgdynamicsti_ducurrent",  path=current_source_file_path+"/data/quadrotor2d" )
# dgdynamicsti_ddtcurrent_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_dgdynamicsti_ddtcurrent", path=current_source_file_path+"/data/quadrotor2d" )

H_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_2d_H_vectorized", path=current_source_file_path+"/data/quadrotor_2d" )
C_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_2d_C_vectorized", path=current_source_file_path+"/data/quadrotor_2d" )
G_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_2d_G_vectorized", path=current_source_file_path+"/data/quadrotor_2d" )
B_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor_2d_B_vectorized", path=current_source_file_path+"/data/quadrotor_2d" )

# x_dot_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_x_dot_vectorized", path=current_source_file_path+"/data/quadrotor2d" )
# df_dx_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_df_dx_vectorized", path=current_source_file_path+"/data/quadrotor2d" )
# df_du_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_df_du_vectorized", path=current_source_file_path+"/data/quadrotor2d" )

# g_dynamics_ti_vectorized_autowrap           = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_g_dynamics_ti_vectorized",           path=current_source_file_path+"/data/quadrotor2d" )
# dgdynamicsti_dxcurrent_vectorized_autowrap  = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_dgdynamicsti_dxcurrent_vectorized",  path=current_source_file_path+"/data/quadrotor2d" )
# dgdynamicsti_dxnext_vectorized_autowrap     = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_dgdynamicsti_dxnext_vectorized",     path=current_source_file_path+"/data/quadrotor2d" )
# dgdynamicsti_ducurrent_vectorized_autowrap  = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_dgdynamicsti_ducurrent_vectorized",  path=current_source_file_path+"/data/quadrotor2d" )
# dgdynamicsti_ddtcurrent_vectorized_autowrap = sympy_utils.import_anon_func_from_from_module_autowrap( module_name="quadrotor2d_dgdynamicsti_ddtcurrent_vectorized", path=current_source_file_path+"/data/quadrotor2d" )

sys_time_end = time.time()
print "flashlight.quadrotor_2d: Finished loading sympy modules (%.03f seconds)." % (sys_time_end - sys_time_begin)



def pack_state(p, theta, p_dot, theta_dot):

    x     = matrix( [ p.item(0), p.item(1), theta, p_dot.item(0), p_dot.item(1), theta_dot ] ).T
    q     = matrix( [ p.item(0), p.item(1), theta ] ).T
    q_dot = matrix( [ p_dot.item(0), p_dot.item(1), theta_dot ] ).T

    return x, q, q_dot

def unpack_state(x):

    p         = matrix( [ x.item(0), x.item(1) ] ).T
    theta     = x.item(2)
    p_dot     = matrix( [ x.item(3), x.item(4) ] ).T
    theta_dot = x.item(5)
    q         = matrix( [ p.item(0),     p.item(1),     theta ] ).T
    q_dot     = matrix( [ p_dot.item(0), p_dot.item(1), theta_dot ] ).T

    return p, theta, p_dot, theta_dot, q, q_dot

def pack_state_space_trajectory(p, theta, p_dot, theta_dot):

    x     = c_[ p, theta, p_dot, theta_dot ]
    q     = c_[ p, theta ]
    q_dot = c_[ p_dot, theta_dot ]

    return x, q, q_dot, q_dot_dot

def unpack_state_space_trajectory(x):

    p         = x[:,0:2]
    theta     = x[:,2]
    p_dot     = x[:,3:5]
    theta_dot = x[:,5]
    q         = x[:,0:3]
    q_dot     = x[:,3:6]

    return p, theta, p_dot, theta_dot, q, q_dot

def pack_state_space_trajectory_and_derivatives(q_qdot_qdotdot):

    p, p_dot, p_dot_dot, theta, theta_dot, theta_dot_dot = q_qdot_qdotdot

    x         = c_[ p, theta, p_dot, theta_dot ]
    q         = c_[ p, theta ]
    q_dot     = c_[ p_dot, theta_dot ]
    q_dot_dot = c_[ p_dot_dot, theta_dot_dot ]

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

#         gi_begin            = (ti+0)*num_dims_g_dynamics_ti
#         gi_end              = (ti+1)*num_dims_g_dynamics_ti
#         ai_x_current_begin  = (ti+0)*num_x_dims
#         ai_x_current_end    = (ti+1)*num_x_dims
#         ai_x_next_begin     = (ti+1)*num_x_dims
#         ai_x_next_end       = (ti+2)*num_x_dims
#         ai_u_current_begin  = (ti+0)*num_u_dims
#         ai_u_current_end    = (ti+1)*num_u_dims

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



# def compute_dynamics_constraints_jacobian_direct_transcription_nonzero(X,U,dt):

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

#         gi_begin            = (ti+0)*num_dims_g_dynamics_ti
#         gi_end              = (ti+1)*num_dims_g_dynamics_ti
#         ai_x_current_begin  = (ti+0)*num_x_dims
#         ai_x_current_end    = (ti+1)*num_x_dims
#         ai_x_next_begin     = (ti+1)*num_x_dims
#         ai_x_next_end       = (ti+2)*num_x_dims
#         ai_u_current_begin  = (ti+0)*num_u_dims
#         ai_u_current_end    = (ti+1)*num_u_dims

#         if isinstance(dt, ndarray) or isinstance(dt, matrix):
#             ai_dt_current_begin = (ti+0)*num_dt_dims
#             ai_dt_current_end   = (ti+1)*num_dt_dims
#         else:
#             ai_dt_current_begin = 0*num_dt_dims
#             ai_dt_current_end   = 1*num_dt_dims

#         dgdynamics_dX[gi_begin:gi_end,ai_x_current_begin:ai_x_current_end]    = 1
#         dgdynamics_dX[gi_begin:gi_end,ai_x_next_begin:ai_x_next_end]          = 1
#         dgdynamics_dU[gi_begin:gi_end,ai_u_current_begin:ai_u_current_end]    = 1
#         dgdynamics_ddt[gi_begin:gi_end,ai_dt_current_begin:ai_dt_current_end] = 1

#     return dgdynamics_dX, dgdynamics_dU, dgdynamics_ddt



def compute_state_space_trajectory_and_derivatives(p,dt):

    f_thrust_world_norm_threshold = 0.01

    num_timesteps = len(p)

    p_dotN    = gradient_utils.gradients_vector_wrt_scalar_smooth_boundaries(p,dt,max_gradient=2,poly_deg=5)
    p_dot     = p_dotN[1]
    p_dot_dot = p_dotN[2]

    f_thrust_world      = m*p_dot_dot - f_external_world.T.A
    f_thrust_world_norm = linalg.norm(f_thrust_world,axis=1)

    theta_raw  = arctan2(f_thrust_world[:,0],f_thrust_world[:,1]) - (pi/2.0)
    t_tmp      = arange(num_timesteps)
    theta_func = scipy.interpolate.interp1d(t_tmp[f_thrust_world_norm > f_thrust_world_norm_threshold], theta_raw[f_thrust_world_norm > f_thrust_world_norm_threshold], kind="linear")
    theta      = theta_func(t_tmp)

    theta         = trig_utils.compute_continuous_angle_array(theta)
    theta_dotN    = gradient_utils.gradients_scalar_wrt_scalar_smooth_boundaries(theta,dt,max_gradient=2,poly_deg=5)
    theta_dot     = theta_dotN[1]
    theta_dot_dot = theta_dotN[2]

    return p, p_dot, p_dot_dot, theta, theta_dot, theta_dot_dot



def compute_control_trajectory(q_qdot_qdotdot):

    x, q, q_dot, q_dot_dot = pack_state_space_trajectory_and_derivatives(q_qdot_qdotdot)
    num_timesteps          = q.shape[0]
    u                      = zeros((num_timesteps,num_u_dims))

    for ti in arange(num_timesteps):

        q_ti         = matrix( q[ti] ).T
        q_dot_ti     = matrix( q_dot[ti] ).T
        q_dot_dot_ti = matrix( q_dot_dot[ti] ).T
        x_ti         = matrix( x[ti] ).T
        H, C, G, B   = compute_manipulator_matrices(x_ti)
        u_ti         = linalg.pinv(B)*(H*q_dot_dot_ti + C*q_dot_ti + G)

        assert allclose(B*u_ti, H*q_dot_dot_ti + C*q_dot_ti + G)

        u[ti,:] = matrix(u_ti).A1

    return u



def draw(t, x, t_nominal=None, x_nominal=None, inline=False):

    if t_nominal is not None:
        assert x_nominal is not None

    if x_nominal is not None:
        assert t_nominal is not None

    if t_nominal is not None and x_nominal is not None:
        x_nominal_interp_func = interpolate_utils.interp1d_vector_wrt_scalar(t_nominal, x_nominal, kind="linear")
        p_nominal, theta_nominal, p_dot_nominal, theta_dot_nominal, q_nominal, q_dot_nominal = unpack_state_space_trajectory(x_nominal)

    x_interp_func = interpolate_utils.interp1d_vector_wrt_scalar(t, x, kind="linear")
    p, theta, p_dot, theta_dot, q, q_dot = unpack_state_space_trajectory(x)

    p_0_y       = p[:,0] + d*sin(theta+pi)
    p_0_x       = p[:,1] + d*cos(theta+pi)
    p_1_y       = p[:,0] + d*sin(theta)
    p_1_x       = p[:,1] + d*cos(theta)
    offset_up_y = 0.2*sin(theta+(pi/2.0))
    offset_up_x = 0.2*cos(theta+(pi/2.0))

    offset_up = matrix(c_[offset_up_y, offset_up_x])
    p_0       = matrix(c_[p_0_y, p_0_x])
    p_1       = matrix(c_[p_1_y, p_1_x])
    p_0_up    = p_0 + offset_up
    p_1_up    = p_1 + offset_up

    p_y_min = np.min(p[:,0]) - d
    p_y_max = np.max(p[:,0]) + d
    p_x_min = np.min(p[:,1]) - d
    p_x_max = np.max(p[:,1]) + d
    p_y_pad = 0.1*(p_y_max - p_y_min)
    p_x_pad = 0.1*(p_x_max - p_x_min)

    if not inline:
        plt.switch_backend("Qt4Agg")

    fig = figure()
    ax  = fig.add_subplot(111, autoscale_on=False, xlim=(p_x_min-p_x_pad, p_x_max+p_x_pad), ylim=(p_y_min-p_y_pad, p_y_max+p_y_pad))

    ax.set_aspect("equal")
    ax.grid()

    if inline:
        plt.close()

    if t_nominal is not None and x_nominal is not None:
        ax.scatter(p_nominal[:,1], p_nominal[:,0], c=t_nominal, cmap="jet", s=50, alpha=0.25)
        p_nominal_plot, = ax.plot([], [], "o-", lw=4, c="w", markersize=10)

    p0_p1_line,          = ax.plot([], [], "o-", lw=2)
    p0_pup0_line,        = ax.plot([], [], "o-", lw=2)
    p1_pup1_line,        = ax.plot([], [], "o-", lw=2)

    time_text            = ax.text(0.05, 0.95, "", transform=ax.transAxes)
    debug_text           = ax.text(0.05, 0.90, "", transform=ax.transAxes)
    time_text_template   = "t = %.1fs"
    debug_text_template  = "E = %.1fJ, T = %.1fJ, U = %.1fJ"

    def init():

        if t_nominal is not None and x_nominal is not None:
            return p_nominal_plot, p0_p1_line, p0_pup0_line, p1_pup1_line, time_text, debug_text
        else:
            return p0_p1_line, p0_pup0_line, p1_pup1_line, time_text, debug_text

    def animate(i):

        t_i = t[i]

        if t_nominal is not None and x_nominal is not None:
            x_nominal_current = x_nominal_interp_func(t_i).T.A
            p_nominal_current, theta_nominal_current, p_dot_nominal_current, theta_dot_nominal_current, q_nominal_current, q_dot_nominal_current = unpack_state_space_trajectory(x_nominal_current)
            p_nominal_plot.set_data(p_nominal_current[:,1], p_nominal_current[:,0])

        p0_p1_y_current = [p_0[i,0], (p_0[i,0]+p_1[i,0])/2.0, p_1[i,0]]
        p0_p1_x_current = [p_0[i,1], (p_0[i,1]+p_1[i,1])/2.0, p_1[i,1]]
        p0_p1_line.set_data(p0_p1_x_current, p0_p1_y_current)

        p0_pup0_y_current = [p_0[i,0], p_0_up[i,0]]
        p0_pup0_x_current = [p_0[i,1], p_0_up[i,1]]
        p0_pup0_line.set_data(p0_pup0_x_current, p0_pup0_y_current)

        p1_pup1_y_current = [p_1[i,0], p_1_up[i,0]]
        p1_pup1_x_current = [p_1[i,1], p_1_up[i,1]]
        p1_pup1_line.set_data(p1_pup1_x_current, p1_pup1_y_current)

        time_text.set_text(time_text_template % t_i)
        debug_text.set_text("")

        if t_nominal is not None and x_nominal is not None:
            return p_nominal_plot, p0_p1_line, p0_pup0_line, p1_pup1_line, time_text, debug_text
        else:
            return p0_p1_line, p0_pup0_line, p1_pup1_line, time_text, debug_text

    animation = matplotlib.animation.FuncAnimation(fig, animate, arange(x.shape[0]), interval=25, blit=True, init_func=init)
    #animation.save("quadcopter_2d_open_loop_trajectory_no_disturbance.mp4", fps=30)

    if inline:
        return animation
    else:
        plt.show()
        return None