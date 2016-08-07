from pylab import *

import os, sys, time
import pdb
import sets
import scipy.interpolate
import matplotlib.cm
import sklearn.metrics

import pathutils
pathutils.add_relative_to_current_source_file_path_to_sys_path("..")

import flashlight.transformations   as transformations
import flashlight.splineutils       as splineutils
import flashlight.curveutils        as curveutils
import flashlight.gradientutils     as gradientutils
import flashlight.interpolateutils  as interpolateutils
import flashlight.trigutils         as trigutils
import flashlight.sympyutils        as sympyutils
import flashlight.quadrotor3d       as quadrotor3d
import flashlight.quadrotorcamera3d as quadrotorcamera3d

import flashlight.trajectory_optimization.quadrotor3d_fixed_path                       as quadrotor3d_fixed_path
import flashlight.trajectory_optimization.quadrotor3d_direct_transcription_nonconst_dt as quadrotor3d_direct_transcription_nonconst_dt
import flashlight.trajectory_optimization.quadrotor3d_uniform_time_stretch             as quadrotor3d_uniform_time_stretch
import flashlight.trajectory_optimization.quadrotor3d_gaussian_time_stretch            as quadrotor3d_gaussian_time_stretch

# teaser
# x_min_ti = matrix([ -1000.0, -1000.0, -1000.0, -pi/3, -200*pi, -pi/3, -10.0, -10.0, -10.0, -2*pi, -2*pi, -2*pi ]).T
# x_max_ti = matrix([ 1000.0,  1000.0,  1000.0,   pi/3,  200*pi,  pi/3,  10.0,  10.0,  10.0,  2*pi,  2*pi,  2*pi ]).T

# u_min_ti = matrix([ 0.0, 0.0, 0.0, 0.0 ]).T
# u_max_ti = matrix([ 5.0, 5.0, 5.0, 5.0 ]).T

# infeasible trajectory dataset, also works for teaser
x_min_ti = matrix([ -1000.0, -1000.0, -1000.0, -pi/3, -200*pi, -pi/3, -10.0, -3.0, -10.0, -2*pi, -2*pi, -2*pi ]).T
x_max_ti = matrix([ 1000.0,  1000.0,  1000.0,   pi/3,  200*pi,  pi/3,  10.0,  3.0,  10.0,  2*pi,  2*pi,  2*pi ]).T

u_min_ti = matrix([ 0.0, 0.0, 0.0, 0.0 ]).T
u_max_ti = matrix([ 5.0, 5.0, 5.0, 5.0 ]).T

num_timesteps = 100

def _compute_param_spacing_L2_norm(P, alpha):
    D                   = sklearn.metrics.pairwise_distances(P,P)
    dist                = diag(D,k=1) + 0.01
    l                   = pow(dist,alpha)
    l_cum               = r_[0.0,cumsum(l)]

    T = np.tile(c_[l_cum], P.shape[1])
    return T

def _get_easing_spline_coefficients(P,T=None,S=None,Z=None,degree=9):
    return splineutils.compute_minimum_variation_nonlocal_interpolating_b_spline_coefficients(P,T=T,S=S,Z=Z,degree=degree,lamb=[0,0,0,1,0],return_derivatives=False)
    #return compute_catmull_rom_spline_coefficients(P,T=T,S=None,Z=None,degree=3)

def _evaluate_easing_spline(C,T,sd,T_eval=None,num_samples=200):
    return splineutils.evaluate_minimum_variation_nonlocal_interpolating_b_spline(C,T,sd,T_eval=T_eval,num_samples=num_samples)    
    #return evaluate_catmull_rom_spline(C,T,sd,T_eval=T_eval,num_samples=num_samples)

def _compute_easing_curve(P,T=None,num_samples=200):

    has_valid_spline = False
    C = None
    sd = None
    S = sets.Set([])
    Z = [0, -1]

    # Calculate the spline
    C,T,sd = _get_easing_spline_coefficients(P,T=T,S=list(S),Z=Z)

    Pev = None

    i = 0
    MAX_ITERS = 50
    while not has_valid_spline or i < MAX_ITERS:
        has_valid_spline = True
        # Calculate the spline
        C,T,sd = _get_easing_spline_coefficients(P,T=T,S=list(S),Z=Z)

        # then sample it
        Pev,Tev,dT = _evaluate_easing_spline(C,T,sd,num_samples=num_samples)
        
        Pev[0] = 0
        Pev[-1] = 1

        currentSection = 0
        invalidSection = False
        for i in range(len(Tev)):
            if i == len(Tev)-1 or Tev[i] > T[currentSection+1]:
                if invalidSection:
                    S = S.union(set([currentSection]))
                    if currentSection == 0:
                        try:
                            Z.remove(0)
                        except:
                            pass
                    if currentSection == len(T)-2:
                        try:
                            Z.remove(-1)
                        except:
                            pass

                currentSection = min(currentSection+1,len(T)-1)
                invalidSection = False

            if Pev[i] < -0.0000001 or Pev[i] > 1.0000001:
                invalidSection = True
                has_valid_spline = False

        i += 1
 
    assert min(Pev) > -0.0000001
    assert max(Pev) <  1.0000001

    return C,T,sd

def _get_spatial_spline_coefficients(P,T=None,S=None,degree=9,return_derivatives=False,uniformKnots=False):
    #return compute_catmull_rom_spline_coefficients(P,T=T,S=S,degree=degree)

    if T is None and not uniformKnots:
        T = _compute_param_spacing_L2_norm(P, 0.5)
        if T[-1,-1] > 0:
            T = T / T[-1,-1]
        #S = [0]
    if uniformKnots:
        T = None
    return splineutils.compute_minimum_variation_nonlocal_interpolating_b_spline_coefficients(P,T=T,S=S,degree=9,lamb=[0,0,0,1,0],return_derivatives=False)

def _evaluate_spatial_spline(C,T,sd,T_eval=None,num_samples=200):
  #return evaluate_catmull_rom_spline(C,T,sd,num_samples=num_samples)
  return splineutils.evaluate_minimum_variation_nonlocal_interpolating_b_spline(C,T,sd,T_eval=T_eval,num_samples=num_samples)

def _compute_spatial_trajectory_and_arc_distance(P,T=None,S=None,num_samples=200,inNED=True):

    C,T,sd = _get_spatial_spline_coefficients(P,T=T,S=S,degree=9,return_derivatives=False)
    
    p,T_eval,dT = _evaluate_spatial_spline(C,T,sd,num_samples=num_samples)
    
    # Turn into NED:
    if not inNED:
        p = np.array([llh2ned(point, p[0]) for point in p])
    
    if len(p.shape) == 1:
        p = matrix(p).T
    else:
        p = matrix(p)

    num_samples_p             = p.shape[0]
    num_dimensions            = p.shape[1]

    t_p_linspace = linspace(0.0,T[-1,0],num_samples_p)
    
    D                   = sklearn.metrics.pairwise_distances(p,p)
    l                   = diag(D,k=1)
    l_cum               = r_[0.0,cumsum(l)]
    l_cum_f             = scipy.interpolate.interp1d(t_p_linspace, l_cum)
    
    knot_arc_distances  = l_cum_f(T[:,0])
    
    return C,T,sd,knot_arc_distances

def _reparameterize_spline(P_spline, T_spline, P_ease, T_ease, num_samples=200, ref_llh = None, isNED=True):
    """

    This assumes the easing curve in position and time is normalized
        - P_Ease in [0,1]
        - T_Ease in [0,1]

    Input: A description of a spline, and an easing curve for time to distance (normalized).

    Calculates the (time -> distance -> spline parameter) mapping. 
    Returns the resulting table of time to spline parameter values, such that
    sweeping linearly through time will result in spline parameters that move along the spline
    according to the time->distance easing curve.

    """

    # First, calculate a spline for P_spline and P_ease
    C_spline,T_spline,sd_spline,dist = _compute_spatial_trajectory_and_arc_distance(P_spline,T=T_spline)
    C_ease,T_ease,sd_ease = _compute_easing_curve(P_ease,T=T_ease)

    # Then sample that densely
    Spline_eval,T_spline_eval,dT_spline = _evaluate_spatial_spline(C_spline,T_spline,sd_spline,num_samples=num_samples)
    Ease_eval,T_ease_eval,dT_ease = _evaluate_easing_spline(C_ease,T_ease,sd_ease,num_samples=num_samples)    
    
    if not isNED:
        if ref_llh is None:
            ref_llh = Spline_eval[0]
        # Move into NED space, where everything is in meters.
        Spline_eval = np.array([llh2ned(point, ref_llh) for point in Spline_eval])

    assert min(Ease_eval) > -0.0001
    assert max(Ease_eval) <  1.0001
    Ease_eval = Ease_eval[:,0]/Ease_eval[-1,0]
    Ease_eval = clip(Ease_eval,0,1)

    # Finally, reparameterize the spline curve first into dist then modulate with ease
    p_user_progress, t_user_progress, cumLength, t_user_progress_linspace_norm = curveutils.reparameterize_curve(Spline_eval,Ease_eval)
    
    # Then return a table of t_user_progress_linspace_norm
    return t_user_progress_linspace_norm, t_user_progress, p_user_progress, ref_llh

def _evaluate_splines_and_convert_to_meters(P_spline, T_spline, P_ease, T_ease, num_samples=200, ref_llh=None, isNED=True):

    # First, calculate a spline for P_spline and P_ease
    C_spline,T_spline,sd_spline,dist = _compute_spatial_trajectory_and_arc_distance(P_spline,T=T_spline)
    C_ease,T_ease,sd_ease = _compute_easing_curve(P_ease,T=T_ease)

    # Then sample that densely
    Spline_eval,T_spline_eval,dT_spline = _evaluate_spatial_spline(C_spline,T_spline,sd_spline,num_samples=num_samples)
    Ease_eval,T_ease_eval,dT_ease = _evaluate_easing_spline(C_ease,T_ease,sd_ease,num_samples=num_samples)    
    
    if not isNED:
        if ref_llh is None:
            ref_llh = Spline_eval[0]
        # Move into NED space, where everything is in meters.
        Spline_eval = np.array([llh2ned(point, ref_llh) for point in Spline_eval])

    assert min(Ease_eval) > -0.0001
    assert max(Ease_eval) <  1.0001
    Ease_eval = Ease_eval[:,0]/Ease_eval[-1,0]
    Ease_eval = clip(Ease_eval,0,1)

    return Spline_eval,Ease_eval

def calculate_feasibility_ned(P_lookFrom_spline, T_lookFrom_spline, P_lookAt_spline, T_lookAt_spline, P_lookFrom_ease, T_lookFrom_ease, P_lookAt_ease, T_lookAt_ease, total_time, refLLH):

    lookFrom_t_user_progress_linspace_norm, lookFrom_t_user_progress, lookFrom_p_user_progress, lookFrom_ref_llh  = _reparameterize_spline(P_lookFrom_spline, T_lookFrom_spline, P_lookFrom_ease, T_lookFrom_ease)
    lookAt_t_user_progress_linspace_norm, lookAt_t_user_progress, lookAt_p_user_progress, lookAt_ref_llh          = _reparameterize_spline(P_lookAt_spline, T_lookAt_spline, P_lookAt_ease, T_lookAt_ease)

    y_axis_cam_hint_nominal = c_[ zeros_like(lookAt_t_user_progress),  ones_like(lookAt_t_user_progress), zeros_like(lookAt_t_user_progress) ]

    #north, negative down, east
    i = np.array([0, 2, 1])
    lookFrom_p_user_progress        = lookFrom_p_user_progress[:,[0,2,1]]
    lookAt_p_user_progress          = lookAt_p_user_progress[:,[0,2,1]]
    lookFrom_p_user_progress[:, 1] *= -1
    lookAt_p_user_progress[:, 1]   *= -1

    dt = lookAt_t_user_progress_linspace_norm[1] * total_time;

    q_q_dot_q_dot_dot_nominal = quadrotorcamera3d.compute_state_space_trajectory_and_derivatives(lookFrom_p_user_progress,lookAt_p_user_progress,y_axis_cam_hint_nominal,dt)
    u_nominal                 = quadrotorcamera3d.compute_control_trajectory(q_q_dot_q_dot_dot_nominal)

    p_body_nominal, p_body_dot_nominal, p_body_dot_dot_nominal, theta_body_nominal, theta_body_dot_nominal, theta_body_dot_dot_nominal, psi_body_nominal, psi_body_dot_nominal, psi_body_dot_dot_nominal, phi_body_nominal, phi_body_dot_nominal, phi_body_dot_dot_nominal, theta_cam_nominal, theta_cam_dot_nominal, theta_cam_dot_dot_nominal, psi_cam_nominal, psi_cam_dot_nominal, psi_cam_dot_dot_nominal, phi_cam_nominal, phi_cam_dot_nominal, phi_cam_dot_dot_nominal  = q_q_dot_q_dot_dot_nominal

    q_body_nominal     = c_[ p_body_nominal,     theta_body_nominal,     psi_body_nominal,     phi_body_nominal ]
    q_body_dot_nominal = c_[ p_body_dot_nominal, theta_body_dot_nominal, psi_body_dot_nominal, phi_body_dot_nominal ]
    x_body_nominal     = c_[ q_body_nominal, q_body_dot_nominal ]
    u_body_nominal     = u_nominal[:,0:4]

    constraint_violation_score_norm_n1p1,constraint_violation_score_norm_01,constraint_violation_colors = \
        quadrotor3d.compute_normalized_constraint_violation_score(x_body_nominal,u_body_nominal,x_min_ti,x_max_ti,u_min_ti,u_max_ti)

    #
    # save_results
    #
    np.savez("tmp_calculate_feasibility_ned_result", \
             x_body_nominal=x_body_nominal, \
             u_body_nominal=u_body_nominal, \
             constraint_violation_score_norm_n1p1=constraint_violation_score_norm_n1p1, \
             constraint_violation_score_norm_01=constraint_violation_score_norm_01, \
             constraint_violation_colors=constraint_violation_colors)

    return u_nominal, q_q_dot_q_dot_dot_nominal, constraint_violation_colors

def calculate_optimized_easing_curve_ned(P_lookFrom_spline, T_lookFrom_spline, P_lookAt_spline, T_lookAt_spline, P_lookFrom_ease, T_lookFrom_ease, P_lookAt_ease, T_lookAt_ease, total_time, refLLH):

    t_begin   = 0.0
    t_end     = total_time
    t_nominal = linspace(t_begin,t_end,num_timesteps)
    dt        = (t_end-t_begin) / num_timesteps

    look_from_eval,look_from_easing_eval = _evaluate_splines_and_convert_to_meters(P_lookFrom_spline, T_lookFrom_spline, P_lookFrom_ease, T_lookFrom_ease, num_samples=num_timesteps)
    look_at_eval,  look_at_easing_eval   = _evaluate_splines_and_convert_to_meters(P_lookAt_spline,   T_lookAt_spline,   P_lookAt_ease,   T_lookAt_ease,   num_samples=num_timesteps)

    #
    # adjust to match flashlight library convention of z,y,x axis ordering where y is up
    #
    look_from_eval        = look_from_eval[:,[0,2,1]]
    look_at_eval          = look_at_eval[:,[0,2,1]]
    look_from_eval[:, 1] *= -1
    look_at_eval[:, 1]   *= -1

    #
    # old strategy: use reparameterized curve to compute yaw values
    #
    # look_from_user_progress, look_from_v_norm_user_progress, _, _ = reparameterize_curve(look_from_eval,look_from_easing_eval)
    # look_at_user_progress,   look_at_v_norm_user_progress, _, _   = reparameterize_curve(look_at_eval,look_at_easing_eval)

    # y_axis_cam_hint_nominal   = c_[ zeros_like(t_nominal),  ones_like(t_nominal), zeros_like(t_nominal) ]
    # q_q_dot_q_dot_dot_nominal = compute_state_space_trajectory_and_derivatives(look_from_user_progress,look_at_user_progress,y_axis_cam_hint_nominal,dt)

    # p_body_nominal, p_body_dot_nominal, p_body_dot_dot_nominal, theta_body_nominal, theta_body_dot_nominal, theta_body_dot_dot_nominal, psi_body_nominal, psi_body_dot_nominal, psi_body_dot_dot_nominal, phi_body_nominal, phi_body_dot_nominal, phi_body_dot_dot_nominal, theta_cam_nominal, theta_cam_dot_nominal, theta_cam_dot_dot_nominal, psi_cam_nominal, psi_cam_dot_nominal, psi_cam_dot_dot_nominal, phi_cam_nominal, phi_cam_dot_nominal, phi_cam_dot_dot_nominal  = q_q_dot_q_dot_dot_nominal

    # assert allclose(look_from_user_progress,p_body_nominal)

    # p_body   = p_body_nominal
    # psi_body = psi_body_nominal

    #
    # new strategy: figure out the look-at values that correspond to the original sampling of look-from values
    #
    look_from_original = look_from_eval
    v_norm_eval        = linspace(0.0,1.0,num_timesteps)

    look_from_user_progress, look_from_v_norm_user_progress, _, _ = curveutils.reparameterize_curve(look_from_eval,look_from_easing_eval)
    look_at_user_progress,   look_at_v_norm_user_progress, _, _   = curveutils.reparameterize_curve(look_at_eval,look_at_easing_eval)

    t_look_from_original    = interpolateutils.resample_scalar_wrt_scalar(look_from_v_norm_user_progress,t_nominal,v_norm_eval)
    look_at_v_norm_original = interpolateutils.resample_scalar_wrt_scalar(t_nominal,look_at_v_norm_user_progress,t_look_from_original)

    look_at_original = interpolateutils.resample_vector_wrt_scalar(v_norm_eval,look_at_eval,look_at_v_norm_original)

    #
    # use quadrotor camera algorithm to compute psi values
    #
    p_body    = look_from_original
    p_look_at = look_at_original

    p_body_dotN    = gradientutils.gradients_vector_wrt_scalar_smooth_boundaries_nonconst_dt(p_body,t_look_from_original,max_gradient=2,poly_deg=5)
    p_body_dot     = p_body_dotN[1]
    p_body_dot_dot = p_body_dotN[2]

    f_thrust_world            = quadrotor3d.m*p_body_dot_dot - quadrotor3d.f_external_world.T.A
    f_thrust_world_normalized = sklearn.preprocessing.normalize(f_thrust_world)

    y_axis_body = f_thrust_world_normalized
    z_axis_body = sklearn.preprocessing.normalize(cross(y_axis_body, p_look_at - p_body))
    x_axis_body = sklearn.preprocessing.normalize(cross(z_axis_body, y_axis_body))

    psi_body = zeros(num_timesteps)

    for ti in range(num_timesteps):

        z_axis_body_ti = c_[matrix(z_axis_body[ti]),0].T
        y_axis_body_ti = c_[matrix(y_axis_body[ti]),0].T
        x_axis_body_ti = c_[matrix(x_axis_body[ti]),0].T

        R_world_from_body_ti                  = c_[z_axis_body_ti,y_axis_body_ti,x_axis_body_ti,[0,0,0,1]]
        psi_body_ti,theta_body_ti,phi_body_ti = transformations.euler_from_matrix(R_world_from_body_ti,"ryxz")

        assert allclose(R_world_from_body_ti, transformations.euler_matrix(psi_body_ti,theta_body_ti,phi_body_ti,"ryxz"))

        psi_body[ti] = psi_body_ti

    psi_body = trigutils.compute_continuous_angle_array(psi_body)

    #
    # call the fixed path optimizer
    #
    p_eval        = p_body
    psi_eval      = psi_body
    user_progress = look_from_easing_eval

    const_vals_ti = hstack( [ quadrotor3d.alpha, quadrotor3d.beta, quadrotor3d.gamma, quadrotor3d.d, quadrotor3d.m, quadrotor3d.I.A1, quadrotor3d.f_external_world.A1 ] )

    opt_problem_type = "track"

    fixed_path_optimized_trajectory = quadrotor3d_fixed_path.optimize( p_eval,psi_eval,            \
                                                                       t_nominal,user_progress,dt, \
                                                                       const_vals_ti,              \
                                                                       x_min_ti,         x_max_ti, \
                                                                       u_min_ti,         u_max_ti, \
                                                                       opt_problem_type )

    s_opt,Beta_opt,Gamma_opt,X_opt,U_opt,t_opt,T_final_opt,solver_time,obj_vals = fixed_path_optimized_trajectory

    #
    # compute look-at position and easing curve
    #
    s_lookat_opt = look_at_easing_eval
    p_lookat_opt = look_at_user_progress

    #
    # color values
    #
    constraint_violation_score_norm_n1p1, constraint_violation_score_norm_01, constraint_violation_colors = \
        quadrotor3d.compute_normalized_constraint_violation_score(X_opt,U_opt,x_min_ti,x_max_ti,u_min_ti,u_max_ti)

    #
    # save_results
    #
    np.savez("tmp_calculate_optimized_easing_curve_ned_result", \
             p_eval=p_eval, \
             psi_eval=psi_eval, \
             t_nominal=t_nominal, \
             user_progress=user_progress, \
             dt=array(dt), \
             const_vals_ti=const_vals_ti, \
             x_min_ti=x_min_ti, \
             x_max_ti=x_max_ti, \
             u_min_ti=u_min_ti, \
             u_max_ti=u_max_ti, \
             opt_problem_type=array(opt_problem_type), \
             s_opt=s_opt, \
             Beta_opt=Beta_opt, \
             Gamma_opt=Gamma_opt, \
             X_opt=X_opt, \
             U_opt=U_opt, \
             t_opt=t_opt, \
             T_final_opt=T_final_opt, \
             solver_time=solver_time, \
             obj_vals=obj_vals, \
             constraint_violation_colors=constraint_violation_colors)

    return s_opt,X_opt,U_opt,t_opt,s_lookat_opt,p_lookat_opt,constraint_violation_colors

def calculate_optimized_trajectory_ned(P_lookFrom_spline, T_lookFrom_spline, P_lookAt_spline, T_lookAt_spline, P_lookFrom_ease, T_lookFrom_ease, P_lookAt_ease, T_lookAt_ease, total_time, refLLH):

    t_begin   = 0.0
    t_end     = total_time
    t_nominal = linspace(t_begin,t_end,num_timesteps)
    dt        = (t_end-t_begin) / num_timesteps

    look_from_eval,look_from_easing_eval = _evaluate_splines_and_convert_to_meters(P_lookFrom_spline, T_lookFrom_spline, P_lookFrom_ease, T_lookFrom_ease, num_samples=num_timesteps)
    look_at_eval,  look_at_easing_eval   = _evaluate_splines_and_convert_to_meters(P_lookAt_spline,   T_lookAt_spline,   P_lookAt_ease,   T_lookAt_ease,   num_samples=num_timesteps)

    #
    # adjust to match flashlight library convention of z,y,x axis ordering where y is up
    #
    look_from_eval        = look_from_eval[:,[0,2,1]]
    look_at_eval          = look_at_eval[:,[0,2,1]]
    look_from_eval[:, 1] *= -1
    look_at_eval[:, 1]   *= -1

    #
    # call the trajectory optimizer
    #
    p_eval        = look_from_eval
    psi_eval      = zeros(num_timesteps)
    user_progress = look_from_easing_eval

    const_vals_ti = hstack( [ quadrotor3d.alpha, quadrotor3d.beta, quadrotor3d.gamma, quadrotor3d.d, quadrotor3d.m, quadrotor3d.I.A1, quadrotor3d.f_external_world.A1 ] )

    direct_transcription_nonconst_dt_optimized_trajectory = quadrotor3d_direct_transcription_nonconst_dt.optimize(p_eval,psi_eval,            \
                                                                                                                  t_nominal,user_progress,dt, \
                                                                                                                  const_vals_ti,              \
                                                                                                                  x_min_ti,x_max_ti,          \
                                                                                                                  u_min_ti,u_max_ti)

    X_opt,U_opt,DT_opt,t_opt,T_final_opt,solver_time,obj_vals = direct_transcription_nonconst_dt_optimized_trajectory

    #
    # color values
    #
    constraint_violation_score_norm_n1p1, constraint_violation_score_norm_01, constraint_violation_colors = \
        quadrotor3d.compute_normalized_constraint_violation_score(X_opt,U_opt,x_min_ti,x_max_ti,u_min_ti,u_max_ti)

    #
    # save_results
    #
    np.savez("tmp_calculate_optimized_trajectory_ned_result", \
             p_eval=p_eval, \
             psi_eval=psi_eval, \
             t_nominal=t_nominal, \
             user_progress=user_progress, \
             dt=array(dt), \
             const_vals_ti=const_vals_ti, \
             x_min_ti=x_min_ti, \
             x_max_ti=x_max_ti, \
             u_min_ti=u_min_ti, \
             u_max_ti=u_max_ti, \
             X_opt=X_opt, \
             U_opt=U_opt, \
             DT_opt=DT_opt, \
             t_opt=t_opt, \
             T_final_opt=T_final_opt, \
             solver_time=solver_time, \
             obj_vals=obj_vals, \
             constraint_violation_colors=constraint_violation_colors)

    return X_opt,U_opt,DT_opt,t_opt,constraint_violation_colors

def calculate_unoptimized_trajectory_ned(P_lookFrom_spline, T_lookFrom_spline, P_lookAt_spline, T_lookAt_spline, P_lookFrom_ease, T_lookFrom_ease, P_lookAt_ease, T_lookAt_ease, total_time, refLLH):

    t_begin   = 0.0
    t_end     = total_time
    t_nominal = linspace(t_begin,t_end,num_timesteps)
    dt        = (t_end-t_begin) / num_timesteps

    look_from_eval,look_from_easing_eval = _evaluate_splines_and_convert_to_meters(P_lookFrom_spline, T_lookFrom_spline, P_lookFrom_ease, T_lookFrom_ease, num_samples=num_timesteps)
    look_at_eval,  look_at_easing_eval   = _evaluate_splines_and_convert_to_meters(P_lookAt_spline,   T_lookAt_spline,   P_lookAt_ease,   T_lookAt_ease,   num_samples=num_timesteps)

    #
    # adjust to match flashlight library convention of z,y,x axis ordering where y is up
    #
    look_from_eval        = look_from_eval[:,[0,2,1]]
    look_at_eval          = look_at_eval[:,[0,2,1]]
    look_from_eval[:, 1] *= -1
    look_at_eval[:, 1]   *= -1

    look_from_user_progress, _, _, _ = curveutils.reparameterize_curve(look_from_eval,look_from_easing_eval)
    look_at_user_progress,   _, _, _ = curveutils.reparameterize_curve(look_at_eval,look_at_easing_eval)

    #
    # use quadrotor camera algorithm to compute nominal trajectory
    #
    p_body    = look_from_user_progress
    p_look_at = look_at_user_progress

    p_body_dotN    = gradientutils.gradients_vector_wrt_scalar_smooth_boundaries(p_body,dt,max_gradient=2,poly_deg=5)
    p_body_dot     = p_body_dotN[1]
    p_body_dot_dot = p_body_dotN[2]

    f_thrust_world            = quadrotor3d.m*p_body_dot_dot - quadrotor3d.f_external_world.T.A
    f_thrust_world_normalized = sklearn.preprocessing.normalize(f_thrust_world)

    y_axis_body = f_thrust_world_normalized
    z_axis_body = sklearn.preprocessing.normalize(cross(y_axis_body, p_look_at - p_body))
    x_axis_body = sklearn.preprocessing.normalize(cross(z_axis_body, y_axis_body))

    theta_body = zeros(num_timesteps)
    psi_body   = zeros(num_timesteps)
    phi_body   = zeros(num_timesteps)

    for ti in range(num_timesteps):

        z_axis_body_ti = c_[matrix(z_axis_body[ti]),0].T
        y_axis_body_ti = c_[matrix(y_axis_body[ti]),0].T
        x_axis_body_ti = c_[matrix(x_axis_body[ti]),0].T

        R_world_from_body_ti                  = c_[z_axis_body_ti,y_axis_body_ti,x_axis_body_ti,[0,0,0,1]]
        psi_body_ti,theta_body_ti,phi_body_ti = transformations.euler_from_matrix(R_world_from_body_ti,"ryxz")

        assert allclose(R_world_from_body_ti, transformations.euler_matrix(psi_body_ti,theta_body_ti,phi_body_ti,"ryxz"))

        theta_body[ti] = theta_body_ti
        psi_body[ti]   = psi_body_ti
        phi_body[ti]   = phi_body_ti

    psi_body   = trigutils.compute_continuous_angle_array(psi_body)
    theta_body = trigutils.compute_continuous_angle_array(theta_body)
    phi_body   = trigutils.compute_continuous_angle_array(phi_body)

    theta_body_dotN = gradientutils.gradients_scalar_wrt_scalar_smooth_boundaries(theta_body,dt,max_gradient=2,poly_deg=5)
    psi_body_dotN   = gradientutils.gradients_scalar_wrt_scalar_smooth_boundaries(psi_body,dt,max_gradient=2,poly_deg=5)
    phi_body_dotN   = gradientutils.gradients_scalar_wrt_scalar_smooth_boundaries(phi_body,dt,max_gradient=2,poly_deg=5)

    theta_body_dot     = theta_body_dotN[1]
    psi_body_dot       = psi_body_dotN[1]
    phi_body_dot       = phi_body_dotN[1]

    theta_body_dot_dot = theta_body_dotN[2]
    psi_body_dot_dot   = psi_body_dotN[2]
    phi_body_dot_dot   = phi_body_dotN[2]

    q_q_dot_q_dot_dot_nominal = p_body, p_body_dot, p_body_dot_dot, theta_body, theta_body_dot, theta_body_dot_dot, psi_body, psi_body_dot, psi_body_dot_dot, phi_body, phi_body_dot, phi_body_dot_dot
    u_nominal                 = quadrotor3d.compute_control_trajectory(q_q_dot_q_dot_dot_nominal)

    p_nominal, p_dot_nominal, p_dot_dot_nominal, theta_nominal, theta_dot_nominal, theta_dot_dot_nominal, psi_nominal, psi_dot_nominal, psi_dot_dot_nominal, phi_nominal, phi_dot_nominal, phi_dot_dot_nominal = q_q_dot_q_dot_dot_nominal

    q_nominal    = c_[ p_nominal,     theta_nominal,     psi_nominal,     phi_nominal ]
    qdot_nominal = c_[ p_dot_nominal, theta_dot_nominal, psi_dot_nominal, phi_dot_nominal ]
    x_nominal    = c_[ q_nominal, qdot_nominal ]

    dt_nominal      = dt
    T_final_nominal = t_nominal[-1]

    #
    # color values
    #
    constraint_violation_score_norm_n1p1, constraint_violation_score_norm_01, constraint_violation_colors = \
        quadrotor3d.compute_normalized_constraint_violation_score(x_nominal,u_nominal,x_min_ti,x_max_ti,u_min_ti,u_max_ti)

    #
    # save_results
    #
    np.savez("tmp_calculate_unoptimized_trajectory_ned_result", \
             p_nominal=p_nominal, \
             psi_nominal=psi_nominal, \
             user_progress=look_from_easing_eval, \
             dt=array(dt), \
             x_nominal=x_nominal, \
             u_nominal=u_nominal, \
             dt_nominal=dt, \
             t_nominal=t_nominal, \
             T_final_nominal=T_final_nominal,
             constraint_violation_colors=constraint_violation_colors)

    return x_nominal,u_nominal,dt_nominal,t_nominal

def calculate_optimized_easing_curve_uniform_ned(P_lookFrom_spline, T_lookFrom_spline, P_lookAt_spline, T_lookAt_spline, P_lookFrom_ease, T_lookFrom_ease, P_lookAt_ease, T_lookAt_ease, total_time, refLLH):

    t_begin   = 0.0
    t_end     = total_time
    t_nominal = linspace(t_begin,t_end,num_timesteps)
    dt        = (t_end-t_begin) / num_timesteps

    look_from_eval,look_from_easing_eval = _evaluate_splines_and_convert_to_meters(P_lookFrom_spline, T_lookFrom_spline, P_lookFrom_ease, T_lookFrom_ease, num_samples=num_timesteps)
    look_at_eval,  look_at_easing_eval   = _evaluate_splines_and_convert_to_meters(P_lookAt_spline,   T_lookAt_spline,   P_lookAt_ease,   T_lookAt_ease,   num_samples=num_timesteps)

    #
    # adjust to match flashlight library convention of z,y,x axis ordering where y is up
    #
    look_from_eval        = look_from_eval[:,[0,2,1]]
    look_at_eval          = look_at_eval[:,[0,2,1]]
    look_from_eval[:, 1] *= -1
    look_at_eval[:, 1]   *= -1

    #
    # new strategy: figure out the look-at values that correspond to the original sampling of look-from values
    #
    look_from_original      = look_from_eval
    v_norm_eval             = linspace(0.0,1.0,num_timesteps)

    look_from_user_progress, look_from_v_norm_user_progress, _, _ = curveutils.reparameterize_curve(look_from_eval,look_from_easing_eval)
    look_at_user_progress,   look_at_v_norm_user_progress, _, _   = curveutils.reparameterize_curve(look_at_eval,look_at_easing_eval)

    t_look_from_original    = interpolateutils.resample_scalar_wrt_scalar(look_from_v_norm_user_progress,t_nominal,v_norm_eval)
    look_at_v_norm_original = interpolateutils.resample_scalar_wrt_scalar(t_nominal,look_at_v_norm_user_progress,t_look_from_original)

    look_at_original = interpolateutils.resample_vector_wrt_scalar(v_norm_eval,look_at_eval,look_at_v_norm_original)

    #
    # use quadrotor camera algorithm to compute psi values
    #
    p_body    = look_from_original
    p_look_at = look_at_original

    p_body_dotN    = gradientutils.gradients_vector_wrt_scalar_smooth_boundaries_nonconst_dt(p_body,t_look_from_original,max_gradient=2,poly_deg=5)
    p_body_dot     = p_body_dotN[1]
    p_body_dot_dot = p_body_dotN[2]

    f_thrust_world            = quadrotor3d.m*p_body_dot_dot - quadrotor3d.f_external_world.T.A
    f_thrust_world_normalized = sklearn.preprocessing.normalize(f_thrust_world)

    y_axis_body = f_thrust_world_normalized
    z_axis_body = sklearn.preprocessing.normalize(cross(y_axis_body, p_look_at - p_body))
    x_axis_body = sklearn.preprocessing.normalize(cross(z_axis_body, y_axis_body))

    psi_body = zeros(num_timesteps)

    for ti in range(num_timesteps):

        z_axis_body_ti = c_[matrix(z_axis_body[ti]),0].T
        y_axis_body_ti = c_[matrix(y_axis_body[ti]),0].T
        x_axis_body_ti = c_[matrix(x_axis_body[ti]),0].T

        R_world_from_body_ti                  = c_[z_axis_body_ti,y_axis_body_ti,x_axis_body_ti,[0,0,0,1]]
        psi_body_ti,theta_body_ti,phi_body_ti = transformations.euler_from_matrix(R_world_from_body_ti,"ryxz")

        assert allclose(R_world_from_body_ti, transformations.euler_matrix(psi_body_ti,theta_body_ti,phi_body_ti,"ryxz"))

        psi_body[ti] = psi_body_ti

    psi_body = trigutils.compute_continuous_angle_array(psi_body)

    #
    # compute nominal trajectory
    #
    p_eval        = p_body
    psi_eval      = psi_body
    user_progress = look_from_easing_eval
    dt_nominal    = dt

    max_bin_search_iters_feasible = 10
    dt_upper_init_feasible        = 4.0

    p_nominal, _, _, _   = curveutils.reparameterize_curve( p_eval, user_progress )
    psi_nominal, _, _, _ = curveutils.reparameterize_curve( psi_eval, user_progress )

    #
    # uniform time stretch
    #
    uniform_time_stretch_optimized_trajectory = quadrotor3d_uniform_time_stretch.optimize_feasible( p_nominal,psi_nominal,dt_nominal, \
                                                                                                    x_min_ti,x_max_ti,                \
                                                                                                    u_min_ti,u_max_ti,                \
                                                                                                    max_bin_search_iters_feasible,    \
                                                                                                    dt_upper_init_feasible )

    s_opt                    = user_progress
    X_opt,U_opt,dt_scale_opt = uniform_time_stretch_optimized_trajectory
    t_opt                    = t_nominal*dt_scale_opt
    T_final_opt              = t_opt[-1]
    dt_opt                   = dt_nominal*dt_scale_opt

    #
    # color values
    #
    constraint_violation_score_norm_n1p1, constraint_violation_score_norm_01, constraint_violation_colors = \
        quadrotor3d.compute_normalized_constraint_violation_score(X_opt,U_opt,x_min_ti,x_max_ti,u_min_ti,u_max_ti)

    #
    # save_results
    #
    np.savez("tmp_calculate_optimized_easing_curve_uniform_ned_result", \
             p_eval=p_eval, \
             psi_eval=psi_eval, \
             user_progress=user_progress, \
             p_nominal=p_nominal, \
             psi_nominal=psi_nominal, \
             dt_nominal=dt_nominal, \
             x_min_ti=x_min_ti, \
             x_max_ti=x_max_ti, \
             u_min_ti=u_min_ti, \
             u_max_ti=u_max_ti, \
             max_bin_search_iters_feasible=max_bin_search_iters_feasible, \
             dt_upper_init_feasible=dt_upper_init_feasible, \
             s_opt=s_opt, \
             X_opt=X_opt, \
             U_opt=U_opt, \
             t_opt=t_opt, \
             T_final_opt=T_final_opt, \
             constraint_violation_colors=constraint_violation_colors)

    return s_opt,X_opt,U_opt,t_opt,constraint_violation_colors



def calculate_optimized_easing_curve_gaussian_ned(P_lookFrom_spline, T_lookFrom_spline, P_lookAt_spline, T_lookAt_spline, P_lookFrom_ease, T_lookFrom_ease, P_lookAt_ease, T_lookAt_ease, total_time, refLLH):

    t_begin   = 0.0
    t_end     = total_time
    t_nominal = linspace(t_begin,t_end,num_timesteps)
    dt        = (t_end-t_begin) / num_timesteps

    look_from_eval,look_from_easing_eval = _evaluate_splines_and_convert_to_meters(P_lookFrom_spline, T_lookFrom_spline, P_lookFrom_ease, T_lookFrom_ease, num_samples=num_timesteps)
    look_at_eval,  look_at_easing_eval   = _evaluate_splines_and_convert_to_meters(P_lookAt_spline,   T_lookAt_spline,   P_lookAt_ease,   T_lookAt_ease,   num_samples=num_timesteps)

    #
    # adjust to match flashlight library convention of z,y,x axis ordering where y is up
    #
    look_from_eval        = look_from_eval[:,[0,2,1]]
    look_at_eval          = look_at_eval[:,[0,2,1]]
    look_from_eval[:, 1] *= -1
    look_at_eval[:, 1]   *= -1

    #
    # new strategy: figure out the look-at values that correspond to the original sampling of look-from values
    #
    look_from_original      = look_from_eval
    v_norm_eval             = linspace(0.0,1.0,num_timesteps)

    look_from_user_progress, look_from_v_norm_user_progress, _, _ = curveutils.reparameterize_curve(look_from_eval,look_from_easing_eval)
    look_at_user_progress,   look_at_v_norm_user_progress, _, _   = curveutils.reparameterize_curve(look_at_eval,look_at_easing_eval)

    t_look_from_original    = interpolateutils.resample_scalar_wrt_scalar(look_from_v_norm_user_progress,t_nominal,v_norm_eval)
    look_at_v_norm_original = interpolateutils.resample_scalar_wrt_scalar(t_nominal,look_at_v_norm_user_progress,t_look_from_original)

    look_at_original = interpolateutils.resample_vector_wrt_scalar(v_norm_eval,look_at_eval,look_at_v_norm_original)

    #
    # use quadrotor camera algorithm to compute psi values
    #
    p_body    = look_from_original
    p_look_at = look_at_original

    p_body_dotN    = gradientutils.gradients_vector_wrt_scalar_smooth_boundaries_nonconst_dt(p_body,t_look_from_original,max_gradient=2,poly_deg=5)
    p_body_dot     = p_body_dotN[1]
    p_body_dot_dot = p_body_dotN[2]

    f_thrust_world            = quadrotor3d.m*p_body_dot_dot - quadrotor3d.f_external_world.T.A
    f_thrust_world_normalized = sklearn.preprocessing.normalize(f_thrust_world)

    y_axis_body = f_thrust_world_normalized
    z_axis_body = sklearn.preprocessing.normalize(cross(y_axis_body, p_look_at - p_body))
    x_axis_body = sklearn.preprocessing.normalize(cross(z_axis_body, y_axis_body))

    psi_body = zeros(num_timesteps)

    for ti in range(num_timesteps):

        z_axis_body_ti = c_[matrix(z_axis_body[ti]),0].T
        y_axis_body_ti = c_[matrix(y_axis_body[ti]),0].T
        x_axis_body_ti = c_[matrix(x_axis_body[ti]),0].T

        R_world_from_body_ti                  = c_[z_axis_body_ti,y_axis_body_ti,x_axis_body_ti,[0,0,0,1]]
        psi_body_ti,theta_body_ti,phi_body_ti = transformations.euler_from_matrix(R_world_from_body_ti,"ryxz")

        assert allclose(R_world_from_body_ti, transformations.euler_matrix(psi_body_ti,theta_body_ti,phi_body_ti,"ryxz"))

        psi_body[ti] = psi_body_ti

    psi_body = trigutils.compute_continuous_angle_array(psi_body)

    #
    # call the gaussian time stretching optimizer
    #
    p_eval        = p_body
    psi_eval      = psi_body
    user_progress = look_from_easing_eval

    const_vals_ti = hstack( [ quadrotor3d.alpha, quadrotor3d.beta, quadrotor3d.gamma, quadrotor3d.d, quadrotor3d.m, quadrotor3d.I.A1, quadrotor3d.f_external_world.A1 ] )

    max_stretch_iters_feasible          = 100
    gauss_width_in_terms_of_dt_feasible = 200.0
    gauss_max_in_terms_of_dt_feasible   = 0.2
    extra_iters_feasible                = 1

    #
    # gaussian time stretch
    #
    gaussian_time_stretch_optimized_trajectory = quadrotor3d_gaussian_time_stretch.optimize_feasible( p_eval,psi_eval,                     \
                                                                                                      t_nominal,user_progress,dt,          \
                                                                                                      x_min_ti,         x_max_ti,          \
                                                                                                      u_min_ti,         u_max_ti,          \
                                                                                                      max_stretch_iters_feasible,          \
                                                                                                      gauss_width_in_terms_of_dt_feasible, \
                                                                                                      gauss_max_in_terms_of_dt_feasible,   \
                                                                                                      extra_iters_feasible )

    X_opt,U_opt,t_opt,user_progress_opt,dt_opt = gaussian_time_stretch_optimized_trajectory

    s_opt       = user_progress_opt
    T_final_opt = t_opt[-1]

    #
    # color values
    #
    constraint_violation_score_norm_n1p1, constraint_violation_score_norm_01, constraint_violation_colors = \
        quadrotor3d.compute_normalized_constraint_violation_score(X_opt,U_opt,x_min_ti,x_max_ti,u_min_ti,u_max_ti)

    #
    # save_results
    #
    np.savez("tmp_calculate_optimized_easing_curve_gaussian_ned_result", \
             p_eval=p_eval, \
             psi_eval=psi_eval, \
             t_nominal=t_nominal, \
             user_progress=user_progress, \
             dt=array(dt), \
             const_vals_ti=const_vals_ti, \
             x_min_ti=x_min_ti, \
             x_max_ti=x_max_ti, \
             u_min_ti=u_min_ti, \
             u_max_ti=u_max_ti, \
             s_opt=s_opt, \
             X_opt=X_opt, \
             U_opt=U_opt, \
             t_opt=t_opt, \
             T_final_opt=T_final_opt, \
             constraint_violation_colors=constraint_violation_colors)

    return s_opt,X_opt,U_opt,t_opt,constraint_violation_colors
