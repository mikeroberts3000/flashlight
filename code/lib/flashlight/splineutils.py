from pylab import *

import scipy.linalg
import cvxopt
import sklearn.metrics



def compute_catmull_rom_spline_coefficients(P,T=None,S=None,Z=None,degree=3):

    """

Computes Catmull Rom spline coefficients for the knot sequences in T, the
points in P. S is a list of straight line segments, Z is a list of keyframes
for which the derivative must be zero, and degree must be in the set [3,5,7,9].

Returns C an array of coefficients for each segment, T an array of knot
sequences, and sd a spline description object.

    """

    P = P.astype(float64)

    if T is None:
        t_sampled = arange(P.shape[0]).reshape(-1,1)
        T         = tile(t_sampled,(1,P.shape[1]))

    T  = T.astype(float64)
    sd = _compute_catmull_rom_spline_desc(P,T,S,Z,degree)
    C  = zeros((sd["num_unknowns_per_segment"]*sd["num_segments"],sd["num_dimensions"]))

    for di in range(sd["num_dimensions"]):
        C[:,di] = _compute_catmull_rom_spline_coefficients_scalar(T[:,di],P[:,di],sd)

    return C,T,sd



def compute_minimum_variation_catmull_rom_spline_coefficients(P,T=None,S=None,Z=None,degree=5,lamb=[1,0,0,0,0,0]):

    P = P.astype(float64)

    if T is None:
        t_sampled = arange(P.shape[0]).reshape(-1,1)
        T         = tile(t_sampled,(1,P.shape[1]))

    T  = T.astype(float64)
    sd = _compute_minimum_variation_catmull_rom_spline_desc(P,T,S,Z,degree,lamb)
    C  = zeros((sd["num_unknowns_per_segment"]*sd["num_segments"],sd["num_dimensions"]))

    for di in range(sd["num_dimensions"]):
        C[:,di] = _compute_minimum_variation_catmull_rom_spline_coefficients_scalar(T[:,di],P[:,di],sd)

    return C,T,sd



def compute_nonlocal_interpolating_b_spline_coefficients(P,T=None,S=None,Z=None,degree=3):

    P = P.astype(float64)

    if T is None:
        t_sampled = arange(P.shape[0]).reshape(-1,1)
        T         = tile(t_sampled,(1,P.shape[1]))

    T  = T.astype(float64)
    sd = _compute_nonlocal_interpolating_b_spline_desc(P,T,S,Z,degree)
    C  = zeros((sd["num_unknowns_per_segment"]*sd["num_segments"] + sd["num_unknowns_per_knot"]*sd["num_knots"],sd["num_dimensions"]))

    for di in range(sd["num_dimensions"]):
        C[:,di] = _compute_nonlocal_interpolating_b_spline_coefficients_scalar(T[:,di],P[:,di],sd)

    return C,T,sd



def compute_minimum_variation_nonlocal_interpolating_b_spline_coefficients(P,T=None,S=None,Z=None,degree=5,continuity=4,lamb=[1,0,0,0,0,0],return_derivatives=True):

    P = P.astype(float64)

    if T is None:
        t_sampled = arange(P.shape[0]).reshape(-1,1)
        T         = tile(t_sampled,(1,P.shape[1]))

    T  = T.astype(float64)
    sd = _compute_minimum_variation_nonlocal_interpolating_b_spline_desc(P,T,S,Z,degree,continuity,lamb)
    C  = zeros((sd["num_unknowns_per_segment"]*sd["num_segments"] + sd["num_unknowns_per_knot"]*sd["num_knots"],sd["num_dimensions"]))

    for di in range(sd["num_dimensions"]):
        C[:,di] = _compute_minimum_variation_nonlocal_interpolating_b_spline_coefficients_scalar(T[:,di],P[:,di],sd)

    #
    # added in case client code is using the length of the returned array
    # to determine the degree of the system
    #
    if not return_derivatives:
        C = C[0:sd["num_unknowns_per_segment"]*sd["num_segments"],:]

    return C,T,sd



def compute_trigonometric_spline_coefficients(P,T=None,S=None,Z=None,order=2):

    P = P.astype(float64)

    if T is None:
        t_sampled = arange(P.shape[0]).reshape(-1,1) * (pi/4)
        T         = tile(t_sampled,(1,P.shape[1]))

    T  = T.astype(float64)
    sd = _compute_trigonometric_spline_desc(P,T,S,Z,order)
    C  = zeros((sd["num_unknowns_per_segment"]*sd["num_segments"],sd["num_dimensions"]))

    for di in range(sd["num_dimensions"]):
        C[:,di] = _compute_trigonometric_spline_coefficients_scalar(T[:,di],P[:,di],sd)

    return C,T,sd



def evaluate_catmull_rom_spline(C,T,sd,T_eval=None,num_samples=None):

    """

Evaluates Catmull Rom spline at the parameter values stored in T_eval
using the coefficients stored in C and the knot values stored in T.

Returns the positions of the spline evaluated at each parameter value.

    """

    return _evaluate_spline(C,T,sd,_evaluate_polynomial_spline_scalar,T_eval,num_samples)



def evaluate_minimum_variation_catmull_rom_spline(C,T,sd,T_eval=None,num_samples=None):
    return _evaluate_spline(C,T,sd,_evaluate_polynomial_spline_scalar,T_eval,num_samples)



def evaluate_nonlocal_interpolating_b_spline(C,T,sd,T_eval=None,num_samples=None):
    return _evaluate_spline(C,T,sd,_evaluate_polynomial_spline_scalar,T_eval,num_samples)



def evaluate_minimum_variation_nonlocal_interpolating_b_spline(C,T,sd,T_eval=None,num_samples=None):
    return _evaluate_spline(C,T,sd,_evaluate_polynomial_spline_scalar,T_eval,num_samples)



def evaluate_trigonometric_spline(C,T,sd,T_eval=None,num_samples=None):
    return _evaluate_spline(C,T,sd,_evaluate_trigonometric_spline_scalar,T_eval,num_samples)



def _compute_catmull_rom_spline_desc(T,P,S,Z,degree):

    assert P.shape == T.shape
    assert degree in [3,5,7,9]

    sd = {}

    sd["num_segments"] = P.shape[0] - 1

    if S is None:
        S = []

    if Z is None:
        Z = []

    S = array(S,dtype=int32)
    Z = array(Z,dtype=int32)

    S[S == -1] = sd["num_segments"] - 1
    Z[Z == -1] = sd["num_segments"]

    sd["degree"]                      = degree
    sd["num_dimensions"]              = P.shape[1]
    sd["num_constraints_per_segment"] = degree + 1
    sd["num_unknowns_per_segment"]    = degree + 1

    sd["straight_line_segments"]      = S
    sd["zero_derivative_keyframes"]   = Z

    assert not any(diff(sd["straight_line_segments"]) <= 1)

    assert intersect1d(sd["straight_line_segments"],sd["zero_derivative_keyframes"]).size   == 0
    assert intersect1d(sd["straight_line_segments"],sd["zero_derivative_keyframes"]-1).size == 0

    return sd



def _compute_minimum_variation_catmull_rom_spline_desc(T,P,S,Z,degree,lamb):

    assert S is None
    assert Z is None

    assert P.shape == T.shape
    assert degree in [5,7,9,11]

    sd = {}

    sd["num_segments"] = P.shape[0] - 1

    if S is None:
        S = []

    if Z is None:
        Z = []

    S = array(S,dtype=int32)
    Z = array(Z,dtype=int32)

    S[S == -1] = sd["num_segments"] - 1
    Z[Z == -1] = sd["num_segments"]

    sd["degree"]                      = degree
    sd["lamb"]                        = lamb
    sd["num_dimensions"]              = P.shape[1]
    sd["num_constraints_per_segment"] = degree - 1
    sd["num_unknowns_per_segment"]    = degree + 1

    sd["straight_line_segments"]      = S
    sd["zero_derivative_keyframes"]   = Z

    assert intersect1d(sd["straight_line_segments"],sd["zero_derivative_keyframes"]).size   == 0
    assert intersect1d(sd["straight_line_segments"],sd["zero_derivative_keyframes"]-1).size == 0

    return sd



def _compute_nonlocal_interpolating_b_spline_desc(T,P,S,Z,degree):

    assert S is None
    assert Z is None

    assert P.shape == T.shape
    assert degree in [3]

    sd = {}

    sd["num_segments"] = P.shape[0] - 1

    if S is None:
        S = []

    if Z is None:
        Z = []

    S = array(S,dtype=int32)
    Z = array(Z,dtype=int32)

    S[S == -1] = sd["num_segments"] - 1
    Z[Z == -1] = sd["num_segments"]

    sd["degree"]                       = degree
    sd["num_knots"]                    = P.shape[0]
    sd["num_dimensions"]               = P.shape[1]

    sd["num_constraints_per_segment"]  = 6
    sd["num_unknowns_per_segment"]     = 4
    sd["num_unknowns_per_knot"]        = 2
    sd["num_constraints_begin_spline"] = 1
    sd["num_constraints_end_spline"]   = 1

    sd["straight_line_segments"]       = S
    sd["zero_derivative_keyframes"]    = Z

    assert intersect1d(sd["straight_line_segments"],sd["zero_derivative_keyframes"]).size   == 0
    assert intersect1d(sd["straight_line_segments"],sd["zero_derivative_keyframes"]-1).size == 0

    return sd



def _compute_minimum_variation_nonlocal_interpolating_b_spline_desc(T,P,S,Z,degree,continuity,lamb):

    assert P.shape == T.shape
    assert degree in [5,7,9]
    assert continuity in [1,2,3,4]

    sd = {}

    sd["num_segments"] = P.shape[0] - 1

    if S is None:
        S = []

    if Z is None:
        Z = []

    S = array(S,dtype=int32)
    Z = array(Z,dtype=int32)

    S[S == -1] = sd["num_segments"] - 1
    Z[Z == -1] = sd["num_segments"]

    sd["degree"]                                    = degree
    sd["continuity"]                                = continuity
    sd["lamb"]                                      = lamb
    sd["num_knots"]                                 = P.shape[0]
    sd["num_dimensions"]                            = P.shape[1]

    sd["num_unknowns_per_segment"]                  = degree + 1
    sd["num_unknowns_per_knot"]                     = continuity

    sd["num_constraints_per_polynomial_segment"]    = 2*(continuity+1)
    sd["num_constraints_per_straight_line_segment"] = sd["num_unknowns_per_segment"] + 2*sd["num_unknowns_per_knot"]

    sd["polynomial_segments"]                       = setdiff1d(arange(sd["num_segments"]),S)
    sd["straight_line_segments"]                    = S
    sd["zero_derivative_keyframes"]                 = Z

    sd["num_polynomial_segments"]                   = len(sd["polynomial_segments"])
    sd["num_straight_line_segments"]                = len(sd["straight_line_segments"])
    sd["num_zero_derivative_keyframes"]             = len(sd["zero_derivative_keyframes"])

    if 0 in sd["zero_derivative_keyframes"] or 0 in sd["straight_line_segments"]:
        sd["num_constraints_begin_spline"] = 0
    else:
        sd["num_constraints_begin_spline"] = 1

    if sd["num_knots"]-1 in sd["zero_derivative_keyframes"]  or sd["num_segments"]-1 in sd["straight_line_segments"]:
        sd["num_constraints_end_spline"] = 0
    else:
        sd["num_constraints_end_spline"] = 1

    assert intersect1d(sd["straight_line_segments"],sd["zero_derivative_keyframes"]).size   == 0
    assert intersect1d(sd["straight_line_segments"],sd["zero_derivative_keyframes"]-1).size == 0

    return sd



def _compute_trigonometric_spline_desc(T,P,S,Z,order):

    assert S is None
    assert Z is None
    assert P.shape == T.shape
    assert order in [2,5]

    if S is None:
        S = []

    if Z is None:
        Z = []

    sd = {}

    sd["num_segments"] = P.shape[0] - 1

    S = array(S,dtype=int32)
    Z = array(Z,dtype=int32)

    S[S == -1] = sd["num_segments"] - 1
    Z[Z == -1] = sd["num_segments"]

    sd["order"]                       = order
    sd["num_dimensions"]              = P.shape[1]
    sd["num_constraints_per_segment"] = 2*order
    sd["num_unknowns_per_segment"]    = 2*order

    sd["straight_line_segments"]    = S
    sd["zero_derivative_keyframes"] = Z

    assert intersect1d(sd["straight_line_segments"],sd["zero_derivative_keyframes"]).size   == 0
    assert intersect1d(sd["straight_line_segments"],sd["zero_derivative_keyframes"]-1).size == 0

    return sd



def _compute_catmull_rom_spline_derivatives(t,p,sd):

    t_keyframes = t["keyframes"]
    p_keyframes = p["keyframes"]

    t_keyframes_diff = diff(t["keyframes"])
    p_keyframes_diff = diff(p["keyframes"])

    assert all(t_keyframes_diff > 0.0)

    #
    # compute unscaled derivatives at keyframes
    #
    pdot_keyframes                                  = zeros_like(p_keyframes)
    pdot_keyframes[0]                               = p_keyframes_diff[0]/t_keyframes_diff[0]
    pdot_keyframes[-1]                              = p_keyframes_diff[-1]/t_keyframes_diff[-1]
    pdot_keyframes[1:-1]                            = (p_keyframes[2:] - p_keyframes[0:-2]) / (t_keyframes[2:] - t_keyframes[0:-2])
    pdot_keyframes[sd["zero_derivative_keyframes"]] = 0

    #
    # constrain the derivatives for straight line segments
    #
    for si in sd["straight_line_segments"]:
        pdot_straight_line_segment = (p_keyframes[si+1]-p_keyframes[si]) / (t_keyframes[si+1] - t_keyframes[si])
        pdot_keyframes[si]         = pdot_straight_line_segment
        pdot_keyframes[si+1]       = pdot_straight_line_segment

    #
    # compute scaled derivatives
    #
    pdot_segments_begin = pdot_keyframes[:-1] * t_keyframes_diff
    pdot_segments_end   = pdot_keyframes[1:]  * t_keyframes_diff

    #
    # adjust scaled derivatives on either side of straight line constraints
    #
    for si in sd["straight_line_segments"]:

        pdot_straight_line_segment = (p_keyframes[si+1]-p_keyframes[si]) / (t_keyframes[si+1] - t_keyframes[si])

        if si > 0:
            pdot_segments_end[si-1]   = pdot_straight_line_segment * t_keyframes_diff[si-1]
        if si+1 < sd["num_segments"]:
            pdot_segments_begin[si+1] = pdot_straight_line_segment * t_keyframes_diff[si+1]

    #
    # return all relevant values in a dict
    #
    pdot = {}
    pdot["keyframes"]      = pdot_keyframes
    pdot["segments_begin"] = pdot_segments_begin
    pdot["segments_end"]   = pdot_segments_end

    return pdot



def _compute_catmull_rom_spline_coefficients_scalar(t_keyframes,p_keyframes,sd):

    assert all(diff(t_keyframes) > 0.0)
    assert sorted(t_keyframes)

    t = {}
    t["keyframes"]       = t_keyframes
    t["segments_begin"]  = t_keyframes[:-1]
    t["segments_end"]    = t_keyframes[1:]

    p = {}
    p["keyframes"]      = p_keyframes
    p["segments_begin"] = p_keyframes[:-1]
    p["segments_end"]   = p_keyframes[1:]

    if sd["degree"] == 3:
        pd1 = _compute_catmull_rom_spline_derivatives(t,p,sd)
        A,b = _compute_catmull_rom_spline_linear_system_degree_3(t,p,pd1,sd)

    if sd["degree"] == 5:
        pd1 = _compute_catmull_rom_spline_derivatives(t,p,sd)
        pd2 = _compute_catmull_rom_spline_derivatives(t,pd1,sd)
        A,b = _compute_catmull_rom_spline_linear_system_degree_5(t,p,pd1,pd2,sd)

    if sd["degree"] == 7:
        pd1 = _compute_catmull_rom_spline_derivatives(t,p,sd)
        pd2 = _compute_catmull_rom_spline_derivatives(t,pd1,sd)
        pd3 = _compute_catmull_rom_spline_derivatives(t,pd2,sd)
        A,b = _compute_catmull_rom_spline_linear_system_degree_7(t,p,pd1,pd2,pd3,sd)

    if sd["degree"] == 9:
        pd1 = _compute_catmull_rom_spline_derivatives(t,p,sd)
        pd2 = _compute_catmull_rom_spline_derivatives(t,pd1,sd)
        pd3 = _compute_catmull_rom_spline_derivatives(t,pd2,sd)
        pd4 = _compute_catmull_rom_spline_derivatives(t,pd3,sd)
        A,b = _compute_catmull_rom_spline_linear_system_degree_9(t,p,pd1,pd2,pd3,pd4,sd)

    c = linalg.solve(A,b)

    return c.squeeze()



def _compute_minimum_variation_catmull_rom_spline_coefficients_scalar(t_keyframes,p_keyframes,sd):

    assert all(diff(t_keyframes) > 0.0)
    assert sorted(t_keyframes)

    t = {}
    t["keyframes"]       = t_keyframes
    t["segments_begin"]  = t_keyframes[:-1]
    t["segments_end"]    = t_keyframes[1:]

    p = {}
    p["keyframes"]      = p_keyframes
    p["segments_begin"] = p_keyframes[:-1]
    p["segments_end"]   = p_keyframes[1:]

    if sd["degree"] == 5:
        pd1 = _compute_catmull_rom_spline_derivatives(t,p,sd)
        A,b = _compute_minimum_variation_catmull_rom_spline_linear_system_degree_5(t,p,pd1,sd)
        P,q = _compute_minimum_variation_catmull_rom_spline_quadratic_objective_degree_5(sd)

    if sd["degree"] == 7:
        pd1 = _compute_catmull_rom_spline_derivatives(t,p,sd)
        pd2 = _compute_catmull_rom_spline_derivatives(t,pd1,sd)
        A,b = _compute_minimum_variation_catmull_rom_spline_linear_system_degree_7(t,p,pd1,pd2,sd)
        P,q = _compute_minimum_variation_catmull_rom_spline_quadratic_objective_degree_7(sd)

    if sd["degree"] == 9:
        pd1 = _compute_catmull_rom_spline_derivatives(t,p,sd)
        pd2 = _compute_catmull_rom_spline_derivatives(t,pd1,sd)
        pd3 = _compute_catmull_rom_spline_derivatives(t,pd2,sd)
        A,b = _compute_minimum_variation_catmull_rom_spline_linear_system_degree_9(t,p,pd1,pd2,pd3,sd)
        P,q = _compute_minimum_variation_catmull_rom_spline_quadratic_objective_degree_9(sd)

    if sd["degree"] == 11:
        pd1 = _compute_catmull_rom_spline_derivatives(t,p,sd)
        pd2 = _compute_catmull_rom_spline_derivatives(t,pd1,sd)
        pd3 = _compute_catmull_rom_spline_derivatives(t,pd2,sd)
        pd4 = _compute_catmull_rom_spline_derivatives(t,pd3,sd)
        A,b = _compute_minimum_variation_catmull_rom_spline_linear_system_degree_11(t,p,pd1,pd2,pd3,pd4,sd)
        P,q = _compute_minimum_variation_catmull_rom_spline_quadratic_objective_degree_11(sd)

    # print "flashlight.splineutils: Solving QP with cvxopt.solvers.qp..."

    # A_cvx  = cvxopt.matrix(A)
    # b_cvx  = cvxopt.matrix(b)
    # P2_cvx = cvxopt.matrix(2*P)
    # q_cvx  = cvxopt.matrix(q)

    # c_sol_cvx = cvxopt.solvers.qp(P2_cvx, q_cvx, A=A_cvx, b=b_cvx)
    # c_cvx     = array(c_sol_cvx["x"]).squeeze()

    # print "flashlight.splineutils: Solving QP with numpy.linalg.solve..."

    Alpha = bmat([[ 2*P, A.T ],[ A, zeros((A.shape[0],A.shape[0])) ]])
    beta  = bmat([[ zeros((P.shape[0],1)) ],[ b ]])
    chi   = linalg.solve(Alpha,beta)
    c     = chi[0:P.shape[0]].A1
    
    # assert allclose(c,c_cvx)

    # print "flashlight.splineutils: Both solutions are equal."

    return c



def _compute_nonlocal_interpolating_b_spline_coefficients_scalar(t_keyframes,p_keyframes,sd):

    assert all(diff(t_keyframes) > 0.0)
    assert sorted(t_keyframes)

    t = {}
    t["keyframes"]       = t_keyframes
    t["segments_begin"]  = t_keyframes[:-1]
    t["segments_end"]    = t_keyframes[1:]

    p = {}
    p["keyframes"]      = p_keyframes
    p["segments_begin"] = p_keyframes[:-1]
    p["segments_end"]   = p_keyframes[1:]

    A,b = _compute_nonlocal_interpolating_b_spline_linear_system_degree_3(t,p,sd)

    c = linalg.solve(A,b)

    return c.squeeze()



def _compute_minimum_variation_nonlocal_interpolating_b_spline_coefficients_scalar(t_keyframes,p_keyframes,sd):

    assert all(diff(t_keyframes) > 0.0)
    assert sorted(t_keyframes)

    t = {}
    t["keyframes"]       = t_keyframes
    t["segments_begin"]  = t_keyframes[:-1]
    t["segments_end"]    = t_keyframes[1:]

    p = {}
    p["keyframes"]      = p_keyframes
    p["segments_begin"] = p_keyframes[:-1]
    p["segments_end"]   = p_keyframes[1:]

    if sd["degree"] == 5:

        A,b = _compute_minimum_variation_nonlocal_interpolating_b_spline_linear_system_degree_5(t,p,sd)
        P,q = _compute_minimum_variation_catmull_rom_spline_quadratic_objective_degree_5(sd)

        Z = zeros((sd["num_knots"]*sd["num_unknowns_per_knot"],sd["num_knots"]*sd["num_unknowns_per_knot"]))
        z = zeros((sd["num_knots"]*sd["num_unknowns_per_knot"],1))

        P = scipy.linalg.block_diag(P,Z)
        q = bmat([[q],[z]])

    if sd["degree"] == 7:

        A,b = _compute_minimum_variation_nonlocal_interpolating_b_spline_linear_system_degree_7(t,p,sd)
        P,q = _compute_minimum_variation_catmull_rom_spline_quadratic_objective_degree_7(sd)

        Z = zeros((sd["num_knots"]*sd["num_unknowns_per_knot"],sd["num_knots"]*sd["num_unknowns_per_knot"]))
        z = zeros((sd["num_knots"]*sd["num_unknowns_per_knot"],1))

        P = scipy.linalg.block_diag(P,Z)
        q = bmat([[q],[z]])

    if sd["degree"] == 9:

        A,b = _compute_minimum_variation_nonlocal_interpolating_b_spline_linear_system_degree_9(t,p,sd)
        P,q = _compute_minimum_variation_catmull_rom_spline_quadratic_objective_degree_9(sd)

        Z = zeros((sd["num_knots"]*sd["num_unknowns_per_knot"],sd["num_knots"]*sd["num_unknowns_per_knot"]))
        z = zeros((sd["num_knots"]*sd["num_unknowns_per_knot"],1))

        P = scipy.linalg.block_diag(P,Z)
        q = bmat([[q],[z]])

    # print "flashlight.splineutils: Solving QP with cvxopt.solvers.qp..."

    # A_cvx  = cvxopt.matrix(A)
    # b_cvx  = cvxopt.matrix(b)
    # P2_cvx = cvxopt.matrix(2*P)
    # q_cvx  = cvxopt.matrix(q)

    # c_sol_cvx = cvxopt.solvers.qp(P2_cvx, q_cvx, A=A_cvx, b=b_cvx)
    # c_cvx     = array(c_sol_cvx["x"]).squeeze()

    # print "flashlight.splineutils: Solving QP with numpy.linalg.solve..."

    Alpha = bmat([[ 2*P, A.T ],[ A, zeros((A.shape[0],A.shape[0])) ]])
    beta  = bmat([[ zeros((P.shape[0],1)) ],[ b ]])
    chi   = linalg.solve(Alpha,beta)
    c     = chi[0:P.shape[0]].A1

    # assert allclose(c,c_cvx)

    # print "flashlight.splineutils: Both solutions are equal."

    return c



def _compute_trigonometric_spline_coefficients_scalar(t_keyframes,p_keyframes,sd):

    assert all(diff(t_keyframes) > 0.0)
    assert sorted(t_keyframes)

    t = {}
    t["keyframes"]       = t_keyframes
    t["segments_begin"]  = t_keyframes[:-1]
    t["segments_end"]    = t_keyframes[1:]

    p = {}
    p["keyframes"]      = p_keyframes
    p["segments_begin"] = p_keyframes[:-1]
    p["segments_end"]   = p_keyframes[1:]

    if sd["order"] == 2:
        pd1 = _compute_catmull_rom_spline_derivatives(t,p,sd)
        A,b = _compute_trigonometric_spline_linear_system_order_2(t,p,pd1,sd)

    if sd["order"] == 5:
        pd1 = _compute_catmull_rom_spline_derivatives(t,p,sd)
        pd2 = _compute_catmull_rom_spline_derivatives(t,pd1,sd)
        pd3 = _compute_catmull_rom_spline_derivatives(t,pd2,sd)
        pd4 = _compute_catmull_rom_spline_derivatives(t,pd3,sd)
        A,b = _compute_trigonometric_spline_linear_system_order_5(t,p,pd1,pd2,pd3,pd4,sd)

    c = linalg.solve(A,b)

    return c.squeeze()



def _compute_catmull_rom_spline_linear_system_degree_3(t,p,pd1,sd):

    A = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))
    b = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [ 1, t_begin, t_begin**2, t_begin**3 ]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 1, t_end, t_end**2, t_end**3 ]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_begin, 3*t_begin**2 ]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_end, 3*t_end**2 ]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = pd1["segments_begin"][si]
        b[r+3]                                    = pd1["segments_end"][si]

    for si in sd["straight_line_segments"]:

        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [1,t_begin,0,0]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [1,t_end,0,0]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [0,0,1,0]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,1]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = 0
        b[r+3]                                    = 0

    return A,b



def _compute_catmull_rom_spline_linear_system_degree_5(t,p,pd1,pd2,sd):

    A = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))
    b = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [ 1, t_begin, t_begin**2, t_begin**3, t_begin**4, t_begin**5 ]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 1, t_end, t_end**2, t_end**3, t_end**4, t_end**5 ]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_begin, 3*t_begin**2, 4*t_begin**3, 5*t_begin**4 ]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_end, 3*t_end**2, 4*t_end**3, 5*t_end**4 ]
        A[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_begin, 12*t_begin**2, 20*t_begin**3 ]
        A[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_end, 12*t_end**2, 20*t_end**3 ]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = pd1["segments_begin"][si]
        b[r+3]                                    = pd1["segments_end"][si]
        b[r+4]                                    = pd2["segments_begin"][si]
        b[r+5]                                    = pd2["segments_end"][si]
        
    for si in sd["straight_line_segments"]:
        
        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [1,t_begin,0,0,0,0]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [1,t_end,0,0,0,0]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [0,0,1,0,0,0]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,1,0,0]
        A[r+4,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,1,0]
        A[r+5,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,0,1]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = 0
        b[r+3]                                    = 0
        b[r+4]                                    = 0
        b[r+5]                                    = 0

    return A,b



def _compute_catmull_rom_spline_linear_system_degree_7(t,p,pd1,pd2,pd3,sd):

    A = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))
    b = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [ 1, t_begin, t_begin**2, t_begin**3, t_begin**4, t_begin**5, t_begin**6, t_begin**7 ]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 1, t_end, t_end**2, t_end**3, t_end**4, t_end**5, t_end**6, t_end**7 ]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_begin, 3*t_begin**2, 4*t_begin**3, 5*t_begin**4, 6*t_begin**5, 7*t_begin**6 ]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_end, 3*t_end**2, 4*t_end**3, 5*t_end**4, 6*t_end**5, 7*t_end**6 ]
        A[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_begin, 12*t_begin**2, 20*t_begin**3, 30*t_begin**4, 42*t_begin**5 ]
        A[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_end, 12*t_end**2, 20*t_end**3, 30*t_end**4, 42*t_end**5 ]
        A[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 6, 24*t_begin, 60*t_begin**2, 120*t_begin**3, 210*t_begin**4 ]
        A[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 6, 24*t_end, 60*t_end**2, 120*t_end**3, 210*t_end**4 ]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = pd1["segments_begin"][si]
        b[r+3]                                    = pd1["segments_end"][si]
        b[r+4]                                    = pd2["segments_begin"][si]
        b[r+5]                                    = pd2["segments_end"][si]
        b[r+6]                                    = pd3["segments_begin"][si]
        b[r+7]                                    = pd3["segments_end"][si]
        
    for si in sd["straight_line_segments"]:
        
        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [1,t_begin,0,0,0,0,0,0]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [1,t_end,0,0,0,0,0,0]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [0,0,1,0,0,0,0,0]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,1,0,0,0,0]
        A[r+4,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,1,0,0,0]
        A[r+5,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,0,1,0,0]
        A[r+6,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,0,0,1,0]
        A[r+7,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,0,0,0,1]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = 0
        b[r+3]                                    = 0
        b[r+4]                                    = 0
        b[r+5]                                    = 0
        b[r+6]                                    = 0
        b[r+7]                                    = 0

    return A,b



def _compute_catmull_rom_spline_linear_system_degree_9(t,p,pd1,pd2,pd3,pd4,sd):

    A = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))
    b = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [ 1, t_begin, t_begin**2, t_begin**3, t_begin**4, t_begin**5, t_begin**6, t_begin**7, t_begin**8, t_begin**9 ]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 1, t_end, t_end**2, t_end**3, t_end**4, t_end**5, t_end**6, t_end**7, t_end**8, t_end**9 ]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_begin, 3*t_begin**2, 4*t_begin**3, 5*t_begin**4, 6*t_begin**5, 7*t_begin**6, 8*t_begin**7, 9*t_begin**8 ]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_end, 3*t_end**2, 4*t_end**3, 5*t_end**4, 6*t_end**5, 7*t_end**6, 8*t_end**7, 9*t_end**8 ]
        A[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_begin, 12*t_begin**2, 20*t_begin**3, 30*t_begin**4, 42*t_begin**5, 56*t_begin**6, 72*t_begin**7 ]
        A[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_end, 12*t_end**2, 20*t_end**3, 30*t_end**4, 42*t_end**5, 56*t_end**6, 72*t_end**7 ]
        A[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 6, 24*t_begin, 60*t_begin**2, 120*t_begin**3, 210*t_begin**4, 336*t_begin**5, 504*t_begin**6 ]
        A[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 6, 24*t_end, 60*t_end**2, 120*t_end**3, 210*t_end**4, 336*t_end**5, 504*t_end**6 ]
        A[r+8,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 0, 24, 120*t_begin, 360*t_begin**2, 840*t_begin**3, 1680*t_begin**4, 3024*t_begin**5 ]
        A[r+9,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 0, 24, 120*t_end, 360*t_end**2, 840*t_end**3, 1680*t_end**4, 3024*t_end**5 ]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = pd1["segments_begin"][si]
        b[r+3]                                    = pd1["segments_end"][si]
        b[r+4]                                    = pd2["segments_begin"][si]
        b[r+5]                                    = pd2["segments_end"][si]
        b[r+6]                                    = pd3["segments_begin"][si]
        b[r+7]                                    = pd3["segments_end"][si]
        b[r+8]                                    = pd4["segments_begin"][si]
        b[r+9]                                    = pd4["segments_end"][si]

    for si in sd["straight_line_segments"]:
        
        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [1,t_begin,0,0,0,0,0,0,0,0]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [1,t_end,0,0,0,0,0,0,0,0]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [0,0,1,0,0,0,0,0,0,0]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,1,0,0,0,0,0,0]
        A[r+4,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,1,0,0,0,0,0]
        A[r+5,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,0,1,0,0,0,0]
        A[r+6,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,0,0,1,0,0,0]
        A[r+7,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,0,0,0,1,0,0]
        A[r+8,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,0,0,0,0,1,0]
        A[r+9,c:c+sd["num_unknowns_per_segment"]] = [0,0,0,0,0,0,0,0,0,1]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = 0
        b[r+3]                                    = 0
        b[r+4]                                    = 0
        b[r+5]                                    = 0
        b[r+6]                                    = 0
        b[r+7]                                    = 0
        b[r+8]                                    = 0
        b[r+9]                                    = 0

    return A,b



def _compute_minimum_variation_catmull_rom_spline_linear_system_degree_5(t,p,pd1,sd):

    A = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))
    b = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [ 1, t_begin, t_begin**2, t_begin**3, t_begin**4, t_begin**5 ]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 1, t_end, t_end**2, t_end**3, t_end**4, t_end**5 ]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_begin, 3*t_begin**2, 4*t_begin**3, 5*t_begin**4 ]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_end, 3*t_end**2, 4*t_end**3, 5*t_end**4 ]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = pd1["segments_begin"][si]
        b[r+3]                                    = pd1["segments_end"][si]

    return A,b



def _compute_minimum_variation_catmull_rom_spline_linear_system_degree_7(t,p,pd1,pd2,sd):

    A = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))
    b = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [ 1, t_begin, t_begin**2, t_begin**3, t_begin**4, t_begin**5, t_begin**6, t_begin**7 ]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 1, t_end, t_end**2, t_end**3, t_end**4, t_end**5, t_end**6, t_end**7 ]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_begin, 3*t_begin**2, 4*t_begin**3, 5*t_begin**4, 6*t_begin**5, 7*t_begin**6 ]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_end, 3*t_end**2, 4*t_end**3, 5*t_end**4, 6*t_end**5, 7*t_end**6 ]
        A[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_begin, 12*t_begin**2, 20*t_begin**3, 30*t_begin**4, 42*t_begin**5 ]
        A[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_end, 12*t_end**2, 20*t_end**3, 30*t_end**4, 42*t_end**5 ]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = pd1["segments_begin"][si]
        b[r+3]                                    = pd1["segments_end"][si]
        b[r+4]                                    = pd2["segments_begin"][si]
        b[r+5]                                    = pd2["segments_end"][si]

    return A,b



def _compute_minimum_variation_catmull_rom_spline_linear_system_degree_9(t,p,pd1,pd2,pd3,sd):

    A = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))
    b = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [ 1, t_begin, t_begin**2, t_begin**3, t_begin**4, t_begin**5, t_begin**6, t_begin**7, t_begin**8, t_begin**9 ]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 1, t_end, t_end**2, t_end**3, t_end**4, t_end**5, t_end**6, t_end**7, t_end**8, t_end**9 ]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_begin, 3*t_begin**2, 4*t_begin**3, 5*t_begin**4, 6*t_begin**5, 7*t_begin**6, 8*t_begin**7, 9*t_begin**8 ]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_end, 3*t_end**2, 4*t_end**3, 5*t_end**4, 6*t_end**5, 7*t_end**6, 8*t_end**7, 9*t_end**8 ]
        A[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_begin, 12*t_begin**2, 20*t_begin**3, 30*t_begin**4, 42*t_begin**5, 56*t_begin**6, 72*t_begin**7 ]
        A[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_end, 12*t_end**2, 20*t_end**3, 30*t_end**4, 42*t_end**5, 56*t_end**6, 72*t_end**7 ]
        A[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 6, 24*t_begin, 60*t_begin**2, 120*t_begin**3, 210*t_begin**4, 336*t_begin**5, 504*t_begin**6 ]
        A[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 6, 24*t_end, 60*t_end**2, 120*t_end**3, 210*t_end**4, 336*t_end**5, 504*t_end**6 ]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = pd1["segments_begin"][si]
        b[r+3]                                    = pd1["segments_end"][si]
        b[r+4]                                    = pd2["segments_begin"][si]
        b[r+5]                                    = pd2["segments_end"][si]
        b[r+6]                                    = pd3["segments_begin"][si]
        b[r+7]                                    = pd3["segments_end"][si]

    return A,b



def _compute_minimum_variation_catmull_rom_spline_linear_system_degree_11(t,p,pd1,pd2,pd3,pd4,sd):

    A = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))
    b = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r       = si*sd["num_constraints_per_segment"]
        c       = si*sd["num_unknowns_per_segment"]
        t_begin = 0
        t_end   = 1

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [ 1, t_begin, t_begin**2, t_begin**3, t_begin**4, t_begin**5, t_begin**6, t_begin**7, t_begin**8, t_begin**9, t_begin**10, t_begin**11 ]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 1, t_end, t_end**2, t_end**3, t_end**4, t_end**5, t_end**6, t_end**7, t_end**8, t_end**9, t_end**10, t_end**11 ]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_begin, 3*t_begin**2, 4*t_begin**3, 5*t_begin**4, 6*t_begin**5, 7*t_begin**6, 8*t_begin**7, 9*t_begin**8, 10*t_begin**9, 11*t_begin**10 ]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0, 1, 2*t_end, 3*t_end**2, 4*t_end**3, 5*t_end**4, 6*t_end**5, 7*t_end**6, 8*t_end**7, 9*t_end**8, 10*t_end**9, 11*t_end**10 ]
        A[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_begin, 12*t_begin**2, 20*t_begin**3, 30*t_begin**4, 42*t_begin**5, 56*t_begin**6, 72*t_begin**7, 90*t_begin**8, 110*t_begin**9 ]
        A[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 2, 6*t_end, 12*t_end**2, 20*t_end**3, 30*t_end**4, 42*t_end**5, 56*t_end**6, 72*t_end**7, 90*t_end**8, 110*t_end**9 ]
        A[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 6, 24*t_begin, 60*t_begin**2, 120*t_begin**3, 210*t_begin**4, 336*t_begin**5, 504*t_begin**6, 720*t_begin**7, 990*t_begin**8 ]
        A[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 6, 24*t_end, 60*t_end**2, 120*t_end**3, 210*t_end**4, 336*t_end**5, 504*t_end**6, 720*t_end**7, 990*t_end**8 ]
        A[r+8,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 0, 24, 120*t_begin, 360*t_begin**2, 840*t_begin**3, 1680*t_begin**4, 3024*t_begin**5, 5040*t_begin**6, 7920*t_begin**7 ]
        A[r+9,c:c+sd["num_unknowns_per_segment"]] = [ 0, 0, 0, 0, 24, 120*t_end, 360*t_end**2, 840*t_end**3, 1680*t_end**4, 3024*t_end**5, 5040*t_end**6, 7920*t_end**7 ]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = pd1["segments_begin"][si]
        b[r+3]                                    = pd1["segments_end"][si]
        b[r+4]                                    = pd2["segments_begin"][si]
        b[r+5]                                    = pd2["segments_end"][si]
        b[r+6]                                    = pd3["segments_begin"][si]
        b[r+7]                                    = pd3["segments_end"][si]
        b[r+8]                                    = pd4["segments_begin"][si]
        b[r+9]                                    = pd4["segments_end"][si]

    return A,b



def _compute_minimum_variation_catmull_rom_spline_quadratic_objective_degree_5(sd):

    P = zeros((sd["num_segments"]*sd["num_unknowns_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))

    P_d1 = zeros_like(P)
    P_d2 = zeros_like(P)
    P_d3 = zeros_like(P)
    P_d4 = zeros_like(P)
    P_d5 = zeros_like(P)

    q = zeros((sd["num_segments"]*sd["num_unknowns_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r                                            = si*sd["num_unknowns_per_segment"]
        c                                            = si*sd["num_unknowns_per_segment"]

        P_d1[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d1[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]
        P_d1[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 4.0/3.0, 3.0/2.0, 8.0/5.0, 5.0/3.0 ]
        P_d1[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 3.0/2.0, 9.0/5.0, 2.0, 15.0/7.0 ]
        P_d1[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 8.0/5.0, 2.0, 16.0/7.0, 5.0/2.0 ]
        P_d1[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 5.0/3.0, 15.0/7.0, 5.0/2.0, 25.0/9.0 ]

        P_d2[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d2[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d2[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 4.0, 6.0, 8.0, 10.0 ]
        P_d2[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 6.0, 12.0, 18.0, 24.0 ]
        P_d2[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 8.0, 18.0, 144.0/5.0, 40.0 ]
        P_d2[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 10.0, 24.0, 40.0, 400.0/7.0 ]

        P_d3[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 36.0, 72.0, 120.0 ]
        P_d3[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 72.0, 192.0, 360.0 ]
        P_d3[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 120.0, 360.0, 720.0 ]

        P_d4[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 576.0, 1440.0 ]
        P_d4[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 1440.0, 4800.0 ]

        P_d5[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 14400.0 ]

    P = sd["lamb"][0]*P_d1 + sd["lamb"][1]*P_d2 + sd["lamb"][2]*P_d3 + sd["lamb"][3]*P_d4 + sd["lamb"][4]*P_d5

    return P,q



def _compute_minimum_variation_catmull_rom_spline_quadratic_objective_degree_7(sd):

    P = zeros((sd["num_segments"]*sd["num_unknowns_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))

    P_d1 = zeros_like(P)
    P_d2 = zeros_like(P)
    P_d3 = zeros_like(P)
    P_d4 = zeros_like(P)
    P_d5 = zeros_like(P)

    q = zeros((sd["num_segments"]*sd["num_unknowns_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r                                            = si*sd["num_unknowns_per_segment"]
        c                                            = si*sd["num_unknowns_per_segment"]

        P_d1[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d1[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]
        P_d1[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 4.0/3.0, 3.0/2.0, 8.0/5.0, 5.0/3.0, 12.0/7.0, 7.0/4.0 ]
        P_d1[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 3.0/2.0, 9.0/5.0, 2.0, 15.0/7.0, 9.0/4.0, 7.0/3.0 ]
        P_d1[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 8.0/5.0, 2.0, 16.0/7.0, 5.0/2.0, 8.0/3.0, 14.0/5.0 ]
        P_d1[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 5.0/3.0, 15.0/7.0, 5.0/2.0, 25.0/9.0, 3.0, 35.0/11.0 ]
        P_d1[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 12.0/7.0, 9.0/4.0, 8.0/3.0, 3.0, 36.0/11.0, 7.0/2.0 ]
        P_d1[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 7.0/4.0, 7.0/3.0, 14.0/5.0, 35.0/11.0, 7.0/2.0, 49.0/13.0 ]

        P_d2[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d2[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d2[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0 ]
        P_d2[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0 ]
        P_d2[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 8.0, 18.0, 144.0/5.0, 40.0, 360.0/7.0, 63.0 ]
        P_d2[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 10.0, 24.0, 40.0, 400.0/7.0, 75.0, 280.0/3.0 ]
        P_d2[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 12.0, 30.0, 360.0/7.0, 75.0, 100.0, 126.0 ]
        P_d2[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 14.0, 36.0, 63.0, 280.0/3.0, 126.0, 1764.0/11.0 ]

        P_d3[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 36.0, 72.0, 120.0, 180.0, 252.0 ]
        P_d3[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 72.0, 192.0, 360.0, 576.0, 840.0 ]
        P_d3[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 120.0, 360.0, 720.0, 1200.0, 1800.0 ]
        P_d3[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 180.0, 576.0, 1200.0, 14400.0/7.0, 3150.0 ]
        P_d3[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 252.0, 840.0, 1800.0, 3150.0, 4900.0 ]

        P_d4[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 576.0, 1440.0, 2880.0, 5040.0 ]
        P_d4[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 1440.0, 4800.0, 10800.0, 20160.0 ]
        P_d4[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 2880.0, 10800.0, 25920.0, 50400.0 ]
        P_d4[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 5040.0, 20160.0, 50400.0, 100800.0 ]

        P_d5[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 14400.0, 43200.0, 100800.0 ]
        P_d5[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 43200.0, 172800.0, 453600.0 ]
        P_d5[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 100800.0, 453600.0, 1270080.0 ]

    P = sd["lamb"][0]*P_d1 + sd["lamb"][1]*P_d2 + sd["lamb"][2]*P_d3 + sd["lamb"][3]*P_d4 + sd["lamb"][4]*P_d5

    return P,q



def _compute_minimum_variation_catmull_rom_spline_quadratic_objective_degree_9(sd):

    P = zeros((sd["num_segments"]*sd["num_unknowns_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))

    P_d1 = zeros_like(P)
    P_d2 = zeros_like(P)
    P_d3 = zeros_like(P)
    P_d4 = zeros_like(P)
    P_d5 = zeros_like(P)

    q = zeros((sd["num_segments"]*sd["num_unknowns_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r                                            = si*sd["num_unknowns_per_segment"]
        c                                            = si*sd["num_unknowns_per_segment"]

        P_d1[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d1[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]
        P_d1[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 4.0/3.0, 3.0/2.0, 8.0/5.0, 5.0/3.0, 12.0/7.0, 7.0/4.0, 16.0/9.0, 9.0/5.0 ]
        P_d1[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 3.0/2.0, 9.0/5.0, 2.0, 15.0/7.0, 9.0/4.0, 7.0/3.0, 12.0/5.0, 27.0/11.0 ]
        P_d1[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 8.0/5.0, 2.0, 16.0/7.0, 5.0/2.0, 8.0/3.0, 14.0/5.0, 32.0/11.0, 3.0 ]
        P_d1[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 5.0/3.0, 15.0/7.0, 5.0/2.0, 25.0/9.0, 3.0, 35.0/11.0, 10.0/3.0, 45.0/13.0 ]
        P_d1[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 12.0/7.0, 9.0/4.0, 8.0/3.0, 3.0, 36.0/11.0, 7.0/2.0, 48.0/13.0, 27.0/7.0 ]
        P_d1[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 7.0/4.0, 7.0/3.0, 14.0/5.0, 35.0/11.0, 7.0/2.0, 49.0/13.0, 4.0, 21.0/5.0 ]
        P_d1[r+8,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 16.0/9.0, 12.0/5.0, 32.0/11.0, 10.0/3.0, 48.0/13.0, 4.0, 64.0/15.0, 9.0/2.0 ]
        P_d1[r+9,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 9.0/5.0, 27.0/11.0, 3.0, 45.0/13.0, 27.0/7.0, 21.0/5.0, 9.0/2.0, 81.0/17.0 ]

        P_d2[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d2[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d2[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0 ]
        P_d2[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0 ]
        P_d2[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 8.0, 18.0, 144.0/5.0, 40.0, 360.0/7.0, 63.0, 224.0/3.0, 432.0/5.0 ]
        P_d2[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 10.0, 24.0, 40.0, 400.0/7.0, 75.0, 280.0/3.0, 112.0, 1440.0/11.0 ]
        P_d2[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 12.0, 30.0, 360.0/7.0, 75.0, 100.0, 126.0, 1680.0/11.0, 180.0 ]
        P_d2[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 14.0, 36.0, 63.0, 280.0/3.0, 126.0, 1764.0/11.0, 196.0, 3024.0/13.0 ]
        P_d2[r+8,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 16.0, 42.0, 224.0/3.0, 112.0, 1680.0/11.0, 196.0, 3136.0/13.0, 288.0 ]
        P_d2[r+9,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 18.0, 48.0, 432.0/5.0, 1440.0/11.0, 180.0, 3024.0/13.0, 288.0, 1728.0/5.0 ]

        P_d3[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 36.0, 72.0, 120.0, 180.0, 252.0, 336.0, 432.0 ]
        P_d3[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 72.0, 192.0, 360.0, 576.0, 840.0, 1152.0, 1512.0 ]
        P_d3[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 120.0, 360.0, 720.0, 1200.0, 1800.0, 2520.0, 3360.0 ]
        P_d3[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 180.0, 576.0, 1200.0, 14400.0/7.0, 3150.0, 4480.0, 6048.0 ]
        P_d3[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 252.0, 840.0, 1800.0, 3150.0, 4900.0, 7056.0, 105840.0/11.0 ]
        P_d3[r+8,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 336.0, 1152.0, 2520.0, 4480.0, 7056.0, 112896.0/11.0, 14112.0 ]
        P_d3[r+9,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 432.0, 1512.0, 3360.0, 6048.0, 105840.0/11.0, 14112.0, 254016.0/13.0 ]

        P_d4[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 576.0, 1440.0, 2880.0, 5040.0, 8064.0, 12096.0 ]
        P_d4[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 1440.0, 4800.0, 10800.0, 20160.0, 33600.0, 51840.0 ]
        P_d4[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 2880.0, 10800.0, 25920.0, 50400.0, 86400.0, 136080.0 ]
        P_d4[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 5040.0, 20160.0, 50400.0, 100800.0, 176400.0, 282240.0 ]
        P_d4[r+8,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 8064.0, 33600.0, 86400.0, 176400.0, 313600.0, 508032.0 ]
        P_d4[r+9,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 12096.0, 51840.0, 136080.0, 282240.0, 508032.0, 9144576.0/11.0 ]

        P_d5[r,  c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+1,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+2,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+3,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+4,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+5,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 14400.0, 43200.0, 100800.0, 201600.0, 362880.0 ]
        P_d5[r+6,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 43200.0, 172800.0, 453600.0, 967680.0, 1814400.0 ]
        P_d5[r+7,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 100800.0, 453600.0, 1270080.0, 2822400.0, 5443200.0 ]
        P_d5[r+8,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 201600.0, 967680.0, 2822400.0, 6451200.0, 12700800.0 ]
        P_d5[r+9,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 362880.0, 1814400.0, 5443200.0, 12700800.0, 25401600.0 ]

    P = sd["lamb"][0]*P_d1 + sd["lamb"][1]*P_d2 + sd["lamb"][2]*P_d3 + sd["lamb"][3]*P_d4 + sd["lamb"][4]*P_d5

    return P,q



def _compute_minimum_variation_catmull_rom_spline_quadratic_objective_degree_11(sd):

    P = zeros((sd["num_segments"]*sd["num_unknowns_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))

    P_d1 = zeros_like(P)
    P_d2 = zeros_like(P)
    P_d3 = zeros_like(P)
    P_d4 = zeros_like(P)
    P_d5 = zeros_like(P)
    P_d6 = zeros_like(P)

    q = zeros((sd["num_segments"]*sd["num_unknowns_per_segment"],1))

    for si in range(sd["num_segments"]):
        
        r                                             = si*sd["num_unknowns_per_segment"]
        c                                             = si*sd["num_unknowns_per_segment"]

        P_d1[r,  c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d1[r+1,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]
        P_d1[r+2,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 1.0, 4.0/3.0, 3.0/2.0, 8.0/5.0, 5.0/3.0, 12.0/7.0, 7.0/4.0, 16.0/9.0, 9.0/5.0, 20.0/11.0, 11.0/6.0 ]
        P_d1[r+3,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 1.0, 3.0/2.0, 9.0/5.0, 2.0, 15.0/7.0, 9.0/4.0, 7.0/3.0, 12.0/5.0, 27.0/11.0, 5.0/2.0, 33.0/13.0 ]
        P_d1[r+4,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 1.0, 8.0/5.0, 2.0, 16.0/7.0, 5.0/2.0, 8.0/3.0, 14.0/5.0, 32.0/11.0, 3.0, 40.0/13.0, 22.0/7.0 ]
        P_d1[r+5,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 1.0, 5.0/3.0, 15.0/7.0, 5.0/2.0, 25.0/9.0, 3.0, 35.0/11.0, 10.0/3.0, 45.0/13.0, 25.0/7.0, 11.0/3.0 ]
        P_d1[r+6,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 1.0, 12.0/7.0, 9.0/4.0, 8.0/3.0, 3.0, 36.0/11.0, 7.0/2.0, 48.0/13.0, 27.0/7.0, 4.0, 33.0/8.0 ]
        P_d1[r+7,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 1.0, 7.0/4.0, 7.0/3.0, 14.0/5.0, 35.0/11.0, 7.0/2.0, 49.0/13.0, 4.0, 21.0/5.0, 35.0/8.0, 77.0/17.0 ]
        P_d1[r+8,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 1.0, 16.0/9.0, 12.0/5.0, 32.0/11.0, 10.0/3.0, 48.0/13.0, 4.0, 64.0/15.0, 9.0/2.0, 80.0/17.0, 44.0/9.0 ]
        P_d1[r+9,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 1.0, 9.0/5.0, 27.0/11.0, 3.0, 45.0/13.0, 27.0/7.0, 21.0/5.0, 9.0/2.0, 81.0/17.0, 5.0, 99.0/19.0 ]
        P_d1[r+10,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 20.0/11.0, 5.0/2.0, 40.0/13.0, 25.0/7.0, 4.0, 35.0/8.0, 80.0/17.0, 5.0, 100.0/19.0, 11.0/2.0 ]
        P_d1[r+11,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 1.0, 11.0/6.0, 33.0/13.0, 22.0/7.0, 11.0/3.0, 33.0/8.0, 77.0/17.0, 44.0/9.0, 99.0/19.0, 11.0/2.0, 121.0/21.0 ]

        P_d2[r,  c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d2[r+1,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d2[r+2,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0 ]
        P_d2[r+3,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0 ]
        P_d2[r+4,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 8.0, 18.0, 144.0/5.0, 40.0, 360.0/7.0, 63.0, 224.0/3.0, 432.0/5.0, 1080.0/11.0, 110.0 ]
        P_d2[r+5,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 10.0, 24.0, 40.0, 400.0/7.0, 75.0, 280.0/3.0, 112.0, 1440.0/11.0, 150.0, 2200.0/13.0 ]
        P_d2[r+6,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 12.0, 30.0, 360.0/7.0, 75.0, 100.0, 126.0, 1680.0/11.0, 180.0, 2700.0/13.0, 1650.0/7.0 ]
        P_d2[r+7,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 14.0, 36.0, 63.0, 280.0/3.0, 126.0, 1764.0/11.0, 196.0, 3024.0/13.0, 270.0, 308.0 ]
        P_d2[r+8,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 16.0, 42.0, 224.0/3.0, 112.0, 1680.0/11.0, 196.0, 3136.0/13.0, 288.0, 336.0, 385.0 ]
        P_d2[r+9,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 18.0, 48.0, 432.0/5.0, 1440.0/11.0, 180.0, 3024.0/13.0, 288.0, 1728.0/5.0, 405.0, 7920.0/17.0 ]
        P_d2[r+10,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 20.0, 54.0, 1080.0/11.0, 150.0, 2700.0/13.0, 270.0, 336.0, 405.0, 8100.0/17.0, 550.0 ]
        P_d2[r+11,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 22.0, 60.0, 110.0, 2200.0/13.0, 1650.0/7.0, 308.0, 385.0, 7920.0/17.0, 550.0, 12100.0/19.0 ]

        P_d3[r,  c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+1,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+2,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d3[r+3,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 36.0, 72.0, 120.0, 180.0, 252.0, 336.0, 432.0, 540.0, 660.0 ]
        P_d3[r+4,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 72.0, 192.0, 360.0, 576.0, 840.0, 1152.0, 1512.0, 1920.0, 2376.0 ]
        P_d3[r+5,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 120.0, 360.0, 720.0, 1200.0, 1800.0, 2520.0, 3360.0, 4320.0, 5400.0 ]
        P_d3[r+6,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 180.0, 576.0, 1200.0, 14400.0/7.0, 3150.0, 4480.0, 6048.0, 86400.0/11.0, 9900.0 ]
        P_d3[r+7,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 252.0, 840.0, 1800.0, 3150.0, 4900.0, 7056.0, 105840.0/11.0, 12600.0, 207900.0/13.0 ]
        P_d3[r+8,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 336.0, 1152.0, 2520.0, 4480.0, 7056.0, 112896.0/11.0, 14112.0, 241920.0/13.0, 23760.0 ]
        P_d3[r+9,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 432.0, 1512.0, 3360.0, 6048.0, 105840.0/11.0, 14112.0, 254016.0/13.0, 25920.0, 33264.0 ]
        P_d3[r+10,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 540.0, 1920.0, 4320.0, 86400.0/11.0, 12600.0, 241920.0/13.0, 25920.0, 34560.0, 44550.0 ]
        P_d2[r+11,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 660.0, 2376.0, 5400.0, 9900.0, 207900.0/13.0, 23760.0, 33264.0, 44550.0, 980100.0/17.0 ]

        P_d4[r,  c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+1,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+2,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+3,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d4[r+4,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 576.0, 1440.0, 2880.0, 5040.0, 8064.0, 12096.0, 17280.0, 23760.0 ]
        P_d4[r+5,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 1440.0, 4800.0, 10800.0, 20160.0, 33600.0, 51840.0, 75600.0, 105600.0 ]
        P_d4[r+6,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 2880.0, 10800.0, 25920.0, 50400.0, 86400.0, 136080.0, 201600.0, 285120.0 ]
        P_d4[r+7,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 5040.0, 20160.0, 50400.0, 100800.0, 176400.0, 282240.0, 423360.0, 604800.0 ]
        P_d4[r+8,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 8064.0, 33600.0, 86400.0, 176400.0, 313600.0, 508032.0, 8467200.0/11.0, 1108800.0 ]
        P_d4[r+9,c:c+sd["num_unknowns_per_segment"]]  = [ 0.0, 0.0, 0.0, 0.0, 12096.0, 51840.0, 136080.0, 282240.0, 508032.0, 9144576.0/11.0, 1270080.0, 23950080.0/13.0 ]
        P_d4[r+10,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 17280.0, 75600.0, 201600.0, 423360.0, 8467200.0/11.0, 1270080.0, 25401600.0/13.0, 2851200.0 ]
        P_d4[r+11,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 23760.0, 105600.0, 285120.0, 604800.0, 1108800.0, 23950080.0/13.0, 2851200.0, 4181760.0 ]

        P_d5[r,   c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+1, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+2, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+3, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+4, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d5[r+5, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 14400.0, 43200.0, 100800.0, 201600.0, 362880.0, 604800.0, 950400.0 ]
        P_d5[r+6, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 43200.0, 172800.0, 453600.0, 967680.0, 1814400.0, 3110400.0, 4989600.0 ]
        P_d5[r+7, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 100800.0, 453600.0, 1270080.0, 2822400.0, 5443200.0, 9525600.0, 15523200.0 ]
        P_d5[r+8, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 201600.0, 967680.0, 2822400.0, 6451200.0, 12700800.0, 22579200.0, 37255680.0 ]
        P_d5[r+9, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 362880.0, 1814400.0, 5443200.0, 12700800.0, 25401600.0, 45722880.0, 76204800.0 ]
        P_d5[r+10,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 604800.0, 3110400.0, 9525600.0, 22579200.0, 45722880.0, 914457600.0/11.0, 139708800.0 ]
        P_d5[r+11,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 950400.0, 4989600.0, 15523200.0, 37255680.0, 76204800.0, 139708800.0, 3073593600.0/13.0 ]

        P_d6[r,   c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d6[r+1, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d6[r+2, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d6[r+3, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d6[r+4, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d6[r+5, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        P_d6[r+6, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 518400.0, 1814400.0, 4838400.0, 10886400.0, 21772800.0, 39916800.0 ]
        P_d6[r+7, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1814400.0, 8467200.0, 25401600.0, 60963840.0, 127008000.0, 239500800.0 ]
        P_d6[r+8, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4838400.0, 25401600.0, 81285120.0, 203212800.0, 435456000.0, 838252800.0 ]
        P_d6[r+9, c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10886400.0, 60963840.0, 203212800.0, 522547200.0, 1143072000.0, 2235340800.0 ]
        P_d6[r+10,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 21772800.0, 127008000.0, 435456000.0, 1143072000.0, 2540160000.0, 5029516800.0 ]
        P_d6[r+11,c:c+sd["num_unknowns_per_segment"]] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 39916800.0, 239500800.0, 838252800.0, 2235340800.0, 5029516800.0, 10059033600.0 ]

    P = sd["lamb"][0]*P_d1 + sd["lamb"][1]*P_d2 + sd["lamb"][2]*P_d3 + sd["lamb"][3]*P_d4 + sd["lamb"][4]*P_d5 + sd["lamb"][5]*P_d6

    return P,q



def _compute_nonlocal_interpolating_b_spline_linear_system_degree_3(t,p,sd):

    A_num_rows    = sd["num_segments"]*sd["num_constraints_per_segment"] + sd["num_constraints_begin_spline"] + sd["num_constraints_end_spline"]
    A_num_columns = sd["num_segments"]*sd["num_unknowns_per_segment"]    + sd["num_knots"]*sd["num_unknowns_per_knot"]
    b_num_rows    = A_num_rows

    A = zeros((A_num_rows,A_num_columns))
    b = zeros((b_num_rows,1))

    for si in range(sd["num_segments"]):
        
        r              = si*sd["num_constraints_per_segment"]
        c_coeffs_begin = si*sd["num_unknowns_per_segment"]
        c_coeffs_end   = (si + 1)*sd["num_unknowns_per_segment"]

        A[r,  c_coeffs_begin:c_coeffs_end] = [ 1.0, 0.0, 0.0, 0.0 ]
        A[r+1,c_coeffs_begin:c_coeffs_end] = [ 1.0, 1.0, 1.0, 1.0 ]
        A[r+2,c_coeffs_begin:c_coeffs_end] = [ 0.0, 1.0, 0.0, 0.0 ]
        A[r+3,c_coeffs_begin:c_coeffs_end] = [ 0.0, 1.0, 2.0, 3.0 ]
        A[r+4,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 2.0, 0.0 ]
        A[r+5,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 2.0, 6.0 ]

        c_d1 = sd["num_segments"]*sd["num_unknowns_per_segment"] + (0*sd["num_knots"]) + si
        c_d2 = sd["num_segments"]*sd["num_unknowns_per_segment"] + (1*sd["num_knots"]) + si

        A[r+2,c_d1]   = -(t["segments_end"][si]-t["segments_begin"][si])**1
        A[r+3,c_d1+1] = -(t["segments_end"][si]-t["segments_begin"][si])**1
        A[r+4,c_d2]   = -(t["segments_end"][si]-t["segments_begin"][si])**2
        A[r+5,c_d2+1] = -(t["segments_end"][si]-t["segments_begin"][si])**2

        b[r]   = p["segments_begin"][si]
        b[r+1] = p["segments_end"][si]
        b[r+2] = 0.0
        b[r+3] = 0.0
        b[r+4] = 0.0
        b[r+5] = 0.0

    r                 = sd["num_segments"]*sd["num_constraints_per_segment"]
    c_d1_spline_begin = sd["num_segments"]*sd["num_unknowns_per_segment"] + (0*sd["num_knots"]) + 0
    c_d1_spline_end   = sd["num_segments"]*sd["num_unknowns_per_segment"] + (0*sd["num_knots"]) + sd["num_knots"] - 1

    A[r,  c_d1_spline_begin] = 1.0
    A[r+1,c_d1_spline_end]   = 1.0

    b[r]   = p["segments_end"][0]  - p["segments_begin"][0]
    b[r+1] = p["segments_end"][-1] - p["segments_begin"][-1]

    return A,b



def _compute_minimum_variation_nonlocal_interpolating_b_spline_linear_system_matrices(t,p,sd):

    A_num_columns = sd["num_segments"]*sd["num_unknowns_per_segment"] + sd["num_knots"]*sd["num_unknowns_per_knot"]

    #
    # polynomial line segments constraints
    #
    polynomial_segment_num_rows = sd["num_polynomial_segments"]*sd["num_constraints_per_polynomial_segment"]

    A_polynomial_segments = zeros((polynomial_segment_num_rows,A_num_columns))
    b_polynomial_segments = zeros((polynomial_segment_num_rows,1))

    #
    # straight line segments constraints
    #
    straight_line_segment_num_rows = sd["num_straight_line_segments"]*sd["num_constraints_per_straight_line_segment"]

    A_straight_line_segments = zeros((straight_line_segment_num_rows,A_num_columns))
    b_straight_line_segments = zeros((straight_line_segment_num_rows,1))

    #
    # zero derivative keyframes constraints
    #
    zero_derivative_keyframes_num_rows = sd["num_zero_derivative_keyframes"]

    A_zero_derivative_keyframes = zeros((zero_derivative_keyframes_num_rows,A_num_columns))
    b_zero_derivative_keyframes = zeros((zero_derivative_keyframes_num_rows,1))

    for zdki in range(sd["num_zero_derivative_keyframes"]):

        ki = sd["zero_derivative_keyframes"][zdki]

        r    = zdki
        c_d1 = sd["num_segments"]*sd["num_unknowns_per_segment"] + ki

        A_zero_derivative_keyframes[r,c_d1] = 1.0
        b_zero_derivative_keyframes[r]      = 0.0

    #
    # begin spline constraints
    #
    begin_spline_num_rows = sd["num_constraints_begin_spline"]

    A_begin_spline = zeros((begin_spline_num_rows,A_num_columns))
    b_begin_spline = zeros((begin_spline_num_rows,1))

    if sd["num_constraints_begin_spline"] > 0:
        c_d1_spline_begin                   = sd["num_segments"]*sd["num_unknowns_per_segment"] + (0*sd["num_knots"]) + 0
        A_begin_spline[0,c_d1_spline_begin] = 1.0
        b_begin_spline[0]                   = p["segments_end"][0] - p["segments_begin"][0]

    #
    # end spline constraints
    #
    end_spline_num_rows = sd["num_constraints_end_spline"]

    A_end_spline = zeros((end_spline_num_rows,A_num_columns))
    b_end_spline = zeros((end_spline_num_rows,1))

    if sd["num_constraints_end_spline"] > 0:
        c_d1_spline_end                 = sd["num_segments"]*sd["num_unknowns_per_segment"] + (1*sd["num_knots"]) - 1
        A_end_spline[0,c_d1_spline_end] = 1.0
        b_end_spline[0]                 = p["segments_end"][-1] - p["segments_begin"][-1]

    return A_polynomial_segments,b_polynomial_segments,A_straight_line_segments,b_straight_line_segments,A_zero_derivative_keyframes,b_zero_derivative_keyframes,A_begin_spline,b_begin_spline,A_end_spline,b_end_spline



def _compute_minimum_variation_nonlocal_interpolating_b_spline_full_rank_linear_system(A,b):

    assert A.shape[0] < A.shape[1]

    A_T         = A.T
    tol         = 1e-5
    Q, R        = linalg.qr(A_T.A)
    independent = np.where(np.abs(R.diagonal()) > tol)[0]

    A_T_independent = A_T[:,independent]
    A_independent   = A_T_independent.T
    b_independent   = b[independent]

    return A_independent, b_independent



def _compute_minimum_variation_nonlocal_interpolating_b_spline_linear_system_degree_5(t,p,sd):

    minimum_variation_nonlocal_interpolating_b_spline_linear_system_matrices = _compute_minimum_variation_nonlocal_interpolating_b_spline_linear_system_matrices(t,p,sd)
    A_polynomial_segments,b_polynomial_segments,A_straight_line_segments,b_straight_line_segments,A_zero_derivative_keyframes,b_zero_derivative_keyframes,A_begin_spline,b_begin_spline,A_end_spline,b_end_spline = minimum_variation_nonlocal_interpolating_b_spline_linear_system_matrices

    #
    # polynomial segment constraints
    #
    for psi in range(sd["num_polynomial_segments"]):
        
        si             = sd["polynomial_segments"][psi]
        r              = psi*sd["num_constraints_per_polynomial_segment"]
        c_coeffs_begin = si*sd["num_unknowns_per_segment"]
        c_coeffs_end   = (si + 1)*sd["num_unknowns_per_segment"]
        c_d1           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (0*sd["num_knots"]) + si
        c_d2           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (1*sd["num_knots"]) + si
        c_d3           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (2*sd["num_knots"]) + si
        c_d4           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (3*sd["num_knots"]) + si

        A_polynomial_segments[r,  c_coeffs_begin:c_coeffs_end] = [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_polynomial_segments[r+1,c_coeffs_begin:c_coeffs_end] = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]

        if sd["continuity"] >= 1:
            A_polynomial_segments[r+2,c_coeffs_begin:c_coeffs_end] = [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ]
            A_polynomial_segments[r+3,c_coeffs_begin:c_coeffs_end] = [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ]
        if sd["continuity"] >= 2:
            A_polynomial_segments[r+4,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 2.0, 0.0, 0.0, 0.0 ]
            A_polynomial_segments[r+5,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 2.0, 6.0, 12.0, 20.0 ]
        if sd["continuity"] >= 3:
            A_polynomial_segments[r+6,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 6.0, 0.0, 0.0 ]
            A_polynomial_segments[r+7,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 6.0, 24.0, 60.0 ]
        if sd["continuity"] >= 4:
            A_polynomial_segments[r+8,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 24.0, 0.0 ]
            A_polynomial_segments[r+9,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 ]

        if sd["continuity"] >= 1:
            A_polynomial_segments[r+2,c_d1]   = -(t["segments_end"][si]-t["segments_begin"][si])**1
            A_polynomial_segments[r+3,c_d1+1] = -(t["segments_end"][si]-t["segments_begin"][si])**1
        if sd["continuity"] >= 2:
            A_polynomial_segments[r+4,c_d2]   = -(t["segments_end"][si]-t["segments_begin"][si])**2
            A_polynomial_segments[r+5,c_d2+1] = -(t["segments_end"][si]-t["segments_begin"][si])**2
        if sd["continuity"] >= 3:
            A_polynomial_segments[r+6,c_d3]   = -(t["segments_end"][si]-t["segments_begin"][si])**3
            A_polynomial_segments[r+7,c_d3+1] = -(t["segments_end"][si]-t["segments_begin"][si])**3
        if sd["continuity"] >= 4:
            A_polynomial_segments[r+8,c_d4]   = -(t["segments_end"][si]-t["segments_begin"][si])**4
            A_polynomial_segments[r+9,c_d4+1] = -(t["segments_end"][si]-t["segments_begin"][si])**4

        b_polynomial_segments[r]   = p["segments_begin"][si]
        b_polynomial_segments[r+1] = p["segments_end"][si]

        if sd["continuity"] >= 1:
            b_polynomial_segments[r+2] = 0.0
            b_polynomial_segments[r+3] = 0.0
        if sd["continuity"] >= 2:
            b_polynomial_segments[r+4] = 0.0
            b_polynomial_segments[r+5] = 0.0
        if sd["continuity"] >= 3:
            b_polynomial_segments[r+6] = 0.0
            b_polynomial_segments[r+7] = 0.0
        if sd["continuity"] >= 4:
            b_polynomial_segments[r+8] = 0.0
            b_polynomial_segments[r+9] = 0.0

    #
    # straight line segment constraints
    #
    for slsi in range(sd["num_straight_line_segments"]):
        
        si             = sd["straight_line_segments"][slsi]
        r              = slsi*sd["num_constraints_per_straight_line_segment"]
        c_coeffs_begin = si*sd["num_unknowns_per_segment"]
        c_coeffs_end   = (si + 1)*sd["num_unknowns_per_segment"]
        c_d1           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (0*sd["num_knots"]) + si
        c_d2           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (1*sd["num_knots"]) + si
        c_d3           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (2*sd["num_knots"]) + si
        c_d4           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (3*sd["num_knots"]) + si

        A_straight_line_segments[r,  c_coeffs_begin:c_coeffs_end] = [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+1,c_coeffs_begin:c_coeffs_end] = [ 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+2,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+3,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ]
        A_straight_line_segments[r+4,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ]
        A_straight_line_segments[r+5,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]

        if sd["continuity"] >= 1:
            A_straight_line_segments[r+6,c_d1]   = 1.0
            A_straight_line_segments[r+7,c_d1+1] = 1.0
        if sd["continuity"] >= 2:
            A_straight_line_segments[r+8,c_d2]   = 1.0
            A_straight_line_segments[r+9,c_d2+1] = 1.0
        if sd["continuity"] >= 3:
            A_straight_line_segments[r+10,c_d3]   = 1.0
            A_straight_line_segments[r+11,c_d3+1] = 1.0
        if sd["continuity"] >= 4:
            A_straight_line_segments[r+12,c_d4]   = 1.0
            A_straight_line_segments[r+13,c_d4+1] = 1.0

        b_straight_line_segments[r]   = p["segments_begin"][si]
        b_straight_line_segments[r+1] = p["segments_end"][si]
        b_straight_line_segments[r+2] = 0.0
        b_straight_line_segments[r+3] = 0.0
        b_straight_line_segments[r+4] = 0.0
        b_straight_line_segments[r+5] = 0.0

        if sd["continuity"] >= 1:
            b_straight_line_segments[r+6] = (p["segments_end"][si]-p["segments_begin"][si]) / (t["segments_end"][si]-t["segments_begin"][si])
            b_straight_line_segments[r+7] = (p["segments_end"][si]-p["segments_begin"][si]) / (t["segments_end"][si]-t["segments_begin"][si])
        if sd["continuity"] >= 2:
            b_straight_line_segments[r+8] = 0.0
            b_straight_line_segments[r+9] = 0.0
        if sd["continuity"] >= 3:
            b_straight_line_segments[r+10] = 0.0
            b_straight_line_segments[r+11] = 0.0
        if sd["continuity"] >= 4:
            b_straight_line_segments[r+12] = 0.0
            b_straight_line_segments[r+13] = 0.0

    #
    # concatenate all constraints
    #
    A = bmat([[A_polynomial_segments],[A_straight_line_segments],[A_zero_derivative_keyframes],[A_begin_spline],[A_end_spline]])
    b = bmat([[b_polynomial_segments],[b_straight_line_segments],[b_zero_derivative_keyframes],[b_begin_spline],[b_end_spline]])

    A,b = _compute_minimum_variation_nonlocal_interpolating_b_spline_full_rank_linear_system(A,b)

    return A,b



def _compute_minimum_variation_nonlocal_interpolating_b_spline_linear_system_degree_7(t,p,sd):

    minimum_variation_nonlocal_interpolating_b_spline_linear_system_matrices = _compute_minimum_variation_nonlocal_interpolating_b_spline_linear_system_matrices(t,p,sd)
    A_polynomial_segments,b_polynomial_segments,A_straight_line_segments,b_straight_line_segments,A_zero_derivative_keyframes,b_zero_derivative_keyframes,A_begin_spline,b_begin_spline,A_end_spline,b_end_spline = minimum_variation_nonlocal_interpolating_b_spline_linear_system_matrices

    #
    # polynomial segment constraints
    #
    for psi in range(sd["num_polynomial_segments"]):
        
        si             = sd["polynomial_segments"][psi]
        r              = psi*sd["num_constraints_per_polynomial_segment"]
        c_coeffs_begin = si*sd["num_unknowns_per_segment"]
        c_coeffs_end   = (si + 1)*sd["num_unknowns_per_segment"]
        c_d1           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (0*sd["num_knots"]) + si
        c_d2           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (1*sd["num_knots"]) + si
        c_d3           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (2*sd["num_knots"]) + si
        c_d4           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (3*sd["num_knots"]) + si

        A_polynomial_segments[r,  c_coeffs_begin:c_coeffs_end] = [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_polynomial_segments[r+1,c_coeffs_begin:c_coeffs_end] = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]

        if sd["continuity"] >= 1:
            A_polynomial_segments[r+2,c_coeffs_begin:c_coeffs_end] = [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
            A_polynomial_segments[r+3,c_coeffs_begin:c_coeffs_end] = [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 ]
        if sd["continuity"] >= 2:
            A_polynomial_segments[r+4,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
            A_polynomial_segments[r+5,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 2.0, 6.0, 12.0, 20.0, 30.0, 42.0 ]
        if sd["continuity"] >= 3:
            A_polynomial_segments[r+6,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0 ]
            A_polynomial_segments[r+7,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 6.0, 24.0, 60.0, 120.0, 210.0 ]
        if sd["continuity"] >= 4:
            A_polynomial_segments[r+8,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 24.0, 0.0, 0.0, 0.0 ]
            A_polynomial_segments[r+9,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 24.0, 120.0, 360.0, 840.0 ]

        if sd["continuity"] >= 1:
            A_polynomial_segments[r+2,c_d1]   = -(t["segments_end"][si]-t["segments_begin"][si])**1
            A_polynomial_segments[r+3,c_d1+1] = -(t["segments_end"][si]-t["segments_begin"][si])**1
        if sd["continuity"] >= 2:
            A_polynomial_segments[r+4,c_d2]   = -(t["segments_end"][si]-t["segments_begin"][si])**2
            A_polynomial_segments[r+5,c_d2+1] = -(t["segments_end"][si]-t["segments_begin"][si])**2
        if sd["continuity"] >= 3:
            A_polynomial_segments[r+6,c_d3]   = -(t["segments_end"][si]-t["segments_begin"][si])**3
            A_polynomial_segments[r+7,c_d3+1] = -(t["segments_end"][si]-t["segments_begin"][si])**3
        if sd["continuity"] >= 4:
            A_polynomial_segments[r+8,c_d4]   = -(t["segments_end"][si]-t["segments_begin"][si])**4
            A_polynomial_segments[r+9,c_d4+1] = -(t["segments_end"][si]-t["segments_begin"][si])**4

        b_polynomial_segments[r]   = p["segments_begin"][si]
        b_polynomial_segments[r+1] = p["segments_end"][si]

        if sd["continuity"] >= 1:
            b_polynomial_segments[r+2] = 0.0
            b_polynomial_segments[r+3] = 0.0
        if sd["continuity"] >= 2:
            b_polynomial_segments[r+4] = 0.0
            b_polynomial_segments[r+5] = 0.0
        if sd["continuity"] >= 3:
            b_polynomial_segments[r+6] = 0.0
            b_polynomial_segments[r+7] = 0.0
        if sd["continuity"] >= 4:
            b_polynomial_segments[r+8] = 0.0
            b_polynomial_segments[r+9] = 0.0

    #
    # straight line segment constraints
    #
    for slsi in range(sd["num_straight_line_segments"]):
        
        si             = sd["straight_line_segments"][slsi]
        r              = slsi*sd["num_constraints_per_straight_line_segment"]
        c_coeffs_begin = si*sd["num_unknowns_per_segment"]
        c_coeffs_end   = (si + 1)*sd["num_unknowns_per_segment"]
        c_d1           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (0*sd["num_knots"]) + si
        c_d2           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (1*sd["num_knots"]) + si
        c_d3           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (2*sd["num_knots"]) + si
        c_d4           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (3*sd["num_knots"]) + si

        A_straight_line_segments[r,  c_coeffs_begin:c_coeffs_end] = [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+1,c_coeffs_begin:c_coeffs_end] = [ 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+2,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+3,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+4,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+5,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ]
        A_straight_line_segments[r+6,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ]
        A_straight_line_segments[r+7,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]

        if sd["continuity"] >= 1:
            A_straight_line_segments[r+8,c_d1]   = 1.0
            A_straight_line_segments[r+9,c_d1+1] = 1.0
        if sd["continuity"] >= 2:
            A_straight_line_segments[r+10,c_d2]   = 1.0
            A_straight_line_segments[r+11,c_d2+1] = 1.0
        if sd["continuity"] >= 3:
            A_straight_line_segments[r+12,c_d3]   = 1.0
            A_straight_line_segments[r+13,c_d3+1] = 1.0
        if sd["continuity"] >= 4:
            A_straight_line_segments[r+14,c_d4]   = 1.0
            A_straight_line_segments[r+15,c_d4+1] = 1.0

        b_straight_line_segments[r]   = p["segments_begin"][si]
        b_straight_line_segments[r+1] = p["segments_end"][si]
        b_straight_line_segments[r+2] = 0.0
        b_straight_line_segments[r+3] = 0.0
        b_straight_line_segments[r+4] = 0.0
        b_straight_line_segments[r+5] = 0.0
        b_straight_line_segments[r+6] = 0.0
        b_straight_line_segments[r+7] = 0.0

        if sd["continuity"] >= 1:
            b_straight_line_segments[r+8] = (p["segments_end"][si]-p["segments_begin"][si]) / (t["segments_end"][si]-t["segments_begin"][si])
            b_straight_line_segments[r+9] = (p["segments_end"][si]-p["segments_begin"][si]) / (t["segments_end"][si]-t["segments_begin"][si])
        if sd["continuity"] >= 2:
            b_straight_line_segments[r+10] = 0.0
            b_straight_line_segments[r+11] = 0.0
        if sd["continuity"] >= 3:
            b_straight_line_segments[r+12] = 0.0
            b_straight_line_segments[r+13] = 0.0
        if sd["continuity"] >= 4:
            b_straight_line_segments[r+14] = 0.0
            b_straight_line_segments[r+15] = 0.0

    #
    # concatenate all constraints
    #
    A = bmat([[A_polynomial_segments],[A_straight_line_segments],[A_zero_derivative_keyframes],[A_begin_spline],[A_end_spline]])
    b = bmat([[b_polynomial_segments],[b_straight_line_segments],[b_zero_derivative_keyframes],[b_begin_spline],[b_end_spline]])

    A,b = _compute_minimum_variation_nonlocal_interpolating_b_spline_full_rank_linear_system(A,b)

    return A,b



def _compute_minimum_variation_nonlocal_interpolating_b_spline_linear_system_degree_9(t,p,sd):

    minimum_variation_nonlocal_interpolating_b_spline_linear_system_matrices = _compute_minimum_variation_nonlocal_interpolating_b_spline_linear_system_matrices(t,p,sd)
    A_polynomial_segments,b_polynomial_segments,A_straight_line_segments,b_straight_line_segments,A_zero_derivative_keyframes,b_zero_derivative_keyframes,A_begin_spline,b_begin_spline,A_end_spline,b_end_spline = minimum_variation_nonlocal_interpolating_b_spline_linear_system_matrices

    #
    # polynomial segment constraints
    #
    for psi in range(sd["num_polynomial_segments"]):
        
        si             = sd["polynomial_segments"][psi]
        r              = psi*sd["num_constraints_per_polynomial_segment"]
        c_coeffs_begin = si*sd["num_unknowns_per_segment"]
        c_coeffs_end   = (si + 1)*sd["num_unknowns_per_segment"]
        c_d1           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (0*sd["num_knots"]) + si
        c_d2           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (1*sd["num_knots"]) + si
        c_d3           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (2*sd["num_knots"]) + si
        c_d4           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (3*sd["num_knots"]) + si

        A_polynomial_segments[r,  c_coeffs_begin:c_coeffs_end] = [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_polynomial_segments[r+1,c_coeffs_begin:c_coeffs_end] = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]

        if sd["continuity"] >= 1:
            A_polynomial_segments[r+2,c_coeffs_begin:c_coeffs_end] = [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
            A_polynomial_segments[r+3,c_coeffs_begin:c_coeffs_end] = [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
        if sd["continuity"] >= 2:
            A_polynomial_segments[r+4,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
            A_polynomial_segments[r+5,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0 ]
        if sd["continuity"] >= 3:
            A_polynomial_segments[r+6,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
            A_polynomial_segments[r+7,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 6.0, 24.0, 60.0, 120.0, 210.0, 336.0, 504.0 ]
        if sd["continuity"] >= 4:
            A_polynomial_segments[r+8,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
            A_polynomial_segments[r+9,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 24.0, 120.0, 360.0, 840.0, 1680.0, 3024.0 ]

        if sd["continuity"] >= 1:
            A_polynomial_segments[r+2,c_d1]   = -(t["segments_end"][si]-t["segments_begin"][si])**1
            A_polynomial_segments[r+3,c_d1+1] = -(t["segments_end"][si]-t["segments_begin"][si])**1
        if sd["continuity"] >= 2:
            A_polynomial_segments[r+4,c_d2]   = -(t["segments_end"][si]-t["segments_begin"][si])**2
            A_polynomial_segments[r+5,c_d2+1] = -(t["segments_end"][si]-t["segments_begin"][si])**2
        if sd["continuity"] >= 3:
            A_polynomial_segments[r+6,c_d3]   = -(t["segments_end"][si]-t["segments_begin"][si])**3
            A_polynomial_segments[r+7,c_d3+1] = -(t["segments_end"][si]-t["segments_begin"][si])**3
        if sd["continuity"] >= 4:
            A_polynomial_segments[r+8,c_d4]   = -(t["segments_end"][si]-t["segments_begin"][si])**4
            A_polynomial_segments[r+9,c_d4+1] = -(t["segments_end"][si]-t["segments_begin"][si])**4

        b_polynomial_segments[r]   = p["segments_begin"][si]
        b_polynomial_segments[r+1] = p["segments_end"][si]

        if sd["continuity"] >= 1:
            b_polynomial_segments[r+2] = 0
            b_polynomial_segments[r+3] = 0
        if sd["continuity"] >= 2:
            b_polynomial_segments[r+4] = 0
            b_polynomial_segments[r+5] = 0
        if sd["continuity"] >= 3:
            b_polynomial_segments[r+6] = 0
            b_polynomial_segments[r+7] = 0
        if sd["continuity"] >= 4:
            b_polynomial_segments[r+8] = 0
            b_polynomial_segments[r+9] = 0

    #
    # straight line segment constraints
    #
    for slsi in range(sd["num_straight_line_segments"]):
        
        si             = sd["straight_line_segments"][slsi]
        r              = slsi*sd["num_constraints_per_straight_line_segment"]
        c_coeffs_begin = si*sd["num_unknowns_per_segment"]
        c_coeffs_end   = (si + 1)*sd["num_unknowns_per_segment"]
        c_d1           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (0*sd["num_knots"]) + si
        c_d2           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (1*sd["num_knots"]) + si
        c_d3           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (2*sd["num_knots"]) + si
        c_d4           = sd["num_segments"]*sd["num_unknowns_per_segment"] + (3*sd["num_knots"]) + si

        A_straight_line_segments[r,  c_coeffs_begin:c_coeffs_end] = [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+1,c_coeffs_begin:c_coeffs_end] = [ 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+2,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+3,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+4,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+5,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+6,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ]
        A_straight_line_segments[r+7,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ]
        A_straight_line_segments[r+8,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ]
        A_straight_line_segments[r+9,c_coeffs_begin:c_coeffs_end] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]

        if sd["continuity"] >= 1:
            A_straight_line_segments[r+10,c_d1]   = 1.0
            A_straight_line_segments[r+11,c_d1+1] = 1.0
        if sd["continuity"] >= 2:
            A_straight_line_segments[r+12,c_d2]   = 1.0
            A_straight_line_segments[r+13,c_d2+1] = 1.0
        if sd["continuity"] >= 3:
            A_straight_line_segments[r+14,c_d3]   = 1.0
            A_straight_line_segments[r+15,c_d3+1] = 1.0
        if sd["continuity"] >= 4:
            A_straight_line_segments[r+16,c_d4]   = 1.0
            A_straight_line_segments[r+17,c_d4+1] = 1.0

        b_straight_line_segments[r]   = p["segments_begin"][si]
        b_straight_line_segments[r+1] = p["segments_end"][si]
        b_straight_line_segments[r+2] = 0.0
        b_straight_line_segments[r+3] = 0.0
        b_straight_line_segments[r+4] = 0.0
        b_straight_line_segments[r+5] = 0.0
        b_straight_line_segments[r+6] = 0.0
        b_straight_line_segments[r+7] = 0.0
        b_straight_line_segments[r+8] = 0.0
        b_straight_line_segments[r+9] = 0.0

        if sd["continuity"] >= 1:
            b_straight_line_segments[r+10] = (p["segments_end"][si]-p["segments_begin"][si]) / (t["segments_end"][si]-t["segments_begin"][si])
            b_straight_line_segments[r+11] = (p["segments_end"][si]-p["segments_begin"][si]) / (t["segments_end"][si]-t["segments_begin"][si])
        if sd["continuity"] >= 2:
            b_straight_line_segments[r+12] = 0.0
            b_straight_line_segments[r+13] = 0.0
        if sd["continuity"] >= 3:
            b_straight_line_segments[r+14] = 0.0
            b_straight_line_segments[r+15] = 0.0
        if sd["continuity"] >= 4:
            b_straight_line_segments[r+16] = 0.0
            b_straight_line_segments[r+17] = 0.0

    #
    # concatenate all constraints
    #
    A = bmat([[A_polynomial_segments],[A_straight_line_segments],[A_zero_derivative_keyframes],[A_begin_spline],[A_end_spline]])
    b = bmat([[b_polynomial_segments],[b_straight_line_segments],[b_zero_derivative_keyframes],[b_begin_spline],[b_end_spline]])

    A,b = _compute_minimum_variation_nonlocal_interpolating_b_spline_full_rank_linear_system(A,b)

    return A,b



def _compute_trigonometric_spline_linear_system_order_2(t,p,pd1,sd):

    A = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))
    b = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],1))

    t_begin = 0
    t_end   = pi/4

    for si in range(sd["num_segments"]):
        
        r                                         = si*sd["num_constraints_per_segment"]
        c                                         = si*sd["num_unknowns_per_segment"]
        A[r,  c:c+sd["num_unknowns_per_segment"]] = [1, cos(t_begin),  cos(2*t_begin),    sin(t_begin)]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [1, cos(t_end),    cos(2*t_end),      sin(t_end)]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [0, -sin(t_begin), -2*sin(2*t_begin), cos(t_begin)]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [0, -sin(t_end),   -2*sin(2*t_end),   cos(t_end)]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = pd1["segments_begin"][si]
        b[r+3]                                    = pd1["segments_end"][si]

    return A,b



def _compute_trigonometric_spline_linear_system_order_5(t,p,pd1,pd2,pd3,pd4,sd):

    A = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],sd["num_segments"]*sd["num_unknowns_per_segment"]))
    b = zeros((sd["num_segments"]*sd["num_constraints_per_segment"],1))

    t_begin = 0
    t_end   = pi/4

    for si in range(sd["num_segments"]):
        
        r                                         = si*sd["num_constraints_per_segment"]
        c                                         = si*sd["num_unknowns_per_segment"]

        A[r,  c:c+sd["num_unknowns_per_segment"]] = [1, cos(t_begin),    cos(2*t_begin),    cos(3*t_begin),    cos(4*t_begin),    cos(5*t_begin), sin(t_begin),    sin(2*t_begin),    sin(3*t_begin),    sin(4*t_begin)]
        A[r+1,c:c+sd["num_unknowns_per_segment"]] = [1, cos(t_end),      cos(2*t_end),      cos(3*t_end),      cos(4*t_end),      cos(5*t_end),   sin(t_end),      sin(2*t_end),      sin(3*t_end),      sin(4*t_end)]
        A[r+2,c:c+sd["num_unknowns_per_segment"]] = [0,-sin(t_begin), -2*sin(2*t_begin), -3*sin(3*t_begin), -4*sin(4*t_begin), -5*sin(5*t_begin), cos(t_begin),  2*cos(2*t_begin),  3*cos(3*t_begin),  4*cos(4*t_begin)]
        A[r+3,c:c+sd["num_unknowns_per_segment"]] = [0,-sin(t_end),   -2*sin(2*t_end),   -3*sin(3*t_end),   -4*sin(4*t_end),   -5*sin(5*t_end),   cos(t_end),    2*cos(2*t_end),    3*cos(3*t_end),    4*cos(4*t_end)]
        A[r+4,c:c+sd["num_unknowns_per_segment"]] = [0,-cos(t_begin), -4*cos(2*t_begin), -9*cos(3*t_begin),-16*cos(4*t_begin),-25*cos(5*t_begin),-sin(t_begin), -4*sin(2*t_begin), -9*sin(3*t_begin),-16*sin(4*t_begin)]
        A[r+5,c:c+sd["num_unknowns_per_segment"]] = [0,-cos(t_end),   -4*cos(2*t_end),   -9*cos(3*t_end),  -16*cos(4*t_end),  -25*cos(5*t_end),  -sin(t_end),   -4*sin(2*t_end),   -9*sin(3*t_end),  -16*sin(4*t_end)]
        A[r+6,c:c+sd["num_unknowns_per_segment"]] = [0, sin(t_begin),  8*sin(2*t_begin), 27*sin(3*t_begin), 64*sin(4*t_begin),125*sin(5*t_begin),-cos(t_begin), -8*cos(2*t_begin),-27*cos(3*t_begin),-64*cos(4*t_begin)]
        A[r+7,c:c+sd["num_unknowns_per_segment"]] = [0, sin(t_end),    8*sin(2*t_end),   27*sin(3*t_end),   64*sin(4*t_end),  125*sin(5*t_end),  -cos(t_end),   -8*cos(2*t_end),  -27*cos(3*t_end),  -64*cos(4*t_end)]
        A[r+8,c:c+sd["num_unknowns_per_segment"]] = [0, cos(t_begin), 16*cos(2*t_begin), 81*cos(3*t_begin),256*cos(4*t_begin),625*cos(5*t_begin), sin(t_begin), 16*sin(2*t_begin), 81*sin(3*t_begin),256*sin(4*t_begin)]
        A[r+9,c:c+sd["num_unknowns_per_segment"]] = [0, cos(t_end),   16*cos(2*t_end),   81*cos(3*t_end),  256*cos(4*t_end),  625*cos(5*t_end),   sin(t_end),   16*sin(2*t_end),   81*sin(3*t_end),  256*sin(4*t_end)]
        b[r]                                      = p["segments_begin"][si]
        b[r+1]                                    = p["segments_end"][si]
        b[r+2]                                    = pd1["segments_begin"][si]
        b[r+3]                                    = pd1["segments_end"][si]
        b[r+4]                                    = pd2["segments_begin"][si]
        b[r+5]                                    = pd2["segments_end"][si]
        b[r+6]                                    = pd3["segments_begin"][si]
        b[r+7]                                    = pd3["segments_end"][si]
        b[r+8]                                    = pd4["segments_begin"][si]
        b[r+9]                                    = pd4["segments_end"][si]

    return A,b



def _evaluate_spline(C,T,sd,eval_scalar_func,T_eval,num_samples):

    C = C.astype(float64)
    T = T.astype(float64)

    if num_samples is None:
        assert T_eval is not None

    if T_eval is None:
        assert num_samples is not None

    if T_eval is None:
        T_eval = empty((num_samples,0))
        for di in range(sd["num_dimensions"]):
            t_eval = linspace(T[0,di],T[-1,di],num_samples).reshape(-1,1)
            T_eval = c_[T_eval,t_eval]

    dT = empty((1,0))
    for di in range(sd["num_dimensions"]):
        dt = float(T[-1,di] - T[0,di])/T_eval.shape[0]
        dT = c_[dT,dt]

    P = zeros_like(T_eval)

    for di in range(sd["num_dimensions"]):
        P[:,di] = eval_scalar_func(T_eval[:,di],C[:,di],T[:,di],sd)

    return P,T_eval,dT



def _evaluate_polynomial_spline_scalar(t_eval,c,t_keyframes,sd):

    p_eval = zeros_like(t_eval)
    si     = 0

    for ti in range(t_eval.shape[0]):

        t_eval_val = t_eval[ti]
        
        while si+1 < sd["num_segments"] and t_eval_val >= t_keyframes[si+1]:
            si += 1

        ci             = si*sd["num_unknowns_per_segment"]
        t_eval_bar_val = float(t_eval_val-t_keyframes[si])/(t_keyframes[si+1]-t_keyframes[si])

        if sd["degree"] == 3:
            t_eval_bar_pows = matrix([t_eval_bar_val**0,t_eval_bar_val**1,t_eval_bar_val**2,t_eval_bar_val**3])
        if sd["degree"] == 5:
            t_eval_bar_pows = matrix([t_eval_bar_val**0,t_eval_bar_val**1,t_eval_bar_val**2,t_eval_bar_val**3,t_eval_bar_val**4,t_eval_bar_val**5])
        if sd["degree"] == 7:
            t_eval_bar_pows = matrix([t_eval_bar_val**0,t_eval_bar_val**1,t_eval_bar_val**2,t_eval_bar_val**3,t_eval_bar_val**4,t_eval_bar_val**5,t_eval_bar_val**6,t_eval_bar_val**7])
        if sd["degree"] == 9:
            t_eval_bar_pows = matrix([t_eval_bar_val**0,t_eval_bar_val**1,t_eval_bar_val**2,t_eval_bar_val**3,t_eval_bar_val**4,t_eval_bar_val**5,t_eval_bar_val**6,t_eval_bar_val**7,t_eval_bar_val**8,t_eval_bar_val**9])
        if sd["degree"] == 11:
            t_eval_bar_pows = matrix([t_eval_bar_val**0,t_eval_bar_val**1,t_eval_bar_val**2,t_eval_bar_val**3,t_eval_bar_val**4,t_eval_bar_val**5,t_eval_bar_val**6,t_eval_bar_val**7,t_eval_bar_val**8,t_eval_bar_val**9,t_eval_bar_val**10,t_eval_bar_val**11])

        c_segment = matrix(c[ci:ci+sd["num_unknowns_per_segment"]]).T
        
        p_eval[ti] = t_eval_bar_pows*c_segment

    return p_eval



def _evaluate_trigonometric_spline_scalar(t_eval,c,t_keyframes,sd):

    p_eval = zeros_like(t_eval)
    si     = 0

    for ti in range(t_eval.shape[0]):

        t_eval_val = t_eval[ti]

        while si+1 < sd["num_segments"] and t_eval_val >= t_keyframes[si+1]:
            si += 1

        ci             = si*sd["num_unknowns_per_segment"]
        t_eval_bar_val = (pi/4)*float(t_eval_val-t_keyframes[si])/(t_keyframes[si+1]-t_keyframes[si])

        if sd["order"] == 2:
            t_eval_bar_trig_funcs = matrix([1, cos(t_eval_bar_val), cos(2*t_eval_bar_val), sin(t_eval_bar_val)])
        if sd["order"] == 5:
            t_eval_bar_trig_funcs = matrix([1, cos(t_eval_bar_val), cos(2*t_eval_bar_val), cos(3*t_eval_bar_val), cos(4*t_eval_bar_val), cos(5*t_eval_bar_val), sin(t_eval_bar_val), sin(2*t_eval_bar_val), sin(3*t_eval_bar_val), sin(4*t_eval_bar_val)])

        c_segment = matrix(c[ci:ci+sd["num_unknowns_per_segment"]]).T
        
        p_eval[ti] = t_eval_bar_trig_funcs*c_segment

    return p_eval