from pylab import *

import os
import re
import subprocess
import sympy
import sympy.utilities.autowrap
# import sympy.printing.theanocode
# import theano.compile.function_module



#
# constructing matrices
#

def construct_matrix_and_entries(prefix_name,shape,real=True):
    
    A = sympy.Matrix.zeros(shape[0],shape[1])
    for r in range(A.rows):
        for c in range(A.cols):
            a      = sympy.Symbol("%s_%s_%s"%(prefix_name,r,c),real=real)
            A[r,c] = a
    return A, matrix(A)

def construct_matrix_from_block_matrix(A_expr):

    A_expr           = sympy.BlockMatrix(A_expr)
    A_collapsed_expr = sympy.Matrix.zeros(A_expr.rows,A_expr.cols)
    for r in range(A_expr.rows):
        for c in range(A_expr.cols):
            A_collapsed_expr[r,c] = A_expr[r,c]
    return A_collapsed_expr

    # return sympy.block_collapse(A_expr)

def construct_cross_product_left_term_matrix_from_vector(a_expr):

    return sympy.Matrix([[0,-a_expr[2],a_expr[1]],[a_expr[2],0,-a_expr[0]],[-a_expr[1],a_expr[0],0]])

def construct_axis_aligned_rotation_matrix_right_handed(angle_expr,axis):

    assert axis in [0,1,2]
    if axis == 0:
        return sympy.Matrix( [ [1, 0, 0], [0, sympy.cos(angle_expr), -sympy.sin(angle_expr)], [0, sympy.sin(angle_expr), sympy.cos(angle_expr)] ] )
    if axis == 1:
        return sympy.Matrix( [ [sympy.cos(angle_expr), 0, sympy.sin(angle_expr)], [0, 1, 0], [-sympy.sin(angle_expr), 0, sympy.cos(angle_expr)] ] )
    if axis == 2:
        return sympy.Matrix( [ [sympy.cos(angle_expr), -sympy.sin(angle_expr), 0], [sympy.sin(angle_expr), sympy.cos(angle_expr), 0], [0, 0, 1] ] )



#
# numpy compatibility functions
#

def sum(A_expr,axis=None):

    assert axis is None or axis == 0 or axis == 1
    if axis is None:
        A_sum_expr = sympy.Add(*A_expr)
        return A_sum_expr
    if axis == 0:
        A_sum_expr = sympy.Matrix.zeros(1,A_expr.cols)
        for c in range(A_expr.cols):
            A_sum_expr[0,c] = sympy.Add(*A_expr[:,c])
        return A_sum_expr
    if axis == 1:
        A_sum_expr = sympy.Matrix.zeros(A_expr.rows,1)
        for r in range(A_expr.rows):
            A_sum_expr[r,0] = sympy.Add(*A_expr[r,:])
        return A_sum_expr

def square(A_expr):

    if not isinstance(A_expr,sympy.Matrix):
        A_square_expr = A_expr**2
    else:
        A_square_expr = sympy.Matrix.zeros(A_expr.rows,A_expr.cols)
        for r in range(A_expr.rows):
            for c in range(A_expr.cols):
                A_square_expr[r,c] = A_expr[r,c]**2
    return A_square_expr

def norm(A_expr,axis=None,ord=2):

    assert axis is None or axis == 0 or axis == 1
    if axis is None:
        if not A_expr.is_Matrix:
            return sympy.Abs(A_expr)
        assert A_expr.cols == 1 or A_expr.rows == 1
        if A_expr.cols == 1 and A_expr.rows == 1:
            return sympy.Abs(A_expr)
        if A_expr.cols == 1:
            return sympy.root( sympy.Add( *[ a_expr**ord for a_expr in A_expr[:,0] ] ), ord )
        if A_expr.rows == 1:
            return sympy.root( sympy.Add( *[ a_expr**ord for a_expr in A_expr[0,:] ] ), ord )
    if axis == 0:
        A_norm_expr = sympy.Matrix.zeros(1,A_expr.cols)
        for c in range(A_expr.cols):
            A_norm_expr[0,c] = sympy.root( sympy.Add( *[ a_expr**ord for a_expr in A_expr[:,c] ] ), ord )
        return A_norm_expr
    if axis == 1:
        A_norm_expr = sympy.Matrix.zeros(A_expr.rows,1)
        for r in range(A_expr.rows):
            A_norm_expr[r,0] = sympy.root( sympy.Add( *[ a_expr**ord for a_expr in A_expr[r,:] ] ), ord )
        return A_norm_expr
    assert False
    return None

def arctan2(y_expr,x_expr):

    return sympy.functions.elementary.trigonometric.atan2(y_expr,x_expr)

def pinv(A_expr):

    a_wild_expr = sympy.Wild("a_wild")
    assert A_expr == A_expr.replace(sympy.conjugate(a_wild_expr),a_wild_expr)
    A_pinv_expr = A_expr.pinv()
    A_pinv_expr = A_pinv_expr.replace(sympy.conjugate(a_wild_expr),a_wild_expr)
    return A_pinv_expr

def ravel(A_expr,order="C"):

    return sympy.Matrix(array(A_expr).ravel(order))



#
# sklearn compatibility functions
#

def normalize(A_expr,axis=0,ord=2):

    assert axis == 0 or axis == 1
    if A_expr.cols == 1 or A_expr.rows == 1:
        return A_expr / norm(A_expr,ord=ord)
    if axis == 0:
        A_normalized_expr = sympy.Matrix.zeros(1,A_expr.cols)
        for c in range(A_expr.cols):
            A_normalized_expr[:,c] = A_expr[:,c] / norm(A_expr[:,c],ord=ord)
        return A_normalized_expr
    if axis == 1:
        A_norm_expr = sympy.Matrix.zeros(A_expr.rows,1)
        for r in range(A_expr.rows):
            A_normalized_expr[r,:] = A_expr[r,:] / norm(A_expr[r,:],ord=ord)
        return A_normalized_expr



#
# transformations compatibility functions
#

def euler_from_matrix(A_expr,axes="sxyz"):

    import transformations

    try:
        firstaxis, parity, repetition, frame = transformations._AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        transformations._TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = transformations._NEXT_AXIS[i+parity]
    k = transformations._NEXT_AXIS[i-parity+1]

    M = A_expr[0:3,0:3]

    if repetition:

        # sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        # if sy > _EPS:
        #     ax = math.atan2( M[i, j],  M[i, k])
        #     ay = math.atan2( sy,       M[i, i])
        #     az = math.atan2( M[j, i], -M[k, i])
        # else:
        #     ax = math.atan2(-M[j, k],  M[j, j])
        #     ay = math.atan2( sy,       M[i, i])
        #     az = 0.0
        
        sy = sympy.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])

        ax = sympy.Piecewise( ( arctan2(  M[i, j],  M[i, k] ), sy > transformations._EPS ), \
                              ( arctan2( -M[j, k],  M[j, j] ), True                      )  )

        ay = sympy.Piecewise( ( arctan2(  sy,       M[i, i] ), sy > transformations._EPS ), \
                              ( arctan2(  sy,       M[i, i] ), True                      )  )

        az = sympy.Piecewise( ( arctan2(  M[j, i], -M[k, i] ), sy > transformations._EPS ), \
                              ( 0,                             True                      )  )

    else:

        # cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        # if cy > _EPS:
        #     ax = math.atan2( M[k, j],  M[k, k])
        #     ay = math.atan2(-M[k, i],  cy)
        #     az = math.atan2( M[j, i],  M[i, i])
        # else:
        #     ax = math.atan2(-M[j, k],  M[j, j])
        #     ay = math.atan2(-M[k, i],  cy)
        #     az = 0.0

        cy = sympy.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])

        ax = sympy.Piecewise( ( arctan2(  M[k, j],  M[k, k] ), cy > transformations._EPS ), \
                              ( arctan2( -M[j, k],  M[j, j] ), True                      )  )

        ay = sympy.Piecewise( ( arctan2( -M[k, i],  cy      ), cy > transformations._EPS ), \
                              ( arctan2( -M[k, i],  cy      ), True                      )  )

        az = sympy.Piecewise( ( arctan2(  M[j, i],  M[i, i] ), cy > transformations._EPS ), \
                              ( 0,                             True                      )  )

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az



#
# computing derivative expressions
#

def diff_scalar_wrt_vector(a_expr,b_expr,shape="column"):

    assert shape == "column" or shape == "row"
    if shape == "column":
        return sympy.Matrix([[a_expr]]).jacobian(b_expr).transpose()
    if shape == "row":
        return sympy.Matrix([[a_expr]]).jacobian(b_expr)

def diff_scalar_using_chain_rule(f_expr,g_expr,h_expr,ord,dNf_dgN_subs,dNg_dhN_subs):

    dNf_dgN_subs = dNf_dgN_subs[0:ord+1]
    dNg_dhN_subs = dNg_dhN_subs[0:ord+1]

    dNf_dgN_expr      = sympy.Matrix.zeros(ord+1,1)
    dNf_dgN_expr[0] = f_expr
    dNf_dgN_expr[1] = f_expr.diff(g_expr)
    for i in range(2,ord+1):
        dNf_dgN_expr[i] = dNf_dgN_expr[i-1].diff(g_expr)/2

    dNg_dhN_expr      = sympy.Matrix.zeros(ord+1,1)
    dNg_dhN_expr[0] = g_expr
    dNg_dhN_expr[1] = g_expr.diff(h_expr)
    for i in range(2,ord+1):
        dNg_dhN_expr[i] = dNg_dhN_expr[i-1].diff(h_expr)

    dNf_dhN_expr      = sympy.Matrix.zeros(ord+1,1)
    dNf_dhN_expr[0] = f_expr
    dNf_dhN_expr[1] = f_expr.diff(h_expr)
    for i in range(2,ord+1):
        dNf_dhN_expr[i] = dNf_dhN_expr[i-1].diff(h_expr)

    f_wild_expr                    = sympy.Wild("p_wild",real=True)
    x_wild_expr                    = sympy.Wild("x_wild",real=True)
    dNf_dgN_not_yet_evaluated_expr = sympy.Matrix.zeros(ord+1,1)
    for i in range(1,ord+1):
        derivative_args                   = [f_wild_expr] + [x_wild_expr]*i
        dNf_dgN_not_yet_evaluated_expr[i] = sympy.Subs(sympy.Derivative(*derivative_args),x_wild_expr,g_expr)

    dNf_dhN_in_terms_of_subs_expr    = sympy.Matrix.zeros(ord+1,1)
    dNf_dhN_in_terms_of_subs_expr[0] = dNf_dgN_subs[0]
    for i in range(1,ord+1):
        dNf_dhN_in_terms_of_subs_expr[i] = dNf_dhN_expr[i]
        dNf_dhN_in_terms_of_subs_expr[i] = dNf_dhN_in_terms_of_subs_expr[i].subs(zip(dNg_dhN_expr[::-1],dNg_dhN_subs[::-1])[:-1],simultaneous=True)
        for j in range(1,ord+1):
            dNf_dhN_in_terms_of_subs_expr[i] = dNf_dhN_in_terms_of_subs_expr[i].replace(dNf_dgN_not_yet_evaluated_expr[j],dNf_dgN_subs[j])

    return dNf_dhN_in_terms_of_subs_expr



#
# processing expressions
#

def simplify_assume_real(expr,syms):
    return sympy.refine( expr, sympy.And( *[ sympy.Q.real(sym) for sym in matrix(syms).A1 ] ) )

def nsimplify_matrix(A_expr,constants=[],tolerance=None,full=False,rational=False):

    A_nsimplified_expr = sympy.Matrix.zeros(A_expr.rows,A_expr.cols)
    for r in range(A_expr.rows):
        for c in range(A_expr.cols):
            A_nsimplified_expr[r,c] = sympy.nsimplify(A_expr[r,c],constants,tolerance,full,rational)
    return A_nsimplified_expr

def subs_matrix_verbose(A_expr,subs,simultaneous=False):

    print "flashlight.sympy: subs_matrix_verbose(...) begin..."
    A_subs_expr = sympy.Matrix.zeros(A_expr.rows,A_expr.cols)
    for r in range(A_expr.rows):
        for c in range(A_expr.cols):
            print "    ",r,c,len(subs),sympy.count_ops(A_expr[r,c])
            A_subs_expr[r,c] = A_expr[r,c].subs(subs,simultaneous=simultaneous)
    print "flashlight.sympy: subs_matrix_verbose(...) end."
    return A_subs_expr

def dummify(expr,syms,pretty_dummy_symbol_names=False):

    old_expr = expr
    old_syms = syms

    if pretty_dummy_symbol_names:
        old_syms_to_new_syms_non_derivatives = dict( [ (old_sym, sympy.Symbol("\Delta_\mathrm{%s}"%str(old_sym))) for old_sym in old_syms if not old_sym.is_Derivative ] )
        old_syms_to_new_syms_derivatives     = dict( [ (old_sym, sympy.Symbol("\Delta_\mathrm{%s}"%str(old_sym))) for old_sym in old_syms if old_sym.is_Derivative ] )
    else:
        old_syms_to_new_syms_non_derivatives = dict( [ (old_sym, sympy.Symbol("_dummy_%s"%str(old_sym).replace(", ","_").replace("(","_").replace(")","_"))) for old_sym in old_syms if not old_sym.is_Derivative ] )
        old_syms_to_new_syms_derivatives     = dict( [ (old_sym, sympy.Symbol("_dummy_%s"%str(old_sym).replace(", ","_").replace("(","_").replace(")","_"))) for old_sym in old_syms if old_sym.is_Derivative ] )

    old_syms_to_new_syms = dict(old_syms_to_new_syms_non_derivatives.items() + old_syms_to_new_syms_derivatives.items())
    new_syms             = matrix([ old_syms_to_new_syms[old_sym] for old_sym in old_syms ]).A1

    if isinstance(expr,list):
        new_expr = [ e.subs(old_syms_to_new_syms_derivatives).subs(old_syms_to_new_syms_non_derivatives) for e in expr ]
    else:
        new_expr = expr.subs(old_syms_to_new_syms_derivatives).subs(old_syms_to_new_syms_non_derivatives)

    return new_expr,new_syms

def collect_into_dict_include_zero_and_constant_terms(expr, syms):
    
    expr_terms_dict = sympy.collect(expr,syms,exact=True,evaluate=False)
    for sym in syms:
        if sym not in expr_terms_dict.keys(): expr_terms_dict[sym] = 0
    if 1 not in expr_terms_dict: expr_terms_dict[1] = 0
    return expr_terms_dict



#
# constructing and evaluating anonymous functions
#

def construct_anon_func_lambdify(expr,syms,dummify=False,pretty_dummy_symbol_names=False):
    return _construct_anon_func(expr=expr,syms=syms,dummify=dummify,pretty_dummy_symbol_names=pretty_dummy_symbol_names,verbose=False,construct_func=_construct_anon_func_lambdify)

def construct_anon_func_autowrap(expr,syms,dummify=False,pretty_dummy_symbol_names=False,verbose=False):
    return _construct_anon_func(expr=expr,syms=syms,dummify=dummify,pretty_dummy_symbol_names=pretty_dummy_symbol_names,verbose=verbose,construct_func=_construct_anon_func_autowrap)

def construct_anon_func_ufuncify(expr,syms,dummify=False,pretty_dummy_symbol_names=False,verbose=False):
    return _construct_anon_func(expr=expr,syms=syms,dummify=dummify,pretty_dummy_symbol_names=pretty_dummy_symbol_names,verbose=verbose,construct_func=_construct_anon_func_ufuncify)

# def construct_anon_func_theano(expr,syms,dummify=False,pretty_dummy_symbol_names=False,verbose=False):
#     return _construct_anon_func(expr=expr,syms=syms,dummify=dummify,pretty_dummy_symbol_names=pretty_dummy_symbol_names,verbose=verbose,construct_func=_construct_anon_func_theano)

def construct_matrix_anon_funcs_lambdify(matrix_expr,syms,dummify=False,pretty_dummy_symbol_names=False):
    return _construct_matrix_anon_funcs(matrix_expr=matrix_expr,syms=syms,dummify=dummify,pretty_dummy_symbol_names=pretty_dummy_symbol_names,verbose=False,construct_func=_construct_anon_func_lambdify)

def construct_matrix_anon_funcs_autowrap(matrix_expr,syms,dummify=False,pretty_dummy_symbol_names=False,verbose=False):
    return _construct_matrix_anon_funcs(matrix_expr=matrix_expr,syms=syms,dummify=dummify,pretty_dummy_symbol_names=pretty_dummy_symbol_names,verbose=verbose,construct_func=_construct_anon_func_autowrap)

def construct_matrix_anon_funcs_ufuncify(matrix_expr,syms,dummify=False,pretty_dummy_symbol_names=False,verbose=False):
    return _construct_matrix_anon_funcs(matrix_expr=matrix_expr,syms=syms,dummify=dummify,pretty_dummy_symbol_names=pretty_dummy_symbol_names,verbose=verbose,construct_func=_construct_anon_func_ufuncify)

def evaluate_anon_func(anon_func,vals):

    vals = matrix(vals)

    num_evaluations = vals.shape[0]
    num_arguments   = vals.shape[1]

    if isinstance(anon_func,ufunc):
        assert anon_func.nin == num_arguments

    # if isinstance(anon_func,theano.compile.function_module.Function):
    #     return anon_func(*vals.A1)
    # else:
    #     return anon_func(*vals.T)

    return anon_func(*vals.T)

def evaluate_matrix_anon_funcs(matrix_anon_funcs,vals):

    vals = matrix(vals)

    num_evaluations = vals.shape[0]
    num_arguments   = vals.shape[1]
    num_rows        = matrix_anon_funcs.shape[0]
    num_cols        = matrix_anon_funcs.shape[1]

    matrix_anon_funcs_eval = zeros((num_evaluations,num_rows,num_cols))

    for r in range(num_rows):
        for c in range(num_cols):
            if isinstance(matrix_anon_funcs[r,c],ufunc):
                assert matrix_anon_funcs[r,c].nin == num_arguments
            matrix_anon_funcs_eval[:,r,c] = matrix_anon_funcs[r,c](*vals.T)

    return matrix_anon_funcs_eval

def _dummify(expr,syms,pretty_dummy_symbol_names):
    return dummify(expr,syms,pretty_dummy_symbol_names)

def _construct_anon_func(expr,syms,dummify,pretty_dummy_symbol_names,verbose,construct_func):

    if dummify:
        expr,syms = _dummify(expr,syms,pretty_dummy_symbol_names)

    return construct_func(expr=expr,syms=syms,verbose=verbose)

def _construct_matrix_anon_funcs(matrix_expr,syms,dummify,pretty_dummy_symbol_names,verbose,construct_func):

    matrix_anon_funcs = []
    for r in range(matrix_expr.rows):
        matrix_r_anon_funcs = []
        for c in range(matrix_expr.cols):
            matrix_expr_rc      = matrix_expr[r,c]
            matrix_rc_anon_func = _construct_anon_func(expr=matrix_expr_rc,syms=syms,dummify=dummify,pretty_dummy_symbol_names=pretty_dummy_symbol_names,verbose=verbose,construct_func=construct_func)
            if isinstance(matrix_r_anon_funcs,list): matrix_r_anon_funcs = matrix_rc_anon_func
            else:                                    matrix_r_anon_funcs = hstack((matrix_r_anon_funcs,matrix_rc_anon_func))
        if isinstance(matrix_anon_funcs,list): matrix_anon_funcs = matrix_r_anon_funcs
        else:                                  matrix_anon_funcs = vstack((matrix_anon_funcs,matrix_r_anon_funcs))

    return matrix_anon_funcs

def _construct_anon_func_lambdify(expr,syms,verbose):
    return sympy.lambdify(syms,expr,"numpy")

def _construct_anon_func_autowrap(expr,syms,verbose):
    return sympy.utilities.autowrap.autowrap(expr=expr,backend="cython",args=syms,verbose=verbose)

def _construct_anon_func_ufuncify(expr,syms,verbose):
    return sympy.utilities.autowrap.ufuncify(expr=expr,backend="numpy",args=syms,verbose=verbose)

# def _construct_anon_func_theano(expr,syms,verbose):
#     return sympy.printing.theanocode.theano_function(syms,expr,on_unused_input="ignore")



#
# generating C code
#
def _wrap_code_skip_compile(self, routine, helpers=[]):

    workdir = self.filepath or tempfile.mkdtemp("_sympy_compile")
    if not os.access(workdir, os.F_OK):
        os.mkdir(workdir)
    oldwork = os.getcwd()
    os.chdir(workdir)
    try:
        sys.path.append(workdir)
        self._generate_code(routine, helpers)
        self._prepare_files(routine)
    finally:
        sys.path.remove(workdir)
        sympy.utilities.autowrap.CodeWrapper._module_counter += 1
        os.chdir(oldwork)
        if not self.filepath:
            shutil.rmtree(workdir)

    return

def _autowrap_skip_compile(expr, language=None, backend='f2py', tempdir=None, args=None, flags=None, verbose=False, helpers=None):

    if language:
        sympy.utilities.autowrap._validate_backend_language(backend, language)
    else:
        language = sympy.utilities.autowrap._infer_language(backend)

    helpers = helpers if helpers else ()
    flags = flags if flags else ()

    code_generator = sympy.utilities.autowrap.get_code_generator(language, "autowrap")
    CodeWrapperClass = sympy.utilities.autowrap._get_code_wrapper_class(backend)
    code_wrapper = CodeWrapperClass(code_generator, tempdir, flags, verbose)

    try:
        routine = sympy.utilities.autowrap.make_routine('autofunc', expr, args)
    except sympy.utilities.autowrap.CodeGenArgumentListError as e:
        new_args = []
        for missing in e.missing_args:
            if not isinstance(missing, sympy.utilities.autowrap.OutputArgument):
                raise
            new_args.append(missing.name)
        routine = sympy.utilities.autowrap.make_routine('autofunc', expr, args + new_args)

    helps = []
    for name, expr, args in helpers:
        helps.append(sympy.utilities.autowrap.make_routine(name, expr, args))

    return _wrap_code_skip_compile(code_wrapper, routine, helpers=helps)

def generate_c_code(expr,syms,func_name,tmp_dir,out_dir,verbose=False,request_delete_tmp_dir=True):
    
    tmp_dir_exists_before_call = os.path.exists(tmp_dir)

    syms = list(syms)

    if isinstance(expr,sympy.Matrix) or isinstance(expr,sympy.MutableMatrix) or isinstance(expr,sympy.ImmutableMatrix):
        out_sym = sympy.MatrixSymbol('out_%s' % abs(hash(sympy.ImmutableMatrix(expr))), *expr.shape)
        syms    = syms + [out_sym]

    if verbose: print "flashlight.sympy: Generating Cython code for %s..." % func_name
    _autowrap_skip_compile(expr=expr,backend="cython",verbose=verbose,args=syms,tempdir=tmp_dir)
    if verbose: print "flashlight.sympy: Modifying autogenerated Cython files and building..."

    with open("%s/setup.py" % tmp_dir, "r") as f:
        setup_py_str = f.read()
        tmp_module_name        = re.findall("wrapper_module_[0-9]*", setup_py_str)
        tmp_autofunc_code_name = re.findall("wrapped_code_[0-9]*",   setup_py_str)
        assert len(tmp_module_name)        == 2
        assert len(tmp_autofunc_code_name) == 1
        assert tmp_module_name[0] == tmp_module_name[1]
        tmp_module_name        = tmp_module_name[0]
        tmp_autofunc_code_name = tmp_autofunc_code_name[0]
    with open("%s/%s.c" % (tmp_dir,tmp_autofunc_code_name), "r") as f:
        autofunc_c_str = f.read()
    with open("%s/%s.h" % (tmp_dir,tmp_autofunc_code_name), "r") as f:
        autofunc_h_str = f.read()

    with open("%s/%s_autofunc.c" % (tmp_dir,func_name), "w") as f:
        autofunc_c_str_mod = autofunc_c_str
        autofunc_c_str_mod = autofunc_c_str_mod.replace("autofunc(", "%s_autofunc(" % func_name)
        autofunc_c_str_mod = autofunc_c_str_mod.replace('#include "%s.h"' % tmp_autofunc_code_name, '#include "%s.h"' % (func_name+"_autofunc"))
        f.write(autofunc_c_str_mod)
    with open("%s/%s_autofunc.h" % (tmp_dir,func_name), "w") as f:
        autofunc_h_str_mod = autofunc_h_str
        autofunc_h_str_mod = autofunc_h_str.replace("autofunc(", "%s_autofunc(" % func_name)
        autofunc_h_str_mod = autofunc_h_str_mod.replace("AUTOWRAP__%s__H" % tmp_autofunc_code_name.upper(), "%s_H" % (func_name+"_autofunc").upper())
        f.write(autofunc_h_str_mod)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cmd = "cp %s/%s_autofunc.c %s/%s_autofunc.c" % (tmp_dir,func_name,out_dir,func_name)
    if verbose: print cmd
    output = subprocess.check_output(cmd, shell=True)
    if verbose and len(output) > 0: print output

    cmd = "cp %s/%s_autofunc.h %s/%s_autofunc.h" % (tmp_dir,func_name,out_dir,func_name)
    if verbose: print cmd
    output = subprocess.check_output(cmd, shell=True)
    if verbose and len(output) > 0: print output

    if not tmp_dir_exists_before_call and request_delete_tmp_dir:
        cmd = "rm -rf %s" % tmp_dir
        if verbose: print cmd
        output = subprocess.check_output(cmd, shell=True)
    if verbose and len(output) > 0: print output

    if verbose: print

cse_autofunc_h_str = \
"""
/******************************************************************************
 *              Code generated with flashlight.sympyutils v0.0.1              *
 ******************************************************************************/

#ifndef %s_AUTOFUNC_H
#define %s_AUTOFUNC_H

%s autofunc(%s);

#endif
"""

cse_autofunc_c_str = \
"""
/******************************************************************************
 *              Code generated with flashlight.sympyutils v0.0.1              *
 ******************************************************************************/

#include "%s_autofunc.h"
#include <math.h>

%s
%s autofunc(%s) {
%s

}
"""

def generate_c_code_cse(expr,syms,func_name,tmp_dir,out_dir,cse_ordering="canonical",verbose=False,request_delete_tmp_dir=True):

    if verbose: print "flashlight.sympy: Performing common subexpression elimination..."
    subexpr_list,expr_in_terms_of_subexprs = sympy.cse(expr,order=cse_ordering)
    if verbose: print "flashlight.sympy: Finished performing common subexpression elimination."

    include_str             = ""
    subexpr_eval_str        = "\n"
    syms_not_incl_temp_vars = list(syms)
    syms_incl_temp_vars     = list(syms)

    relational_expr_types     = [ sympy.Equality, sympy.Unequality, sympy.LessThan, sympy.StrictLessThan, sympy.GreaterThan, sympy.StrictGreaterThan ]
    relational_subexprs       = [ (k,v)                           for (k,v) in subexpr_list if type(v) in relational_expr_types or v == True or v == False ]
    nonrelational_subexprs    = [ (k,v)                           for (k,v) in subexpr_list if (k,v) not in relational_subexprs ]
    subexpr_list              = [ (k,v.subs(relational_subexprs)) for (k,v) in nonrelational_subexprs ]
    expr_in_terms_of_subexprs = [ expr.subs(relational_subexprs)  for expr  in expr_in_terms_of_subexprs ]

    is_matrix_expr = isinstance(expr,sympy.Matrix) or isinstance(expr,sympy.MutableMatrix) or isinstance(expr,sympy.ImmutableMatrix)
    
    for subexpr_kv in subexpr_list:

        subexpr_name_expr   = subexpr_kv[0]
        subexpr_func_expr   = subexpr_kv[1]
        subexpr_c_func_name = func_name+"_"+str(subexpr_name_expr)
        include_str         = include_str      + '#include "%s_autofunc.c"\n' % subexpr_c_func_name
        subexpr_eval_str    = subexpr_eval_str + "    double %s = %s_autofunc(%s);\n" % (subexpr_name_expr,subexpr_c_func_name,str(syms_incl_temp_vars)[1:-1].replace("'",""))

        generate_c_code(subexpr_func_expr,syms_incl_temp_vars,subexpr_c_func_name,tmp_dir,out_dir,verbose=verbose,request_delete_tmp_dir=request_delete_tmp_dir)    
        syms_incl_temp_vars.append(subexpr_name_expr)

    expr_in_terms_of_subexprs_c_func_name = func_name+"_in_terms_of_subexprs"
    include_str                           = include_str      + '#include "%s_autofunc.c"\n' % expr_in_terms_of_subexprs_c_func_name

    generate_c_code(expr_in_terms_of_subexprs,syms_incl_temp_vars,expr_in_terms_of_subexprs_c_func_name,tmp_dir,out_dir,verbose=verbose,request_delete_tmp_dir=request_delete_tmp_dir)

    if is_matrix_expr:
        return_type_str    = "void"
        out_var_str        = "out_123456789"
        func_signature_str = str( [ "double %s" % str(sym) for sym in syms_not_incl_temp_vars ] )[1:-1].replace("'","") + (", double *%s")%out_var_str
        subexpr_eval_str   = subexpr_eval_str + "\n    %s_autofunc(%s, %s);\n\n" % (expr_in_terms_of_subexprs_c_func_name,str(syms_incl_temp_vars)[1:-1].replace("'",""),out_var_str)
    else:
        return_type_str    = "double"
        func_signature_str = str( [ "double %s" % str(sym) for sym in syms_not_incl_temp_vars ] )[1:-1].replace("'","")
        subexpr_eval_str   = subexpr_eval_str + "\n    return %s_autofunc(%s);" % (expr_in_terms_of_subexprs_c_func_name,str(syms_incl_temp_vars)[1:-1].replace("'",""))
        
    cse_autofunc_h_str_eval = cse_autofunc_h_str % (func_name.upper(),func_name.upper(),return_type_str,func_signature_str)
    cse_autofunc_c_str_eval = cse_autofunc_c_str % (func_name,        include_str,      return_type_str,func_signature_str,subexpr_eval_str)

    with open("%s/%s_autofunc.h" % (tmp_dir,func_name), "w") as f:
        f.write(cse_autofunc_h_str_eval)
    with open("%s/%s_autofunc.c" % (tmp_dir,func_name), "w") as f:
        f.write(cse_autofunc_c_str_eval)

    cmd = "cp %s/%s_autofunc.c %s/%s_autofunc.c" % (tmp_dir,func_name,out_dir,func_name)
    if verbose: print cmd
    output = subprocess.check_output(cmd, shell=True)
    if verbose and len(output) > 0: print output

    cmd = "cp %s/%s_autofunc.h %s/%s_autofunc.h" % (tmp_dir,func_name,out_dir,func_name)
    if verbose: print cmd
    output = subprocess.check_output(cmd, shell=True)
    if verbose and len(output) > 0: print output

    return include_str,subexpr_eval_str



#
# building modules
#

vectorized_setup_py_str = \
"""
from distutils.core      import setup
from distutils.extension import Extension
from Cython.Build        import cythonize

import numpy

ext_name           = "%s_vectorized"
src_files          = [ "%s_vectorized.pyx", "%s_autofunc.c" ]
include_dirs       = [ numpy.get_include() ]
extra_compile_args = [ "-fopenmp", "-fno-var-tracking", "-fno-var-tracking-assignments" ]
extra_link_args    = [ "-fopenmp" ]

extensions = [ Extension( ext_name, src_files, include_dirs=include_dirs, extra_compile_args=extra_compile_args, extra_link_args=extra_link_args ) ]
setup( name=ext_name, ext_modules=cythonize(extensions) )
"""

vectorized_pyx_str = \
"""
#******************************************************************************
#*                Code generated with flashlight.sympy v0.0.1                 *
#******************************************************************************

import numpy
import cython.parallel

cimport cython
cimport numpy

ctypedef numpy.float64_t FLOAT64_DTYPE_t

cdef extern from "%s_autofunc.h":
    %s autofunc(%s) nogil

def autofunc_c(
    numpy.ndarray[FLOAT64_DTYPE_t, ndim=2] args ):

    cdef %s out = numpy.empty(%s)

    cdef FLOAT64_DTYPE_t [:,:] args_v = args
    cdef %s out_v = out

    cdef int i
    with nogil:
        for i in range(args.shape[0]):
            %s

    return out.squeeze()
"""

def build_module_autowrap(expr,syms,module_name,tmp_dir,out_dir,dummify=False,cse=False,cse_ordering="canonical",build_vectorized=False,pretty_dummy_symbol_names=False,verbose=False,request_delete_tmp_dir=True):

    tmp_dir_exists_before_call = os.path.exists(tmp_dir)

    is_matrix_expr = isinstance(expr,sympy.Matrix) or isinstance(expr,sympy.MutableMatrix) or isinstance(expr,sympy.ImmutableMatrix)

    if dummify:
        if verbose: print "flashlight.sympy: Generating dummy symbols for %s..." % module_name        
        expr,syms = _dummify(expr,syms,pretty_dummy_symbol_names)
        if verbose: print "flashlight.sympy: Finished generating dummy symbols for %s." % module_name

    if cse:
        include_str,subexpr_eval_str = generate_c_code_cse(expr,syms,module_name,tmp_dir,out_dir,cse_ordering=cse_ordering,verbose=verbose,request_delete_tmp_dir=request_delete_tmp_dir)

        if is_matrix_expr:
            expr = sympy.Matrix.zeros(expr.rows,expr.cols)
        else:
            expr = 0

    syms = list(syms)

    if is_matrix_expr:
        out_sym = sympy.MatrixSymbol('out_%s' % abs(hash(sympy.ImmutableMatrix(expr))), *expr.shape)
        syms    = syms + [out_sym]

    if verbose: print "flashlight.sympy: Generating Cython code for %s..." % module_name
    _autowrap_skip_compile(expr=expr,backend="cython",verbose=verbose,args=syms,tempdir=tmp_dir)
    if verbose: print "flashlight.sympy: Modifying autogenerated Cython files for %s" % module_name

    with open("%s/setup.py" % tmp_dir, "r") as f:
        setup_py_str = f.read()
        tmp_module_name        = re.findall("wrapper_module_[0-9]*", setup_py_str)
        tmp_autofunc_code_name = re.findall("wrapped_code_[0-9]*",   setup_py_str)
        assert len(tmp_module_name)        == 2
        assert len(tmp_autofunc_code_name) == 1
        assert tmp_module_name[0] == tmp_module_name[1]
        tmp_module_name        = tmp_module_name[0]
        tmp_autofunc_code_name = tmp_autofunc_code_name[0]
    with open("%s/%s.pyx" % (tmp_dir,tmp_module_name), "r") as f:
        wrapper_module_pyx_str = f.read()
    with open("%s/%s.c" % (tmp_dir,tmp_autofunc_code_name), "r") as f:
        autofunc_c_str = f.read()
    with open("%s/%s.h" % (tmp_dir,tmp_autofunc_code_name), "r") as f:
        autofunc_h_str = f.read()

    with open("%s/%s_setup.py" % (tmp_dir,module_name), "w") as f:
        setup_py_str_mod = setup_py_str
        setup_py_str_mod = setup_py_str_mod.replace("from Cython.Distutils import build_ext","from Cython.Distutils import build_ext\nimport numpy")
        setup_py_str_mod = setup_py_str_mod.replace("extra_compile_args=['-std=c99']","extra_compile_args=['-std=c99','-fno-var-tracking','-fno-var-tracking-assignments'], include_dirs=[numpy.get_include()]")
        setup_py_str_mod = setup_py_str_mod.replace("%s" % tmp_module_name,        "%s"          % module_name)
        setup_py_str_mod = setup_py_str_mod.replace("%s" % tmp_autofunc_code_name, "%s_autofunc" % module_name)
        f.write(setup_py_str_mod)
    with open("%s/%s.pyx" % (tmp_dir,module_name), "w") as f:
        wrapper_module_pyx_str_mod = wrapper_module_pyx_str
        wrapper_module_pyx_str_mod = wrapper_module_pyx_str_mod.replace("%s" % tmp_autofunc_code_name, "%s_autofunc" % module_name)
        f.write(wrapper_module_pyx_str_mod)

    if not cse:
        with open("%s/%s_autofunc.c" % (tmp_dir,module_name), "w") as f:
            autofunc_c_str_mod = autofunc_c_str
            autofunc_c_str_mod = autofunc_c_str_mod.replace('#include "%s.h"' % tmp_autofunc_code_name, '#include "%s.h"' % (module_name+"_autofunc"))
            f.write(autofunc_c_str_mod)
        with open("%s/%s_autofunc.h" % (tmp_dir,module_name), "w") as f:
            autofunc_h_str_mod = autofunc_h_str
            autofunc_h_str_mod = autofunc_h_str_mod.replace("AUTOWRAP__%s__H" % tmp_autofunc_code_name.upper(), "%s_H" % (module_name+"_autofunc").upper())
            f.write(autofunc_h_str_mod)

    if build_vectorized:
        if verbose: print "flashlight.sympy: Generating Cython code for %s_vectorized..." % module_name
        if is_matrix_expr:
            c_return_type_str     = "void"
            c_out_var_str         = "out_123456789"
            c_func_signature_str  = str( [ "double %s" % str(sym) for sym in syms[:-1] ] )[1:-1].replace("'","") + ", double *%s" % syms[-1]
            cython_out_type_str   = "numpy.ndarray[FLOAT64_DTYPE_t, ndim=3]"
            cython_out_shape_str  = "(args.shape[0],%s,%s)" % (expr.rows,expr.cols)
            cython_out_v_type_str = "FLOAT64_DTYPE_t [:,:,:]"
            cython_args_str       = str( [ "args[i,%d]" % i for i in range(len(syms[:-1])) ] )[1:-1].replace("'","") + ", &out_v[i,0,0]"
            cython_loop_body_str  = "autofunc(%s)" % cython_args_str
        else:
            c_return_type_str     = "double"
            c_func_signature_str  = str( [ "double %s" % str(sym) for sym in syms ] )[1:-1].replace("'","")
            cython_out_type_str   = "numpy.ndarray[FLOAT64_DTYPE_t, ndim=1]"
            cython_out_shape_str  = "(args.shape[0])"
            cython_out_v_type_str = "FLOAT64_DTYPE_t [:]"
            cython_args_str       = str( [ "args[i,%d]" % i for i in range(len(syms)) ] )[1:-1].replace("'","")
            cython_loop_body_str  = "out_v[i] = autofunc(%s)" % cython_args_str
        vectorized_setup_py_str_eval = vectorized_setup_py_str % (module_name, module_name, module_name)
        vectorized_pyx_str_eval      = vectorized_pyx_str % (module_name, c_return_type_str, c_func_signature_str, cython_out_type_str, cython_out_shape_str, cython_out_v_type_str, cython_loop_body_str)
        with open("%s/%s_vectorized_setup.py" % (tmp_dir,module_name), "w") as f:
            vectorized_setup_py_str_mod = vectorized_setup_py_str_eval
            f.write(vectorized_setup_py_str_mod)
        with open("%s/%s_vectorized.pyx" % (tmp_dir,module_name), "w") as f:
            vectorized_pyx_str_mod = vectorized_pyx_str_eval
            f.write(vectorized_pyx_str_mod)

    cwd = os.getcwd()
    try:
        os.chdir(tmp_dir)

        if verbose: print "flashlight.sympy: Building Cython code for %s..." % module_name
        cmd = "python %s_setup.py build_ext --inplace" % module_name
        if verbose: print cmd
        output = subprocess.check_output(cmd, shell=True)
        if verbose and len(output) > 0: print output

        if build_vectorized:
            if verbose: print "flashlight.sympy: Building Cython code for %s_vectorized..." % module_name
            cmd = "python %s_vectorized_setup.py build_ext --inplace" % module_name
            if verbose: print cmd
            output = subprocess.check_output(cmd, shell=True)
            if verbose and len(output) > 0: print output

    finally:
        os.chdir(cwd)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cmd = "cp %s/%s.so %s/%s.so" % (tmp_dir,module_name,out_dir,module_name)
    if verbose: print cmd
    output = subprocess.check_output(cmd, shell=True)
    if verbose and len(output) > 0: print output

    if build_vectorized:
        cmd = "cp %s/%s_vectorized.so %s/%s_vectorized.so" % (tmp_dir,module_name,out_dir,module_name)
        if verbose: print cmd
        output = subprocess.check_output(cmd, shell=True)
        if verbose and len(output) > 0: print output

    # cmd = "cp %s/%s_autofunc.c %s/%s_autofunc.c" % (tmp_dir,module_name,out_dir,module_name)
    # if verbose: print cmd
    # output = subprocess.check_output(cmd, shell=True)
    # if verbose and len(output) > 0: print output

    # cmd = "cp %s/%s_autofunc.h %s/%s_autofunc.h" % (tmp_dir,module_name,out_dir,module_name)
    # if verbose: print cmd
    # output = subprocess.check_output(cmd, shell=True)
    # if verbose and len(output) > 0: print output

    if not tmp_dir_exists_before_call and request_delete_tmp_dir:
        cmd = "rm -rf %s" % tmp_dir
        if verbose: print cmd
        output = subprocess.check_output(cmd, shell=True)
        if verbose and len(output) > 0: print output

def import_anon_func_from_from_module_autowrap(module_name,path):
    assert os.path.exists(path)
    oldcwd = os.getcwd()
    try:
        os.chdir(path)
        tmpcwd = os.getcwd()
        sys.path.insert(0,tmpcwd)
        module = __import__(module_name)
    finally:
        sys.path.remove(tmpcwd)
        os.chdir(oldcwd)
    return getattr(module,"autofunc_c")
