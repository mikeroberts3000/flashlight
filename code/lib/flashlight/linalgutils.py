from pylab import *

import sklearn
import sklearn.preprocessing



def cross_product_left_term_matrix_from_vector(a):
    
    return matrix([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])



def project_vectors_onto_vectors(a,b,axis=1):

    assert axis == 0 or axis == 1
    
    a = matrix(a,dtype(float64))
    b = matrix(b,dtype(float64))
    
    if axis == 0:
        a = a.T
        b = b.T
        
    b_normalized = matrix(sklearn.preprocessing.normalize(b))
    a_scalar     = a*b_normalized.T

    if a.shape == b.shape:
        a_scalar = matrix(diag(a_scalar))
        
    if a_scalar.shape[0] == 1:
        a_scalar_column = a_scalar.T
    else:
        a_scalar_column = a_scalar

    if a.shape == b.shape:
        a_projection    = matrix(a_scalar_column.A*b_normalized.A)
        a_orthogonal    = a - a_projection
    else:
        a_projection = matrix(a_scalar_column.A*tile(b_normalized,(a_scalar.shape[0],1)).A)
        a_orthogonal = a - a_projection
        
    if axis == 0:
        a_projection = a_projection.T
        a_orthogonal = a_orthogonal.T

    return a_projection,a_orthogonal