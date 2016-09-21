from pylab import *

def compute_smallest_angular_diff(a,b):

    # analytically, this dot product will always be in the range [-1.0,1.0],
    # but we clip to account for small numerical errors in the dot product
    # function when a and b are almost identical
    omega = arccos( clip( dot([sin(a),cos(a)],[sin(b),cos(b)]), -1.0, 1.0 ) )

    if fmod(b,2*pi) > fmod(a,2*pi):
        if abs(fmod(b,2*pi)-fmod(a,2*pi)) < pi:
            return omega
        else:
            return -omega
    else:
        if abs(fmod(b,2*pi)-fmod(a,2*pi)) < pi:
            return -omega
        else:
            return omega

def compute_continuous_angle_array(a,on_half_circle=False):

    num_timesteps = len(a)

    a_out = zeros_like(a)

    for ti in range(num_timesteps):
        if ti == 0:
            a_out[ti] = a[ti]
        else:
            a_diff    = compute_smallest_angular_diff(a[ti-1],a[ti])
            a_out[ti] = a_out[ti-1] + a_diff

    return a_out
