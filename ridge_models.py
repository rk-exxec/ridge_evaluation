import cupy as np 
from cupyx.scipy.special import j0, j1

# copy-pasted from
# https://github.com/scipy/scipy/blob/v1.12.0/scipy/integrate/_tanhsinh.py
# then removed comments to reduce length; other optimizations possible
# There are plans to make this Array-API compatible and provide a public 
# interface

_N_BASE_STEPS = 8

# -----------------
# functions for GPU
# -----------------

def _get_base_step(dtype=np.float64):
    fmin = 4*np.finfo(dtype).tiny
    tmax = np.arcsinh(np.log(2/fmin - 1) / np.pi)
    h0 = tmax / _N_BASE_STEPS
    return h0.astype(dtype)

def _compute_pair(k, h0):
    h = h0 / 2**k
    max = _N_BASE_STEPS * 2**k
    j = np.arange(max+1) if k == 0 else np.arange(1, max+1)
    jh = j * h
    pi_2 = np.pi / 2
    u1 = pi_2*np.cosh(jh)
    u2 = pi_2*np.sinh(jh)
    wj = u1 / np.cosh(u2)**2
    xjc = 1 / (np.exp(u2) * np.cosh(u2))
    wj[0] = wj[0] / 2 if k == 0 else wj[0]
    return xjc, wj

def _transform_to_limits(xjc, wj, a, b):
    alpha = (b - a) / 2
    xj = np.concatenate((-alpha * xjc + b, alpha * xjc + a), axis=-1)
    wj = wj*alpha
    wj = np.concatenate((wj, wj), axis=-1)
    invalid = (xj <= a) | (xj >= b)
    wj[invalid] = 0
    return xj, wj

# simple fixed-step integration function
def integrate(func, a, b, args):
    k = 11  # increase this to improve accuracy 
    step0 = _get_base_step()
    step = step0 / 2**k
    xjc, wj = _compute_pair(k, step0)
    xj, wj = _transform_to_limits(xjc, wj, a, b)
    fj = func(xj, *args)
    return fj @ wj * step

# --------------------------------------

def _integrand_ld_ts(s, x_ ,upsilon, E,h):
        B = 3*upsilon/(2*E)
        numerator = (1+2*s**2)*np.exp(-2*s) + 0.5*(1+np.exp(-4*s))
        denominator = 0.5*(1-np.exp(-4*s)) - 2*s*np.exp(-2*s)
        return np.cos(s*x_) / ( (numerator/denominator)*s + (B*s**2)/h)

def style_ld_ts(r, gamma, R, upsilon, E, h):
    r = np.abs(np.asarray(r))
    x_ = (r-R)/h 
    G = E/3       
    int_out = integrate(_integrand_ld_ts, 0, 1e9, args=(x_[..., np.newaxis],upsilon,E,h))
    return np.asnumpy(gamma/(2*np.pi*G) * int_out)

# ---------------------------------------

# @nb.jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),cache=True)
def _QSzz_1(s,h,z, upsilon, E):
    # exponential form of the QSzz^-1 function to prevent float overflow,
    # also assumes nu==0.5
    numerator = (1+2*s**2*h**2)*np.exp(-2*s*z) + 0.5*(1+np.exp(-4*s*z))
    denominator = 0.5*(1-np.exp(-4*s*z)) - 2*s*h*np.exp(-2*s*z)
    return (3/(2*s*E)) / (numerator/denominator + (3*upsilon/(2*E))*s)

# @nb.jit(nb.float64(nb.float64, nb.float64,nb.float64, nb.float64, nb.float64, nb.float64),cache=True)
def _integrand_exact(s, r, R, upsilon, E, h):
    return s*(R*j0(s*R) - 2*j1(s*R)/s) * _QSzz_1(s,h,h, upsilon, E) * j0(s*r)

def style_exact(r, gamma, R, upsilon, E, h):
    r = np.asarray(r)          
    int_out = integrate(_integrand_exact, 0, 1e7, args=(r[..., np.newaxis],R,upsilon,E,h))
    return np.asnumpy(gamma * int_out)


# ----------------------------------------

def shanahan(x, gamma, R, theta, E, d):
    G =E/3
    x = (np.asarray(x)-R)
    x[x==0] = 0.000001
    zeta = gamma*np.sin(np.deg2rad(theta)) / (2*np.pi*G) * np.log(d/np.abs(x))
    # zeta = gamma * np.log(d/np.abs(x)-displ) + shft
    # xi = np.log(d/np.abs(x))
    return np.asnumpy(zeta)


def limat_symmetric(x, gamma, R, gamma_s, E, theta):
    G= E/3
    x = np.asarray(x)
    l_l = gamma*np.sin(np.deg2rad(theta))/(2*np.pi*G)
    l_s= gamma_s/(np.pi*G)
    ls_R = l_s/R

    def smaller_r(x):
        return l_l * (ls_R*np.log((l_s+x+R)/l_s) + ls_R*np.log((l_s-x+R)/l_s) + (x/R)*np.log((l_s+x+R)/(l_s-x+R)) - 2)
    
    def at_r(x):
        return l_l * np.log(x/l_s)

    def larger_r(x):
        return l_l * (((x+l_s)/R)*np.log((l_s+x+R)/(l_s+x-R)) - 2)
    
    zeta = x.copy()
    zeta[x<R] = smaller_r(x[x<R])
    zeta[x==R] = at_r(x[x==R])
    zeta[x>R] = larger_r(x[x>R])
    
    return np.asnumpy(zeta)