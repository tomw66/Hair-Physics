# -*- coding: utf-8 -*-
"""
This program models simple hair physics as Boundary Value Problem (BVP),
using the shooting method in order to find the coordinates of the 
hairs, given various forces.
The shooting method method is used due to its general effectiveness and the
given boundary conditions. However, extra care had to be taken ( problem
specific lines 156-163) to ensure the shooting method converges on the correct
solution. For a more universal model, the relaxation method may be more
applicable.
Function fsolve() is used to find the roots, as the newton() function
only accepts a scalar initial estimate.
"""

import numpy
from matplotlib import pyplot
from scipy.integrate import odeint
from scipy.optimize import fsolve

def f(t, s, f_g, f_w):
    """
    Defines the IVP to be solved.

    Parameters
    ----------

    t : array
        Contains angular solutions, and their derivatives
        
    t[0] : Theta(s)
    t[1] : Theta'(s)
    t[2] : Phi(s)
    t[3] : Phi'(s)
    
    s : scalar
        Domain over which t is evaluated

    Returns
    -------
    
    dtds : array
           Outputs the first and second order derivatives
           
    dtds[0] : Theta'(s)
    dtds[1] : Theta''(s)
    dtds[2] : Phi'(s)
    dtds[3] : Phi''(s)
    """
    assert(len(t) == 4),\
    "t must be array of length 4"
    
    assert((numpy.isscalar(s)) and (not numpy.any(numpy.isnan(s))) and\
    numpy.all(numpy.isfinite(s)) and numpy.all(numpy.isreal(s))), \
    "s must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(f_g)) and (not numpy.any(numpy.isnan(f_g))) and\
    numpy.all(numpy.isfinite(f_g)) and numpy.all(numpy.isreal(f_g))),\
    "f_g must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(f_w)) and (not numpy.any(numpy.isnan(f_w))) and\
    numpy.all(numpy.isfinite(f_w)) and numpy.all(numpy.isreal(f_w))),\
    "f_w must be a scalar that is real, finite and not NaN"
    
    dtds = numpy.zeros_like(t)
    dtds[0] = t[1]
    dtds[1] = s*f_g*numpy.cos(t[0])+s*f_w*numpy.sin(t[0])*numpy.cos(t[2])
    dtds[2] = t[3]
    dtds[3] = -s*f_w*numpy.sin(t[2])*numpy.sin(t[0])
    return dtds

def shooting_ivp(zp, a, i, N, t0, p0, L, f_g, f_w):
    """
    Given the guesses for the boundary, solves the IVP and returns
    the difference between the calculated and true solution.
    
    Parameters
    ----------

    zp : Guesses for Theta(L) and Phi(L)
    a, i : Iteration of hairs from Theta(0) and Phi(0) lists

    Returns
    -------
    
    [t_boundary - t0[a], p_boundary - p0[i]] : array of length 2
                                               difference between true and
                                               calculated solutions
    """
    assert(len(zp) == 2), \
    "zp must be array of length 2"
    
    assert((numpy.isscalar(a)) and (a<=N)),\
    "a must be a scalar less than or equal to N"
    
    assert((numpy.isscalar(i)) and (i<=N)),\
    "i must be a scalar less than or equal to N"
    
    assert((numpy.isscalar(L)) and (not numpy.any(numpy.isnan(L))) and\
    numpy.all(numpy.isfinite(L)) and numpy.all(numpy.isreal(L))),\
    "L must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(f_g)) and (not numpy.any(numpy.isnan(f_g))) and\
    numpy.all(numpy.isfinite(f_g)) and numpy.all(numpy.isreal(f_g))),\
    "f_g must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(f_w)) and (not numpy.any(numpy.isnan(f_w))) and\
    numpy.all(numpy.isfinite(f_w)) and numpy.all(numpy.isreal(f_w))),\
    "f_w must be a scalar that is real, finite and not NaN"
    
    assert(len(t0) == N),\
    "t0 must be array of length N"
    
    assert(len(p0) == N), \
    "p0 must be array of length N"
    
    soln = odeint(f, [zp[0], 0, zp[1], 0], [L,0], args=(f_g, f_w))
    t_boundary = soln[-1, 0]
    p_boundary = soln[-1, 2]
    return [t_boundary - t0[a], p_boundary - p0[i]]
    
def shooting(a, i, N, t0, p0, L, f_g, f_w):
    """
    Shooting method for BVP.
    
    Parameters
    ----------

    a, i : Iteration of hairs from Theta(0) and Phi(0) lists
    
    Returns
    -------
    
    s : Position on hair axis
    soln : Solution of Theta and Phi at s
    """
    assert((numpy.isscalar(a)) and (a<=N)),\
    "a must be a scalar less than or equal to N"
    
    assert((numpy.isscalar(i)) and (i<=N)),\
    "i must be a scalar less than or equal to N"
    
    assert((numpy.isscalar(f_g)) and (not numpy.any(numpy.isnan(f_g))) and\
    numpy.all(numpy.isfinite(f_g)) and numpy.all(numpy.isreal(f_g))),\
    "f_g must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(f_w)) and (not numpy.any(numpy.isnan(f_w))) and\
    numpy.all(numpy.isfinite(f_w)) and numpy.all(numpy.isreal(f_w))),\
    "f_w must be a scalar that is real, finite and not NaN"
    
    assert(len(t0) == N),\
    "t0 must be array of length N"
    
    assert(len(p0) == N), \
    "p0 must be array of length N"
    
    if t0[a]<numpy.pi/2 or f_w>0.0:
        t_guess=0.0
    else:
        t_guess=numpy.pi
    if p0[i]<numpy.pi/2 or f_w>0.0:
        p_guess=0.0
    else:
        p_guess=numpy.pi
    guess = [t_guess, p_guess]
    proper = fsolve(shooting_ivp, guess, args=(a, i, N, t0, p0, L, f_g, f_w))   #nonlinear root finding
    s = numpy.linspace(L,0)
    soln = odeint(f, [proper[0], 0, proper[1], 0], s, args=(f_g, f_w))
    return s, soln[:, 0], soln[:, 2]

def task1(N, t0, p0, L, R, f_g, f_w, shooting, plot):
    """
    Computes and returns the (x, z) coordinates of the hairs, for the case
    when Phi(0) = 0.
    
    Parameters
    ----------

    N : Number of hairs
    t0 : List of Theta(0), angle in (x, z) plane at which hairs meet head
    p0 : List of Phi(0), angle in (y, z) plane at which hairs meet head
    L : Length of hair
    R : Radius of 'head'
    f_g : Force due to gravity
    f_w : Force due to wind
    
    plot : If true, plots hairs in (x, z) plane

    Returns
    -------
    
    x, z : 2D array
           Position coordinates of hair N
    """
    assert((numpy.isscalar(N)) and (not numpy.any(numpy.isnan(N))) and\
    numpy.all(numpy.isfinite(N)) and numpy.all(numpy.isreal(N))),\
    "N must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(L)) and (not numpy.any(numpy.isnan(L))) and\
    numpy.all(numpy.isfinite(L)) and numpy.all(numpy.isreal(L))),\
    "L must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(R)) and (not numpy.any(numpy.isnan(R))) and\
    numpy.all(numpy.isfinite(R)) and numpy.all(numpy.isreal(R))),\
    "R must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(f_g)) and (not numpy.any(numpy.isnan(f_g))) and\
    numpy.all(numpy.isfinite(f_g)) and numpy.all(numpy.isreal(f_g))),\
    "f_g must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(f_w)) and (not numpy.any(numpy.isnan(f_w))) and\
    numpy.all(numpy.isfinite(f_w)) and numpy.all(numpy.isreal(f_w))),\
    "f_w must be a scalar that is real, finite and not NaN"
    
    assert(len(t0) == N),\
    "t0 must be array of length N"
    
    assert(len(p0) == N) and (numpy.all(p0) == 0),\
    "p0 must be array of length N, with every element as zero"
    
    assert(hasattr(shooting, '__call__')), \
    "shooting must be a callable function"
    
    x=numpy.zeros((N,50))
    z=numpy.zeros((N,50))
    for i in range(N):
        x[i]=-(R*numpy.cos(t0[i])*numpy.cos(p0[i])) - \
        shooting(i, i, N, t0, p0, L, f_g, f_w)[0]*numpy.cos(shooting(i, i, N, t0, p0, L, f_g, f_w)[1])*numpy.cos(shooting(i, i, N, t0, p0, L, f_g, f_g)[2])
        
        z[i]=R*numpy.sin(t0[i])+shooting(i, i, N, t0, p0, L, f_g, f_w)[0]*(numpy.sin(shooting(i, i, N, t0, p0, L, f_g, f_w)[1]))
        if plot==True:
            pyplot.plot(x[i],z[i],'r')
    if plot==True:
        xc = numpy.linspace(-R, R, 100)
        zc = numpy.linspace(-R, R, 100)
        X, Z = numpy.meshgrid(xc,zc)
        F = X**2 + Z**2 - R**2
        pyplot.contour(X,Z,F,[0])   #plot head
        pyplot.ylim(-2*R,2*R)
        pyplot.xlim(-2*R,2*R)
        pyplot.xlabel('X axis')
        pyplot.ylabel('Z axis')
        pyplot.show()
    return x, z
#------------------------------------------------------------------------------
def task4(N, t0, p0, L, R, f_g, f_w, shooting, plot):
    """
    Computes and returns the (x, z) coordinates of the hairs, for the case
    when Phi(0) = 0.
    
    Parameters
    ----------

    Inputs are as in task1()
    
    plot : If true, plots hairs in (x, z), (y, z), and (x, y, z) planes

    Returns
    -------
    
    x, y, z : 2D array
              Position coordinates of hair N
    """
    assert((numpy.isscalar(N)) and (not numpy.any(numpy.isnan(N))) and\
    numpy.all(numpy.isfinite(N)) and numpy.all(numpy.isreal(N))), \
    "N must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(L)) and (not numpy.any(numpy.isnan(L))) and\
    numpy.all(numpy.isfinite(L)) and numpy.all(numpy.isreal(L))),\
    "L must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(R)) and (not numpy.any(numpy.isnan(R))) and\
    numpy.all(numpy.isfinite(R)) and numpy.all(numpy.isreal(R))),\
    "R must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(f_g)) and (not numpy.any(numpy.isnan(f_g))) and\
    numpy.all(numpy.isfinite(f_g)) and numpy.all(numpy.isreal(f_g))),\
    "f_g must be a scalar that is real, finite and not NaN"
    
    assert((numpy.isscalar(f_w)) and (not numpy.any(numpy.isnan(f_w))) and\
    numpy.all(numpy.isfinite(f_w)) and numpy.all(numpy.isreal(f_w))),\
    "f_w must be a scalar that is real, finite and not NaN"
    
    assert(len(t0) == N),\
    "t0 must be array of length N"
    
    assert(len(p0) == N), \
    "p0 must be array of length N"
    
    assert(hasattr(shooting, '__call__')), \
    "shooting must be a callable function"
    
    x=numpy.zeros((N*N,50))
    y=numpy.zeros((N*N,50))
    z=numpy.zeros((N*N,50))
    for i in range(N):
        for j in range(N):
            x[10*i+j]=-(R*numpy.cos(t0[j])*numpy.cos(p0[i]))-\
            shooting(j, i, N, t0, p0, L, f_g, f_w)[0]*numpy.cos(shooting(j, i, N, t0, p0, L, f_g, f_w)[1])*numpy.cos(shooting(j, i, N, t0, p0, L, f_g, f_w)[2])
            
            y[10*i+j]=-(R*numpy.cos(t0[j])*numpy.sin(p0[i]))-\
            shooting(j, i, N, t0, p0, L, f_g, f_w)[0]*numpy.cos(shooting(j, i, N, t0, p0, L, f_g, f_w)[1])*numpy.sin(shooting(j, i, N, t0, p0, L, f_g, f_w)[2])
            
            z[10*i+j]=R*numpy.sin(t0[j])+shooting(j, i, N, t0, p0, L, f_g, f_w)[0]*(numpy.sin(shooting(j, i, N, t0, p0, L, f_g, f_w)[1]))
    if plot==True:
        xc = numpy.linspace(-R, R, 100)
        zc = numpy.linspace(-R, R, 100)
        X, Z = numpy.meshgrid(xc,zc)
        F = X**2 + Z**2 - R**2
        
        "(x, z) plot"
        fig = pyplot.figure(1)
        pyplot.contour(X,Z,F,[0])
        pyplot.ylim(-2*R,2*R)
        pyplot.xlim(-2*R,2*R)
        for i in range(N*N):
            pyplot.plot(x[i], z[i], color='r')
        pyplot.xlabel('X axis')
        pyplot.ylabel('Z axis')
        
        "(y, z) plot"
        fig = pyplot.figure(2)
        pyplot.contour(X,Z,F,[0])
        pyplot.ylim(-2*R,2*R)
        pyplot.xlim(-2*R,2*R)
        for i in range(N*N):
            pyplot.plot(y[i], z[i], color='r')
        pyplot.xlabel('X axis')
        pyplot.ylabel('Y axis')
        
        "(x, y, z) plot"
        fig = pyplot.figure(3)
        ax = fig.add_subplot(111, projection='3d')
        u = numpy.linspace(0, 2 * numpy.pi, 100)
        v = numpy.linspace(0, numpy.pi, 100)
        xd = R * numpy.outer(numpy.cos(u), numpy.sin(v))
        yd = R * numpy.outer(numpy.sin(u), numpy.sin(v))
        zd = R * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))
        ax.plot_surface(xd, yd, zd, color='w')
        for i in range(N*N):
            ax.plot(x[i], y[i], z[i] , color='r')
        
        ax.view_init(45,270)    #3D viewing angles
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        pyplot.show()
    return x, y, z

if __name__ == "__main__":
    "Task 2"
    f_g1=0.1
    f_w1=0.0
    N1=20
    t01=numpy.linspace(0,numpy.pi,N1)
    p01=numpy.linspace(0,0,N1)
    L1=4.0
    R1=10.0
    task1(N1, t01, p01, L1, R1, f_g1, f_w1, shooting, True)
    
    "Task 3"
    f_w3=0.1
    task1(N1, t01, p01, L1, R1, f_g1, f_w3, shooting, True)
    
    "Task 5"
    f_g2=0.1
    f_w2=0.05
    N2=10
    t02=numpy.linspace(0, 0.049*numpy.pi, N2)
    p02 = numpy.linspace(0, numpy.pi, N2)
    L2=4.0
    R2 = 10.0
    task4(N2, t02, p02, L2, R2, f_g2, f_w2, shooting, True)
    