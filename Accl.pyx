# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@lisence: MIT
@status: alpha
"""
import numpy as np
cimport numpy as np
import scipy.constants as sconst

cdef class gl_1D_cache:
    cdef dict c
    def __init__(self):
        self.c = {}
    
    cpdef cache(self, int n):
        return self.c[n]
        
    cpdef add_cache(self, int n, val):
        self.c[n] = val
    
    cpdef gl_1D(self, int n):
        """ Outputs points, weights for 1D gauss quadrature with n points
        
        Gauss-Legendre quadrature can use n points to integrate a polynomial
        of degree 2n-1 exactly:\n
        int(f(x), x, -1, 1) = sum(w_i * f(x_i))\n
        where w_i are the weights corresponding to Guass-Legendre points x_i.\n
        Arguments:
         n: number of guass points to generate.\n\n
        Returns points, weights:
         points: numpy 1d array of values between -1, 1 corresponding to 
         the n Guass-Legendre points.\n
         weights: corespoinding weights for the points output."""
        cdef double m, z, z1, p1, p2, p3, pp
        cdef np.ndarray points, weights
        try:
            points, weights = self.cache(n)
        except KeyError:
            points = np.zeros(n)
            weights = np.zeros(n)
            m = (n+1.0)/2.0
            for i in range(0,int(np.floor(m-1))+1):
                z = np.cos(np.pi*(i+0.75)/(n+0.5))
                z1 = 12.0
                while abs(z - z1) > 1e-14:
                    p1 = 1.0
                    p2 = 0.0
                    for j in range(0, n):
                        p3=p2
                        p2=p1
                        p1=((2.0*j+1.0)*z*p2-j*p3)/(j+1.0)
                    pp = n*(z*p1 - p2)/(z**2 - 1.0)
                    z1 = z;
                    z = z1-p1/pp
                points[i]=z
                points[n-i-1]=-z
                weights[i]= 2/((1.0-z**2)*pp**2)
                weights[n-i-1]=weights[i]
                self.add_cache(n, (points, weights))
        return (points, weights)

the_one_gl1D_cache_object_global_horror = gl_1D_cache()
gl_1D = the_one_gl1D_cache_object_global_horror.gl_1D

cpdef int_1D(f, double a, double b, int pts):
    """ Integrate a function using a given method.
    
    Args:\n
    f: f(x) to integrate wrt/x \n
    a: lower limit\n
    b: upper limit\n
    method: a(pts) where n is number of points. Eg use gl_1D. Produce 
    quadrature for limits -1,1\n
    pts: number of integration points to use.\n
    """
    cdef np.ndarray points, weights
    cdef double integral
    cdef int idx
    
    points, weights = gl_1D(pts)
    # linear remap from -1,1 to a,b:
    weights = weights * (b-a)/2
    points = points + 1 - a + (points+1) * (b - a)/2
    integral = 0.0
    for idx in range(0,pts):
        integral += weights[idx] * f(points[idx])
    return integral

def B_int_function(double T, double n, double vk, double vk_minus):
    """ The B^{(k)}(T, n) function.
    
    T is temperature.\n
    n is refractive index\n
    vk & vk_minus are frequencies used as the limits of integration.
    """
    
    i = int_1D(lambda x: B_function(T, x, n),
                          vk_minus, vk,
                          4) # 4 chosen arbitrarily.
    return i

cdef double B_function(double T, double v, double n):
    """ Calculate the spectral intensity of the black-body radiation
    
    T: temperature
    v: frequency of radiation
    n: refractive index of material
    """
    cdef double a, b
    a = (np.exp((sconst.h*v)/(sconst.k * T))-1.0)
    b = (2.0*sconst.h*(v**3)*(n**2))/(sconst.c**2)
    return b / a

if __name__ == "__main__":
     pts, weights = gl_1D(2)
     print(pts)
     print(weights)












