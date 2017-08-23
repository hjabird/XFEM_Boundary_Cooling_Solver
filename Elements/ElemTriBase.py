# -*- coding: utf-8 -*-
"""
@author: "Hugh Bird"
@copyright "Copyright 2016, Hugh Bird"
@lisence: MIT
@status: "alpha"
"""
import numpy as np
from .ElemBaseClass import ElemBaseClass

class ElemTriBase(ElemBaseClass):
    """ Base class for triangular elements
    
    Element is a right angle triangle with vertices at
    (0,0), (1,0) and (0,1).
    
    Triangle base class adds methods to define 
    number of dimensions and generate Gauss Points
    """
    
    gen_gp_cache = {}
    
    @staticmethod
    def nd():
        return 2

    @staticmethod
    def local_in_element(local_coord):
        if local_coord[0] + ElemBaseClass._in_tol >= 0 \
                      and local_coord[1] + ElemBaseClass._in_tol >= 0 \
                      and local_coord[0] + local_coord[1] \
                      - + ElemBaseClass._in_tol <= 1:
            return True
        else:
            return False

    def gen_gp(self, gauss_order):
        """ Generate Guass points for a triangle.
        
        Based on Hussain et al 2012, Int J App. Math. and Comp.
        GQUTM (Gaussian Quadrature for Unit Triangles M?)
        
        Transforms square to right angle triangle.
        (-1,-1),(1,-1),(1,1),(-1,1) -> (0,0),(1,0),(0,1)
        The right edge of the square is collapsed to point.
        """
        assert(len(gauss_order) == 2)
        go1 = gauss_order[0] + 1
        go2 = gauss_order[1] + 1
        # Non symetric gauss points has not been implemented yet, so we take
        # the highest order we need in both directions. Sad, I know.
        gauss_order = max(gauss_order) + 1
        try:
            gp_2d, wp_2d = ElemTriBase.gen_gp_cache[(go1, go2)]
        except KeyError:
            def t(xsi):
                return 0.5*(1.0-xsi[:,0])*(1.0+xsi[:,1])
                
            def s(xsi):
                return 0.5*(1.0+xsi[:,0])
            
            # Output ngp and 1 dimension of Gauss points can be calculated now.
            ngp = int((gauss_order**2 + gauss_order)/2)
            xsi_p, xsi_w = self.gauss_legendre_1D(gauss_order)
            gp_2d = np.zeros((ngp, 2), dtype=np.float64)
            wp_2d = np.zeros((ngp, 1), dtype=np.float64)
            
            # whilst a indexes could probably be computed independently of 
            # the iteration, having a index to keep track is easier...
            idx = 0
            for i in range(gauss_order, 0, -1):
                # NGP in dir eta changes  to account for traingleness in transform.
                ngp_eta = int(gauss_order-i+1)
                eta_p, eta_w = self.gauss_legendre_1D(ngp_eta)
                # GP = [ const (at gp in xsi), eta gauss points]
                # Will be transformed to triangle after loop.
                gp_2d[idx:idx+ngp_eta,:] = np.row_stack((-xsi_p[i-1]*np.ones((1,ngp_eta)),
                                                          eta_p)).transpose()
                # Wp = (1-xsi)/8  *  W_xsi * W_eta (Already in triangle here)
                wp_2d[idx:idx+ngp_eta] = np.matrix(0.125*(1.0-xsi_p[i-1])*xsi_w[i-1]*eta_w).transpose()
                idx += ngp_eta
                
            # Transform GP to triangle
            gp_2d[:,0] = s(gp_2d)
            gp_2d[:,1] = t(gp_2d)
            # Swap dir if go2 > go1
            if go2 > go1:
                gp_2d_tmp = np.array(gp_2d[:,0])
                gp_2d[:, 0] = gp_2d[:, 1]
                gp_2d[:, 1] = gp_2d_tmp
            ElemTriBase.gen_gp_cache[(go1, go2)] = (gp_2d, wp_2d)
        return gp_2d, wp_2d
        