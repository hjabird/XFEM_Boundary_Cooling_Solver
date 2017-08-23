# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@lisence: MIT
@status: alpha
"""
import numpy as np
from .ElemBaseClass import ElemBaseClass

class ElemQuadBase(ElemBaseClass):
    """ An abstract class for 2d quad elements 
    
    Elements should have unit square shape:
    (-1,1) ------ (1,1)
       |            |
       |            |
       |            |
    (-1, -1)------(1,-1)
    
    Class adds definition of nd() (number of dimensions 
    for element) and gauss point generation (gen_gp())
    """
    @staticmethod
    def nd():
        return  2
    
    
    @staticmethod
    def local_in_element(local_coord):
        if local_coord[0] + ElemBaseClass._in_tol >= -1.0 \
                and local_coord[0] - ElemBaseClass._in_tol <= 1.0 \
                and local_coord[1] + ElemBaseClass._in_tol >= -1.0 \
                and local_coord[1] - ElemBaseClass._in_tol <= 1.0:
            return True 
        else:
            return False
        
    def gen_gp(self, gauss_order):
        # gauss_order should be tuple of length 2.
        # gauss_order[0] = xi 1 dir
        # gauss_order[1] = xi 2 dir
        assert(len(gauss_order) == 2)
        gp_1d1, wp_1d1 = self.gauss_legendre_1D(gauss_order[0])
        gp_1d2, wp_1d2 = self.gauss_legendre_1D(gauss_order[1])
        gp_2dx, gp_2dy = np.meshgrid(gp_1d1, gp_1d2)
        wp_2dx, wp_2dy = np.meshgrid(wp_1d1, wp_1d2)
        ngp = gauss_order[0]*gauss_order[1]
        gp_2d = np.column_stack((np.reshape(gp_2dx, ngp),
                                np.reshape(gp_2dy, ngp)))
        wp_2d = np.reshape(wp_2dx, ngp)*np.reshape(wp_2dy, ngp)
        return gp_2d, wp_2d