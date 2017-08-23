# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@lisence: MIT
@status: alpha
"""
import numpy as np
from .ElemLineBase import ElemLineBase

class ElemLine3(ElemLineBase):
    """ A 3 noded linear line element
    
    0 ----- 2 ----- 1\n
    (See source ascii art!)
    """
    def dnen(self):
        return 3
        
    def _shape_func_def_gauss_order(self):
        return (3,)
        
    @staticmethod
    def edge_adjacentcy():
        return [(2,), (0,1), (2,)]

    @staticmethod
    def node_locals():
        return [(-1.0,), (1.0,), (0.0,)]

    @staticmethod
    def shape_funct(x):
        Nr = np.array([ 0.5*(x[0]-1.0)*x[0],
                        0.5*(x[0]+1.0)*x[0],
                        -1.0*(x[0]-1.0)*(x[0]+1.0)])
        return Nr
        
    @staticmethod
    def der_shape_funct(x):
        dNr = np.matrix([[x[0]-0.5],
                         [x[0]+0.5]],
                         [-2.0*x[0]])
        return dNr

