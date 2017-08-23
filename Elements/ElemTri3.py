# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@license: MIT
@status: alpha
"""
import numpy as np
from .ElemTriBase import ElemTriBase

class ElemTri3(ElemTriBase):
    """3 Noded triangle element
    
    v
    | 
    2\n
    |‘\ 
    | ‘\ 
    |  ‘\ 
    |   ‘\ 
    |    ‘\ \n
    0------1 --> u\n
    See source for ascii art!
    
    This class adds shape functions and derivative
    shape functions to ElemTriBase, in addition to
    number of endpoints function and default 
    gauss order function.
    """
    def  dnen(self):
        return 3
        
    def _shape_func_def_gauss_order(self):
        return (2, 2)
        
        
    @staticmethod
    def edge_adjacentcy():
        return [(2,1), (0,2), (1,0)]
        
    @staticmethod
    def node_locals():
        return [(0.0,0.0), (1.0,0.0), (0.0,1.0)]

    @staticmethod
    def shape_func(x):
        Nr = np.array([-1.0*(x[0] + x[1] - 1.0),
                    x[0],
                    x[1]])
        return Nr
        
    @staticmethod
    def der_shape_func(x):
        dNr = np.matrix([[-1.0, -1.0],
                         [1.0, 0.0],
                         [0.0, 1.0]])
        return dNr