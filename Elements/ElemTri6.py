# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@lisence: MIT
@status: alpha
"""
import numpy as np
from .ElemTriBase import ElemTriBase

class ElemTri6(ElemTriBase):
    """6 Noded triangle element
    
    2\n
    | \ 
    |  ‘\ 
    5   ‘4\n
    |     ‘\
    |       ‘\ 
    0-----3----1\n
    See source for ascii art!
    
    This class adds shape functions and derivative
    shape functions to ElemTriBase, in addition to
    number of endpoints function and default 
    gauss order function.
    """    
    def dnen(self):
        return 6
        
    def _shape_func_def_gauss_order(self):
        return (3, 3)
        
    @staticmethod
    def edge_adjacentcy():
        return [(5,3), (3,4), (4,5), (0,1), (1,2), (2,0)]

    @staticmethod
    def node_locals():
        return [(0.0,0.0),
                (1.0,0.0),
                (0.0,1.0),
                (0.5,0.0),
                (0.5,0.5),
                (0.0,0.5)]
        
    @staticmethod
    def shape_func(x):
        Nr = np.array([2.0*(x[0]+x[1]-1.0)*(x[0]+x[1]-0.5), #0
                        2.0*(x[0])*(x[0]-0.5), #1
                        2.0*(x[1])*(x[1]-0.5), #2
                        -4.0*(x[0])*(x[0]+x[1]-1.0), #3
                        4.0*(x[0])*(x[1]), #4
                        -4.0*(x[0]+x[1]-1.0)*x[1]]) #5
        return Nr
    
    @staticmethod
    def der_shape_func(x):
        dNr = np.matrix([[4.0*(x[0]+x[1])-3, # 0/ 0
                          4.0*(x[0]+x[1])-3], # 0/1
                        [4.0*x[0]-1.0, #1/0
                         0.0], #1/1
                        [0.0, #2/0
                         4.0*x[1]-1.0], #2/1
                        [-4.0*(2*x[0]+x[1]-1.0),#3/0
                         -4.0*x[0]],#3/1
                        [4.0*x[1], #4/0
                         4.0*x[0]], #4/1
                        [-4.0*x[1], #5/0
                         -4.0*(x[0]+2.0*x[1]-1)]]) #5/1
        return dNr