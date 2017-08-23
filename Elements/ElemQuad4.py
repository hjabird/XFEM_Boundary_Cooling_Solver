# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@license: MIT
@status: alpha
"""
import numpy as np
from .ElemQuadBase import ElemQuadBase

class ElemQuad4(ElemQuadBase):  
    """ A 4 noded quad element

    3 ----- 2 \n
    |       |
    |       | 
    0 ----- 1\n
    (See ascii art in source file)
    """      
    def dnen(self):
        return 4
        
    def _shape_func_def_gauss_order(self):
        return (2, 2)
        
    @staticmethod
    def edge_adjacentcy():
        return [(3,1), (0,2), (1,3), (2,0)]

    @staticmethod
    def node_locals():
        return [(-1.0, -1.0),
                (1.0, -1.0),
                (1.0, 1.0),
                (-1.0, 1.0)]
                
    @staticmethod
    def shape_func(x):
        Nr = np.array([0.25*(x[0] - 1)*(x[1] - 1),
              -0.25*(x[0] + 1)*(x[1] - 1), 
              0.25*(x[0] + 1)*(x[1] + 1),
              -0.25*(x[0] - 1)*(x[1] + 1)])
        return Nr
        
    @staticmethod
    def der_shape_func(x):
        dNr = np.matrix([[0.25*(x[1]-1),0.25*(x[0]-1) ],
                [-0.25*(x[1]-1), -0.25*(x[0]+1) ],
                [0.25*(1+x[1]), 0.25*(1+x[0]) ],
                [-0.25*(x[1]+1), -0.25*(x[0]-1) ]])         
        return dNr