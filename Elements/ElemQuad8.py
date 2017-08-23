# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@licence: MIT
@status: alpha
"""
import numpy as np
from .ElemQuadBase import ElemQuadBase

class ElemQuad8(ElemQuadBase):
    """ A 8 noded quad element

    3 -- 6 -- 2 \n
    |         | \n
    7         5 \n 
    |         | \n
    0 -- 4 -- 1\n
    (See ascii art in source file)
    """ 
    def dnen(self):
        return 8
        
    def _shape_func_def_gauss_order(self):
        return (3, 3)
        
    @staticmethod
    def edge_adjacentcy():
        x = [(7,4),
             (4,5),
             (5,6),
             (6,7),
             (0,1),
             (1,2),
             (2,3),
             (3,0)]
        return x
        
        
    @staticmethod
    def node_locals():
        return [(-1.0, -1.0),
                (1.0, -1.0),
                (1.0, 1.0),
                (-1.0, 1.0),
                (0.0, -1.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (-1.0, 0.0)]
        
    @staticmethod
    def shape_func(x):
        Nr = np.array([   -0.25*(x[0]-1.0)*(x[1]-1.0)*(x[0]+x[1]+1.0), #0
                 0.25*(x[0]+1.0)*(x[1]-1.0)*(x[1]-x[0]+1.0), #1
                 0.25*(x[0]+1.0)*(x[1]+1.0)*(x[0]+x[1]-1.0), #2
                 -0.25*(x[0]-1.0)*(x[1]+1.0)*(x[1]-x[0]-1.0), #3
                 0.5*(x[0]+1.0)*(x[0]-1.0)*(x[1]-1.0), #4
                 -0.5*(x[0]+1.0)*(x[1]-1.0)*(x[1]+1.0), #5
                 -0.5*(x[0]+1.0)*(x[0]-1.0)*(x[1]+1.0), #6
                 0.5*(x[0]-1.0)*(x[1]-1.0)*(x[1]+1.0)#7              
                  ])
        return Nr
        
    @staticmethod
    def der_shape_func(x):
        dNr = np.matrix([[-0.25*(x[1]-1.0)*((x[0]+x[1]+1.0)+(x[0]-1.0)), # nd0 / 0
               -0.25*(x[0]-1.0)*((x[0]+x[1]+1.0)+(x[1]-1.0))], # nd0 / 1
              [0.25*(x[1]-1.0)*((x[1]-x[0]+1.0)-(x[0]+1.0)),  # nd1 / 0
               0.25*(x[0]+1.0)*((x[1]-x[0]+1.0)+(x[1]-1.0))], # nd1 / 1
              [0.25*(x[1]+1.0)*((x[0]+x[1]-1.0)+(x[0]+1.0)), # nd2 / 0
               0.25*(x[0]+1.0)*((x[0]+x[1]-1.0)+(x[1]+1.0))], # nd2 / 1
              [-0.25*(x[1]+1.0)*((x[1]-x[0]-1.0)-(x[0]-1.0)), # nd3 / 0
               -0.25*(x[0]-1.0)*((x[1]-x[0]-1.0)+(x[1]+1.0))], # nd3 / 1
              [x[0]*(x[1]-1.0),  # nd4 / 0
               0.50*(x[0]**2-1.0)], # nd4 / 1
              [-0.50*(x[1]**2-1.0),  # nd5 / 0
               -x[1]*(x[0]+1.0)], # nd5 / 1
              [-x[0]*(x[1]+1.0), #nd6 / 0
               -0.50*(x[0]**2-1.0)], #nd6 / 1
              [0.50*(x[1]**2-1.0),# nd7 /0
               x[1]*(x[0]-1.0)]]) # nd7 /1
        return dNr