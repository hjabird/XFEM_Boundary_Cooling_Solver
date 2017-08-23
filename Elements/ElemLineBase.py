# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@licence: MIT
@status: alpha
"""
import numpy as np
from .ElemBaseClass import ElemBaseClass

class ElemLineBase(ElemBaseClass):
    """ Base class for 1D line element
    
    (-1)------(1)
    
    Class adds nd() and gen_gp()
    """
    @staticmethod
    def nd():
        return 1;
        
    def gen_gp(self, gauss_order):
        # gauss_order should be 1 element tuple.
        assert(len(gauss_order) == 1)
        return self.gauss_legendre_1D(gauss_order[0])