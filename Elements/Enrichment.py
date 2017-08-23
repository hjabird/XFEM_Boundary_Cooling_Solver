# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@licence: MIT
@status: alpha
"""

class Enrichment:
    """ Enrichment class.
    """
    def __init__(self, elem_to_enrich_class, ident):
        try:
            assert(ident != 0)
        except:
            print("Enrichment:\tFATAL ERROR!")
            print("Bad enrichment tag. Enrichment identety '0' was used"
                  + " in enrichment setup. This is reserved for for element"
                  + " shape functions. Please use another number.")
            raise AssertionError
        self.enrichment_type = elem_to_enrich_class
        self.identity = ident
        self.local_transforms = {}
        self.gauss_order = self.enrichment_type.nd() * (30,)
    
    def define_func(self, func):
        """ Define the function to use for enrichement"""
        self.function = func
        
    def define_deriv_func(self, deriv_func):
        """ Define the derivative of the enrichment"""
        self.deriv_function = deriv_func
        
    def define_gauss_order(self, order):
        """ Define the gauss order per dimension. Should be tuple[nd]"""
        assert(type(order) is tuple)
        self.gauss_order = order
        
    def enrichment_gauss_order(self):
        """ Returns the minimum ngp per dimension for this enrichment"""
        return self.gauss_order
    
    def func(self, eta):
        """ Returns the enrichement evaluated at local coordinate eta
        """
        return self.function(eta)
    
    def der_func(self, eta):
        """ Returns the dG/dx_n evaluated at local coordinate eta
        
        numpy array [dG/dx, dG/dy...]
        """
        return self.deriv_function(eta)
    
    def ident(self):
        """ Returns an identifier for the enrichement.
        """
        return self.identity
    

    def enrich_elem(self, elem):
        """ Enrich element given by elem.
        """
        assert(issubclass(elem.__class__, self.enrichment_type))
        elem._add_enrichment(self)
        # A transformation so enrichment works in the element's coordinate
        # system as opposed to the coordinates the enrichement was defined
        # in.
        self.local_transforms[elem] = 1