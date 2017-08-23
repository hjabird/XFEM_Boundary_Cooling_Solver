# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@lisence: MIT
@status: alpha
"""

import numpy as np
import Accelerated as accl


class ElemBaseClass:
    _in_tol = 1e-11

    def __init__(self, node_coord_dict, nodes, gauss_order=0):
        #        if gauss_order == 0:
        #            self.gauss_order = self.def_gauss_order()
        #        else:
        #            self.gauss_order = gauss_order
        self.node_x = node_coord_dict
        self.nodes = tuple(nodes) #Node ids, not nodes themselves.
        self.enrichment = []

    def nen(self):
        """no. endpoints given to element\n
        endpoints == nodes"""
        return len(self.nodes)

    def dnen(self):
        """no. endpoints the element should have"""
        raise NotImplementedError

    @staticmethod
    def nd():
        """no. dimensions in element\n
        1D returns 1, 2D returns 2, etc..."""
        raise NotImplementedError

    def ndof(self):
        """No. degrees of freedom in an element.
        
        Includes enriched degrees of freedom.
        """
        return self.dnen() * (1 + len(self.enrichment))

    def node_coords(self):
        """ Return node coordinates for element.
        
        Returns numpy array of form:\n
        [[x1, y1, z1],\n
        [x2, y2, z2],\n
        ...,\n
        [xn, yn, zn]]\n
        Where the numbers correspond to the node number 
        of the element, and x y and z are cartesian coordinates.
        """
        coords = np.matrix([self.node_x[a] for a in self.nodes],
                           dtype=np.double)
        return coords

    def local_to_global(self, eta):
        """ Turn element coordinate to global coordinate
        
        Eta is a coordinate to translate to a global coordinate. 
        """
        coords = self.node_coords()
        glob = np.matmul(self.shape_func(eta), coords)
        return np.squeeze(np.asarray(glob))

    def global_to_local1(self, x, eta=None):
        """ Turn global coordinate to element coordinate
        
        x is a global coordinate [x, y, z]. 
        Assumed to be in element domain.
        eta is an optional argument that is a guess of the local coord of
        x in the element.
        """
        # Solving to find f directly might be difficult - the inverse
        # of the element shape function would presumable have to be known and
        # then solved simultaneously, which would be a major pain. Instead,
        # we'll try and use an iterative method. 1 step for linear elements.
        if eta is None: eta = np.ones(self.nd()) * 0.0000000001;
        # Newton Raphson iteration:
        # x_{n+1} = x_{n} - (f(x_{n})) / (f'(x_{n})):
        # Problem: it is possible for one dimension to be tangential to the
        # direction of the error vector, leading to f' having a zero. This
        # would then result in a division by zero.
        # Solution: if there f'[i] = 0, f is not a function of eta_i, and so
        # this compenent can be ignored for the current iteration.

        def f(c): return np.array(self.local_to_global(c) - x)

        def f_prime(c):
            a = self.jacobian(c)
            return np.array(a.sum(axis=1)).squeeze()

        iterations = 0
        a = f(eta)
        from .ElemTriBase import ElemTriBase
        if issubclass(self.__class__, ElemTriBase) \
            and x[0] < 1.4 \
            and x[1] > 1.0 \
            and x[1] < x[0] + 0.3:
            mprint = print
        else:
            def mprint(c): pass
        
        mprint("\n\nWorking on Tri.")
        mprint("X:\t" + str(x))
        while np.linalg.norm(a) > 1e-12:
            f_p_actual = f_prime(eta)
            mprint("F_p:\t" + str(f_p_actual))
            is_zero = np.equal(f_p_actual, 0.0)
            modifier = np.divide(a[0:self.nd()], f_p_actual[0:self.nd()])
            modifier[is_zero] = 0
            eta = eta - modifier
            mprint("Eta:\t" + str(eta))
            a = f(eta)
            mprint("Error:\t" + str(a))
            iterations += 1
            if iterations > 10:
                #raise AssertionError
                break
        return eta
        
    def global_to_local(self, x, eta=None):
        """ Turn global coordinate to element coordinate
        
        x is a global coordinate [x, y, z]. 
        Assumed to be in element domain.
        eta is an optional argument that is a guess of the local coord of
        x in the element.
        """
        # Solving to find f directly might be difficult - the inverse
        # of the element shape function would presumable have to be known and
        # then solved simultaneously, which would be a major pain. Instead,
        # we'll try and use an iterative method. 1 step for linear elements.
        if eta is None: eta = np.ones(self.nd()) * 0.0000000001;
        # Newton Raphson iteration:
        # x_{n+1} = x_{n} - (f(x_{n})) / (f'(x_{n})):
        # Problem: it is possible for one dimension to be tangential to the
        # direction of the error vector, leading to f' having a zero. This
        # would then result in a division by zero.
        # Solution: if there f'[i] = 0, f is not a function of eta_i, and so
        # this compenent can be ignored for the current iteration.

        def f(c): return np.array(self.local_to_global(c) - x)
        # df is jacobian.transpose()
        iterations = 0
        a = f(eta)

        ndim = self.nd()
        while np.linalg.norm(a) > 1e-12:
            f_p = self.jacobian(eta).transpose()#
            u =  np.matmul(f_p, np.matrix(eta).transpose()) - np.matrix(a[0:ndim]).transpose()#
            eta = np.array(np.linalg.solve(f_p, u).transpose()).squeeze()
            a = f(eta)
            iterations += 1
            if iterations > 100:
                #raise AssertionError
                break
        return eta
        
    def is_near(self, point_ids_set, point_dict):
        """ Returns a set of points close to element.
        
        Given a set of points, the set is reduced to one
        which is near the element. Returned points are 
        not guarenteed to be in the element. 
        """
        out_set = set()
        # Make a box:
        ncs = self.node_coords()
        maxs = ncs.max(0)
        mins = ncs.min(0)
        tol = (maxs-mins) * (0.05)
        maxs += tol
        mins -= tol
        # See whats in the box.
        for point_id in point_ids_set:
            pc = point_dict[point_id]
            if not np.greater(pc, maxs).any() \
                and not np.less(pc, mins).any():
                out_set.add(point_id)
        # print("is_near:\t"+str(len(out_set))+" of " + 
            # str(len(point_ids_set)) + " in box.")
        return out_set
        
    
    @staticmethod
    def local_in_element(local_coord):
        """Returns true if a local coord is in/on element"""
        raise NotImplementedError

    @staticmethod
    def edge_adjacentcy():
        """Returns a list where l[i] returns tuple of nodes adjacent to node i
        """
        raise NotImplementedError

    @staticmethod
    def node_locals():
        """Returns the local coordinate of each node.
        
        In form: [(x0, y0), (x1,y1), ... , (xdnen, ydnen)]
        """
        raise NotImplementedError

    @staticmethod
    def der_shape_func(x):
        """Returns partial derivatives of element shape functions at given 
        point\n
        
        Arugments:
         x: A coordinate for der shape functions to be evaluated at.
         
        Output:
        Derivatives of nodal shape functions evaluated at coordinates given
        by x:\n
        DNr = [[dN1/dx@x, dN1/dy@x]\n
               [dN2/dx@x, dN2/dy@x]\n
               [dN3/dx@x, dN3/dy@x]]\n"""
        raise NotImplementedError

    @staticmethod
    def shape_func(x):
        """Returns element shape functions evaluated at given point\n
        
        Arugments:
         x: A coordinate for shape functions to be evaluated at.
         
        Output:
        Derivatives of nodal shape functions evaluated at coordinates given
        by x:\n
        DNr = [[N1@x]\n
               [N2@x]\n
               [N3@x]]\n"""
        raise NotImplementedError

    def def_gauss_order(self):
        """The default number of guass points in each dimension for
        the element.
        
        Gauss points are used to integrate the strain over an element.
        Typically, the number of gauss points are used are the same as
        number of nodes along an elements edge. Example, 4 noded
        quad would use 4 GPs.
        
        Gaussian integration is not the only way to go, but for polynomial 
        shape function cannot be beaten. For GFEM or XFEM, which have 
        non-polynomial enrichment, more Gauss points would be needed.
        
        To adjust for this, this returns the minimum of the suggested gauss
        order of the shape functions or the enrichment.

        Returns tuple (for anisotropic GPs as might be needed with
        enrichment.
        """
        n = self._shape_func_def_gauss_order()

        # Given 2 tuples get a tuple that is the elementwise maximum.
        # Ie: a = (1,6), b = (5,4) return (5,6)
        def elemwise_max(inp1, inp2):
            return tuple(map(max, zip(inp1, inp2)))

        for e in self.enrichment:
            a = e.enrichment_gauss_order()
            n = elemwise_max(a, n)
        return n

    def _shape_func_def_gauss_order(self):
        """ The default number of G-L points to use when integrating over the
        element in 1D

        Should be tuple with entries = number of dimensions of element.
        """
        raise NotImplementedError

    def correct_nen(self):
        """ True if the element has been given correct no. end nodes """
        return self.dnen() == self.nen()

    def gauss_legendre_1D(self, n):
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
        return accl.gl_1D(n)

    def gen_gp(self, guass_order):
        """ Generate multidimensional gauss points with internal element
        coordinate system.\n
        Arguments:\n
         gauss_order: number of guass points per dimension.\n
        Returns:\n
         coords: numpy matrix of form [[xsi_1, eta_1],...,[xsi_n, eta_n]] or
         equivalent for however many dimensions (xsi, eta, zet)\n
         weights: numpy array of for [wp1, wp2, wp3, ..., wpn] for weighting
         of points at given coords.
        """
        raise NotImplementedError

    def funcs(self, x):
        """ Element functions without coefficients.
        
        Defaults to N_{i} - the polynomial shape function
        unless modified by enrichment or otherwise.
        """
        dnen = self.dnen()
        f = np.zeros(dnen * (len(self.enrichment) + 1), dtype=np.float64)
        f[0:dnen] = self.shape_func(x)
        idx = dnen
        for enrch in self.enrichment:
            f[idx:idx + dnen] = f[0:dnen] * enrch.func(x)
            idx += dnen
        return f

    def der_funcs(self, x):
        """ Element derivative function matrix without coefficients.
        
        Defaults to N_{i,(x,y,...)} - the polynomial sf 
        derivatives unless modified.
        """
        dnen = self.dnen()
        nd = self.nd()
        f = np.zeros((dnen * (len(self.enrichment) + 1), nd),
                     dtype=np.float64)
        f[0:dnen, 0:nd] = self.der_shape_func(x)
        f_nd = self.shape_func(x)
        idx = dnen
        for enrch in self.enrichment:
            f[idx:idx + dnen, :] = f[0:dnen, :] * enrch.func(x) \
                                   + np.outer(f_nd, enrch.der_func(x))
            idx += dnen
        return f

    def elem_node_tag_gen(self):
        """ Generate tags for nodes which are different for enrichment.
        
        tag:
        (nid, <function attached to coeff>)
        """
        tags = []
        for nid in self.nodes:
            tags.append((nid, 0))
        for enrch in self.enrichment:
            for nid in self.nodes:
                tags.append((nid, enrch.ident()))
        return tags

    def jacobian(self, x):
        """ Calculates Jacobian of element at points x
        
        Returns matrix as square nd() * nd() matrix: \n
        dx/d(eta0)     dy/d(eta0) ... \n
        dx/d(eta1)     dy/d(eta1) ... \n
        dx/d(eta2)     dy/d(eta2) ...
        """
        coords = self.node_coords()
        dsf = self.der_shape_func(x)
        J = np.matmul(dsf.transpose(), coords[:, 0:self.nd()])
        return J

    def extract_elem_sol(self, node_map, solution):
        """ Extract a solution vector for individual element from a global 
        solution.
        
        elem is an element\n
        node_map is a node mapping\n
        solution is a np.array containing solution.
        """
        tags = self.elem_node_tag_gen()
        idxs = node_map.tags_to_idxs(tags)
        return solution[idxs]

    def eval_elem(self, node_map, solution, eta):
        """ Takes the value of the element at given positions using given 
        coeffs.
        
        node_map: Node_mapping object\n
        solution: a global solution corresponding to node_mapping\n
        eta: a list of tuples(eta1, eta2,...) at which to evaluate the element
        """
        coeffs = self.extract_elem_sol(node_map, solution)

        fx = []
        for a in eta:
            fx.append(np.sum(np.dot(coeffs, self.funcs(a))))
        return fx

    def guess_normal_vector_local(self, eta):
        """ For a point on the boundary of the element, guess the direction
        of the normal to the boundary
        """
        # METHOD
        # Assume that eta is already on element boundary, but not on a node.
        # 1: Find the vector from eta to all nodes (local coords)
        # 2: Find the minimum of the dot product of all the vectors from step 1
        # 3: The pair with the minimum dot product must be vectors in opposite 
        # directions, and therefor lead to nodes on the edge of the element.
        # 4: Take the difference of this pair to find a vector that we think
        # is the direction of the edge at eta.
        # 5: Generatea normal to the edge by {x,y} = {-y, x}
        # 6: Test whether this points into the element. This can be done by
        # summing all vectors from 1 and taking the dot product with our normal
        # vector. If the dot product is positive, reverese the direction of 
        # the normal.
        # 7: Make unit vector, becuase that just seems like a nice thing to do.

        # 1
        loc_coords = self.node_locals()
        eta_to_point = []
        for loc in loc_coords:
            eta_to_point.append(np.array(loc) - np.array(eta))
        # 2 / 3
        min_dot = 1.0
        pair = (-1, -1)
        for i in range(0, len(eta_to_point)):
            for j in range(i, len(eta_to_point)):
                cos_ang = np.dot(eta_to_point[i], eta_to_point[j]) \
                          / (np.linalg.norm(eta_to_point[i])
                             * np.linalg.norm(eta_to_point[j]))
                if cos_ang < min_dot:
                    min_dot = cos_ang
                    pair = (i, j)
        # 4
        a = eta_to_point[pair[0]] - eta_to_point[pair[1]]
        # 5
        n = np.array([-a[1], a[0]])
        # 6
        vect_sum = np.array(eta_to_point[0])
        for i in range(1, len(eta_to_point)):
            vect_sum += eta_to_point[i]
        if np.dot(vect_sum, n) > 0:
            n = -n
        # 7
        return n / np.linalg.norm(n)

    def guess_normal_vector_global(self, eta):
        """ From a point on the boundary of the element (local coords), guess
        the direction of the normal to the boundary """
        # Initially I used the jacobian to transform local normal to global
        # normal, but this is wrong - you want to use the coordinate deriv
        # tangential to face, but use the one which is normal locally, but
        # not generally globaly.

        local_normal = self.guess_normal_vector_local(eta)
        # Transform 90 degrees so tangential to face before transform:
        rot90 = np.matrix(((0, -1), (1, 0)), dtype=np.float64)
        tangential = np.matmul(local_normal, rot90)
        J = self.jacobian(eta)
        global_tangential = np.matmul(tangential, J)
        # Rotate back and convert to array.
        if np.linalg.det(J) > 0:
            global_n = np.matmul(global_tangential, -1 * rot90)
        else:
            global_n = np.matmul(global_tangential, rot90)
        global_n = np.asarray(global_n).squeeze()
        # Everyone loves a unit vector, right?
        return global_n / np.linalg.norm(global_n)

    def _add_enrichment(self, enrichment):
        """ Friend function to enrichment
        
        Add to the enrichement list. "enrichment" is Enrichment class instance.
        """
        this_id = enrichment.ident()
        for enrich in self.enrichment:
            try:
                assert(this_id != enrich.ident())
            except:
                print("######################################################")
                print("Elements:\tSETUP ERROR!")
                print("Elements:")
                print("Elements:\tTried to add enrichment to element.")
                print("Elements:\tEnrichment ID is already assigned in element.")
                try:
                    print("Elements:\tProblem enrichment ID:\t"+str(this_id))
                except:
                    print("Element:\t[Enrichment ID could not be converted to" \
                        + " string for error message]")
                print("Elements:\tHave you define the same enrichment twice?")
                print("######################################################")
                raise AssertionError
        self.enrichment.append(enrichment)
