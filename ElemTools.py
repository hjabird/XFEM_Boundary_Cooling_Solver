# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@lisence: MIT
@status: alpha
"""
import numpy as np
import scipy.sparse as ssparse
import time as python_time
from multiprocessing import Pool

class j_cache:
    """ Cache element jacobians and jacobian determinants.
    """
    cj = {}
    cdet_j = {}
    def __init__(self):
        pass
    
    @staticmethod
    def det_j(elem, eta):
        coord = tuple(a for a in eta)
        try:
            return j_cache.cdet_j[(elem, coord)]
        except KeyError:
            j_cache._add_to_cache(elem, eta, coord)
            return j_cache.cdet_j[(elem, coord)]
    
    @staticmethod
    def j(elem, eta):
        coord = tuple(a for a in eta)
        try:
            return j_cache.cj[(elem, coord)]
        except KeyError:
            j_cache._add_to_cache(elem, eta, coord)
            return j_cache.cj[(elem, coord)]
            
    @staticmethod
    def _add_to_cache(elem, eta, coord):
        j = elem.jacobian(eta)
        # Force positive jacobian.... 
        detj = abs(np.linalg.det(elem.jacobian(eta)))
        try:
            # You might have noted this isn't going to happen right now...
            assert(detj > 0)
        except AssertionError:
            print("\n\n\n")
            print("##########################################################")
            print("FATAL ERROR!")
            print("Encountered bad jacobian! (Negative)")
            print("Element type" +  str(elem.__class__))
            print("Local coordinates are normally: \n")
            print(str(elem.node_locals()))
            print("Actual node coordinates:\n")
            print(str(elem.node_coords()))
            print("Resulted in a jacobian matrix:")
            print(str(j))
            print("With det(J) = " + str(detj))
            print("##########################################################")
            raise AssertionError
        j_cache.cj[(elem, coord)] = j
        j_cache.cdet_j[(elem, coord)] = detj
            

def uv_mtrx(elem, eta):
    """ Compute product the outer product of the sf at x"""
    feval_local = elem.funcs(eta)
    fp = np.outer(feval_local, feval_local)
    return fp

def gu_gv_mtrx(elem, eta):
    """ Compute grad of element shape functions.
        
    grad(v) (DOT) grad(u) at x
    """
    grad = np.linalg.solve(j_cache.j(elem, eta), 
                           elem.der_funcs(eta).transpose())
    gd = np.zeros((elem.ndof(), elem.ndof()))
    for i in range(0, elem.nd()):
        gd += np.outer(grad[i, :], grad[i, :])
    #print("DEBUG/ElemTools/gu_gv_mtrx:\tgd = \n" + str(gd))
    return gd
    
    
def integrate_elem(elem, funct, gauss_ord=None, gauss_mult=1):
    """ Integrates f(x) over an element's area.
    
    int(f(x), dA) =~ sum(w_i * f(p_i))\n
    Args
    
     elem: An element\n
     funct: a function that takes the element and a point as arguments.
        returns f(x), be it a matrix or otherwise.\n
     gauss_ord: optional (default=element.def_gauss_ord()). Order of 
        integration to use over element.\n
     gauss_mult: Multiple of default gauss num gauss points to use.\n
    
    Returns
      integral: weighted sum of funct at different points.\n
    
    """
    # Gen mtrx indexes for nodes
    if gauss_ord is None and gauss_mult == 1:
        gauss_ord = elem.def_gauss_order()
    elif gauss_ord is None:
        def smult(inp):
            return gauss_mult * inp
        gauss_ord = tuple(map(smult, elem.def_gauss_order()))
    points, weights = elem.gen_gp(gauss_ord)
    ngp = len(weights)
    
    integral = weights[0] * funct(elem, points[0, :])* \
                    j_cache.det_j(elem, points[0, :])
    for i in range(1, ngp):
        integral += weights[i] * funct(elem, points[i, :]) * \
                        j_cache.det_j(elem, points[i, :])
    return integral
    
    
def elems_2_csc(mesh, funct, node_mapping, gauss_ord=None):
    """Assemble a global matrix for function "funct"
    
    mesh is mesh object.\n
    funct is a function(elem, eta) over which to integrate.\n
    node_mapping is NodeMapping object.\n \n
    
    Optional:\n
    gauss_ord is dictionary. If node tag is not found in dictionary default
    gauss order will be used. If no dictionary the all is defualt gauss order.
    """
    ticy = python_time.clock()
    print("elems_2_csc:\tIntegrating "+funct.__name__+" mesh.")
    # Find the number of entries that we'll get from all our element matrices.
    numijk = 0
    for elem in mesh.elems.values():
        numijk += (elem.dnen() * (1 + len(elem.enrichment))) ** 2
    # Allocate arrays to store the ijk data before matrix assembly.
    row = np.zeros(numijk, dtype=np.int32)
    col = np.zeros(numijk, dtype=np.int32)
    data = np.zeros(numijk, dtype=np.float64)
    
    position = 0
    for eleid, elem in mesh.elems.items():
        # Check guass order.
        try:
            go = gauss_ord[eleid]
            intermed = integrate_elem(elem, funct, gauss_ord=go)
        except:
            intermed = integrate_elem(elem, funct)
        # Add to our ijk data.
        idxs = node_mapping.tags_to_idxs(elem.elem_node_tag_gen())
        ndof = len(idxs)**2
        xv, yv = np.meshgrid(idxs, idxs)
        xv = np.reshape(xv, -1)
        yv = np.reshape(yv, -1)
        intermed = np.reshape(intermed, -1)
        col[position: position + ndof] = xv
        row[position: position + ndof] = yv
        data[position: position + ndof] = intermed
        position += ndof
    
    glo_ndof = node_mapping.count
    mtrx = ssparse.coo_matrix((data, (row, col)), shape=(glo_ndof, glo_ndof))
    tot_time = python_time.clock() - ticy
    print("elems_2_csc:\tGenerated " +str(mtrx.shape)+" matrix with nnz " +\
          str(mtrx.nnz) + " in " + "{:10.4f}".format(tot_time) + " s.")
    return mtrx.tocsc()
    
def elems_2_array(mesh, funct, node_mapping, gauss_ord=None, gauss_mult=1):
    """Assemble a global array for function "funct" from all elems in mesh
    
    mesh is mesh object.\n
    funct is a function(elem, eta) over which to integrate.\n
    node_mapping is NodeMapping object.\n \n
    
    Optional:\n
    gauss_ord is dictionary. If node tag is not found in dictionary default
    gauss order will be used. If no dictionary the all is defualt gauss order.
    """
    ticy = python_time.clock()
    print("elems_2_array:\tIntegrating "+funct.__name__+" mesh.")

    vect = np.zeros(node_mapping.count, dtype=np.float64)    
    for eleid, elem in mesh.elems.items():
        # Check guass order.
        try:
            go = gauss_ord[eleid]
        except:
            go = elem.def_gauss_order() * gauss_mult
            
        intermed = integrate_elem(elem, funct, gauss_ord = go)
        # Add to our ijk data.
        idxs = node_mapping.tags_to_idxs(elem.elem_node_tag_gen())
        #print("DEBUG/ElemTools/elems_to_array:\tintermed = " + str(intermed))
        vect[idxs] += intermed
        
    tot_time = python_time.clock() - ticy
    print("elems_2_array:\tGenerated " +str(vect.shape)+" array in "
           + "{:10.4f}".format(tot_time) + " s.")
    return vect

## Parallel version not working yet or benchmarked.
#def elems_2_array_p(mesh, funct, node_mapping, gauss_ord=None):
#    """Assemble a global array for function "funct" from all elems in mesh
#    
#    mesh is mesh object.\n
#    funct is a function(elem, eta) over which to integrate.\n
#    node_mapping is NodeMapping object.\n \n
#    
#    Optional:\n
#    gauss_ord is dictionary. If node tag is not found in dictionary default
#    gauss order will be used. If no dictionary the all is defualt gauss order.
#    """
#    ticy = python_time.clock()
#    print("elems_2_array_p:\tIntegrating "+funct.__name__+" mesh.")
#
#    vect = np.zeros(node_mapping.count, dtype=np.float64)    
#    pool = Pool()
#    se_args = []
#    idx_mapping = []
#    # Generate se_args
#    for eleid, elem in mesh.elems.items():
#        se_args.append((funct, elem))
#        idx_mapping.append(node_mapping.tags_to_idxs(elem.elem_node_tag_gen()))
#    
#    # sols are small numpy arrays that need to be added to the main one.
#    solns = pool.map(single_elem_2_array_p, se_args)
#    
#    for i in range(len(se_args)):
#        vect[idx_mapping[i]] += solns[i]
#        
#    tot_time = python_time.clock() - ticy
#    print("elems_2_array_p:\tGenerated " +str(vect.shape)+" array in "
#           + "{:10.4f}".format(tot_time) + " s.")
#    return vect
    
    
def single_elem_2_array_p(one_arg):
    """A function for a single element integration for multiprocessing
    """
    elem, funct = one_arg
    intermed = integrate_elem(elem, funct)
    return intermed
    
    
def integrate_edge(elem, edge_nids, funct, gauss_order=None, gauss_mult=1):
    """ Integrate over the edge given by edge nids
    
    CONTAINS POTENTIAL BUG!
    
    Currently, all edge_nids must be on the edge of the element or who knows
    what will happen. Integration is currently only between adjacent nodes
    on an edge, not along the whole edge. This is an area for improvement.
    
    elem: The element we're interested in.\n
     edge_nids: integration edge node node ids/ tags. \n
     funct: a function(elem, eta) that we want to integrate over.\n
    Optional:\n
        gauss_order: number of gauss points to use in integration. Defaults to 
    elem.def_gauss_order()\n
     gauss_mult: Multiple of default gauss num gauss points to use.\n
    
    ___\n
    BUG:\n
    An two consecutive nodes on an element can be on the edges of a domain,
    but the edge of the element itself may not be. Eg:(See ascii art 
    in source code)\n
    |\n
    2\n
    | '\n
    |  '\n
    0---1----\n
    Both edges 1 & 2 are on boundary, but edge 1-2 is not.    
    """
    # Gen mtrx indexes for nodes
    if gauss_order is None:
        gauss_order = max(elem.def_gauss_order()) * gauss_mult
    points, weights = elem.gauss_legendre_1D(gauss_order)
    ngp = len(weights)
    # We write signitures for each edge we've already done, since we'll
    # get A-B and B-A otherwise. If B-A is about to be done we'd check if
    # A-B is already in the set.
    done = set()
    # Convert node tags to element local node indexes.
    nids = elem.nodes
    loc_ids = []
    node_locals = elem.node_locals()
    for nid in edge_nids:
        loc_ids.append(nids.index(nid))
    # Now work through node connectivities integrating:
    for loc_id in loc_ids:
        con = elem.edge_adjacentcy()[loc_id]
        for con_id in con:
            if con_id in loc_ids and (con_id, loc_id) not in done:
                done.add((loc_id, con_id))
                #print("DEBUG/ElemTools/integrate_edge:\tElem:" 
                #      + str(elem))
                #print("DEBUG/ElemTools/integrate_edge:\tEdge:"
                #      + str((loc_id, con_id)))
                
                # And we can FINALLY integrate over something.
                # Since we don't know what a whole edge is without adding this
                # to the elements, we'll just integrate between adjacent nodes
                int_vector = np.array(node_locals[con_id]) - \
                                      np.array(node_locals[loc_id])
                vec_len = np.linalg.norm(int_vector)
                # Weights add up to 2 in 1D Gauss, so /2.0
                int_wghts = weights * vec_len / 2.0
                int_pts = np.repeat( np.matrix(node_locals[loc_id]+
                                             int_vector/2.0),
                                  len(weights), axis=0) \
                                  +np.outer(points, 0.5*int_vector)
                toarr = lambda x: np.squeeze(np.asarray(x))
                #print("EDGE INT ON ELEM")
                #print(elem.__class__)
                #n = elem.node_coords()
                #print(n[loc_id])
                #print(n[con_id])
                #print(int_vector)
                todx = lambda eta: np.linalg.norm(
                        np.matmul(int_vector/vec_len,  j_cache.j(elem, eta)))
                try:
                    for i in range(0, ngp):
                        p = toarr(int_pts[i, :])
                        integral += int_wghts[i] * funct(elem, p) * todx(p)
                except NameError:
                    p = toarr(int_pts[0, :])
                    integral = int_wghts[0] * funct(elem, p) * todx(p)
                    for i in range(1, ngp):
                        p = toarr(int_pts[i, :])
                        #print("To dx: " + str(todx(p)))
                        integral += int_wghts[i] * funct(elem, p) * todx(p)
    # If we shouldn't have integrated over this element, we'll return None.
    try:
        return integral
    except:
        return None
    

def edge_2_csc(mesh, edge_physgrp, funct, node_mapping, gauss_ord=None,
               gauss_mult=1):
    """ Integrates over an edge defined by a group of nodes.
    
    mesh is ElemMesh.\n
    edge_physgrp is the name of the physical group representing the boundary
    of that we want to integrate over.\n
    funct is a function(elem, eta)=matrix that we wish to integrate over.\n
    node_mapping is a NodeMapping object that we're using.\n
    Optional: gauss_ord={eleid:uint} Allows the gauss order to be individually
    set for any element tag. If not in dictionary, defaults to 
    elem.def_gauss_order()\n
    gauss_mult: Multiple of default gauss num gauss points to use.\n
    """
    
    print("edge_2_csc:\tIntegrating "+funct.__name__+" over physical " + \
          "group " + edge_physgrp + ".")
    ticy = python_time.clock()
    # First estimate the number of degrees of freedom we're dealing with.
    # This may be an overestimate since some elements will just have
    numijk = 0
    grp_num = [key for key, value in mesh.phys_group_names.items() \
               if value == edge_physgrp][0]
    for eleid in mesh.elems_in_physical_groups[grp_num]:
        elem = mesh.elems[eleid]
        numijk += (elem.dnen() * (1 + len(elem.enrichment))) ** 2
    
    # Allocate arrays to store the ijk data before matrix assembly.
    row = np.zeros(numijk, dtype=np.int32)
    col = np.zeros(numijk, dtype=np.int32)
    data = np.zeros(numijk, dtype=np.float64)
    
    edge_set = set(mesh.nodes_in_physical_groups[grp_num])
    
    position = 0
    for eleid in mesh.elems_in_physical_groups[grp_num]:
        elem = mesh.elems[eleid]
        # Check guass order.
        try:
            go = gauss_ord[eleid]
        except:
            go = elem.def_gauss_order() * gauss_mult
        # Get edge node tags for this element.
        elem_nid_set = set(elem.nodes)
        edge_nids = elem_nid_set.intersection(edge_set)
        go = max(go)
        intermed = integrate_edge(elem, edge_nids, funct, gauss_order=go)
        if intermed is not None:
            # Add to our ijk data.
            idxs = node_mapping.tags_to_idxs(elem.elem_node_tag_gen())
            ndof = len(idxs)**2
            xv, yv = np.meshgrid(idxs, idxs)
            xv = np.reshape(xv, -1)
            yv = np.reshape(yv, -1)
            intermed = np.reshape(intermed, -1)
            col[position : position + ndof] = xv
            row[position : position + ndof] = yv
            data[position : position + ndof] = intermed
            position += ndof
    
    glo_ndof = node_mapping.count
    mtrx = ssparse.coo_matrix((data, (row, col)), shape=(glo_ndof, glo_ndof))
    tot_time = python_time.clock() - ticy
    print("edge_2_csc:\tGenerated " +str(mtrx.shape)+" matrix with nnz " +\
          str(mtrx.nnz) + " in " + "{:10.4f}".format(tot_time) + " s.")
    return mtrx.tocsc()
    
    
def edge_2_array(mesh, edge_physgrp, funct, node_mapping, gauss_ord=None,
                 gauss_mult=1):
    """Integrates over an edge defined by a group of nodes.
    
    mesh is ElemMesh.\n
    edge_physgrp is the name of the physical group representing the boundary
    of that we want to integrate over.\n
    funct is a function(elem, eta)=array that we wish to integrate over.\n
    node_mapping is a NodeMapping object that we're using.\n
    Optional: gauss_ord={eleid:uint} Allows the gauss order to be individually
    set for any element tag. If not in dictionary, defaults to 
    elem.def_gauss_order()\n
    gauss_mult: Multiple of default gauss num gauss points to use.\n
    """
    print("edge_2_array:\tIntegrating "+funct.__name__+" over physical " + \
          "group " + edge_physgrp + ".")
    
    ticy = python_time.clock()
    grp_num = [key for key, value in mesh.phys_group_names.items() \
               if value == edge_physgrp][0]
    edge_set = set(mesh.nodes_in_physical_groups[grp_num])
    
    # First estimate the number of degrees of freedom we're dealing with.
    # This may be an overestimate since some elements will just have
    vect = np.zeros(node_mapping.count, dtype=np.float64)
    
    for eleid in mesh.elems_in_physical_groups[grp_num]:
        elem = mesh.elems[eleid]
        # Check guass order.
        try:
            go = gauss_ord[eleid]
        except:
            go = elem.def_gauss_order() * gauss_mult
        # Get edge node tags for this element.
        elem_nid_set = set(elem.nodes)
        edge_nids = elem_nid_set.intersection(edge_set)
        try:
            go = gauss_ord[eleid]
            intermed = integrate_edge(elem, edge_nids, funct, gauss_order=go)
        except:
            intermed = integrate_edge(elem, edge_nids, funct,
                gauss_mult=gauss_mult)
        #print("DEBUG/ElemTools/edge_to_array:\tintermed = " + str(intermed))
        if intermed is not None:
            # Add to our ind/data.
            idxs = node_mapping.tags_to_idxs(elem.elem_node_tag_gen())
            vect[idxs] += intermed

    tot_time = python_time.clock() - ticy
    print("edge_2_array:\tGenerated " + str(vect.shape) +" array in "
           + "{:10.4f}".format(tot_time) + " s.")
    
    #true_idxs = []
    #for nid in mesh.nodes_in_physical_groups[grp_num]:
    #    true_idxs.append(node_mapping.tag_to_idx((nid, 0)))
    #true_array = np.zeros(len(vect))
    #true_array[true_idxs] = 1
    #print(np.hstack((np.matrix(vect).transpose(), np.matrix(true_array).transpose())))
    return vect
    
    
    
    
    
    