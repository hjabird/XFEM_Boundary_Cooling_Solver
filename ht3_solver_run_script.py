
# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2017, Hugh Bird
@lisence: MIT
@status: alpha
"""

import ht3_solver as ht3s
import ElemMesh as em
import Elements as Elements
import numpy as np
import pickle

from ScriptTools import *

# Convenience...
run_id = "TEST_ID"
print("Run id is "+str(run_id)) 

## MESH INPUTS
mesh = em.ElemMesh()

# WE CAN BUILD A MESH FROM SCRATCH:
# 1x3 Mesh:
# ---------------
# |   |    |    |
# |   |    |    |
# ---------------
#mesh.nodes[0] = np.array([0.0, 0.0, 0.0])
#mesh.nodes[1] = np.array([1.0, 0.0, 0.0])
#mesh.nodes[2] = np.array([2.0, 0.0, 0.0])
#mesh.nodes[3] = np.array([3.0, 0.0, 0.0])
#mesh.nodes[4] = np.array([0.0, 1.0, 0.0])
#mesh.nodes[5] = np.array([1.0, 1.0, 0.0])
#mesh.nodes[6] = np.array([2.0, 1.0, 0.0])
#mesh.nodes[7] = np.array([3.0, 1.0, 0.0])
#
#mesh.elems[0] = Elements.ElemQuad4(mesh.nodes, (0,1,5,4))
#mesh.elems[1] = Elements.ElemQuad4(mesh.nodes, (1,2,6,5))
#mesh.elems[2] = Elements.ElemQuad4(mesh.nodes, (2,3,7,6))
#
#mesh.nodes_in_physical_groups = {}
#mesh.nodes_in_physical_groups[0] = [0,4]
#mesh.nodes_in_physical_groups[1] = [3,7]
#mesh.nodes_in_physical_groups[2] = [1,2,3,4,5,6,7]
#mesh.nodes_in_physical_groups[3] = [0,4,3,7]
#mesh.phys_group_names = {0:"Left",
#                         1:"Right",
#                         2:"Volume",
#                         3:"Boundary"}

# OR IMPORT OUR MESH FROM A .msh FILE:
mesh.build_from_gmsh("./RMesh/MESH_FILE.msh")

mesh.print_elem_counts()
mesh.remove_line_elems()            # Remove line elements on boundary
mesh.print_elem_counts()
mesh.calc_elems_in_physgrps()       # [Boilerplate]
mesh.print_group_elem_counts()
mesh.elem_quad9_to_quad8()          # Currently, quad9s don't work. This converts to quad8.

# ARE WE USING ENRICHMENT? IF YES:
# We need a mesh to project results onto:
outmesh = em.ElemMesh()                                         # Mesh object
outmesh.build_from_gmsh("./RMesh/MESH_FILE_2.msh")              # Import mesh from .msh
outmesh.print_elem_counts()                                     #(Boilerplate)
outmesh.remove_line_elems()                                     # Again, remove line elements.
outmesh.print_elem_counts()
outmesh.calc_elems_in_physgrps()
outmesh.print_group_elem_counts()

# DEFINE OUR ENRICHMENT:
# Enrichment needs to define an enrichment function and its partial derivatives.
# We can make a function generate these functions for similar enrichments.
def gen_tanhkx2d(k, dim, scalar):
    offset = 1.0 - np.tanh(2*k)
    f = lambda x:np.tanh(scalar*k*x[dim]+k) + offset - 1.0  
    f_prime0 = lambda x: scalar * k * (1.0/np.cosh(scalar*k*x[dim] + k))**2
    f_prime1 = lambda x: 0
    if dim == 0:
        f_prime = lambda x: np.array((f_prime0(x), f_prime1(x)))
    if dim == 1:
        f_prime = lambda x: np.array((f_prime1(x), f_prime0(x)))
    return (f,  f_prime)

# We can also write a function to apply multiple enrichments to a single element:
def enrich_me(group, dim, pm, k_list, ids_start):
    if dim == 0:
        quadrature = (70, 1)    # Quadratures is not symettric.
    else:
        quadrature = (1, 70)
    for k in k_list:
        print("SCRIPT:\tAdding enrichment to quad with id "+ str(ids_start))
        enr = gen_tanhkx2d(k, dim, pm)
        mesh.enrich_elems(group, enr[0], 
                          enr[1], 
                          quadrature,
                          Elements.ElemQuadBase, 
                          ids_start)
        ids_start += 1

# Enrichment IDs - Enrichments on the same node that share the same id will 
# share a degree of freedom.
k_list = [2, 3, 6, 12, 24]
enrich_me("Bottom", 1, 1, k_list, 100)
enrich_me("Right", 1, -1, k_list, 100)
enrich_me("Arc", 0, -1, k_list, 200)

# END DEFINE ENRICHEMENT



## CREATE A NEW SOLVER OBJECT
solver = ht3s.ht3_solver(mesh)
# solver.norm_path = "./ROut/ht3_"+run_id+"_norm.csv"   # If norm output is desired, this must be defined.
# solver.export_mesh = outmesh                          # If XFEM, an output mesh must be defined.
solver.save_path = "./ROut/ht3_"+run_id+ "_"            # A path to save the solution .vtus must be defined.
mesh.export_to_vtk(solver.save_path+"mesh")             # It is useful to save the input mesh as a VTU. Good for debugging.

# We can specify that saving and norm calculation is only done on specific steps:
# def norm_reporting_rule(step, dt):
    # if step % np.floor(5e-6 / dt) == 0:
        # return True
    # else:
        # return False
# solver.norm_saving_rule = norm_reporting_rule
def saving_rule(step, dt): return False
solver.save_rule = saving_rule

# We can use a predfined solution:
#f(x,y,t) = exp(- x^c kt) + exp(- y^c kt)
# c = 1
# k = 1
# solution = lambda x, t: np.exp(-1 * x[0]**c *k*t) + np.exp(-1 * x[1]**c *k*t) # The solution
# oneD1 = lambda x, t: -1 * c * k * t * x**(c-1) * np.exp(-1 * x**c *k *t)      # Partial deriv 1
# def oneD2(x, t):                                                              # Partial deriv 2
    # a = c*k*t*np.exp(-x**c * k*t)
    # b = c*k*t*x**(2*c-2)
    # d = (c - 1) * x**(c-2)
    # return a * ( b - d)
# laplacian = lambda x, t: oneD2(x[0], t) + oneD2(x[1], t)                      # Laplacian
# def norm_grad(x, t, n):                                                       # Grad in given dir.
    # dfdx = np.array((oneD1(x[0], t), oneD1(x[1], t)))
    # return np.dot(n, dfdx)
# dTdt = lambda x, t: -k * (x[0]**c * np.exp(-k * t*x[0]**c) + \                # DT / Dt
                          # x[1]**c * np.exp(-k * t*x[1]**c))

# solver.redef_F_laplacian = lambda x, y, t: laplacian((x,y), t)
# solver.redef_f_norm_grad = lambda x, y, t, n: norm_grad((x,y), t, n)
# solver.redef_dTdt = lambda x, y, t: dTdt((x, y), t)
# solver.expected_solution = solution



# SIMULATION CONSTANTS
# Some parts are optical for SP1 radiation approximation included in code.
# If len(fq_list) == 0, no radiation will be modelled. Radiation consts like diff scale
# must still be defined however - assertions will (should) occur otherwise.
    #mesh
    #time
solver.zero_timings()
solver.d_T = 1e-7
solver.max_T = 1.01e-5
    # simulation setup optical
solver.v0_frequency = 2.933e13
solver.fq_list = []#[3.422, 3.733, 4.563, 5.133, 5.866, 6.844, 102.671, 10e6]
    # simulation setup temperatures
solver.background_temperature = 300.00
solver.initial_temperature = 1000.0
solver.diff_scale = 0.5

#material properties
    #optical
solver.absorb_coeffs = []#[7136.00, 576.32, 276.98, 27.98, 15.45, 7.70, 0.50, 0.40]
solver.alpha = 0.92 #(Hemisperic emssivity)
solver.refr_idx_vol = 1.46
solver.refr_idx_background = 1.00
solver.r1 = 0.0
solver.r2 = 0.0
    #conductive
solver.density = 2514.8
solver.heat_capacity = 1239.6
solver.thermal_conductivity = 1.672
solver.convect_coeff = 1.0


# Set solver running.
# Solver can be called with initial solution for FEM problems. 
# solver.run(initial= lambda x,y: solution(np.array((x,y)),solver.current_T))
solver.run()
# Solver runs till it ends.

# FEM: a solution can be saved (IE Mesh + degrees of freedom). Not possible currently with XFEM.
# f = open("ROut/SOLUTION.pkl", 'wb')
# ts = ht3s.saved_solver(solver)
# pickle.dump(ts, f)
# f.close()

# FEM OR XFEM: a reference solution can be opened and compared to (Ie calc rel error L2 Norms)
f = open("../v0.6_FEM/ROut/SOLUTION.pkl", 'rb')
fem_ref = pickle.load(f).return_solver()
f.close()
mapping = solver.compare_solutions(fem_ref, 1e-7)
solver.compare_solutions(fem_ref, 2e-7, mesh_mapping = mapping)
solver.compare_solutions(fem_ref, 4e-7, mesh_mapping = mapping)
print("DONE!")



