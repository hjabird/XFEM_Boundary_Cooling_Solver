# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@lisence: MIT
@status: alpha
"""
import numpy as np
import NodeMapping
import ElemTools as et
import scipy.constants as sconst
from scipy.sparse.linalg import spsolve as scipy_sparse_linsolve
from scipy.sparse.linalg import eigsh as scipy_sparse_eigens
import time as python_time
import Accelerated as Accl
import csv


class ht3_solver:
    def __init__(self, mesh):
        # USER FIELDS:
        # simulation setup
        # mesh
        self.mesh = mesh
        self.export_mesh = None  # For XFEM we use this mesh for data export.
        self.export_mesh_geomfix_nearest = None
        self.save_path = None
        # time
        self.max_T = None
        self.d_T = None
        # simulation setup optical
        self.v0_frequency = None
        self.fq_list = []
        # simulation setup temperatures
        self.background_temperature = None
        self.initial_temperature = None
        self.diff_scale = None

        # material properties
        # optical
        self.absorb_coeffs = []  # Absorbtion coefficients.
        self.alpha = None  # (Hemisperic emssivity)
        self.refr_idx_vol = None
        self.refr_idx_background = None
        self.r1 = None
        self.r2 = None
        # conductive
        self.density = None
        self.heat_capacity = None
        self.thermal_conductivity = None
        self.convect_coeff = None

        # SOLVER INTERNALS
        self.step = None
        self.current_T = None
        # cond would be true if currently solving for conduction
        self.cond = False
        # Last set of temperature coefficients
        self.lst_tmp = None
        # rad is none whilst not solving for radiation
        # takes value of index of frequency in fq list when solving for freq.
        self.rad = None
        # List of the last set(s) or radiative intensity coefficients
        self.lst_rad = []
        # Data retention
        self.saved_data = {}
        self.node_map = NodeMapping.NodeIdxMap()
        # Mesh to export point / element mapping
        self.export_to_elem = None
        self.norm_path = None
        self.expected_solution = None
        self.redefined = False
        def save_rule(step, dt): return True
        self.save_rule = save_rule
        def norm_saving_rule(step, dt): return True
        self.norm_saving_rule = norm_saving_rule

    def advance(self, sol):
        """ Prepare for next simulation step
        
        Saves last solution in correct place. Changes constants and functions
        to be approriate for next step. Returns false if the simulation should
        end, true otherwise.
        """
        self.data_saving(sol)
        simulation_continues = self._advance_settings(sol)
        self.redef_vars()
        self.reporting(sol)
        self.norm_reporting()
        return simulation_continues

    def _advance_settings(self, sol):
        """ Changes settings that indicate what the solver should be solving #
        for next. Save sol correctly.
        
        sol is the solution vector of the last step.
        sol = anything can be used on first step.
        """
        if self.cond == True:
            # Save last solution...
            self.lst_tmp = sol
            # Check if all timesteps are complete.
            self.current_T += self.d_T
            self.step += 1
            if self.current_T > self.max_T:
                return False
            # Set to not be conduction any more
            self.cond = False
            if len(self.fq_list) > 0:
                self.rad = 0
            else:
                # There are radiation steps to do.
                self.cond = True
            return True

        # If we're here, we're either not done anything yet or have
        # just done a radiation step.
        if self.rad != None:
            # Save last solution
            self.lst_rad[self.rad] = sol
            # Advance to next radiation stage if one exists. Else cond.
            if self.rad + 1 != len(self.fq_list):
                self.rad += 1
            else:
                self.rad = None
                self.cond = True
            return True

        # If we've made it to here, we must just setting the simulation
        # going.
        assert (len(self.fq_list) == len(self.lst_rad))
        if len(self.lst_rad) > 0:
            assert (len(self.fq_list) == len(self.absorb_coeffs))
            assert (self.refr_idx_vol >= 0.0)
        # Could set to zero, but that might limit restarts. Just check
        # Validity....
        assert (self.step != None)
        assert (self.d_T > 0.0)
        assert (self.current_T != None)
        assert (self.max_T != None)
        assert (self.max_T > self.current_T)
        assert (self.diff_scale >= 0.0)
        assert (self.diff_scale <= 1.0)
        assert (self.thermal_conductivity > 0.0)
        assert (self.alpha >= 0.0)
        assert (self.refr_idx_background >= 0.0)
        # Set the ball rolling:
        if len(self.fq_list) > 0:
            # We can set solver for frequencies first...
            self.rad = 0
        else:
            self.cond = True
        return True

    def redef_vars(self):
        """ Redefines constants and vectors used to be appropriate for time
        and solver step.
        """

        # Try using redefined source / boundary terms
        if self.redefined == True:
            self._redef_via_predef_eqn()
        else:  # If they haven't been set you'll get an exception.
            self._redef_sp1_vars()

    def _redef_via_predef_eqn(self):
        """ If the solver has been given predefined boundary conditions
        and source terms, redefine the variables used in the simulation as
        these instead """
        time = self.current_T  # + self.d_T

        self.Beta = (self.diff_scale * self.thermal_conductivity) / \
                    (self.convect_coeff) 
        self.Epsilon = self.d_T * self.thermal_conductivity / \
                    (self.density * self.heat_capacity)

        # Source term.
        def F_func(elem, eta):
            x = elem.local_to_global(eta)
            F = elem.eval_elem(self.node_map, self.lst_tmp, [eta])[0]
            F -= self.Epsilon * self.redef_F_laplacian(x[0], x[1], time)
            F += self.redef_dTdt(x[0], x[1], time) * self.d_T
            return elem.funcs(eta) * F

        self.vF_vect_vol = et.elems_2_array(self.mesh,
                                            F_func,
                                            self.node_map,
                                            gauss_mult=2)  # Use double gp_1D

        # Boundary term.
        def f_func(elem, eta):
            n = elem.guess_normal_vector_global(eta)
            f = elem.eval_elem(self.node_map, self.lst_tmp, [eta])[0]
            x = elem.local_to_global(eta)
            # Evaluate our boundary term.
            f += self.Beta * self.redef_f_norm_grad(x[0], x[1], time, n)
            f += self.redef_dTdt(x[0], x[1], time) * self.d_T
            return elem.funcs(eta) * f

        self.vf_vect_bound = et.edge_2_array(self.mesh,
                                             "Boundary",
                                             f_func,
                                             self.node_map,
                                             gauss_mult=2)

    def _redef_sp1_vars(self):
        """ Redefines constants and vectors to be appropriate for time and
        solver step in an SP1 approximation of RHT / heat transfer."""

        if len(self.fq_list) == 0:
            no_rad = True
            lst_tmp = np.matrix(np.reshape(self.lst_tmp, 
                (self.lst_tmp.size, 1)))
        else: no_rad = False
        # The practically constants...
        # Big Epsilon:
        if self.cond == True:
            self.Epsilon = self.d_T * self.thermal_conductivity
        else:
            self.Epsilon = (self.diff_scale ** 2) / \
                           (3.0 * self.absorb_coeffs[self.rad] ** 2)
        # Beta:
        if self.cond == True:
            self.Beta = (self.diff_scale * self.thermal_conductivity) / \
                        (self.convect_coeff)
        else:
            self.Beta = (1.0 + 3.0 * self.r2) * (2.0 * self.diff_scale) / \
                        ((1.0 - 2.0 * self.r1) * (
                        3.0 * self.absorb_coeffs[self.rad]))

        # The feild solutions at the last timestep.
        # The integral vF:
        if self.cond == True:
            # The horrifically complicated F:
            def F_func_cond(elem, eta):
                F = 0.0
                Tn = elem.eval_elem(self.node_map, self.lst_tmp, [eta])[0]
                F += Tn
                for k in range(0, len(self.fq_list)):
                    vk = self.fq_list[k]
                    try:
                        vk_m = self.fq_list[k - 1]
                    except:
                        vk_m = self.v0_frequency
                    absorbtion = self.absorb_coeffs[k]
                    phi = elem.eval_elem(self.node_map, self.lst_rad[k],
                                         [eta])[0]
                    inter1 = phi - 4.0 * sconst.pi * \
                                   self.B_int_function(Tn, self.refr_idx_vol,
                                                       vk, vk_m)
                    inter2 = absorbtion * self.d_T / (self.diff_scale ** 2)
                    F += inter2 * inter1
                return elem.funcs(eta) * F
            if not no_rad:
                # We're integrating something non-linear for SP1
                self.vF_vect_vol = et.elems_2_array(self.mesh,
                                                F_func_cond,
                                                self.node_map)
            else:
                # Or something easier if we're only looking at heat.
                self.vF_vect_vol = np.array(self.uv_vol * lst_tmp).reshape(-1)
        else:
            def F_func_radiative(elem, eta):
                T = elem.eval_elem(self.node_map, self.lst_tmp, [eta])[0]
                vk = self.fq_list[self.rad]
                try:
                    vk_minus = self.fq_list[self.rad - 1]
                except:
                    vk_minus = self.v0_frequency
                n = self.refr_idx_vol
                F = 4.0 * sconst.pi * self.B_int_function(T, n, vk, vk_minus)
                return elem.funcs(eta) * F

            self.vF_vect_vol = et.elems_2_array(self.mesh,
                                                F_func_radiative,
                                                self.node_map)
        # The path integral vf:
        if self.cond == True:
            def f_func_cond(elem, eta):
                Tb = self.background_temperature
                Tn = elem.eval_elem(self.node_map, self.lst_tmp, [eta])[0]
                n = self.refr_idx_background
                vk = self.v0_frequency
                vk_minus = 0
                Bb0 = self.B_int_function(Tb, n, vk, vk_minus)
                Bn0 = self.B_int_function(Tn, n, vk, vk_minus)
                B_coeff = (self.alpha * sconst.pi) / self.convect_coeff
                f = Tb + B_coeff * (Bb0 - Bn0)
                return elem.funcs(eta) * f
            if not no_rad:
                self.vf_vect_bound = et.edge_2_array(self.mesh,
                                                 "Boundary",
                                                 f_func_cond,
                                                 self.node_map)
            else:
                try:
                    self.vf_vect_bound = self.cache_tb_integral_array
                except AttributeError:
                    def elem_functor(elem, eta): return elem.funcs(eta)
                    self.cache_tb_integral_array = et.edge_2_array(self.mesh,
                                                 "Boundary",
                                                 elem_functor,
                                                 self.node_map)
                    self.cache_tb_integral_array *= self.background_temperature
                    self.vf_vect_bound = self.cache_tb_integral_array
                
        else:
            # Radiation f = 4*pi*B^{(k)}(T_b, n_g)
            def f_func_radiative(elem, eta):
                T = self.background_temperature
                vk = self.fq_list[self.rad]
                try:
                    vk_minus = self.fq_list[self.rad - 1]
                except:
                    vk_minus = self.v0_frequency
                n = self.refr_idx_vol
                f = 4 * sconst.pi * self.B_int_function(T, n, vk, vk_minus)
                return elem.funcs(eta) * f

            self.vf_vect_bound = et.edge_2_array(self.mesh,
                                                 "Boundary",
                                                 f_func_radiative,
                                                 self.node_map)
        assert (self.vF_vect_vol.size == self.vF_vect_vol.shape[0])
        assert (self.vf_vect_bound.size == self.vf_vect_bound.shape[0])
        assert (self.vf_vect_bound.shape[0] == \
                        self.vF_vect_vol.shape[0])

    def initialise(self, initial=None):
        """ Prepare for the start of simulation.
        
        Build nodemapping object.
        Prepare T0, defaults as initial temperatures or
        initial(x, y) if provided as argument.
        Setup constant matrices.
        """
        ticy = python_time.clock()
        if hasattr(self, 'redef_F_laplacian') or \
                hasattr(self, 'redef_f_norm_grad'):
            print("ht3_solver:\tVariables resassigned to known solution.")
            assert (hasattr(self, 'redef_F_laplacian'))
            assert (hasattr(self, 'redef_f_norm_grad'))
            self.redefined = True

        self._print_setup()

        # Add all elem DoFs to NodeMapping
        for elem in self.mesh.elems.values():
            self.node_map.tags_to_idxs(elem.elem_node_tag_gen())

        # Set initial condition.
        t0 = np.zeros(self.node_map.count, dtype=np.float64)
        if initial is None:
            for elem in self.mesh.elems.values():
                idxs = self.node_map.tags_to_idxs(elem.elem_node_tag_gen())
                t0[idxs[:elem.dnen()]] = self.initial_temperature
        else:
            for elem in self.mesh.elems.values():
                idxs = self.node_map.tags_to_idxs(elem.elem_node_tag_gen())
                coords = elem.node_coords()
                for i in range(len(idxs)):
                    t0[idxs[i]] = initial(coords[i, 0], coords[i, 1])
        self.lst_tmp = t0

        # Just to have the correct length list. Should be skipped over anyway.
        self.lst_rad = [np.zeros(len(t0), dtype=np.float64)
                        for a in self.fq_list]
        # Setup constant matrices
        self.uv_vol = et.elems_2_csc(self.mesh,
                                     et.uv_mtrx,
                                     self.node_map)
        self.uv_vol.description = "Integral of test function * weight " \
                                  + "over element volumes."
        self.guv_vol = et.elems_2_csc(self.mesh,
                                      et.gu_gv_mtrx,
                                      self.node_map)
        self.guv_vol.description = "Integral of test function laplacian *" \
                                   + " weight function laplacian over element volumes."
        self.uv_bound = et.edge_2_csc(self.mesh,
                                      "Boundary",
                                      et.uv_mtrx,
                                      self.node_map)
        self.uv_bound.description = "Integral of test function * weight " \
                                    + " function over domain boundary."
        self._print_matrix_info(self.uv_vol, "UV over volume")
        self._print_matrix_info(self.guv_vol, "Grad U dot Grad V over volume")
        self._print_matrix_info(self.uv_bound, "UV over boundary")
        tocy = python_time.clock()
        print("ht3_solver:\tCompleted initialisation in " + str(tocy - ticy)
              + " s.")
        

    @staticmethod
    def _print_matrix_info(mtrx, name):
        """ Print infomation about a matrix
        """
        pr = lambda t: print("ht3_solver:\t" + t)
        pr("MATRIX INFO:")
        pr("Matrix:\t" + name)
        pr("Description:\t" + str(mtrx.description))
        pr("Shape:\t" + str(mtrx.shape))

    def _print_setup(self):
        """ Prints a load of settings for the solver.
        """
        pr = lambda x: print("ht3_solver:\t" + x)
        pr("Start time is " + str(python_time.asctime()))
        pr("")
        pr("TIME SETTINGS:")
        pr("Current time:\t\t\t\t" + str(self.current_T))
        pr("Delta T:\t\t\t\t" + str(self.d_T))
        pr("Finish time:\t\t\t\t" + str(self.max_T))
        pr("")
        pr("Using predefined funtions?:\t\t" + str(self.redefined))
        pr("")
        pr("PHYSICAL MODEL: ")
        pr("Background temperature:\t\t\t" + str(self.background_temperature))
        pr("Starting temp (maybe overrided):\t" + str(self.initial_temperature))
        pr("Diffusion scale:\t\t\t" + str(self.diff_scale))
        pr("Solid refractive index:\t\t\t" + str(self.refr_idx_vol))
        pr("Background refractive index:\t\t" + str(self.refr_idx_background))
        pr("Solid density:\t\t\t\t" + str(self.density))
        pr(
            "Solid specific heat capacity:\t\t" + str(
                self.heat_capacity))
        pr("Solid thermal conductivity:\t\t" + str(self.thermal_conductivity))
        pr("Solid hemispheric emissivity:\t\t" + str(self.alpha))
        pr("SP1 setting - r1:\t\t\t" + str(self.r1))
        pr("SP1 setting - r2:\t\t\t" + str(self.r2))
        pr("Convective coefficient:\t\t\t" + str(self.convect_coeff))
        pr("")
        pr("RADIATION - FREQUENCIES:")
        pr("Frequencies defined beyond base:\t" + str(len(self.fq_list)))
        pr("-----------------------------------------------------------------")
        pr("Frequency (Hz)\t\tAbsorbtion coeff")
        pr("-----------------------------------------------------------------")
        pr(str(self.v0_frequency) + "\t\t" + "-")
        for i in range(0, len(self.fq_list)):
            pr(str(self.fq_list[i]) + "\t" + str(self.absorb_coeffs[i]))
        pr("-----------------------------------------------------------------")

    def zero_timings(self):
        """ Zero step counter and current time """
        self.step = 0
        self.current_T = 0.0

    def make_k_matrix(self):
        """ Generate ht3_solver 'stiffness' matrix
        """
        K = self.uv_vol + self.Epsilon * self.guv_vol + \
            (self.Epsilon / self.Beta) * self.uv_bound
        return K
    
    def matrix_spy(self, mtrx):
        """ Use matplotlib to spy a matrix
        """
        import matplotlib.pylab as pl
        pl.spy(mtrx,precision=0.01, markersize=1)
        pl.show()
    
    def check_k_matrix_stability(self):
        """ Check stability of solution.
        
        Finds primary eigenvalue of system. Asserts if more than 1.
        """
        K = self.make_k_matrix()
        vals, vects = scipy_sparse_eigens(K)
        principal_val = vals.max()
        print("ht3_solver:\t'Stiffness' matrix principal eigenvalue was "
            + str(principal_val))
        if principal_val > 1:
            print("##########################################################")
            print("ht3_solver:\tWARNING")
            print("ht3_solver:\tPrincipal eigenvalue is more than one.")
            print("ht3_solver:\tThe analysis will be unstable.")
            print("ht3_solver:\tIf this is OK, just go and modify the code "
                + "or something.")
            print("##########################################################")
            raise(AssertionError)
        
    def one_step(self):
        """ Do a single simulation step. Returns step's solution.
        
        Forms linear expresson to solve and solves it for solution.
        """
        assert (self.uv_vol is not None)
        assert (self.guv_vol is not None)
        assert (self.uv_bound is not None)
        assert (self.vf_vect_bound is not None)
        assert (self.vF_vect_vol is not None)
        # Shape checks
        assert (self.vF_vect_vol.size == self.vF_vect_vol.shape[0])
        assert (self.vf_vect_bound.size == self.vf_vect_bound.shape[0])
        assert (self.vF_vect_vol.shape == self.vf_vect_bound.shape)
        assert (self.uv_vol.shape[0] == self.uv_vol.shape[1])
        assert (self.uv_vol.shape == self.guv_vol.shape)
        assert (self.uv_vol.shape == self.uv_bound.shape)
        assert (self.uv_vol.shape[0] == self.vF_vect_vol.shape[0])
        
        if self.step == 0:
            self.check_k_matrix_stability()
        # print("Epsilon is :"+str(self.Epsilon))
        # print("Beta is :"+str(self.Beta))

        # Form "Stiffness" matrix:
        K = self.make_k_matrix()
        # Form "Force" vector:        
        f = self.vF_vect_vol + (self.Epsilon / self.Beta) * self.vf_vect_bound

        #        print("FORCE VECTOR:")
        #        print(f)
        #        print("STIFFNESS MATRIX")
        #        print(K)
        #        print("UV_VOL")
        #        print(self.uv_vol)
        #        print("EPSILON * GUV_VOL")
        #        print(self.Epsilon * self.guv_vol)
        #        print("UV_BOUND * COEFF")
        #        print((self.Epsilon / self.Beta) * self.uv_bound)
        sol = scipy_sparse_linsolve(K, f)
        #        print("SOLUTION")
        #        print(sol)
        return sol

    def run(self, initial=None):
        """ Run the simulation.
        """
        self.initialise(initial=initial)
        sol = None
        while self.advance(sol):
            sol = self.one_step()

    B_int_function = Accl.B_int_function

    # """ The B^{(k)}(T, n) function.
    # 
    # T is temperature.\n
    # n is refractive index\n
    # vk & vk_minus are frequencies used as the limits of integration.
    # """


    def data_saving(self, sol):
        """ Saves given solution as solution to CURRENT solver state
        """
        # Only export data once per time-step. We do this on the conduction
        # step.
        if self.save_rule is not None:
            save_rule_true = self.save_rule(self.step, self.d_T)
        else:
            save_rule_true = True
            
        if self.cond == True:
            series = "Temperature"
        elif self.rad is not None:
            series = ("Radiation", self.fq_list[self.rad])
        else:
            # before sim starts... EARLY EXIT
            return

        if self.cond == True and save_rule_true:
            # Save data to file with step no.
            # First, generate dictionaries with {nid:value}

            # CASE 1: Export mesh = FEM mesh (ie, no enrichment, easier!)
            if self.export_mesh is None:
                data_temp = {}
                for nid in self.mesh.nodes.keys():
                    idx = self.node_map.tag_to_idx((nid, 0))
                    data_temp[nid] = self.lst_tmp[idx]
                data_rad = {}
                for i in self.fq_list:
                    data_rad[i] = {}
                for nid in self.mesh.nodes.keys():
                    idx = self.node_map.tag_to_idx((nid, 0))
                    for i in range(0, len(self.fq_list)):
                        data_rad[self.fq_list[i]][nid] = self.lst_rad[i][idx]
            # End CASE 1 - see after case two for finishing export.


            # CASE 2: Exporting to a different mesh to the the XFEM / FEM
            # mesh.            
            else:
                # We need to a mapping from global to local element 
                # coordinates. We'll do this once and then store it.
                # We store it in self.export_to_elem dictionary.
                if self.export_to_elem is None:
                    self.export_to_elem = \
                        self.mesh.project_points(self.export_mesh.nodes,
                                                    failure_rule='closest')
                # Dictionaries to export:
                data_temp = {}
                data_rad = {}
                # Setup frequency data:
                for i in self.fq_list:
                    data_rad[i] = {}

                for node_id, expt_data in self.export_to_elem.items():
                    # Unpack the value of the dictionary value for clarity:
                    elem = expt_data[0]
                    eta = expt_data[1]  # local coord

                    # Get element / solution indexes:
                    val = elem.eval_elem(self.node_map, self.lst_tmp, [eta])[0]
                    data_temp[node_id] = val
                    # And for all the frequencies:
                    for i in range(0, len(self.fq_list)):
                        data_rad[self.fq_list[i]][node_id] \
                            = elem.eval_elem(self.node_map,
                                             self.lst_rad[i],
                                             [eta])[0]
            # END CASE 2

            expt_data = {"Temperature": data_temp}
            for freq, nvals in data_rad.items():
                expt_data[str(freq * 10) + "THz"] = data_rad[freq]

            # Send to be exported as a VTK.
            if self.export_mesh is None:
                self.mesh.export_to_vtk(self.save_path + str(self.step),
                                        expt_data)
            else:
                self.export_mesh.export_to_vtk(self.save_path + str(self.step),
                                               expt_data)

        try:
            container = self.saved_data[series]
        except:
            self.saved_data[series] = {}
            container = self.saved_data[series]
        if self.step % 10 == 0 or self.step < 10:
            container[self.step] = saved_data(sol, self.step, self.current_T)

    class _reporting_statics:
        """ Really just a static variable....
        """
        time = python_time.clock()
        last_report = -1000

    def reporting(self, sol):
        """ Generate printouts to show simulation progress
        """
        if self.cond == True:
            time = python_time.clock()
            dt = time - self._reporting_statics.time

            def rp(txt):
                print("ht3_solver:\t" + txt)

            if self._reporting_statics.last_report - time < 0:
                rp("Completed step " + str(self.step - 1) + " in " \
                   + str(dt) + " s.")
                steps_rem = (self.max_T - self.current_T) / self.d_T
                completion = 1 - steps_rem / (self.step + steps_rem)
                rp(str(int(completion * 100)) + "% complete.")
                more_steps = np.ceil((self.max_T - self.current_T) / self.d_T)
                more_time = more_steps * dt
                exp_fin = python_time.asctime(python_time.localtime(
                    python_time.time() + int(more_time)))
                rp("Expected completion is " + exp_fin)
                print("\n")
                rp("Starting step " + str(self.step) + ".")
                self._reporting_statics.last_report = time
            self._reporting_statics.time = time

    def norm_reporting(self):
        """ Calculate L1, L2 and Linf norms and print to file.
        
        File is given by self.norm_path
        If an expected solution is given, expected L1, L2 and abs erros will
        also be computed. Expected solution is f(x, t) where x is global
        coordinate and t is time.
        """
        if self.norm_saving_rule is not None:
            norm_rule = self.norm_saving_rule(self.step, self.d_T)
        else:
            norm_rule = True

        if self.norm_path is not None and norm_rule:
            f = open(self.norm_path, 'a', newline="")
            csvf = csv.writer(f)

            if self.step == 0:
                out_row = ["Step", "Time (s)", "Matrix condition", "L1 u", "L2 u", "Linf u"]
                if self.expected_solution is not None:
                    out_row.append("L1 Expected")
                    out_row.append("L2 Expected")
                    out_row.append("L1 Error")
                    out_row.append("L2 Error")
                    out_row.append("L1 Abs Error")
                    out_row.append("L2 Abs Error")
                csvf.writerow(out_row)
            
            condition_number = np.linalg.cond((self.uv_vol + self.Epsilon * self.guv_vol + \
                                    (self.Epsilon / self.Beta) * self.uv_bound).todense())
            out_row = [self.step, self.current_T, condition_number]

            # Calculate the l2 norm or l2 error norm:
            def current_u(elem, eta):
                T = elem.eval_elem(self.node_map, self.lst_tmp, [eta])[0]
                return T

            current_u2 = lambda elem, eta: current_u(elem, eta) ** 2
            cu_i = 0
            cu2_i = 0
            cuinf = 0

            if self.expected_solution is not None:
                def expct(elem, eta):
                    glob_x = elem.local_to_global(eta)
                    true_sol = self.expected_solution(glob_x, self.current_T)
                    return true_sol

                # A bunch of expressons that we can integrate over.
                expct2 = lambda elem, eta: expct(elem, eta) ** 2
                l1_err = lambda elem, eta: current_u(elem, eta) \
                                           - expct(elem, eta)
                l2_err = lambda elem, eta: l1_err(elem, eta) ** 2
                # Initialise variables for reduction to zero.
                expct_i = 0
                expct2_i = 0
                l1_err_i = 0
                l2_err_i = 0
                l1_abs_i = 0
                l2_abs_i = 0

            for elem in self.mesh.elems.values():
                cu_i += et.integrate_elem(elem, current_u)
                cu2_i += et.integrate_elem(elem, current_u2)
                for loc in elem.node_locals():
                    tmp_u = current_u(elem, loc)
                    if tmp_u > cuinf:
                        cuinf = tmp_u

            cu2_i = np.sqrt(cu2_i)

            out_row.append(cu_i)
            out_row.append(cu2_i)
            out_row.append(cuinf)

            if self.expected_solution is not None:
                for elem in self.mesh.elems.values():
                    expct_i += et.integrate_elem(elem, expct,
                                                  gauss_mult=2)
                    expct2_i += et.integrate_elem(elem, expct2,
                                                  gauss_mult=2)
                    l1_err_i += et.integrate_elem(elem, l1_err,
                                                  gauss_mult=2)
                    l2_err_i += et.integrate_elem(elem, l2_err,
                                                  gauss_mult=2)

                expct2_i = np.sqrt(expct2_i)
                l2_err_i = np.sqrt(l2_err_i)
                l1_abs_i = abs(l1_err_i) / abs(expct_i)
                l2_abs_i = abs(l2_err_i) / abs(expct2_i)

                out_row.append(expct_i)
                out_row.append(expct2_i)
                out_row.append(l1_err_i)
                out_row.append(l2_err_i)
                out_row.append(l1_abs_i)
                out_row.append(l2_abs_i)

            csvf.writerow(out_row)
            print("Norm reporting: Wrote norms to " + self.norm_path + ".")
            f.close()

    def compare_solutions(self, FEM_ref_sol, time,
                            series='Temperature',
                            path=None,
                            mesh_mapping=None,
                            save_as_vtu=True):
        """ Compare a solution to a FEM reference solution
        """
        if path == None:
            path = self.save_path+"_comp_sols.csv"
        print("ht3_solver:\tComparing solutions as t = " \
                + str(time) + " for series " + str(series) \
                + " and writing to " +  path, flush=True)
        step_no_this = int(np.floor(time / self.d_T))
        step_no_FEM = int(np.floor(time / FEM_ref_sol.d_T))
        # First, map between FEM_ref solution nodes and this solution.
        if mesh_mapping == None:
            mapping = self.mesh.project_points(FEM_ref_sol.mesh.nodes,
                                            failure_rule='closest')
        else:
            mapping = mesh_mapping
        # Project xfem solution onto these points.
        this_sol = self.saved_data[series][step_no_this].data
        xfem_mapped_sol = np.zeros(FEM_ref_sol.node_map.num())
        for nid, info in mapping.items():
            elem, loc_coord = info
            p_val = elem.eval_elem(self.node_map, \
                                    this_sol, \
                                    (loc_coord,))[0]
            idx = FEM_ref_sol.node_map.tag_to_idx((nid, 0))
            xfem_mapped_sol[idx] = p_val
        
        fem_sol = FEM_ref_sol.saved_data[series][step_no_FEM].data
        # Now we have two solutions, with all value valid at nodes:
        #   fem_sol and xfem_mapped_sol
        f = open(path, 'a', newline="")
        
        def write_pair(a,b):
            f.write(a + ", "+ str(b)+", ")
        write_pair("Series", series)
        write_pair("Time", time)
        # L2 errors
        Err = fem_sol - xfem_mapped_sol
        Ex = fem_sol
        
        #Save into vtu...
        if save_as_vtu == True:
            data_err = {}
            for nid in FEM_ref_sol.mesh.nodes.keys():
                idx = FEM_ref_sol.node_map.tag_to_idx((nid, 0))
                data_err[nid] = Err[idx]        
            data_abs = {}
            for nid in FEM_ref_sol.mesh.nodes.keys():
                idx = FEM_ref_sol.node_map.tag_to_idx((nid, 0))
                data_abs[nid] = Ex[idx]   
            data_calc = {}
            for nid in FEM_ref_sol.mesh.nodes.keys():
                idx = FEM_ref_sol.node_map.tag_to_idx((nid, 0))
                data_calc[nid] = xfem_mapped_sol[idx]
            expt_data = {"Error": data_err,
                         "Reference": data_abs,
                         "Calculated": data_calc}
            FEM_ref_sol.mesh.export_to_vtk(self.save_path + str(step_no_this)+"comp",
                                        expt_data)
                
        L2Ex = 0.0
        L2Abs = 0.0
        def ev_elemSqErr(elem, eta): 
            return np.square(elem.eval_elem(FEM_ref_sol.node_map, Err, [eta])[0])
        def ev_elemSqEx(elem, eta): 
            return np.square(elem.eval_elem(FEM_ref_sol.node_map, Ex, [eta])[0])
        for elem in FEM_ref_sol.mesh.elems.values():
            L2Ex += et.integrate_elem(elem, ev_elemSqEx,
                                                  gauss_mult=2)
            L2Abs += et.integrate_elem(elem, ev_elemSqErr,
                                                  gauss_mult=2)
        L2Ex = np.sqrt(L2Ex)
        L2Abs = np.sqrt(L2Abs)
        write_pair("L2 Err", L2Abs)
        write_pair("L2 Abs Err", L2Abs / L2Ex)
        f.write("\n")
        f.close()
        return mapping

class saved_solver:
    """ Picklable object for saving solver state
    
    Saves: Mesh and enrichment, solutions to steps
    """
    def __init__(self, to_save):
        """ Save essential bits of a solver to a new object that can be pickled.
        """
        self.mesh = to_save.mesh
        self.max_T = to_save.max_T
        self.d_T = to_save.d_T
        # Data retention
        self.saved_data = to_save.saved_data
        self.node_map = to_save.node_map
        
    def return_solver(self):
        """ Create a skelaton solver from which past solution can be obtained.
        """
        sol = ht3_solver(self.mesh)
        sol.max_T = self.max_T
        sol.d_T = self.d_T
        sol.saved_data = self.saved_data
        sol.node_map = self.node_map
        return sol
    
class saved_data:
    """
    Save solution with metadata.
    """
    def __init__(self, data, step_no, sim_time):
        self.data = data
        self.time_stamp = python_time.asctime()
        self.step = step_no
        self.sim_time = sim_time
