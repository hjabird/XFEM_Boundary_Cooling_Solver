# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@lisence: MIT
@status: alpha
"""


import Elements as Elements
import numpy as np

class ElemMesh:
    
    def __init__(self):
        self.nodes = {}
        self.elems = {}
        self.nodes_in_physical_groups = {}
        self.phys_group_names = {}
        self.elems_in_physical_groups = {}

    def enrich_elems(self, physgrp, enrich, deriv_enrich, ngp, eclass, ident):
        """ Enrich a set of elements within a physical group
        
        physgrp - string representing group to enrich.\n
        enrich - enrichment function - function of form f(x) where x is an 
        array.\n
        deriv_enrich - tuple of derivatives of the enrichment function.\n
        ngp - number of gauss points per dimension for integration.\n
        ident - the identity of the enrichment. Shares degrees of freedom with
        enrichment of the same id.
        """
        try:        
            grp_num = [key for key, value in self.phys_group_names.items() \
                    if value == physgrp][0]
            elemset = set(self.elems_in_physical_groups[grp_num])
        except:
            print("##########################################################"
                  + "#####################")
            print("ElemMesh:\tFATAL ERROR!")
            print("ElemMesh:\tTrying to build enrichement with identity "
                  + str(ident) + " on physical group " + physgrp + ".")
            print("ElemMesh:\tCould not find group " + physgrp + ".")
            print("ElemMesh:\tAvailible groups are:")
            for key, val in self.phys_group_names.items():
                print(str(key)+":\t"+str(val))
            print("##########################################################"
                  + "#####################")
            raise KeyError
            
        enrichment = Elements.Enrichment(eclass, ident)
        enrichment.define_func(enrich)
        enrichment.define_deriv_func(deriv_enrich)
        enrichment.define_gauss_order(ngp)
        
        for eleid in elemset:
            enrichment.enrich_elem(self.elems[eleid])
        print("ElemMesh:\tApplied enrichment with id "+str(ident)+\
              " to " + str(len(elemset)) +" elements in set " + physgrp + ".")
        

    def build_from_gmsh(self, file_path):
        """ Build elemement mesh from gmsh ascii .msh file
        
        Arg: takes path to .msh file
        """
    
        import Translators.gmshtranslator as gmshtranslator
        self.nodes.clear()
        self.elems.clear()
        self.nodes_in_physical_groups.clear()
        
        gt = gmshtranslator.gmshTranslator(file_path)
        # Setup for phys groups
        for grp in gt.physical_groups:
            self.nodes_in_physical_groups[grp] = []
        logical_nodes_in_physical_groups = gt.nodes_in_physical_groups
        self.phys_group_names = gt.physical_group_names
        
        # Check that physical group names are not repeated:
        grp_nams = set(self.phys_group_names.keys())
        for key in self.phys_group_names.keys():
            try:
                grp_nams.remove(key)
            except:
                print("gmshTranslator:\tParent: Physical group name repeated.")
                print("gmshTranslator:\tParent: Phys grp name repetition " +
                      "lead to errors later on.")
                print("FATAL ERROR")
                raise ValueError
        del grp_nams
        
        # Element to gmsh element mapping:
        ele_to_Element_dict = {
                               gt.line_2_node:Elements.ElemLine2,
                               gt.line_3_node:Elements.ElemLine3,
                               gt.quadrangle_4_node:Elements.ElemQuad4,
                               gt.quadrangle_8_node:Elements.ElemQuad8,
                               gt.quadrangle_9_node:Elements.ElemQuad9,
                               gt.triangle_3_node:Elements.ElemTri3,
                               gt.triangle_6_node:Elements.ElemTri6}
        
        def elem_supported(eletag,eletype,physgrp,nodes):
            if eletype in ele_to_Element_dict.keys():
                return True
            else:
                print("FATAL ERROR.")
                print(str(eletype) + " is not implemented")
                raise NotImplementedError
                return False
            
        def add_elem(eletag,eletype,physgrp,nodes):
            Element_obj = ele_to_Element_dict[eletype]
            self.elems[eletag] = Element_obj(self.nodes, nodes)
        
        def node_always_true(tag,x,y,z,physgroups):
            return True
        
        class node_idx:
            c = 0
            
        def add_node(tag,x,y,z):
            self.nodes[tag] = np.array((x,y,z))
            for grp_name, grp_nodes in logical_nodes_in_physical_groups.items():
                # Oddly there is offset of 1 in logical arrays...
                if logical_nodes_in_physical_groups[grp_name][node_idx.c+1] == 1:
                    self.nodes_in_physical_groups[grp_name].append(tag)
            node_idx.c += 1
                
        gt.add_nodes_rule(node_always_true, add_node)
        gt.add_elements_rule(elem_supported, add_elem)
        
        gt.parse()
    
        
    def export_to_vtk(self, export_path, NodeData={}):
        """ Export mesh as VTK unstructured grid 
        
        Uses evtk
        Argument: export_path - path to save mesh to.\n
        NodeData: List of data dictionaries. {'varname':data}\n
        Data is {nid : float64}.
        """
        import Translators.pyevtk.vtk as vtk
            
        # Check we have a mesh - forgetting to load one confused me once.
        if len(self.elems) == 0:
            print("##########################################################"
                  + "#####################")
            print("ElemMesh:\tFATAL ERROR!")
            print("ElemMesh:\tTrying to export mesh as VTU to file " + 
                export_path)
            print("ElemMesh:\tMesh is empty. Nothing to export!")
            print("ElemMesh:\tDid you remember to build or load a mesh?")
            print("##########################################################"
                  + "#####################")
            raise(AssertionError)
            
            
            
        def inform(text ):
            print("mesh2vtk:\t" + text)
        # Convert nodes to consecutive numbering.
        # Additionally sort out the NodeData to somethat can be exported.
        idx = 0
        cnodes_x = []
        cnodes_y = []
        cnodes_z = []
        nid_to_idx = {}
        # Prep data variable
        expt_data = {}
        for key in NodeData.keys():
            expt_data[key] = list()
        
        for nid, coord in self.nodes.items():
            cnodes_x.append(coord[0])
            cnodes_y.append(coord[1])
            cnodes_z.append(coord[2])
            nid_to_idx[nid] = idx
            idx += 1
            # And now our export data;
            for key, dict in NodeData.items():
                expt_data[key].append(dict[nid])
            
        cnodes = (np.array(cnodes_x), np.array(cnodes_y), np.array(cnodes_z))
        # nid_to_idx[node_tag] returns node index
        # cnodes contains (x, y, z) for nodes
        
        # lists to np.arrays in export data:
        for key in expt_data.keys():
            expt_data[key] = np.array(expt_data[key], dtype=np.float64)
        
        expt = vtk.VtkFile(export_path, vtk.VtkUnstructuredGrid)
        inform("Writing to "+expt.getFileName() + ".")
        expt.openGrid()
        expt.openPiece(npoints=len(cnodes_x),ncells=len(self.elems.keys()))
        del cnodes_x, cnodes_y, cnodes_z
        
        # Add nodes to file
        expt.openElement("Points")
        expt.addData("Points", cnodes)
        expt.closeElement("Points")
        
        
        # the VTK file wants the ordering as a continious array of node ids,
        # with offsets which contain the final idx of each element along with
        # an array of element types.
        ele_cnids = []
        offsets = []
        vtkeletype = []
        eletype_to_vtk = {
                          Elements.ElemLine2:   vtk.VtkLine,
                          Elements.ElemLine3:   vtk.VtkQuadraticEdge,
                          Elements.ElemTri3:    vtk.VtkTriangle,
                          Elements.ElemTri6:    vtk.VtkQuadraticTriangle,
                          Elements.ElemQuad4:   vtk.VtkQuad,
                          Elements.ElemQuad8:   vtk.VtkQuadraticQuad,
                          Elements.ElemQuad9:   vtk.VtkBiQuadraticQuad}
        for elem in self.elems.values():
            for nid in elem.nodes:
                ele_cnids.append(nid_to_idx[nid])
            try:
                offsets.append(len(elem.nodes) + offsets[-1])
            except:
                offsets.append(len(elem.nodes))
            vtkeletype.append(eletype_to_vtk[elem.__class__].tid)
        ele_cnids = np.array(ele_cnids, dtype=np.int32)
        offsets = np.array(offsets, dtype=np.uint32)
        vtkeletype= np.array(vtkeletype, dtype=np.uint8)
        
        
        expt.openElement("Cells")
        expt.addData("connectivity", ele_cnids)
        expt.addData("offsets", offsets)
        expt.addData("types", vtkeletype)
        expt.closeElement("Cells")
        
        # Add point data info to file
        if len(expt_data.keys()) > 0:
            _addDataToFile(expt, cellData=None, pointData=expt_data)
        
        expt.closePiece()
        expt.closeGrid()
        
        # ACTUAL DATA APPENDING
        expt.appendData(cnodes)
        expt.appendData(ele_cnids).appendData(offsets).appendData(vtkeletype)
        
        # Add point data data to file
        if len(expt_data.keys()) > 0:
            _appendDataToFile(expt, cellData=None, pointData=expt_data)
        
        expt.save()
        inform("Saved data.")
            
    
    def elem_quad9_to_quad8(self):
        """ Substitutes elem quad 9s for quad 8s """
        counter = 0
        for eleid, elem in self.elems.items():
            if isinstance(elem, Elements.ElemQuad9):
                old = elem
                nodes = elem.nodes
                self.elems[eleid] = Elements.ElemQuad8(self.nodes, nodes[0:8])
                counter += 1
        print("ElemMesh:\tSwapped " + str(counter) + " quad8 elements " + \
            "for quad9s.")
    
    def disp_nodes(self):
        """ Draws dots where the nodes are in XY plane """
        colours = ['red', 'blue', 'green', 'magenta', 'cyan']
        idx=0
        for grp in self.nodes_in_physical_groups.values():
            X = [self.nodes[a][0] for a in grp]
            Y = [self.nodes[a][1] for a in grp]
            plt.scatter(X,Y, color=colours[idx], alpha=0.5)
            idx+=1
        plt.show()
        
        
    def calc_elems_in_physgrps(self):
        """ Calculates the elements attached to nodes in physical groups.
        """
        for grp_num, grp_nodes in self.nodes_in_physical_groups.items():
            self.elems_in_physical_groups[grp_num] = set()
            grp_n_set = set(grp_nodes)
            for eleid, elem in self.elems.items():
                if grp_n_set.intersection(set(elem.nodes)):
                    self.elems_in_physical_groups[grp_num].add(eleid)
        
        for key in self.elems_in_physical_groups.keys():
            self.elems_in_physical_groups[key]=[a for a \
                                        in self.elems_in_physical_groups[key]]
    
    def print_elem_counts(self):
        """ Prints out the counts of different element types.
        """
        def rep_str(txt):
            print("ElemMesh:\t" + txt)
        rep_str("Mesh has following contents...")
        counts = {}
        for elem in self.elems.values():
            try:
                counts[elem.__class__] += 1
            except:
                counts[elem.__class__] = 1
        for key, val in counts.items():
            rep_str("\t" + str(val) + " elements of type " + str(key) + ".")
            
    def print_group_elem_counts(self):
        """ Prints out the counts of nodes and elements in physical groups
        """
        def rep_str(txt):
            print("ElemMesh:\t" + txt)
        rep_str("Groups contain following number of elements...")
        for num, name in self.phys_group_names.items():
            elec = len(self.elems_in_physical_groups[num])
            nidc = len(self.nodes_in_physical_groups[num])
            rep_str("\t" + name + " contained " + str(elec) + " elements and "
                    + str(nidc) + " nodes.")
            
    def remove_line_elems(self):
        """ Removes 1D elements from mesh
        """
        to_remove = []
        for eletag, elem in self.elems.items():
            if elem.nd() == 1:
                to_remove.append(eletag)
        print("ElemMesh:\tRemoving " + str(len(to_remove)) + " line elements.")
        for tag in to_remove:
            self.elems.pop(tag)
            
    def project_points(self, ext_points, failure_rule=None):
        """ Map nodes from external points onto this mesh.
        
        ext_points: a dictionary of {point_id : (numpy_array)point_coord}
        failure_rule: default to Assertion error, ='closest' assigns
        save value as nearest point.
        
        Outputs: {point_id : (<element>, <local_coord>)}
        """
        
        print("ElemMesh:\tCalculating element local " +
              "coordinates for interpolating nodes.")
        print("ElemMesh:\tMay take a while... (1 time cost)")
        
        to_place = set(ext_points.keys())
        rntp = np.floor(len(to_place)/100)
        placed = set()
        mapping = {}
        
        def print_completion(comp):
            print("\rElemMesh:\tDone " + str(comp) + "%",
                end="")
                
        for eleid, elem in self.elems.items():
            # Get nids near elem... A must for speed!
            trial_set = elem.is_near(to_place.difference(placed),
                                    ext_points)
            for nid in trial_set:
                # Find the node's coordinate in the element's
                # coordinate system.
                coord = ext_points[nid]
                loc_coord = elem.global_to_local(coord)
                # If the node is in the element, add it to dict
                # and remove from test set.
                if elem.local_in_element(loc_coord):
                    mapping[nid] = (elem, loc_coord)
                    placed.add(nid)
                    if len(placed) % rntp == 0:
                        print_completion(len(placed)/rntp)
        print("") # Next line after completion thing.
        
        # See if we have any homeless nodes:
        if len(to_place.difference(placed)) != 0:
            print("ElemMesh:\tExport mesh contains nodes " +
                  "outside of XFEM/FEM mesh!")
            print("ElemMesh:\tRemaining nodes count:" + str(
                len(to_place.difference(placed))) + " of " +
                str(len(ext_points.keys())))
            
            if failure_rule == 'closest':
                print("ElemMesh:\tSetting to match nearest"
                        + " known node")
                maxerr = 0
                for nid in to_place.difference(placed):
                        coord = ext_points[nid]
                        dist = 9e99
                        nearest_node = None
                        dvect = np.zeros(3)
                        # Find the nearest node...
                        for nid_ext in placed:
                                ext_coord = ext_points[nid_ext]
                                dvect = coord - ext_coord
                                dist_ext = np.linalg.norm(dvect)
                                if dist_ext < dist:
                                        nearest_node = nid_ext
                                        dist = dist_ext
                        if dist > maxerr:
                                maxerr = dist
                        mapping[nid] = \
                                mapping[nid_ext]
            elif failure_rule is None:
                # Handle default.
                print("ElemMesh:\tNo failure rule set.")
                print("ElemMesh:\tHomeless projected nodes stops simulation.")
                print("ElemMesh:\tConsider choosing 'closest' rule.")
                raise Error
            else:
                # Handle invalid.
                print("ElemMesh:\tInvalid failure rule chosen for point" \
                        + " projection with nodes out of bounds!")
                print("ElemMesh:\tGiven failure rule was:" + str(failure_rule))
                print("ElemMesh:\tRaising error.")
                raise InputError
        return mapping
        
def _addDataToFile(vtkFile, cellData, pointData):
    # Point data
    if pointData is not None:
        keys = list(pointData.keys())
        vtkFile.openData("Point", scalars=keys[0])
        for key in keys:
            data = pointData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Point")
 
     # Cell data
    if cellData is not None:
        keys = list(cellData.keys())
        vtkFile.openData("Cell", scalars=keys[0])
        for key in keys:
            data = cellData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Cell")
 
def _appendDataToFile(vtkFile, cellData, pointData):
    # Append data to binary section
    if pointData is not None:
        keys = list(pointData.keys())
        for key in keys:
            data = pointData[key]
            vtkFile.appendData(data)
 
    if cellData is not None:
        keys = list(cellData.keys())
        for key in keys:
            data = cellData[key]
            vtkFile.appendData(data)
             
             
if __name__ == "__main__":
    print("QUICK TEST OF ELEMMESH.PY")
    a = ElemMesh()
    print("\tBUINDING ./RMesh/structuredSquare.msh")
    a.build_from_gmsh("./RMesh//structuredSquare.msh")
    print("\tDISPLAYING NODES")
    a.disp_nodes()
    print("\tEXPORTING to ./vtkfile")
    a.export_to_vtk("./vtkfile")
    print("\tDONE\n")
    
