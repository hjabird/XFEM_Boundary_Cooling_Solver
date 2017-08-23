# XFEM_BOUNDARY_COOLING_SOLVER

Simulation of transient heat transfer using the XFEM with a SP1 radiative heat
transfer aproximation. Input is via GMSH .msh file and python script. Output
is via .vtk files for visualisation in paraview. 

This was part of an MEng dissertation project and is no longer under 
active development.

## Dependencies
Requirements:
numpy 
scipy 
Certain features:
cython - needed for recompilation of accelerated functions.

Includes: [Please see repositories for appropriate lisence]
gmshtranslator - Original author: jaabel (GitHub). Updated, corrected & expanded by HJA Bird.
			-> https://github.com/jaabell/gmshtranslator
evtk - Original author: paulo.herrera.eirl@gmail.com. Updated to python 3.3 compatible & expanded by HJA Bird
			-> https://bitbucket.org/pauloh/pyevtk

## Notes
Theoretically Python 2.7 / Python 3.3 compliant. Tested only on python 3.3.
Written to PEP8 std except for run script.
It is suggested that the Anaconda python distribution is used.
PyPy JIT was not found to achieve a useful speedup.

Memory: For situations where "large" FEM simutions are compared to XFEM 
significant memory may be used - most observed use was 6GB.


Running:
Open powershell / bash:
python ht3_solver_run_script.py
Often logging to a file is useful:
python ht3_solver_run_script.py > ./ROut/my_log_file.log


## Issues
- Quad9 elements are incorrectly formulated.
- Projecting meshes onto each other:
	- Occasionally completely misplaces nodes -> BAD
		- Only apparent with refined mesh / refined mesh projection.
		- Time intensive to reproduce!
	- Horrifically slow generally
	- Scales horrifically: num_elem_mesh_a * num_elem_mesh_b
- XFEM solutions cannot be saved:
	- Due to inability of cPickle to support serialisation of some functions
		- IE Python std.lib does not support serialiation of all first class objects.
	- Supposedly, Dill package allows this.
	- Serialisation of function objects also precludes use of the multiprocessing module.

## Further work
- Integration system:
	- Currently based on min number of points in each direction.
	- Support for including integral transforms would be nice
	- A proper integrand settings object.
- Nodemapping:
	- Is an object used to map degrees of freedom to nodes/functionsTag
	- Automatically allocates new DoFs to unseen node/functionTag pairs
	- Suggestion:
		- Add a lock mechanism so new DoFs can't silently introduced later on by mistake.
- Rewrite the whole damned thing in C++.
	- Avoid the choice between nice abstractions and speed.
	- Use native libVTK.
	- Easy shared memory multithreading w/ OpenMP
	- Probably massively speed up integration speeds - a must for SP1 rad use.
	- You get really nice debuggers / profiling tools.

