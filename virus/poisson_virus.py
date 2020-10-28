# import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.plotting
import ufl
from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh, solve
from dolfinx.cpp.mesh import CellType, entities_to_geometry, exterior_facet_indices
from dolfinx.fem import locate_dofs_topological
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from ufl import ds, dx, grad, inner

from scipy.sparse import csr_matrix

# based on: https://fenicsproject.org/docs/dolfinx/dev/python/demos/poisson/demo_poisson.py.html

def load_virus():
    path = "./1igt.xdmf"
    comm = MPI.COMM_WORLD
    encoding = XDMFFile.Encoding.HDF5
    infile = XDMFFile(comm, path, 'r', encoding) 
    mesh = infile.read_mesh(name='Grid')
    name = "virus"
    mesh.name = name
    return mesh

mesh = load_virus()
# mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)
print("loaded mesh") 
print(mesh.name)

tets = mesh.topology.connectivity(3, 0)
mesh.topology.create_connectivity(2, 0)
tris = mesh.topology.connectivity(2, 0)
mesh.topology.create_connectivity(2, 3)
tri_to_tet = mesh.topology.connectivity(2, 3)

surface_tris = []
for i in range(tris.num_nodes):
    if (len(tri_to_tet.links(i)) == 1):
        surface_tris += [i]
surface_tris = np.array(surface_tris)
len(surface_tris)
print("Surface tris done")

triangles = surface_tris 
# print(surface_tris)

facets = exterior_facet_indices(mesh)
# print('facets ', facets)


# now apply boundary conditions
V = FunctionSpace(mesh, ("Lagrange", 1))
u0 = Function(V)
u0.vector.set(0.0)

bc = DirichletBC(u0, locate_dofs_topological(V, 2, facets))

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
f = 10 * ufl.exp(-((x[0] - 30)**2 + (x[1] + 50)**2) / 200)
g = ufl.sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds
print("Done setup")
# Compute solution
u = Function(V)
solve(a == L, u, bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
print("Solved")

# Save solution in XDMF format
with XDMFFile(MPI.COMM_WORLD, "virus_3D-5.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u)

# u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
# dolfinx.plotting.plot(u)
# plt.show()



# print(bm_nodes)
#print(exterior_facet_indices(fenics_mesh))
#print(dir(fenics_mesh))
#print(fenics_mesh.geometry.x)
#print(boundary)

