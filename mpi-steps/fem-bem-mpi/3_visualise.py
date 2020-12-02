from mpi4py import MPI
import numpy as np
from dolfinx import *
from dolfinx.io import XDMFFile

comm = MPI.COMM_WORLD
mesh = UnitCubeMesh(comm, 5, 5, 5)
tdim = mesh.topology.dim
print(tdim)
mesh.topology.create_connectivity(0, tdim)
indices = np.arange(mesh.topology.index_map(tdim).size_local)
values = np.ones_like(indices, dtype=np.int32) * MPI.COMM_WORLD.rank
cell_owner = MeshTags(mesh, tdim, indices, values)

rank = comm.Get_rank()
size = comm.Get_size()
with XDMFFile(comm, "3_vis/cell_owner_" + str(size) + ".xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_meshtags(cell_owner)

vertex_indices = np.arange(mesh.topology.index_map(0).size_local)
vertex_values = np.ones_like(vertex_indices, dtype=np.int32) * MPI.COMM_WORLD.rank
vertex_owner = MeshTags(mesh, 0, vertex_indices, vertex_values)
print(vertex_indices, vertex_values)

with XDMFFile(comm, "3_vis/vertex_owner_" + str(size) + ".xdmf", "w") as file:
   file.write_mesh(mesh)
   file.write_meshtags(vertex_owner)


