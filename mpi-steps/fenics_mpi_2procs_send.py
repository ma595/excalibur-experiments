import numpy as np
import dolfinx
import dolfinx.geometry
from mpi4py import MPI
import ufl
from scipy.sparse import csr_matrix
import pytest
import sys
sys.path.append("./fem-bem/")


def test_mpi():
    N = 2
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank() 
    world_size = comm.Get_size()
    print("world rank ", world_rank)
    if world_rank == 1:
        # fenics bit
        print(MPI.COMM_SELF)
        fenics_mesh = dolfinx.UnitCubeMesh(MPI.COMM_SELF, N, N, N)
        facet_indices = exterior_facet_indices(fenics_mesh)
        print(facet_indices)
	boundary = entities_to_geometry(
		fenics_mesh,
		fenics_mesh.topology.dim - 1,
		exterior_facet_indices(fenics_mesh),
		True,
		)

	bm_nodes = set()
	for tri in boundary:
	    for node in tri:
		bm_nodes.add(node)
	bm_nodes = list(bm_nodes)
	bm_cells = np.array([[bm_nodes.index(i) for i in tri] for tri in boundary])
	bm_coords = fenics_mesh.geometry.x[bm_nodes]




def test_subcomms(N):
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    group = comm.Get_group()
    newgroup = group.Excl([0])
    newcomm = comm.Create(newgroup)

    if world_rank == 0:
        assert newcomm == MPI.COMM_NULL
    else:
        assert newcomm.size == comm.size - 1
        assert newcomm.rank == comm.rank - 1
        fenics_mesh = dolfinx.UnitCubeMesh(newcomm, N, N, N)

    group.Free(); newgroup.Free()
    if newcomm: newcomm.Free()

    print("done")

