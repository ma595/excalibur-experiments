import numpy as np
import dolfinx
import dolfinx.geometry
from mpi4py import MPI
import ufl
from scipy.sparse import csr_matrix
import pytest
import sys
sys.path.append("./fem-bem/")

from dolfinx.io import XDMFFile

def test_mpi():
    N = 2
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank() 
    world_size = comm.Get_size()
    print("world rank ", world_rank)
    if world_rank == 1:
        print(MPI.COMM_SELF)
        fenics_mesh = dolfinx.UnitCubeMesh(MPI.COMM_SELF, N, N, N)


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

def test_subcomms_split(N):
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank() 
    world_size = comm.Get_size()
    print("world rank ", world_rank)
    subcomm = comm.Split(comm.rank)

    if world_rank < world_size//2:
        color = 55
        key = -world_rank
    else:
        color = 77
        key = +world_rank

    newcomm = MPI.COMM_WORLD.Split(color, key)
    if comm.rank == 0:
        print("size ", newcomm.size)
        print("rank ", newcomm.rank)
    else:
        print("size ", comm.size)
        print("rank ", comm.rank)

    from dolfinx.cpp.mesh import entities_to_geometry, exterior_facet_indices, CellType
    # if comm.rank == 1:
    #     fenics_mesh = dolfinx.UnitCubeMesh(newcomm, N, N, N)

    # if comm.rank == 0:
    #     subcomm = MPI.COMM_NULL

        # print(exterior_facet_indices(fenics_mesh))
    print("done", comm.Get_rank())
    MPI.Finalize()
# test_subcomms_split(2)

test_subcomms(2)
# test_mpi()
