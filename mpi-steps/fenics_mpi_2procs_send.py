import numpy as np
import dolfinx
import dolfinx.geometry
from mpi4py import MPI
import ufl
from scipy.sparse import csr_matrix
import pytest
import sys, os
sys.path.append("./fem-bem/")
from dolfinx.cpp.mesh import entities_to_geometry, exterior_facet_indices
from scipy.sparse import coo_matrix
from dolfinx.io import XDMFFile
import bempp.api

def test_mpi_p2p_alldata():
    N = 2
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()
    print(comm)
    print("world rank ", world_rank)

    # FEniCS mesh on rank 1
    if world_rank == 1:
        # print("self ", r_comm)
        r_comm = MPI.COMM_SELF
        fenics_mesh = dolfinx.UnitCubeMesh(r_comm, N, N, N)
        fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))
        bm_nodes, bm_cells, bm_coords = bm_from_fenics_mesh(fenics_mesh)
        bm_nodes_arr = np.asarray(bm_nodes, dtype='i')
        # comm.send(bm_nodes, dest=0, tag=11)
        # comm.Send([bm_nodes_arr, MPI.INT], dest=0, tag=0)
        print(len(bm_cells))
        # print(type(bm_cells[0,0]))
        comm.Send([bm_cells, MPI.LONG], dest=0, tag=0)

        comm.Send([bm_coords, MPI.DOUBLE], dest=0,tag=1)
        # convert list to numpy array
        comm.Send([np.array(bm_nodes, np.int32), MPI.LONG], dest=0, tag=2)

    elif world_rank == 0:
        # comm = MPI.COMM_SELF
        info = MPI.Status()

        # BM_CELLS
        comm.Probe(MPI.ANY_SOURCE,0,info)
        elements = info.Get_elements(MPI.LONG)
        bm_cells = np.zeros(elements, dtype=np.int64)
        comm.Recv([bm_cells, MPI.LONG], source=1, tag=0)
        bm_cells = bm_cells.reshape(int(elements/3),3)

        # BM_COORDS
        comm.Probe(MPI.ANY_SOURCE,1,info)
        elements = info.Get_elements(MPI.DOUBLE)
        bm_coords = np.zeros(elements, dtype=np.float64)
        comm.Recv([bm_coords, MPI.DOUBLE], source=1, tag=1)
        bm_coords = bm_coords.reshape(int(elements/3),3)

        # BM_NODES
        comm.Probe(MPI.ANY_SOURCE,2,info)
        elements = info.Get_elements(MPI.INT)
        bm_nodes = np.zeros(elements, dtype=np.int32)
        comm.Recv([bm_nodes, MPI.INT], source=1, tag=2)
        print(bm_nodes)
        print(bm_coords)
        print(bm_cells)

# transfer bm_nodes, bm_cells, bm_coords

def test_mpi_p2p():
    N = 2
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank() 
    world_size = comm.Get_size()
    print(comm)
    print("world rank ", world_rank)

    # FEniCS mesh on rank 1
    if world_rank == 1:
        # print("self ", r_comm)
        r_comm = MPI.COMM_SELF
        fenics_mesh = dolfinx.UnitCubeMesh(r_comm, N, N, N)
        fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))
        bm_nodes, bm_cells, bm_coords = bm_from_fenics_mesh(fenics_mesh)
        bm_nodes_arr = np.asarray(bm_nodes, dtype='i')
        # comm.send(bm_nodes, dest=0, tag=11)
        # comm.Send([bm_nodes_arr, MPI.INT], dest=0, tag=0)
        print(len(bm_cells))
        print(type(bm_cells[0,0]))
        comm.Send([bm_cells, MPI.LONG], dest=0, tag=0)

    elif world_rank == 0:
        # comm = MPI.COMM_SELF
        # data = comm.recv(source=1, tag=11)
        info = MPI.Status()
        comm.Probe(MPI.ANY_SOURCE,MPI.ANY_TAG,info)
        elements = info.Get_elements(MPI.LONG)
        # print(elements)
        data = np.zeros(elements, dtype=np.int64)
        comm.Recv([data, MPI.LONG], source=1, tag=0)
        data = data.reshape(int(elements/3),3)
        print(data)
        # print(len(data))

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
        fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))

    group.Free(); newgroup.Free()
    if newcomm: newcomm.Free()

    print("done")

# boundary now returns dofs 
# bm_coords are now ordered by dof
# bm_nodes are now dofs
def bm_from_fenics_mesh(fenics_mesh):
    boundary = entities_to_geometry( fenics_mesh,
            fenics_mesh.topology.dim - 1,
            exterior_facet_indices(fenics_mesh),
            True,
            )

    # print("number of facets ", len(exterior_facet_indices(fenics_mesh)))
    bm_nodes = set()
    for tri in boundary:
        for node in tri:
            bm_nodes.add(node)
    bm_nodes = list(bm_nodes)
    # bm_cells - remap cell indices between 0-len(bm_nodes) 
    bm_cells = np.array([[bm_nodes.index(i) for i in tri] for tri in boundary])
    bm_coords = fenics_mesh.geometry.x[bm_nodes]

    # print(boundary)
    # print("fenics mesh dofs\n", fm_dofs)
    # # print("type of bm_coords ", type(bm_nodes), len(bm_nodes))
    # print('shape bm_cells ', bm_cells.shape)
    # print('type bm_cells ', type(bm_cells))
    # print('bm_cells \n', bm_cells)
    # print('bm_nodes \n', bm_nodes)
    return bm_nodes, bm_cells, bm_coords

test_mpi_p2p_alldata()
