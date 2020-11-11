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

# return global node indices. 
def bm_from_fenics_mesh_mpi(fenics_mesh, fenics_space):
    boundary = entities_to_geometry( fenics_mesh,
            fenics_mesh.topology.dim - 1,
            exterior_facet_indices(fenics_mesh),
            True,
            )
    dofmap = fenics_space.dofmap.index_map.global_indices(False)
    geom_map = fenics_mesh.geometry.index_map().global_indices(False)
    dofmap_mesh = fenics_mesh.geometry.dofmap

    assert dofmap == geom_map
    # print("dofmap ", dofmap)
    # print("geometry map ", geom_map)
    # print("dofmap mesh", dofmap_mesh)
    # print("number of facets ", len(exterior_facet_indices(fenics_mesh)))
    bm_nodes = set()
    for i, tri in enumerate(boundary):
        for j, node in enumerate(tri):
            # print(node, boundary[i][j])
            glob_geom_node = geom_map[node]
            boundary[i][j] = glob_geom_node
            bm_nodes.add(node)

    bm_nodes_global = [ geom_map[i] for i in bm_nodes ]
    bm_nodes = list(bm_nodes)
    bm_coords = fenics_mesh.geometry.x[bm_nodes]
    # bm_cells - remap cell indices between 0-len(bm_nodes) 
    # bm_cells = np.array([[bm_nodes.index(i) for i in tri] for tri in boundary])
    # print("bm_coords\n", bm_coords)
    # print("bm_nodes\n", bm_nodes)
    # print("boundary\n", boundary)
    # # print("type of bm_coords ", type(bm_nodes), len(bm_nodes))
    # print('shape bm_cells ', bm_cells.shape)
    # print('type bm_cells ', type(bm_cells))
    # print('bm_cells \n', bm_cells)
    return bm_nodes_global, bm_coords, boundary


def test_mpi_p2p_alldata_gather():
    N = 2
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank() 
    world_size = comm.Get_size()
    print(comm)
    print("world rank ", world_rank)

    group = comm.Get_group()
    newgroup = group.Excl([0])
    newcomm = comm.Create(newgroup)

    if world_rank == 0:
        assert newcomm == MPI.COMM_NULL
        info = MPI.Status()
        
        # BM_CELLS
        comm.Probe(MPI.ANY_SOURCE,100,info)
        elements = info.Get_elements(MPI.LONG)
        bm_cells = np.zeros(elements, dtype=np.int64)
        comm.Recv([bm_cells, MPI.LONG], source=1, tag=100)
        bm_cells = bm_cells.reshape(int(elements/3),3)

        # BM_COORDS
        comm.Probe(MPI.ANY_SOURCE,101,info)
        elements = info.Get_elements(MPI.DOUBLE)
        bm_coords = np.zeros(elements, dtype=np.float64) 
        comm.Recv([bm_coords, MPI.DOUBLE], source=1, tag=101)
        bm_coords = bm_coords.reshape(int(elements/3),3)

        # BM_NODES
        comm.Probe(MPI.ANY_SOURCE,102,info)
        elements = info.Get_elements(MPI.INT)
        bm_nodes = np.zeros(elements, dtype=np.int32)
        comm.Recv([bm_nodes, MPI.INT], source=1, tag=102)
        # print(bm_nodes)
        # print(bm_coords)
        # print(bm_cells)
        bempp_boundary_grid = bempp.api.Grid(bm_coords.transpose(), bm_cells.transpose())
        out = os.path.join("mpi-steps/bempp_out", "test_mesh.msh")
        bempp.api.export(out, grid=bempp_boundary_grid)
        print("exported mesh to", out)

    else: # world rank = 1, 2
        fenics_mesh = dolfinx.UnitCubeMesh(newcomm, N, N, N)
        with XDMFFile(newcomm, "box.xdmf", "w") as file:
            file.write_mesh(fenics_mesh)

        fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))
        bm_nodes_global, bm_coords, boundary = bm_from_fenics_mesh_mpi(fenics_mesh, fenics_space)
        bm_nodes_global_list = list(bm_nodes_global)
        bm_nodes_arr = np.asarray(bm_nodes_global_list, dtype=np.int64)
        sendbuf_bdry = boundary
        sendbuf_coords = bm_coords
        sendbuf_nodes = bm_nodes_arr
        recvbuf_boundary = None
        recvbuf_coords = None
        recvbuf_nodes = None

        # Allocate memory for gathered data on subprocess 0. 
        if newcomm.rank == 0:
            info = MPI.Status()
            # The 3 factor corresponds to fact that the array is concatenated
            recvbuf_boundary = np.empty(newcomm.size * len(boundary) * 3, dtype=np.int32)
            recvbuf_coords = np.empty(newcomm.size * len(bm_coords) * 3, dtype=np.float64)
            recvbuf_nodes = np.empty(newcomm.size * len(bm_nodes_arr), dtype=np.int64)

        # Receive on subprocess 0. 
        newcomm.Gather(sendbuf_bdry, recvbuf_boundary, root=0)
        newcomm.Gather(sendbuf_coords, recvbuf_coords, root=0)
        newcomm.Gather(sendbuf_nodes, recvbuf_nodes, root=0)

        # when we do the gather we get boundary node indices repetitions 
        # therefore we find unique nodes in the gathered array. 
        if newcomm.rank == 0:
            all_boundary = recvbuf_boundary.reshape(int(len(recvbuf_boundary)/3),3) # 48 (48)
            bm_coords = recvbuf_coords.reshape(int(len(recvbuf_coords)/3),3) # 34 (26) 
            bm_nodes = recvbuf_nodes # 34 (26)
            # print(len(bm_nodes))
            # print(len(all_boundary))
            # print(len(bm_coords))

            # Sort the nodes (on global geom node indices) to make the unique faster? 
            sorted_indices = recvbuf_nodes.argsort()
            bm_nodes_sorted = recvbuf_nodes[sorted_indices]
            bm_coords_sorted = bm_coords[sorted_indices]
            # print("sorted indices, ", sorted_indices)
            
            bm_nodes, unique = np.unique(bm_nodes_sorted, return_index=True) 
            bm_coords = bm_coords_sorted[unique]
            bm_nodes_list = list(bm_nodes)
            print("bm_nodes_list", bm_nodes_list) 
            # bm_cells - remap boundary triangle indices between 0-len(bm_nodes) - this can be improved
            bm_cells = np.array([[bm_nodes_list.index(i) for i in tri] for tri in all_boundary])
            
            # print(len(bm_nodes))
            # print(len(bm_cells))
            # print(len(bm_coords))
            # print(len(all_boundary))
            # print(bm_cells)
            # send to world process 0. 
            comm.Send([bm_cells, MPI.LONG], dest=0, tag=100)
            comm.Send([bm_coords, MPI.DOUBLE], dest=0,tag=101)
            comm.Send([np.array(bm_nodes, np.int32), MPI.LONG], dest=0, tag=102)

test_mpi_p2p_alldata_gather()
