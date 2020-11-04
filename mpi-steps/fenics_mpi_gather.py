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

def bm_from_fenics_mesh_mpi(fenics_mesh, fenics_space):
    boundary = entities_to_geometry( fenics_mesh,
            fenics_mesh.topology.dim - 1,
            exterior_facet_indices(fenics_mesh),
            True,
            )
    dofmap = fenics_space.dofmap.index_map.global_indices(False)
    print("dofmap ", dofmap)
    # print("number of facets ", len(exterior_facet_indices(fenics_mesh)))
    bm_nodes = set()
    for i, tri in enumerate(boundary):
        for j, node in enumerate(tri):
            # print(node, boundary[i][j])
            glob_node = dofmap[node]
            boundary[i][j] = glob_node
            bm_nodes.add(node)
    
    bm_nodes_global = [ dofmap[i] for i in bm_nodes ]
    bm_nodes = list(bm_nodes)
    # bm_cells - remap cell indices between 0-len(bm_nodes) 
    # bm_cells = np.array([[bm_nodes.index(i) for i in tri] for tri in boundary])
    bm_coords = fenics_mesh.geometry.x[bm_nodes]
    print(bm_coords)
    print(bm_nodes)
    print(boundary)
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
        # comm = MPI.COMM_SELF
        assert newcomm == MPI.COMM_NULL
        info = MPI.Status()
        
        # BM_CELLS
        # comm.Probe(MPI.ANY_SOURCE,0,info)
        # elements = info.Get_elements(MPI.LONG)
        # bm_cells = np.zeros(elements, dtype=np.int64)
        # comm.Recv([bm_cells, MPI.LONG], source=1, tag=0)
        # bm_cells = bm_cells.reshape(int(elements/3),3)

        # # BM_COORDS
        # comm.Probe(MPI.ANY_SOURCE,1,info)
        # elements = info.Get_elements(MPI.DOUBLE)
        # bm_coords = np.zeros(elements, dtype=np.float64) 
        # comm.Recv([bm_coords, MPI.DOUBLE], source=1, tag=1)
        # bm_coords = bm_coords.reshape(int(elements/3),3)

        # BM_NODES
        # comm.Probe(MPI.ANY_SOURCE,2,info)
        # elements = info.Get_elements(MPI.INT)
        # bm_nodes = np.zeros(elements, dtype=np.int32) 
        # comm.Recv([bm_nodes, MPI.INT], source=1, tag=2)
        # print(bm_nodes)
        # print(bm_coords)
        # print(bm_cells)
    else:
        fenics_mesh = dolfinx.UnitCubeMesh(newcomm, N, N, N)
        with XDMFFile(newcomm, "box.xdmf", "w") as file:
            file.write_mesh(fenics_mesh)

        fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))

        bm_nodes_global, bm_coords, boundary = bm_from_fenics_mesh_mpi(fenics_mesh, fenics_space)

        bm_nodes_global_list = list(bm_nodes_global)
        bm_nodes_arr = np.asarray(bm_nodes_global_list, dtype=np.int64)
        sendbuf = boundary
        sendbuf_coords = bm_coords
        sendbuf_nodes = bm_nodes_arr
        
        recvbuf_boundary = None
        recvbuf_coords = None
        recvbuf_nodes = None
        if newcomm.rank == 0:
            # recvbuf = np.empty(newcomm.size*len(bm_nodes_arr), dtype=np.int64)
            info = MPI.Status()
            # comm.Probe(MPI.ANY_SOURCE,MPI.ANY_TAG,info)
            # elements = info.Get_elements(MPI.LONG)
            # print("elements ", elements)
            recvbuf_boundary = np.empty(newcomm.size * len(boundary) * 3, dtype=np.int32)
            recvbuf_coords = np.empty(newcomm.size * len(bm_coords) * 3, dtype=np.float64)
            recvbuf_nodes = np.empty(newcomm.size * len(bm_nodes_arr), dtype=np.int64)

        newcomm.Gather(sendbuf, recvbuf_boundary, root=0)
        newcomm.Gather(sendbuf_coords, recvbuf_coords, root=0)
        newcomm.Gather(sendbuf_nodes, recvbuf_nodes, root=0)
        if newcomm.rank == 0:
            all_boundary = recvbuf_boundary.reshape(int(len(recvbuf_boundary)/3),3)
            bm_coords = recvbuf_coords.reshape(int(len(recvbuf_coords)/3),3)
            bm_nodes = recvbuf_nodes
            # check if mapping is correct
            if newcomm.size == 2:
                bm_nodes = list(recvbuf_nodes)
                print("bm_nodes ", bm_nodes)
                indices = [i for i, x in enumerate(bm_nodes) if x == 3]
                assert bm_coords[indices[0]].all() == bm_coords[indices[1]].all()

            arr1inds = recvbuf_nodes.argsort()
            bm_nodes_sorted = recvbuf_nodes[arr1inds]
            bm_coords_sorted = bm_coords[arr1inds]
            
            bm_nodes, uni = np.unique(bm_nodes_sorted, return_index=True) 
            bm_coords = bm_coords_sorted[uni]
            
            bm_nodes_list = list(bm_nodes)
            
            # bm_cells - remap cell indices between 0-len(bm_nodes) 
            # this can be better
            bm_cells = np.array([[bm_nodes_list.index(i) for i in tri] for tri in all_boundary])
            print(bm_cells)
            # print(bm_nodes)
            # print(bm_coords)
            exit(0)
            bm_cells = np.array([[bm_nodes.index(i) for i in tri] for tri in all_boundary])
            print(len(bm_nodes))
            print(len(bm_cells))
            print(len(bm_coords))
            print(len(all_boundary))

            # bempp_boundary_grid = bempp.api.Grid(bm_coords.transpose(), bm_cells.transpose())
            # out = os.path.join("mpi-steps/bempp_out", fenics_mesh.name+".msh")
            # bempp.api.export(out, grid=bempp_boundary_grid)
            # print("exported mesh to", out)

test_mpi_p2p_alldata_gather()
