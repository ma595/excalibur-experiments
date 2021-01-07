
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
from scipy.sparse import coo_matrix, csr_matrix
from dolfinx.io import XDMFFile
import gather_fns as gfns
from petsc4py import PETSc

import bempp.api
class Counter:
    def __init__(self):
        self.count = 0

    def add(self, *args):
        self.count += 1

def read_mesh_from_file():
    comm = MPI.COMM_WORLD
    # encoding = dolfinx.cpp.io.XDMFFile.Encoding.ASCII
    encoding = XDMFFile.Encoding.HDF5
    path = "square.xdmf"
    infile = XDMFFile(comm, path, 'r', encoding)
    fenics_mesh = infile.read_mesh(name='square')
    print(fenics_mesh.name)

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

    # print("dofmap ", fenics_mesh.geometry.dofmap)
    return bm_nodes_global, bm_coords, boundary


# def p1_trace(fenics_space, fenics_mesh)

def FEniCS_dofs_to_vertices(newcomm, fenics_space, fenics_mesh):
    num_fenics_vertices_proc = fenics_mesh.topology.connectivity(0, 0).num_nodes
    # print("num fenics vertices ", num_fenics_vertices_proc)

    tets = fenics_mesh.geometry.dofmap
    # print("num fenics tets ", len(tets))

    # print(tets)
    dofmap = fenics_space.dofmap.index_map.global_indices(False)
    geom_map = fenics_mesh.geometry.index_map().global_indices(False)
    dofmap_mesh = fenics_mesh.geometry.dofmap
    # print("dofmap space ", dofmap)
    # print("dofmap geometry ", dofmap_mesh)
    
def test_mpi_p2p_alldata_gather():
        fenics_mesh = dolfinx.UnitCubeMesh(newcomm, N, N, N)
        with XDMFFile(newcomm, "box.xdmf", "w") as file:
            file.write_mesh(fenics_mesh)

        fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))

        u = ufl.TrialFunction(fenics_space)
        v = ufl.TestFunction(fenics_space)
        k = 2

        form = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k ** 2 * ufl.inner(u, v)) * ufl.dx

        bm_nodes_global, bm_coords, bm_tris = bm_from_fenics_mesh_mpi(fenics_mesh, fenics_space)
        A = dolfinx.fem.assemble_matrix(form)
        A.assemble()

        actual = dolfinx.Function(fenics_space)
        actual.interpolate(lambda x: np.exp(1j * k * x[0]))
        actual_vec = actual.vector[:]

        # ai, aj, av = A.getValuesCSR()
        # Asp = csr_matrix((av, aj, ai))
        # print(Asp)
        # Asp_array = Asp.toarray()
        # Asp_1 = csr_matrix(Asp_array)
        # assert Asp_1.all() == Asp.all()
        # print(Asp_1)
        # print(Asp)

        global_alldofs_proc = fenics_space.dofmap.index_map.global_indices(False)
        bm_nodes_global_list = list(bm_nodes_global)
        bm_nodes_nparr = np.asarray(bm_nodes_global_list, dtype=np.int64)
        pregathbuf_bm_tris = bm_tris
        pregathbuf_bm_coords = bm_coords
        pregathbuf_bm_nodes = bm_nodes_nparr
        pregathbuf_global_alldofs = np.asarray(global_alldofs_proc, dtype=np.int32)
        # pregathbuf_global_vertices = 
        gathered_bm_tris = None
        gathered_bm_coords = None
        gathered_bm_nodes = None
        gathered_global_alldofs = None
        
        rank = newcomm.Get_rank()
        # number cols = total num rows?
        print("PRINT counts ")
        # print("pregathbuf_bdry", len(pregathbuf_bdry), rank)
        # print("bm_coords ", len(pregathbuf_coords), rank)
        # print("nodes ", len(pregathbuf_bm_nodes), rank)
        recvbuf_A = gfns.gather_petsc_matrix(A, newcomm)
        recvbuf_actual = gfns.create_gather_to_zero_vec(actual.vector)(actual.vector)[:]
        # print("recvbuf_actual ", recvbuf_actual)

        sendcounts_bm_tris = np.array(newcomm.gather(len(bm_tris), root=0))
        sendcounts_bm_coords = np.array(newcomm.gather(len(bm_coords), root=0))
        sendcounts_alldofs = np.array(newcomm.gather(len(global_alldofs_proc), root=0))
        print("alldofs ", sendcounts_alldofs)
        print(len(bm_nodes_nparr), len(pregathbuf_bm_coords), len(bm_tris))
        # print(sendcounts)
        # Allocate memory for gathered data on subprocess 0. 
        if newcomm.rank == 0:
            info = MPI.Status()
            gathered_bm_tris = np.empty(sum(sendcounts_bm_tris) * 3, dtype=np.int32)
            gathered_bm_coords = np.empty(sum(sendcounts_bm_coords) * 3, dtype=np.float64)
            gathered_bm_nodes = np.empty(sum(sendcounts_bm_coords), dtype=np.int64)
            gathered_global_alldofs = np.empty(sum(sendcounts_alldofs), dtype=np.int32)
            # recvbuf_allnodes = np.empty(newcomm.size * len())
            # recvbuf_dofs = np.empty(newcomm.size * len(bm_dofs))
            # recvbuf_soln = np.empty(newcomm.size*
        # Receive on subprocess 0. 
        newcomm.Gather(pregathbuf_bm_tris, gathered_bm_tris, root=0)
        newcomm.Gather(pregathbuf_bm_coords, gathered_bm_coords, root=0)
        newcomm.Gather(pregathbuf_bm_nodes, gathered_bm_nodes, root=0)
        newcomm.Gather(pregathbuf_global_alldofs, gathered_global_alldofs, root=0)
        # print(gathered_bdry_nodes)
        # print(fenics_space.dim)
        FEniCS_dofs_to_vertices(newcomm, fenics_space, fenics_mesh)
        
        # Find unique nodes in the gathered array. 
        if newcomm.rank == 0:
            # Sort and get unique nodes
            num_fenics_vertices = len(np.unique(np.sort(gathered_global_alldofs)))

            all_boundary = gathered_bm_tris.reshape(int(len(gathered_bm_tris)/3),3) # 48 (48)
            bm_coords = gathered_bm_coords.reshape(int(len(gathered_bm_coords)/3),3) # 34 (26) 
            bm_nodes = gathered_bm_nodes # 34 (26)
            # print(len(bm_nodes))
            # print(len(all_boundary))
            # print(len(bm_coords))
            # Sort the nodes (on global geom node indices) to make the unique faster? 
            sorted_indices = gathered_bm_nodes.argsort()
            bm_nodes_sorted = gathered_bm_nodes[sorted_indices]
            bm_coords_sorted = bm_coords[sorted_indices]
            # print("sorted indices, ", sorted_indices)
            
            bm_nodes, unique = np.unique(bm_nodes_sorted, return_index=True) 
            bm_coords = bm_coords_sorted[unique]
            bm_nodes_list = list(bm_nodes)
            # print("bm_nodes_list", bm_nodes_list) 
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
            comm.Send([recvbuf_A, MPI.DOUBLE_COMPLEX], dest=0, tag=112)
            comm.send(num_fenics_vertices, dest=0, tag=103)
            # comm.Send([recvbuf_av, MPI.DOUBLE_COMPLEX], dest=0, tag=112)
            comm.Send([recvbuf_actual, MPI.DOUBLE_COMPLEX], dest=0, tag=104)
            print("num_fenics_vertices presend", num_fenics_vertices)


test_mpi_p2p_alldata_gather()
# read_mesh_from_file()
