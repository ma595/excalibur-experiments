import numpy as np
import dolfinx
import dolfinx.geometry
from mpi4py import MPI
import ufl
from scipy.sparse import csr_matrix
import pytest
import sys
sys.path.append("./fem-bem/")
from dolfinx.cpp.mesh import entities_to_geometry, exterior_facet_indices

# the old incorrect way
def get_surface_tris(fenics_mesh):
    tets = fenics_mesh.topology.connectivity(3, 0)
    fenics_mesh.topology.create_connectivity(2, 0)
    tris = fenics_mesh.topology.connectivity(2, 0)
    fenics_mesh.topology.create_connectivity(2, 3)
    tri_to_tet = fenics_mesh.topology.connectivity(2, 3)
    surface_tris = []
    surface_verts = []
    for i in range(tris.num_nodes):
        if (len(tri_to_tet.links(i)) == 1): 
            surface_tris += [i]
            for v in tris.links(i):
                if v not in surface_verts:
                    surface_verts.append(v)
    surface_tris = np.array(surface_tris)
    bm_cells = []
    for cell in surface_tris:
        bm_cells.append([surface_verts.index(v) for v in tris.links(cell)])
    bm_cells = np.array(bm_cells)

    bm_coords = np.array([np.zeros(3) for i in surface_verts])

    # TODO: this can be made much better
    for tet in range(fenics_mesh.geometry.dofmap.num_nodes):
        for top_v, geo_v in zip(tets.links(tet), fenics_mesh.geometry.dofmap.links(tet)):
            if top_v in surface_verts:
                bm_coords[surface_verts.index(top_v)] = fenics_mesh.geometry.x[geo_v] 
    return surface_tris, surface_verts, bm_cells, bm_coords
# surface_tris, surface_verts, surface_cells, bm_coords_old = get_surface_tris(fenics_mesh)

# boundary now returns dofs 
# bm_coords are now ordered by dof
# bm_nodes are now dofs
def bm_from_fenics_mesh(fenics_mesh):
    boundary = entities_to_geometry( fenics_mesh,
            fenics_mesh.topology.dim - 1,
            exterior_facet_indices(fenics_mesh),
            True,
            )

    print("number of facets ", len(exterior_facet_indices(fenics_mesh)))
    bm_nodes = set()
    for tri in boundary:
        for node in tri:
            bm_nodes.add(node)
    bm_nodes = list(bm_nodes)
    # bm_cells - remap cell indices between 0-len(bm_nodes) 
    bm_cells = np.array([[bm_nodes.index(i) for i in tri] for tri in boundary])
    bm_coords = fenics_mesh.geometry.x[bm_nodes]
    fm_dofs = fenics_mesh.geometry.dofmap
    # print("type of bm_coords ", type(bm_nodes), len(bm_nodes))
    return bm_nodes, bm_cells, bm_coords

def test_mpi_p2p():
    N = 2
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank() 
    world_size = comm.Get_size()
    print(comm)
    print("world rank ", world_rank)

    # FEniCS mesh on rank 1 
    if world_rank == 1:
        # print("self ", r_comm)
        r_comm = MPI.COMM_SELF
        fenics_mesh = dolfinx.UnitCubeMesh(r_comm, N, N, N)
        bm_nodes, bm_cells, bm_coords = bm_from_fenics_mesh(fenics_mesh)
        bm_nodes_arr = np.asarray(bm_nodes, dtype='i')
        # comm.send(bm_nodes, dest=0, tag=11)
        comm.Send([bm_nodes_arr, MPI.INT], dest=0, tag=0)

    elif world_rank == 0:
        # comm = MPI.COMM_SELF
        # data = comm.recv(source=1, tag=11)
        info = MPI.Status()
        comm.Probe(MPI.ANY_SOURCE,MPI.ANY_TAG,info)
        elements = info.Get_elements(MPI.INT)
        # print(elements)
        data = np.zeros(elements, dtype='i')
        comm.Recv([data, MPI.INT], source=1, tag=0)
        print(data)




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

test_mpi_p2p()
