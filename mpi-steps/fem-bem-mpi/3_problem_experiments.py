
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
from collections import OrderedDict, Counter

# N = 2 
comm = MPI.COMM_WORLD
# fenics_mesh = dolfinx.UnitCubeMesh(comm, N, N, N)

# print(tris)

def vertex_dofmap(fenics_mesh):
    tets = fenics_mesh.topology.connectivity(3, 0)
    # mesh_dofmap = fenics_mesh.geometry.dofmap
    dofmap_mesh = fenics_mesh.geometry.dofmap

    mapping = {}
    dof_map = {}
    # print(dofmap_mesh)
    l2g = fenics_mesh.topology.index_map(0).global_indices(False)
    geom_map = fenics_mesh.geometry.index_map().global_indices(False)
    for tetra in range(tets.num_nodes):
        for v in range(4):
        # print(tets.links(tetra))
            mapping[l2g[tets.links(tetra)[v]]] = geom_map[dofmap_mesh.links(tetra)[v]]
            dof_map[geom_map[dofmap_mesh.links(tetra)[v]]] = l2g[tets.links(tetra)[v]]

    # print(tets)

    # print(fenics_mesh.topology.index_map(2).global_indices(False))
    return mapping, dof_map

def space_dofmap(fenics_mesh, fenics_space):
    tets = fenics_mesh.topology.connectivity(3, 0)
    dofmap = fenics_space.dofmap.index_map.global_indices(False)
    l2g = fenics_mesh.topology.index_map(0).global_indices(False)
    # print(dofmap)
    # print(fenics_space.dofmap)
    mapping_space = {}
    for tetra in range(tets.num_nodes):
        cell_dofs = fenics_space.dofmap.cell_dofs(tetra)
        tet = tets.links(tetra)
        for v in range(4):
            mapping_space[l2g[tet[v]]] = dofmap[cell_dofs[v]]

    return mapping_space


def get_num_bdry_dofs(fenics_mesh):
    exterior_facets = exterior_facet_indices(fenics_mesh)
    boundary = entities_to_geometry( fenics_mesh,
            fenics_mesh.topology.dim - 1,
            exterior_facet_indices(fenics_mesh),
            True,
            )
    bm_nodes = set()
    fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))
    global_dofs_map = fenics_space.dofmap.index_map.global_indices(False)

    # print(fenics_space.dofmap.index_map.ghosts)
    mapping, dof_map = vertex_dofmap(fenics_mesh)
    # value of 9 
    vertex_err_val = 9

    if vertex_err_val in mapping.keys():
        print("value of corresponding dof is ", mapping[9])

    
    for i, tri in enumerate(boundary):
        for j, node in enumerate(tri):
            # print(node, boundary[i][j])
            glob_geom_node = global_dofs_map[node]
            # glob_geom_node = geom_map[node]
            boundary[i][j] = glob_geom_node
            bm_nodes.add(node)
    bm_nodes_global = [ global_dofs_map[i] for i in bm_nodes ]
    ghosts = fenics_space.dofmap.index_map.ghosts
    ghosts_list = list(ghosts)
    ghost_owner = fenics_space.dofmap.index_map.ghost_owner_rank()
    ownership = np.ones_like(bm_nodes_global, dtype=np.int32) * comm.rank
    for i in range(len(bm_nodes_global)):
        if bm_nodes_global[i] in ghosts_list:
            index = ghosts_list.index(bm_nodes_global[i])
            ownership[i] = ghost_owner[index]
    print(ownership)
    print(bm_nodes_global)
    print("RANK {} \n".format(comm.Get_rank()))


    root = 0
    rank = comm.Get_rank()
    sendbuf = np.array(bm_nodes_global)
    sendcounts = np.array(comm.gather(len(sendbuf), root))

    # print(sendcounts)

    if rank == root:
        print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
        recvbuf = np.empty(sum(sendcounts), dtype=np.int64)
    else:
        recvbuf = None

    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)
    if rank == root:
        bm_nodes_unique = sorted(np.unique(recvbuf))
        print("Gathered array: {}, unique length: {}".format(bm_nodes_unique, len(bm_nodes_unique)))
        return len(bm_nodes_unique)
    return 0
        # missing = 0
        # for i in range(len(bm_nodes_unique)):
        #     if i != bm_nodes_unique[i]:
        #         missing = i
        #         break
        # mapping, dof_map = vertex_dofmap(fenics_mesh)
        # print(missing, dof_map[missing])
        
def get_num_bdry_verts(fenics_mesh):
    num_fenics_vertices = fenics_mesh.topology.connectivity(0, 0).num_nodes
    fenics_mesh.topology.create_connectivity(2, 0)
    tris = fenics_mesh.topology.connectivity(2, 0)
    local2global_nodes = fenics_mesh.topology.index_map(0).global_indices(False)

    exterior_facets = exterior_facet_indices(fenics_mesh)
    exterior_nodes = set()

    for i in exterior_facets:
        for j in tris.links(i):
            exterior_nodes.add(local2global_nodes[j])
    root = 0
    rank = comm.Get_rank()

    ghosts = fenics_mesh.topology.index_map(0).ghosts
    ghost_owner = fenics_mesh.topology.index_map(0).ghost_owner_rank()
    all_global = fenics_mesh.topology.index_map(0).global_indices(False)
    all_indices = fenics_mesh.topology.index_map(0).indices(False)
    # print(sendbuf_vertices)
    ownership = np.ones_like(list(exterior_nodes), dtype=np.int32) * comm.rank
    ext_nodes_list = list(exterior_nodes)
    ghosts_list = list(ghosts)
    for i in range(len(ext_nodes_list)):
        if ext_nodes_list[i] in ghosts_list:
            index = ghosts_list.index(ext_nodes_list[i])
            ownership[i] = ghost_owner[index]
    # print(np.sum(ownership == rank))
    # print(sendbuf_vertices[ownership == rank])
    # print("\n")
    # print("exterior nodes length (global)", len(exterior_nodes))
    sendbuf_vertices = np.asarray(list(exterior_nodes), dtype=np.int64)
    print("ownership ", ownership) 
    print("external nodes from exterior_facet_indices ", ext_nodes_list)
    print("size local ", fenics_mesh.topology.index_map(0).size_local)
    print("all indices ", all_indices)
    print("RANK {} \n".format(rank))
    sendbuf_vertices = sendbuf_vertices[ownership == rank]
    # print(sum(sendbuf_vertices))
    sendcounts = np.array(comm.gather(len(sendbuf_vertices), root))
    # print("ghosts", fenics_mesh.topology.index_map(0).ghosts)
    # print("ghost owner rank ", fenics_mesh.topology.index_map(0).ghost_owner_rank())

    # print(sendcounts)
    if rank == root:
        # print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
        recvbuf = np.empty(sum(sendcounts), dtype=np.int64)
    else:
        recvbuf = None

    comm.Gatherv(sendbuf=sendbuf_vertices, recvbuf=(recvbuf, sendcounts), root=root)
    if rank == root:
        bdry_vertices = sorted(np.unique(recvbuf))
        # print("Gathered array: {}, unique length: {}".format(bdry_vertices, len(bdry_vertices)))
        # print("unique length: {}".format(len(bdry_vertices)))
        return len(bdry_vertices)
    return 0
        # get missing integer:
        # mapping, dof_map = vertex_dofmap(fenics_mesh)
   
        # missing = 0
        # for i in range(len(bdry_vertices)):
        #     if i != bdry_vertices[i]:
        #         missing = i
        #         break
        # print(missing, mapping[missing])

def test_mesh_bdry(fenics_mesh):
    rank = comm.Get_rank()
    num_bdry_verts = get_num_bdry_verts(fenics_mesh)
    if rank == 0:
        print(comm.Get_size(), num_bdry_verts)

    # num_bdry_dofs = get_num_bdry_dofs(fenics_mesh)
    # if rank == 0:
    #     print(comm.Get_size(), num_bdry_dofs)



def play(fenics_mesh):
    exterior_facets = exterior_facet_indices(fenics_mesh)
    num_fenics_vertices = fenics_mesh.topology.connectivity(0, 0).num_nodes
    tets = fenics_mesh.topology.connectivity(3, 0)
    fenics_mesh.topology.create_connectivity(2, 0)
    tris = fenics_mesh.topology.connectivity(2, 0)
    fenics_mesh.topology.create_connectivity(2, 3)
    tri_to_tet = fenics_mesh.topology.connectivity(2, 3)
    exterior_nodes = set()
    local2global_nodes = fenics_mesh.topology.index_map(0).global_indices(False)

    # from exterior facets get all nodes (global) on bdry
    # is the bdry triangle index the same as in tris?
    for i in exterior_facets:
        for j in tris.links(i):
            exterior_nodes.add(local2global_nodes[j])

    # print("ghost owner rank ", fenics_mesh.topology.index_map(0))
    # print("index_map ", dir(fenics_mesh.topology.index_map(0)))
    ghosts = fenics_mesh.topology.index_map(0).ghosts
    # print("l2g ", local2global_nodes)
    # ghosts_global = [i for i in ghosts]
    print("ghosts", fenics_mesh.topology.index_map(0).ghosts)
    # print("ghosts_global", fenics_mesh.topology.index_map(0).ghosts)
    print("ghost owner rank ", fenics_mesh.topology.index_map(0).ghost_owner_rank())
    print("all global indices on this process", fenics_mesh.topology.index_map(0).indices(True))
    # print(dir(fenics_mesh.topology.index_map(0)))
    size_local = fenics_mesh.topology.index_map(0).size_local
    size_global = fenics_mesh.topology.index_map(0).size_global
    # shared = fenics_mesh.topology.index_map(0).shared_indices
    print("size local ", size_local, " size global ", size_global)
    print("exterior nodes (global)", exterior_nodes)
    print("exterior nodes length (global)", len(exterior_nodes))
    sendbuf_nodes = np.asarray(list(exterior_nodes), dtype=np.int64)
    print("length of sendbuf ", len(sendbuf_nodes))
    # print("exterior facets ", exterior_facets)
    # print("exterior facet 0", [ local2global_nodes[i] for i in tris.links(0)])
    print("RANK ", comm.Get_rank())
# print(local2global_nodes)
    boundary = entities_to_geometry( fenics_mesh,
            fenics_mesh.topology.dim - 1,
            exterior_facet_indices(fenics_mesh),
            True,
            )
    bm_nodes = set()
    # this map is wrong:
    geom_map = fenics_mesh.geometry.index_map().global_indices(False)
    # use this instead:
    fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))
    global_dofs_map = fenics_space.dofmap.index_map.global_indices(False)
    mapping_space = space_dofmap(fenics_mesh, fenics_space)
    # print("mapping space ", mapping_space)
    # print(mapping)
    exterior_dofs = []
    for i in exterior_nodes:
        exterior_dofs.append(mapping_space[i])
    
    print("exterior dofs from vertices", exterior_dofs)
    for i, tri in enumerate(boundary):
        for j, node in enumerate(tri):
            # print(node, boundary[i][j])
            glob_geom_node = global_dofs_map[node]
            # glob_geom_node = geom_map[node]
            boundary[i][j] = glob_geom_node
            bm_nodes.add(node)
    bm_nodes_global = [ geom_map[i] for i in bm_nodes ]
    print("bm nodes ", bm_nodes_global)
    mapping, dof_map = vertex_dofmap(fenics_mesh)
    # print("geom nodes     ", bm_nodes_global)
    # print(exterior_facets)
    # get mapping between dofs and vertices
    mesh_dofmap = fenics_mesh.geometry.dofmap
    # mapping = vertex_dofmap(fenics_mesh)
    # print(mapping[13])
    mesh_dofs = []
    for i in exterior_nodes:
        # print(i)
        mesh_dofs.append(mapping[i])
    # print("mapping     ", mapping)
    # print("dof_map     ", dof_map)
    ma = [dof_map[i] for i in bm_nodes_global]
    print(ma)
    print("\n")

    # print("bm_nodes global ", sorted(bm_nodes_global))

    print("do a gather")
   
    recvbuf_nodes = None
    if comm.rank == 0:
        info = MPI.Status()
        recvbuf_nodes = np.empty(comm.Get_size() * len(exterior_nodes), dtype=np.int64)
    comm.Gather(sendbuf_nodes, recvbuf_nodes, root=0)

    print("received nodes ", len(np.unique(recvbuf_nodes)))
    exit(0)

def play_with_meshtags():
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 5, 5, 5)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(0, tdim)
    indices = np.arange(mesh.topology.index_map(tdim).size_local)
    values = np.ones_like(indices, dtype=np.int32) * MPI.COMM_WORLD.rank
    cell_owner = MeshTags(mesh, tdim, indices, values)
    with XDMFFile(MPI.COMM_WORLD, "cell_owner.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_meshtags(cell_owner)

def old_way(fenics_mesh):
    fenics_mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)
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

    print(sorted(surface_verts))
    print(surface_tris)
    print(tris)

# play(fenics_mesh)
# old_way(fenics_mesh)
# get_num_bdry_dofs(fenics_mesh)
# get_num_bdry_verts(fenics_mesh)
N = 5
fenics_mesh = dolfinx.UnitCubeMesh(comm, N, N, N)
test_mesh_bdry(fenics_mesh)



# fenics_mesh.topology.create_connectivity(2, 0)
# tris = fenics_mesh.topology.connectivity(2, 0)
# exterior_facets = exterior_facet_indices(fenics_mesh)
# print(tris)
# print(exterior_facets)
# tets = fenics_mesh.topology.connectivity(3, 0)
# print(fenics_mesh.geometry.dofmap)
# print(tets)
# how to get dofs on bdry corresponding to vertices. 
# print(tris)
# print("\n")
# # print(tris)
# print("index_map faces", fenics_mesh.topology.index_map(2).indices(True))
# print("index_map nodes", fenics_mesh.topology.index_map(0).indices(True))
# # print("index_map 2", dir(fenics_mesh.topology.index_map(2)))
# # print("index_map 1", dir(fenics_mesh.topology.index_map(1)))
# # print("index_map 0", dir(fenics_mesh.topology.index_map(0)))
# print("global node indices ", fenics_mesh.topology.index_map(0).global_indices(False))
# print("global facet indices ", fenics_mesh.topology.index_map(2).global_indices(False))
# print("topology fns ", dir(fenics_mesh.topology.index_map(0).global_indices(False)))
# print("facet indices\n ", facet_indices)
# print(fenics_mesh.geometry.dofmap)
# print(fenics_mesh.geometry.index_map().global_indices(False))
# fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))
# print(fenics_space.dofmap.index_map.global_indices(False))
# print("topology index map", fenics_mesh.topology.index_map(0))
# print(fenics_mesh.topology.connevtivity(0, 0).num_nodes)
