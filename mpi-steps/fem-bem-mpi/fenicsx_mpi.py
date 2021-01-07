# Get all relevant info from fenics_mesh (boundary mesh, dofs, everything we need) gather then send
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

def bm_from_fenics_mesh(comm, fenics_comm, fenics_mesh, fenics_space):
    from dolfinx.cpp.mesh import entities_to_geometry, exterior_facet_indices

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
    # print('shape bm_cells ', bm_cells.shape)
    # print('bm_cells \n', bm_cells)
    # print("dofmap ", fenics_mesh.geometry.dofmap)
    gathered_bm_coords = gather(fenics_comm, bm_coords, np.float64)
    gathered_bm_tris = gather(fenics_comm, boundary, np.int32)
    gathered_bm_nodes = gather(fenics_comm, np.asarray(bm_nodes_global, np.int32), np.int32)

    global_alldofs = np.asarray(fenics_space.dofmap.index_map.global_indices(False), dtype=np.int32)
    gathered_global_alldofs = gather(fenics_comm, global_alldofs, np.int32)

    if fenics_comm.rank == 0:
        all_bm_coords = gathered_bm_coords.reshape(int(len(gathered_bm_coords)/3),3) # 34 (26) 
        all_bm_tris = gathered_bm_tris.reshape(int(len(gathered_bm_tris)/3),3) # 48 (48)
        all_bm_nodes = gathered_bm_nodes # 34 (26)

        # Sort gathered nodes and remove repetitions (ghosts on bdry)
        sorted_indices = all_bm_nodes.argsort()
        all_bm_nodes_sorted = all_bm_nodes[sorted_indices]
        all_bm_coords_sorted = all_bm_coords[sorted_indices]
        # print("sorted indices, ", sorted_indices)
        all_bm_nodes, unique = np.unique(all_bm_nodes_sorted, return_index=True) 
        all_bm_coords = all_bm_coords_sorted[unique]
        all_bm_nodes_list = list(all_bm_nodes)
        # bm_cells - remap boundary triangle indices between 0-len(bm_nodes) - this can be improved
        all_bm_cells = np.array([[all_bm_nodes_list.index(i) for i in tri] for tri in all_bm_tris], dtype=np.int32)
        all_bm_nodes = np.asarray(all_bm_nodes_list, dtype=np.int32)
        # Send to Bempp process 
        send(comm, all_bm_coords, MPI.DOUBLE, 100)
        send(comm, all_bm_cells, MPI.LONG, 101)
        send(comm, all_bm_nodes, MPI.LONG, 102)

        num_fenics_vertices = len(np.unique(np.sort(gathered_global_alldofs)))
        comm.send(num_fenics_vertices, dest=0, tag=103)


def recv_fenicsx_bm(comm):
    info = MPI.Status() 
    bm_coords = recv(comm, MPI.DOUBLE, np.float64, info, 100)
    bm_coords = bm_coords.reshape(int(len(bm_coords)/3),3)
    bm_tris = recv(comm, MPI.INT, np.int32, info, 101)
    bm_tris = bm_tris.reshape(int(len(bm_tris)/3), 3)
    bm_nodes = recv(comm, MPI.INT, np.int32, info, 102)

    num_fenics_vertices = comm.recv(source=1, tag=103)
    return bm_nodes, bm_tris, bm_coords, num_fenics_vertices


def recv(comm, mpi_type, np_type, info, tag):
    comm.Probe(MPI.ANY_SOURCE, tag, info)
    elements = info.Get_elements(mpi_type)
    # print(elements)
    arr = np.zeros(elements, dtype=np_type)
    comm.Recv([arr, mpi_type], source=1, tag=tag)
    return arr

def gather(comm, arr, dtype):
    gathered_arr = None
    sendcounts = np.array(comm.gather(len(arr), root=0))
    if comm.rank == 0:
        info = MPI.Status()
        m = 1
        if arr.ndim == 2:
            m = 3
        gathered_arr = np.empty(sum(sendcounts)*m, dtype=dtype)
    comm.Gather(arr, gathered_arr, root=0)
    return gathered_arr

def send(comm, arr, dtype, tag):
    # comm.Send([recvbuf_A, MPI.DOUBLE_COMPLEX], dest=0, tag=112)
    comm.Send([arr, dtype], dest=0, tag=tag)
    
def send_A_actual_hack(comm, fenicsx_comm, A, actual):
    import gather_fns as gfns
    recvbuf_A = gfns.gather_petsc_matrix(A, fenicsx_comm)
    recvbuf_actual = gfns.create_gather_to_zero_vec(actual.vector)(actual.vector)[:]
    if fenicsx_comm.rank == 0: 
        comm.Send([recvbuf_actual, MPI.DOUBLE_COMPLEX], dest=0, tag=104)
        comm.Send([recvbuf_A, MPI.DOUBLE_COMPLEX], dest=0, tag=112)

def get_A_actual_hack(comm, num_fenics_vertices=27):
    info = MPI.Status()
    av = recv(comm, MPI.DOUBLE_COMPLEX, np.cdouble, info, 112)
    A = PETSc.Mat().create(comm=MPI.COMM_SELF)
    A.setSizes([num_fenics_vertices, num_fenics_vertices])
    A.setType("aij")
    A.setUp()
    A.setValues(list(range(0, num_fenics_vertices)), list(range(0, num_fenics_vertices)), av.reshape(num_fenics_vertices, num_fenics_vertices))
    A.assemble()
    actual = recv(comm, MPI.DOUBLE_COMPLEX, np.cdouble, info, 104)
    return A, actual


def p1_trace():
    dof_to_vertex_map = np.zeros(num_fenics_vertices, dtype=np.int64)

    b_vertices_from_vertices = coo_matrix(
        (np.ones(len(bm_nodes)), (np.arange(len(bm_nodes)), bm_nodes)),
        shape=(len(bm_nodes), num_fenics_vertices),
        dtype="float64",
    ).tocsc()

    dof_to_vertex_map = np.arange(num_fenics_vertices, dtype=np.int64)

    # print(dof_to_vertex_map)
    
    vertices_from_fenics_dofs = coo_matrix(
        (
        np.ones(num_fenics_vertices),
        (dof_to_vertex_map, np.arange(num_fenics_vertices)),
        ),
        shape=(num_fenics_vertices, num_fenics_vertices),
        dtype="float64",
    ).tocsc()
