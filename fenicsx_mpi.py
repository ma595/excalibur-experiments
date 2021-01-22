import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

def bm_from_fenics_mesh(comm, fenics_comm, fenics_mesh, fenics_space):
    """
    Create a Bempp boundary grid from a FEniCS Mesh.
    Return the Bempp grid and a map from the node numberings of the FEniCS
    mesh to the node numbers of the boundary grid.
    """
    from dolfinx.cpp.mesh import entities_to_geometry, exterior_facet_indices

    boundary = entities_to_geometry( fenics_mesh,
            fenics_mesh.topology.dim - 1,
            exterior_facet_indices(fenics_mesh),
            True,
            )

    dofmap = fenics_space.dofmap.index_map.global_indices(False)
    geom_map = fenics_mesh.geometry.index_map().global_indices(False)
    dofmap_mesh = fenics_mesh.geometry.dofmap

    # assert dofmap == geom_map
    # print("number of facets ", len(exterior_facet_indices(fenics_mesh)))
    bm_nodes = set()
    for i, tri in enumerate(boundary):
        for j, node in enumerate(tri):
            # print(node, boundary[i][j])
            glob_dof_node = dofmap[node]
            boundary[i][j] = glob_dof_node
            bm_nodes.add(node)

    bm_nodes_global = [ dofmap[i] for i in bm_nodes ]
    bm_nodes = list(bm_nodes)
    bm_coords = fenics_mesh.geometry.x[bm_nodes]
    # bm_cells - remap cell indices between 0-len(bm_nodes) 
    # bm_cells = np.array([[bm_nodes.index(i) for i in tri] for tri in boundary])
    # print('shape bm_cells ', bm_cells.shape)
    # print('bm_cells \n', bm_cells)
    # print('bm_coords len ', len(bm_coords))
    # print('bm_coords type ', type(bm_coords[0][0]))
    # print("RANK ", fenics_comm.rank)
    gathered_bm_coords = gather(fenics_comm, bm_coords, 3, np.float64)
    gathered_bm_tris = gather(fenics_comm, boundary, 3, np.int32)
    gathered_bm_nodes = gather(fenics_comm, np.asarray(bm_nodes_global, np.int32), 1, np.int32)

    global_alldofs = np.asarray(fenics_space.dofmap.index_map.global_indices(False), dtype=np.int32)
    gathered_global_alldofs = gather(fenics_comm, global_alldofs, 1, np.int32)

    if fenics_comm.rank == 0:
        all_bm_coords = gathered_bm_coords.reshape(int(len(gathered_bm_coords)/3),3)
        all_bm_tris = gathered_bm_tris.reshape(int(len(gathered_bm_tris)/3),3) 
        all_bm_nodes = gathered_bm_nodes 

        # sort gathered nodes and remove repetitions (ghosts on bdry)
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
        # send to Bempp process 
        send(comm, all_bm_coords, MPI.DOUBLE, 100)
        send(comm, all_bm_cells, MPI.INT, 101)
        send(comm, all_bm_nodes, MPI.INT, 102)
        # print("all_bm_cells ", type(all_bm_cells))

        num_fenics_vertices = len(np.unique(np.sort(gathered_global_alldofs)))

        # hack - to change
        comm.send(num_fenics_vertices, dest=0, tag=103)

def get_num_fenics_vertices_unique_fenics(fenicsx_comm, fenics_space):
    global_alldofs = np.asarray(fenics_space.dofmap.index_map.global_indices(False), dtype=np.int32)
    gathered_global_alldofs = gather(fenicsx_comm, global_alldofs, 1, np.int32)
    
    num_fenics_vertices = len(np.unique(np.sort(gathered_global_alldofs)))
    return num_fenics_vertices

def recv_fenicsx_bm(comm):
    info = MPI.Status() 
    bm_coords = recv(comm, MPI.DOUBLE, np.float64, info, 100)
    bm_coords = bm_coords.reshape(int(len(bm_coords)/3),3)
    bm_tris = recv(comm, MPI.INT, np.int32, info, 101)
    bm_tris = bm_tris.reshape(int(len(bm_tris)/3), 3)
    bm_nodes = recv(comm, MPI.INT, np.int32, info, 102)

    # print(bm_coords, bm_tris, bm_nodes)
    num_fenics_vertices = comm.recv(source=1, tag=103)
    return bm_nodes, bm_tris, bm_coords, num_fenics_vertices

def recv(comm, mpi_type, np_type, info, tag):
    comm.Probe(MPI.ANY_SOURCE, tag, info)
    elements = info.Get_elements(mpi_type)
    # print(elements)
    arr = np.zeros(elements, dtype=np_type)
    comm.Recv([arr, mpi_type], source=1, tag=tag)
    return arr

def gather(comm, arr, mdim, dtype):
    gathered_arr = None
    # gathered_arr_1 = None
    sendcounts = np.array(comm.gather(len(arr)*mdim, root=0))
    if comm.rank == 0:
        gathered_arr = np.empty(sum(sendcounts), dtype=dtype)
        # gathered_arr_1 = np.empty([sum(sendcounts), mdim], dtype=dtype)
        # print("gathered_arr.shape ", gathered_arr.shape)
    comm.Gatherv(sendbuf=arr, recvbuf=(gathered_arr, sendcounts), root=0)
    return gathered_arr

def send(comm, arr, dtype, tag):
    # comm.Send([recvbuf_A, MPI.DOUBLE_COMPLEX], dest=0, tag=112)
    comm.Send([arr, dtype], dest=0, tag=tag)
    

def p1_trace(comm, fenicsx_comm, fenics_mesh, fenics_space):
    
    # not always guaranteed to be equivalent. 
    fs_dofs = fenics_space.dofmap.cell_dofs(0)
    tets = fenics_mesh.geometry.dofmap
    # print(fenics_mesh.geometry.index_map())
    # print("tetra ", fenics_mesh.topology.index_map(3).global_indices(False))
    # print("indices ", fenics_mesh.topology.index_map(0).global_indices(False))
    # print(fenics_mesh.topology.connectivity(3,0))
    # print(fenics_space.dofmap.dof_layout.entity_dofs(0,1))

    # vertices on single process
    num_fenics_vertices = fenics_mesh.topology.connectivity(0, 0).num_nodes
    # this map won't work because our num_fenics_vertices number is lower than the global number of vertices;
    # therefore will go out of bounds.
    # print("check ", fenics_space.dofmap.dof_layout.entity_dofs(0, 3)[0])
    
    geom_map = fenics_mesh.geometry.index_map().global_indices(False)
    dofs_map = fenics_space.dofmap.index_map.global_indices(False)
   
    # FEniCS dofs to vertices.
    dof_to_vertex_map = np.zeros(num_fenics_vertices, dtype=np.int64)
    dof_vertex_map_global = {}
    tets = fenics_mesh.geometry.dofmap
    for tet in range(tets.num_nodes):
        cell_dofs = fenics_space.dofmap.cell_dofs(tet)
        cell_verts = tets.links(tet)

        for v in range(4):
            vertex_n = cell_verts[v]
            dof = cell_dofs[fenics_space.dofmap.dof_layout.entity_dofs(0, v)[0]]
            dof_to_vertex_map[dof] = vertex_n
            
            dof_vertex_map_global[dofs_map[dof]] = geom_map[vertex_n]

    dof_vertex_map_global_np = np.array(list(dof_vertex_map_global.items()), dtype=np.int64)
    dof_vertex_map_global_np = dof_vertex_map_global_np[np.argsort(dof_vertex_map_global_np[:,0])]

    # print(dof_vertex_map_global_np)
    gathered_dof_vertex_map = gather(fenicsx_comm, dof_vertex_map_global_np, 2, np.int64)
    if fenicsx_comm.rank == 0:
        gathered_dof_vertex_map = gathered_dof_vertex_map.reshape(int(len(gathered_dof_vertex_map)/2),2)

        gathered_dof_vertex_map = gathered_dof_vertex_map[gathered_dof_vertex_map[:,0].argsort(kind='mergesort')] 
        # print(gathered_dof_vertex_map)
        dof_to_vertex_map = np.unique(gathered_dof_vertex_map, axis=0)[:,1]
        send(comm, dof_to_vertex_map.ravel(), MPI.LONG, 15)

# def fenics_to_bempp_trace_data():
#     """Returns tuple (space,trace_matrix)."""
#     family, degree = fenics_space_info(fenics_space)

#     if family == "Lagrange":
#         if degree == 1:
#             return p1_trace_recv()
#     else:
#         raise NotImplementedError()

def fenics_to_bempp_trace_data(comm):
    """
    Return the P1 trace operator.
    This function returns a pair (space, trace_matrix),
    where space is a Bempp space object and trace_matrix is the corresponding 
    matrix that maps the coefficients of a FEniCS function to its boundary trace coefficients in the corresponding Bempp space.
    """

    import bempp.api
    from scipy.sparse import coo_matrix
    import numpy as np

    # Recv fenics boundary 
    bm_nodes, bm_cells, bm_coords, num_fenics_vertices = recv_fenicsx_bm(comm)

    bempp_boundary_grid = bempp.api.Grid(bm_coords.transpose(), bm_cells.transpose())
    
    # First get trace space
    space = bempp.api.function_space(bempp_boundary_grid, "P", 1)

    # FEniCS vertices to bempp dofs
    b_vertices_from_vertices = coo_matrix(
        (np.ones(len(bm_nodes)), (np.arange(len(bm_nodes)), bm_nodes)),
        shape=(len(bm_nodes), num_fenics_vertices),
        dtype="float64",
    ).tocsc()

    info = MPI.Status()
    dof_to_vertex_map = recv(comm, MPI.LONG, np.int64, info, 15)
    # dof_to_vertex_map = np.arange(num_fenics_vertices, dtype=np.int32)
    
    vertices_from_fenics_dofs = coo_matrix(
        (
            np.ones(num_fenics_vertices),
            (dof_to_vertex_map, np.arange(num_fenics_vertices)),
        ),
        shape=(num_fenics_vertices, num_fenics_vertices),
        dtype="float64",
    ).tocsc()

    # Get trace matrix by multiplication
    trace_matrix = b_vertices_from_vertices @ vertices_from_fenics_dofs

    # Now return everything
    return space, trace_matrix, num_fenics_vertices


def send_A_actual_hack(comm, fenicsx_comm, A, actual):
    """
    Hack function (temporary) - send part
    """
    import gather_fns as gfns
    recvbuf_A = gfns.gather_petsc_matrix(A, fenicsx_comm)
    recvbuf_actual = gfns.create_gather_to_zero_vec(actual.vector)(actual.vector)[:]
    if fenicsx_comm.rank == 0: 
        comm.Send([recvbuf_actual, MPI.DOUBLE_COMPLEX], dest=0, tag=104)
        comm.Send([recvbuf_A, MPI.DOUBLE_COMPLEX], dest=0, tag=112)

def get_A_actual_hack(comm, num_fenics_vertices=27):
    """
    Hack function (temporary) - recv part
    """
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

