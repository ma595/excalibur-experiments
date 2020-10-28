import dolfinx as _dolfin
import dolfinx.cpp.mesh as _dmesh
import time, os

def boundary_grid_from_fenics_mesh(fenics_mesh):
    """
    Create a Bempp boundary grid from a FEniCS Mesh.
    Return the Bempp grid and a map from the node numberings of the FEniCS
    mesh to the node numbers of the boundary grid.
    """
    import bempp.api
    import numpy as np

    # boundary_mesh = _dolfin.BoundaryMesh(fenics_mesh, "exterior", False) ##
    # bm_coords = boundary_mesh.coordinates()
    # bm_cells = boundary_mesh.cells()
    # bm_nodes = boundary_mesh.entity_map(0).array().astype(np.int64)
    # bempp_boundary_grid = bempp.api.Grid(bm_coords.transpose(), bm_cells.transpose())
    # return bempp_boundary_grid, bm_nodes

    tets = fenics_mesh.topology.connectivity(3, 0)
    fenics_mesh.topology.create_connectivity(2, 0)
    tris = fenics_mesh.topology.connectivity(2, 0)
    fenics_mesh.topology.create_connectivity(2, 3)
    tri_to_tet = fenics_mesh.topology.connectivity(2, 3)
    fenics_mesh.topology.create_connectivity(0, 3)
    node_to_tet = fenics_mesh.topology.connectivity(0, 3)
    # im2 = mesh.topology.index_map(2)
    # print('Index Map 2= ',im2.size_local, im2.size_global, im2.num_ghosts)
    # print("tets ", tets) 	
    # print("tri_to_tet ", tri_to_tet)
    start = time.time()
    surface_tris = []
    surface_verts = []
    surface_dof = []
    surface_verts_dict = {}
    # loops over all triangles 
    count = 0
    for i in range(tris.num_nodes):
      #  if i % 10**4 == 0:
      #      print(i)
        if (len(tri_to_tet.links(i)) == 1):
            surface_tris += [i]
#            for v in tris.links(i):
#                if v not in surface_verts: # this is slow (but number of vertices are relatively low)
#                    surface_verts.append(v)
            for v in tris.links(i):
                if v not in surface_verts_dict:
                    surface_verts_dict[v] = count # probably redundant as it's an ordered dict anyway (insertion).
                    count += 1 

    surface_tris = np.array(surface_tris)
    end = time.time()
    print("Number of total verts ", node_to_tet.num_nodes) # 154
    print("Number of total tetra ", len(tets))
    print("Number of surface verts ", len(surface_verts_dict))
    print("Number of surface triangles ", len(surface_tris))
   
    print("get surface vertices time ", end - start)
    # print(dir(fenics_mesh.geometry))
    
    # TODO: loop through and flip triangles whose normals are inwards
    #     if centre of tet is on the same side as normal is facing, flip it
    # mapping between nodes and dofs - glob_ind.index(0) = 5 , glob_ind.index(1) = 14, glob_ind.index(4) = 6  
    glob_ind = fenics_mesh.geometry.input_global_indices
    geo_x = fenics_mesh.geometry.x
    #  print("dofmap", fenics_mesh.geometry.dofmap)
    #  print(geo_x.sort)
    #  print(geo_x)
    # tri_normals = np.array([np.zeros(3) for i in surface_tris])
    start = time.time()

    nodes2dof = dict(zip(glob_ind, range(len(glob_ind)))) # storing redundant info because ordered perhaps?
    surface_tris_fixed = np.array([np.zeros(3) for i in surface_tris])
    for i, tri in enumerate(surface_tris):
        tri_nodes = tris.links(tri)
        #print("triangle ", tri, " nodes ", tris.links(tri), " Point ", fenics_mesh.geometry.x[glob_ind.index(tri_nodes[2])])
        p1 = geo_x[nodes2dof[tri_nodes[0]]]
        p2 = geo_x[nodes2dof[tri_nodes[1]]]
        p3 = geo_x[nodes2dof[tri_nodes[2]]]
        N = np.cross(p1-p2,p1-p3) 
        # get the corresponding tetrahedron and store centre
        tet = tri_to_tet.links(tri)
        tri_nodes = tris.links(tri)
        tet_nodes = tets.links(tet)
        centre = [0,0,0]
        for v in tet_nodes:
            centre += geo_x[nodes2dof[v]]
        centre = centre/4.
        #p1 = geo_x_sorted[tri_nodes[0]]
        f2c = centre - p1
        if np.dot(f2c, N) > 0:
            triangle = tris.links(tri)
            triangle[[0,1,2]] = triangle[[1,0,2]]
            surface_tris_fixed[i] = triangle
        else:
            surface_tris_fixed[i] = tris.links(tri)
  
    #print("surface tris\n", surface_tris)
    #print("fixed\n", surface_tris_fixed)
    end = time.time()
    print("correct normals time ", end - start)
    start = time.time()
    bm_coords = np.array([np.zeros(3) for i in surface_verts_dict])
    for top_v in surface_verts_dict:
        bm_coords[surface_verts_dict[top_v]] = fenics_mesh.geometry.x[nodes2dof[top_v]]
    # print(bm_coords)
    end = time.time()
    print("loop time", end - start)
    bm_cells = []
    for cell in surface_tris_fixed:
        #print(cell)
        bm_cells.append([surface_verts_dict[v] for v in cell])
    bm_cells = np.array(bm_cells)
   
   # print("Check output")
   # print("surface_verts\n", [*surface_verts_dict])
   # print("bm_coords\n", bm_coords)
   # print("bm_cells\n", bm_cells)
   
    out = False
    if out:
        filo = open(r"./check.out", "w")
        filo.write("surface_verts\n")
        filo.write(str([*surface_verts_dict]))
        filo.write("\nbm_coords\n")
        filo.write(str(bm_coords))
        filo.write("\nbm_cells\n")
        filo.write(str(bm_cells))

    bempp_boundary_grid = bempp.api.Grid(bm_coords.transpose(), bm_cells.transpose())
    out = os.path.join("bempp_out", fenics_mesh.name+".msh")
    bempp.api.export(out, grid=bempp_boundary_grid)
    print("exported mesh to", out)



    surface_verts = [i for i in range(len(surface_verts_dict))]
    inv_surface_vert_dict = {v: k for k, v in surface_verts_dict.items()}

    for i in range(len(surface_verts)):
        surface_verts[i] = inv_surface_vert_dict[i]
   # print(surface_verts)
    return bempp_boundary_grid, surface_verts


def fenics_to_bempp_trace_data(fenics_space):
    """
    Returns tuple (space,trace_matrix)
    """
    family, degree = fenics_space_info(fenics_space)

    if family == "Lagrange":
        if degree == 1:
            return p1_trace(fenics_space)
    else:
        raise NotImplementedError()


def fenics_space_info(fenics_space):
    """
    Returns tuple (family,degree) containing information about a FEniCS space
    """
    element = fenics_space.ufl_element()
    family = element.family()
    degree = element.degree()
    return (family, degree)


# pylint: disable=too-many-locals
def p1_trace(fenics_space):
    """
    Return the P1 trace operator.

    This function returns a pair (space, trace_matrix),
    where space is a Bempp space object and trace_matrix is the corresponding
    matrix that maps the coefficients of a FEniCS function to its boundary
    trace coefficients in the corresponding Bempp space.
    """

    import bempp.api
    from scipy.sparse import coo_matrix
    import numpy as np

    # Temporarily make a space and trace_matrix
    #bempp_space = bempp.api.function_space(bempp.api.shapes.cube(h=1), "P", 1)
    #min_dim = min(fenics_space.dim, bempp_space.global_dof_count)
    #trace_matrix = coo_matrix(
    #    (np.ones(min_dim), (list(range(min_dim)), list(range(min_dim)))),
    #    shape=(bempp_space.global_dof_count, fenics_space.dim),
    #    dtype="float64").tocsc()
    #return bempp_space, trace_matrix
    # End temp

    family, degree = fenics_space_info(fenics_space)
    if not (family == "Lagrange" and degree == 1):
        raise ValueError("fenics_space must be a p1 Lagrange space")

    fenics_mesh = fenics_space.mesh
    bempp_boundary_grid, bm_nodes = boundary_grid_from_fenics_mesh(fenics_mesh)

    # First get trace space
    space = bempp.api.function_space(bempp_boundary_grid, "P", 1)

    num_fenics_vertices = fenics_mesh.topology.connectivity(0,0).num_nodes

    # FEniCS vertices to bempp dofs
    b_vertices_from_vertices = coo_matrix(
        (np.ones(len(bm_nodes)), (np.arange(len(bm_nodes)), bm_nodes)),
        shape=(len(bm_nodes), num_fenics_vertices),
        dtype="float64",
    ).tocsc()

    # Finally FEniCS dofs to vertices.
    dof_to_vertex_map = np.zeros(num_fenics_vertices, dtype=np.int64)
    tets = fenics_mesh.topology.connectivity(3, 0)
    for tet in range(tets.num_nodes):
        cell_dofs = fenics_space.dofmap.cell_dofs(tet)
        cell_verts = tets.links(tet)
        for v in range(4):
            vertex_n = cell_verts[v]
            dof = cell_dofs[fenics_space.dofmap.dof_layout.entity_dofs(0, v)[0]]
            dof_to_vertex_map[dof] = vertex_n

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
    return space, trace_matrix
