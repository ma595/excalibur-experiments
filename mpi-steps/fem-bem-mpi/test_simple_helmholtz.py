import numpy as np
import dolfinx
import dolfinx.geometry
from mpi4py import MPI
import ufl
from scipy.sparse import csr_matrix
import pytest

from scipy.sparse import coo_matrix, csr_matrix

class Counter:
    def __init__(self):
        self.count = 0

    def add(self, *args):
        self.count += 1


def test_simple_helmholtz_problem(N):
    N = 2
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank() 
    world_size = comm.Get_size()
    # print(comm)
    print("world rank ", world_rank)

    # Set up subcommunicator
    group = comm.Get_group()
    newgroup = group.Excl([0])
    fenicsx_comm = comm.Create(newgroup)

    # FENics work on processes != 0 
    if world_rank != 0:
        fenics_mesh = dolfinx.UnitCubeMesh(fenicsx_comm, N, N, N)
        fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))

        u = ufl.TrialFunction(fenics_space)
        v = ufl.TestFunction(fenics_space)
        k = 2

        form = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k ** 2 * ufl.inner(u, v)) * ufl.dx
        A = dolfinx.fem.assemble_matrix(form)
        A.assemble()

        actual = dolfinx.Function(fenics_space)
        actual.interpolate(lambda x: np.exp(1j * k * x[0]))
        
        from fenicsx_mpi import bm_from_fenics_mesh, send_A_actual_hack

        bm_from_fenics_mesh(comm, fenicsx_comm, fenics_mesh, fenics_space)

        send_A_actual_hack(comm, fenicsx_comm, A, actual)
    # PROCESS 0: (BEMPP process)
    # Receive fenics_mesh from PROCESS 1 (fenics mesh distributed
    # across PROCESS 1 and PROCESS 2. 
    else:
    # assert newcomm == MPI.COMM_NULL
        import bempp.api
        from fenicsx_mpi import recv_fenicsx_bm, get_A_actual_hack
        bm_nodes, bm_cells, bm_coords, num_fenics_vertices = recv_fenicsx_bm(comm)
        # print(num_fenics_vertices)
        A, actual = get_A_actual_hack(comm, num_fenics_vertices)
        k = 2
        # print(len(bm_cells), len(bm_coords))
        bempp_boundary_grid = bempp.api.Grid(bm_coords.transpose(), bm_cells.transpose())
        space = bempp.api.function_space(bempp_boundary_grid, "P", 1)
        trace_space = space


	# this all need to be delegated to fenicsx_mpi	
	####
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
	#### 

        trace_matrix = b_vertices_from_vertices @ vertices_from_fenics_dofs
        bempp_space = bempp.api.function_space(trace_space.grid, "DP", 0)

        id_op = bempp.api.operators.boundary.sparse.identity(
            trace_space, bempp_space, bempp_space
        )
        mass = bempp.api.operators.boundary.sparse.identity(
            bempp_space, bempp_space, trace_space
        )
        dlp = bempp.api.operators.boundary.helmholtz.double_layer(
            trace_space, bempp_space, bempp_space, k
        )
        slp = bempp.api.operators.boundary.helmholtz.single_layer(
            bempp_space, bempp_space, bempp_space, k
        )

        rhs_fem = np.zeros(num_fenics_vertices)

        @bempp.api.complex_callable
        def u_inc(x, n, domain_index, result):
            result[0] = np.exp(1j * k * x[0])

        u_inc = bempp.api.GridFunction(bempp_space, fun=u_inc)
        rhs_bem = u_inc.projections(bempp_space)

        rhs = np.concatenate([rhs_fem, rhs_bem])

        from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
        from scipy.sparse.linalg.interface import LinearOperator

        blocks = [[None, None], [None, None]]

        trace_op = LinearOperator(trace_matrix.shape, lambda x: trace_matrix @ x)

        ai, aj, av = A.getValuesCSR()
        Asp = csr_matrix((av, aj, ai))

        blocks[0][0] = Asp
        blocks[0][1] = -trace_matrix.T * mass.weak_form().A
        blocks[1][0] = (0.5 * id_op - dlp).weak_form() * trace_op
        blocks[1][1] = slp.weak_form()

        blocked = BlockedDiscreteOperator(np.array(blocks))

        from scipy.sparse.linalg import gmres

        c = Counter()
        soln, info = gmres(blocked, rhs, callback=c.add)

        print("Solved in", c.count, "iterations")
        computed = soln[: num_fenics_vertices]

        # print(actual)
        print("L2 error:", np.linalg.norm(actual - computed))
        assert np.linalg.norm(actual - computed) < 1 / N

        # out = os.path.join("./bempp_out", "test_mesh.msh")
        # bempp.api.export(out, grid=bempp_boundary_grid)
        # print("exported mesh to", out)

test_simple_helmholtz_problem(2)
