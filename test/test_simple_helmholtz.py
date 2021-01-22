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
    print(fenicsx_comm)
    # FENics work on all processes != 0 
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
        
        from fenicsx_mpi import bm_from_fenics_mesh, p1_trace, send_A_actual_hack

        bm_from_fenics_mesh(comm, fenicsx_comm, fenics_mesh, fenics_space)

        send_A_actual_hack(comm, fenicsx_comm, A, actual)

        p1_trace(comm, fenicsx_comm, fenics_mesh, fenics_space)
    # PROCESS 0: (BEMPP process)
    # Receive fenics_mesh from PROCESS 1 (fenics mesh distributed
    # across PROCESS 1 and PROCESS 2. 
    else:
    # assert newcomm == MPI.COMM_NULL
        import bempp.api
        from fenicsx_mpi import  get_A_actual_hack, fenics_to_bempp_trace_data

        # print(num_fenics_vertices)

        trace_space, trace_matrix, num_fenics_vertices = fenics_to_bempp_trace_data(comm)

        A, actual = get_A_actual_hack(comm, num_fenics_vertices)
        k = 2
        # print(len(bm_cells), len(bm_coords))
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
        B = -trace_matrix.T * mass.weak_form().A
        C = (0.5 * id_op - dlp).weak_form() * trace_op
        D = slp.weak_form()
       
        # print("A ", A)
        # print("B ", B)
        # print("C ", C)
        # print("D ", D)
        
        lambda_0 = np.ones(48) * 0.1
        
        lambda_0 = np.array([ -0.3356277 -2.11393401j,  0.02514609+0.04718278j, -0.33520395-2.11406802j,
            0.02488649+0.0469925j,  0.23409561-0.21446884j, 0.23400113-0.21425613j, 
            -0.45940246-2.15994469j,  0.15385037+0.06197709j,  0.13949044-1.95379861j,
            0.23264441-0.14574814j,  0.30841645+0.16531784j,  0.13737633-0.10828095j,
            0.13644371-1.95408152j, -0.45718109-2.15976116j,  0.15361352+0.06301156j,
            0.23301017-0.14551304j, -0.02710572-2.00445983j, -0.02426736-2.00420245j, 
            0.31511422+0.14649406j,  0.11312288-0.02620189j,  0.31130568+0.16580552j,
            0.13298721-0.11031612j,  0.31892437+0.14608164j,  0.11018384-0.02619779j, 
            -1.81867122-0.84790512j,  0.00656279+0.34571095j, -1.82075977-0.84283439j,
            0.00732981+0.34381219j, -1.79530298-1.29931097j,  0.02859708+0.34507824j, 
            -1.85169611-0.66703723j, -0.07317962+0.0905053j , -0.01419597+0.16957234j, 
            -0.15886253+0.0816724j , -1.85071354-0.66968136j, -1.79650239-1.29485921j,
            0.02923799+0.3427498j , -0.07320608+0.0903135j , -1.81409468-1.16122048j,
            -1.81371538-1.16208464j,  0.02441639+0.04823583j, -0.236166  +0.15833326j,
            -0.01450401+0.16621932j, -0.15799446+0.08310207j,  0.02294841+0.05062881j,
            -0.23427732+0.15907834j, -0.30055134+0.12987961j, -0.30162485+0.1306874j ])
 
        blocks[0][0] = Asp
        blocks[0][1] = B
        blocks[1][0] = C
        blocks[1][1] = D

        from scipy.sparse.linalg import gmres

        # Example of one single pass through a naive implementation
        its = 5000
        i = 0
        sigma = 0.0001
        delta = 0.0001


        # u_n = np.zeros(27)
        # u_nm1 = np.zeros(27)
        # print(u_nm1)

        # print(hex(id(u_n)))
        # print(hex(id(u_nm1)))
        # print(hex(id(u_n)))
        # print(hex(id(u_nm1)))
        # u_nm1 = actual

        # u_n0 = actual
        # u_n = u_n0
        # lambda_tilde = lambda_0

        # while i < its:
        #     # first step 
        #     lambda_n = lambda_tilde
        #     b = rhs_fem - B*lambda_n
        #     u_np1, info = gmres(Asp, b)
        #     u_tilde = sigma*u_np1 + (1-sigma)*u_n
        #     u_n = u_tilde

        #     # second step 
        #     b = rhs_bem - C*u_tilde
        #     lambda_np1, info = gmres(D, b)
        #     lambda_tilde = delta*lambda_np1 + (1-delta)*lambda_n
        #     i += 1

        #     if i % 1000 == 0:
        #         print("L2 error:", np.linalg.norm(actual - u_np1))

        # arr = A.getSize()
        # print(B.todense().shape)
        # print("A ", type(A))
        # print("B ", type(B))
        # print("multiply 1", B.multiply(lambda_tilde))
        # print("multiply 2", B*lambda_tilde)
        # print("multiply 3", B.dot(lambda_tilde))


        # print("C ", C)
        # print("D ", D)

        # print(u_n)

        # print(actual)
    
        blocked = BlockedDiscreteOperator(np.array(blocks))

        c = Counter()
        soln, info = gmres(blocked, rhs, callback=c.add)

        print("Solved in", c.count, "iterations")
        computed = soln[: num_fenics_vertices]
        # print(soln[num_fenics_vertices :])
        # print(actual)
        print("L2 error:", np.linalg.norm(actual - computed))
        assert np.linalg.norm(actual - computed) < 1 / N

        # out = os.path.join("./bempp_out", "test_mesh.msh")
        # bempp.api.export(out, grid=bempp_boundary_grid)
        # print("exported mesh to", out)

test_simple_helmholtz_problem(2)
