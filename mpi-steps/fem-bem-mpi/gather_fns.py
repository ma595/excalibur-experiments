import numpy as np
from petsc4py import PETSc
def create_gather_to_zero_mat(pmat):
    """
    Create the ``gather_to_zero()`` function for collecting the global PETSc
    matrix on the task of rank zero.
    """
    g20, pvec_full =  PETSc.Scatter().toZero(pvec)

    def gather_to_zero(pvec):
        """
        Return the global PETSc vector, corresponding to `pvec`, on the task of
        rank zero. The vector is reused between calls!
        """
        g20.scatter(pvec, pvec_full, PETSc.InsertMode.INSERT,
                    PETSc.ScatterMode.FORWARD)

        return pvec_full

    return gather_to_zero

def create_gather_to_zero_vec(pvec):
    """
    Create the ``gather_to_zero()`` function for collecting the global PETSc
    vector on the task of rank zero.
    """
    g20, pvec_full =  PETSc.Scatter().toZero(pvec)

    def gather_to_zero(pvec):
        """
        Return the global PETSc vector, corresponding to `pvec`, on the task of
        rank zero. The vector is reused between calls!
        """
        g20.scatter(pvec, pvec_full, PETSc.InsertMode.INSERT,
                    PETSc.ScatterMode.FORWARD)

        return pvec_full

    return gather_to_zero


        # comm.Probe(MPI.ANY_SOURCE,111,info)
        # elements = info.Get_elements(MPI.INT)
        # aj = np.zeros(elements, dtype=np.int32)
        # comm.Recv([aj, MPI.INT], source=1, tag=111)
        # # print(aj)
        # comm.Probe(MPI.ANY_SOURCE,110,info)
        # elements = info.Get_elements(MPI.INT)
        # ai = np.zeros(elements, dtype=np.int32)
        # comm.Recv([ai, MPI.INT], source=1, tag=110)
        # # print("ai shape ", ai)

# def testGetSubMatrix(self):
#     if 'baij' in self.A.getType(): return # XXX
#     self._preallocate()
#     self._set_values_ijv()
#     self.A.assemble()
#     #
#     rank = self.A.getComm().getRank()
#     rs, re = self.A.getOwnershipRange()
#     cs, ce = self.A.getOwnershipRangeColumn()
#     rows = N.array(range(rs, re), dtype=PETSc.IntType)
#     cols = N.array(range(cs, ce), dtype=PETSc.IntType)
#     rows = PETSc.IS().createGeneral(rows, comm=self.A.getComm())
#     cols = PETSc.IS().createGeneral(cols, comm=self.A.getComm())
#     #
#     S = self.A.getSubMatrix(rows, None)
#     S.zeroEntries()
#     self.A.getSubMatrix(rows, None, S)
#     S.destroy()
#     #
#     S = self.A.getSubMatrix(rows, cols)
#     S.zeroEntries()
#     self.A.getSubMatrix(rows, cols, S)
#     S.destroy()

def gather_petsc_array(x, comm, out_shape=None):
    """Gather the petsc vector/matrix `x` to a single array on the master
    process, assuming that owernership is sliced along the first dimension.

    Parameters
    ----------
    x : petsc4py.PETSc Mat or Vec
        Distributed petsc array to gather.
    comm : mpi4py.MPI.COMM
        MPI communicator
    out_shape : tuple, optional
        If not None, reshape the output array to this.

    Returns
    -------
    gathered : np.array master, None on workers (rank > 0)
    """
    # get local numpy array
    lx = x.getArray() # this function doesn't work (though we can use the stuff below)
    # lx = 
    ox = np.empty(2, dtype=int)
    ox[:] = x.getOwnershipRange()

    # master only
    if comm.Get_rank() == 0:

        # create total array
        ax = np.empty(x.getSize(), dtype=lx.dtype)
        # set master's portion
        ax[ox[0]:ox[1], ...] = lx

        # get ownership ranges and data from worker processes
        for i in range(1, comm.Get_size()):
            comm.Recv(ox, source=i, tag=11)

            # receive worker's part of ouput vector
            comm.Recv(ax[ox[0]:ox[1], ...], source=i, tag=42)

        if out_shape is not None:
            ax = ax.reshape(*out_shape)

    # Worker only
    else:
        # send ownership range
        comm.Send(ox, dest=0, tag=11)
        # send local portion of eigenvectors as buffer
        comm.Send(lx, dest=0, tag=42)
        ax = None

    return ax

def gather_petsc_matrix(x, comm, out_shape=None):
    """Gather the petsc vector/matrix `x` to a single array on the master
    process, assuming that owernership is sliced along the first dimension.

    Parameters
    ----------
    x : petsc4py.PETSc Mat or Vec
        Distributed petsc array to gather.
    comm : mpi4py.MPI.COMM
        MPI communicator
    out_shape : tuple, optional
        If not None, reshape the output array to this.

    Returns
    -------
    gathered : np.array master, None on workers (rank > 0)
    """
    # get local numpy array
    # lx = x.getArray() # this function doesn't work (though we can use the stuff below)
    # lx = 
    ox = np.empty(2, dtype=int)
    ox[:] = x.getOwnershipRange()

    size = x.getSize()
    # print("size ", size)
    # print("ox ", ox)
    lx =  x.getValues(list(range(ox[0], ox[1])), list(range(0,27)))
    # print(lx)
    # lx = A.getValues(list(range
    # master only
    print(comm.Get_rank())

    if comm.Get_rank() == 0:

        # create total array
        ax = np.empty(x.getSize(), dtype=lx.dtype)
        # set master's portion
        ax[ox[0]:ox[1], ...] = lx

        # get ownership ranges and data from worker processes
        for i in range(1, comm.Get_size()):
            comm.Recv(ox, source=i, tag=11)

            # receive worker's part of ouput vector
            comm.Recv(ax[ox[0]:ox[1], ...], source=i, tag=42)

        if out_shape is not None:
            ax = ax.reshape(*out_shape)

    # Worker only
    else:
        # send ownership range
        comm.Send(ox, dest=0, tag=11)
        # send local portion of eigenvectors as buffer
        comm.Send(lx, dest=0, tag=42)
        ax = None

    return ax
