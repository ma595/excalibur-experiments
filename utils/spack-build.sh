
git clone https://github.com/spack/spack
. spack/share/spack/setup-env.sh
module load gcc/8
spack compiler find
spack install py-fenics-dolfinx ^petsc@3.13+mumps+hypre cflags="-O3" fflags="-O3" ^intel-mkl ^intel-mpi
