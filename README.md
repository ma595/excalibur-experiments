# excalibur-experiments

## Progress
- [ ] Test larger process count 
- [ ] Test larger meshes
- [ ] Create fenics mesh on three processes
  - [ ] Rank ordering
- [x] Create fenics mesh on two processes and send mesh to second (bempp) process
  - [x] Check solution against serial version
- [x] Create fenics mesh on one process and send mesh content to second (bempp) process
- [x] Create fenics mesh on two processes - gather() 

## Issues 
https://github.com/FEniCS/dolfinx/issues/1170
