import meshio
msh = meshio.read("./Bunny.msh")
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "tetra":
        tetra_cells = cell.data

tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})

#for key in msh.cell_data_dict["gmsh:physical"].keys():
#    if key == "triangle":
#        triangle_data = msh.cell_data_dict["gmsh:physical"][key]
#    elif key == "tetra":
#        tetra_data = msh.cell_data_dict["gmsh:physical"][key]
#tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})
#triangle_mesh =meshio.Mesh(points=msh.points,
#                           cells=[("triangle", triangle_cells)],
#                           cell_data={"name_to_read":[triangle_data]})
meshio.write("Bunny_High.xdmf", tetra_mesh)

#meshio.write("mf.xdmf", triangle_mesh)
