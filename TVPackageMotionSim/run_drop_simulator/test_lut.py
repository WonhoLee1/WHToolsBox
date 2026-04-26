import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np

plotter = pv.Plotter(off_screen=True)
mesh = pv.Plane()
mesh.point_data["values"] = mesh.points[:, 0]  # Min: -0.5, Max: 0.5

lut = pv.LookupTable()
cmap_obj = plt.get_cmap("jet")
colors = cmap_obj(np.linspace(0, 1, 256))
for i in range(256):
    lut.SetTableValue(i, colors[i, 0], colors[i, 1], colors[i, 2], colors[i, 3])

plotter.add_mesh(mesh, scalars="values", cmap=lut)
plotter.screenshot("test_lut.png")
