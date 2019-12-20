import json
import numpy as np
import pyvista as pv

# mesh points
vertices = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0],
                     [0, 1, 0],
                     [0.5, 0.5, -1]])

# mesh faces
faces = np.hstack([[4, 0, 1, 2, 3],  # square
                   [3, 0, 1, 4],     # triangle
                   [3, 1, 2, 4]])    # triangle

surf = pv.PolyData(vertices, faces)

# plot each face with a different color
plotter = pv.Plotter(notebook=True)
actor = plotter.add_mesh(surf, scalars=np.arange(3))
#plotter.show()

import panel.pane.vtk.dev.vtk_render_serializer as rws

import panel as pn
pn.extension('vtk')
pan = pn.panel(plotter.ren_win)
plotter.ren_win.OffScreenRenderingOn()
plotter.ren_win.Render()

context = rws.VTKSerializer()
state = context.serialize(None, actor)

res = context.todict(state)
with open('actor.json', 'w') as f:
    json.dump(res, f, indent=2, separators=(',', ': '), sort_keys=True)