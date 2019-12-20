import vtk
import json
import panel.pane.vtk.dev.vtk_render_serializer as rws

if __name__ == "__main__":    
    cone = vtk.vtkConeSource()

    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())

    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)

    ren = vtk.vtkRenderer()
    ren.AddActor(coneActor)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    ren.ResetCamera()
    renWin.OffScreenRenderingOn()
    renWin.Render()

    context = rws.VTKSerializer()
    state = context.serialize(None, coneActor)

    res = context.todict(state)
    with open('actor.json', 'w') as f:
        json.dump(res, f, indent=2, separators=(',', ': '), sort_keys=True)
