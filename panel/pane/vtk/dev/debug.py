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

    rws.initializeSerializers()
    context = rws.Context()
    instance_to_serialize = coneActor
    state = rws.serializeInstance(None, coneActor, rws.getReferenceId(coneActor), context)

    with open('actor.json', 'w') as f:
        json.dump(rws.dump(state, context), f, indent=4, separators=(',', ': '), sort_keys=True)
