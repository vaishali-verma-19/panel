import sys

import vtk
from vtk.util import numpy_support
from vtk.vtkFiltersGeometry import vtkCompositeDataGeometryFilter
import copy

py3 = sys.version_info >= (3,0)

arrayTypesMapping = [
  ' ', # VTK_VOID            0
  ' ', # VTK_BIT             1
  'b', # VTK_CHAR            2
  'B', # VTK_UNSIGNED_CHAR   3
  'h', # VTK_SHORT           4
  'H', # VTK_UNSIGNED_SHORT  5
  'i', # VTK_INT             6
  'I', # VTK_UNSIGNED_INT    7
  'l', # VTK_LONG            8
  'L', # VTK_UNSIGNED_LONG   9
  'f', # VTK_FLOAT          10
  'd', # VTK_DOUBLE         11
  'L', # VTK_ID_TYPE        12
]

javascriptMapping = {
    'b': 'Int8Array',
    'B': 'Uint8Array',
    'h': 'Int16Array',
    'H': 'Int16Array',
    'i': 'Int32Array',
    'I': 'Uint32Array',
    'l': 'Int32Array',
    'L': 'Uint32Array',
    'f': 'Float32Array',
    'd': 'Float64Array'
}

import json

class Context:

  def  __init__(self):
    self._register = {}

  def register(self, obj_id, obj):
    self._register.update({obj_id: obj})
  
  def get(self, obj_id):
    return self._register.get(obj_id)

  def todict(self, state):
    res = {k:v for k,v in state.items() if k != 'calls'}
    if 'calls' in state:
      for name, ref_id in state['calls']:
        ref = self.get(ref_id)
        if isinstance(ref, dict) and 'calls' in ref:
          res[name] = self.todict(ref)
        else:
          res[name] = ref
    return res


class VTKSerializer(Context):

  _serializers = {}

  def __init__(self):
    super().__init__()

  @classmethod
  def registerInstanceSerializer(cls, name, method):
    cls._serializers[name] = method

  def serialize(self, parent, instance):
    instanceType = instance.GetClassName()
    serializer = self._serializers.get(instanceType, None)

    if serializer:
      return serializer(parent, instance, self)
    else:
      raise TypeError('!!!No serializer for %s with id %s' % (instanceType))

    return None


def getJSArrayType(dataArray):
    return javascriptMapping[arrayTypesMapping[dataArray.GetDataType()]]


def _dump_data_array(array):

    if array.GetDataType() == 12:
        # IdType need to be converted to Uint32
        arraySize = array.GetNumberOfTuples() * array.GetNumberOfComponents()
        newArray = vtk.vtkTypeUInt32Array()
        newArray.SetNumberOfTuples(arraySize)
        for i in range(arraySize):
            newArray.SetValue(i, -1 if array.GetValue(i) < 0 else array.GetValue(i))
    else:
        newArray = array
    return {
      'name': "temp",
      'dataType': javascriptMapping[arrayTypesMapping[newArray.GetDataType()]],
      'numberOfComponents': array.GetNumberOfComponents(),
      'size': newArray.GetNumberOfComponents() * newArray.GetNumberOfTuples(),
      'values': numpy_support.vtk_to_numpy(newArray).ravel().tolist(),
      'vtkClass': 'vtkDataArray',
    }

def _dump_tcoords(dataset, root):
  tcoords = dataset.GetPointData().GetTCoords()
  if tcoords:
    dumpedArray = _dump_data_array(tcoords)
    root['pointData']['activeTCoords'] = len(root['pointData']['arrays'])
    root['pointData']['arrays'].append({ 'data': dumpedArray })


def _dump_normals(dataset, root):
  normals = dataset.GetPointData().GetNormals()
  if normals:
      dumpedArray = _dump_data_array(normals)
      root['pointData']['activeNormals'] = len(root['pointData']['arrays'])
      root['pointData']['arrays'].append({ 'data': dumpedArray })

  
def _dump_all_arrays(dataset):
  root = {}
  for data_loc in ['pointData', 'cellData', 'fieldData']:
    root[data_loc] = {
      'vtkClass': 'vtkDataSetAttributes',
      "activeGlobalIds":-1,
      "activeNormals":-1,
      "activePedigreeIds":-1,
      "activeScalars":-1,
      "activeTCoords":-1,
      "activeTensors":-1,
      "activeVectors":-1,
      "arrays": []
    }

  # Point data
  pd = dataset.GetPointData()
  pd_size = pd.GetNumberOfArrays()
  for i in range(pd_size):
    array = pd.GetArray(i)
    if array:
      dumpedArray = _dump_data_array(array)
      root['pointData']['activeScalars'] = 0
      root['pointData']['arrays'].append({'data': dumpedArray})
  scalars = pd.GetScalars()
  if scalars:
    dumpedArray = _dump_data_array(scalars)
    root['pointData']['activeScalars'] = len(root['pointData']['arrays'])
    root['pointData']['arrays'].append({'data': dumpedArray})
  
  # Cell data
  cd = dataset.GetCellData()
  cd_size = pd.GetNumberOfArrays()
  for i in range(cd_size):
    array = cd.GetArray(i)
    if array:
      dumpedArray = _dump_data_array(array)
      root['cellData']['activeScalars'] = 0
      root['cellData']['arrays'].append({'data': dumpedArray})
  scalars = cd.GetScalars()
  if scalars:
    dumpedArray = _dump_data_array(scalars)
    root['cellData']['activeScalars'] = len(root['cellData']['arrays'])
    root['cellData']['arrays'].append({'data': dumpedArray})

  _dump_tcoords(dataset, root)

  _dump_normals(dataset, root)

  return root


# -----------------------------------------------------------------------------

def getReferenceId(ref):
  return ref.__this__ if hasattr(ref, '__this__') else str(id(ref))


# -----------------------------------------------------------------------------

def getRangeInfo(array, component):
  r = array.GetRange(component)
  compRange = {}
  compRange['min'] = r[0]
  compRange['max'] = r[1]
  compRange['component'] = array.GetComponentName(component)
  return compRange


# -----------------------------------------------------------------------------
# Concrete instance serializers
# -----------------------------------------------------------------------------

def genericActorSerializer(parent, actor, context):
  # This kind of actor has two "children" of interest, a property and a mapper
  actorId = getReferenceId(actor)
  actorVisibility = actor.GetVisibility()
  mapperInstance = None
  propertyInstance = None
  calls = []

  if actorVisibility:
    mapper = None
    mapper = actor.GetMapper()
    if mapper:
      mapperId = getReferenceId(mapper)
      mapperInstance = context.serialize(actor, mapper)
      if mapperInstance:
        context.register(mapperId, mapperInstance)
        calls.append(['mapper', mapperId])

    prop = actor.GetProperty()
    if prop:
      propId = getReferenceId(prop)
      propertyInstance = context.serialize(actor, prop)
      if propertyInstance:
        context.register(propId, propertyInstance)
        calls.append(['property', propId])

  if actorVisibility == 0 or (mapperInstance and propertyInstance):
    return {
      'parent': getReferenceId(parent),
      'id': actorId,
      'type': actor.GetClassName(),
      'vtkClass': 'vtkActor',
      # vtkProp
      'visibility': actorVisibility,
      'pickable': actor.GetPickable(),
      'dragable': actor.GetDragable(),
      'useBounds': actor.GetUseBounds(),
      # vtkProp3D
      'origin': actor.GetOrigin(),
      'position': actor.GetPosition(),
      'scale': actor.GetScale(),
      # vtkActor
      'forceOpaque': actor.GetForceOpaque(),
      'forceTranslucent': actor.GetForceTranslucent(),
      'calls': calls,
    }

  return None

# -----------------------------------------------------------------------------

def genericMapperSerializer(parent, mapper, context):
  # This kind of mapper requires us to get 2 items: input data and lookup table
  mapperId = getReferenceId(mapper)
  dataObject = None
  dataObjectInstance = None
  lookupTableInstance = None
  calls = []

  dataObject = mapper.GetInputDataObject(0, 0)

  if dataObject:
    dataObjectId = getReferenceId(dataObject)
    dataObjectInstance = context.serialize(mapper, dataObject)
    if dataObjectInstance:
      context.register(dataObjectId, dataObjectInstance)
      calls.append(['inputData', dataObjectId ])

  lookupTable = None

  if hasattr(mapper, 'GetLookupTable'):
    lookupTable = mapper.GetLookupTable()

  if lookupTable:
    lookupTableId = getReferenceId(lookupTable)
    lookupTableInstance = context.serialize(mapper, lookupTable)
    if lookupTableInstance:
      context.register(lookupTableId, lookupTableInstance)
      calls.append(['lookupTable', lookupTableId])

  if dataObjectInstance and lookupTableInstance:
    colorArrayName = mapper.GetArrayName() if mapper.GetArrayAccessMode() == 1 else mapper.GetArrayId()
    return {
      'parent': getReferenceId(parent),
      'id': mapperId,
      'type': mapper.GetClassName(),
      'vtkClass': 'vtkMapper',
      'scalarRange': mapper.GetScalarRange(),
      'useLookupTableScalarRange': True if mapper.GetUseLookupTableScalarRange() else False,
      'scalarVisibility': mapper.GetScalarVisibility(),
      'colorByArrayName': colorArrayName,
      'colorMode': mapper.GetColorMode(),
      'scalarMode': mapper.GetScalarMode(),
      'interpolateScalarsBeforeMapping': True if mapper.GetInterpolateScalarsBeforeMapping() else False,
      'calls': calls,
    }

  return None

# -----------------------------------------------------------------------------

def lookupTableSerializer(parent, lookupTable, context):
  # No children in this case, so no additions to bindings and return empty list
  # But we do need to add instance
  lookupTableId = getReferenceId(lookupTable)
  lookupTableRange = lookupTable.GetRange()

  lookupTableHueRange = lookupTable.GetHueRange() if hasattr(lookupTable, 'GetHueRange') else [0.5, 0]
  lutSatRange = lookupTable.GetSaturationRange()

  return {
    'parent': getReferenceId(parent),
    'id': lookupTableId,
    'type': lookupTable.GetClassName(),
    'vtkClass': 'vtkLookupTable',
    'numberOfColors': lookupTable.GetNumberOfColors(),
    'valueRange': lookupTableRange,
    'hueRange': lookupTableHueRange,
    # 'alphaRange': lutAlphaRange,  # Causes weird rendering artifacts on client
    'saturationRange': lutSatRange,
    'nanColor': lookupTable.GetNanColor(),
    'belowRangeColor': lookupTable.GetBelowRangeColor(),
    'aboveRangeColor': lookupTable.GetAboveRangeColor(),
    'useAboveRangeColor': True if lookupTable.GetUseAboveRangeColor() else False,
    'useBelowRangeColor': True if lookupTable.GetUseBelowRangeColor() else False,
    'alpha': lookupTable.GetAlpha(),
    'vectorSize': lookupTable.GetVectorSize(),
    'vectorComponent': lookupTable.GetVectorComponent(),
    'vectorMode': lookupTable.GetVectorMode(),
    'indexedLookup': lookupTable.GetIndexedLookup(),
  }

# -----------------------------------------------------------------------------

def propertySerializer(parent, propObj, context):
  propObjId = getReferenceId(propObj)
  representation = propObj.GetRepresentation() if hasattr(propObj, 'GetRepresentation') else 2
  colorToUse = propObj.GetDiffuseColor() if hasattr(propObj, 'GetDiffuseColor') else [1, 1, 1]
  if representation == 1 and hasattr(propObj, 'GetColor'):
    colorToUse = propObj.GetColor()

  return {
    'parent': getReferenceId(parent),
    'id': propObjId,
    'type': propObj.GetClassName(),
    'vtkClass': 'vtkProperty',
    'representation': representation,
    'diffuseColor': colorToUse,
    'color': propObj.GetColor(),
    'ambientColor': propObj.GetAmbientColor(),
    'specularColor': propObj.GetSpecularColor(),
    'edgeColor': propObj.GetEdgeColor(),
    'ambient': propObj.GetAmbient(),
    'diffuse': propObj.GetDiffuse(),
    'specular': propObj.GetSpecular(),
    'specularPower': propObj.GetSpecularPower(),
    'opacity': propObj.GetOpacity(),
    'interpolation': propObj.GetInterpolation(),
    'edgeVisibility': True if propObj.GetEdgeVisibility() else False,
    'backfaceCulling': True if propObj.GetBackfaceCulling() else False,
    'frontfaceCulling': True if propObj.GetFrontfaceCulling() else False,
    'pointSize': propObj.GetPointSize(),
    'lineWidth': propObj.GetLineWidth(),
    'lighting': propObj.GetLighting(),
  }

# -----------------------------------------------------------------------------

def polydataSerializer(parent, dataset, context):
  datasetId = getReferenceId(dataset)
  datasetType = dataset.GetClassName()

  if dataset and dataset.GetPoints():
    root = {
      'parent': getReferenceId(parent),
      'id': datasetId,
      'type': datasetType,
      'vtkClass': 'vtkPolyData'
    }


    #Geometry
    # Points
    points = _dump_data_array(dataset.GetPoints().GetData())
    points['vtkClass'] = 'vtkPoints'
    root['points'] = points

    # Cells
    for cell_type in ['verts', 'lines', 'polys', 'strips']:
        cell = getattr(dataset, 'Get' + cell_type.capitalize())()
        if cell and cell.GetData().GetNumberOfTuples() > 0:
            root[cell_type] = _dump_data_array(cell.GetData())
            root[cell_type]['vtkClass'] = 'vtkCellArray'

    # Data
    root.update(_dump_all_arrays(dataset))

    return [root]

  raise ValueError('This dataset has no points!')

# -----------------------------------------------------------------------------

def mergeToPolydataSerializer(parent, dataObject, context):
  dataset = None

  if dataObject.IsA('vtkCompositeDataSet'):
    gf = vtkCompositeDataGeometryFilter()
    gf.SetInputData(dataObject)
    gf.Update()
    tempDS = gf.GetOutput()
    dataset = tempDS
  else:
    raise RuntimeError('dataset = mapper.GetInput()')

  return polydataSerializer(parent, dataset, context)

# -----------------------------------------------------------------------------

def colorTransferFunctionSerializer(parent, instance, context):
  objId = getReferenceId(instance)
  nodes = []

  for i in range(instance.GetSize()):
    # x, r, g, b, midpoint, sharpness
    node = [0, 0, 0, 0, 0, 0]
    instance.GetNodeValue(i, node)
    nodes.append(node)

  return {
    'parent': getReferenceId(parent),
    'id': objId,
    'type': instance.GetClassName(),
    'vtkClass': 'vtkColorTransferFunction',
    'clamping': True if instance.GetClamping() else False,
    'colorSpace': instance.GetColorSpace(),
    'hSVWrap': True if instance.GetHSVWrap() else False,
    # 'nanColor': instance.GetNanColor(),                  # Breaks client
    # 'belowRangeColor': instance.GetBelowRangeColor(),    # Breaks client
    # 'aboveRangeColor': instance.GetAboveRangeColor(),    # Breaks client
    # 'useAboveRangeColor': True if instance.GetUseAboveRangeColor() else False,
    # 'useBelowRangeColor': True if instance.GetUseBelowRangeColor() else False,
    'allowDuplicateScalars': True if instance.GetAllowDuplicateScalars() else False,
    'alpha': instance.GetAlpha(),
    'vectorComponent': instance.GetVectorComponent(),
    'vectorSize': instance.GetVectorSize(),
    'vectorMode': instance.GetVectorMode(),
    'indexedLookup': instance.GetIndexedLookup(),
    'nodes': nodes,
  }

# -----------------------------------------------------------------------------

def rendererSerializer(parent, instance, context):
  objId = getReferenceId(instance)
  raise NotImplementedError('TODO')
  calls = []

  # Camera
  camera = instance.GetActiveCamera()
  cameraId = getReferenceId(camera)
  cameraInstance = context.serialize(instance, camera)
  if cameraInstance:
      context.register(cameraInstance)
      calls.append(['activeCamera', cameraId])

  # View prop as representation containers
  viewPropCollection = instance.GetViewProps()
  for rpIdx in range(viewPropCollection.GetNumberOfItems()):
    viewProp = viewPropCollection.GetItemAsObject(rpIdx)
    viewPropId = getReferenceId(viewProp)
    viewPropInstance = context.serialize(instance, viewProp)
    if viewPropInstance:
      context.register(viewPropId, viewPropInstance)
      calls.append(['addViewProp', viewPropId])

  # Lights
  lightCollection = instance.GetLights()
  for lightIdx in range(lightCollection.GetNumberOfItems()):
    light = lightCollection.GetItemAsObject(lightIdx)
    lightId = getReferenceId(light)
    lightInstance = context.serialize(instance, light)
    if lightInstance:
        context.register(lightId, lightInstance)
        calls.append(['addLight', lightId])


  return {
    'parent': getReferenceId(parent),
    'id': objId,
    'type': instance.GetClassName(),
    'vtkClass': 'vtkRenderer',

    'background': instance.GetBackground(),
    'background2': instance.GetBackground2(),
    'viewport': instance.GetViewport(),
    ### These commented properties do not yet have real setters in vtk.js
    # 'gradientBackground': instance.GetGradientBackground(),
    # 'aspect': instance.GetAspect(),
    # 'pixelAspect': instance.GetPixelAspect(),
    # 'ambient': instance.GetAmbient(),
    'twoSidedLighting': instance.GetTwoSidedLighting(),
    'lightFollowCamera': instance.GetLightFollowCamera(),
    'layer': instance.GetLayer(),
    'preserveColorBuffer': instance.GetPreserveColorBuffer(),
    'preserveDepthBuffer': instance.GetPreserveDepthBuffer(),
    'nearClippingPlaneTolerance': instance.GetNearClippingPlaneTolerance(),
    'clippingRangeExpansion': instance.GetClippingRangeExpansion(),
    'useShadows': instance.GetUseShadows(),
    'useDepthPeeling': instance.GetUseDepthPeeling(),
    'occlusionRatio': instance.GetOcclusionRatio(),
    'maximumNumberOfPeels': instance.GetMaximumNumberOfPeels(),
    'calls': calls,
  }

# -----------------------------------------------------------------------------

def cameraSerializer(parent, instance, context):
  objId = getReferenceId(instance)
  return {
    'parent': getReferenceId(parent),
    'id': objId,
    'type': instance.GetClassName(),
    'vtkClass': 'vtkCamera',
    'focalPoint': instance.GetFocalPoint(),
    'position': instance.GetPosition(),
    'viewUp': instance.GetViewUp(),
  }

# -----------------------------------------------------------------------------

def lightTypeToString(value):
  """
  #define VTK_LIGHT_TYPE_HEADLIGHT    1
  #define VTK_LIGHT_TYPE_CAMERA_LIGHT 2
  #define VTK_LIGHT_TYPE_SCENE_LIGHT  3
  'HeadLight';
  'SceneLight';
  'CameraLight'
  """
  if value == 1:
    return 'HeadLight'
  elif value == 2:
    return 'CameraLight'

  return 'SceneLight'

def lightSerializer(parent, instance, context):
  return {
    'parent': getReferenceId(parent),
    'id': getReferenceId(instance),
    'type': instance.GetClassName(),
    'vtkClass': 'vtkLight',
    # 'specularColor': instance.GetSpecularColor(),
    # 'ambientColor': instance.GetAmbientColor(),
    'switch': instance.GetSwitch(),
    'intensity': instance.GetIntensity(),
    'color': instance.GetDiffuseColor(),
    'position': instance.GetPosition(),
    'focalPoint': instance.GetFocalPoint(),
    'positional': instance.GetPositional(),
    'exponent': instance.GetExponent(),
    'coneAngle': instance.GetConeAngle(),
    'attenuationValues': instance.GetAttenuationValues(),
    'lightType': lightTypeToString(instance.GetLightType()),
    'shadowAttenuation': instance.GetShadowAttenuation()
  }

# -----------------------------------------------------------------------------

def renderWindowSerializer(parent, instance, context):
  objId = getReferenceId(instance)
  calls = []
  rendererIds = []

  rendererCollection = instance.GetRenderers()
  for rIdx in range(rendererCollection.GetNumberOfItems()):
    # Grab the next vtkRenderer
    renderer = rendererCollection.GetItemAsObject(rIdx)
    rendererId = getReferenceId(renderer)
    rendererInstance = context.serialize(instance, renderer)
    if rendererInstance:
      context.register(rendererId, rendererInstance)
      calls.append(['addRenderer', rendererId])

  calls = context.buildDependencyCallList(objId, rendererIds, 'addRenderer', 'removeRenderer')

  return {
    'parent': getReferenceId(parent),
    'id': objId,
    'type': instance.GetClassName(),
    'vtkClass': 'vtkRenderWindow',
    'numberOfLayers': instance.GetNumberOfLayers(),
    'calls': calls,
  }


# Actors/viewProps
VTKSerializer.registerInstanceSerializer('vtkOpenGLActor', genericActorSerializer)
VTKSerializer.registerInstanceSerializer('vtkPVLODActor', genericActorSerializer)

# Mappers
VTKSerializer.registerInstanceSerializer('vtkOpenGLPolyDataMapper', genericMapperSerializer)
VTKSerializer.registerInstanceSerializer('vtkDataSetMapper', genericMapperSerializer)
VTKSerializer.registerInstanceSerializer('vtkCompositePolyDataMapper2', genericMapperSerializer)

# LookupTables/TransferFunctions
VTKSerializer.registerInstanceSerializer('vtkLookupTable', lookupTableSerializer)
VTKSerializer.registerInstanceSerializer('vtkPVDiscretizableColorTransferFunction', colorTransferFunctionSerializer)

# Property
VTKSerializer.registerInstanceSerializer('vtkOpenGLProperty', propertySerializer)

# Datasets
VTKSerializer.registerInstanceSerializer('vtkPolyData', polydataSerializer)
VTKSerializer.registerInstanceSerializer('vtkMultiBlockDataSet', mergeToPolydataSerializer)

# RenderWindows
VTKSerializer.registerInstanceSerializer('vtkCocoaRenderWindow', renderWindowSerializer)
VTKSerializer.registerInstanceSerializer('vtkXOpenGLRenderWindow', renderWindowSerializer)
VTKSerializer.registerInstanceSerializer('vtkWin32OpenGLRenderWindow', renderWindowSerializer)
VTKSerializer.registerInstanceSerializer('vtkEGLRenderWindow', renderWindowSerializer)
VTKSerializer.registerInstanceSerializer('vtkOpenVRRenderWindow', renderWindowSerializer)
VTKSerializer.registerInstanceSerializer('vtkGenericOpenGLRenderWindow', renderWindowSerializer)
VTKSerializer.registerInstanceSerializer('vtkOSOpenGLRenderWindow', renderWindowSerializer)
VTKSerializer.registerInstanceSerializer('vtkOpenGLRenderWindow', renderWindowSerializer)
VTKSerializer.registerInstanceSerializer('vtkIOSRenderWindow', renderWindowSerializer)
VTKSerializer.registerInstanceSerializer('vtkExternalOpenGLRenderWindow', renderWindowSerializer)

# Renderers
VTKSerializer.registerInstanceSerializer('vtkOpenGLRenderer', rendererSerializer)

# Cameras
VTKSerializer.registerInstanceSerializer('vtkOpenGLCamera', cameraSerializer)

# Lights
VTKSerializer.registerInstanceSerializer('vtkPVLight', lightSerializer)
VTKSerializer.registerInstanceSerializer('vtkOpenGLLight', lightSerializer)
