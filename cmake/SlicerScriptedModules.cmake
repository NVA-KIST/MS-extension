# Called after include(${Slicer_USE_FILE}).
# Resolves module sources relative to this file so it works whether CMake is
# run from the repository root or from slicer_modules/.

get_filename_component(_KUPET_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_KUPET_MODULE_SRC "${_KUPET_REPO_ROOT}/slicer_modules")
set(_KUPET_MODULE_DEST "${CMAKE_BINARY_DIR}/${Slicer_QTSCRIPTEDMODULES_LIB_DIR}")

set(KUPETCTMS_SCRIPTED_MODULES
  ModuleLauncher
  PETCTSegmentationModule
  VesselSegmenter
  SegmentDilator
  UreterPostProcess
  DistanceMeasurer
  PETHotspotNavigator
  ScribbleTool
  PETCTQuantAnalysis
  PETBiomarkerStudio
)

foreach(_module ${KUPETCTMS_SCRIPTED_MODULES})
  set(_scripts ${_module}.py)
  file(GLOB_RECURSE _lib_scripts
    RELATIVE "${_KUPET_MODULE_SRC}"
    "${_KUPET_MODULE_SRC}/${_module}Lib/*.py"
  )
  list(APPEND _scripts ${_lib_scripts})

  ctkMacroCompilePythonScript(
    TARGET_NAME ${_module}
    SOURCE_DIR ${_KUPET_MODULE_SRC}
    SCRIPTS ${_scripts}
    DESTINATION_DIR ${_KUPET_MODULE_DEST}
    INSTALL_DIR ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}
    NO_INSTALL_SUBDIR
  )
endforeach()

file(GLOB_RECURSE _bootstrap_scripts
  RELATIVE "${_KUPET_MODULE_SRC}"
  "${_KUPET_MODULE_SRC}/KUPETCTMSLib/*.py"
)
ctkMacroCompilePythonScript(
  TARGET_NAME KUPETCTMSLib
  SOURCE_DIR ${_KUPET_MODULE_SRC}
  SCRIPTS ${_bootstrap_scripts}
  DESTINATION_DIR ${_KUPET_MODULE_DEST}
  INSTALL_DIR ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}
  NO_INSTALL_SUBDIR
)
