cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(KUPETCTMS)

#-----------------------------------------------------------------------------
# Extension meta-information
# This repository is a Python-only 3D Slicer extension. Daily development does
# not require building Slicer from source: add slicer_modules/ as an Additional
# Module Path. CMake is used to package the extension for the Extension Manager.
set(EXTENSION_HOMEPAGE "https://github.com/IshitaSinghFaujdar/MS-extension")
set(EXTENSION_CATEGORY "Quantification")
set(EXTENSION_CONTRIBUTORS "Ishita Singh Faujdar (KIST), Sumin Cho (KIST)")
set(EXTENSION_DESCRIPTION "PET/CT metabolic-syndrome toolkit: TotalSegmentator and visceral-fat segmentation, KU post-processing, and SUV / radiomics quantification.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Scripted modules live in slicer_modules/ (flat *.py layout for Additional
# Module Paths). Do not put a CMakeLists.txt in that folder — Slicer will treat
# it as an extension and Extension Wizard will fail.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SlicerScriptedModules.cmake)

#-----------------------------------------------------------------------------
# Ship the shared pipeline package + CLI scripts next to the scripted modules
# so `import lib` and scripts/run_pipeline.py work after installation.
file(GLOB_RECURSE KUPETCTMS_LIB_SCRIPTS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.py"
)
file(GLOB_RECURSE KUPETCTMS_SCRIPT_SCRIPTS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py"
)

ctkMacroCompilePythonScript(
  TARGET_NAME kupetctms_python_lib
  SCRIPTS ${KUPETCTMS_LIB_SCRIPTS} ${KUPETCTMS_SCRIPT_SCRIPTS}
  DESTINATION_DIR ${CMAKE_BINARY_DIR}/${Slicer_QTSCRIPTEDMODULES_LIB_DIR}
  INSTALL_DIR ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}
  NO_INSTALL_SUBDIR
)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
