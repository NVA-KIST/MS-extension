cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

project(KUPETCTMS)

#-----------------------------------------------------------------------------
# Extension meta-information
# Extension Wizard parses this file when you select the slicer_modules folder.
# A full CMake package build should still use the repository-root CMakeLists.txt.
set(EXTENSION_HOMEPAGE "https://github.com/IshitaSinghFaujdar/MS-extension")
set(EXTENSION_CATEGORY "Quantification")
set(EXTENSION_CONTRIBUTORS "Ishita Singh Faujdar (KIST), Sumin Cho (KIST)")
set(EXTENSION_DESCRIPTION "PET/CT metabolic-syndrome toolkit: TotalSegmentator and visceral-fat segmentation, KU post-processing, and SUV / radiomics quantification.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA")

# NEXT_MODULE

#-----------------------------------------------------------------------------
# Optional: allow configuring this directory if Slicer_DIR is already set.
get_filename_component(_KUPET_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

if(NOT DEFINED Slicer_DIR AND NOT DEFINED Slicer_SOURCE_DIR)
  return()
endif()

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

include(${_KUPET_REPO_ROOT}/cmake/SlicerScriptedModules.cmake)

file(GLOB_RECURSE KUPETCTMS_LIB_SCRIPTS
  RELATIVE "${_KUPET_REPO_ROOT}"
  "${_KUPET_REPO_ROOT}/lib/*.py"
)
file(GLOB_RECURSE KUPETCTMS_SCRIPT_SCRIPTS
  RELATIVE "${_KUPET_REPO_ROOT}"
  "${_KUPET_REPO_ROOT}/scripts/*.py"
)

ctkMacroCompilePythonScript(
  TARGET_NAME kupetctms_python_lib
  SOURCE_DIR ${_KUPET_REPO_ROOT}
  SCRIPTS ${KUPETCTMS_LIB_SCRIPTS} ${KUPETCTMS_SCRIPT_SCRIPTS}
  DESTINATION_DIR ${CMAKE_BINARY_DIR}/${Slicer_QTSCRIPTEDMODULES_LIB_DIR}
  INSTALL_DIR ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}
  NO_INSTALL_SUBDIR
)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
