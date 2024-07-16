# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT doa_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT doa_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND doa_python_files
    autocorrelate_python.cc
    calibrate_lin_array_python.cc
    find_local_max_python.cc
    MUSIC_lin_array_python.cc
    rootMUSIC_linear_array_python.cc python_bindings.cc)

GR_PYBIND_MAKE_OOT(doa
   ../..
   gr::doa
   "${doa_python_files}")

# copy bindings extension for use in QA test module
add_custom_command(TARGET doa_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:doa_python>
    ${PROJECT_BINARY_DIR}/test_modules/gnuradio/doa/
)

install(TARGETS doa_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/doa COMPONENT pythonapi)
