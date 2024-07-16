/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually
 * edited  */
/* The following lines can be configured to regenerate this file during cmake */
/* If manual edits are made, the following tags should be modified accordingly.
 */
/* BINDTOOL_GEN_AUTOMATIC(0) */
/* BINDTOOL_USE_PYGCCXML(0) */
/* BINDTOOL_HEADER_FILE(MUSIC_lin_array.h) */
/* BINDTOOL_HEADER_FILE_HASH(0f653b7a2695d31f9f80c6dbfc8bd8be) */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <doa/MUSIC_lin_array.h>
// pydoc.h is automatically generated in the build directory
#include <MUSIC_lin_array_pydoc.h>

void bind_MUSIC_lin_array(py::module &m) {

  using MUSIC_lin_array = ::gr::doa::MUSIC_lin_array;

  py::class_<MUSIC_lin_array, gr::sync_block, gr::block, gr::basic_block,
             std::shared_ptr<MUSIC_lin_array>>(m, "MUSIC_lin_array",
                                               D(MUSIC_lin_array))

      .def(py::init(&MUSIC_lin_array::make), py::arg("norm_spacing"),
           py::arg("num_targets"), py::arg("num_ant_ele"),
           py::arg("pspectrum_len"), D(MUSIC_lin_array, make))

      ;
}