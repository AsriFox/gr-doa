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
/* BINDTOOL_HEADER_FILE(antenna_correction.h) */
/* BINDTOOL_HEADER_FILE_HASH(8f3e9a991311f04c14a6a66836f4a0ea) */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <doa/antenna_correction.h>
// pydoc.h is automatically generated in the build directory
#include <antenna_correction_pydoc.h>

void bind_antenna_correction(py::module &m) {

  using antenna_correction = ::gr::doa::antenna_correction;

  py::class_<antenna_correction, gr::sync_block, gr::block, gr::basic_block,
             std::shared_ptr<antenna_correction>>(m, "antenna_correction",
                                                  D(antenna_correction))

      .def(py::init(&antenna_correction::make), py::arg("num_ant_ele"),
           py::arg("config_filename"), D(antenna_correction, make))

      ;
}