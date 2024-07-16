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
/* BINDTOOL_HEADER_FILE(calibrate_lin_array.h) */
/* BINDTOOL_HEADER_FILE_HASH(6a918e8eccc9fa567fd46338485dafe6) */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <doa/calibrate_lin_array.h>
// pydoc.h is automatically generated in the build directory
#include <calibrate_lin_array_pydoc.h>

void bind_calibrate_lin_array(py::module &m) {

  using calibrate_lin_array = ::gr::doa::calibrate_lin_array;

  py::class_<calibrate_lin_array, gr::sync_block, gr::block, gr::basic_block,
             std::shared_ptr<calibrate_lin_array>>(m, "calibrate_lin_array",
                                                   D(calibrate_lin_array))

      .def(py::init(&calibrate_lin_array::make), py::arg("norm_spacing"),
           py::arg("num_ant_ele"), py::arg("pilot_angle"),
           D(calibrate_lin_array, make))

      ;
}
