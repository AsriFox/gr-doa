#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2016
# Srikanth Pagadarai <srikanth.pagadarai@gmail.com>
# Travis F. Collins <travisfcollins@gmail.com>
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

from gnuradio import gr, gr_unittest
from gnuradio import blocks
import numpy as np
import doa_swig as doa
from doa_testbench.autocorrelate import autocorrelate_test_input_gen

class qa_autocorrelate (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None

    def test_001_t (self):
		# length of each snapshot
		len_ss = 2048
		# overlap size of each snapshot
		overlap_size = 512
		# num of inputs
		num_inputs = 4
		# apply Forward-Backward Averaging?
		FB = False

        # Generate auto-correlation vector
        expected_S_x, data = autocorrelate_test_input_gen(len_ss, overlap_size, num_inputs, FB)
        expected_S_x = np.ndarray.flatten(expected_S_x)
		# num of snapshots
		n_ss = len(expected_S_x)/(num_inputs*num_inputs)

		##################################################
		# Blocks & Connections
		##################################################
		self.doa_autocorrelate_0 = doa.autocorrelate(num_inputs, len_ss, overlap_size, FB)
		self.vec_sink = blocks.vector_sink_c(num_inputs*num_inputs)
		# setup sources
		for p in range(num_inputs):
		    # Add vector source
		    object_name_vs = 'vec_source_'+str(p)
		    setattr(self, object_name_vs, blocks.vector_source_c(data[:, p].tolist(), False) )
		    # Connect
		    self.tb.connect((getattr(self,object_name_vs), 0), (self.doa_autocorrelate_0, p))

		self.tb.connect((self.doa_autocorrelate_0, 0), (self.vec_sink, 0))

		# set up fg
		self.tb.run ()
		observed_S_x = self.vec_sink.data()

		# check data
        if np.any(np.abs(expected_S_x - observed_S_x) > 1.0):
            self.fail("Expected S_x is not equal to observed S_x")

    def test_002_t (self):
		# length of each snapshot
		len_ss = 1024
		# overlap size of each snapshot
		overlap_size = 256
		# num of inputs
		num_inputs = 8
		# apply Forward-Backward Averaging?
		FB = True

        # Generate auto-correlation vector
        expected_S_x, data = autocorrelate_test_input_gen(len_ss, overlap_size, num_inputs, FB)
        expected_S_x = np.ndarray.flatten(expected_S_x)
		# num of snapshots
		n_ss = len(expected_S_x)/(num_inputs*num_inputs)

		##################################################
		# Blocks & Connections
		##################################################
		self.doa_autocorrelate_0 = doa.autocorrelate(num_inputs, len_ss, overlap_size, FB)
		self.vec_sink = blocks.vector_sink_c(num_inputs*num_inputs)
		# setup sources
		for p in range(num_inputs):
		    # Add vector source
		    object_name_vs = 'vec_source_'+str(p)
		    setattr(self, object_name_vs, blocks.vector_source_c(data[:, p].tolist(), False) )
		    # Connect
		    self.tb.connect((getattr(self,object_name_vs), 0), (self.doa_autocorrelate_0, p))

		self.tb.connect((self.doa_autocorrelate_0, 0), (self.vec_sink, 0))

		# set up fg
		self.tb.run ()
		observed_S_x = self.vec_sink.data()

		# check data
        if np.any(np.abs(expected_S_x - observed_S_x) > 1.0):
            self.fail("Expected S_x is not equal to observed S_x")

    def test_003_t (self):
		# length of each snapshot
		len_ss = 256
		# overlap size of each snapshot
		overlap_size = 32
		# num of inputs
		num_inputs = 4
		# apply Forward-Backward Averaging?
		FB = True

        # Generate auto-correlation vector
        expected_S_x, data = autocorrelate_test_input_gen(len_ss, overlap_size, num_inputs, FB)
        expected_S_x = np.ndarray.flatten(expected_S_x)
		# num of snapshots
		n_ss = len(expected_S_x)/(num_inputs*num_inputs)

		##################################################
		# Blocks & Connections
		##################################################
		self.doa_autocorrelate_0 = doa.autocorrelate(num_inputs, len_ss, overlap_size, FB)
		self.vec_sink = blocks.vector_sink_c(num_inputs*num_inputs)
		# setup sources
		for p in range(num_inputs):
		    # Add vector source
		    object_name_vs = 'vec_source_'+str(p)
		    setattr(self, object_name_vs, blocks.vector_source_c(data[:, p].tolist(), False) )
		    # Connect
		    self.tb.connect((getattr(self,object_name_vs), 0), (self.doa_autocorrelate_0, p))

		self.tb.connect((self.doa_autocorrelate_0, 0), (self.vec_sink, 0))

		# set up fg
		self.tb.run ()
		observed_S_x = self.vec_sink.data()

		# check data
        if np.any(np.abs(expected_S_x - observed_S_x) > 1.0):
            self.fail("Expected S_x is not equal to observed S_x")

if __name__ == '__main__':
    gr_unittest.run(qa_autocorrelate, "qa_autocorrelate.xml")
