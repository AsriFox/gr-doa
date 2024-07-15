#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2016 
# Srikanth Pagadarai<srikanth.pagadarai@gmail.com>
# Travis Collins<travisfcollins@gmail.com>.	
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
import doa_swig as doa
import numpy
from doa_testbench_create.music_test_input_gen import music_test_input_gen


class qa_rootMUSIC (gr_unittest.TestCase):

	def setUp(self):
		self.tb = gr.top_block()
		self.samp_rate = 32000        

	def tearDown(self):
		self.tb = None

	def test_001_rootMUSIC_aoa23(self):
		# length of each snapshot
		len_ss = 256
		# overlap size of each snapshot
		overlap_size = 32
		# apply Forward-Backward Averaging?
		FB = True
		# normalized_spacing
		norm_spacing = 0.5
		# number of sources 
		num_srcs = 1
		# expected angle-of-arrival
		expected_aoa = 23.0
		# number of array elements
		num_arr_ele = 8
		# simulate perturbation?
		PERTURB = False

		# Generate auto-correlation vector
		rootmusic_linear_input, _, _ = music_test_input_gen(
			len_ss, overlap_size, num_arr_ele, norm_spacing, PERTURB, [expected_aoa]
		)
		rootmusic_linear_input = numpy.ndarray.flatten(rootmusic_linear_input)

		##################################################
		# Blocks
		##################################################
		self.doa_rootMUSIC_0 = doa.rootMUSIC_linear_array(norm_spacing, num_srcs, num_arr_ele)
		self.vec_sink = blocks.vector_sink_f(1)
		self.vec_source = blocks.vector_source_c(rootmusic_linear_input, False, num_arr_ele * num_arr_ele)

		##################################################
		# Connections
		##################################################
		self.tb.connect((self.vec_source, 0), (self.doa_rootMUSIC_0, 0))
		self.tb.connect((self.doa_rootMUSIC_0, 0), (self.vec_sink, 0))

		# set up fg
		self.tb.run()

		# get data from sink
		aoa_output_23 = self.vec_sink.data()

		# check
        if numpy.any(numpy.abs(numpy.array(aoa_output_23) - expected_aoa) > 2.0):
            self.fail("Measured AoA is not equal to " + str(expected_aoa))

	def test_002_rootMUSIC_aoa52(self):
		# length of each snapshot
		len_ss = 1024
		# overlap size of each snapshot
		overlap_size = 64
		# apply Forward-Backward Averaging?
		FB = True
		# normalized_spacing
		norm_spacing = 0.5
		# number of sources 
		num_srcs = 1
		# expected angle-of-arrival
		expected_aoa = 52.0
		# number of array elements
		num_arr_ele = 4
		# simulate perturbation?
		PERTURB = False

		# Generate auto-correlation vector
		rootmusic_linear_input, _, _ = music_test_input_gen(
			len_ss, overlap_size, num_arr_ele, norm_spacing, PERTURB, [expected_aoa]
		)
		rootmusic_linear_input = numpy.ndarray.flatten(rootmusic_linear_input)

		##################################################
		# Blocks
		##################################################
		self.doa_rootMUSIC_0 = doa.rootMUSIC_linear_array(norm_spacing, num_srcs, num_arr_ele)
		self.vec_sink = blocks.vector_sink_f(1)
		self.vec_source = blocks.vector_source_c(rootmusic_linear_input, False, num_arr_ele * num_arr_ele)

		##################################################
		# Connections
		##################################################
		self.tb.connect((self.vec_source, 0), (self.doa_rootMUSIC_0, 0))
		self.tb.connect((self.doa_rootMUSIC_0, 0), (self.vec_sink, 0))

		# set up fg
		self.tb.run()

		# get data from sink
		aoa_output_52 = self.vec_sink.data()

		# check
        if numpy.any(numpy.abs(numpy.array(aoa_output_52) - expected_aoa) > 2.0):
            self.fail("Measured AoA is not equal to " + str(expected_aoa))
	
if __name__ == '__main__':
	gr_unittest.run(qa_rootMUSIC, "qa_rootMUSIC.xml")
