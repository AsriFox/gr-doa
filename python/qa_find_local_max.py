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
import numpy as np
from numpy import array, pi, sin, cos
from scipy.signal import find_peaks
import os


def findpeaks(num_max_vals: int, y: np.ndarray):
	data = np.abs(y)
	all_pks_idx, all_pks_props = find_peaks(data, height=0.0)
	all_pks_idx = np.argsort(-all_pks_props['peak_height'])
	pks_idx = all_pks_idx[0:num_max_vals]
	pks = data[pks_idx]
	return array(pks), array(pks_idx)


class qa_find_local_max (gr_unittest.TestCase):

	def setUp (self):
		self.tb = gr.top_block ()	

	def tearDown (self):
		self.tb = None

	def test_001_t (self):
		self.vector_len = 2**11
		self.num_max_vals = 3

		# generate vector
		t = 2.0 * pi * np.linspace(0, 1, self.vector_len)
		y = sin(3.14 * t) + 0.5 * cos(6.09 * t) + 0.1 * sin(10.11 * t + 1 / 6) + 0.1 * sin(15.3 * t + 1 / 3)
		data = y.flatten().tolist()
		expected_pks, expected_pks_locs = findpeaks(self.num_max_vals, y)
		expected_t_pks = t[expected_pks_locs]

		##################################################
		# Blocks
		##################################################
		self.blocks_vector_source_x_0 = blocks.vector_source_f(data, False, self.vector_len, [])
		self.doa_find_local_max_0 = doa.find_local_max(self.num_max_vals, self.vector_len, 0.0, 2*numpy.pi)
		self.blocks_vector_sink_x_0 = blocks.vector_sink_f(self.num_max_vals)
		self.blocks_vector_sink_x_1 = blocks.vector_sink_f(self.num_max_vals)       

		##################################################
		# Connections
		##################################################
		self.tb.connect((self.blocks_vector_source_x_0, 0), (self.doa_find_local_max_0, 0))    
		self.tb.connect((self.doa_find_local_max_0, 0), (self.blocks_vector_sink_x_0, 0))    
		self.tb.connect((self.doa_find_local_max_0, 1), (self.blocks_vector_sink_x_1, 0))  

		# set up fg
		self.tb.run ()

		# get data from sink
		measured_pks = array(self.blocks_vector_sink_x_0.data())
		measured_pks_locs = array([int(p) for p in self.blocks_vector_sink_x_1.data()])
		measured_t_pks = t[measured_pks_locs]

		# check data
		if np.any(np.abs(expected_pks - measured_pks) > 1e-5):
			self.fail("Expected peak heights are not equal to measured peak heights")
		if np.any(np.abs(expected_t_pks - measured_t_pks) > 1e-5):
			self.fail("Expected peak locations are not equal to measured peak locations")

	def test_002_t (self):
		self.vector_len = 2**12
		self.num_max_vals = 5

		# generate vector
		t = 2.0 * pi * np.linspace(0, 1, self.vector_len)
		y = sin(0.25 * 3.14 * t) + 5 * cos(6.09 * t) + 0.6 * sin(1.11 * t + 1 / 6) + 2 * sin(5.3 * t + 1 / 3)
		data = y.flatten().tolist()
		expected_pks, expected_pks_locs = findpeaks(self.num_max_vals, y)
		expected_t_pks = t[expected_pks_locs]

		##################################################
		# Blocks
		##################################################
		self.blocks_vector_source_x_0 = blocks.vector_source_f(data, False, self.vector_len, [])
		self.doa_find_local_max_0 = doa.find_local_max(self.num_max_vals, self.vector_len, 0.0, 2*numpy.pi)
		self.blocks_vector_sink_x_0 = blocks.vector_sink_f(self.num_max_vals)
		self.blocks_vector_sink_x_1 = blocks.vector_sink_f(self.num_max_vals)       

		##################################################
		# Connections
		##################################################
		self.tb.connect((self.blocks_vector_source_x_0, 0), (self.doa_find_local_max_0, 0))    
		self.tb.connect((self.doa_find_local_max_0, 0), (self.blocks_vector_sink_x_0, 0))    
		self.tb.connect((self.doa_find_local_max_0, 1), (self.blocks_vector_sink_x_1, 0))  

		# set up fg
		self.tb.run ()

		# get data from sink
		measured_pks = array(self.blocks_vector_sink_x_0.data())
		measured_pks_locs = array([int(p) for p in self.blocks_vector_sink_x_1.data()])
		measured_t_pks = t[measured_pks_locs]

		# check data
		if np.any(np.abs(expected_pks - measured_pks) > 1e-5):
			self.fail("Expected peak heights are not equal to measured peak heights")
		if np.any(np.abs(expected_t_pks - measured_t_pks) > 1e-5):
			self.fail("Expected peak locations are not equal to measured peak locations")


if __name__ == '__main__':
	gr_unittest.run(qa_find_local_max, "qa_find_local_max.xml")
