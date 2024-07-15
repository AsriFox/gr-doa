#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2016
# Srikanth Pagadarai <srikanth.pagadarai@gmail.com>
# Travis Collins <travisfcollins@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np


def autocorrelate_test_input_gen(
    len_ss: int,
    overlap_size: int,
    num_inputs: int,
    FB: bool,
    num_ss: int = 1500,
    in_sig_stream: np.ndarray = None,
):
    """
    Generate the autocorrelation matrix needed for subspace-based DoA estimation.
    If no input signal is provided, a standard normal random complex matrix is
    used as input data.
    """
    nonoverlap_size = len_ss - overlap_size
    # length of the signal
    len_input = nonoverlap_size * num_ss + overlap_size

    # autocorrelate input
    xx = (
        in_sig_stream
        if in_sig_stream is not None
        else (np.random.normal(size=(len_input, num_inputs)) + 1j * np.random.normal(size=(len_input, num_inputs)))
    )

    # autocorrelate output
    S_xx = np.zeros(shape=(num_inputs, num_ss * num_inputs), dtype=np.complex64)

    J = np.fliplr(np.eye(num_inputs, num_inputs))
    for ii in range(num_ss):
        x = xx[(ii * nonoverlap_size) : (ii * nonoverlap_size + len_ss), :]

        # sample spectral matrix
        S_x = (np.transpose(x) @ np.conj(x)) / len_ss

        if FB:
            S_x = 0.5 * S_x + 0.5 * (J @ np.conj(S_x) @ J) / len_ss

        S_xx[:, (ii * num_inputs) : (ii * num_inputs + num_inputs)] = S_x[:]

    return S_xx, xx
