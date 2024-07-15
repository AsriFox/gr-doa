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
from autocorrelate import autocorrelate_test_input_gen


def music_test_input_gen(
    len_ss: int,
    overlap_size: int,
    num_ant_ele: int,
    norm_spacing: float,
    perturb: bool,
    doas: list[float],
    num_ss: int = 1500,
    snr: float = 1000.0,
):
    """
    Generate simulated data to be provided as input
    to MUSIC algorithm for DoA estimation.
    """
    num_targets = len(doas)

    nonoverlap_size = len_ss - overlap_size
    # length of the signal
    len_input = nonoverlap_size * num_ss + overlap_size

    # direction of arrival
    D = np.deg2rad(doas)
    # digital frequency
    denom = rand

    array_loc = norm_spacing * np.linspace(start=(num_ant_ele - 1) / 2, stop=-(num_ant_ele - 1) / 2, num=num_ant_ele)

    def amv(theta: float):
        return np.exp(-1j * 2 * np.pi * cos(theta) * array_loc)

    # TODO: WTF is prod(N)???
    prod_N = num_ant_ele
    denom = np.random.permutation(prod_N)
    w = np.transpose(np.pi / denom[0:num_targets])

    if perturb:
        # unknown antenna perturbations to be calibrated
        ant_gain = np.random.rand(prod_N, 1)
        ant_phase = np.exp(-1j * np.pi * np.rand(prod_N, 1))
        # reference element
        ant_gain[0] = 1
        ant_phase[0] = 1
        ant_pert_vec = ant_gain * ant_phase
        ant_pert_mat = np.diag(ant_pert_vec)

        # known pilot DoA
        pilot_doa = np.deg2rad(30)
        V_true_pilot = amv(pilot_doa)
        V_pilot = ant_pert_mat @ V_true_pilot

        # pilot's digital frequency
        w_pilot = np.pi / 3
        x_pilot = np.transpose(V_pilot * np.exp(1j * w_pilot * np.arange(len_input)))
        S_x_pilot, _ = autocorrelate_test_input_gen(len_ss, overlap_size, num_inputs, FB, num_ss, x_pilot)
        S_x_pilot = np.reshape(S_x_pilot, prod_N, len(S_x_pilot) / prod_N)

        # This approach is based on
        # V. C. Soon, L. Tong, Y. F. Huang and R. Liu,
        # "A Subspace Method for Estimating Sensor Gains and Phases,"
        # in IEEE Transactions on Signal Processing,
        # vol. 42, no. 4, pp. 973-976, Apr 1994.
        ant_pert_est = np.zeros(prod_N, 1)
        for kk in range(num_ss):
            _, eig_vec = np.linalg.eig(S_x_pilot[:, ((kk - 1) * prod_N) : (kk * prod_N)])
            # signal subspace
            U_S = eig_vec[:, -1]
            U_S_sq = U_S @ np.transpose(U_S)

            W = np.transpose(np.diag(V_true_pilot)) @ U_S_sq @ np.diag(V_true_pilot)
            W_eig_val, W_eig_vec = np.linalg.eig(W)
            I = np.argsort(np.diag(W_eig_val))
            W_eig_vec = W_eig_vec[:, I]
            ant_pert_est = ant_pert_est + W_eig_vec[:, -1]

        ant_pert_est = ant_pert_est / num_ss
        # print(After antenna calibration, the array coefficients are:)
        # print(ant_pert_vec / ant_pert_est)

    V_true_targets = np.zeros(prod_N, len(D))
    for k in range(len(D)):
        V_true_targets[:, k] = amv(D[k])

    if perturb:
        V_targets = ant_pert_mat * V_true_targets
        x = np.transpose(V_targets @ np.exp(1j * w * np.arange(len_input)))
        xx = x * np.diag(1.0 / ant_pert_est)

    else:
        xx = np.tranpose(V_true_targets @ np.exp(1j * w * np.arange(len_input)))

    # add noise
    SNR_lin = 10 ** (0.1 * snr)
    sig_En = np.sum(np.abs(xx) ** 2) / len_input
    N0 = sig_En / SNR_lin
    N_sigma = np.sqrt(N0 / 2)
    n1 = (np.random.randn(len_input, prod_N) + 1j * np.random.randn(len_input, prod_N)) @ np.diag(N_sigma)
    xxx1 = xx + n1
    S_x, _ = autocorrelate_test_input_gen(len_ss, overlap_size, num_inputs, FB, num_ss, xxx1)

    if perturb:
        n2 = (np.random.randn(len_input, prod_N) + 1j * np.random.randn(len_input, prod_N)) @ np.diag(N_sigma)
        xxx2 = x + n2
        S_x_uncalibrated, _ = autocorrelate_test_input_gen(len_ss, overlap_size, num_inputs, FB, num_ss, xxx2)
        return S_x, S_x_uncalibrated, ant_pert_vec

    else:
        return S_x, None, None
