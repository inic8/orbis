# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import numpy as np


def fit_biquadratic_polynomial(noise_values, acc_values):
    noise = np.array(noise_values, dtype=np.float64)
    acc = np.array(acc_values, dtype=np.float64)

    x = np.concatenate([noise, -noise])
    y = np.concatenate([acc, acc])

    coeffs = np.polyfit(x, y, 4)
    coeffs[1] = 0.0
    coeffs[3] = 0.0
    return coeffs.tolist()


def compute_sensitivity(poly_coeffs):
    a2 = poly_coeffs[2]
    return 2.0 * abs(a2)