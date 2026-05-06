# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import numpy as np


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(value) for value in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj