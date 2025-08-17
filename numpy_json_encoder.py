#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom JSON encoder for NumPy types

This module provides a custom JSON encoder that can handle NumPy data types,
which is useful when saving registration results that include NumPy arrays,
floats and integers.
"""

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types.
    
    This encoder converts NumPy data types to their Python equivalents:
    - numpy.integer types → Python int
    - numpy.floating types → Python float
    - numpy.ndarray → Python list
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)