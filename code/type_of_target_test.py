# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd

from sklearn.utils.multiclass import type_of_target

a = [1.0, 2.3, 4.5, 5.2, 6.3, 7.5]
type_of_target(a)

b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
type_of_target(b)

c = [1, 2, 1, 2, 2, 1, 1, 1, 2]
type_of_target(c)


