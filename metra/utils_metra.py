import math
import torch
import os
import sys

import imageio
import numpy as np


def generate_skill(dim, eval_idx = -1):

    vector = np.full(dim, -1/(dim-1))

    if eval_idx != -1:
        vector[eval_idx] = 1
    else:
        idx = np.random.randint(dim)
        vector[idx] = 1
    
    return vector
