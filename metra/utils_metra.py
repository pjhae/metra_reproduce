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


def generate_skill_cont(dim):
    vector = np.random.normal(0, 1, dim)
    norm = np.linalg.norm(vector)
    normalized_vector = vector / norm
    return normalized_vector