# -*- coding: utf-8 -*-
"""
Created on Thu Jan  11 13:37:51 2024

@author: aad
"""

import numpy as np
from PIL import Image
from holography_functions import precompute_ft_terms, precompute_propagator_constants, FT2Dc,  find_best_focus, reconstruct_and_plot

# PARAMETERS
N = 200
lambda_val = 530e-9
area = 0.0015

# READING HOLOGRAM OBJECT
hologram = np.array(Image.open('U01.png')).astype(float)
Nx, Ny = hologram.shape
f1 = precompute_ft_terms(Nx, Ny)
alpha, beta, mask = precompute_propagator_constants(N, lambda_val, area)
hologram_FT = FT2Dc(hologram, f1)

# Set the range and step size for z
z_values = np.arange(0, 0.1, 0.02)

# Autofocus
best_focus, best_phase_range = find_best_focus(hologram_FT, f1, z_values, N, lambda_val, area, alpha, beta, mask)
print(f'Best focus found at z = {best_focus} with phase value of = {best_phase_range}')

# Reconstruct and display the images at the best focus
reconstruct_and_plot(N, lambda_val, area, best_focus, alpha, beta, mask, hologram_FT, f1)