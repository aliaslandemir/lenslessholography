# main.py

# Importing required libraries
import numpy as np
from PIL import Image


# Importing functions from holography_func.py
from holography_func import precompute_ft_terms,  Propagator, IFT2Dc, FT2Dc, find_best_focus, reconstruct_and_plot

# PARAMETERS
N = 200
lambda_val = 530e-9
area = 0.0015

# READING HOLOGRAM OBJECT
hologram = np.array(Image.open('U01.png')).astype(float)
Nx, Ny = hologram.shape
f1 = precompute_ft_terms(Nx, Ny)
hologram_FT = FT2Dc(hologram)

# Autofocus
z_values = np.arange(0, 0.1, 0.02)
best_focus, best_phase_range = find_best_focus(hologram_FT, f1, z_values, N, lambda_val, area)
print(f'Best focus found at z = {best_focus} with phase value of = {best_phase_range}')

# Parameters for Phase Retrieval
Iterations = 20
wavelength = 530e-9
z = best_focus  # Use the best focus value from autofocus


# Phase retrieval process
amplitude = None  # Placeholder, replace with actual calculation
phase = None      # Placeholder, replace with actual calculation
for i in range(Iterations):
    print(f'Iteration: {i + 1}')

    prop = Propagator(N, wavelength, area, z)
    field_detector = np.sqrt(hologram) * np.exp(1j * np.zeros((N, N)))
    t = IFT2Dc(FT2Dc(field_detector) * np.conj(prop))
    am = np.abs(t)
    ph = np.angle(t)
    abso = -np.log(am)

    # Apply constraint in the object plane
    abso[abso < 0] = 0
    ph[abso < 0] = 0
    am = np.exp(-abso)

    # Update transmission function in the object plane
    t = am * np.exp(1j * ph)

    # Calculate complex-valued wavefront in the detector plane
    field_detector_updated = IFT2Dc(FT2Dc(t) * prop)
    amplitude = np.abs(field_detector_updated)
    phase = np.angle(field_detector_updated)
    
# Ensure 'amplitude' and 'phase' are computed and available here
if amplitude is not None and phase is not None:
    reconstruct_and_plot(N, lambda_val, area, best_focus, hologram, amplitude, phase, hologram_FT, f1)
else:
    print("Amplitude or phase not defined")

