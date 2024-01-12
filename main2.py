# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 22:58:10 2024

@author: aad
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from holography_functions2 import FT2Dc, IFT2Dc, Propagator

# Parameters
N = 200
Iterations = 20
wavelength = 530e-9
area = 0.0045
z = 0.61

# Read hologram
hologram = np.array(Image.open('U01.png')).astype(float)
measured = np.sqrt(hologram)


prop = Propagator(N, wavelength, area, z)
phase = np.zeros((N, N))

# Iterative reconstruction
plt.figure(figsize=(10, 20))

for kk in range(Iterations):
    print(f'Iteration: {kk + 1}')
    field_detector = measured * np.exp(1j * phase)
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

    
    field_detector_amp = measured * np.exp(1j * phase)
    t_amp  = IFT2Dc(FT2Dc(field_detector_amp) * np.conj(prop))
    am_amp = np.abs(t_amp)
    
    # Clear the previous figure and create new subplots
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(am_amp, cmap='gray')
    plt.title(f'Reconstructed Amplitude (Iteration {kk + 1})')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(ph, cmap='gray')
    plt.title(f'Reconstructed Phase (Iteration {kk + 1})')
    plt.colorbar()

    # This command updates the figure with the new plots
    plt.pause(0.1)  # you can adjust the pause time as needed

# After all iterations, you may want to show the final figure
plt.show()


# Save reconstructed amplitude and phase
amplitude_norm = (am_amp - np.min(am_amp)) / (np.max(am_amp) - np.min(am_amp))
Image.fromarray((amplitude_norm * 255).astype(np.uint8)).save('reconstructed_amplitude.jpg')

phase_norm = (ph - np.min(ph)) / (np.max(ph) - np.min(ph))
Image.fromarray((phase_norm * 255).astype(np.uint8)).save('reconstructed_phase.jpg')
