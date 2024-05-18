# main.py

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.restoration import unwrap_phase

# Functions

def precompute_ft_terms(Nx, Ny):
    return np.exp(1j * np.pi * (np.arange(Nx)[:, None] + np.arange(Ny)))

def precompute_propagator_constants(N, lambda_val, area):
    indices = np.fft.fftfreq(N) * N
    alpha = lambda_val * indices / area
    beta = lambda_val * indices[:, None] / area
    mask = alpha**2 + beta**2 <= 1
    return alpha, beta, mask

def FT2Dc(in_val, f1):
    return f1 * np.fft.fft2(f1 * in_val)

def IFT2Dc(in_val, f1):
    return f1 * np.fft.ifft2(f1 * in_val)

def Propagator(N, lambda_val, area, z, alpha, beta, mask):
    p = np.exp(-2j * np.pi * z * np.sqrt(1 - alpha**2 - beta**2) / lambda_val) * mask
    return p

def reconstruct_amplitude_phase(N, lambda_val, area, z, alpha, beta, mask, hologram_FT, f1):
    prop = Propagator(N, lambda_val, area, z, alpha, beta, mask)
    hologram_FT_prop = hologram_FT * prop
    amplitude = np.abs(IFT2Dc(hologram_FT_prop, f1))
    phase = np.angle(IFT2Dc(hologram_FT_prop, f1))
    unwrapped_phase = unwrap_phase(-phase)
    return amplitude, unwrapped_phase

def plot_reconstruction(ax1, ax2, cax, z, amplitude, unwrapped_phase):
    ax1.clear()
    ax2.clear()

    ax1.imshow(amplitude, cmap='gray')
    ax1.set_title(f'Amplitude at z = {z*1e3:.3f} mm')  # Convert z to mm for display
    ax1.axis('off')

    im = ax2.imshow(unwrapped_phase, cmap='jet')
    ax2.set_title(f'Unwrapped Phase at z = {z*1e3:.3f} mm')  # Convert z to mm for display
    ax2.axis('off')

    cax.cla()  # Clear the previous colorbar
    plt.colorbar(im, cax=cax)
    plt.draw()
    plt.pause(0.5)  # Pause to allow the plot to be updated iteratively

def reconstruct_and_plot(N, lambda_val, area, best_focus, alpha, beta, mask, hologram_FT, f1):
    prop = Propagator(N, lambda_val, area, best_focus, alpha, beta, mask)
    hologram_FT_prop = hologram_FT * prop
    best_recO_amplitude = np.abs(IFT2Dc(hologram_FT_prop, f1))
    best_recO_phase = np.angle(IFT2Dc(hologram_FT_prop, f1))
    best_recO_unwrapped_phase = unwrap_phase(-best_recO_phase)

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(best_recO_amplitude, cmap='gray')
    ax1.set_title(f'Amplitude at {best_focus*1e3:.3f} mm')  # Convert best_focus to mm for display
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(best_recO_unwrapped_phase, cmap='gray')
    ax2.set_title(f'Unwrapped Phase at {best_focus*1e3:.3f} mm')  # Convert best_focus to mm for display
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(-best_recO_unwrapped_phase, cmap='jet')
    ax3.set_title(f'Phase Colormap: {best_focus*1e3:.3f} mm')  # Convert best_focus to mm for display
    ax3.axis('off')

    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.suptitle(f'Reconstruction at Best Focus: {best_focus*1e3:.3f} mm', fontsize=16)  # Convert best_focus to mm for display
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'best_focus_subplot_{int(best_focus*1e6)}um.png')  # Save as micrometers for filename
    plt.show()

# PARAMETERS
N = 200
lambda_val = 530e-9  # Wavelength in meters
area = 0.0015  # Area in square meters

# READING HOLOGRAM OBJECT
hologram = np.array(Image.open('U01.png')).astype(float)
Nx, Ny = hologram.shape
f1 = precompute_ft_terms(Nx, Ny)
alpha, beta, mask = precompute_propagator_constants(N, lambda_val, area)
hologram_FT = FT2Dc(hologram, f1)

# Set the range and step size for z
z_values = np.arange(0, 0.004, 0.0004)  # z in meters (0 to 0.1 meters in steps of 0.001 meters)

# Create a figure and axes for interactive plotting
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Add a separate axis for the colorbar

# Iterate through the z_values and display the plots
for z in z_values:
    amplitude, unwrapped_phase = reconstruct_amplitude_phase(N, lambda_val, area, z, alpha, beta, mask, hologram_FT, f1)
    plot_reconstruction(ax1, ax2, cax, z, amplitude, unwrapped_phase)

plt.ioff()  # Turn off interactive mode

# After deciding the best focus value, input it here
best_focus = float(input("Enter the best focus value from the plots (in meters): "))

# Iterative reconstruction using the chosen best focus value
Iterations = 20
wavelength = 530e-9
z = best_focus  # Use the chosen best focus value

# Phase retrieval process
plt.ion()  # Turn on interactive mode

# Precompute propagator for the chosen best focus
prop = Propagator(N, wavelength, area, z, alpha, beta, mask)

for i in range(Iterations):
    print(f'Iteration: {i + 1}')

    field_detector = np.sqrt(hologram) * np.exp(1j * np.zeros((N, N)))
    t = IFT2Dc(FT2Dc(field_detector, f1) * np.conj(prop), f1)
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
    field_detector_updated = IFT2Dc(FT2Dc(t, f1) * prop, f1)
    amplitude = np.abs(field_detector_updated)
    phase = np.angle(field_detector_updated)

    # Clear the previous plots
    ax1.clear()
    ax2.clear()

    # Plot reconstructed amplitude and phase for each iteration
    ax1.imshow(amplitude, cmap='gray')
    ax1.set_title(f'Reconstructed Amplitude - Iteration {i + 1}')
    ax1.axis('off')

    im = ax2.imshow(phase, cmap='gray')
    ax2.set_title(f'Reconstructed Phase - Iteration {i + 1}')
    ax2.axis('off')

    cax.cla()  # Clear the previous colorbar
    plt.colorbar(im, cax=cax)
    plt.draw()
    plt.pause(0.1)  # Pause to allow the plot to be updated

plt.ioff()  # Turn off interactive mode

# Final reconstruction plot
reconstruct_and_plot(N, lambda_val, area, best_focus, alpha, beta, mask, hologram_FT, f1)
