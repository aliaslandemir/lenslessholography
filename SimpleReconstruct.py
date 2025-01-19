import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.restoration import unwrap_phase
import time

# Functions

def precompute_ft_terms(Nx, Ny):
    return np.exp(1j * np.pi * (np.arange(Nx)[:, None] + np.arange(Ny)))

def precompute_propagator_constants(N, lambda_val, area):
    alpha = lambda_val * (np.arange(N) - N / 2 - 1) / area
    beta = lambda_val * (np.arange(N)[:, None] - N / 2 - 1) / area
    mask = alpha**2 + beta**2 <= 1
    return alpha, beta, mask

def FT2Dc(in_val, f1):
    return f1 * np.fft.fft2(f1 * in_val)

def IFT2Dc(in_val, f1):
    return f1 * np.fft.ifft2(f1 * in_val)

def Propagator(N, lambda_val, area, z, alpha, beta, mask):
    p = np.exp(-2j * np.pi * z * np.sqrt(1 - alpha**2 - beta**2) / lambda_val) * mask
    return p

def calculate_phase_range(phh):
    return np.ptp(phh)

def find_best_focus(hologram_FT, f1, z_values, N, lambda_val, area, alpha, beta, mask):
    best_focus = None
    best_phase_range = -np.inf

    for z in z_values:
        prop = Propagator(N, lambda_val, area, z, alpha, beta, mask)
        hologram_FT_prop = hologram_FT * prop
        ph = np.angle(IFT2Dc(hologram_FT_prop, f1))
        phh = unwrap_phase(-ph)

        phh_range = calculate_phase_range(phh)
        if phh_range > best_phase_range:
            best_phase_range = phh_range
            best_focus = z

    return best_focus, best_phase_range

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
    ax1.set_title(f'Amplitude at {best_focus:.3f} micrometer')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(best_recO_unwrapped_phase, cmap='gray')
    ax2.set_title(f'Unwrapped Phase at {best_focus:.3f} micrometer')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(-best_recO_unwrapped_phase, cmap='jet')
    ax3.set_title(f'Phase Colormap: {best_phase_range:.3f} radian')
    ax3.axis('off')

    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.suptitle(f'Reconstruction at Best Focus: {best_focus:.3f} micrometer', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'best_focus_subplot_{int(best_focus*1e6)}um.png')
    plt.show()

# Measure total processing time
start_time_total = time.time()


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
z_values = np.arange(0, 1, 0.001)

# Autofocus
best_focus, best_phase_range = find_best_focus(hologram_FT, f1, z_values, N, lambda_val, area, alpha, beta, mask)
print(f'Best focus found at z = {best_focus} with phase value of = {best_phase_range}')

# Reconstruct and display the images at the best focus
reconstruct_and_plot(N, lambda_val, area, best_focus, alpha, beta, mask, hologram_FT, f1)

end_time_total = time.time()
print(f'Total processing time: {end_time_total - start_time_total:.2f} seconds')
