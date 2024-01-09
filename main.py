
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from holography_functions import precompute_ft_terms, precompute_propagator_constants, FT2Dc, IFT2Dc, unwrap_phase, Propagator

N = 200
lambda_val = 530e-9
area = 0.0015
z_start, z_end, z_step = 0, 0.1, 0.02

hologram = np.array(Image.open('U01.png')).astype(float)
Nx, Ny = hologram.shape
f1 = precompute_ft_terms(Nx, Ny)
alpha, beta, mask = precompute_propagator_constants(N, lambda_val, area)
hologram_FT = FT2Dc(hologram, f1)

S = int((z_end - z_start) / z_step)
for ii in range(S):
    z = z_start + ii * z_step
    prop = Propagator(N, lambda_val, area, z, alpha, beta, mask)
    recO = np.abs(IFT2Dc(hologram_FT * prop, f1))
    ph = np.angle(IFT2Dc(hologram_FT * prop, f1))
    phh = unwrap_phase(-ph)
    phh_range = np.abs(np.max(phh) - np.min(phh))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(recO, cmap='gray')
    axs[0].set_title(f'Amplitude: {z:.3f} micrometer')
    axs[0].axis('off')
    axs[1].imshow(phh, cmap='gray')
    axs[1].set_title(f'Unwrapped Phase: {z:.3f} micrometer')
    axs[1].axis('off')
    im = axs[2].imshow(-phh, cmap='jet')
    axs[2].set_title(f'Phase Colormap: {phh_range:.3f} radian')
    axs[2].axis('off')
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f'aad_subplot_{int(z*1e6)}um.png')
    plt.show()
