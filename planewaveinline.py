import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage

def FT2Dc(in_val):
    Nx, Ny = in_val.shape
    f1 = np.exp(1j * np.pi * (np.arange(Nx)[:, None] + np.arange(Ny)))
    FT = np.fft.fft2(f1 * in_val)
    return f1 * FT

def IFT2Dc(in_val):
    Nx, Ny = in_val.shape
    f1 = np.exp(-1j * np.pi * (np.arange(Nx)[:, None] + np.arange(Ny)))
    FT = np.fft.ifft2(f1 * in_val)
    return f1 * FT

def Phase_unwrapping(in_val):
    Nx, Ny = in_val.shape
    x = np.arange(Nx) - Nx/2 - 1
    y = np.arange(Ny) - Ny/2 - 1
    f = x[:, None]**2 + y**2
    a = IFT2Dc(FT2Dc(np.cos(in_val) * IFT2Dc(FT2Dc(np.sin(in_val)) * f) / (f + 0.000001)))
    b = IFT2Dc(FT2Dc(np.sin(in_val) * IFT2Dc(FT2Dc(np.cos(in_val)) * f) / (f + 0.000001)))
    return np.real(a - b)

def Propagator(N, lambda_val, area, z):
    alpha = lambda_val * (np.arange(N) - N/2 - 1) / area
    beta = lambda_val * (np.arange(N)[:, None] - N/2 - 1) / area
    mask = alpha**2 + beta**2 <= 1
    p = np.exp(-2j * np.pi * z * np.sqrt(1 - alpha**2 - beta**2) / lambda_val) * mask
    return p

# PARAMETERS
N = 200                  # number of pixels
lambda_val = 530e-9      # wavelength in meters
area = 0.0015            # area size in meters
z_start = 0.000          # z start in meters
z_end = 0.1              # z end in meters
z_step = 0.005           # z step in meters

# READING HOLOGRAM OBJECT
hologram = np.array(Image.open('U01.png')).astype(float)

# OBJECT RECONSTRUCTED AT DIFFERENT Z-DISTANCES
S = int((z_end - z_start) / z_step)
for ii in range(S):
    z = z_start + ii * z_step
    prop = Propagator(N, lambda_val, area, z)
    recO = np.abs(IFT2Dc(FT2Dc(hologram) * prop))

    # SAVE RECONSTRUCTION AS JPG FILE
    amp = 255 * (recO - np.min(recO)) / (np.max(recO) - np.min(recO))
    Image.fromarray(amp.astype(np.uint8), 'L').save(f'aad_rec_amp_{int(z*1e6)}um.jpg')

    ph = np.angle(IFT2Dc(FT2Dc(hologram) * prop))
    # ph = Phase_unwrapping(-phh)
    pht = 255 * (ph - np.min(ph)) / (np.max(ph) - np.min(ph))
    Image.fromarray(pht.astype(np.uint8), 'L').save(f'aad_rec_phase{int(z*1000000)}um.jpg')


    # Display pht image
    plt.imshow(amp, cmap='gray')
    plt.title(f'Z-dist-phase: {z:.3f} micrometer')
    plt.show()
    
    