# Importing necessary libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
area = 0.0015

def precompute_ft_terms(Nx, Ny):
    x = np.arange(Nx)
    y = np.arange(Ny)
    return np.exp(1j * np.pi * (x[:, None] + y))

def precompute_propagator_constants(N, lambda_val, area):
    indices = np.arange(N) - N / 2 - 1
    alpha = lambda_val * indices / area
    beta = lambda_val * indices[:, None] / area
    mask = alpha**2 + beta**2 <= 1
    return alpha, beta, mask

def FT2Dc(in_val):
    Nx, Ny = in_val.shape
    f1 = np.exp(1j * np.pi * (np.arange(Nx)[:, None] + np.arange(Ny)))
    return f1 * np.fft.fft2(f1 * in_val)

def IFT2Dc(in_val):
    Nx, Ny = in_val.shape
    f1 = np.exp(-1j * np.pi * (np.arange(Nx)[:, None] + np.arange(Ny)))
    return f1 * np.fft.ifft2(f1 * in_val)

def unwrap_phase(phase):
    Ny, Nx = phase.shape
    reliability = get_reliability(phase)
    h_edges, v_edges = get_edges(reliability)
    edges = np.concatenate((h_edges.ravel(), v_edges.ravel()))
    edge_bound_idx = Ny * (Nx - 1)
    edge_sort_idx = np.argsort(-edges)
    idxs1 = edge_sort_idx % edge_bound_idx
    idxs2 = idxs1 + 1 + Ny * (edge_sort_idx >= edge_bound_idx)
    group = np.arange(Ny * Nx)
    is_grouped = np.zeros(Ny * Nx, dtype=bool)
    group_members = {i: [i] for i in range(Ny * Nx)}
    res_img = np.copy(phase)
    for i in range(len(edge_sort_idx)):
        idx1, idx2 = idxs1[i], idxs2[i]
        if idx2 >= Ny * Nx:
            continue
        if group[idx1] == group[idx2]:
            continue
        dval = np.floor((res_img.flat[idx2] - res_img.flat[idx1] + np.pi) / (2 * np.pi)) * 2 * np.pi
        g1, g2 = group[idx1], group[idx2]
        res_img.flat[group_members[g1]] += dval
        group_members[g2].extend(group_members[g1])
        group[group_members[g1]] = g2
        is_grouped[idx1] = is_grouped[idx2] = True
    return res_img.reshape(Ny, Nx)

def get_reliability(img):
    D = np.abs(np.diff(img, axis=0, prepend=img[0:1])) + np.abs(np.diff(img, axis=1, prepend=img[:, 0:1]))
    return 1 / (D + 1e-6)

def get_edges(rel):
    h_edges = rel[:, :-1] + rel[:, 1:]
    v_edges = rel[:-1, :] + rel[1:, :]
    return h_edges, v_edges

def Propagator(N, lambda_val, area, z):
    indices = np.arange(N) - N / 2 - 1
    alpha = lambda_val * indices / area
    beta = lambda_val * indices[:, None] / area
    mask = alpha**2 + beta**2 <= 1
    p = np.exp(-2j * np.pi * z * np.sqrt(1 - alpha**2 - beta**2) / lambda_val) * mask
    return p

def normalize_and_save_image(image, filename):
    normalized = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    Image.fromarray(np.clip(normalized, 0, 255).astype(np.uint8), 'L').save(filename)

def calculate_phase_range(phh):
    return np.ptp(phh)

def find_best_focus(hologram_FT, f1, z_values, N, lambda_val, area):
    best_focus, best_phase_range = None, -np.inf
    alpha, beta, mask = precompute_propagator_constants(N, lambda_val, area)

    for z in z_values:
        prop = Propagator(N, lambda_val, area, z)
        hologram_FT_prop = hologram_FT * prop
        phh = unwrap_phase(-np.angle(IFT2Dc(hologram_FT_prop)))  # Adjusted to pass only one argument

        phh_range = calculate_phase_range(phh)
        if phh_range > best_phase_range:
            best_phase_range, best_focus = phh_range, z

    return best_focus, best_phase_range



def reconstruct_and_plot(N, lambda_val, area, best_focus, hologram, amplitude, phase, hologram_FT, f1):
    prop = Propagator(N, lambda_val, area, best_focus)
    hologram_FT_prop = hologram_FT * prop

    # Adjusted to pass the correct number of arguments to IFT2Dc
    best_recO_amplitude = np.abs(IFT2Dc(hologram_FT_prop))
    best_recO_phase = np.angle(IFT2Dc(hologram_FT_prop))
    best_recO_unwrapped_phase = unwrap_phase(-best_recO_phase)

        # Prepare for combined plotting
    fig = plt.figure(figsize=(18, 12))  # Adjust the figure size as needed
    gs = GridSpec(2, 3, figure=fig)

    # Plotting hologram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(hologram, cmap='gray')
    ax1.set_title('Hologram')
    ax1.axis('off')

    # Plotting Amplitude at Best Focus
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(best_recO_amplitude, cmap='gray')
    ax2.set_title(f'Amplitude at {best_focus:.3f} micrometer')
    ax2.axis('off')

    # Plotting Unwrapped Phase at Best Focus
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(best_recO_unwrapped_phase, cmap='gray')
    ax3.set_title(f'Unwrapped Phase at {best_focus:.3f} micrometer')
    ax3.axis('off')

    # Plotting Phase Colormap
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(-best_recO_unwrapped_phase, cmap='jet')
    ax4.set_title(f'Phase Colormap: {best_focus:.3f} radian')
    ax4.axis('off')

    # Plotting Reconstructed Amplitude
    amplitude_norm = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude))
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(amplitude_norm, cmap='gray')
    ax5.set_title('Reconstructed Amplitude')
    ax5.axis('off')

    # Plotting Reconstructed Phase
    phase_norm = (phase - np.min(phase)) / (np.max(phase) - np.min(phase))
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(phase_norm, cmap='gray')
    ax6.set_title('Reconstructed Phase')
    ax6.axis('off')

    # Adding colorbar for the colormap
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


