import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def precompute_ft_terms(Nx, Ny):
    return np.exp(1j * np.pi * (np.arange(Nx)[:, None] + np.arange(Ny)))

def precompute_propagator_constants(N, lambda_val, area):
    alpha = lambda_val * (np.arange(N) - N/2 - 1) / area
    beta = lambda_val * (np.arange(N)[:, None] - N/2 - 1) / area
    mask = alpha**2 + beta**2 <= 1
    return alpha, beta, mask

def FT2Dc(in_val, f1):
    return f1 * np.fft.fft2(f1 * in_val)

def IFT2Dc(in_val, f1):
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
    D = np.zeros_like(img)
    D[1:-1, 1:-1] = np.abs(img[:-2, 1:-1] - 2 * img[1:-1, 1:-1] + img[2:, 1:-1]) + np.abs(img[1:-1, :-2] - 2 * img[1:-1, 1:-1] + img[1:-1, 2:])
    return 1 / (D + 1e-6)

def get_edges(rel):
    Ny, Nx = rel.shape
    h_edges = rel[:, :-1] + rel[:, 1:]
    v_edges = rel[:-1, :] + rel[1:, :]
    return h_edges, v_edges

def Propagator(N, lambda_val, area, z, alpha, beta, mask):
    p = np.exp(-2j * np.pi * z * np.sqrt(1 - alpha**2 - beta**2) / lambda_val) * mask
    return p

def normalize_and_save_image(image, filename):
    image = np.nan_to_num(image)
    normalized = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    Image.fromarray(np.clip(normalized, 0, 255).astype(np.uint8), 'L').save(filename)

def calculate_phase_range(phh):
    return np.abs(np.max(phh) - np.min(phh))

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
    ax3.set_title(f'Phase Colormap: {best_focus:.3f} radian')
    ax3.axis('off')

    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # Adding a suptitle to the figure
    plt.suptitle(f'Reconstruction at Best Focus: {best_focus:.3f} micrometer', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
    plt.savefig(f'best_focus_subplot_{int(best_focus*1e6)}um.png')
    plt.show()
