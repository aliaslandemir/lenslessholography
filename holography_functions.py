
import numpy as np
from PIL import Image

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
    D[1:-1, 1:-1] = np.abs(img[:-2, 1:-1] - 2 * img[1:-1, 1:-1] + img[2:, 1:-1]) +                     np.abs(img[1:-1, :-2] - 2 * img[1:-1, 1:-1] + img[1:-1, 2:])
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
