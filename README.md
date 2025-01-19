# Inline Holography Reconstruction

This project provides a PyQt-based GUI for inline holography reconstruction. It supports both **direct** and **iterative phase retrieval** methods for reconstructing amplitude and phase from holograms. The GUI enables scientists and researchers to analyze holograms with customizable parameters, constraints, and visualization tools.

---

## What Does This Code Do?

The application allows users to:

1. **Load a Hologram**: Load a hologram image in standard formats (e.g., PNG, JPG, TIF).
2. **Direct Reconstruction**: Use the **Angular Spectrum** or **Fresnel propagation** to reconstruct amplitude and phase directly.
3. **Iterative Phase Retrieval**: Apply **Gershberg-Saxton-like** iterations with constraints to suppress twin images and enhance reconstruction:
   - **Positivity Enforcement**: Enforces non-negative amplitude in the object plane.
   - **Finite Support**: Limits reconstruction to a specific spatial region (adaptive support threshold available).
   - **Phase Extrapolation**: Pads the hologram with zeros or random noise to recover missing boundary information.
4. **Autofocus**: Perform a brute-force search for the propagation distance (**z**) that maximizes the unwrapped phase range.
5. **Twin-Image Filtering**: Suppress twin images using Fourier domain filtering.
6. **3D Deconvolution (Placeholder)**: Enhance resolution using a simple Wiener-like filter (demonstrates potential for iterative 3D deconvolution).
7. **Visualization**:
   - Compare **Direct** vs. **Iterative Reconstruction** results in a **2×3 grid** of subplots (amplitude, wrapped phase, and unwrapped phase).
   - Analyze **line profiles** in a **2×2 grid**, including comparisons of amplitude, wrapped phase, and unwrapped phase.

---

## Screenshots

### Main Application
![Main Application](docs/screenshot.png)

---

## Features

### Direct Reconstruction
- Supports **Angular Spectrum** and **Fresnel propagation** models.
- Adjustable parameters for the wavelength, hologram size, and propagation distance.

### Iterative Phase Retrieval
- Flexible iteration count (`max_iter`) with constraints:
  - **Positivity Threshold**: Force amplitude ≥ 0.
  - **Support Threshold**: Create a mask based on amplitude thresholds and morphological dilation.
  - **Adaptive Thresholding**: Dynamically relax the support threshold over iterations.

### Autofocus
- Brute-force scan over a user-defined range of `z` values to find the best focus.
- Automatically updates reconstruction parameters for the optimal focus distance.

### Visualization
- **Main 2×3 Plot**: Direct and iterative reconstruction results for amplitude, wrapped phase, and unwrapped phase.
- **Line Profile Window**: 2×2 comparison of amplitude and phase profiles (wrapped and unwrapped).

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual Environment (optional but recommended)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/aliaslandemir/lenslessholography.git



---

## License

This project is licensed under the [MIT License](License). You are free to use, modify, and distribute this software under the terms of the license.
