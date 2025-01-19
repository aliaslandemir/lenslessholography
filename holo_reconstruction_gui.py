#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lensless Inline Holography Reconstruction GUI


This script provides a PyQt-based GUI for loading a single-plane hologram,
performing either Angular Spectrum or Fresnel propagation, and iteratively
retrieving phase with additional constraints (positivity, finite support).
It supports zero or random padding (phase extrapolation), optional twin-image
filtering, autofocus, and optional 3D deconvolution.

Features:
- Angular/Fresnel propagator
- Positivity & finite support constraints
- Adaptive support threshold (optional)
- 3D Deconvolution (placeholder Wiener filter)
- 2×3 subplot for amplitude & phase (wrapped/unwrapped) comparisons
- Separate line-profile window with 2×2 subplots comparing direct vs. iterative reconstructions:
  1) Amplitude profiles
  2) Wrapped phase profiles
  3) Unwrapped phase profiles
  4) (Optional) amplitude difference or any other user-defined data

Usage:
  python advanced_holography_gui.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PIL import Image

from skimage.restoration import unwrap_phase
from skimage.morphology import binary_dilation, disk

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGroupBox, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog, QDoubleSpinBox,
    QSpinBox, QMessageBox, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt

###############################################################################
# 1) FFT Utilities
###############################################################################

def precompute_ft_terms(Nx, Ny):
    """
    Precompute a phase factor for "centered" FFT/iFFT.
    This factor is multiplied before and after transforms
    to emulate a shift to/from the image center.
    """
    x = np.arange(Nx)
    y = np.arange(Ny)
    return np.exp(1j * np.pi * (x[:, None] + y))

def FT2Dc(in_val, f1):
    """2D centered FFT."""
    return f1 * np.fft.fft2(f1 * in_val)

def IFT2Dc(in_val, f1):
    """2D centered IFFT."""
    return f1 * np.fft.ifft2(f1 * in_val)

###############################################################################
# 2) Propagation Methods (Angular Spectrum or Fresnel)
###############################################################################

def precompute_propagator_constants(N, lambda_val, area):
    """
    Precompute alpha, beta, and a mask for the Angular Spectrum approach.
    (Assumes Nx=Ny=N.)
    """
    coords = np.arange(N) - N // 2
    alpha = coords / area * lambda_val
    beta  = coords / area * lambda_val
    
    alpha = alpha.reshape(-1, 1)  # Nx x 1
    beta  = beta.reshape(1, -1)   # 1 x N
    
    # Mask for valid spatial frequencies (circle in alpha-beta space)
    mask = (alpha**2 + beta**2) <= 1.0
    return alpha, beta, mask

def propagator_angular_spectrum(N, lambda_val, area, z, alpha, beta, mask):
    """
    Returns the angular-spectrum propagator for distance z.
    """
    inside = 1.0 - (alpha**2 + beta**2)
    inside[inside < 0] = 0  # clamp negatives
    phase_term = -2j * np.pi * z * np.sqrt(inside) / lambda_val
    Pz = np.exp(phase_term) * mask
    return Pz

def propagator_fresnel(N, lambda_val, area, z):
    """
    Returns a Fresnel propagator kernel for distance z:
      H(fx,fy) = exp(1j*k*z) * exp(-1j * pi * lambda * z * (fx^2 + fy^2))
    We fftshift for convenience.
    """
    k = 2 * np.pi / lambda_val
    dx = area / N  # real-space sampling
    # freq coordinates
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * lambda_val * z * (FX**2 + FY**2))
    return np.fft.fftshift(H)

def generate_propagator(N, lambda_val, area, z, alpha, beta, mask, method='angular'):
    """
    Wrapper that returns the appropriate propagator kernel
    based on the chosen method.
    """
    if method.lower() == 'angular':
        return propagator_angular_spectrum(N, lambda_val, area, z, alpha, beta, mask)
    elif method.lower() == 'fresnel':
        return propagator_fresnel(N, lambda_val, area, z)
    else:
        raise ValueError(f"Unsupported propagation method: {method}")

###############################################################################
# 3) Filtering / Preprocessing
###############################################################################

def sideband_filter(hologram):
    """
    Example twin-image filter: A naive band-pass in Fourier space.
    """
    H = np.fft.fft2(hologram)
    H_shifted = np.fft.fftshift(H)
    
    nx, ny = hologram.shape
    cx, cy = nx // 2, ny // 2
    
    # Example radial band
    radius_min = 5
    radius_max = min(cx, cy) // 2
    
    X, Y = np.meshgrid(np.arange(ny), np.arange(nx))
    r2 = (X - cy)**2 + (Y - cx)**2
    mask = (r2 >= radius_min**2) & (r2 <= radius_max**2)
    
    H_shifted[~mask] = 0
    H_filt = np.fft.ifftshift(H_shifted)
    
    holo_filtered = np.fft.ifft2(H_filt)
    return np.abs(holo_filtered)

###############################################################################
# 4) Advanced Constraints & 3D Deconvolution
###############################################################################

def generate_object_support(amp_obj, threshold, dilation_radius):
    """
    Create a binary support mask in the object plane
    by thresholding amplitude and (optionally) applying morphological dilation.
    """
    mask = amp_obj > threshold
    if dilation_radius > 0:
        mask = binary_dilation(mask, disk(dilation_radius))
    return mask

def apply_3d_deconvolution(field_obj, wiener_param=1e-3):
    """
    Placeholder for advanced 3D deconvolution/resolution enhancement.
    Demonstrates a simple Wiener-like filter in Fourier space:
        E_deconv = IFFT( FFT(E) / (OTF + wiener_param) )
    Here, OTF=1 for demonstration, so E_deconv = E / (1 + wiener_param).
    """
    F = np.fft.fft2(field_obj)
    F_deconv = F / (1.0 + wiener_param)
    return np.fft.ifft2(F_deconv)

###############################################################################
# 5) Iterative Phase Retrieval
###############################################################################

def iterative_phase_retrieval(
    holo_amp,
    f1,
    Pz_forward,
    Pz_backward,
    max_iter=10,
    positivity_enforce=True,
    support_threshold=0.0,
    dilation_radius=0,
    adaptive_threshold=False
):
    """
    A Gershberg–Saxton-like iterative phase retrieval with constraints:
      - positivity_enforce: ensures amplitude >= 0
      - support_threshold + dilation_radius: finite support in object plane
      - adaptive_threshold: reduces threshold each iteration
    """
    Nx, Ny = holo_amp.shape
    
    # Start with random phase in hologram plane
    phase_init = 2.0 * np.pi * np.random.rand(Nx, Ny)
    Eholo = holo_amp * np.exp(1j * phase_init)
    
    current_threshold = support_threshold
    
    for _ in range(max_iter):
        # Forward: Hologram plane -> Object plane
        Eobj = IFT2Dc(FT2Dc(Eholo, f1) * Pz_forward, f1)
        
        # Apply constraints
        amp_obj = np.abs(Eobj)
        phase_obj = np.angle(Eobj)
        
        # Positivity
        if positivity_enforce:
            amp_obj[amp_obj < 0] = 0
        
        # Finite support
        if current_threshold > 0:
            support_mask = generate_object_support(amp_obj, current_threshold, dilation_radius)
            amp_obj[~support_mask] = 0
        
        Eobj = amp_obj * np.exp(1j * phase_obj)
        
        # Backward: Object plane -> Hologram plane
        Eholo_back = IFT2Dc(FT2Dc(Eobj, f1) * Pz_backward, f1)
        
        # Enforce known amplitude in hologram plane
        Eholo = holo_amp * np.exp(1j * np.angle(Eholo_back))
        
        # Adaptive threshold (optional)
        if adaptive_threshold and current_threshold > 0:
            current_threshold *= 0.95  # e.g. reduce threshold by 5% each iteration
    
    return Eobj

###############################################################################
# 6) PyQt5 Visualization Classes
###############################################################################

class MplCanvas(FigureCanvas):
    """
    A Matplotlib canvas with 2 rows x 3 columns of subplots:
      Top row   : direct reconstruction (amplitude, wrapped phase, unwrapped phase)
      Bottom row: iterative reconstruction (amplitude, wrapped phase, unwrapped phase)
    """
    def __init__(self, parent=None, width=12, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        axes = fig.subplots(2, 3)
        
        self.fig = fig
        self.ax_amp_direct   = axes[0, 0]
        self.ax_wp_direct    = axes[0, 1]
        self.ax_uwp_direct   = axes[0, 2]
        
        self.ax_amp_iter     = axes[1, 0]
        self.ax_wp_iter      = axes[1, 1]
        self.ax_uwp_iter     = axes[1, 2]
        
        super().__init__(fig)
        self.setParent(parent)
        self.fig.tight_layout()

class LineProfileWindow(QWidget):
    """
    A separate window to display 2×2 line profiles:
      - Top-left:  Amplitude (direct vs. iterative)
      - Top-right: Wrapped phase (direct vs. iterative)
      - Bottom-left: Unwrapped phase (direct vs. iterative)
      - Bottom-right: (Optional) difference or another comparison
    """
    def __init__(self, amp_direct, phase_direct, amp_iter, phase_iter, unwrap_direct, unwrap_iter):
        super().__init__()
        self.setWindowTitle("Line Profile Comparisons")
        
        layout = QVBoxLayout(self)
        
        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        ax = []
        ax.append(self.fig.add_subplot(2, 2, 1))  # amplitude
        ax.append(self.fig.add_subplot(2, 2, 2))  # wrapped phase
        ax.append(self.fig.add_subplot(2, 2, 3))  # unwrapped phase
        ax.append(self.fig.add_subplot(2, 2, 4))  # optional difference or another metric
        
        # Create row-center line profiles
        Nx = amp_direct.shape[0]
        Ny = amp_direct.shape[1]
        row_center = Nx // 2
        
        # 1) Amplitude line profile
        profile_amp_direct = amp_direct[row_center, :]
        profile_amp_iter   = amp_iter[row_center, :]
        
        # 2) Wrapped phase line profile
        profile_phase_direct = phase_direct[row_center, :]
        profile_phase_iter   = phase_iter[row_center, :]
        
        # 3) Unwrapped phase line profile
        profile_uwp_direct = unwrap_direct[row_center, :]
        profile_uwp_iter   = unwrap_iter[row_center, :]
        
        # --- Top-left: Amplitude ---
        ax[0].plot(profile_amp_direct, 'b-', label='Direct Amp')
        ax[0].plot(profile_amp_iter,   'r--', label='Iter Amp')
        ax[0].set_title("Amplitude Profile")
        ax[0].legend()
        
        # --- Top-right: Wrapped Phase ---
        ax[1].plot(profile_phase_direct, 'b-', label='Direct Wrapped')
        ax[1].plot(profile_phase_iter,   'r--', label='Iter Wrapped')
        ax[1].set_title("Wrapped Phase Profile")
        ax[1].legend()
        
        # --- Bottom-left: Unwrapped Phase ---
        ax[2].plot(profile_uwp_direct, 'b-', label='Direct Unwrapped')
        ax[2].plot(profile_uwp_iter,   'r--', label='Iter Unwrapped')
        ax[2].set_title("Unwrapped Phase Profile")
        ax[2].legend()
        
        # --- Bottom-right: Optional difference or 2D correlation, etc.
        # For demonstration, let's do amplitude difference
        difference_amp = profile_amp_direct - profile_amp_iter
        ax[3].plot(difference_amp, 'k-', label='Amp Diff (Dir - Iter)')
        ax[3].set_title("Amplitude Difference")
        ax[3].legend()
        
        self.fig.tight_layout()
        self.canvas.draw()

###############################################################################
# 7) Main PyQt5 GUI
###############################################################################

class HoloReconstructionApp(QMainWindow):
    """
    Main Window for advanced lensless holography reconstruction.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Inline Holography Reconstruction")
        
        # Default parameters
        self.N = 200
        self.lambda_val = 530e-9  # 530 nm
        self.area = 1.5e-3       # 1.5 mm
        self.hologram = None
        self.hologram_filtered = None
        self.hologram_FT = None
        
        # Propagation method
        self.propagation_method = 'angular'
        
        # Precompute for Nx=Ny=200
        self.f1 = precompute_ft_terms(self.N, self.N)
        self.alpha, self.beta, self.mask = precompute_propagator_constants(
            self.N, self.lambda_val, self.area
        )
        
        # Data references (for line profiles)
        self.amp_direct = None
        self.phase_direct = None
        self.amp_iter = None
        self.phase_iter = None
        self.unwrap_direct = None
        self.unwrap_iter = None
        
        # Build UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left: Control panel
        control_panel = self.create_control_panel()
        # Right: Plot canvas
        self.canvas = MplCanvas(self, width=12, height=8)
        
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(self.canvas, 3)
        
        self.colorbars = []
    
    def create_control_panel(self):
        panel = QGroupBox("Controls")
        layout = QVBoxLayout()
        
        # 1) Load hologram + extrapolation
        btn_load = QPushButton("Load Hologram")
        btn_load.clicked.connect(self.load_hologram)
        
        lbl_extrap = QLabel("Extrapolation Factor:")
        self.spin_extrap = QSpinBox()
        self.spin_extrap.setRange(1, 4)
        self.spin_extrap.setValue(1)
        
        lbl_extrap_mode = QLabel("Extrapolation Mode:")
        self.combo_extrap_mode = QComboBox()
        self.combo_extrap_mode.addItems(["Zeros", "Random"])
        
        # 2) Twin-image filter
        btn_filter = QPushButton("Twin-Image Filter")
        btn_filter.clicked.connect(self.run_filter)
        
        # 3) Autofocus
        lbl_zmin = QLabel("z min (m):")
        self.spin_zmin = QDoubleSpinBox()
        self.spin_zmin.setRange(1e-6, 0.5)
        self.spin_zmin.setValue(0.0001)
        self.spin_zmin.setDecimals(8)
        
        lbl_zmax = QLabel("z max (m):")
        self.spin_zmax = QDoubleSpinBox()
        self.spin_zmax.setRange(1e-6, 0.5)
        self.spin_zmax.setValue(0.005)
        self.spin_zmax.setDecimals(8)
        
        lbl_zstep = QLabel("z step (m):")
        self.spin_zstep = QDoubleSpinBox()
        self.spin_zstep.setRange(1e-7, 0.1)
        self.spin_zstep.setValue(0.0001)
        self.spin_zstep.setDecimals(8)
        
        btn_autofocus = QPushButton("Search Best Focus")
        btn_autofocus.clicked.connect(self.run_autofocus)
        
        # 4) Manual z
        lbl_zmanual = QLabel("Manual z (m):")
        self.spin_zmanual = QDoubleSpinBox()
        self.spin_zmanual.setRange(1e-6, 0.5)
        self.spin_zmanual.setValue(0.005)
        self.spin_zmanual.setDecimals(8)
        
        # 5) Propagation method
        lbl_prop_method = QLabel("Propagation Method:")
        self.combo_prop_method = QComboBox()
        self.combo_prop_method.addItems(["Angular Spectrum", "Fresnel"])
        self.combo_prop_method.currentIndexChanged.connect(self.update_propagation_method)
        
        # 6) Iterations & constraints
        lbl_iter = QLabel("Max Iterations:")
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 1000)
        self.spin_iter.setValue(10)
        
        lbl_pos_thresh = QLabel("Positivity Threshold:")
        self.spin_pos_thresh = QDoubleSpinBox()
        self.spin_pos_thresh.setRange(0.0, 1e5)
        self.spin_pos_thresh.setValue(0.0)
        
        lbl_supp_thresh = QLabel("Support Threshold:")
        self.spin_supp_thresh = QDoubleSpinBox()
        self.spin_supp_thresh.setRange(0.0, 1e5)
        self.spin_supp_thresh.setValue(10.0)
        
        lbl_dilation = QLabel("Dilation Radius:")
        self.spin_dilation = QSpinBox()
        self.spin_dilation.setRange(0, 50)
        self.spin_dilation.setValue(2)
        
        self.check_adaptive_thresh = QCheckBox("Adaptive Threshold")
        self.check_3d_deconv = QCheckBox("3D Deconvolution")
        
        # 7) Reconstruct
        btn_reconstruct = QPushButton("Reconstruct @ z")
        btn_reconstruct.clicked.connect(self.run_reconstruction)
        
        # 8) Show line profiles
        btn_line_profiles = QPushButton("Show Line Profiles")
        btn_line_profiles.clicked.connect(self.show_line_profiles)
        
        # Add to layout
        layout.addWidget(btn_load)
        layout.addWidget(lbl_extrap)
        layout.addWidget(self.spin_extrap)
        layout.addWidget(lbl_extrap_mode)
        layout.addWidget(self.combo_extrap_mode)
        
        layout.addWidget(btn_filter)
        
        layout.addWidget(lbl_zmin)
        layout.addWidget(self.spin_zmin)
        layout.addWidget(lbl_zmax)
        layout.addWidget(self.spin_zmax)
        layout.addWidget(lbl_zstep)
        layout.addWidget(self.spin_zstep)
        layout.addWidget(btn_autofocus)
        
        layout.addWidget(lbl_zmanual)
        layout.addWidget(self.spin_zmanual)
        
        layout.addWidget(lbl_prop_method)
        layout.addWidget(self.combo_prop_method)
        
        layout.addWidget(lbl_iter)
        layout.addWidget(self.spin_iter)
        
        layout.addWidget(lbl_pos_thresh)
        layout.addWidget(self.spin_pos_thresh)
        
        layout.addWidget(lbl_supp_thresh)
        layout.addWidget(self.spin_supp_thresh)
        
        layout.addWidget(lbl_dilation)
        layout.addWidget(self.spin_dilation)
        
        layout.addWidget(self.check_adaptive_thresh)
        layout.addWidget(self.check_3d_deconv)
        
        layout.addWidget(btn_reconstruct)
        layout.addWidget(btn_line_profiles)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def update_propagation_method(self):
        text = self.combo_prop_method.currentText()
        if text == "Angular Spectrum":
            self.propagation_method = 'angular'
        else:
            self.propagation_method = 'fresnel'
    
    def load_hologram(self):
        """Load and optionally extrapolate the hologram."""
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Hologram", "",
            "Images (*.png *.jpg *.bmp *.tif)"
        )
        if fname:
            img = np.array(Image.open(fname)).astype(float)
            Nx, Ny = img.shape
            
            # Extrapolation
            factor = self.spin_extrap.value()
            Nx_new = Nx * factor
            Ny_new = Ny * factor
            
            self.N = int(max(Nx_new, Ny_new))
            
            # Recompute for new N
            self.f1 = precompute_ft_terms(self.N, self.N)
            self.alpha, self.beta, self.mask = precompute_propagator_constants(
                self.N, self.lambda_val, self.area
            )
            
            holo_padded = np.zeros((self.N, self.N), dtype=float)
            
            mode = self.combo_extrap_mode.currentText().lower()
            if mode == 'random':
                holo_padded[:] = np.random.rand(self.N, self.N) * np.mean(img)
            
            # Center the original in holo_padded
            start_x = (self.N - Nx) // 2
            start_y = (self.N - Ny) // 2
            holo_padded[start_x:start_x+Nx, start_y:start_y+Ny] = img
            
            self.hologram = holo_padded
            self.hologram_filtered = None
            self.hologram_FT = FT2Dc(self.hologram, self.f1)
            
            QMessageBox.information(self, "Load Hologram", f"Loaded {fname}")
    
    def run_filter(self):
        """Apply sideband (twin-image) filter."""
        if self.hologram is None:
            QMessageBox.warning(self, "Warning", "No hologram loaded.")
            return
        self.hologram_filtered = sideband_filter(self.hologram)
        QMessageBox.information(self, "Filter", "Twin-image filtering done.")
    
    def run_autofocus(self):
        """Brute-force search for best z by maximizing unwrapped phase range."""
        if self.hologram is None:
            QMessageBox.warning(self, "Warning", "No hologram loaded.")
            return
        
        zmin = self.spin_zmin.value()
        zmax = self.spin_zmax.value()
        zstep = self.spin_zstep.value()
        if zstep <= 0:
            QMessageBox.warning(self, "Invalid Input", "z step must be > 0.")
            return
        
        z_values = np.arange(zmin, zmax, zstep)
        if len(z_values) < 1:
            QMessageBox.warning(self, "Invalid Range",
                                "No valid z-values. Check your min, max, step.")
            return
        
        if self.hologram_filtered is not None:
            holo_ft = FT2Dc(self.hologram_filtered, self.f1)
        else:
            holo_ft = self.hologram_FT
        
        best_focus = None
        best_range = -np.inf
        
        for z in z_values:
            Pz = generate_propagator(
                self.N, self.lambda_val, self.area, z,
                self.alpha, self.beta, self.mask,
                method=self.propagation_method
            )
            field_prop = IFT2Dc(holo_ft * Pz, self.f1)
            phase = np.angle(field_prop)
            uwp = unwrap_phase(-phase)
            rng = uwp.max() - uwp.min()
            if rng > best_range:
                best_range = rng
                best_focus = z
        
        QMessageBox.information(
            self, "AutoFocus Results",
            f"Best focus = {best_focus*1e6:.2f} µm\n"
            f"Phase range = {best_range:.4f} rad"
        )
        
        self.spin_zmanual.setValue(best_focus)
        self.run_reconstruction()
    
    def run_reconstruction(self):
        """Perform direct and iterative reconstructions at the chosen z."""
        if self.hologram is None:
            QMessageBox.warning(self, "Warning", "No hologram loaded.")
            return
        
        if self.hologram_filtered is not None:
            holo = self.hologram_filtered
            holo_ft = FT2Dc(holo, self.f1)
        else:
            holo = self.hologram
            holo_ft = self.hologram_FT
        
        z = self.spin_zmanual.value()
        
        # Create forward/backward propagators
        Pz_forward = generate_propagator(
            self.N, self.lambda_val, self.area, z,
            self.alpha, self.beta, self.mask,
            method=self.propagation_method
        )
        Pz_backward = generate_propagator(
            self.N, self.lambda_val, self.area, -z,
            self.alpha, self.beta, self.mask,
            method=self.propagation_method
        )
        
        # Direct reconstruction
        field_prop = IFT2Dc(holo_ft * Pz_forward, self.f1)
        self.amp_direct = np.abs(field_prop)
        self.phase_direct = np.angle(field_prop)
        self.unwrap_direct = unwrap_phase(-self.phase_direct)
        
        # Iterative reconstruction
        holo_amp = np.sqrt(holo)
        
        max_iter = self.spin_iter.value()
        pos_thresh = self.spin_pos_thresh.value()
        supp_thresh = self.spin_supp_thresh.value()
        dilation_radius = self.spin_dilation.value()
        adapt_thresh = self.check_adaptive_thresh.isChecked()
        
        Eobj_iter = iterative_phase_retrieval(
            holo_amp,
            self.f1,
            Pz_forward,
            Pz_backward,
            max_iter=max_iter,
            positivity_enforce=(pos_thresh > 0.0),
            support_threshold=supp_thresh,
            dilation_radius=dilation_radius,
            adaptive_threshold=adapt_thresh
        )
        
        # Optional 3D Deconvolution
        if self.check_3d_deconv.isChecked():
            Eobj_iter = apply_3d_deconvolution(Eobj_iter, wiener_param=1e-3)
        
        self.amp_iter = np.abs(Eobj_iter)
        self.phase_iter = np.angle(Eobj_iter)
        self.unwrap_iter = unwrap_phase(-self.phase_iter)
        
        # Update the 2×3 matplotlib plots
        self.update_plots(z)
    
    def update_plots(self, z):
        # Clear old images
        for ax in [
            self.canvas.ax_amp_direct,
            self.canvas.ax_wp_direct,
            self.canvas.ax_uwp_direct,
            self.canvas.ax_amp_iter,
            self.canvas.ax_wp_iter,
            self.canvas.ax_uwp_iter
        ]:
            ax.clear()
        
        # Remove colorbars
        for cb in self.colorbars:
            try:
                cb.remove()
            except:
                pass
        self.colorbars = []
        
        # ---------- Top Row (Direct) -----------
        im_amp_d = self.canvas.ax_amp_direct.imshow(self.amp_direct, cmap='gray')
        self.canvas.ax_amp_direct.set_title(f"Direct Amplitude @ z={1e6*z:.2f} µm")
        self.canvas.ax_amp_direct.axis("off")
        cb_d1 = self.canvas.fig.colorbar(im_amp_d, ax=self.canvas.ax_amp_direct,
                                         fraction=0.046, pad=0.04)
        self.colorbars.append(cb_d1)
        
        im_wp_d = self.canvas.ax_wp_direct.imshow(self.phase_direct, cmap='jet')
        prange_d = self.phase_direct.max() - self.phase_direct.min()
        self.canvas.ax_wp_direct.set_title(f"Direct Wrapped\nRange={prange_d:.2f}")
        self.canvas.ax_wp_direct.axis("off")
        cb_d2 = self.canvas.fig.colorbar(im_wp_d, ax=self.canvas.ax_wp_direct,
                                         fraction=0.046, pad=0.04)
        self.colorbars.append(cb_d2)
        
        im_uwp_d = self.canvas.ax_uwp_direct.imshow(self.unwrap_direct, cmap='jet')
        urange_d = self.unwrap_direct.max() - self.unwrap_direct.min()
        self.canvas.ax_uwp_direct.set_title(f"Direct Unwrapped\nRange={urange_d:.2f}")
        self.canvas.ax_uwp_direct.axis("off")
        cb_d3 = self.canvas.fig.colorbar(im_uwp_d, ax=self.canvas.ax_uwp_direct,
                                         fraction=0.046, pad=0.04)
        self.colorbars.append(cb_d3)
        
        # ---------- Bottom Row (Iterative) -----------
        im_amp_i = self.canvas.ax_amp_iter.imshow(self.amp_iter, cmap='gray')
        self.canvas.ax_amp_iter.set_title("Iterative Amplitude")
        self.canvas.ax_amp_iter.axis("off")
        cb_i1 = self.canvas.fig.colorbar(im_amp_i, ax=self.canvas.ax_amp_iter,
                                         fraction=0.046, pad=0.04)
        self.colorbars.append(cb_i1)
        
        im_wp_i = self.canvas.ax_wp_iter.imshow(self.phase_iter, cmap='jet')
        prange_i = self.phase_iter.max() - self.phase_iter.min()
        self.canvas.ax_wp_iter.set_title(f"Iter Wrapped\nRange={prange_i:.2f}")
        self.canvas.ax_wp_iter.axis("off")
        cb_i2 = self.canvas.fig.colorbar(im_wp_i, ax=self.canvas.ax_wp_iter,
                                         fraction=0.046, pad=0.04)
        self.colorbars.append(cb_i2)
        
        im_uwp_i = self.canvas.ax_uwp_iter.imshow(self.unwrap_iter, cmap='jet')
        urange_i = self.unwrap_iter.max() - self.unwrap_iter.min()
        self.canvas.ax_uwp_iter.set_title(f"Iter Unwrapped\nRange={urange_i:.2f}")
        self.canvas.ax_uwp_iter.axis("off")
        cb_i3 = self.canvas.fig.colorbar(im_uwp_i, ax=self.canvas.ax_uwp_iter,
                                         fraction=0.046, pad=0.04)
        self.colorbars.append(cb_i3)
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def show_line_profiles(self):
        """Open a separate window with 2×2 line profile plots."""
        if self.amp_direct is None or self.amp_iter is None:
            QMessageBox.warning(self, "Warning", "No reconstruction to plot. Please reconstruct first.")
            return
        
        self.line_profile_window = LineProfileWindow(
            self.amp_direct, self.phase_direct,
            self.amp_iter, self.phase_iter,
            self.unwrap_direct, self.unwrap_iter
        )
        self.line_profile_window.show()

###############################################################################
# 8) Main Entry Point
###############################################################################

def main():
    app = QApplication(sys.argv)
    window = HoloReconstructionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
