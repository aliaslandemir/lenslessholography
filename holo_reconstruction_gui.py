#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Inline Holography Reconstruction GUI
=============================================
- PyQt5 + Matplotlib, single-file GUI
- Direct propagation (Angular Spectrum / Fresnel)
- Iterative phase retrieval (GS / HIO) with positivity & finite support
- Adaptive support threshold and morphological dilation
- Twin-image (sideband) filter (naive band-pass)
- Autofocus (metric: unwrapped phase range OR Laplacian variance sharpness)
- Linked, publication-grade 3x4 figure:
    Row 1: Direct — Amp / Wrapped / Unwrapped / Fourier |E| of object
    Row 2: Iterative — Amp / Wrapped / Unwrapped / Fourier |E| of object
    Row 3: Differences — ΔAmp / ΔWrapped / ΔUnwrapped / Phase histogram (iter)
- Interactive line profiles: click anywhere to update overlaid row/column profiles
- Metrics: RMSE (amp), SSIM (amp), Phase corr. (unwrapped), support coverage
- Export: PNG/TIFF figure, NPZ fields, JSON params, CSV metrics

Dependencies:
  Python >= 3.8
  numpy, matplotlib, pillow, scikit-image, PyQt5

Run:
  python advanced_holography_gui.py
"""

import json
import sys
import time
import math
import numpy as np
from dataclasses import dataclass, asdict

from PIL import Image
from skimage.restoration import unwrap_phase
from skimage.morphology import binary_dilation, disk
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGroupBox, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDoubleSpinBox, QSpinBox, QMessageBox,
    QCheckBox, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem, QDockWidget,
    QLineEdit
)

# -------------------------------
# FFT helpers (centered versions)
# -------------------------------

def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(X):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

# -------------------------------
# Propagators
# -------------------------------

def make_freq_grids(N, dx):
    fx = np.fft.fftfreq(N, d=dx)  # cycles/m
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    return FX, FY

def angular_spectrum_kernel(N, wavelength, dx, z, evanescent="zero"):
    """
    H(fx,fy) = exp(i * kz * z), kz = 2π * sqrt( (1/λ)^2 - fx^2 - fy^2 )
    If evanescent='zero', drop evanescent components (where fx^2+fy^2 > (1/λ)^2).
    """
    FX, FY = make_freq_grids(N, dx)
    k = 2 * np.pi / wavelength
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    k_cut2 = (1.0 / wavelength) ** 2

    mask_prop = (FX**2 + FY**2) <= k_cut2
    kz = np.zeros_like(FX, dtype=np.complex128)
    kz[mask_prop] = np.sqrt(np.maximum(0.0, k**2 - kx[mask_prop]**2 - ky[mask_prop]**2))
    if evanescent == "keep":
        # decay for evanescent (optional): kz becomes imaginary
        ev_mask = ~mask_prop
        kz[ev_mask] = 1j * np.sqrt(kx[ev_mask]**2 + ky[ev_mask]**2 - k**2)
    else:
        # zero-out evanescent
        pass

    H = np.exp(1j * kz * z)
    if evanescent != "keep":
        H[~mask_prop] = 0.0
    return H

def fresnel_kernel(N, wavelength, dx, z):
    """
    H(fx,fy) = exp(i*k*z) * exp(-i*pi*λ*z*(fx^2+fy^2))
    """
    k = 2 * np.pi / wavelength
    FX, FY = make_freq_grids(N, dx)
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    return H

# -------------------------------
# Filters / preprocessing
# -------------------------------

def twin_image_sideband_filter(img, rmin=5, rmax_frac=0.45):
    """
    Naive radial band-pass in Fourier domain to suppress DC & very-high-freqs.
    """
    X = fft2c(img)
    N, M = img.shape
    cy, cx = N // 2, M // 2
    yy, xx = np.ogrid[:N, :M]
    r2 = (yy - cy)**2 + (xx - cx)**2
    rmax = int(min(cy, cx) * rmax_frac)
    mask = (r2 >= rmin**2) & (r2 <= rmax**2)
    Xf = np.zeros_like(X)
    Xf[mask] = X[mask]
    return np.abs(ifft2c(Xf))

# -------------------------------
# Support mask
# -------------------------------

def make_support(amp, mode="percentile", thresh_val=10.0, percentile=90.0, dilation=2):
    """
    mode:
      - "percentile": threshold at given percentile of amplitude
      - "otsu": Otsu threshold on amplitude
      - "fixed": fixed absolute threshold (thresh_val)
    """
    if mode == "percentile":
        t = np.percentile(amp, percentile)
    elif mode == "otsu":
        t = threshold_otsu(amp / (amp.max() + 1e-12)) * (amp.max() + 1e-12)
    elif mode == "fixed":
        t = float(thresh_val)
    else:
        t = np.percentile(amp, 90.0)

    mask = amp > t
    if dilation > 0:
        mask = binary_dilation(mask, disk(int(dilation)))
    return mask, float(t)

# -------------------------------
# Autofocus metrics
# -------------------------------

def laplacian_variance(image):
    # simple focus proxy on amplitude; expects float
    # using numpy 2D laplacian kernel
    kern = np.array([[0, 1, 0],
                     [1,-4, 1],
                     [0, 1, 0]], dtype=np.float32)
    from scipy.signal import convolve2d
    L = convolve2d(image, kern, mode='same', boundary='symm')
    return float(L.var())

def autofocus_metric(field, metric="phase_range"):
    """
    field: complex object-plane field
    """
    if metric == "phase_range":
        ph = np.angle(field)
        uw = unwrap_phase(ph)
        return float(uw.max() - uw.min())
    else:
        amp = np.abs(field)
        return laplacian_variance(amp)

# -------------------------------
# Iterative Retrieval (GS / HIO)
# -------------------------------

@dataclass
class IterSettings:
    method: str = "GS"          # "GS" or "HIO"
    max_iter: int = 50
    beta: float = 0.9           # only for HIO
    positivity: bool = True
    support_mode: str = "percentile"  # "percentile"|"otsu"|"fixed"
    support_thresh: float = 10.0
    support_percentile: float = 90.0
    dilation: int = 2
    adaptive_support: bool = True
    adaptive_rate: float = 0.97  # multiplicative decay per iter for threshold

def iterative_reconstruction(holo_amp, Hfwd, Hbwd, settings: IterSettings):
    """
    holo_amp: measured amplitude at hologram plane (sqrt of hologram intensity)
    Hfwd/Hbwd: transfer functions for forward/backward
    """
    N, M = holo_amp.shape
    rng = np.random.default_rng(0)
    phase0 = rng.uniform(-np.pi, np.pi, size=(N, M))
    Eholo = holo_amp * np.exp(1j * phase0)

    Eobj_prev = None
    support = None
    support_thresh = settings.support_thresh

    # precompute support every iter
    for it in range(settings.max_iter):
        # hologram -> object
        Eobj = ifft2c(fft2c(Eholo) * Hfwd)

        # constraints in object plane
        amp = np.abs(Eobj)
        pha = np.angle(Eobj)

        if support is None:
            support, used_t = make_support(
                amp,
                mode=settings.support_mode,
                thresh_val=settings.support_thresh,
                percentile=settings.support_percentile,
                dilation=settings.dilation
            )
            support_thresh = used_t

        # optional adaptive support (gradually relax threshold)
        if settings.adaptive_support and (it > 0) and (settings.support_mode != "fixed"):
            if settings.support_mode == "percentile":
                # slowly lower percentile toward 80
                settings.support_percentile = max(80.0, settings.support_percentile * settings.adaptive_rate)
            else:
                support_thresh *= settings.adaptive_rate
            support, _ = make_support(
                amp,
                mode=settings.support_mode,
                thresh_val=support_thresh,
                percentile=settings.support_percentile,
                dilation=settings.dilation
            )

        # positivity (on amplitude – trivial non-negativity) + zero outside support
        if settings.method.upper() == "GS":
            if settings.positivity:
                amp = np.clip(amp, 0, None)
            amp[~support] = 0.0
            Eobj_new = amp * np.exp(1j * pha)

        elif settings.method.upper() == "HIO":
            if Eobj_prev is None:
                Eobj_prev = Eobj.copy()
            # Inside support: enforce positivity
            Ein = Eobj.copy()
            if settings.positivity:
                Ein = np.clip(np.abs(Ein), 0, None) * np.exp(1j * np.angle(Ein))
            # HIO outside support
            Eout = Eobj_prev[~support] - settings.beta * Eobj[~support]
            Eobj_new = Eobj.copy()
            Eobj_new[support] = Ein[support]
            Eobj_new[~support] = Eout
            Eobj_prev = Eobj_new.copy()

        else:
            raise ValueError("Unknown iterative method")

        # object -> hologram and enforce measured amplitude
        Eholo_back = ifft2c(fft2c(Eobj_new) * Hbwd)
        Eholo = holo_amp * np.exp(1j * np.angle(Eholo_back))

    return Eobj_new, support

# -------------------------------
# Metrics
# -------------------------------

def safe_norm(a):
    m = a.max() - a.min()
    if m <= 1e-12:
        return a*0
    return (a - a.min()) / m

def compute_metrics(amp_d, amp_i, uwp_d, uwp_i, support):
    # RMSE amplitude
    rmse = float(np.sqrt(np.mean((amp_d - amp_i)**2)))
    # SSIM amplitude (normalize to [0,1])
    A1 = safe_norm(amp_d)
    A2 = safe_norm(amp_i)
    ssim_val = float(ssim(A1, A2, data_range=1.0))
    # Phase correlation on unwrapped phase (mask finite & support)
    mask = np.isfinite(uwp_d) & np.isfinite(uwp_i)
    if support is not None:
        mask &= support
    if mask.sum() > 10:
        p1 = uwp_d[mask].ravel()
        p2 = uwp_i[mask].ravel()
        # Pearson correlation
        p1 = (p1 - p1.mean()) / (p1.std() + 1e-12)
        p2 = (p2 - p2.mean()) / (p2.std() + 1e-12)
        corr = float(np.mean(p1 * p2))
    else:
        corr = float("nan")
    cov = float(mask.mean()) if mask.size else 0.0
    sup_cov = float(support.mean()) if support is not None else float("nan")
    return {
        "rmse_amp": rmse,
        "ssim_amp": ssim_val,
        "phase_corr_unwrapped": corr,
        "valid_mask_coverage": cov,
        "support_coverage": sup_cov,
    }

# -------------------------------
# Main Canvas (3x4 grid + interactivity)
# -------------------------------

class ReconCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=9, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        axs = self.fig.subplots(3, 4, sharex=True, sharey=True)
        self.axs = axs
        super().__init__(self.fig)
        self.setParent(parent)

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self._profiles_callback = None
        self._last_images = {}  # store images for colorbars etc.
        self.colorbars = []

    def clear(self):
        for ax in self.axs.flat:
            ax.clear()
        for cb in self.colorbars:
            try:
                cb.remove()
            except Exception:
                pass
        self.colorbars = []
        self.draw()

    def set_profiles_callback(self, fn):
        self._profiles_callback = fn

    def on_click(self, event):
        # Report click in data coords to callback (for line profiles)
        if event.inaxes in self.axs.flat:
            if self._profiles_callback is not None:
                self._profiles_callback(int(round(event.xdata)), int(round(event.ydata)))

    def _show(self, ax, img, title="", cmap="gray", add_cb=True, vmin=None, vmax=None):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if add_cb:
            cb = self.fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
            self.colorbars.append(cb)
        return im

    def plot_all(self, D_amp, D_wr, D_uw,
                 I_amp, I_wr, I_uw,
                 diff_amp, diff_wr, diff_uw,
                 fftD_mag, fftI_mag, phase_hist,
                 link_color=True):

        self.clear()

        # Optional linked scales
        vA = (D_amp.min(), D_amp.max()) if not link_color else (
            min(D_amp.min(), I_amp.min()), max(D_amp.max(), I_amp.max())
        )
        vW = (-np.pi, np.pi)
        vU = (min(D_uw.min(), I_uw.min()), max(D_uw.max(), I_uw.max())) if link_color else (None, None)

        # Row 1: Direct
        self._show(self.axs[0,0], D_amp, "Direct Amplitude", "gray", True, *vA)
        self._show(self.axs[0,1], D_wr, "Direct Wrapped Phase", "twilight", True, *vW)
        self._show(self.axs[0,2], D_uw, "Direct Unwrapped Phase", "twilight", True, *vU)
        self._show(self.axs[0,3], fftD_mag, "Direct |FFT(E)|", "magma", True, None, None)

        # Row 2: Iterative
        self._show(self.axs[1,0], I_amp, "Iterative Amplitude", "gray", True, *vA)
        self._show(self.axs[1,1], I_wr, "Iterative Wrapped Phase", "twilight", True, *vW)
        self._show(self.axs[1,2], I_uw, "Iterative Unwrapped Phase", "twilight", True, *vU)
        self._show(self.axs[1,3], fftI_mag, "Iterative |FFT(E)|", "magma", True, None, None)

        # Row 3: Differences + histogram
        self._show(self.axs[2,0], diff_amp, "ΔAmplitude (Iter-Direct)", "bwr", True, None, None)
        self._show(self.axs[2,1], diff_wr, "ΔWrapped Phase (wrap)", "twilight", True, -np.pi, np.pi)
        self._show(self.axs[2,2], diff_uw, "ΔUnwrapped Phase", "coolwarm", True, None, None)

        # Phase histogram (iterative)
        axh = self.axs[2,3]
        axh.clear()
        axh.hist(phase_hist.ravel(), bins=80)
        axh.set_title("Iterative Phase Histogram")
        axh.set_xlabel("Phase [rad]")
        axh.set_ylabel("Count")
        axh.grid(True)

        self.draw()

# -------------------------------
# Main App
# -------------------------------

class HoloGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Inline Holography Reconstruction")

        # Data
        self.hologram = None  # float 2D (intensity)
        self.filtered = None
        self.N = 256
        self.area = 1.5e-3  # meters (FOV length), square assumed
        self.wavelength = 530e-9
        self.z = 5e-3

        # Reconstruction fields
        self.direct_field = None
        self.iter_field = None
        self.support = None

        # GUI layout
        self.canvas = ReconCanvas(self, width=12, height=9)
        self.setCentralWidget(self.canvas)

        # Controls (tabs in a dock)
        self.ctrl_dock = QDockWidget("Controls", self)
        self.ctrl_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.ctrl_dock)

        self.tabs = QTabWidget()
        self.ctrl_dock.setWidget(self.tabs)

        self._build_tab_input()
        self._build_tab_prop()
        self._build_tab_iter()
        self._build_tab_viz_export()

        # Metrics dock
        self.metrics_dock = QDockWidget("Metrics", self)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(self.metrics_table)
        self.metrics_dock.setWidget(w)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.metrics_dock)

        # Profiles dock
        self.profiles_dock = QDockWidget("Line Profiles", self)
        self.profiles_fig = Figure(figsize=(5,4), dpi=100)
        self.profiles_canvas = FigureCanvas(self.profiles_fig)
        w2 = QWidget()
        v2 = QVBoxLayout(w2)
        v2.addWidget(self.profiles_canvas)
        self.profiles_dock.setWidget(w2)
        self.addDockWidget(Qt.RightDockWidgetArea, self.profiles_dock)

        self.canvas.set_profiles_callback(self.update_profiles)

    # ---------------- UI Tabs ----------------

    def _build_tab_input(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        # load
        btn_load = QPushButton("Load Hologram")
        btn_load.clicked.connect(self.load_hologram)

        # extrapolation
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Extrapolation factor"))
        self.spin_extrap = QSpinBox()
        self.spin_extrap.setRange(1, 8)
        self.spin_extrap.setValue(1)
        hl.addWidget(self.spin_extrap)

        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Extrapolation mode"))
        self.combo_extrap = QComboBox()
        self.combo_extrap.addItems(["Zeros", "Random"])
        hl2.addWidget(self.combo_extrap)

        # filter
        btn_filter = QPushButton("Twin-Image Filter")
        btn_filter.clicked.connect(self.apply_filter)

        # fields
        lay.addWidget(btn_load)
        lay.addLayout(hl)
        lay.addLayout(hl2)
        lay.addWidget(btn_filter)
        lay.addStretch()

        self.tabs.addTab(tab, "Input")

    def _build_tab_prop(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        # wavelength
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Wavelength [nm]"))
        self.spin_lambda = QDoubleSpinBox()
        self.spin_lambda.setDecimals(3)
        self.spin_lambda.setRange(200.0, 2000.0)
        self.spin_lambda.setValue(self.wavelength*1e9)
        h1.addWidget(self.spin_lambda)

        # area
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("FOV side length [mm]"))
        self.spin_area = QDoubleSpinBox()
        self.spin_area.setDecimals(6)
        self.spin_area.setRange(0.01, 100.0)
        self.spin_area.setValue(self.area*1e3)
        h2.addWidget(self.spin_area)

        # N
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Grid size N (square)"))
        self.spin_N = QSpinBox()
        self.spin_N.setRange(64, 4096)
        self.spin_N.setValue(self.N)
        h3.addWidget(self.spin_N)

        # z manual
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("z [mm]"))
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setDecimals(6)
        self.spin_z.setRange(0.001, 100.0)
        self.spin_z.setValue(self.z*1e3)
        h4.addWidget(self.spin_z)

        # propagator
        h5 = QHBoxLayout()
        h5.addWidget(QLabel("Propagation method"))
        self.combo_prop = QComboBox()
        self.combo_prop.addItems(["Angular Spectrum", "Fresnel"])
        h5.addWidget(self.combo_prop)

        # autofocus controls
        self.combo_focus_metric = QComboBox()
        self.combo_focus_metric.addItems(["Unwrapped phase range", "Laplacian variance"])
        h6 = QHBoxLayout()
        h6.addWidget(QLabel("Autofocus metric"))
        h6.addWidget(self.combo_focus_metric)

        self.spin_zmin = QDoubleSpinBox(); self.spin_zmin.setDecimals(6); self.spin_zmin.setRange(0.001, 100.0); self.spin_zmin.setValue(0.05)
        self.spin_zmax = QDoubleSpinBox(); self.spin_zmax.setDecimals(6); self.spin_zmax.setRange(0.001, 100.0); self.spin_zmax.setValue(10.0)
        self.spin_zstep= QDoubleSpinBox(); self.spin_zstep.setDecimals(6); self.spin_zstep.setRange(0.0001, 10.0); self.spin_zstep.setValue(0.05)
        hz = QHBoxLayout()
        hz.addWidget(QLabel("z min [mm]")); hz.addWidget(self.spin_zmin)
        hz.addWidget(QLabel("z max [mm]")); hz.addWidget(self.spin_zmax)
        hz.addWidget(QLabel("z step [mm]")); hz.addWidget(self.spin_zstep)

        btn_autofocus = QPushButton("Autofocus (scan)")
        btn_autofocus.clicked.connect(self.autofocus)

        # reconstruct button
        btn_recon = QPushButton("Reconstruct @ z")
        btn_recon.clicked.connect(self.reconstruct)

        lay.addLayout(h1); lay.addLayout(h2); lay.addLayout(h3)
        lay.addLayout(h4); lay.addLayout(h5); lay.addLayout(h6); lay.addLayout(hz)
        lay.addWidget(btn_autofocus)
        lay.addWidget(btn_recon)
        lay.addStretch()
        self.tabs.addTab(tab, "Propagation")

    def _build_tab_iter(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        # method
        h0 = QHBoxLayout()
        h0.addWidget(QLabel("Iterative method"))
        self.combo_iter_method = QComboBox()
        self.combo_iter_method.addItems(["GS", "HIO"])
        h0.addWidget(self.combo_iter_method)

        # iter count, beta
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Max iterations"))
        self.spin_iters = QSpinBox(); self.spin_iters.setRange(1, 2000); self.spin_iters.setValue(50)
        h1.addWidget(self.spin_iters)

        h1b = QHBoxLayout()
        h1b.addWidget(QLabel("HIO beta"))
        self.spin_beta = QDoubleSpinBox(); self.spin_beta.setDecimals(3); self.spin_beta.setRange(0.0, 1.0); self.spin_beta.setValue(0.9)
        h1b.addWidget(self.spin_beta)

        # positivity
        self.chk_pos = QCheckBox("Enforce positivity (amplitude)")
        self.chk_pos.setChecked(True)

        # support
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Support mode"))
        self.combo_support = QComboBox()
        self.combo_support.addItems(["percentile", "otsu", "fixed"])
        h2.addWidget(self.combo_support)

        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Support percentile / fixed thr."))
        self.spin_support_val = QDoubleSpinBox(); self.spin_support_val.setDecimals(3)
        self.spin_support_val.setRange(0.0, 100000.0); self.spin_support_val.setValue(90.0)
        h3.addWidget(self.spin_support_val)

        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Dilation radius (px)"))
        self.spin_dilation = QSpinBox(); self.spin_dilation.setRange(0, 100); self.spin_dilation.setValue(2)
        h4.addWidget(self.spin_dilation)

        self.chk_adapt = QCheckBox("Adaptive support threshold")
        self.chk_adapt.setChecked(True)

        lay.addLayout(h0); lay.addLayout(h1); lay.addLayout(h1b)
        lay.addWidget(self.chk_pos)
        lay.addLayout(h2); lay.addLayout(h3); lay.addLayout(h4)
        lay.addWidget(self.chk_adapt)
        lay.addStretch()
        self.tabs.addTab(tab, "Iterative")

    def _build_tab_viz_export(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        self.chk_link_color = QCheckBox("Link color scales (amp/phase)")
        self.chk_link_color.setChecked(True)

        self.chk_save_npz = QCheckBox("Save NPZ fields on export")
        self.chk_save_npz.setChecked(True)

        self.chk_save_params = QCheckBox("Save JSON params on export")
        self.chk_save_params.setChecked(True)

        self.chk_save_metrics = QCheckBox("Save CSV metrics on export")
        self.chk_save_metrics.setChecked(True)

        hfmt = QHBoxLayout()
        hfmt.addWidget(QLabel("Export base name"))
        self.edit_basename = QLineEdit("recon")
        hfmt.addWidget(self.edit_basename)

        btn_export = QPushButton("Export Figure + Data")
        btn_export.clicked.connect(self.export_all)

        lay.addWidget(self.chk_link_color)
        lay.addWidget(self.chk_save_npz)
        lay.addWidget(self.chk_save_params)
        lay.addWidget(self.chk_save_metrics)
        lay.addLayout(hfmt)
        lay.addWidget(btn_export)
        lay.addStretch()
        self.tabs.addTab(tab, "Viz & Export")

    # --------------- Actions ----------------

    def load_hologram(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Hologram", "", "Images (*.png *.jpg *.bmp *.tif *.tiff)")
        if not fname:
            return
        img = np.array(Image.open(fname)).astype(np.float64)
        if img.ndim == 3:
            img = img[..., 0]  # take first channel
        # normalize to [0, max]
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        # extrapolate if needed
        factor = int(self.spin_extrap.value())
        mode = self.combo_extrap.currentText().lower()
        Nx, Ny = img.shape
        N = int(np.ceil(max(Nx, Ny) * factor / 2) * 2)  # even
        pad = np.zeros((N, N), dtype=np.float64)
        if mode == "random":
            pad[:] = np.random.RandomState(0).rand(N, N) * img.mean()
        sx = (N - Nx)//2; sy = (N - Ny)//2
        pad[sx:sx+Nx, sy:sy+Ny] = img
        self.hologram = pad
        self.filtered = None
        # update internal params
        self.N = int(self.spin_N.value())
        # snap N to hologram size if extrapolated
        self.N = max(self.N, N)
        self.spin_N.setValue(self.N)
        QMessageBox.information(self, "Loaded", f"Loaded hologram: {fname}\nShape: {img.shape} -> padded {pad.shape}")

    def apply_filter(self):
        if self.hologram is None:
            QMessageBox.warning(self, "Warning", "Load a hologram first.")
            return
        self.filtered = twin_image_sideband_filter(self.hologram)
        QMessageBox.information(self, "Filter", "Twin-image filtering done.")

    def autofocus(self):
        if self.hologram is None:
            QMessageBox.warning(self, "Warning", "Load a hologram first.")
            return
        use = self.filtered if self.filtered is not None else self.hologram
        N = int(self.spin_N.value())
        L = float(self.spin_area.value()) * 1e-3
        dx = L / N
        lam = float(self.spin_lambda.value()) * 1e-9

        method = self.combo_prop.currentText()
        metric_name = self.combo_focus_metric.currentText()

        zmin = float(self.spin_zmin.value()) * 1e-3
        zmax = float(self.spin_zmax.value()) * 1e-3
        zstep = float(self.spin_zstep.value()) * 1e-3
        zs = np.arange(zmin, zmax + 0.5*zstep, zstep)

        # choose kernel maker
        def kernel(z):
            if method == "Angular Spectrum":
                return angular_spectrum_kernel(N, lam, dx, z, evanescent="zero")
            else:
                return fresnel_kernel(N, lam, dx, z)

        X = fft2c(resize_or_pad(use, (N, N)))
        best_val, best_z = -np.inf, None

        for z in zs:
            H = kernel(z)
            field = ifft2c(X * H)
            val = autofocus_metric(field, "phase_range" if "phase" in metric_name.lower() else "laplacian")
            if val > best_val:
                best_val, best_z = val, z

        self.z = best_z
        self.spin_z.setValue(best_z * 1e3)
        QMessageBox.information(self, "Autofocus",
                                f"Best z = {best_z*1e3:.3f} mm | metric = {best_val:.4f}")
        self.reconstruct()

    def reconstruct(self):
        if self.hologram is None:
            QMessageBox.warning(self, "Warning", "Load a hologram first.")
            return

        # params
        self.N = int(self.spin_N.value())
        self.area = float(self.spin_area.value()) * 1e-3
        self.wavelength = float(self.spin_lambda.value()) * 1e-9
        self.z = float(self.spin_z.value()) * 1e-3
        dx = self.area / self.N

        # select image
        holo = self.filtered if self.filtered is not None else self.hologram
        Himg = resize_or_pad(holo, (self.N, self.N))
        HFT = fft2c(Himg)

        # kernels
        if self.combo_prop.currentText() == "Angular Spectrum":
            Hfwd = angular_spectrum_kernel(self.N, self.wavelength, dx, self.z, evanescent="zero")
            Hbwd = angular_spectrum_kernel(self.N, self.wavelength, dx, -self.z, evanescent="zero")
        else:
            Hfwd = fresnel_kernel(self.N, self.wavelength, dx, self.z)
            Hbwd = fresnel_kernel(self.N, self.wavelength, dx, -self.z)

        # direct
        Fdir = ifft2c(HFT * Hfwd)
        amp_d = np.abs(Fdir)
        wrp_d = np.angle(Fdir)
        uwp_d = unwrap_phase(wrp_d)

        # iterative settings
        itset = IterSettings(
            method=self.combo_iter_method.currentText(),
            max_iter=int(self.spin_iters.value()),
            beta=float(self.spin_beta.value()),
            positivity=self.chk_pos.isChecked(),
            support_mode=self.combo_support.currentText(),
            support_thresh=float(self.spin_support_val.value()),
            support_percentile=float(self.spin_support_val.value()),
            dilation=int(self.spin_dilation.value()),
            adaptive_support=self.chk_adapt.isChecked(),
            adaptive_rate=0.97
        )

        holo_amp = np.sqrt(np.clip(Himg, 0, None))
        Fiter, sup = iterative_reconstruction(holo_amp, Hfwd, Hbwd, itset)
        self.support = sup
        amp_i = np.abs(Fiter)
        wrp_i = np.angle(Fiter)
        uwp_i = unwrap_phase(wrp_i)

        # diffs
        diff_amp = amp_i - amp_d
        diff_wr = wrap_to_pi(wrp_i - wrp_d)
        diff_uw = uwp_i - uwp_d

        # FFT magnitudes (object plane)
        fftD_mag = np.log1p(np.abs(fft2c(Fdir)))
        fftI_mag = np.log1p(np.abs(fft2c(Fiter)))

        # Plot grid
        self.canvas.plot_all(
            amp_d, wrp_d, uwp_d,
            amp_i, wrp_i, uwp_i,
            diff_amp, diff_wr, diff_uw,
            fftD_mag, fftI_mag, wrp_i,
            link_color=self.chk_link_color.isChecked()
        )

        # store fields
        self.direct_field = Fdir
        self.iter_field = Fiter

        # metrics
        m = compute_metrics(amp_d, amp_i, uwp_d, uwp_i, self.support)
        self.update_metrics_table(m)

    def update_metrics_table(self, metrics_dict):
        self.metrics_table.setRowCount(0)
        for k, v in metrics_dict.items():
            r = self.metrics_table.rowCount()
            self.metrics_table.insertRow(r)
            self.metrics_table.setItem(r, 0, QTableWidgetItem(k))
            self.metrics_table.setItem(r, 1, QTableWidgetItem(f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)))

    def update_profiles(self, x, y):
        # Draw line profiles through (y, x)
        if self.direct_field is None or self.iter_field is None:
            return
        D = self.direct_field
        I = self.iter_field
        N, M = D.shape
        x = np.clip(x, 0, M-1)
        y = np.clip(y, 0, N-1)

        # Extract profiles
        Ad = np.abs(D); Ai = np.abs(I)
        Pd = np.angle(D); Pi = np.angle(I)
        Ud = unwrap_phase(Pd); Ui = unwrap_phase(Pi)

        row = slice(None)
        col = slice(None)
        amp_row_d = Ad[y, :]
        amp_row_i = Ai[y, :]
        amp_col_d = Ad[:, x]
        amp_col_i = Ai[:, x]

        # Re-plot small figure: 2x2 (row/col for amplitude & unwrapped)
        self.profiles_fig.clf()
        ax1 = self.profiles_fig.add_subplot(2,2,1); ax2 = self.profiles_fig.add_subplot(2,2,2)
        ax3 = self.profiles_fig.add_subplot(2,2,3); ax4 = self.profiles_fig.add_subplot(2,2,4)

        ax1.plot(amp_row_d, label="Dir"); ax1.plot(amp_row_i, '--', label="Iter")
        ax1.set_title(f"Amp Row y={y}"); ax1.legend(); ax1.grid(True)
        ax2.plot(amp_col_d, label="Dir"); ax2.plot(amp_col_i, '--', label="Iter")
        ax2.set_title(f"Amp Col x={x}"); ax2.legend(); ax2.grid(True)

        ax3.plot(Ud[y, :], label="Dir"); ax3.plot(Ui[y, :], '--', label="Iter")
        ax3.set_title("Unwrapped Row"); ax3.legend(); ax3.grid(True)
        ax4.plot(Ud[:, x], label="Dir"); ax4.plot(Ui[:, x], '--', label="Iter")
        ax4.set_title("Unwrapped Col"); ax4.legend(); ax4.grid(True)

        self.profiles_fig.tight_layout()
        self.profiles_canvas.draw()

    def export_all(self):
        if self.direct_field is None or self.iter_field is None:
            QMessageBox.warning(self, "Warning", "Reconstruct first.")
            return
        base = self.edit_basename.text().strip() or "recon"
        # figure
        fig_path = ask_save(self, f"{base}.png", "PNG (*.png);;TIFF (*.tif *.tiff)")
        if not fig_path:
            return
        self.canvas.fig.savefig(fig_path, dpi=300)
        # npz
        if self.chk_save_npz.isChecked():
            np.savez_compressed(
                fpath_with_new_ext(fig_path, ".npz"),
                direct=self.direct_field.astype(np.complex64),
                iterative=self.iter_field.astype(np.complex64),
                support=self.support.astype(np.uint8) if self.support is not None else None
            )
        # params
        if self.chk_save_params.isChecked():
            params = {
                "N": self.N,
                "area_m": self.area,
                "dx_m": self.area / self.N,
                "wavelength_m": self.wavelength,
                "z_m": self.z,
                "propagation": self.combo_prop.currentText()
            }
            with open(fpath_with_new_ext(fig_path, ".json"), "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2)
        # metrics
        if self.chk_save_metrics.isChecked():
            # reconstruct metrics dict from table
            metrics = {}
            for r in range(self.metrics_table.rowCount()):
                k = self.metrics_table.item(r, 0).text()
                v = self.metrics_table.item(r, 1).text()
                metrics[k] = v
            save_csv_dict(metrics, fpath_with_new_ext(fig_path, ".csv"))
        QMessageBox.information(self, "Export", f"Saved outputs with base: {fig_path}")

# -------------------------------
# Utilities
# -------------------------------

def resize_or_pad(img, shape):
    """
    If img is smaller than shape, center-pad with zeros.
    If larger, center-crop.
    """
    N, M = shape
    h, w = img.shape
    out = np.zeros((N, M), dtype=img.dtype)

    # target window
    if h <= N:
        sy = (N - h)//2; ey = sy + h
        ys = 0; ye = h
    else:
        sy = 0; ey = N
        ys = (h - N)//2; ye = ys + N

    if w <= M:
        sx = (M - w)//2; ex = sx + w
        xs = 0; xe = w
    else:
        sx = 0; ex = M
        xs = (w - M)//2; xe = xs + M

    out[sy:ey, sx:ex] = img[ys:ye, xs:xe]
    return out

def wrap_to_pi(phase):
    return (phase + np.pi) % (2*np.pi) - np.pi

def ask_save(parent, default_name, filter_spec):
    path, _ = QFileDialog.getSaveFileName(parent, "Save As", default_name, filter_spec)
    return path

def fpath_with_new_ext(path, new_ext):
    import os
    root, _ = os.path.splitext(path)
    return root + new_ext

def save_csv_dict(d, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in d.items():
            f.write(f"{k},{v}\n")

# -------------------------------
# Entry
# -------------------------------

def main():
    app = QApplication(sys.argv)
    w = HoloGUI()
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
