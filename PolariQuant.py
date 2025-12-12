"""
Polariscope analysis and figure generation

This script

  * Loads an aligned, cropped RGB TIFF stack from a homemade polariscope
  * Computes intensity vs angle for three ROIs and fits a cosine model
  * Builds sliding window "stress" maps, orientations, and fit quality
  * Repeats the analysis per RGB channel and builds a combined orientation
  * Provides figure functions corresponding to Figures 1 through 12
  * Lets you choose which figure to generate after analysis is complete
"""

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tifffile as tiff


# ----------------------------------------------------------------------
# Colors and small utilities
# ----------------------------------------------------------------------


def darken(color, amount: float = 0.7):
    """
    Darken a Matplotlib color by multiplying its RGB values.
    amount < 1 gives a darker color.
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb * amount)


TIP_COLOR = "tab:blue"
TIP_DARK_COLOR = darken(TIP_COLOR, 0.7)

MID_COLOR = "tab:orange"
MID_DARK_COLOR = darken(MID_COLOR, 0.7)

BASE_COLOR = "tab:green"
BASE_DARK_COLOR = darken(BASE_COLOR, 0.7)


# Analyzer angles in degrees and radians
ANGLES_DEG = np.array([0, 15, 30, 45, 60, 75, 90], dtype=float)
ANGLES_RAD = np.deg2rad(ANGLES_DEG)


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------


@dataclass
class DatasetConfig:
    key: str
    label: str
    tif_path: str
    rois: Dict[str, Tuple[int, int, int, int]]
    

# Preconfigured data sets.
# Fork uses your existing ROIs. The stressed plastic starts with no ROIs.
# You can fill in rois["region1"] etc once you pick coordinates.
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "fork": DatasetConfig(
        key="fork",
        label="Fork (aligned, cropped)",
        tif_path="data/fork_aligned_cropped.tif",
        rois={
            "tip": (132, 145, 938, 949),
            "middle": (481, 487, 693, 700),
            "base": (710, 720, 617, 621),
        },
    ),
    "plastic": DatasetConfig(
        key="plastic",
        label="Stressed plastic (aligned, cropped)",
        tif_path="data/stressed_plastic_aligned_cropped.tif",
        rois={
            "bend": (646, 654, 1162, 1171),
            "middle": (498, 507, 681, 693),
            "base": (356, 362, 176, 183),
        },
    ),
}


@dataclass
class ROIResult:
    name: str
    roi: Tuple[int, int, int, int]
    intensity: np.ndarray
    a: float
    b: float
    c: float
    I_mod: float
    phase: float
    R2: float
    RMSE: float


@dataclass
class SlidingGrayResult:
    box_h: int
    box_w: int
    a_map: np.ndarray
    I_mod_map: np.ndarray
    phase_map: np.ndarray
    theta_map: np.ndarray
    theta_deg_map: np.ndarray
    R2_map: np.ndarray


@dataclass
class ChannelMaps:
    mod: np.ndarray
    norm_mod: np.ndarray
    phase: np.ndarray
    theta: np.ndarray
    R2: np.ndarray


@dataclass
class CombinedRGBResult:
    theta_comb: np.ndarray
    mod_comb: np.ndarray
    R2_comb: np.ndarray
    diff_map_orig_vs_comb: np.ndarray
    channel_vs_comb_stats: Dict[str, Dict[str, float]]
    orig_vs_comb_stats: Dict[str, float]


@dataclass
class AnalysisResults:
    tif_path: str
    angles_deg: np.ndarray
    angles_rad: np.ndarray
    stack_raw: np.ndarray
    stack_gray: np.ndarray
    roi_results: Dict[str, ROIResult]
    sliding_gray: SlidingGrayResult
    has_color: bool
    channel_R: Optional[ChannelMaps] = None
    channel_G: Optional[ChannelMaps] = None
    channel_B: Optional[ChannelMaps] = None
    combined_rgb: Optional[CombinedRGBResult] = None


# ----------------------------------------------------------------------
# Core analysis helpers
# ----------------------------------------------------------------------


def load_stack(tif_path: str):
    """
    Load the TIFF stack and return
      stack_raw: original stack as read (float32)
      stack_gray: grayscale stack of shape (N, H, W)
      has_color: True if stack_raw is RGB and color analysis is possible
    """
    stack_raw = tiff.imread(tif_path).astype(np.float32)
    print(f"Raw stack shape: {stack_raw.shape}")

    if stack_raw.ndim == 4:
        # (Nangles, H, W, 3)
        N, H, W, C = stack_raw.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")
        R = stack_raw[..., 0]
        G = stack_raw[..., 1]
        B = stack_raw[..., 2]
        stack_gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
        has_color = True
    elif stack_raw.ndim == 3:
        # Already grayscale, assume (Nangles, H, W)
        stack_gray = stack_raw
        N, H, W = stack_gray.shape
        has_color = False
    else:
        raise ValueError(f"Unexpected stack dimensions: {stack_raw.shape}")

    print(f"Grayscale stack shape (N, H, W): {stack_gray.shape}")

    if N != len(ANGLES_DEG):
        print(
            "Warning: number of slices does not match len(ANGLES_DEG). "
            "Make sure ANGLES_DEG matches your acquisition."
        )

    return stack_raw, stack_gray, has_color


def mean_intensity_vs_angle(stack_gray: np.ndarray, roi: Tuple[int, int, int, int]):
    """
    stack_gray: (Nangles, H, W)
    roi: (r0, r1, c0, c1)
    returns: intensity vs angle as length Nangles array
    """
    r0, r1, c0, c1 = roi
    patch = stack_gray[:, r0:r1, c0:c1]
    I = patch.mean(axis=(1, 2))
    return I


def fit_cos2A(angles_rad: np.ndarray, I: np.ndarray):
    """
    Fit I(A) = a + b cos(2A) + c sin(2A) using least squares.

    Returns a, b, c, I_mod, phase, R2, RMSE.
    """
    A = np.asarray(angles_rad)
    I = np.asarray(I)

    X = np.column_stack(
        [
            np.ones_like(A),
            np.cos(2 * A),
            np.sin(2 * A),
        ]
    )

    params, _, _, _ = np.linalg.lstsq(X, I, rcond=None)
    a, b, c = params

    I_fit = X @ params
    residuals = I - I_fit

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((I - I.mean()) ** 2)

    if ss_tot > 0:
        R2 = 1.0 - ss_res / ss_tot
    else:
        R2 = np.nan

    dof = len(I) - 3
    if dof > 0:
        RMSE = np.sqrt(ss_res / dof)
    else:
        RMSE = np.nan

    I_mod = float(np.sqrt(b ** 2 + c ** 2))
    phase = float(np.arctan2(-c, b))  # I(A) ~ a + I_mod cos(2A - phase)

    return float(a), float(b), float(c), I_mod, phase, float(R2), float(RMSE)


def model_I(angles_rad: np.ndarray, a: float, b: float, c: float):
    return a + b * np.cos(2 * angles_rad) + c * np.sin(2 * angles_rad)


def sliding_mean(image: np.ndarray, box_h: int, box_w: int):
    """
    Fast sliding box mean using an integral image.

    image: 2D array (H, W)
    returns: 2D array (H - box_h + 1, W - box_w + 1)
    """
    H, W = image.shape
    integ = np.pad(image, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    integ = integ.cumsum(axis=0).cumsum(axis=1)

    sum_windows = (
        integ[box_h:, box_w:]
        - integ[:-box_h, box_w:]
        - integ[box_h:, :-box_w]
        + integ[:-box_h, :-box_w]
    )
    mean_windows = sum_windows / float(box_h * box_w)
    return mean_windows


def compute_sliding_gray(
    stack_gray: np.ndarray, angles_rad: np.ndarray, box_h: int, box_w: int
) -> SlidingGrayResult:
    """
    Sliding window cosine fit across the grayscale field.
    """
    Nangles, H, W = stack_gray.shape
    H_out = H - box_h + 1
    W_out = W - box_w + 1

    print(
        f"Computing sliding window gray map with box {box_h}x{box_w}, "
        f"output size {H_out} x {W_out}"
    )

    means = np.empty((Nangles, H_out, W_out), dtype=np.float32)
    for k in range(Nangles):
        means[k] = sliding_mean(stack_gray[k], box_h, box_w)

    I_flat = means.reshape(Nangles, -1)

    X = np.column_stack(
        [
            np.ones_like(angles_rad),
            np.cos(2 * angles_rad),
            np.sin(2 * angles_rad),
        ]
    )
    P = np.linalg.pinv(X)

    params_flat = P @ I_flat
    a_flat = params_flat[0]
    b_flat = params_flat[1]
    c_flat = params_flat[2]

    I_mod_flat = np.sqrt(b_flat ** 2 + c_flat ** 2)
    phase_flat = np.arctan2(-c_flat, b_flat)

    I_fit_flat = X @ params_flat
    res_flat = I_flat - I_fit_flat
    ss_res = np.sum(res_flat ** 2, axis=0)
    I_mean_flat = np.mean(I_flat, axis=0)
    ss_tot = np.sum((I_flat - I_mean_flat) ** 2, axis=0)
    R2_flat = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    a_map = a_flat.reshape(H_out, W_out)
    I_mod_map = I_mod_flat.reshape(H_out, W_out)
    phase_map = phase_flat.reshape(H_out, W_out)
    R2_map = R2_flat.reshape(H_out, W_out)

    theta_map = 0.5 * phase_map
    theta_map = np.mod(theta_map, np.pi)
    theta_deg_map = np.rad2deg(theta_map)

    return SlidingGrayResult(
        box_h=box_h,
        box_w=box_w,
        a_map=a_map,
        I_mod_map=I_mod_map,
        phase_map=phase_map,
        theta_map=theta_map,
        theta_deg_map=theta_deg_map,
        R2_map=R2_map,
    )


def compute_mod_map_for_stack(
    stack_chan: np.ndarray,
    angles_rad: np.ndarray,
    box_h: int,
    box_w: int,
    normalize: bool = True,
):
    """
    stack_chan: array of shape (N_angles, H, W) for one color channel.
    Returns:
        a_map, I_mod_map, norm_map, phase_map, R2_map
    """
    N_angles, H, W = stack_chan.shape
    H_out = H - box_h + 1
    W_out = W - box_w + 1

    means = np.empty((N_angles, H_out, W_out), dtype=np.float32)
    for k in range(N_angles):
        means[k] = sliding_mean(stack_chan[k], box_h, box_w)

    I_flat = means.reshape(N_angles, -1)

    X = np.column_stack(
        [
            np.ones_like(angles_rad),
            np.cos(2 * angles_rad),
            np.sin(2 * angles_rad),
        ]
    )
    P = np.linalg.pinv(X)

    params_flat = P @ I_flat
    a_flat = params_flat[0]
    b_flat = params_flat[1]
    c_flat = params_flat[2]

    I_mod_flat = np.sqrt(b_flat ** 2 + c_flat ** 2)
    phase_flat = np.arctan2(-c_flat, b_flat)

    I_fit_flat = X @ params_flat
    res_flat = I_flat - I_fit_flat
    ss_res = np.sum(res_flat ** 2, axis=0)
    I_mean_flat = np.mean(I_flat, axis=0)
    ss_tot = np.sum((I_flat - I_mean_flat) ** 2, axis=0)
    R2_flat = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    a_map = a_flat.reshape(H_out, W_out)
    I_mod_map = I_mod_flat.reshape(H_out, W_out)
    phase_map = phase_flat.reshape(H_out, W_out)
    R2_map = R2_flat.reshape(H_out, W_out)

    if normalize:
        norm_map = I_mod_map / np.where(a_map > 0, a_map, np.nan)
    else:
        norm_map = None

    return a_map, I_mod_map, norm_map, phase_map, R2_map


def compute_color_channel_maps(
    stack_raw: np.ndarray, angles_rad: np.ndarray, box_h: int, box_w: int
):
    """
    Compute modulation, orientation, and R2 maps for R, G, B channels,
    and the combined RGB orientation.
    """
    N, H, W, C = stack_raw.shape
    if C != 3:
        raise ValueError("Color analysis expects stack_raw with 3 channels.")

    stack_R = stack_raw[..., 0]
    stack_G = stack_raw[..., 1]
    stack_B = stack_raw[..., 2]

    print("Computing color channel modulation maps...")

    _, mod_R, norm_R, phase_R, R2_R = compute_mod_map_for_stack(
        stack_R, angles_rad, box_h, box_w, normalize=True
    )
    _, mod_G, norm_G, phase_G, R2_G = compute_mod_map_for_stack(
        stack_G, angles_rad, box_h, box_w, normalize=True
    )
    _, mod_B, norm_B, phase_B, R2_B = compute_mod_map_for_stack(
        stack_B, angles_rad, box_h, box_w, normalize=True
    )

    theta_R = np.mod(0.5 * phase_R, np.pi)
    theta_G = np.mod(0.5 * phase_G, np.pi)
    theta_B = np.mod(0.5 * phase_B, np.pi)

    ch_R = ChannelMaps(mod=mod_R, norm_mod=norm_R, phase=phase_R, theta=theta_R, R2=R2_R)
    ch_G = ChannelMaps(mod=mod_G, norm_mod=norm_G, phase=phase_G, theta=theta_G, R2=R2_G)
    ch_B = ChannelMaps(mod=mod_B, norm_mod=norm_B, phase=phase_B, theta=theta_B, R2=R2_B)

    # Combined RGB orientation, weighted by modulation and R2
    w_R = np.clip(mod_R, 0, None) * np.clip(R2_R, 0, 1)
    w_G = np.clip(mod_G, 0, None) * np.clip(R2_G, 0, 1)
    w_B = np.clip(mod_B, 0, None) * np.clip(R2_B, 0, 1)

    cos2_R = np.cos(2 * theta_R)
    sin2_R = np.sin(2 * theta_R)
    cos2_G = np.cos(2 * theta_G)
    sin2_G = np.sin(2 * theta_G)
    cos2_B = np.cos(2 * theta_B)
    sin2_B = np.sin(2 * theta_B)

    c_num = w_R * cos2_R + w_G * cos2_G + w_B * cos2_B
    s_num = w_R * sin2_R + w_G * sin2_G + w_B * sin2_B
    w_sum = w_R + w_G + w_B

    cos2_avg = c_num / np.where(w_sum > 0, w_sum, np.nan)
    sin2_avg = s_num / np.where(w_sum > 0, w_sum, np.nan)

    theta_comb = 0.5 * np.arctan2(sin2_avg, cos2_avg)
    theta_comb = np.mod(theta_comb, np.pi)

    mod_comb = (mod_R + mod_G + mod_B) / 3.0
    R2_comb = (R2_R + R2_G + R2_B) / 3.0

    return ch_R, ch_G, ch_B, theta_comb, mod_comb, R2_comb


def orientation_diff_deg(theta1: np.ndarray, theta2: np.ndarray):
    """
    Smallest angular difference between two orientations (radians),
    where orientations are equivalent modulo 180 degrees.
    Returns degrees in [0, 90].
    """
    d = theta2 - theta1
    d_wrapped = (d + np.pi / 2.0) % np.pi - np.pi / 2.0
    return np.rad2deg(np.abs(d_wrapped))


def compute_combined_stats(
    gray: SlidingGrayResult,
    ch_R: ChannelMaps,
    ch_G: ChannelMaps,
    ch_B: ChannelMaps,
    theta_comb: np.ndarray,
    mod_comb: np.ndarray,
    R2_comb: np.ndarray,
) -> CombinedRGBResult:
    """
    Compute statistics comparing each channel to the combined RGB orientation,
    and comparing the original gray orientation to the combined RGB orientation.
    Also builds a masked difference map for plotting (Figure 12).
    """
    R2_thr = 0.7
    mod_pctl = 5.0

    channel_results: Dict[str, Dict[str, float]] = {}

    for name, ch in [("R", ch_R), ("G", ch_G), ("B", ch_B)]:
        mod_thr_c = float(np.percentile(ch.mod, mod_pctl))
        mask = (
            (ch.R2 > R2_thr)
            & (ch.mod > mod_thr_c)
            & np.isfinite(ch.theta)
            & np.isfinite(theta_comb)
        )
        diff = orientation_diff_deg(ch.theta[mask], theta_comb[mask])

        print(f"{name} channel vs combined RGB:")
        if diff.size == 0:
            print("  No valid pixels for comparison")
            channel_results[name] = {}
            continue

        mean_diff = float(np.mean(diff))
        median_diff = float(np.median(diff))
        p90_diff = float(np.percentile(diff, 90))
        rms_diff = float(np.sqrt(np.mean(diff ** 2)))

        print(f"  N = {diff.size}")
        print(f"  mean diff   = {mean_diff:.2f} deg")
        print(f"  median diff = {median_diff:.2f} deg")
        print(f"  90th pct    = {p90_diff:.2f} deg")
        print(f"  RMS diff    = {rms_diff:.2f} deg")

        channel_results[name] = {
            "N": int(diff.size),
            "mean": mean_diff,
            "median": median_diff,
            "p90": p90_diff,
            "rms": rms_diff,
        }

    # Original gray vs combined RGB
    R2_thr_orig = 0.7
    R2_thr_comb = 0.7

    mod_thr_orig = float(np.percentile(gray.I_mod_map, mod_pctl))
    mod_thr_comb = float(np.percentile(mod_comb, mod_pctl))

    mask_orig_comb = (
        (gray.R2_map > R2_thr_orig)
        & (R2_comb > R2_thr_comb)
        & (gray.I_mod_map > mod_thr_orig)
        & (mod_comb > mod_thr_comb)
        & np.isfinite(gray.theta_map)
        & np.isfinite(theta_comb)
    )

    diff_all = orientation_diff_deg(gray.theta_map[mask_orig_comb], theta_comb[mask_orig_comb])

    print("Original gray vs combined RGB orientation")
    print(f"  N valid pixels = {diff_all.size}")

    orig_vs_comb_stats: Dict[str, float] = {}
    if diff_all.size > 0:
        mean_diff = float(np.mean(diff_all))
        median_diff = float(np.median(diff_all))
        p90_diff = float(np.percentile(diff_all, 90))
        rms_diff = float(np.sqrt(np.mean(diff_all ** 2)))

        print(f"  mean diff   = {mean_diff:.2f} deg")
        print(f"  median diff = {median_diff:.2f} deg")
        print(f"  90th pct    = {p90_diff:.2f} deg")
        print(f"  RMS diff    = {rms_diff:.2f} deg")

        orig_vs_comb_stats = {
            "N": int(diff_all.size),
            "mean": mean_diff,
            "median": median_diff,
            "p90": p90_diff,
            "rms": rms_diff,
        }

    # Full field difference map, masked by the same criteria
    diff_map = orientation_diff_deg(gray.theta_map, theta_comb)
    diff_map_masked = np.where(mask_orig_comb, diff_map, np.nan)

    return CombinedRGBResult(
        theta_comb=theta_comb,
        mod_comb=mod_comb,
        R2_comb=R2_comb,
        diff_map_orig_vs_comb=diff_map_masked,
        channel_vs_comb_stats=channel_results,
        orig_vs_comb_stats=orig_vs_comb_stats,
    )


# ----------------------------------------------------------------------
# Main analysis driver
# ----------------------------------------------------------------------


def run_analysis(dataset: DatasetConfig, box_h: int = 10, box_w: int = 10) -> AnalysisResults:
    """
    Run all analysis steps once and return a results object that
    plotting functions can use without recomputing.
    """
    stack_raw, stack_gray, has_color = load_stack(dataset.tif_path)

    # ROI analysis, if any ROIs are defined for this data set
    roi_results: Dict[str, ROIResult] = {}

    for name, roi in dataset.rois.items():
        I = mean_intensity_vs_angle(stack_gray, roi)
        a, b, c, I_mod, phase, R2, RMSE = fit_cos2A(ANGLES_RAD, I)
        roi_results[name] = ROIResult(
            name=name,
            roi=roi,
            intensity=I,
            a=a,
            b=b,
            c=c,
            I_mod=I_mod,
            phase=phase,
            R2=R2,
            RMSE=RMSE,
        )

    print("\nROI cosine fit summary (modulation ~ birefringence strength, phase ~ axis)")
    if not roi_results:
        print("  (No ROIs defined for this data set.)")
    else:
        for name, r in roi_results.items():
            print(
                f"{name:10s} I_mod = {r.I_mod:.2f}, "
                f"phase = {r.phase:.2f} rad, R^2 = {r.R2:.3f}, RMSE = {r.RMSE:.3f}"
            )

    # Sliding window gray analysis
    sliding_gray = compute_sliding_gray(stack_gray, ANGLES_RAD, box_h, box_w)

    # Color channel analysis
    channel_R = channel_G = channel_B = None
    combined_rgb = None

    if has_color and stack_raw.ndim == 4 and stack_raw.shape[-1] == 3:
        ch_R, ch_G, ch_B, theta_comb, mod_comb, R2_comb = compute_color_channel_maps(
            stack_raw, ANGLES_RAD, box_h, box_w
        )
        combined_rgb = compute_combined_stats(
            sliding_gray, ch_R, ch_G, ch_B, theta_comb, mod_comb, R2_comb
        )
        channel_R, channel_G, channel_B = ch_R, ch_G, ch_B
    else:
        if not has_color:
            print("\nNote: stack appears to be grayscale only, skipping color analysis.")

    return AnalysisResults(
        tif_path=dataset.tif_path,
        angles_deg=ANGLES_DEG,
        angles_rad=ANGLES_RAD,
        stack_raw=stack_raw,
        stack_gray=stack_gray,
        roi_results=roi_results,
        sliding_gray=sliding_gray,
        has_color=has_color,
        channel_R=channel_R,
        channel_G=channel_G,
        channel_B=channel_B,
        combined_rgb=combined_rgb,
    )



# ----------------------------------------------------------------------
# Plot helpers and figure functions
# ----------------------------------------------------------------------


def draw_quiver_on_axis(
    ax,
    theta_map: np.ndarray,
    I_mod_map: np.ndarray,
    R2_map: np.ndarray,
    title: str,
    arrow_color: str = "lime",
):
    """
    Draw a stress direction quiver plot for one map on a given Matplotlib axis.
    """
    Hc, Wc = theta_map.shape

    target_arrows_across = 20
    step = max(1, int(min(Hc, Wc) / target_arrows_across))

    theta_sample = theta_map[::step, ::step]
    R2_sample = R2_map[::step, ::step]
    mod_sample = I_mod_map[::step, ::step]

    R2_threshold = 0.7
    mod_threshold = np.percentile(I_mod_map, 5)

    good_sample = (R2_sample > R2_threshold) & (mod_sample > mod_threshold)

    arrow_len = 0.7 * step
    U = arrow_len * np.cos(theta_sample)
    V = arrow_len * np.sin(theta_sample)

    U = np.where(good_sample, U, np.nan)
    V = np.where(good_sample, V, np.nan)

    Yc, Xc = np.mgrid[0:Hc:step, 0:Wc:step]

    ax.imshow(I_mod_map, origin="lower", cmap="gray")
    ax.quiver(
        Xc,
        Yc,
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="mid",
        color=arrow_color,
        width=0.006,
        headwidth=2.5,
        headlength=3.0,
        headaxislength=2.5,
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")


def plot_quiver_for_channel(
    theta_map: np.ndarray,
    I_mod_map: np.ndarray,
    R2_map: np.ndarray,
    title: str,
    color: str = "lime",
):
    """
    Convenience wrapper that draws a single panel quiver plot.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_quiver_on_axis(ax, theta_map, I_mod_map, R2_map, title, arrow_color=color)
    plt.tight_layout()
    plt.show()


# Figure 1: intensity in each ROI at each angle
def figure_1_roi_intensity(results: AnalysisResults):
    if not results.roi_results:
        print("Figure 1 needs ROI definitions, but none were found for this data set.")
        return

    angles_deg = results.angles_deg

    plt.figure()
    markers = ["o", "s", "d", "^", "v", "x"]
    color_map = {
        "tip": TIP_COLOR,
        "middle": MID_COLOR,
        "base": BASE_COLOR,
    }

    for idx, (name, r) in enumerate(results.roi_results.items()):
        marker = markers[idx % len(markers)]
        color = color_map.get(name, f"C{idx}")
        plt.plot(
            angles_deg,
            r.intensity,
            marker + "-",
            label=name,
            linewidth=1,
            markersize=4,
            color=color,
        )

    plt.xlabel("Analyzer angle (deg)")
    plt.ylabel("Mean intensity (arb. units)")
    plt.legend()
    plt.title("Raw intensity vs angle for defined ROIs")
    plt.show()



# Figure 2: data and cosine fits for three ROIs
def figure_2_roi_fits(results: AnalysisResults):
    if not results.roi_results:
        print("Figure 2 needs ROI definitions, but none were found for this data set.")
        return

    angles_deg = results.angles_deg

    A_dense = np.linspace(0, np.pi / 2, 200)
    markers = ["o", "s", "d", "^", "v", "x"]
    color_map = {
        "tip": TIP_COLOR,
        "middle": MID_COLOR,
        "base": BASE_COLOR,
    }
    dark_color_map = {
        "tip": TIP_DARK_COLOR,
        "middle": MID_DARK_COLOR,
        "base": BASE_DARK_COLOR,
    }

    plt.figure()

    for idx, (name, r) in enumerate(results.roi_results.items()):
        marker = markers[idx % len(markers)]
        color = color_map.get(name, f"C{idx}")
        dark = dark_color_map.get(name, color)

        plt.plot(
            angles_deg,
            r.intensity,
            marker,
            label=f"{name} data",
            markersize=4,
            color=dark,
        )
        plt.plot(
            np.rad2deg(A_dense),
            model_I(A_dense, r.a, r.b, r.c),
            "-",
            label=f"{name} fit",
            linewidth=1,
            color=color,
        )

    plt.xlabel("Analyzer angle (deg)")
    plt.ylabel("Mean intensity (arb. units)")
    plt.title("Data and cosine fits for defined ROIs")
    plt.legend()
    plt.show()



# Figure 3: heat map of I_mod (gray stack)
def figure_3_modulation_map(results: AnalysisResults):
    g = results.sliding_gray
    plt.figure(figsize=(6, 5))
    im = plt.imshow(g.I_mod_map, origin="lower", cmap="inferno")
    plt.colorbar(im, label="I_mod (arb. units)")
    plt.title(f"Sliding {g.box_h}x{g.box_w} modulation map")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    plt.show()


# Figure 4: heat map of R squared (gray stack)
def figure_4_R2_map(results: AnalysisResults):
    g = results.sliding_gray
    plt.figure(figsize=(6, 5))
    im = plt.imshow(g.R2_map, origin="lower", vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, label="R^2 of cosine fit")
    plt.title("Fit quality map (R^2)")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    plt.show()


# Figure 5: heat map of stress direction (gray stack, angle map)
def figure_5_orientation_map(results: AnalysisResults):
    g = results.sliding_gray

    R2_threshold = 0.9
    mod_threshold = np.percentile(g.I_mod_map, 10)

    good_mask = (g.R2_map > R2_threshold) & (g.I_mod_map > mod_threshold)
    theta_plot = np.where(good_mask, g.theta_deg_map, np.nan)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(theta_plot, origin="lower", cmap="hsv", vmin=0, vmax=180)
    cbar = plt.colorbar(im, label="Estimated optic axis angle (deg)")
    plt.title(f"Sliding {g.box_h}x{g.box_w} stress direction map")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    plt.show()


# Figure 6: quiver map of stress direction vectors (gray stack)
def figure_6_quiver_gray(results: AnalysisResults):
    g = results.sliding_gray
    H, W = g.theta_map.shape

    target_arrows_across = 20
    step = max(1, int(min(H, W) / target_arrows_across))
    print(f"Figure 6: using quiver step = {step}")

    theta_sample = g.theta_map[::step, ::step]
    R2_sample = g.R2_map[::step, ::step]
    mod_sample = g.I_mod_map[::step, ::step]

    R2_threshold = 0.7
    mod_threshold = np.percentile(g.I_mod_map, 5)
    good_sample = (R2_sample > R2_threshold) & (mod_sample > mod_threshold)

    arrow_len = 0.7 * step
    U = arrow_len * np.cos(theta_sample)
    V = arrow_len * np.sin(theta_sample)
    U = np.where(good_sample, U, np.nan)
    V = np.where(good_sample, V, np.nan)

    Y, X = np.mgrid[0:H:step, 0:W:step]

    plt.figure(figsize=(6, 6))
    plt.imshow(g.I_mod_map, origin="lower", cmap="gray")
    plt.quiver(
        X,
        Y,
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="mid",
        color="lime",
        width=0.006,
        headwidth=2.5,
        headlength=3.0,
        headaxislength=2.5,
        alpha=0.9,
    )
    plt.title(
        f"Stress direction vectors (overview, ~{target_arrows_across} across)"
    )
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    plt.show()


# Figure 7: heat map of I_mod for each channel separately
def figure_7_modulation_rgb(results: AnalysisResults):
    if not results.has_color or results.channel_R is None:
        print("Figure 7 requires RGB data, but the stack appears to be grayscale.")
        return

    ch_R = results.channel_R
    ch_G = results.channel_G
    ch_B = results.channel_B

    map_R = ch_R.norm_mod
    map_G = ch_G.norm_mod
    map_B = ch_B.norm_mod
    label_text = "Normalized modulation I_mod / a"

    vmax = np.nanmax([map_R, map_G, map_B])
    vmin = np.nanmin([map_R, map_G, map_B])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(map_R, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("Red channel")
    axes[0].set_xlabel("x (pixels)")
    axes[0].set_ylabel("y (pixels)")

    im1 = axes[1].imshow(map_G, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("Green channel")
    axes[1].set_xlabel("x (pixels)")

    im2 = axes[2].imshow(map_B, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[2].set_title("Blue channel")
    axes[2].set_xlabel("x (pixels)")

    cbar = fig.colorbar(im2, ax=axes.ravel().tolist())
    cbar.set_label(label_text)

    plt.suptitle(f"Sliding {results.sliding_gray.box_h}x"
                 f"{results.sliding_gray.box_w} modulation by color channel")
    plt.show()


# Figure 8: heat map of R squared for each channel separately
def figure_8_R2_rgb(results: AnalysisResults):
    if not results.has_color or results.channel_R is None:
        print("Figure 8 requires RGB data, but the stack appears to be grayscale.")
        return

    ch_R = results.channel_R
    ch_G = results.channel_G
    ch_B = results.channel_B

    vmin_R2 = 0.0
    vmax_R2 = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(
        ch_R.R2, origin="lower", cmap="viridis", vmin=vmin_R2, vmax=vmax_R2
    )
    axes[0].set_title("Red channel R$^2$")
    axes[0].set_xlabel("x (pixels)")
    axes[0].set_ylabel("y (pixels)")

    im1 = axes[1].imshow(
        ch_G.R2, origin="lower", cmap="viridis", vmin=vmin_R2, vmax=vmax_R2
    )
    axes[1].set_title("Green channel R$^2$")
    axes[1].set_xlabel("x (pixels)")

    im2 = axes[2].imshow(
        ch_B.R2, origin="lower", cmap="viridis", vmin=vmin_R2, vmax=vmax_R2
    )
    axes[2].set_title("Blue channel R$^2$")
    axes[2].set_xlabel("x (pixels)")

    cbar = fig.colorbar(im2, ax=axes.ravel().tolist())
    cbar.set_label("R$^2$ of cosine fit")

    plt.suptitle(
        f"Sliding {results.sliding_gray.box_h}x"
        f"{results.sliding_gray.box_w} R$^2$ maps by color channel"
    )
    plt.show()


# Figure 9: quiver map of stress direction vectors per channel
def figure_9_quiver_rgb(results: AnalysisResults):
    if not results.has_color or results.channel_R is None:
        print("Figure 9 requires RGB data, but the stack appears to be grayscale.")
        return

    ch_R = results.channel_R
    ch_G = results.channel_G
    ch_B = results.channel_B

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    draw_quiver_on_axis(
        axes[0],
        ch_R.theta,
        ch_R.mod,
        ch_R.R2,
        "Red channel stress direction",
        arrow_color="red",
    )
    draw_quiver_on_axis(
        axes[1],
        ch_G.theta,
        ch_G.mod,
        ch_G.R2,
        "Green channel stress direction",
        arrow_color="lime",
    )
    draw_quiver_on_axis(
        axes[2],
        ch_B.theta,
        ch_B.mod,
        ch_B.R2,
        "Blue channel stress direction",
        arrow_color="cyan",
    )

    plt.suptitle(
        f"Sliding {results.sliding_gray.box_h}x"
        f"{results.sliding_gray.box_w} stress direction by color channel"
    )
    plt.show()


# Figure 10: quiver map of combined RGB stress direction
def figure_10_quiver_combined(results: AnalysisResults):
    if results.combined_rgb is None:
        print("Figure 10 requires RGB analysis, which was not computed.")
        return

    comb = results.combined_rgb
    # Use average modulation and R2 maps from combined RGB result
    plot_quiver_for_channel(
        comb.theta_comb,
        comb.mod_comb,
        comb.R2_comb,
        "Combined RGB stress direction",
        color="lime",
    )


# Figure 11: comparison of original gray vs combined RGB quiver plots
def figure_11_quiver_comparison(results: AnalysisResults):
    if results.combined_rgb is None:
        print("Figure 11 requires RGB analysis, which was not computed.")
        return

    g = results.sliding_gray
    comb = results.combined_rgb

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    draw_quiver_on_axis(
        axes[0],
        g.theta_map,
        g.I_mod_map,
        g.R2_map,
        "Original gray stress direction",
        arrow_color="lime",
    )
    draw_quiver_on_axis(
        axes[1],
        comb.theta_comb,
        comb.mod_comb,
        comb.R2_comb,
        "Combined RGB stress direction",
        arrow_color="cyan",
    )

    plt.suptitle(
        "Original gray vs combined RGB stress direction\n"
        f"(sliding {g.box_h}x{g.box_w})"
    )
    plt.show()


# Figure 12: heat map of difference between original and combined RGB orientations
def figure_12_difference_map(results: AnalysisResults):
    if results.combined_rgb is None:
        print("Figure 12 requires RGB analysis, which was not computed.")
        return

    diff_map = results.combined_rgb.diff_map_orig_vs_comb

    plt.figure(figsize=(6, 5))
    im = plt.imshow(diff_map, origin="lower", cmap="magma", vmin=0, vmax=30)
    cbar = plt.colorbar(im, label="Angle difference (deg)")
    plt.title("Original vs combined RGB orientation difference")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    plt.show()

    # Degree difference summary has already been printed during analysis,
    # but you could reprint summary from results.combined_rgb.orig_vs_comb_stats here if desired.


def figure_13_diff_histogram(results: AnalysisResults):
    """
    Figure 13: Histogram of number of pixels vs orientation difference
    between original gray orientation and combined RGB orientation.
    """
    if results.combined_rgb is None:
        print("Figure 13 requires RGB analysis, which was not computed.")
        return

    diff_map = results.combined_rgb.diff_map_orig_vs_comb
    diff_vals = diff_map[np.isfinite(diff_map)]

    if diff_vals.size == 0:
        print("No valid pixels to include in the histogram for Figure 13.")
        return

    # You can adjust the binning depending on how wide the distribution is
    max_for_bins = max(30.0, np.percentile(diff_vals, 99))
    bins = np.linspace(0, max_for_bins, 31)

    plt.figure(figsize=(6, 4))
    plt.hist(diff_vals, bins=bins, edgecolor="black", alpha=0.8)
    plt.xlabel("Orientation difference (deg)")
    plt.ylabel("Number of pixels")
    plt.title(
        "Histogram of orientation difference\n"
        "(original gray vs combined RGB)"
    )

    # Overlay mean and median if they are available
    stats = results.combined_rgb.orig_vs_comb_stats
    if stats:
        mean_diff = stats.get("mean")
        median_diff = stats.get("median")

        if mean_diff is not None:
            plt.axvline(
                mean_diff,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean = {mean_diff:.1f} deg",
            )
        if median_diff is not None:
            plt.axvline(
                median_diff,
                color="green",
                linestyle=":",
                linewidth=2,
                label=f"Median = {median_diff:.1f} deg",
            )
        plt.legend()

    plt.tight_layout()
    plt.show()


# Mapping from figure numbers to functions and descriptions
FIGURE_FUNCTIONS = {
    1: figure_1_roi_intensity,
    2: figure_2_roi_fits,
    3: figure_3_modulation_map,
    4: figure_4_R2_map,
    5: figure_5_orientation_map,
    6: figure_6_quiver_gray,
    7: figure_7_modulation_rgb,
    8: figure_8_R2_rgb,
    9: figure_9_quiver_rgb,
    10: figure_10_quiver_combined,
    11: figure_11_quiver_comparison,
    12: figure_12_difference_map,
    13: figure_13_diff_histogram,
}

FIGURE_DESCRIPTIONS = {
    1: "ROI intensity vs angle",
    2: "ROI data and cosine fits",
    3: "Gray modulation (I_mod) heat map",
    4: "Gray R^2 heat map",
    5: "Gray stress direction angle map",
    6: "Gray stress direction quiver map",
    7: "RGB modulation heat maps",
    8: "RGB R^2 heat maps",
    9: "RGB stress direction quiver maps",
    10: "Combined RGB stress direction quiver map",
    11: "Original vs combined RGB quiver comparison",
    12: "Difference between original and combined RGB orientations",
    13: "Histogram of original vs combined RGB orientation difference",
}


def plot_figure(results: AnalysisResults, num: int):
    func = FIGURE_FUNCTIONS.get(num)
    if func is None:
        print(f"Unknown figure number {num}. Valid figures are 1 to 12.")
        return
    func(results)


def choose_dataset_interactive() -> DatasetConfig:
    """
    Simple text menu to choose which sample to analyze.
    """
    while True:
        print("\nAvailable data sets:")
        keys = list(DATASET_CONFIGS.keys())
        for idx, key in enumerate(keys, start=1):
            cfg = DATASET_CONFIGS[key]
            print(f"  {idx}: {cfg.label}  [{key}]")
        print("  c: Custom TIFF path")

        choice = input("Select data set (Enter for 1): ").strip().lower()

        if choice == "":
            return DATASET_CONFIGS[keys[0]]

        if choice == "c":
            tif_path = input("Custom TIFF path: ").strip()
            if not tif_path:
                continue
            return DatasetConfig(
                key="custom",
                label=f"Custom ({tif_path})",
                tif_path=tif_path,
                rois={},  # add custom ROIs later if you want
            )

        try:
            idx = int(choice)
            if 1 <= idx <= len(keys):
                return DATASET_CONFIGS[keys[idx - 1]]
        except ValueError:
            if choice in DATASET_CONFIGS:
                return DATASET_CONFIGS[choice]

        print("Did not understand that choice, try again.")


def interactive_menu(results: AnalysisResults):
    """
    Simple text menu to let you pick which figure to generate.
    """
    while True:
        print("\nAvailable figures:")
        for num in sorted(FIGURE_DESCRIPTIONS):
            print(f"  {num:2d}: {FIGURE_DESCRIPTIONS[num]}")
        print("  all: generate all figures")
        print("  q:   quit")

        choice = input("Figure to generate [q to quit]: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            break
        if choice == "all":
            for num in sorted(FIGURE_DESCRIPTIONS):
                plot_figure(results, num)
            continue
        if not choice:
            continue

        try:
            num = int(choice)
        except ValueError:
            print("Please enter a number, 'all', or 'q'.")
            continue

        plot_figure(results, num)


# ----------------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Polariscope analysis with selectable data set and figure generation."
    )
    parser.add_argument(
        "tif_path",
        nargs="?",
        help="Optional TIFF path. If omitted and no --dataset is given, a menu is shown.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Preconfigured data set key to use (for example fork or plastic).",
    )
    parser.add_argument(
        "-f",
        "--figure",
        help=(
            "Figure number (for example 1) or comma separated list "
            "(for example 1,2,6) or 'all'. "
            "If omitted, an interactive figure menu is shown."
        ),
    )
    parser.add_argument(
        "--box",
        type=int,
        default=10,
        help="Sliding window size in pixels (box_h = box_w = this value).",
    )

    args = parser.parse_args()
    box_h = box_w = args.box

    # Decide which data set to use
    if args.dataset:
        base_cfg = DATASET_CONFIGS[args.dataset]
        tif_path = args.tif_path or base_cfg.tif_path
        dataset_cfg = DatasetConfig(
            key=base_cfg.key,
            label=base_cfg.label,
            tif_path=tif_path,
            rois=base_cfg.rois,
        )
    elif args.tif_path:
        dataset_cfg = DatasetConfig(
            key="custom",
            label=f"Custom ({args.tif_path})",
            tif_path=args.tif_path,
            rois={},
        )
    else:
        dataset_cfg = choose_dataset_interactive()

    print(
        f"Running analysis for {dataset_cfg.tif_path} "
        f"with box {box_h}x{box_w}..."
    )
    results = run_analysis(dataset_cfg, box_h=box_h, box_w=box_w)

    # Figure selection, same logic as before
    if args.figure:
        spec = args.figure.strip().lower()
        if spec == "all":
            nums: List[int] = sorted(FIGURE_DESCRIPTIONS)
        else:
            nums = []
            for part in spec.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    nums.append(int(part))
                except ValueError:
                    print(f"Ignoring invalid figure spec '{part}'.")
        for num in nums:
            plot_figure(results, num)
    else:
        interactive_menu(results)



if __name__ == "__main__":
    main()
