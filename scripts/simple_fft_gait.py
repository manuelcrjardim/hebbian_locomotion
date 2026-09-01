"""
Gait-frequency measurement.

For each trained network, measures the fundamental frequency of the walking
gait from the body-height signal via the discrete Fourier transform, and
writes a per-network spectrum figure plus a summary JSON/CSV.

Operates on the .npz files written by pca.py -- no Isaac Sim launch.

Method
------
The body height z bobs once per gait cycle, so its dominant spectral peak is
the gait fundamental f_gait, and T_gait = 1 / f_gait. Forward velocity v_x is
transformed as a secondary signal and reported alongside, but z defines the
period.

Before transforming, each signal is linearly detrended (removing slow drift as
the robot settles to its steady ride height) and multiplied by a Hann window
(the abrupt end of a finite record is itself a discontinuity, which the FFT
would otherwise smear across all frequencies). The window is compensated by
its coherent gain so amplitudes stay on the original scale.

Edit the CONFIG block, then:

    python gait_frequency.py
"""

# ===========================================================================
# CONFIG
# ===========================================================================

# Directory of *_data.npz files written by pca.py. Every file in it is
# analysed; the label is the filename with "_data" stripped. Groups come from
# manifest.json in the same directory when present, else from the label.
NPZ_DIR = ("/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/analysis/PCA_08:27-14:42_mu/npz")

from datetime import datetime
current_time = datetime.now().strftime("%m:%d-%H:%M")

TAG = f"{current_time}_gait_frequency"

OUTDIR = ("/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/"
          f"isaac_lab_sandbox/workspace/hebbian_locomotion/analysis/spectrum/{TAG}")

TAG = 'gait_frequency'

DT = 0.02            # control step, s (decimation 4 x sim.dt 0.005)
SKIP_STEPS = 100     # drop the startup transient before transforming
F_MIN = 0.2          # ignore peaks below this (residual drift), Hz
F_MAX = 12.0         # x-axis limit for the spectrum figures, Hz
USE_HANN = True      # Hann window: less spectral leakage, wider peaks

PER_NETWORK_FIGS = True   # one spectrum figure per network
# ===========================================================================

import glob as _glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import detrend, find_peaks


# ===========================================================================
# Figure style
# ===========================================================================
try:
    from cmcrameri import cm as cmc
    _ROMA = cmc.roma
except ImportError:
    _ROMA = plt.get_cmap("Spectral")

BG = "#F6F5F1"
FG = "#22252A"
GRID = "#E4E2DC"

FIG_W_FULL = 5.5
FIG_W_HALF = 2.7
FIG_W_THIRD = 1.75

# --- Unordered categorical (e.g. distinct signals, freeze x lock cells) ----
CAT = ["#1A5E63", "#D9A441", "#B4553F", "#6E4A7E"]


def ramp(n, start=0.22, stop=0.92):
    """n ORDERED categorical colours (e.g. M values), sampled from roma.

    Starts ~22% along so the lightest group stays legible on the off-white
    canvas, and stops short of the far end so a two-colour ramp does not
    reach the extremes.
    """
    if n == 1:
        return [_ROMA(start)]
    return [_ROMA(start + (stop - start) * i / (n - 1)) for i in range(n)]


def tints(base_frac, k, spread=0.10):
    """k dark-biased tints around a ramp position, for seeds within a group."""
    return [_ROMA(min(0.98, max(0.02, base_frac + spread * (i / max(1, k - 1)))))
            for i in range(k)]


def cmap():
    """Continuous colormap (e.g. PCA trajectory), never starting near-white."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "roma_trim", [_ROMA(x) for x in np.linspace(0.03, 0.97, 256)])


def set_pub_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": FG,
        "axes.labelcolor": FG,
        "axes.edgecolor": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # Never savefig.bbox="tight": it crops the canvas to the drawn content,
        # so a figure built at FIG_W_FULL is written narrower, and
        # \includegraphics[width=\linewidth] then rescales every font size.
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[FIG] {name}.pdf / .png")


# ===========================================================================
# Spectral analysis
# ===========================================================================
def spectrum(x):
    """Amplitude spectrum of a 1-D signal, DC bin removed.

    Linear detrend handles slow drift as the robot settles; a Hann window
    suppresses the leakage that a hard cut at the end of the record would
    otherwise smear across all frequencies. Dividing by the window mean
    (coherent gain) restores the original amplitude scale.
    """
    x = detrend(np.asarray(x, dtype=float), type="linear")
    n = len(x)
    if USE_HANN:
        w = np.hanning(n)
        x = x * w
        gain = w.mean()
    else:
        gain = 1.0

    amp = np.abs(np.fft.rfft(x)) / (n * gain) * 2.0
    freqs = np.fft.rfftfreq(n, d=DT)
    return freqs[1:], amp[1:]


def dominant_frequency(freqs, amp):
    """Frequency of the largest spectral peak above F_MIN."""
    mask = freqs >= F_MIN
    f, a = freqs[mask], amp[mask]
    if len(f) == 0:
        return np.nan, np.nan

    peaks, _ = find_peaks(a)
    if len(peaks) == 0:
        i = int(np.argmax(a))
        return float(f[i]), float(a[i])

    i = int(peaks[int(np.argmax(a[peaks]))])
    return float(f[i]), float(a[i])


# ===========================================================================
# Network discovery
# ===========================================================================
def infer_group(label):
    """Group from an M=<n> prefix in the label, else 'other'."""
    m = re.match(r"^M(\d+)", label)
    return f"M={m.group(1)}" if m else "other"


def discover():
    """(group, label, path) for every *_data.npz in NPZ_DIR."""
    paths = sorted(_glob.glob(os.path.join(NPZ_DIR, "*_data.npz")))
    if not paths:
        raise SystemExit(f"No *_data.npz found in {NPZ_DIR}")

    manifest_path = os.path.join(NPZ_DIR, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)

    out = []
    for p in paths:
        label = os.path.basename(p).replace("_data.npz", "")
        group = manifest.get(label, {}).get("group") or infer_group(label)
        out.append((group, label, p))
    return out


# ===========================================================================
# Per-network analysis
# ===========================================================================
def analyse(group, label, npz_path):
    d = np.load(npz_path)
    z_full, vx_full = d["z"], d["vx"]

    if SKIP_STEPS >= len(z_full) - 32:
        raise ValueError(f"{label}: SKIP_STEPS leaves too little signal.")

    z, vx = z_full[SKIP_STEPS:], vx_full[SKIP_STEPS:]
    n = len(z)
    duration = n * DT

    f_z, a_z = spectrum(z)
    f_v, a_v = spectrum(vx)

    fz_peak, _ = dominant_frequency(f_z, a_z)
    fv_peak, _ = dominant_frequency(f_v, a_v)

    # Body height defines the period: for a lunging gait it completes one
    # oscillation per stride. The f_z / f_vx ratio corroborates that reading:
    # a lunge drives body height and forward velocity together (ratio ~ 1),
    # whereas a trot pulses v_x twice per stride (ratio ~ 0.5). A ratio ~ 2
    # would mean z is tracking half-strides and T_gait is half a cycle.
    f_gait = fz_peak
    T_gait = 1.0 / f_gait if np.isfinite(f_gait) and f_gait > 0 else np.nan

    if np.isfinite(fz_peak) and np.isfinite(fv_peak) and fv_peak > 0:
        fz_fv_ratio = fz_peak / fv_peak
        if abs(fz_fv_ratio - 1.0) < 0.15:
            gait_type = "lunge (z and v_x in step)"
        elif abs(fz_fv_ratio - 0.5) < 0.10:
            gait_type = "trot (v_x at twice the stride rate)"
        elif abs(fz_fv_ratio - 2.0) < 0.15:
            gait_type = "CHECK: z may track half-strides"
        else:
            gait_type = "unclassified"
    else:
        fz_fv_ratio, gait_type = np.nan, "unclassified"

    summary = {
        "group": group,
        "label": label,
        "n_analysed": int(n),
        "duration_s": duration,
        "freq_resolution_hz": 1.0 / duration,
        "f_peak_z_hz": fz_peak,
        "f_peak_vx_hz": fv_peak,
        "fz_fv_ratio": fz_fv_ratio,
        "gait_type": gait_type,
        "f_gait_hz": f_gait,
        "T_gait_s": T_gait,
        "T_gait_steps": T_gait / DT if np.isfinite(T_gait) else np.nan,
    }
    arrays = {"f_z": f_z, "a_z": a_z, "f_v": f_v, "a_v": a_v}
    return summary, arrays


# ===========================================================================
# Figures
# ===========================================================================
def fig_one_network(arrays, s):
    """Normalised amplitude spectra of body height and forward velocity."""
    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.42))

    # Body height and forward velocity are unordered categories -> CAT.
    ax.plot(arrays["f_z"], arrays["a_z"] / arrays["a_z"].max(),
            color=CAT[0], label=r"body height $z$")
    ax.plot(arrays["f_v"], arrays["a_v"] / arrays["a_v"].max(),
            color=CAT[1], label=r"forward velocity $v_x$", alpha=0.85)

    f_gait = s["f_gait_hz"]
    if np.isfinite(f_gait):
        ax.axvline(f_gait, color=FG, ls="--", lw=0.8, zorder=0)
        ax.annotate(rf"$f_\mathrm{{gait}} = {f_gait:.2f}$ Hz",
                    xy=(f_gait, 1.0), xytext=(4, -2),
                    textcoords="offset points", fontsize=8, color=FG)

    ax.set_xlim(0, F_MAX)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalised amplitude")
    ax.legend(loc="upper right")
    save_fig(fig, f"{TAG}_{s['label']}_spectrum")


def fig_period_by_group(summaries):
    """T_gait for every network, grouped, showing the within-group spread."""
    groups = sorted({s["group"] for s in summaries})
    cols = ramp(len(groups))

    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.42))
    for gi, g in enumerate(groups):
        vals = [s["T_gait_s"] for s in summaries
                if s["group"] == g and np.isfinite(s["T_gait_s"])]
        if not vals:
            continue
        x = np.full(len(vals), gi, dtype=float)
        x += np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=18, color=cols[gi], zorder=3,
                   edgecolor="none", alpha=0.9)
        ax.hlines(np.median(vals), gi - 0.24, gi + 0.24,
                  color=FG, lw=1.2, zorder=4)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_xlabel("averaging window")
    ax.set_ylabel(r"$T_\mathrm{gait}$ (s)")
    save_fig(fig, f"{TAG}_period_by_group")


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    set_pub_style()

    networks = discover()
    print(f"[INFO] {len(networks)} networks found in {NPZ_DIR}")

    summaries, arrays_by_label = [], {}
    for group, label, path in networks:
        try:
            s, a = analyse(group, label, path)
        except Exception as exc:
            print(f"[WARN] {label}: {exc}")
            continue
        summaries.append(s)
        arrays_by_label[label] = a
        print(f"[OK]   {label:24s} group={s['group']:8s} "
              f"f_gait={s['f_gait_hz']:.3f} Hz  T_gait={s['T_gait_s']:.3f} s "
              f"({s['T_gait_steps']:.1f} steps)  "
              f"f_z/f_vx={s['fz_fv_ratio']:.2f}  {s['gait_type']}")

    if not summaries:
        raise SystemExit("No networks analysed.")

    if PER_NETWORK_FIGS:
        for s in summaries:
            fig_one_network(arrays_by_label[s["label"]], s)

    fig_period_by_group(summaries)

    json_path = os.path.join(OUTDIR, f"{TAG}_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"[OUT] {json_path}")

    csv_path = os.path.join(OUTDIR, f"{TAG}_summary.csv")
    cols = ["group", "label", "f_gait_hz", "T_gait_s", "T_gait_steps",
            "f_peak_vx_hz", "fz_fv_ratio", "gait_type",
            "duration_s", "freq_resolution_hz"]
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for s in summaries:
            fh.write(",".join(str(s[c]) for c in cols) + "\n")
    print(f"[OUT] {csv_path}")


if __name__ == "__main__":
    main()