"""
Gait-frequency and co-dynamic spectral analysis, over an arbitrary set of networks.

Operates on the .npz files written by pca.py -- no Isaac Sim launch, runs in
seconds per network.

Answers four questions:

  1. What is the gait fundamental frequency f_gait (and hence T_gait) for EACH
     network? Measured from body height z and forward velocity v_x.

  2. Do the plastic weights oscillate at the same frequency as the body?
     Same peaks  -> weights are slaved to the gait (co-dynamic).
     Extra peak  -> some autonomous internal rhythm.
     (The decisive test is still the constant-observation rollout; this is
     the cheap spectral version of the same question.)

  3. Given T_gait, what M nulls the gait? The moving average of length M is a
     comb filter with transfer function

         |H(f)| = |sin(pi f M dt) / (M sin(pi f dt))|

     whose zeros sit at f = k / (M dt). When M*dt is an integer multiple of
     T_gait, the gait fundamental AND all its harmonics land exactly on zeros,
     the averaged activation becomes constant, and the Hebbian drive vanishes.
     The dimensionless window is

         M_hat = M * dt / T_gait

     M_hat = 2.0 reproduces the strongest fixed-point condition in Dittrich
     et al.; a non-integer M_hat is the off-null control that distinguishes
     comb nulling from plain low-pass smoothing.

  4. WHEN should each network be ablated? Every network has its OWN T_gait, so
     a global step spacing would sample a different arc of the cycle for each
     one, and the mean over strata would become a phase-WEIGHTED average with
     different weights per network -- sitting directly on top of the retention
     effect being measured. This script emits per-network offsets

         t_k = T_ablate + round(k * T_gait_steps / N_ABLATIONS),  k = 0..N-1

     so each network's ablations span exactly one of ITS OWN cycles at evenly
     spaced phases. Note the spacing is T/N, not T/(N-1): with T/(N-1) the last
     stratum lands back on phase 0 and you lose a stratum to duplication.

     The offsets are written to <TAG>_ablation_offsets.json, keyed by network
     label, ready to be read by the eval script as a lookup.

FALLBACK RULE (decided in advance, not after seeing which networks fail):
     If a network's spectral peak has prominence below PEAK_PROM_MIN (relative
     to the largest amplitude), its T_gait is replaced by the MEDIAN T_gait of
     its group, and the substitution is recorded in the JSON under
     "t_gait_source". Deciding this rule after inspecting the failures is where
     a defensible choice turns into a post-hoc one.

Edit the CONFIG block, then:

    python gait_spectrum.py
"""

# ===========================================================================
# CONFIG -- edit everything here
# ===========================================================================

# ---------------------------------------------------------------------------
# NETWORK DISCOVERY -- set NPZ_DIR and nothing else in the normal case.
#
# Point NPZ_DIR at the npz/ directory pca.py wrote. Every *_data.npz in it is
# analysed; the label is the filename with "_data" stripped.
#
# Groups come from pca.py's manifest.json when it is present in that directory
# (exact, recorded at write time). Without a manifest the group is inferred
# from the label -- see infer_group() for the rule and its limits.
# ---------------------------------------------------------------------------
NPZ_DIR = ("/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/analysis/PCA_08:13-18:58_mu/npz")

# Everything below is an escape hatch, used only when NPZ_DIR is None.
BASE = ("/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/"
        "workspace/hebbian_locomotion/analysis")

# Explicit entries: (group, label, npz_path).
NETWORKS = [
    # ("M=1",   "M1_s0",   f"{BASE}/M1_s0_data.npz"),
    # ("M=160", "M160_s0", f"{BASE}/M160_s0_data.npz"),
]

# A specific manifest.json somewhere other than NPZ_DIR.
MANIFEST = None   # e.g. f"{BASE}/PCA_08:13-10:00_mu/npz/manifest.json"

# Glob patterns: (group, glob); the label is taken from the filename stem.
NETWORK_GLOBS = [
    # ("M=1",   f"{BASE}/M1_s*_data.npz"),
]

TAG = "gait_spectrum"
OUTDIR = (
    "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/"
    "workspace/hebbian_locomotion/analysis/spectrum"
)

DT = 0.02                  # control step, s (decimation 4 x sim.dt 0.005)
SKIP_STEPS = 100           # drop the startup transient before transforming
F_MIN = 0.2                # ignore peaks below this (residual drift), Hz
F_MAX = 12.0               # x-axis limit for the spectrum figure, Hz
USE_HANN = True            # Hann window: less spectral leakage, wider peaks
N_HARMONICS = 3            # how many harmonics to report

# --- Ablation stratification ----------------------------------------------
T_ABLATE = 500             # base step at which the first stratum is ablated
N_ABLATIONS = 5            # strata per network; offsets are T_gait/N apart

# --- Peak-quality gate (the fallback rule above) --------------------------
ACF_MIN = 0.30             # min normalised autocorrelation at the gait lag.
                           # This is the gate, and it is deliberately NOT a
                           # spectral one. Neither peak prominence nor spectral
                           # concentration can separate a periodic gait from
                           # red noise: a random walk has a large, prominent,
                           # tightly concentrated low-frequency peak and no
                           # periodicity whatsoever. Autocorrelation asks the
                           # question directly -- does the signal repeat at
                           # this lag? -- and red noise decays monotonically
                           # with no secondary ACF peak.
ACF_TOL = 0.20             # accept the ACF period if within this fraction of
                           # the FFT period; otherwise the two disagree and the
                           # network is flagged.
HARMONIC_CHECK = True      # warn when f_z ~ 2*f_vx (stride vs half-stride)

# --- Figures ---------------------------------------------------------------
PER_NETWORK_FIGS = False   # one spectra/window figure per network. With 25
                           # networks that is 50+ files; off by default, the
                           # cross-network figures carry the story.
CANDIDATE_M = [1, 10, 20, 50, 110, 160]   # windows scored against each T_gait
PLOT_M = [1, 110, 160]                    # windows drawn on the filter figure
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
# Figure style -- ICLR 2027 (5.5 in text block, 10 pt Times)
# ===========================================================================
OKABE_ITO = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "verm":   "#D55E00",
    "purple": "#CC79A7",
}
CYCLE = [OKABE_ITO[k] for k in
         ("blue", "orange", "green", "verm", "purple", "sky", "yellow", "black")]

FIG_W_FULL = 5.5
FIG_W_HALF = 2.65


def set_pub_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # Do NOT use savefig.bbox="tight". It crops the canvas to the drawn
        # content, so a figure built at FIG_W_FULL is written narrower than
        # FIG_W_FULL; \includegraphics[width=\linewidth] then rescales it and
        # every font size above becomes wrong. constrained_layout packs the
        # axes inward instead, leaving the canvas at exactly the width asked
        # for, so the point sizes here are the true printed point sizes.
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[FIG] {name}.pdf / .png")


# ===========================================================================
# Spectral helpers
# ===========================================================================
def spectrum(x):
    """Amplitude spectrum of a 1-D signal. Returns (freqs, amp), DC removed.

    Linear detrend handles the slow drift in body height as the robot settles;
    a Hann window suppresses the spectral leakage that a hard cut at the end
    of the record would otherwise smear across all frequencies.
    """
    x = detrend(np.asarray(x, dtype=float), type="linear")
    n = len(x)
    if USE_HANN:
        w = np.hanning(n)
        x = x * w
        gain = w.mean()          # coherent gain, restores amplitude scale
    else:
        gain = 1.0

    amp = np.abs(np.fft.rfft(x)) / (n * gain) * 2.0
    freqs = np.fft.rfftfreq(n, d=DT)
    return freqs[1:], amp[1:]    # drop the DC bin


def spectrum_matrix(X):
    """Mean amplitude spectrum across the columns of X (steps x n_signals).

    Incoherent averaging: transform each weight separately and average the
    magnitudes. This preserves a shared frequency even when weights differ in
    phase, which coherent averaging would cancel.
    """
    X = detrend(np.asarray(X, dtype=float), axis=0, type="linear")
    n = X.shape[0]
    if USE_HANN:
        w = np.hanning(n)[:, None]
        X = X * w
        gain = w.mean()
    else:
        gain = 1.0

    amp = np.abs(np.fft.rfft(X, axis=0)) / (n * gain) * 2.0
    freqs = np.fft.rfftfreq(n, d=DT)
    return freqs[1:], amp[1:].mean(axis=1)


def dominant_frequency(freqs, amp):
    """Largest spectral peak above F_MIN.

    Returns (f_peak, amp_peak, prominence_frac), where prominence_frac is the
    peak's prominence divided by the largest amplitude in the band -- a
    scale-free measure of how well defined the peak is. A clean periodic gait
    gives ~1.0; broadband, aperiodic motion gives a small number, and that is
    what PEAK_PROM_MIN gates on.
    """
    mask = freqs >= F_MIN
    f, a = freqs[mask], amp[mask]
    if len(f) == 0:
        return np.nan, np.nan, 0.0

    peaks, props = find_peaks(a, prominence=0.0)
    if len(peaks) == 0:
        i = int(np.argmax(a))
        return float(f[i]), float(a[i]), 0.0

    j = int(np.argmax(a[peaks]))
    i = int(peaks[j])
    prom = float(props["prominences"][j])
    denom = float(a.max())
    return float(f[i]), float(a[i]), (prom / denom if denom > 0 else 0.0)


def acf_periodicity(x):
    """Periodicity strength and period from the normalised autocorrelation.

    Returns (acf_peak_height, period_steps).

    The first ACF maximum AFTER the first zero crossing is the natural period
    estimate. Its height on a [-1, 1] scale is a direct measure of how well the
    signal repeats: a clean gait gives ~0.8-1.0, red noise or aperiodic motion
    gives a small value because the ACF simply decays without a secondary peak.

    Searching after the zero crossing (rather than after lag 0) is what makes
    this robust: it skips the central lobe, which is large for ANY smooth
    signal and carries no periodicity information.
    """
    x = detrend(np.asarray(x, dtype=float), type="linear")
    n = len(x)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0, np.nan

    # FFT autocorrelation, zero-padded to avoid circular wraparound.
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(F * np.conj(F), nfft)[:n] / denom

    # First zero crossing, then the largest peak in the plausible band.
    neg = np.where(ac < 0)[0]
    if len(neg) == 0:
        return 0.0, np.nan
    start = int(neg[0])

    lag_max = min(n - 1, int(round(1.0 / (F_MIN * DT))))
    if start >= lag_max:
        return 0.0, np.nan

    seg = ac[start:lag_max]
    peaks, _ = find_peaks(seg)
    if len(peaks) == 0:
        return float(seg.max()), np.nan
    i = int(peaks[np.argmax(seg[peaks])])
    return float(seg[i]), float(start + i)


def harmonic_amplitudes(freqs, amp, f0, n_harm):
    """Amplitude at f0, 2*f0, ... n_harm*f0 (nearest bin)."""
    out = []
    for k in range(1, n_harm + 1):
        target = k * f0
        if target > freqs[-1]:
            out.append(None)
            continue
        i = int(np.argmin(np.abs(freqs - target)))
        out.append({"harmonic": k,
                    "freq_hz": float(freqs[i]),
                    "amplitude": float(amp[i])})
    return out


def boxcar_response(freqs, M):
    """|H(f)| for an M-sample moving average. Zeros at f = k / (M*dt)."""
    num = np.sin(np.pi * freqs * M * DT)
    den = M * np.sin(np.pi * freqs * DT)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.abs(np.where(np.abs(den) < 1e-12, 1.0, num / den))
    return h


def ablation_offsets(t_gait_steps, t_ablate=T_ABLATE, n=N_ABLATIONS):
    """Evenly phase-spaced ablation steps covering ONE gait cycle.

    Spacing is T/n, not T/(n-1): with T/(n-1) the last stratum lands back on
    phase 0 and duplicates the first, costing a stratum and overweighting that
    phase in the mean.
    """
    k = np.arange(n)
    offs = np.rint(k * t_gait_steps / n).astype(int)
    return {
        "t_ablate_base": int(t_ablate),
        "n_ablations": int(n),
        "offsets_steps": offs.tolist(),
        "steps": (t_ablate + offs).tolist(),
        "phases_deg": (360.0 * offs / t_gait_steps).round(1).tolist(),
        "spacing_steps": float(t_gait_steps / n),
    }


# ===========================================================================
# Per-network analysis
# ===========================================================================
def analyse_network(group, label, npz_path):
    """Full spectral analysis of one network. Returns a summary dict + arrays."""
    d = np.load(npz_path)

    z_full, vx_full = d["z"], d["vx"]
    n_total = len(z_full)
    if SKIP_STEPS >= n_total - 32:
        raise ValueError(f"{label}: SKIP_STEPS leaves too little signal.")

    z, vx = z_full[SKIP_STEPS:], vx_full[SKIP_STEPS:]
    n = len(z)
    duration = n * DT
    df = 1.0 / duration

    f_z, a_z = spectrum(z)
    f_v, a_v = spectrum(vx)

    if "W_traj" in d:
        f_w, a_w = spectrum_matrix(d["W_traj"][SKIP_STEPS:])
        w_source = "W_traj"
    else:
        f_w, a_w = spectrum(d["dW"][SKIP_STEPS:])
        w_source = "dW"

    fz_peak, _, fz_prom = dominant_frequency(f_z, a_z)
    fv_peak, _, fv_prom = dominant_frequency(f_v, a_v)
    fw_peak, _, fw_prom = dominant_frequency(f_w, a_w)
    acf_h, acf_T = acf_periodicity(z)

    # Body height is the most reliable gait indicator: it bobs once per stride
    # regardless of whether forward velocity is cleanly periodic.
    f_gait = fz_peak
    T_gait = 1.0 / f_gait if np.isfinite(f_gait) and f_gait > 0 else np.nan

    # Stride vs half-stride ambiguity. In a trot both diagonal pairs land per
    # stride, so z and v_x can peak at 2*f_stride. If z sits at ~2x v_x the two
    # observables disagree about what one cycle IS, and the offsets would cover
    # half a stride rather than a whole one.
    harmonic_flag = None
    if HARMONIC_CHECK and np.isfinite(fz_peak) and np.isfinite(fv_peak) and fv_peak > 0:
        ratio = fz_peak / fv_peak
        # For a trot both diagonal pairs land per stride, so v_x commonly sits
        # at 2*f_stride while z sits at the stride rate. That is EXPECTED and is
        # reported as a count, not a per-network warning. What would be a real
        # problem is the reverse (z at twice v_x), which would mean the offsets
        # cover half a stride.
        if abs(ratio - 2.0) < 0.15:
            harmonic_flag = "f_z ~ 2*f_vx (z may track half-strides)"
        elif abs(ratio - 0.5) < 0.10:
            harmonic_flag = "f_z ~ 0.5*f_vx (expected for a trot)"

    # Cross-check: the ACF period is an independent estimate. Disagreement
    # usually means the FFT locked onto a harmonic (half or double the true
    # stride), which would make the ablation offsets cover the wrong cycle.
    period_flag = None
    T_fft_steps = (1.0 / fz_peak / DT) if (np.isfinite(fz_peak) and fz_peak > 0) else np.nan
    if np.isfinite(acf_T) and np.isfinite(T_fft_steps) and T_fft_steps > 0:
        r = acf_T / T_fft_steps
        if abs(r - 1.0) > ACF_TOL:
            period_flag = (f"FFT/ACF period disagree: FFT {T_fft_steps:.1f} vs "
                           f"ACF {acf_T:.1f} steps (ratio {r:.2f})")

    summary = {
        "group": group,
        "label": label,
        "npz": os.path.abspath(npz_path),
        "n_analysed": int(n),
        "duration_s": duration,
        "freq_resolution_hz": df,
        "weight_signal": w_source,
        "f_peak_z_hz": fz_peak,
        "f_peak_vx_hz": fv_peak,
        "f_peak_weights_hz": fw_peak,
        "prom_z": fz_prom,
        "acf_peak_z": acf_h,
        "T_gait_steps_acf": acf_T,
        "prom_vx": fv_prom,
        "prom_weights": fw_prom,
        "f_gait_hz": f_gait,
        "T_gait_s": T_gait,
        "T_gait_steps": T_gait / DT if np.isfinite(T_gait) else np.nan,
        "peak_ok": bool(acf_h >= ACF_MIN),
        "harmonic_flag": harmonic_flag,
        "period_flag": period_flag,
        "harmonics_z": harmonic_amplitudes(f_z, a_z, f_gait, N_HARMONICS),
        "harmonics_weights": harmonic_amplitudes(f_w, a_w, f_gait, N_HARMONICS),
        "codynamic_rel_err": (abs(fw_peak - f_gait) / f_gait
                              if np.isfinite(f_gait) and f_gait > 0 else np.nan),
    }

    arrays = {"f_z": f_z, "a_z": a_z, "f_v": f_v, "a_v": a_v,
              "f_w": f_w, "a_w": a_w, "z": z, "vx": vx}
    return summary, arrays


def apply_fallback(summaries):
    """Replace low-quality T_gait with the group median. Records the source.

    Applied to every network uniformly, using a threshold fixed in the CONFIG
    block before any network is inspected.
    """
    groups = {}
    for s in summaries:
        if s["peak_ok"] and np.isfinite(s["T_gait_steps"]):
            groups.setdefault(s["group"], []).append(s["T_gait_steps"])

    medians = {g: float(np.median(v)) for g, v in groups.items() if v}
    all_ok = [t for v in groups.values() for t in v]
    global_median = float(np.median(all_ok)) if all_ok else np.nan

    for s in summaries:
        if s["peak_ok"] and np.isfinite(s["T_gait_steps"]):
            s["T_gait_steps_used"] = s["T_gait_steps"]
            s["t_gait_source"] = "measured"
        elif s["group"] in medians:
            s["T_gait_steps_used"] = medians[s["group"]]
            s["t_gait_source"] = f"group_median({s['group']})"
        else:
            s["T_gait_steps_used"] = global_median
            s["t_gait_source"] = "global_median"
    return medians, global_median


# ===========================================================================
# Figures
# ===========================================================================
def fig_spectra_one(arrays, s):
    """Normalised spectra of body height, forward velocity and plastic weights."""
    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.45))
    f_gait = s["f_gait_hz"]

    for f, a, c, lab in (
            (arrays["f_z"], arrays["a_z"], OKABE_ITO["purple"], r"body height $z$"),
            (arrays["f_v"], arrays["a_v"], OKABE_ITO["orange"], r"forward velocity $v_x$"),
            (arrays["f_w"], arrays["a_w"], OKABE_ITO["blue"], r"plastic weights $W$")):
        ax.plot(f, a / a.max(), color=c, label=lab, lw=1.0)

    for k in range(1, N_HARMONICS + 1):
        ax.axvline(k * f_gait, color=OKABE_ITO["black"], ls=":", lw=0.7,
                   alpha=0.6, zorder=0)

    ax.set_xlim(0, F_MAX)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalised amplitude")
    ax.legend(loc="upper right")
    ax.set_title(rf"{s['label']}: $f_\mathrm{{gait}} = {f_gait:.2f}$ Hz "
                 rf"($T_\mathrm{{gait}} = {1/f_gait:.2f}$ s)")
    save_fig(fig, f"{TAG}_{s['label']}_spectra")


def fig_analysed_window_one(arrays, s):
    """Sanity check: the segment actually transformed, transient excluded."""
    z, vx = arrays["z"], arrays["vx"]
    steps = np.arange(len(z)) + SKIP_STEPS
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W_FULL, FIG_W_FULL * 0.50),
                             sharex=True)
    axes[0].plot(steps, z, color=OKABE_ITO["purple"])
    axes[0].set_ylabel(r"$z$ (m)")
    axes[1].plot(steps, vx, color=OKABE_ITO["orange"])
    axes[1].set_ylabel(r"$v_x$ (m s$^{-1}$)")
    axes[1].set_xlabel("step")
    for a in axes:
        a.set_xlim(steps[0], steps[-1])
    save_fig(fig, f"{TAG}_{s['label']}_analysed_window")


def fig_spectra_by_group(results):
    """One panel per group; every network's z-spectrum overlaid.

    This is the figure that shows whether networks within a group share a gait
    frequency, which is the premise the group-median fallback rests on.
    """
    groups = sorted({s["group"] for s, _ in results})
    nrow = len(groups)
    # Height scales with group count but is capped: the ICLR text block is 9 in
    # tall, and a figure plus caption must fit on one page.
    aspect = min(0.22 * nrow + 0.08, 1.30)
    fig, axes = plt.subplots(nrow, 1,
                             figsize=(FIG_W_FULL, FIG_W_FULL * aspect),
                             sharex=True, squeeze=False)

    for r, g in enumerate(groups):
        ax = axes[r, 0]
        members = [(s, a) for s, a in results if s["group"] == g]
        for i, (s, a) in enumerate(members):
            ax.plot(a["f_z"], a["a_z"] / a["a_z"].max(),
                    color=CYCLE[i % len(CYCLE)], lw=0.9, label=s["label"])
            ax.axvline(s["f_gait_hz"], color=CYCLE[i % len(CYCLE)],
                       ls=":", lw=0.7, alpha=0.7, zorder=0)
        ax.set_xlim(0, F_MAX)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{g}\nnorm. amp.")
        ax.legend(loc="upper right", ncol=3, fontsize=6.5)

    axes[-1, 0].set_xlabel("frequency (Hz)")
    save_fig(fig, f"{TAG}_spectra_by_group")


def fig_tgait_across_networks(summaries):
    """Measured T_gait per network, grouped. The spread here is exactly why the
    ablation offsets cannot be a global constant."""
    order = sorted(range(len(summaries)),
                   key=lambda i: (summaries[i]["group"], summaries[i]["label"]))
    s_ord = [summaries[i] for i in order]
    groups = sorted({s["group"] for s in s_ord})
    gcol = {g: CYCLE[i % len(CYCLE)] for i, g in enumerate(groups)}

    x = np.arange(len(s_ord))
    t_meas = np.array([s["T_gait_steps"] for s in s_ord], dtype=float)
    t_used = np.array([s["T_gait_steps_used"] for s in s_ord], dtype=float)
    sub = np.array([s["t_gait_source"] != "measured" for s in s_ord])

    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.42))
    for g in groups:
        m = np.array([s["group"] == g for s in s_ord])
        ax.bar(x[m], t_used[m], 0.7, color=gcol[g], label=g)
    if sub.any():
        ax.plot(x[sub], t_used[sub], ls="none", marker="v", ms=4,
                mfc="white", mec=OKABE_ITO["black"], mew=0.7,
                label="substituted", zorder=4)
        ax.plot(x[sub], t_meas[sub], ls="none", marker="x", ms=4,
                color="0.45", mew=0.7, label="measured (rejected)", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in s_ord], rotation=60, ha="right")
    ax.set_ylabel(r"$T_\mathrm{gait}$ (steps)")
    ax.legend(loc="upper right", ncol=2)
    save_fig(fig, f"{TAG}_T_gait_across_networks")


def fig_filter_by_group(results, summaries):
    """Boxcar transfer functions against each group's mean gait frequency."""
    by_group = {}
    for s in summaries:
        if np.isfinite(s["f_gait_hz"]):
            by_group.setdefault(s["group"], []).append(s["f_gait_hz"])
    groups = sorted(by_group)

    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.45))
    f = np.linspace(1e-6, F_MAX, 4000)

    for M, c in zip(PLOT_M, CYCLE):
        ax.plot(f, boxcar_response(f, M), color=c, lw=1.2, label=rf"$M={M}$")

    for i, g in enumerate(groups):
        fg = float(np.mean(by_group[g]))
        ax.axvline(fg, color=OKABE_ITO["black"], ls=":", lw=0.8, alpha=0.7,
                   zorder=0)
        ax.annotate(g, xy=(fg, 1.0), xytext=(2, -2),
                    textcoords="offset points", rotation=90,
                    va="top", ha="left", fontsize=7)

    ax.set_xlim(0, F_MAX)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"$|H(f)|$")
    ax.legend(loc="upper right")
    save_fig(fig, f"{TAG}_boxcar_response")


def fig_ablation_phases(summaries):
    """Where each network's ablations land within its own gait cycle.

    Plotted in NORMALISED phase, so uniform coverage looks identical across
    networks with different periods. Non-uniform rows would mean the offsets
    are not doing their job.
    """
    order = sorted(range(len(summaries)),
                   key=lambda i: (summaries[i]["group"], summaries[i]["label"]))
    s_ord = [summaries[i] for i in order]

    # Same page-height cap as above; with 25 networks this lands near 6.9 in.
    aspect = min(0.045 * len(s_ord) + 0.12, 1.30)
    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * aspect))
    groups = sorted({s["group"] for s in s_ord})
    gcol = {g: CYCLE[i % len(CYCLE)] for i, g in enumerate(groups)}

    for r, s in enumerate(s_ord):
        ph = np.array(s["ablation"]["phases_deg"], dtype=float)
        ax.plot(ph, np.full_like(ph, r), ls="none", marker="o", ms=3.5,
                color=gcol[s["group"]])
        ax.axhline(r, color="0.85", lw=0.5, zorder=0)

    ax.set_yticks(np.arange(len(s_ord)))
    ax.set_yticklabels([s["label"] for s in s_ord], fontsize=6.5)
    ax.set_ylim(-0.7, len(s_ord) - 0.3)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("gait phase at ablation (deg)")
    ax.invert_yaxis()
    save_fig(fig, f"{TAG}_ablation_phases")


# ===========================================================================
# Main
# ===========================================================================
def infer_group(label):
    """Best-effort group for a label when no manifest is available.

    Rule: the first M-like token in the label. "M1_s860896728" -> "M=1",
    "M_160_523277674" -> "M=160", "HAN_M_20_seed7" -> "M=20". A label with no
    M token falls back to its leading alphabetic run ("LSTM_s3" -> "LSTM"),
    and failing that to "ungrouped".

    This is a heuristic and it can be wrong -- a seed number that happens to
    lead with digits, or a naming scheme this pattern does not anticipate. The
    manifest.json written by pca.py records the group exactly, so prefer that
    whenever it exists; the grouping only matters for the group-median fallback
    and the cross-network figures, but both are wrong if the grouping is wrong.
    """
    m = re.search(r"(?:^|[^A-Za-z0-9])M_?(\d+)(?:$|[^0-9])", label)
    if m:
        return f"M={m.group(1)}"
    head = label.split("_")[0]
    if re.search(r"[A-Za-z]", head):
        return head          # "LSTM_s3" -> "LSTM", "D2_global_s5" -> "D2"
    return "ungrouped"


def _from_manifest(path):
    """Read a pca.py manifest.json -> [(group, label, npz_path)]."""
    with open(path) as fh:
        man = json.load(fh)
    npz_dir = man.get("npz_dir") or os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(npz_dir):
        # The manifest records an absolute path from write time; if the
        # analysis directory has since moved, fall back to the manifest's own
        # location, which is where the npz files actually are.
        npz_dir = os.path.dirname(os.path.abspath(path))
    return [(n["group"], n["label"], os.path.join(npz_dir, n["npz"]))
            for n in man["networks"]]


def _from_dir(npz_dir):
    """Every *_data.npz in a directory -> [(group, label, npz_path)].

    Uses the manifest for groups when one is present alongside the files.
    """
    manifest = os.path.join(npz_dir, "manifest.json")
    if os.path.exists(manifest):
        items = _from_manifest(manifest)
        # Analyse whatever is actually on disk, not merely what the manifest
        # lists: a re-run that added networks without rewriting the manifest
        # would otherwise be silently truncated.
        listed = {os.path.abspath(p) for _, _, p in items}
        extra = [p for p in sorted(_glob.glob(os.path.join(npz_dir, "*_data.npz")))
                 if os.path.abspath(p) not in listed]
        for p in extra:
            label = os.path.basename(p)[:-len("_data.npz")]
            items.append((infer_group(label), label, p))
        note = f" (+{len(extra)} not in manifest)" if extra else ""
        print(f"[INFO] {npz_dir}: {len(items)} network(s), groups from "
              f"manifest.json{note}")
        return items

    paths = sorted(_glob.glob(os.path.join(npz_dir, "*_data.npz")))
    items = []
    for p in paths:
        label = os.path.basename(p)[:-len("_data.npz")]
        items.append((infer_group(label), label, p))
    print(f"[INFO] {npz_dir}: {len(items)} network(s), no manifest.json -- "
          f"groups inferred from labels")
    return items


def collect_networks():
    """Resolve the work list.

    Precedence: NPZ_DIR > MANIFEST > NETWORKS + NETWORK_GLOBS.
    """
    if NPZ_DIR:
        if not os.path.isdir(NPZ_DIR):
            raise SystemExit(f"NPZ_DIR is not a directory: {NPZ_DIR}")
        items = _from_dir(NPZ_DIR)
        if not items:
            raise SystemExit(f"No *_data.npz files in {NPZ_DIR}")
        return _dedupe(items)

    if MANIFEST:
        items = _from_manifest(MANIFEST)
        print(f"[INFO] manifest: {len(items)} network(s) from {MANIFEST}")
        return _dedupe(items)

    items = list(NETWORKS)
    for group, pattern in NETWORK_GLOBS:
        for path in sorted(_glob.glob(pattern)):
            label = os.path.splitext(os.path.basename(path))[0]
            label = label.replace("_data", "")
            items.append((group, label, path))

    return _dedupe(items)


def _dedupe(items):
    """Drop repeated labels, keeping the first occurrence."""
    seen, out = set(), []
    for group, label, path in items:
        if label in seen:
            print(f"[WARN] duplicate label {label!r}; keeping the first.")
            continue
        seen.add(label)
        out.append((group, label, path))
    return out


def main():
    set_pub_style()
    os.makedirs(OUTDIR, exist_ok=True)

    nets = collect_networks()
    if not nets:
        raise SystemExit("No networks found. Set NPZ_DIR, or fill NETWORKS.")
    print(f"[INFO] {len(nets)} network(s) to analyse\n")

    results, summaries = [], []
    for group, label, path in nets:
        if not os.path.exists(path):
            print(f"[WARN] missing, skipping: {path}")
            continue
        s, a = analyse_network(group, label, path)
        results.append((s, a))
        summaries.append(s)
        flag = "" if s["peak_ok"] else "  <-- APERIODIC, will substitute"
        print(f"  {label:14s} [{group:6s}] "
              f"f_gait {s['f_gait_hz']:6.3f} Hz  "
              f"T_gait {s['T_gait_steps']:6.1f} steps  "
              f"acf {s['acf_peak_z']:5.2f}{flag}")
        if s["period_flag"]:
            print(f"      [WARN] {s['period_flag']}")
        # The trot half-stride relation is expected; only the reverse is a problem.
        if s["harmonic_flag"] and s["harmonic_flag"].startswith("f_z ~ 2*"):
            print(f"      [WARN] {s['harmonic_flag']}")

    if not summaries:
        raise SystemExit("No networks could be loaded.")

    # --- fallback + per-network ablation offsets ---------------------------
    medians, global_median = apply_fallback(summaries)
    for s in summaries:
        s["ablation"] = ablation_offsets(s["T_gait_steps_used"])

    n_sub = sum(1 for s in summaries if s["t_gait_source"] != "measured")

    print("\n--- gait periods " + "-" * 40)
    for g in sorted(medians):
        print(f"  {g:8s} median T_gait {medians[g]:6.1f} steps "
              f"({medians[g] * DT:.3f} s)")
    if n_sub:
        print(f"  {n_sub} network(s) below ACF_MIN={ACF_MIN}; "
              f"substituted with the group median.")
        for s in summaries:
            if s["t_gait_source"] != "measured":
                print(f"     {s['label']:14s} measured {s['T_gait_steps']:6.1f} "
                      f"(acf {s['acf_peak_z']:.2f})  ->  {s['T_gait_steps_used']:6.1f} "
                      f"[{s['t_gait_source']}]")
    n_trot = sum(1 for s in summaries
                 if s["harmonic_flag"] and s["harmonic_flag"].startswith("f_z ~ 0.5"))
    if n_trot:
        print(f"  {n_trot}/{len(summaries)} network(s) show v_x at twice the z rate "
              f"(expected for a trot).")

    print("\n--- co-dynamic check " + "-" * 36)
    for s in summaries:
        rel = s["codynamic_rel_err"]
        verdict = "co-dynamic" if rel < 0.10 else "possible autonomous component"
        print(f"  {s['label']:14s} weight peak off gait by {rel*100:5.1f}%  ->  {verdict}")
    print("  The decisive test remains the constant-observation rollout.")

    print("\n--- ablation offsets " + "-" * 36)
    print(f"  base step {T_ABLATE}, {N_ABLATIONS} strata, spacing T_gait/{N_ABLATIONS}")
    for s in summaries:
        ab = s["ablation"]
        print(f"  {s['label']:14s} T={s['T_gait_steps_used']:6.1f}  "
              f"steps {ab['steps']}  "
              f"phases {[f'{p:.0f}' for p in ab['phases_deg']]}")

    print("\n--- window lengths " + "-" * 38)
    print("  network        M     M_hat   |H(f_gait)|")
    for s in summaries:
        T = s["T_gait_steps_used"]
        for M in CANDIDATE_M:
            m_hat = M / T
            gain = float(boxcar_response(np.array([s["f_gait_hz"]]), M)[0])
            print(f"  {s['label']:14s} {M:<5d} {m_hat:6.2f}   {gain:8.4f}")
        break   # full table goes to JSON; print one network as a sample
    print("  |H(f_gait)| near 0 means the window nulls the gait fundamental.")
    print("-" * 57 + "\n")

    # --- JSON outputs ------------------------------------------------------
    for s in summaries:
        T = s["T_gait_steps_used"]
        s["candidate_windows"] = [
            {"M": M,
             "window_s": M * DT,
             "M_hat": M / T,
             "gain_at_f_gait": float(boxcar_response(np.array([s["f_gait_hz"]]), M)[0])}
            for M in CANDIDATE_M
        ]

    full = {
        "tag": TAG,
        "dt": DT,
        "skip_steps": SKIP_STEPS,
        "t_ablate_base": T_ABLATE,
        "n_ablations": N_ABLATIONS,
        "acf_min": ACF_MIN,
        "acf_tol": ACF_TOL,
        "group_median_T_gait_steps": medians,
        "global_median_T_gait_steps": global_median,
        "n_substituted": n_sub,
        "networks": summaries,
    }
    with open(os.path.join(OUTDIR, f"{TAG}_summary.json"), "w") as fh:
        json.dump(full, fh, indent=2, default=str)

    # Compact lookup for the eval script: label -> ablation steps.
    offsets = {
        s["label"]: {
            "group": s["group"],
            "T_gait_steps": s["T_gait_steps_used"],
            "t_gait_source": s["t_gait_source"],
            "steps": s["ablation"]["steps"],
            "phases_deg": s["ablation"]["phases_deg"],
        }
        for s in summaries
    }
    with open(os.path.join(OUTDIR, f"{TAG}_ablation_offsets.json"), "w") as fh:
        json.dump({"t_ablate_base": T_ABLATE,
                   "n_ablations": N_ABLATIONS,
                   "dt": DT,
                   "networks": offsets}, fh, indent=2)

    # --- figures -----------------------------------------------------------
    if PER_NETWORK_FIGS:
        for s, a in results:
            fig_spectra_one(a, s)
            fig_analysed_window_one(a, s)

    fig_spectra_by_group(results)
    fig_tgait_across_networks(summaries)
    fig_filter_by_group(results, summaries)
    fig_ablation_phases(summaries)

    print(f"[DONE] Outputs written to {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()