"""
plot_leglock.py -- Regenerate every figure from a saved eval_leglock_go1 sweep.

Reads the {RUN_TAG}_data.npz and {RUN_TAG}_summary.json written by
eval_leglock_go1.py and emits publication-ready figures. No simulation, no
Isaac Lab, no GPU -- run it in a plain conda env and iterate on the plots
without re-running the sweep.

Every panel is written as its own file at single-column width, so any of them
can be dropped into the thesis independently:

    {TAG}_{condition}_{variable}.pdf      one condition, one variable
    {TAG}_grid_{variable}.pdf             all conditions stacked, one variable
    {TAG}_zoom_{condition}_dW.pdf         plasticity around the event step
    {TAG}_bar_{measure}.pdf               one measure across conditions x cells

Traces show the MEDIAN across repeats with an interquartile band. Set
SHOW_BAND = False for clean median-only lines when several cells overlap.
"""

# ===========================================================================
# CONFIG -- edit everything here
# ===========================================================================
ROOT = ("/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/"
        "workspace/hebbian_locomotion")

RUN_TAG = "leglock_sweep"                    # must match the sweep that wrote the data
INDIR = f"{ROOT}/analysis/leglock"           # where the npz and json live
OUTDIR = f"{ROOT}/analysis/leglock/figures"  # where figures are written
TAG = RUN_TAG                                # prefix for figure filenames

# --- what to draw; ordering here is the ordering in every figure ---
CONDITION_ORDER = None       # None -> order from the json; else e.g. ["M1", "M20", "M160"]
CELL_ORDER = None            # None -> order from the json; else e.g. ["baseline", "lock"]

VARIABLES = ["vx", "dW", "z", "rew", "up"]   # which traces to plot
MEASURES = ["retention", "conv_ratio", "converged"]

MAKE_SINGLES = True          # one file per condition per variable
MAKE_GRIDS = True            # one file per variable, conditions stacked
MAKE_ZOOM = True             # plasticity around the event step
MAKE_BARS = True             # summary bars per measure

SHOW_BAND = True             # interquartile band around the median
BAND_ALPHA = 0.20
ZOOM_BEFORE = 100            # steps before the event in the zoom figure
ZOOM_AFTER = 300             # steps after
LOG_DW = False               # log-scale the plasticity axis (magnitudes differ ~25x)

SHARE_Y_ACROSS_CONDITIONS = True   # comparable axes in the grid figures
SHOW_LEGEND = True         # set False for singles destined for a shared caption
LEGEND_FS_SINGLE = 8       # legend size on narrow single-column panels
# ===========================================================================

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# Style -- NeurIPS 2026 (5.5 in text block, 10 pt Times)
# ===========================================================================
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "verm": "#D55E00", "purple": "#CC79A7",
}
CELL_COLOUR = {
    "baseline": OKABE_ITO["black"], "freeze_only": OKABE_ITO["sky"],
    "lock_only": OKABE_ITO["orange"], "both": OKABE_ITO["verm"],
}
CELL_LABEL = {
    "baseline": "baseline", "freeze_only": "freeze",
    "lock_only": "lock", "both": "freeze + lock",
}
VAR_LABEL = {
    "vx": r"$v_x$ (m s$^{-1}$)",
    "dW": r"$\sum_k \Vert \Delta W_t^{(k)} \Vert_2$",
    "z": r"$z$ (m)",
    "rew": r"$r_t$",
    "up": r"upright (-)",
}
MEASURE_LABEL = {
    "retention": "velocity retention",
    "conv_ratio": r"$\Delta W_\mathrm{late} / \Delta W_\mathrm{early}$",
    "converged": "fraction converged",
}

FIG_W_FULL = 5.5
FIG_W_HALF = 2.65
FIG_W_THIRD = 1.72


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
        "lines.linewidth": 1.0,
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
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[FIG] {name}.pdf / .png")


# ===========================================================================
# Data access
# ===========================================================================
class Sweep:
    """Thin accessor over the saved npz, keyed by (condition, cell, variable)."""

    def __init__(self, npz_path, json_path):
        self.npz = np.load(npz_path)
        with open(json_path) as f:
            self.meta = json.load(f)

        self.event_step = self.meta["event_step"]
        self.steps = self.meta["steps"]
        self.rho = self.meta["measures"]["rho"]

        json_conds = [c["label"] for c in self.meta["conditions"]]
        json_cells = list(self.meta["cells"].keys())
        self.conditions = CONDITION_ORDER or json_conds
        self.cells = CELL_ORDER or json_cells

        # Only keep combinations that were actually written.
        present = {k.split("__")[0] + "__" + k.split("__")[1]
                   for k in self.npz.files if k.count("__") >= 2}
        self.available = present

    def has(self, cond, cell):
        return f"{cond}__{cell}" in self.available

    def trace(self, cond, cell, var):
        """(steps, num_repeats) array, or None if that combination is absent."""
        key = f"{cond}__{cell}__{var}"
        return self.npz[key] if key in self.npz.files else None

    def measure(self, cond, cell, name):
        """(num_repeats,) array of a per-repeat measure, or None."""
        key = f"{cond}__{cell}__per_repeat_{name}"
        return self.npz[key] if key in self.npz.files else None

    def row(self, cond, cell):
        for r in self.meta["results"]:
            if r["condition"] == cond and r["cell"] == cell:
                return r
        return None


# ===========================================================================
# Drawing primitives
# ===========================================================================
def draw_trace(ax, sw, cond, var, cells, xlim=None, label_cells=False):
    """Overlay the requested cells for one condition and one variable."""
    for cell in cells:
        arr = sw.trace(cond, cell, var)
        if arr is None:
            continue
        x = np.arange(arr.shape[0])
        med = np.median(arr, axis=1)
        colour = CELL_COLOUR.get(cell, OKABE_ITO["purple"])
        if SHOW_BAND and arr.shape[1] > 2:
            lo, hi = np.percentile(arr, [25, 75], axis=1)
            ax.fill_between(x, lo, hi, color=colour, alpha=BAND_ALPHA, lw=0)
        ax.plot(x, med, color=colour, lw=1.0,
                label=CELL_LABEL.get(cell, cell) if label_cells else None)

    ax.axvline(sw.event_step, color="0.55", ls="--", lw=0.8, zorder=0)
    ax.set_xlim(*(xlim if xlim else (0, sw.steps)))
    if var == "dW" and LOG_DW:
        ax.set_yscale("log")


def figure_legend(fig, ax, ncol=None, fontsize=None):
    """Legend above the figure -- overlaid traces leave no free space inside."""
    if not SHOW_LEGEND:
        return
    handles, names = ax.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(handles, names, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=ncol or len(handles),
               handlelength=1.4, columnspacing=1.0, labelspacing=0.3,
               handletextpad=0.5, fontsize=fontsize, borderaxespad=0.0)


# ===========================================================================
# Figures
# ===========================================================================
def fig_single(sw, cond, var):
    """One condition, one variable, its own file at single-column width."""
    fig, ax = plt.subplots(figsize=(FIG_W_HALF, FIG_W_HALF * 0.72))
    draw_trace(ax, sw, cond, var, sw.cells, label_cells=True)
    ax.set_xlabel("step")
    ax.set_ylabel(VAR_LABEL.get(var, var))
    figure_legend(fig, ax, ncol=2, fontsize=LEGEND_FS_SINGLE)
    save_fig(fig, f"{TAG}_{cond}_{var}")


def fig_grid(sw, var):
    """All conditions stacked, one variable, full text width."""
    conds = [c for c in sw.conditions
             if any(sw.trace(c, cell, var) is not None for cell in sw.cells)]
    if not conds:
        return
    fig, axes = plt.subplots(
        len(conds), 1, sharex=True,
        sharey=SHARE_Y_ACROSS_CONDITIONS,
        figsize=(FIG_W_FULL, 1.15 * len(conds) + 0.5))
    axes = np.atleast_1d(axes)

    for ax, cond in zip(axes, conds):
        draw_trace(ax, sw, cond, var, sw.cells, label_cells=(cond == conds[0]))
        ax.set_ylabel(f"{cond}\n{VAR_LABEL.get(var, var)}")
    axes[-1].set_xlabel("step")
    figure_legend(fig, axes[0])
    fig.subplots_adjust(hspace=0.14)
    save_fig(fig, f"{TAG}_grid_{var}")


def fig_zoom(sw, cond, var="dW"):
    """Close-up on the event step -- where a plasticity excursion would show."""
    lo = max(0, sw.event_step - ZOOM_BEFORE)
    hi = min(sw.steps, sw.event_step + ZOOM_AFTER)
    fig, ax = plt.subplots(figsize=(FIG_W_HALF, FIG_W_HALF * 0.72))
    draw_trace(ax, sw, cond, var, sw.cells, xlim=(lo, hi), label_cells=True)
    ax.set_xlabel("step")
    ax.set_ylabel(VAR_LABEL.get(var, var))
    figure_legend(fig, ax, ncol=2, fontsize=LEGEND_FS_SINGLE)
    save_fig(fig, f"{TAG}_zoom_{cond}_{var}")


def fig_bar(sw, name):
    """One measure, conditions on x, cells as grouped bars, its own file."""
    conds = sw.conditions
    cells = [c for c in sw.cells
             if any(sw.measure(cond, c, name) is not None for cond in conds)]
    if not cells:
        return

    x = np.arange(len(conds))
    w = 0.8 / len(cells)
    fig, ax = plt.subplots(figsize=(FIG_W_HALF, FIG_W_HALF * 0.72))

    for k, cell in enumerate(cells):
        off = (k - (len(cells) - 1) / 2) * w
        vals, err_lo, err_hi = [], [], []
        for cond in conds:
            v = sw.measure(cond, cell, name)
            if v is None or len(v) == 0:
                vals.append(np.nan); err_lo.append(0.0); err_hi.append(0.0)
                continue
            if name == "converged":
                # A proportion: no spread, plot the fraction itself.
                vals.append(float(np.nanmean(v)))
                err_lo.append(0.0); err_hi.append(0.0)
            else:
                m = float(np.nanmedian(v))
                q1, q3 = np.nanpercentile(v, [25, 75])
                vals.append(m)
                err_lo.append(max(0.0, m - q1)); err_hi.append(max(0.0, q3 - m))
        ax.bar(x + off, vals, w, yerr=np.array([err_lo, err_hi]), capsize=1.5,
               color=CELL_COLOUR.get(cell, OKABE_ITO["purple"]),
               label=CELL_LABEL.get(cell, cell), error_kw={"lw": 0.7})

    if name == "retention":
        ax.axhline(1.0, color="0.55", ls=":", lw=0.8)
    if name == "conv_ratio":
        ax.axhline(sw.rho, color="0.55", ls=":", lw=0.8)
    if name == "converged":
        ax.set_ylim(0, 1.05)

    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel(MEASURE_LABEL.get(name, name))
    figure_legend(fig, ax, ncol=2, fontsize=LEGEND_FS_SINGLE)
    save_fig(fig, f"{TAG}_bar_{name}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    set_pub_style()
    npz_path = os.path.join(INDIR, f"{RUN_TAG}_data.npz")
    json_path = os.path.join(INDIR, f"{RUN_TAG}_summary.json")
    for p in (npz_path, json_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p}")

    sw = Sweep(npz_path, json_path)
    print(f"[INFO] {len(sw.conditions)} conditions x {len(sw.cells)} cells | "
          f"steps={sw.steps} event={sw.event_step} rho={sw.rho}")

    if MAKE_SINGLES:
        for cond in sw.conditions:
            for var in VARIABLES:
                if any(sw.trace(cond, c, var) is not None for c in sw.cells):
                    fig_single(sw, cond, var)

    if MAKE_GRIDS:
        for var in VARIABLES:
            fig_grid(sw, var)

    if MAKE_ZOOM:
        for cond in sw.conditions:
            if any(sw.trace(cond, c, "dW") is not None for c in sw.cells):
                fig_zoom(sw, cond, "dW")
                fig_zoom(sw, cond, "vx")

    if MAKE_BARS:
        for name in MEASURES:
            fig_bar(sw, name)

    # ---- text table, so the numbers behind the figures are visible ----
    print("\n" + "=" * 70)
    print(f"{'condition':>10} {'cell':>12} {'retention':>10} {'conv ratio':>11} "
          f"{'converged':>10}")
    print("-" * 70)
    for cond in sw.conditions:
        for cell in sw.cells:
            r = sw.row(cond, cell)
            if r is None:
                continue
            print(f"{cond:>10} {cell:>12} {r['velocity_retention']:>10.3f} "
                  f"{r['convergence_ratio']:>11.3f} "
                  f"{r['fraction_converged']:>10.2f}")
    print("=" * 70)
    print(f"\n[DONE] figures in {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()