"""
aggregate_leglock_sites.py -- Combine several eval_leg_lock_final.py sweeps
(one per damage SITE) into one cross-M comparison.

Reads only the {RUN_TAG}_summary.json written by each sweep. The npz archives
are not touched, so this runs in a plain conda env with no Isaac Lab, no GPU
and no unpickling -- iterate on the figures without re-running anything.

WHAT IT PRODUCES
----------------
Tables (the thing the sweep script never wrote):

  agg_seed_level.csv      one row per (site, group, network, cell). The unit of
                          analysis. Every downstream number here is derived
                          from this file, so it is what goes in the appendix.
  agg_interaction.csv     one row per (site, group, network): the freeze x lock
                          interaction, the quantity the 2x2 exists to measure.
  agg_group_level.csv     one row per (site, group, cell): mean / min / max
                          across networks.
  agg_summary.json        everything above plus the provenance of each sweep
                          and the results of the seed-level tests.

Figures (publication-ready, ICLR 5.5 in text block):

  agg_retention_sites     retention under lock_only, M-groups on x, one panel
                          per site. The headline comparison.
  agg_retention_pooled    all four cells of the 2x2, pooled over sites.
  agg_interaction         per-seed freeze x lock interaction by M-group.
  agg_severity            retention against damage site, one line per M-group:
                          does the M ordering hold as damage changes?
  agg_baseline_check      undamaged v_x and fallen fraction by M-group. Small
                          baseline v_x makes retention a ratio against a small
                          denominator; this is the figure that says whether the
                          headline ranking can be trusted.

THE MEASURE THAT MATTERS
------------------------
Retention alone conflates "was robust" with "adapted". The 2x2 separates them:

    interaction = (R_lock - R_base) - (R_both - R_freeze)

the first bracket being the cost of damage with plasticity live, the second
the same cost with the weights frozen. Both are negative (damage costs speed);
a network whose recovery is genuinely online pays MORE when frozen, so the
second bracket is the more negative and the interaction comes out POSITIVE.
A network that was merely statically robust pays the same either way and its
interaction is zero. Read it as "what plasticity bought under damage".
Computed per network, never pooled before differencing, so the pairing across
cells is preserved.

STATISTICS
----------
n = 5 networks per group. Every test here is non-parametric and paired where
the design allows it: an exact sign test on per-seed differences, plus the
median difference. No t-tests, no standard errors on n = 5 -- both would claim
a precision the design cannot support. Tests are reported for the pre-declared
contrasts only (each group against the best group, within site); anything else
you read off the figures is exploratory and should be labelled as such.
"""

# ===========================================================================
# CONFIG -- edit everything here
# ===========================================================================

# --- the four sweeps: point each at that run's output ---
# Either the sweep's OUTDIR (the leglock_MM:DD-HH:MM_TAG folder) or the
# {RUN_TAG}_summary.json inside it. A directory is resolved by taking the
# single *_summary.json it contains.
RUN_RL_HIP  = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/analysis/leglock_08:28-13:46_leglock_sweep_RL_hip_only/leglock_sweep_RL_hip_only_summary.json"     # e.g. ".../analysis/leglock_08:28-09:14_leglock_sweep_RL_hip_only"
RUN_FR_HIP  = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/analysis/leglock_08:28-13:48_leglock_sweep_FR_hip_only/leglock_sweep_RL_hip_only_summary.json"
RUN_RR_CALF = '/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/analysis/leglock_08:28-13:50_leglock_sweep_RR_calf_only/leglock_sweep_RL_hip_only_summary.json'
RUN_FL_CALF = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/analysis/leglock_08:28-13:49_leglock_sweep_FL_calf_only/leglock_sweep_RL_hip_only_summary.json"

# Display names, in the order they should appear on every axis. Ordered from
# what you expect to be mildest to most severe -- the severity figure reads as
# a dose-response curve only if this order is meaningful, so set it
# deliberately rather than alphabetically.
SITES = [
    ("RL hip",  RUN_RL_HIP),
    ("FR hip",  RUN_FR_HIP),
    ("RR calf", RUN_RR_CALF),
    ("FL calf", RUN_FL_CALF),
]

OUTDIR = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/analysis/leglock_aggregate"

# --- what to plot ---
PRIMARY_CELL = "lock_only"      # the cell the headline figure shows
CELL_ORDER = ["baseline", "freeze_only", "lock_only", "both"]
CELL_LABELS = {"baseline": "undamaged", "freeze_only": "freeze only",
               "lock_only": "lock only", "both": "freeze + lock"}

# Group ordering. Left as None, groups are sorted by the integer parsed out of
# the group name ("M=1" -> 1), which is what you want for an M sweep. Set an
# explicit list to override.
GROUP_ORDER = None

# --- sanity guards ---
# T_gait is recorded in the sweep JSON in CONTROL STEPS. A value below this is
# almost certainly seconds that were never converted (x 1/dt), which silently
# collapses the gait-phase stratification onto a single phase. Refuse to
# aggregate such a run rather than plot it.
MIN_T_GAIT_STEPS = 10.0
STRICT_T_GAIT = True            # False -> warn and continue

# Retention is a ratio; a near-zero denominator makes it explode. Values above
# this are flagged in the CSV and excluded from the medians.
RETENTION_CLIP = 5.0

# --- figure geometry (ICLR single column) ---
FIG_W_FULL = 5.5
FIG_W_HALF = 2.7

BG, FG, GRID = "#F6F5F1", "#22252A", "#E4E2DC"
CAT = ["#1A5E63", "#D9A441", "#B4553F", "#6E4A7E"]   # unordered categorical
RAMP_START = 0.22               # skip the lightest end of roma
# ===========================================================================

import json
import os
import re
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ===========================================================================
# Style
# ===========================================================================
def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 0.7, "lines.linewidth": 1.1,
        "lines.solid_capstyle": "round",
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": FG,
        "xtick.color": FG, "ytick.color": FG,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300,
        # Never savefig.bbox="tight": it crops the canvas below FIG_W_FULL and
        # \includegraphics then rescales, making every point size above wrong.
        "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def ramp_colours(n):
    """Ordered categorical colours for the M-groups, sampled from roma.

    Starts RAMP_START of the way along so the lightest group is still legible
    on the off-white canvas. Falls back to a fixed roma-like set if cmcrameri
    is not installed, so this script never fails for want of a colourmap.
    """
    try:
        from cmcrameri import cm as cmc
        xs = np.linspace(RAMP_START, 1.0, n)
        return [cmc.roma(x) for x in xs]
    except Exception:
        fallback = ["#7E4A20", "#B98B3A", "#7FA88C", "#2C6E8F", "#28405B"]
        idx = np.linspace(0, len(fallback) - 1, n).round().astype(int)
        return [fallback[i] for i in idx]


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[FIG] {name}.pdf / .png")


# ===========================================================================
# Loading
# ===========================================================================
def resolve_summary(path, site):
    """Accept either the sweep OUTDIR or the summary JSON itself."""
    if not path:
        raise ValueError(
            f"site '{site}': path is empty. Set the four RUN_* variables at "
            f"the top of this script to the four sweep output directories."
        )
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        cands = [f for f in os.listdir(path) if f.endswith("_summary.json")]
        if len(cands) != 1:
            raise ValueError(
                f"site '{site}': expected exactly one *_summary.json in "
                f"{path}, found {len(cands)}: {sorted(cands)}"
            )
        path = os.path.join(path, cands[0])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"site '{site}': no such file {path}")
    return path


def load_site(site, path):
    p = resolve_summary(path, site)
    with open(p) as f:
        meta = json.load(f)
    meta["_summary_path"] = os.path.abspath(p)
    meta["_site"] = site
    return meta


def check_t_gait(meta):
    """Refuse a sweep whose T_gait was given in seconds instead of steps.

    ablation_steps() spaces the strata by T_gait/N in CONTROL STEPS. A period
    passed in seconds (0.5-1.8) rounds every offset to 0 or 1, so all strata
    land on one event step and the per-network number is phase-CONDITIONAL,
    not the phase-marginal quantity the design claims. There is no symptom in
    any figure, so it has to be caught here.
    """
    bad = []
    for c in meta.get("conditions", []):
        t = c.get("T_gait_steps")
        if t is None or not np.isfinite(t) or t < MIN_T_GAIT_STEPS:
            bad.append((c.get("label"), t))
    if bad:
        msg = (f"[T_gait] {meta['_site']}: {len(bad)} network(s) have "
               f"T_gait_steps < {MIN_T_GAIT_STEPS}, which means the period was "
               f"recorded in SECONDS, not control steps. The gait-phase "
               f"stratification in that sweep collapsed onto a single phase. "
               f"Offenders: {bad[:5]}")
        if STRICT_T_GAIT:
            raise ValueError(msg + "\n  Set STRICT_T_GAIT=False to aggregate "
                                   "anyway (and say so in the caption).")
        print("[WARN] " + msg)


def group_sort_key(g):
    m = re.search(r"(\d+)", str(g))
    return (0, int(m.group(1))) if m else (1, str(g))


def short_group(g):
    """Tick label: "M=160" -> "160".

    Four panels of full "M=..." labels overlap at 5.5 in; the M is carried by
    the axis label instead, which is where it belongs.
    """
    m = re.fullmatch(r"\s*M\s*=?\s*(\d+)\s*", str(g))
    return m.group(1) if m else str(g)


def order_groups(groups):
    if GROUP_ORDER is not None:
        seen = [g for g in GROUP_ORDER if g in groups]
        return seen + sorted([g for g in groups if g not in seen],
                             key=group_sort_key)
    return sorted(groups, key=group_sort_key)


# ===========================================================================
# Reshaping into the seed-level table
# ===========================================================================
def build_seed_table(metas):
    """One record per (site, group, network, cell) -- the unit of analysis.

    Everything downstream is computed from this list, so any number in the
    thesis can be traced back to a single row of agg_seed_level.csv.
    """
    rows = []
    for meta in metas:
        site = meta["_site"]
        for r in meta["results"]:
            ret = r.get("velocity_retention", float("nan"))
            flagged = (not np.isfinite(ret)) or abs(ret) > RETENTION_CLIP
            rows.append({
                "site": site,
                "group": r["group"],
                "network": r["condition"],
                "cell": r["cell"],
                "M": r.get("M"),
                "T_gait_steps": r.get("T_gait_steps"),
                "retention": ret,
                "retention_flagged": int(flagged),
                "conv_ratio": r.get("convergence_ratio", float("nan")),
                "fraction_converged": r.get("fraction_converged", float("nan")),
                "v_pre": r.get("v_pre", float("nan")),
                "v_post": r.get("v_post", float("nan")),
                "dW_early": r.get("dW_early", float("nan")),
                "dW_late": r.get("dW_late", float("nan")),
                "mean_reward": r.get("mean_reward", float("nan")),
                "final_upright": r.get("final_upright", float("nan")),
                "fallen_fraction": r.get("fallen_fraction", float("nan")),
                "retention_phase_range": r.get("retention_phase_range",
                                               float("nan")),
            })
    return rows


def clean(vals):
    """Finite, unflagged values only."""
    a = np.asarray(vals, dtype=float)
    return a[np.isfinite(a) & (np.abs(a) <= RETENTION_CLIP)]


def pick(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def scalar(rows, site, group, network, cell, field="retention"):
    hit = pick(rows, site=site, group=group, network=network, cell=cell)
    if not hit:
        return float("nan")
    return float(hit[0][field])


def networks_of(rows, site, group):
    seen = []
    for r in rows:
        if r["site"] == site and r["group"] == group and r["network"] not in seen:
            seen.append(r["network"])
    return seen


def build_interaction_table(rows):
    """freeze x lock interaction, one value per (site, group, network).

    interaction = (R_lock - R_base) - (R_both - R_freeze)

    Positive -> damage costs more when the weights are frozen, i.e. the
    recovery had an online component. Zero -> the same cost either way, i.e.
    static robustness. Computed within a network so the four cells are paired;
    differencing group means instead would discard that pairing and inflate
    the spread by the between-network variance.
    """
    out = []
    sites = sorted({r["site"] for r in rows}, key=lambda s: [x[0] for x in SITES].index(s))
    for site in sites:
        for group in order_groups({r["group"] for r in rows if r["site"] == site}):
            for net in networks_of(rows, site, group):
                g = lambda c: scalar(rows, site, group, net, c)
                base, frz = g("baseline"), g("freeze_only")
                lock, both = g("lock_only"), g("both")
                cost_plastic = lock - base
                cost_frozen = both - frz
                out.append({
                    "site": site, "group": group, "network": net,
                    "M": scalar(rows, site, group, net, "baseline", "M"),
                    "R_baseline": base, "R_freeze_only": frz,
                    "R_lock_only": lock, "R_both": both,
                    "cost_plastic": cost_plastic,
                    "cost_frozen": cost_frozen,
                    "interaction": cost_plastic - cost_frozen,
                })
    return out


# ===========================================================================
# Statistics -- exact, non-parametric, seed-level
# ===========================================================================
def sign_test(diffs):
    """Exact two-sided sign test on paired differences.

    Chosen over Wilcoxon because at n = 5 the signed-rank statistic has too
    few attainable values to reach conventional significance in either
    direction, and over a t-test because five seeds cannot support a
    distributional assumption. Zeros are discarded, as the test requires.
    """
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return {"n": 0, "n_positive": 0, "p_two_sided": float("nan"),
                "median_diff": float("nan")}
    k = int((d > 0).sum())
    # exact binomial tail under p = 0.5
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return {"n": n, "n_positive": k, "p_two_sided": float(p),
            "median_diff": float(np.median(d))}


def paired_group_tests(rows, cell=PRIMARY_CELL, field="retention"):
    """Every pairwise M-group contrast, within site, paired across sites.

    Networks differ between groups, so the pairing is by SITE, not by seed:
    for each site take the group's median across its networks, then difference
    those medians site-by-site. With four sites that is four paired
    observations -- weak, and reported as such -- but it is the only pairing
    the design actually supports, and it is the right unit for asking whether
    an ordering holds across damage conditions.
    """
    sites = [s for s, _ in SITES]
    groups = order_groups({r["group"] for r in rows})
    per = {}
    for g in groups:
        per[g] = []
        for s in sites:
            v = clean([r[field] for r in pick(rows, site=s, group=g, cell=cell)])
            per[g].append(float(np.median(v)) if v.size else float("nan"))

    out = []
    for a, b in combinations(groups, 2):
        d = [x - y for x, y in zip(per[a], per[b])]
        res = sign_test(d)
        res.update({"group_a": a, "group_b": b, "cell": cell, "field": field,
                    "per_site_diff": d, "pairing": "site"})
        out.append(res)
    return out, per


def interaction_tests(inter):
    """Is the freeze x lock interaction non-zero, within each M-group?

    The null is that freezing costs nothing extra under damage; a positive
    median says plasticity contributed. Pairing is by
    (site, network), which is the strongest pairing available: both terms of
    the difference come from the same trained policy under the same damage.
    """
    out = []
    for g in order_groups({r["group"] for r in inter}):
        d = [r["interaction"] for r in inter if r["group"] == g]
        res = sign_test(d)
        res.update({"group": g, "n_observations": len(d)})
        out.append(res)
    return out


# ===========================================================================
# CSV writing -- plain text, no pandas dependency
# ===========================================================================
def write_csv(path, rows, cols=None):
    if not rows:
        print(f"[CSV] {path}: nothing to write")
        return
    cols = cols or list(rows[0].keys())
    def fmt(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else f"{v:.6g}"
        s = str(v)
        return f'"{s}"' if ("," in s or '"' in s) else s
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(fmt(r.get(c, "")) for c in cols) + "\n")
    print(f"[CSV] {os.path.basename(path)}  ({len(rows)} rows)")


# ===========================================================================
# Figures
# ===========================================================================
def _jitter(n, width=0.16):
    """Deterministic symmetric offsets, so overlaid seeds never coincide."""
    if n == 1:
        return np.array([0.0])
    return np.linspace(-width, width, n)


def fig_retention_sites(rows):
    """Headline: retention under the damaged-plastic cell, one panel per site.

    Bar = mean across networks. Whiskers = full min-max range across NETWORKS
    (between-policy spread), not across repeats. Open markers = the individual
    networks, so the reader can see the ranking is not carried by one seed.
    """
    sites = [s for s, _ in SITES]
    groups = order_groups({r["group"] for r in rows})
    cols = ramp_colours(len(groups))

    fig, axes = plt.subplots(1, len(sites), figsize=(FIG_W_FULL, 2.1),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, site in zip(axes, sites):
        for i, (g, c) in enumerate(zip(groups, cols)):
            vals = clean([r["retention"] for r in
                          pick(rows, site=site, group=g, cell=PRIMARY_CELL)])
            if not vals.size:
                continue
            ax.bar(i, vals.mean(), width=0.62, color=c, edgecolor="none",
                   zorder=2)
            ax.vlines(i, vals.min(), vals.max(), color=FG, linewidth=0.8,
                      zorder=3)
            ax.plot(i + _jitter(vals.size), vals, "o", ms=2.6,
                    mfc="none", mec=FG, mew=0.6, linestyle="none", zorder=4)
        ax.axhline(1.0, color=FG, linewidth=0.6, linestyle=(0, (3, 2)),
                   zorder=1)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([short_group(g) for g in groups])
        ax.set_title(site)
    axes[0].set_ylabel("velocity retention")
    for ax in axes:
        ax.set_xlabel(r"$M$")
    save_fig(fig, "agg_retention_sites")


def fig_retention_pooled(rows):
    """All four cells, pooled across sites. Cells are unordered, so CAT."""
    groups = order_groups({r["group"] for r in rows})
    cells = [c for c in CELL_ORDER
             if any(r["cell"] == c for r in rows)]
    w = 0.8 / len(cells)

    fig, ax = plt.subplots(figsize=(FIG_W_FULL, 2.3))
    for j, cell in enumerate(cells):
        xs, ys, los, his = [], [], [], []
        for i, g in enumerate(groups):
            vals = clean([r["retention"] for r in pick(rows, group=g, cell=cell)])
            if not vals.size:
                continue
            x = i - 0.4 + w * (j + 0.5)
            xs.append(x); ys.append(vals.mean())
            los.append(vals.min()); his.append(vals.max())
        ax.bar(xs, ys, width=w * 0.9, color=CAT[j % len(CAT)],
               edgecolor="none", label=CELL_LABELS.get(cell, cell), zorder=2)
        ax.vlines(xs, los, his, color=FG, linewidth=0.7, zorder=3)
    ax.axhline(1.0, color=FG, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([short_group(g) for g in groups])
    ax.set_ylabel("velocity retention")
    ax.set_xlabel(r"Hebbian window $M$")
    ax.legend(ncol=len(cells), loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save_fig(fig, "agg_retention_pooled")


def fig_interaction(inter):
    """The causal quantity: per-(site, network) freeze x lock interaction.

    Points above zero are networks that lost more to damage when frozen, i.e.
    an online contribution to recovery. The bar is the median, not the mean --
    the quantity is a difference of ratios and its tails are heavy.
    """
    groups = order_groups({r["group"] for r in inter})
    cols = ramp_colours(len(groups))
    site_names = [s for s, _ in SITES]
    marks = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(FIG_W_FULL, 2.3))
    for i, (g, c) in enumerate(zip(groups, cols)):
        vals = [r["interaction"] for r in inter if r["group"] == g]
        vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        if not vals.size:
            continue
        ax.bar(i, float(np.median(vals)), width=0.62, color=c,
               edgecolor="none", zorder=2)
        for r in [r for r in inter if r["group"] == g]:
            if not np.isfinite(r["interaction"]):
                continue
            si = site_names.index(r["site"]) if r["site"] in site_names else 0
            off = _jitter(len(site_names))[si]
            ax.plot(i + off, r["interaction"], marks[si % len(marks)], ms=2.8,
                    mfc="none", mec=FG, mew=0.6, zorder=4)
    ax.axhline(0.0, color=FG, linewidth=0.7, zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([short_group(g) for g in groups])
    ax.set_ylabel(r"freeze $\times$ lock interaction")
    ax.set_xlabel(r"Hebbian window $M$")
    handles = [Line2D([], [], marker=marks[i % len(marks)], linestyle="none",
                      mfc="none", mec=FG, mew=0.6, ms=3.0, label=s)
               for i, s in enumerate(site_names)]
    ax.legend(handles=handles, ncol=len(site_names), loc="upper center",
              bbox_to_anchor=(0.5, 1.22))
    save_fig(fig, "agg_interaction")


def fig_severity(rows):
    """Does the M ordering survive a change of damage site?

    One line per M-group across sites. Lines that cross mean the ranking is
    site-specific and must not be reported as a general result.
    """
    sites = [s for s, _ in SITES]
    groups = order_groups({r["group"] for r in rows})
    cols = ramp_colours(len(groups))

    fig, ax = plt.subplots(figsize=(FIG_W_HALF * 2, 2.2))
    for g, c in zip(groups, cols):
        ys, los, his = [], [], []
        for s in sites:
            vals = clean([r["retention"] for r in
                          pick(rows, site=s, group=g, cell=PRIMARY_CELL)])
            ys.append(vals.mean() if vals.size else np.nan)
            los.append(vals.min() if vals.size else np.nan)
            his.append(vals.max() if vals.size else np.nan)
        xs = np.arange(len(sites))
        ax.plot(xs, ys, "-o", color=c, ms=3.2, label=f"$M={short_group(g)}$",
                zorder=3)
        ax.fill_between(xs, los, his, color=c, alpha=0.14, linewidth=0,
                        zorder=2)
    ax.axhline(1.0, color=FG, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels(sites)
    ax.set_xlabel("damage site")
    ax.set_ylabel("velocity retention")
    ax.legend(ncol=len(groups), loc="upper center", bbox_to_anchor=(0.5, 1.20))
    save_fig(fig, "agg_severity")


def fig_baseline_check(rows):
    """Is the retention ratio trustworthy?

    Left: undamaged v_x per group. Retention divides by this, so a group with
    a small baseline speed gets a large and unstable ratio for a small
    absolute recovery -- the ranking cannot be read without it.
    Right: fallen fraction in the damaged cell, the outcome that does not
    depend on a denominator at all.
    """
    groups = order_groups({r["group"] for r in rows})
    cols = ramp_colours(len(groups))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W_FULL, 2.1))
    for i, (g, c) in enumerate(zip(groups, cols)):
        v = clean([r["v_pre"] for r in pick(rows, group=g, cell="baseline")])
        if v.size:
            ax1.bar(i, v.mean(), width=0.62, color=c, edgecolor="none", zorder=2)
            ax1.vlines(i, v.min(), v.max(), color=FG, linewidth=0.8, zorder=3)
            ax1.plot(i + _jitter(v.size), v, "o", ms=2.6, mfc="none", mec=FG,
                     mew=0.6, linestyle="none", zorder=4)
        f = np.asarray([r["fallen_fraction"] for r in
                        pick(rows, group=g, cell=PRIMARY_CELL)], dtype=float)
        f = f[np.isfinite(f)]
        if f.size:
            ax2.bar(i, f.mean(), width=0.62, color=c, edgecolor="none", zorder=2)
            ax2.vlines(i, f.min(), f.max(), color=FG, linewidth=0.8, zorder=3)
            ax2.plot(i + _jitter(f.size), f, "o", ms=2.6, mfc="none", mec=FG,
                     mew=0.6, linestyle="none", zorder=4)
    for ax, lab in ((ax1, r"undamaged $v_x$  [m s$^{-1}$]"),
                    (ax2, "fallen fraction, damaged")):
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([short_group(g) for g in groups])
        ax.set_xlabel(r"Hebbian window $M$")
        ax.set_ylabel(lab)
    save_fig(fig, "agg_baseline_check")


# ===========================================================================
# Console tables
# ===========================================================================
def print_tables(rows, inter, tests, per_site_median, inter_tests):
    groups = order_groups({r["group"] for r in rows})
    sites = [s for s, _ in SITES]

    print("\n" + "=" * 78)
    print(f"retention, cell = {PRIMARY_CELL}   (mean across networks [min, max])")
    print("-" * 78)
    hdr = f"{'group':>8}" + "".join(f"{s:>17}" for s in sites)
    print(hdr)
    for g in groups:
        line = f"{g:>8}"
        for s in sites:
            v = clean([r["retention"] for r in
                       pick(rows, site=s, group=g, cell=PRIMARY_CELL)])
            line += (f"{v.mean():>7.3f} [{v.min():.2f},{v.max():.2f}]"
                     if v.size else f"{'--':>17}")
        print(line)
    print("=" * 78)

    print("\n" + "=" * 78)
    print("freeze x lock interaction  (POSITIVE = online contribution)")
    print(f"{'group':>8} {'n':>4} {'median':>9} {'pos/n':>8} {'sign p':>9}")
    print("-" * 78)
    for t in inter_tests:
        print(f"{t['group']:>8} {t['n_observations']:>4} "
              f"{t['median_diff']:>9.3f} {t['n_positive']:>4}/{t['n']:<3} "
              f"{t['p_two_sided']:>9.3f}")
    print("=" * 78)

    print("\n" + "=" * 78)
    print(f"pairwise group contrasts, cell = {PRIMARY_CELL}, paired by site")
    print(f"{'A':>8} {'B':>8} {'med(A-B)':>10} {'A>B':>7} {'sign p':>9}")
    print("-" * 78)
    for t in tests:
        print(f"{t['group_a']:>8} {t['group_b']:>8} {t['median_diff']:>10.3f} "
              f"{t['n_positive']:>3}/{t['n']:<3} {t['p_two_sided']:>9.3f}")
    print("=" * 78)
    print("n = 4 paired sites; an exact sign test cannot fall below p = 0.125 "
          "at this n.\nRead the direction and the consistency, not the p value.\n")


# ===========================================================================
# Main
# ===========================================================================
def main():
    setup_style()
    os.makedirs(OUTDIR, exist_ok=True)

    metas = []
    for site, path in SITES:
        meta = load_site(site, path)
        check_t_gait(meta)
        metas.append(meta)
        print(f"[LOAD] {site:>8}  {len(meta['results']):>3} rows  "
              f"tag={meta.get('run_tag')}  legs={meta.get('leg_patterns')}")

    # The damage site is the only thing allowed to differ between sweeps. If
    # anything else does, the sites are not comparable and pooling them is a
    # confound rather than a replication.
    keys = ("lock_mode", "hold_position", "mask_locked_actions", "norm_mode",
            "steps", "event_step_base", "n_ablations", "param_source")
    ref = metas[0]
    for m in metas[1:]:
        for k in keys:
            if m.get(k) != ref.get(k):
                print(f"[WARN] {m['_site']}: {k} = {m.get(k)!r} but "
                      f"{ref['_site']} has {ref.get(k)!r}. Sites are not "
                      f"directly comparable on this axis.")
        if m.get("measures") != ref.get("measures"):
            print(f"[WARN] {m['_site']}: measurement windows differ from "
                  f"{ref['_site']}. Retention is not on the same scale.")

    rows = build_seed_table(metas)
    inter = build_interaction_table(rows)

    n_flagged = sum(r["retention_flagged"] for r in rows)
    if n_flagged:
        print(f"[WARN] {n_flagged} retention value(s) non-finite or "
              f"|r| > {RETENTION_CLIP}; flagged in the CSV and excluded from "
              f"every median. Check v_pre for those networks.")

    tests, per_site_median = paired_group_tests(rows)
    itests = interaction_tests(inter)

    # group-level table
    group_rows = []
    for site in [s for s, _ in SITES]:
        for g in order_groups({r["group"] for r in rows}):
            for cell in CELL_ORDER:
                v = clean([r["retention"] for r in
                           pick(rows, site=site, group=g, cell=cell)])
                if not v.size:
                    continue
                cv = clean([r["conv_ratio"] for r in
                            pick(rows, site=site, group=g, cell=cell)])
                group_rows.append({
                    "site": site, "group": g, "cell": cell,
                    "n_networks": int(v.size),
                    "retention_mean": float(v.mean()),
                    "retention_median": float(np.median(v)),
                    "retention_min": float(v.min()),
                    "retention_max": float(v.max()),
                    "conv_ratio_mean": float(cv.mean()) if cv.size else np.nan,
                })

    write_csv(os.path.join(OUTDIR, "agg_seed_level.csv"), rows)
    write_csv(os.path.join(OUTDIR, "agg_interaction.csv"), inter)
    write_csv(os.path.join(OUTDIR, "agg_group_level.csv"), group_rows)

    with open(os.path.join(OUTDIR, "agg_summary.json"), "w") as f:
        json.dump({
            "sites": [{"site": m["_site"], "summary_path": m["_summary_path"],
                       "run_tag": m.get("run_tag"),
                       "leg_patterns": m.get("leg_patterns"),
                       "lock_mode": m.get("lock_mode"),
                       "measures": m.get("measures"),
                       "n_ablations": m.get("n_ablations"),
                       "repeats_per_cell": m.get("repeats_per_cell")}
                      for m in metas],
            "primary_cell": PRIMARY_CELL,
            "retention_clip": RETENTION_CLIP,
            "group_level": group_rows,
            "interaction": inter,
            "interaction_tests": itests,
            "group_contrasts": tests,
            "per_site_group_median": per_site_median,
        }, f, indent=2, default=str)
    print(f"[OUT] agg_summary.json")

    fig_retention_sites(rows)
    fig_retention_pooled(rows)
    fig_interaction(inter)
    fig_severity(rows)
    fig_baseline_check(rows)

    print_tables(rows, inter, tests, per_site_median, itests)
    print(f"[DONE] {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()