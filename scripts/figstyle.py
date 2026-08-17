"""
figstyle.py — shared publication figure style for the thesis / conference papers.

Usage in any plotting script:

    import figstyle
    figstyle.apply()

    fig, ax = figstyle.figure(figstyle.FULL, aspect=0.62)
    ax.plot(...)
    figstyle.save(fig, "pca_variance")                 # -> ./pca_variance.pdf
    figstyle.save(fig, os.path.join(OUT_DIR, "pca"))   # -> your script's own dir

Figures are written wherever you already write them; this module does not
relocate anything. It only fixes the printed WIDTH, fonts, ticks and colours.

Switching venue is a ONE-LINE change: set VENUE below and re-run every script.

--------------------------------------------------------------------------
CONFIG
--------------------------------------------------------------------------
"""

# ==========================  CONFIG  ======================================

# Target venue. Controls column widths and default font size.
#   "iclr"  -> ICLR 2027 / UCL thesis. Single column, 5.5in text block.
#   "icra"  -> ICRA / IEEE two-column. 3.45in column, 7.16in page span.
VENUE = "iclr"

# Default output directory. None = do not impose a directory; save() uses the
# path exactly as each script already passes it (absolute, relative, or bare
# filename in the current working directory). Set to a string only if you want
# a single global override for every script.
OUTPUT_DIR = None

# Write a 300-DPI PNG alongside every PDF (useful for slides / quick viewing).
SAVE_PNG = True
PNG_DPI = 300

# Preferred serif family, in fallback order.
FONT_FAMILY = ["Times New Roman", "DejaVu Serif"]

# Okabe-Ito colourblind-safe palette (fixed order; index by condition).
OKABE_ITO = [
    "#000000",  # 0 black
    "#E69F00",  # 1 orange
    "#56B4E9",  # 2 sky blue
    "#009E73",  # 3 bluish green
    "#F0E442",  # 4 yellow
    "#0072B2",  # 5 blue
    "#D55E00",  # 6 vermillion
    "#CC79A7",  # 7 reddish purple
]

# ==========================  END CONFIG  ==================================


import os
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler


# --- Venue geometry -------------------------------------------------------
# FULL  : a figure spanning the full usable text width.
# HALF  : two figures side by side within the text width.
# THIRD : three figures side by side within the text width.
# WIDE  : ICRA only — a figure spanning both columns. Equals FULL for ICLR.

_VENUES = {
    # ICLR text block is 5.5in wide (33 picas), 10pt Times, 11pt leading.
    "iclr": {
        "FULL": 5.50,
        "HALF": 2.70,
        "THIRD": 1.75,
        "WIDE": 5.50,
        "BASE_PT": 9.0,
    },
    # IEEE two-column: 3.45in column, 0.17in gutter, 7.16in page span.
    "icra": {
        "FULL": 3.45,
        "HALF": 1.68,
        "THIRD": 1.08,
        "WIDE": 7.16,
        "BASE_PT": 8.0,
    },
}

if VENUE not in _VENUES:
    raise ValueError(f"VENUE must be one of {sorted(_VENUES)}, got {VENUE!r}")

_G = _VENUES[VENUE]

FULL = _G["FULL"]
HALF = _G["HALF"]
THIRD = _G["THIRD"]
WIDE = _G["WIDE"]
BASE_PT = _G["BASE_PT"]


def apply(base_pt=None):
    """Install the publication rcParams. Call once, before creating figures.

    Font sizes are absolute points at the *final printed size*, which is only
    correct if figures are inserted with width=\\linewidth and a figsize whose
    width equals the target column width. Use figure() below to guarantee this.
    """
    pt = BASE_PT if base_pt is None else base_pt

    matplotlib.rcParams.update({
        # --- Fonts. fonttype=42 embeds TrueType, required by IEEE PDF eXpress
        # --- and avoids the Type-3 fonts that NeurIPS/ICLR also reject.
        "font.family": "serif",
        "font.serif": FONT_FAMILY,
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # --- Sizes. Body text is 10pt; figure text sits at or just below it.
        "font.size": pt,
        "axes.labelsize": pt,
        "axes.titlesize": pt,
        "xtick.labelsize": pt - 1,
        "ytick.labelsize": pt - 1,
        "legend.fontsize": pt - 1,
        "figure.titlesize": pt,

        # --- Ticks: inward on all four axes.
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        # --- Axes and lines.
        "axes.linewidth": 0.6,
        "axes.prop_cycle": cycler(color=OKABE_ITO),
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "axes.grid": False,

        # --- Legend: no frame shadow, thin box.
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,

        # --- Layout. constrained_layout packs the axes to fit the figure
        # --- rather than growing it, so the saved width is EXACTLY figsize.
        # --- Do NOT set savefig.bbox="tight": it crops to the drawn content,
        # --- which changes the output width and makes \linewidth rescale the
        # --- figure, silently altering every font size.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,

        # --- Saving.
        "figure.dpi": 150,
        "savefig.dpi": PNG_DPI,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
        "savefig.transparent": False,
    })


def figure(width=None, aspect=0.62, nrows=1, ncols=1, **kwargs):
    """Create a figure at an exact printed width.

    Args:
        width:  printed width in inches. Use FULL / HALF / THIRD / WIDE.
                Defaults to FULL.
        aspect: height / width ratio for the whole figure. 0.62 is the golden
                ratio and is a sane default for a single panel; use ~0.75-0.9
                for a 2x2 grid, ~0.35-0.45 for a wide short strip.
        nrows, ncols, **kwargs: passed through to plt.subplots.

    Returns:
        (fig, ax) or (fig, axes) exactly as plt.subplots.
    """
    w = FULL if width is None else width
    return plt.subplots(nrows, ncols, figsize=(w, w * aspect), **kwargs)


def save(fig, name, output_dir=None, close=True):
    """Save a figure as PDF (and 300-DPI PNG if SAVE_PNG).

    The path is whatever you already pass. This does NOT relocate your figures:

        save(fig, "pca_variance")                  -> ./pca_variance.pdf
        save(fig, "results/plots/pca_variance")    -> results/plots/pca_variance.pdf
        save(fig, "/abs/path/pca_variance")        -> /abs/path/pca_variance.pdf
        save(fig, os.path.join(MY_OUT, "pca"))     -> uses your script's own dir

    A trailing ".pdf" or ".png" in `name` is stripped, so passing an existing
    full filename from your current scripts also works.

    Args:
        fig:  the matplotlib Figure.
        name: output path WITHOUT extension (extension tolerated and stripped).
        output_dir: optional directory to join onto `name`. Defaults to the
              module-level OUTPUT_DIR, which is None (= impose nothing).
        close: close the figure afterwards (prevents memory leaks in loops).

    Returns:
        list of paths written.
    """
    base = name
    for ext in (".pdf", ".png", ".svg", ".eps"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break

    d = OUTPUT_DIR if output_dir is None else output_dir
    stem = base if d is None else os.path.join(d, base)

    # Only create a directory if the resulting path actually names one.
    parent = os.path.dirname(stem)
    if parent:
        os.makedirs(parent, exist_ok=True)

    written = []
    pdf_path = f"{stem}.pdf"
    fig.savefig(pdf_path)
    written.append(pdf_path)

    if SAVE_PNG:
        png_path = f"{stem}.png"
        fig.savefig(png_path, dpi=PNG_DPI)
        written.append(png_path)

    w_in = fig.get_size_inches()[0]
    if close:
        plt.close(fig)

    for p in written:
        print(f"[figstyle] wrote {p}  ({w_in:.2f}in wide)")
    return written


def verify(pdf_path, expected_width=None, tol=0.02):
    """Check a saved PDF is exactly the intended printed width, with embedded
    non-Type-3 fonts. Requires pymupdf; skipped silently if unavailable.

    Run this once after regenerating figures — a width mismatch means
    \\linewidth will rescale the figure and your font sizes will be wrong.
    """
    try:
        import pymupdf
    except ImportError:
        print("[figstyle] verify skipped (pymupdf not installed)")
        return None

    want = FULL if expected_width is None else expected_width
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    got = page.rect.width / 72.0

    ok = abs(got - want) <= tol
    print(f"[figstyle] {pdf_path}: {got:.3f}in "
          f"(want {want:.3f}in) {'OK' if ok else 'MISMATCH'}")

    for f in page.get_fonts():
        name, ftype = f[3], f[2]
        bad = ftype == "Type3"
        print(f"[figstyle]   font {name} type={ftype}"
              f"{'  <-- Type 3, will be rejected' if bad else ''}")
    doc.close()
    return ok


def colour(i):
    """Okabe-Ito colour by index, wrapping around."""
    return OKABE_ITO[i % len(OKABE_ITO)]


def latex_width(width=None):
    """Return the \\includegraphics width directive matching a figure width.

    Always insert figures with this so the scale factor is exactly 1.0 and the
    rcParams font sizes are the true rendered sizes.
    """
    w = FULL if width is None else width
    frac = w / WIDE if VENUE == "icra" and w > FULL else w / FULL
    if abs(frac - 1.0) < 1e-6:
        return r"\linewidth" if not (VENUE == "icra" and w > FULL) else r"\textwidth"
    return rf"{frac:.3f}\linewidth"