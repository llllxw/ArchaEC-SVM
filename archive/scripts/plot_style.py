"""Shared publication plotting style for all project figures."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


# Palette sampled/approximated from the user-provided reference figure, ordered
# from neutral grey through cool blue and warm cream to yellow, wine red, pink.
REFERENCE_PALETTE = [
    "#A6A6A6",
    "#5B6F82",
    "#B8CBE0",
    "#F1DDCA",
    "#FDBE2D",
    "#AB263A",
    "#F3B6B6",
]
CLASS_PALETTE = REFERENCE_PALETTE
BASELINE_COLOR = REFERENCE_PALETTE[0]
MODEL_COLOR = REFERENCE_PALETTE[1]
SECONDARY_COLOR = REFERENCE_PALETTE[5]
TARGET_COLOR = "#666666"

# Matching continuous maps for heatmaps.  These preserve the same visual
# identity while retaining enough luminance contrast for numeric annotations.
REFERENCE_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "reference_sequential",
    ["#F7F7F7", REFERENCE_PALETTE[2], REFERENCE_PALETTE[3], REFERENCE_PALETTE[4], REFERENCE_PALETTE[5]],
)
REFERENCE_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "reference_diverging",
    [REFERENCE_PALETTE[1], REFERENCE_PALETTE[2], "#FAFAFA", REFERENCE_PALETTE[3], REFERENCE_PALETTE[5]],
)


def apply_arial_style() -> None:
    """Set Arial for text and editable TrueType fonts for vector exports."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
