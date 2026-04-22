"""
Build A1 landscape poster for Case Study 1.9 — Tokenization of Numerical Expressions.
Author: Artem Panov, NLP 2026, Innopolis University.

Input: CSV files from /mnt/user-data/uploads
Output: poster.pdf (A1 landscape, single-page, vector)
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

UPLOADS = Path("/mnt/user-data/uploads")
OUTPUT  = Path("/home/claude/poster.pdf")

# A1 landscape: 841 × 594 mm = 33.11 × 23.39 inches
FIG_W, FIG_H = 33.11, 23.39

# Palette — must be consistent across all 4 result panels
COLOR_BG       = "#FFFFFF"
COLOR_HEADER   = "#0B3954"   # deep navy
COLOR_ACCENT   = "#BFD7EA"   # pale blue for callouts
COLOR_TLDR_BG  = "#F4F1DE"   # warm cream
COLOR_GOOD     = "#2A9D8F"   # teal (positive result)
COLOR_BAD      = "#E63946"   # red (failure / OOD)
COLOR_TEXT     = "#1D1D1D"
COLOR_MUTED    = "#5A5A5A"

# Consistent per-model / per-tokenizer colors
MODEL_COLORS = {
    "BERT":     "#1f77b4",
    "RoBERTa":  "#2ca02c",
    "GPT-2":    "#9467bd",
    "Pythia":   "#8c564b",
    "BLOOM":    "#17becf",
    "T5":       "#7f7f7f",
    "OPT":      "#bcbd22",
    "XLM-R":    "#e377c2",
    "Qwen2.5":  "#E76F51",   # highlight colour — orange-red
}

# Per-preprocessing colors (notebook 3)
PREP_COLORS = {
    "original":    "#457B9D",
    "mask_nums":   "#F4A261",
    "digit_split": "#2A9D8F",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": COLOR_TEXT,
    "axes.labelcolor": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # TrueType (editable in Illustrator if needed)
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

frag_mat  = pd.read_csv(UPLOADS / "fragmentation_matrix.csv")
mag_df    = pd.read_csv(UPLOADS / "magnitude_probe_summary.csv")
cmp_df    = pd.read_csv(UPLOADS / "comparison_probe.csv")
pb_mat    = pd.read_csv(UPLOADS / "phrasebank_f1_matrix.csv")
pb_over   = pd.read_csv(UPLOADS / "phrasebank_overall.csv")
nb4_sum   = pd.read_csv(UPLOADS / "nb4_summary.csv")

print("Loaded data:")
print("  fragmentation:", frag_mat.shape)
print("  magnitude:",     mag_df.shape)
print("  comparison:",    cmp_df.shape)
print("  phrasebank f1:", pb_mat.shape)
print("  nb4 summary:",   nb4_sum.shape)

# ---------------------------------------------------------------------------
# Figure + layout
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=COLOR_BG)

# Outer grid: 6 rows, 2 cols (we'll span for special rows)
# row heights ratio:
#   0 Header bar         : 1.2
#   1 TL;DR banner       : 0.7
#   2 Motivation + Setup : 2.3
#   3 Results top        : 5.5
#   4 Results bottom     : 5.5
#   5 Conclusions        : 1.8
outer = gridspec.GridSpec(
    nrows=6, ncols=2,
    figure=fig,
    height_ratios=[1.1, 0.65, 2.4, 5.5, 5.5, 2.3],
    width_ratios=[1, 1],
    left=0.025, right=0.975, top=0.985, bottom=0.018,
    hspace=0.70, wspace=0.12,
)

# ---------------------------------------------------------------------------
# Row 0 — Header
# ---------------------------------------------------------------------------
ax_head = fig.add_subplot(outer[0, :])
ax_head.set_xticks([]); ax_head.set_yticks([])
for s in ax_head.spines.values():
    s.set_visible(False)

# solid filled rectangle as header background
ax_head.add_patch(Rectangle((0, 0), 1, 1, transform=ax_head.transAxes,
                            facecolor=COLOR_HEADER, zorder=0))

ax_head.text(0.015, 0.65,
             "Tokenization of Numerical Expressions and Structured Data",
             transform=ax_head.transAxes,
             fontsize=42, fontweight="bold", color="white",
             va="center", ha="left")

ax_head.text(0.015, 0.22,
             "When does tokenizer design actually help language models reason about numbers?",
             transform=ax_head.transAxes,
             fontsize=22, style="italic", color="#DDE7EC",
             va="center", ha="left")

ax_head.text(0.985, 0.65,
             "Case Study 1.9",
             transform=ax_head.transAxes,
             fontsize=28, fontweight="bold", color="white",
             va="center", ha="right")
ax_head.text(0.985, 0.22,
             "Artem Panov · NLP 2026 · Innopolis University",
             transform=ax_head.transAxes,
             fontsize=18, color="#DDE7EC",
             va="center", ha="right")

# ---------------------------------------------------------------------------
# Row 1 — TL;DR banner
# ---------------------------------------------------------------------------
ax_tldr = fig.add_subplot(outer[1, :])
ax_tldr.set_xticks([]); ax_tldr.set_yticks([])
for s in ax_tldr.spines.values():
    s.set_visible(False)
ax_tldr.add_patch(Rectangle((0, 0), 1, 1, transform=ax_tldr.transAxes,
                            facecolor=COLOR_TLDR_BG, zorder=0))

tldr = ("TL;DR   Tokenizer design is a pre-training-time investment, not a free lunch. "
        "Digit-split preprocessing lifts out-of-distribution numerical reasoning by +14 pp, "
        "but no tokenizer — not even Qwen2.5's per-digit one — makes embeddings extrapolate to unseen magnitudes.")

ax_tldr.text(0.5, 0.5, tldr,
             transform=ax_tldr.transAxes,
             fontsize=20, color=COLOR_TEXT,
             va="center", ha="center", wrap=True)

# ---------------------------------------------------------------------------
# Row 2 — Motivation + Setup (two columns)
# ---------------------------------------------------------------------------

# Motivation
ax_mot = fig.add_subplot(outer[2, 0])
ax_mot.set_xticks([]); ax_mot.set_yticks([])
for s in ax_mot.spines.values():
    s.set_visible(False)
# section title bar
ax_mot.add_patch(Rectangle((0, 0.88), 1, 0.12, transform=ax_mot.transAxes,
                           facecolor=COLOR_HEADER, zorder=0))
ax_mot.text(0.015, 0.94, "Motivation & Research Questions",
            transform=ax_mot.transAxes,
            fontsize=19, fontweight="bold", color="white", va="center")

motivation = (
    "LLMs routinely fail at counting, comparing and extrapolating numbers.\n"
    "Suspect: the tokenizer — \"1000\" and \"1 0 0 0\" are different inputs.\n"
    "Modern models (Llama 3, Qwen 2.5) switched to per-digit tokenization.\n"
    "Does it actually help, and when?"
)
ax_mot.text(0.02, 0.82, motivation,
            transform=ax_mot.transAxes,
            fontsize=13, color=COLOR_TEXT, va="top", ha="left",
            linespacing=1.30)

rqs = (
    "RQ 1   How do tokenizer families fragment numbers?\n"
    "RQ 2   Is magnitude encoded in pretrained embeddings?\n"
    "RQ 3   Does number-preprocessing move the needle on\n"
    "              Financial PhraseBank sentiment?\n"
    "RQ 4   Does digit-split enable OOD generalization to\n"
    "              unseen magnitudes?"
)
ax_mot.text(0.02, 0.33, rqs,
            transform=ax_mot.transAxes,
            fontsize=13, color=COLOR_TEXT, va="top", ha="left",
            linespacing=1.40, fontweight="normal")

# Setup
ax_set = fig.add_subplot(outer[2, 1])
ax_set.set_xticks([]); ax_set.set_yticks([])
for s in ax_set.spines.values():
    s.set_visible(False)
ax_set.add_patch(Rectangle((0, 0.88), 1, 0.12, transform=ax_set.transAxes,
                           facecolor=COLOR_HEADER, zorder=0))
ax_set.text(0.015, 0.94, "Experimental Setup",
            transform=ax_set.transAxes,
            fontsize=19, fontweight="bold", color="white", va="center")

setup_left = (
    "9 tokenizers, 5 families:\n"
    "• WordPiece — BERT\n"
    "• Byte-BPE — GPT-2, RoBERTa, OPT, BLOOM\n"
    "• SentencePiece — T5, XLM-R\n"
    "• GPT-NeoX BPE — Pythia-160m\n"
    "• Qwen2-BPE (per-digit) — Qwen2.5-0.5B  ★"
)
ax_set.text(0.02, 0.80, setup_left,
            transform=ax_set.transAxes,
            fontsize=12.5, color=COLOR_TEXT, va="top", ha="left",
            linespacing=1.45)

setup_right = (
    "Tasks / probes:\n"
    "• Fragmentation audit — 9 categories × 9 tokenizers\n"
    "• Linear ridge probe  emb → log₁₀(n)  on [1,1000] & [10⁴,10⁵]\n"
    "• Pairwise comparison probe  a > b ?\n"
    "• Financial PhraseBank (sentences_50agree, 4 840 sents),\n"
    "    3 preprocessings × 3 encoders\n"
    "• Synthetic a vs b — in-dist [0,1000) + OOD [10⁴,10⁵)"
)
ax_set.text(0.52, 0.80, setup_right,
            transform=ax_set.transAxes,
            fontsize=12.5, color=COLOR_TEXT, va="top", ha="left",
            linespacing=1.45)

# ---------------------------------------------------------------------------
# Helper: panel frame with title
# ---------------------------------------------------------------------------
def panel_title(ax, rq, title, color=COLOR_HEADER):
    """Add a coloured title bar inside the panel axis."""
    # We draw the title bar OUTSIDE the subplot using annotate on figure coords
    # But simpler: use ax.set_title with a suptitle-like bold format
    ax.set_title(f"{rq}   {title}",
                 fontsize=18, fontweight="bold", color=color,
                 loc="left", pad=14)

# ---------------------------------------------------------------------------
# Row 3 — Results top: NB1 | NB2
# ---------------------------------------------------------------------------

# --- NB1: fragmentation heatmap ---------------------------------------------
ax1 = fig.add_subplot(outer[3, 0])

# order rows: put Qwen last so it visually stands out at the bottom
row_order = ["BERT", "GPT-2", "RoBERTa", "BLOOM", "T5", "XLM-R", "Pythia", "OPT", "Qwen2.5"]
cols = ["Plain integers", "Formatted integers (commas)", "Decimals",
        "Scientific notation", "Dates", "Times", "Currency",
        "Percentages", "Fractions"]
frag_mat_idx = frag_mat.set_index("tokenizer").loc[row_order, cols]

data = frag_mat_idx.values
im = ax1.imshow(data, cmap="YlOrRd", aspect="auto", vmin=1, vmax=10)

ax1.set_xticks(np.arange(len(cols)))
ax1.set_xticklabels([c.replace(" ", "\n", 1) for c in cols],
                    fontsize=11, rotation=0, ha="center")
ax1.set_yticks(np.arange(len(row_order)))
ax1.set_yticklabels(row_order, fontsize=12)
ax1.tick_params(length=0)

# annotate
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        txt_color = "white" if v > 5.5 else COLOR_TEXT
        ax1.text(j, i, f"{v:.1f}", ha="center", va="center",
                 fontsize=10, color=txt_color, fontweight="bold")

# highlight Qwen row
qwen_idx = row_order.index("Qwen2.5")
ax1.add_patch(Rectangle(
    (-0.5, qwen_idx - 0.5), len(cols), 1,
    fill=False, edgecolor=MODEL_COLORS["Qwen2.5"], linewidth=3,
    zorder=10
))

cbar = plt.colorbar(im, ax=ax1, shrink=0.6, pad=0.02)
cbar.set_label("mean sub-tokens / expression", fontsize=10)
cbar.ax.tick_params(labelsize=9)

panel_title(ax1, "RQ 1",
            "Tokenizers fragment numbers very differently — Qwen2.5 is the only true digit-level")

ax1.text(0.0, -0.15,
         "Qwen2.5 uses ~one token per digit by design (4.9 for plain ints ≈ mean digit count).\n"
         "Pythia's GPT-NeoX tokenizer collapses small integers into 1.4 tokens — family ≠ digit-level.",
         transform=ax1.transAxes,
         fontsize=11, color=COLOR_MUTED, va="top", ha="left", style="italic",
         linespacing=1.4)

# --- NB2: magnitude probe ---------------------------------------------------
ax2 = fig.add_subplot(outer[3, 1])

order2 = ["BERT", "RoBERTa", "GPT-2", "Pythia", "T5", "Qwen2.5"]
mag = mag_df.set_index("tokenizer").loc[order2]
r2_in  = mag["R2_in"].values
r2_ood = mag["R2_ood"].values

x = np.arange(len(order2))
w = 0.38

bars_in  = ax2.bar(x - w/2, r2_in, w, color=COLOR_GOOD, label="in-distribution R² (≤ 1.0, higher = better)")
bars_ood = ax2.bar(x + w/2, r2_ood, w, color=COLOR_BAD,  label="OOD R² (→ −∞ = fails)")

# highlight Qwen in-dist bar border
qwen_i2 = order2.index("Qwen2.5")
bars_in[qwen_i2].set_edgecolor(MODEL_COLORS["Qwen2.5"])
bars_in[qwen_i2].set_linewidth(3)
bars_ood[qwen_i2].set_edgecolor(MODEL_COLORS["Qwen2.5"])
bars_ood[qwen_i2].set_linewidth(3)

ax2.axhline(0, color="black", linewidth=0.8)
ax2.axhline(1, color=COLOR_GOOD, linewidth=0.8, linestyle=":")
ax2.set_xticks(x)
ax2.set_xticklabels(order2, fontsize=12)
ax2.set_ylabel("R²  (linear probe: embedding → log₁₀ n)", fontsize=12)
ax2.set_ylim(-50, 4)
ax2.legend(loc="lower left", fontsize=11, frameon=False)

# annotate numbers
for b, v in zip(bars_in, r2_in):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.5,
             f"{v:.2f}", ha="center", va="bottom", fontsize=10,
             color=COLOR_GOOD, fontweight="bold")
for b, v in zip(bars_ood, r2_ood):
    ax2.text(b.get_x() + b.get_width()/2, v - 1.5,
             f"{v:.1f}", ha="center", va="top", fontsize=10,
             color=COLOR_BAD, fontweight="bold")

panel_title(ax2, "RQ 2",
            "Magnitude is encoded in-distribution — no model extrapolates")

ax2.text(0.0, -0.14,
         "All models hit R²_in > 0.67 — magnitude is linearly decodable. Every probe collapses on OOD\n"
         "(R² ∈ [−45, −15]). Qwen's low R²_in is a mean-pooling artifact; comparison probe still 0.91.",
         transform=ax2.transAxes,
         fontsize=11, color=COLOR_MUTED, va="top", ha="left", style="italic",
         linespacing=1.4)

# inset: comparison accuracy — placed in middle-left region where bars are negative (empty space)
axi = ax2.inset_axes([0.06, 0.38, 0.36, 0.30])
cmp_order = cmp_df.set_index("tokenizer").loc[order2, "comparison_acc"].values
colors_cmp = [MODEL_COLORS[m] for m in order2]
axi.bar(np.arange(len(order2)), cmp_order, color=colors_cmp, edgecolor="white", linewidth=0.8)
axi.set_ylim(0.85, 0.99)
axi.set_xticks(np.arange(len(order2)))
axi.set_xticklabels(order2, fontsize=8, rotation=35, ha="right")
axi.set_title("Comparison probe:  a > b ?", fontsize=10, fontweight="bold", pad=4)
axi.tick_params(labelsize=8)
for i, v in enumerate(cmp_order):
    axi.text(i, v + 0.003, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
axi.spines["top"].set_visible(False); axi.spines["right"].set_visible(False)
axi.set_facecolor("#FBFBFB")

# ---------------------------------------------------------------------------
# Row 4 — Results bottom: NB3 | NB4
# ---------------------------------------------------------------------------

# --- NB3: PhraseBank grouped bars ------------------------------------------
ax3 = fig.add_subplot(outer[4, 0])
models3 = ["DistilBERT", "RoBERTa", "XLM-R"]
preps3  = ["original", "mask_nums", "digit_split"]
pb = pb_mat.set_index("model").loc[models3, preps3]

x = np.arange(len(models3))
w = 0.26

for k, p in enumerate(preps3):
    offset = (k - 1) * w
    vals = pb[p].values
    bars = ax3.bar(x + offset, vals, w,
                   color=PREP_COLORS[p],
                   label=p.replace("_", "-"))
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width()/2, v + 0.006,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9.5,
                 color=COLOR_TEXT)

ax3.set_xticks(x)
ax3.set_xticklabels(models3, fontsize=13)
ax3.set_ylabel("macro-F1 (test)", fontsize=12)
ax3.set_ylim(0.40, 0.92)
ax3.legend(loc="upper left", fontsize=11, frameon=True, ncol=3,
           facecolor="white", edgecolor="#CCCCCC",
           bbox_to_anchor=(0.0, 1.0))
ax3.grid(axis="y", linestyle=":", alpha=0.35)

# Annotate the big XLM-R jump — position annotation in a clean area ABOVE the XLM-R cluster
xlmr_i = models3.index("XLM-R")
orig = pb.loc["XLM-R", "original"]
masked = pb.loc["XLM-R", "mask_nums"]
ax3.annotate(
    f"Δ = +{masked-orig:.3f}\nundertraining,\nnot tokenization",
    xy=(xlmr_i + 0.0, masked + 0.012),
    xytext=(xlmr_i - 0.40, 0.86),
    fontsize=10.5, color=COLOR_BAD, ha="left",
    arrowprops=dict(arrowstyle="->", color=COLOR_BAD, lw=1.3,
                    connectionstyle="arc3,rad=0.25"),
    fontweight="bold",
)

panel_title(ax3, "RQ 3",
            "On sentiment, number-preprocessing barely moves the needle")

ax3.text(0.0, -0.17,
         "For English encoders |ΔF1| ≤ 0.013 — within seed noise; context carries sentiment, not digits.\n"
         "XLM-R's mask_nums gain reflects undertraining (F1 0.48 at acc 0.71 → majority-class collapse).",
         transform=ax3.transAxes,
         fontsize=11, color=COLOR_MUTED, va="top", ha="left", style="italic",
         linespacing=1.4)

# --- NB4: synthetic OOD -----------------------------------------------------
ax4 = fig.add_subplot(outer[4, 1])

conds = nb4_sum["condition"].tolist()
# relabel for compactness
labels4 = ["DistilBERT\nplain", "DistilBERT\n+ digit-split", "Pythia-160m\n(native digit)"]
in_dist = nb4_sum["in_dist_acc"].values
ood     = nb4_sum["ood_acc"].values
seqlen  = nb4_sum["mean_seq_len"].values

x = np.arange(len(conds))
w = 0.38

bars_i = ax4.bar(x - w/2, in_dist, w, color=COLOR_GOOD, label="in-distribution  [0, 1 000)")
bars_o = ax4.bar(x + w/2, ood,     w, color=COLOR_BAD,  label="OOD  [10⁴, 10⁵)")

# highlight digit_split OOD win
bars_o[1].set_edgecolor("#000000")
bars_o[1].set_linewidth(2.2)

for b, v in zip(bars_i, in_dist):
    ax4.text(b.get_x() + b.get_width()/2, v + 0.012,
             f"{v:.3f}", ha="center", va="bottom", fontsize=10,
             color=COLOR_GOOD, fontweight="bold")
for b, v in zip(bars_o, ood):
    ax4.text(b.get_x() + b.get_width()/2, v + 0.012,
             f"{v:.3f}", ha="center", va="bottom", fontsize=10,
             color=COLOR_BAD, fontweight="bold")

ax4.set_xticks(x)
# include seq length in tick labels so there is no axis collision
labels4_with_seq = [
    f"{lbl}\n(seq len {sl:.1f})"
    for lbl, sl in zip(labels4, seqlen)
]
ax4.set_xticklabels(labels4_with_seq, fontsize=11.5)
ax4.set_ylabel("accuracy (a > b classification)", fontsize=12)
ax4.set_ylim(0.55, 1.10)
ax4.legend(loc="upper left", fontsize=11, frameon=True,
           facecolor="white", edgecolor="#CCCCCC")
ax4.grid(axis="y", linestyle=":", alpha=0.35)

# big win annotation — positioned above DistilBERT + digit-split
delta = ood[1] - ood[0]
ax4.annotate(
    f"+{delta*100:.1f} pp OOD",
    xy=(1 + w/2, ood[1]),
    xytext=(1.22, 1.03),
    fontsize=18, color=COLOR_GOOD, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=COLOR_GOOD, lw=1.8),
)

panel_title(ax4, "RQ 4",
            "On a task that actually needs number reasoning, digit-split wins OOD")

ax4.text(0.0, -0.17,
         "Digit-split preprocessing — without retraining the tokenizer — lifts DistilBERT OOD\n"
         "from 75.8% → 89.9% (+14 pp), at ~72% longer sequences. Model size matters too: Pythia lags.",
         transform=ax4.transAxes,
         fontsize=11, color=COLOR_MUTED, va="top", ha="left", style="italic",
         linespacing=1.4)

# ---------------------------------------------------------------------------
# Row 5 — Conclusions + references
# ---------------------------------------------------------------------------
ax_concl = fig.add_subplot(outer[5, :])
ax_concl.set_xticks([]); ax_concl.set_yticks([])
for s in ax_concl.spines.values():
    s.set_visible(False)
ax_concl.add_patch(Rectangle((0, 0.80), 1, 0.20, transform=ax_concl.transAxes,
                             facecolor=COLOR_HEADER, zorder=0))
ax_concl.text(0.01, 0.90, "Conclusions",
              transform=ax_concl.transAxes,
              fontsize=20, fontweight="bold", color="white", va="center")

# Three columns of conclusions
c1 = ("① Embeddings encode magnitude in-distribution across every\n"
      "tokenizer family — but none extrapolates (R²_OOD ∈ [−45, −15]).\n"
      "Regular input ≠ regular representation. Even Qwen2.5's confirmed\n"
      "per-digit tokenizer does not fix this.")

c2 = ("② For sentiment-type tasks, number-preprocessing barely matters\n"
      "(|ΔF1| ≤ 0.013 for English encoders). Context carries the signal,\n"
      "not the digits themselves. XLM-R's apparent gain with masked\n"
      "numbers is undertraining, not tokenization.")

c3 = ("③ For tasks that require actual number comparison, digit-split\n"
      "preprocessing is a cheap +14 pp OOD boost — no retraining\n"
      "needed. Digit-level tokenization is a pre-training investment\n"
      "for a specific class of tasks, not a universal improvement.")

ax_concl.text(0.01, 0.72, c1, transform=ax_concl.transAxes,
              fontsize=13, color=COLOR_TEXT, va="top", linespacing=1.45)
ax_concl.text(0.35, 0.72, c2, transform=ax_concl.transAxes,
              fontsize=13, color=COLOR_TEXT, va="top", linespacing=1.45)
ax_concl.text(0.68, 0.72, c3, transform=ax_concl.transAxes,
              fontsize=13, color=COLOR_TEXT, va="top", linespacing=1.45)

refs = ("References:  Wallace et al. 2019 (EMNLP) · Thawani et al. 2021 (NAACL) · "
        "Malo et al. 2014 (JASIST, PhraseBank) · Yang et al. 2024 (Qwen 2.5 Tech Report) · "
        "Dubey et al. 2024 (Llama 3).    "
        "Code & notebooks: github.com/ArtemPanov5/tokenization-case-study")
ax_concl.text(0.01, 0.04, refs,
              transform=ax_concl.transAxes,
              fontsize=10, color=COLOR_MUTED, va="bottom", style="italic")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(OUTPUT, format="pdf", facecolor=COLOR_BG, bbox_inches=None)
print(f"Saved poster → {OUTPUT}  ({OUTPUT.stat().st_size/1024:.1f} KB)")
