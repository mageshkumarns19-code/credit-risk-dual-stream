"""
generate_architecture_diagram_v2.py
--------------------------------------
Generates a proper neural architecture diagram (not a flowchart) for the
Dual-Stream Temporal-Static Hybrid Network: unrolled GRU cells across the
6 timesteps, attention shown as weighted connection lines, stacked dense
layers for the static encoder, and explicit tensor shape annotations.

Run from PowerShell:
    cd "D:\MITS\08-09-26\Dataset"
    pip install matplotlib numpy
    python generate_architecture_diagram_v2.py

Output: graphs_output/18_architecture_diagram.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import os

OUTPUT_DIR = "graphs_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(-1.6, 11)
ax.axis("off")

C_INPUT = "#EAF0FA"
C_GRU = "#7FA8D9"
C_GRU_EDGE = "#3B6EA5"
C_ATTN = "#B7CBEA"
C_STATIC = "#F4C089"
C_STATIC_EDGE = "#C97A2B"
C_FUSION = "#C9A8E0"
C_FUSION_EDGE = "#7D4E9C"
C_HEAD = "#9ED2AE"
C_HEAD_EDGE = "#3E8B57"
EDGE = "#2B2B2B"

# Realistic per-fold attention weights (recency-weighted, from Fig. 13) used
# to vary the visual weight of each attention connection line.
attn_weights = [0.116, 0.134, 0.155, 0.177, 0.199, 0.218]
attn_weights = np.array(attn_weights) / max(attn_weights)  # normalize for line width scaling


def rounded_box(x, y, w, h, text, facecolor, edgecolor, fontsize=8.5, lw=1.4, zorder=3):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.05",
                        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=zorder + 1)


def arrow(p1, p2, color=EDGE, lw=1.2, style="-|>", alpha=1.0, zorder=2):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                         linewidth=lw, color=color, alpha=alpha, zorder=zorder)
    ax.add_patch(a)


# ============================================================
# TEMPORAL STREAM: unrolled GRU across 6 timesteps
# ============================================================
n_t = 6
t_labels = ["t-5", "t-4", "t-3", "t-2", "t-1", "t"]
gru_xs = np.linspace(0.6, 9.2, n_t)
gru_y = 6.6
gru_w, gru_h = 1.15, 0.85

ax.text(4.9, 9.6, "Temporal Stream", ha="center", fontsize=12, fontweight="bold", color=C_GRU_EDGE)
ax.text(4.9, 9.15, r"$X^t \in \mathbb{R}^{6 \times 3}$   (6 months $\times$ [PAY, BILL, PAY\_AMT])",
        ha="center", fontsize=9)

# Input vectors x_1..x_6
for i, gx in enumerate(gru_xs):
    rounded_box(gx, 8.3, gru_w, 0.5, f"$x_{{{i+1}}}$\n({t_labels[i]})", C_INPUT, "#8FA6C7", fontsize=8)

# GRU cells, unrolled, with recurrence arrows between them
for i, gx in enumerate(gru_xs):
    rounded_box(gx, gru_y, gru_w, gru_h, f"GRU\n$h_{{{i+1}}}$", C_GRU, C_GRU_EDGE, fontsize=9)
    arrow((gx + gru_w / 2, 8.3), (gx + gru_w / 2, gru_y + gru_h), color="#555", lw=1.0)  # x_i -> GRU_i
    if i < n_t - 1:
        arrow((gx + gru_w, gru_y + gru_h / 2), (gru_xs[i + 1], gru_y + gru_h / 2),
              color=C_GRU_EDGE, lw=1.6)  # recurrence h_i -> h_(i+1)

ax.text(gru_xs[0] - 0.5, gru_y + gru_h / 2, r"$h_0=\vec{0}$", ha="right", va="center", fontsize=8)
arrow((gru_xs[0] - 0.45, gru_y + gru_h / 2), (gru_xs[0], gru_y + gru_h / 2), color=C_GRU_EDGE, lw=1.6)

# Attention: weighted connections from each h_i up to a context node
ctx_x, ctx_y = 4.9, 4.55
ctx_w, ctx_h = 1.6, 0.7
for i, gx in enumerate(gru_xs):
    x0, y0 = gx + gru_w / 2, gru_y
    x1, y1 = ctx_x + ctx_w / 2, ctx_y + ctx_h
    lw = 0.8 + 3.2 * attn_weights[i]
    alpha = 0.35 + 0.55 * attn_weights[i]
    arrow((x0, y0), (x1, y1), color="#C0392B", lw=lw, alpha=alpha, style="-", zorder=2)
    ax.text(x0, y0 - 0.18, f"$\\alpha_{{{i+1}}}$={attn_weights[i]*0.218:.2f}", ha="center",
            fontsize=6.3, color="#8B2E1F")

ax.text(0.6, 5.35, "Self-Attention\n" + r"$\alpha_\tau=\mathrm{softmax}(w_a^\top h_\tau)$" + "\n(line weight = attention magnitude)",
        ha="left", fontsize=7.8, style="italic", color="#8B2E1F")

rounded_box(ctx_x, ctx_y, ctx_w, ctx_h, r"Context $c \in \mathbb{R}^{32}$" + "\n" + r"$c=\sum_\tau \alpha_\tau h_\tau$",
            C_ATTN, C_GRU_EDGE, fontsize=8.5)

# ============================================================
# STATIC STREAM: stacked dense layers
# ============================================================
sx = 11.2
ax.text(12.4, 9.6, "Static Stream", ha="center", fontsize=12, fontweight="bold", color=C_STATIC_EDGE)
ax.text(12.4, 9.15, r"$x^s \in \mathbb{R}^{8}$  (demographics + engineered ratios)", ha="center", fontsize=9)

rounded_box(sx, 8.3, 2.4, 0.5, r"Input $x^s$", C_INPUT, "#C99A5A", fontsize=8.5)

# Dense layer 1 drawn as a stack of "neuron" rectangles to look like a real layer
n_neurons_1 = 6
neuron_w = 2.4 / n_neurons_1
for k in range(n_neurons_1):
    rounded_box(sx + k * neuron_w + 0.02, 7.15, neuron_w - 0.04, 0.55, "", C_STATIC, C_STATIC_EDGE, lw=0.8)
ax.text(12.4, 7.42, "Dense(32) + ReLU + Dropout(0.2)", ha="center", fontsize=7.8, zorder=5)
arrow((12.4, 8.3), (12.4, 7.7), color=C_STATIC_EDGE, lw=1.4)

n_neurons_2 = 6
for k in range(n_neurons_2):
    rounded_box(sx + k * neuron_w + 0.02, 6.05, neuron_w - 0.04, 0.55, "", C_STATIC, C_STATIC_EDGE, lw=0.8)
ax.text(12.4, 6.32, "Dense(32) + ReLU  (Algorithm 3)", ha="center", fontsize=7.8, zorder=5)
arrow((12.4, 7.15), (12.4, 6.6), color=C_STATIC_EDGE, lw=1.4)

rounded_box(sx, 4.85, 2.4, 0.65, r"Static rep. $s \in \mathbb{R}^{32}$", C_STATIC, C_STATIC_EDGE, fontsize=8.5)
arrow((12.4, 6.05), (12.4, 5.5), color=C_STATIC_EDGE, lw=1.4)

# ============================================================
# GATED FUSION
# ============================================================
fx, fy, fw, fh = 4.0, 3.15, 5.2, 0.6
arrow((ctx_x + ctx_w / 2, ctx_y), (fx + fw * 0.35, fy + fh), color=C_GRU_EDGE, lw=1.5)
arrow((sx + 1.2, 4.85), (fx + fw * 0.65, fy + fh), color=C_STATIC_EDGE, lw=1.5)
rounded_box(fx, fy, fw, fh, r"Concatenate: $u=[c\,;\,s]\in\mathbb{R}^{64}$", C_INPUT, "#777", fontsize=9)

gate_x, gate_y, gate_w, gate_h = 3.6, 1.9, 6.0, 0.85
arrow((fx + fw / 2, fy), (gate_x + gate_w / 2, gate_y + gate_h), color="#555", lw=1.4)

rounded_box(gate_x, gate_y, gate_w, gate_h,
            r"Gate: $g=\sigma(W_g u+b_g)$" + "\n" + r"Fused: $f=g\odot u \in \mathbb{R}^{64}$  (Algorithm 3)",
            C_FUSION, C_FUSION_EDGE, fontsize=9)

# ============================================================
# CLASSIFICATION HEAD
# ============================================================
hx, hy, hw, hh = 4.75, 0.75, 4.5, 0.75
arrow((gate_x + gate_w / 2, gate_y), (hx + hw / 2, hy + hh), color=C_FUSION_EDGE, lw=1.5)
rounded_box(hx, hy, hw, hh, "Dense(32, ReLU) \u2192 Dropout \u2192 Dense(1)\n(Algorithm 4)",
            C_HEAD, C_HEAD_EDGE, fontsize=9)

ox, oy, ow, oh = 5.9, -0.55, 2.2, 0.65
arrow((hx + hw / 2, hy), (ox + ow / 2, oy + oh), color=C_HEAD_EDGE, lw=1.5)
rounded_box(ox, oy, ow, oh, r"$\hat{y}=\sigma(\mathrm{logit})$", C_HEAD, C_HEAD_EDGE, fontsize=10)

# ============================================================
# Title + legend
# ============================================================
plt.title("Dual-Stream Temporal-Static Hybrid Network with Gated Fusion",
          fontsize=14, fontweight="bold", pad=18)

legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_GRU, markeredgecolor=C_GRU_EDGE, markersize=15, label='Temporal Stream (GRU)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_ATTN, markeredgecolor=C_GRU_EDGE, markersize=15, label='Attention / Context'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_STATIC, markeredgecolor=C_STATIC_EDGE, markersize=15, label='Static Stream'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_FUSION, markeredgecolor=C_FUSION_EDGE, markersize=15, label='Gated Fusion'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_HEAD, markeredgecolor=C_HEAD_EDGE, markersize=15, label='Classification Head'),
    Line2D([0], [0], color='#C0392B', lw=2.5, label='Attention weight (line thickness \u221d weight)'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
          ncol=3, frameon=False, fontsize=9)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "18_architecture_diagram.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")
