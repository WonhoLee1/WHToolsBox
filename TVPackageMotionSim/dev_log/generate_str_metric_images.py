import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

out_dir = r"c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\dev_log"
os.makedirs(out_dir, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
bg_color = '#ffffff'
text_color = '#2c3e50'
accent_color = '#e74c3c'
blue_accent = '#3498db'

def setup_fig(title):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.axis('off')
    ax.text(0.5, 0.95, title, color=text_color, fontsize=20, fontweight='bold', ha='center', va='center')
    return fig, ax

def draw_tv_box(ax, x, y, angle=0):
    # TV Set (Inner)
    tv = patches.Rectangle((x+0.1, y+0.1), 0.4, 0.25, angle=angle, fill=False, color='#7f8c8d', lw=2, ls='--')
    ax.add_patch(tv)
    # Box (Outer)
    box = patches.Rectangle((x, y), 0.6, 0.45, angle=angle, fill=False, color='#34495e', lw=3)
    ax.add_patch(box)
    # Cushions
    c1 = patches.Rectangle((x+0.02, y+0.02), 0.08, 0.1, angle=angle, fill=True, color=blue_accent, alpha=0.5)
    c2 = patches.Rectangle((x+0.5, y+0.02), 0.08, 0.1, angle=angle, fill=True, color=blue_accent, alpha=0.5)
    c3 = patches.Rectangle((x+0.02, y+0.33), 0.08, 0.1, angle=angle, fill=True, color=blue_accent, alpha=0.5)
    c4 = patches.Rectangle((x+0.5, y+0.33), 0.08, 0.1, angle=angle, fill=True, color=blue_accent, alpha=0.5)
    for c in [c1, c2, c3, c4]: ax.add_patch(c)
    ax.text(x+0.3, y+0.22, "TV Set", color='#7f8c8d', fontsize=12, ha='center', va='center', rotation=angle)
    ax.text(x+0.3, y-0.05, "Packaging Box & Cushions", color='#34495e', fontsize=12, ha='center', va='center', rotation=angle)

# ==========================================================
# 1. Bending Stress (BS)
# ==========================================================
fig, ax = setup_fig("1. Bending Stress (BS) & Bending Moment (BM)")
draw_tv_box(ax, 0.1, 0.5, angle=10)
# Arrow indicating drop impact
ax.annotate("", xy=(0.4, 0.45), xytext=(0.4, 0.25), arrowprops=dict(facecolor=accent_color, shrink=0.05, width=3))
ax.text(0.45, 0.35, "Drop Impact", color=accent_color, fontsize=12)

# Zoom in on cushion bending
cushion = patches.Rectangle((0.6, 0.5), 0.2, 0.3, angle=15, fill=True, color=blue_accent, alpha=0.3, lw=2, ec=blue_accent)
ax.add_patch(cushion)
ax.annotate("", xy=(0.8, 0.8), xytext=(0.9, 0.7), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=accent_color, lw=2))
ax.text(0.85, 0.75, r"$\theta_{bend}$", color=accent_color, fontsize=14)

ax.text(0.1, 0.15, r"$\mathbf{Eq 1.}$ Equivalent Rotational Stiffness: $K_{rot} = \frac{E_{eff} \cdot I}{L}$", color=text_color, fontsize=14)
ax.text(0.1, 0.08, r"$\mathbf{Eq 2.}$ Bending Moment: $M = K_{rot} \cdot \theta_{bend}$", color=text_color, fontsize=14)
ax.text(0.1, 0.01, r"$\mathbf{Eq 3.}$ Bending Stress: $\sigma_{bend} = \frac{M \cdot c}{I} \quad \text{(where } c = t/2 \text{)}$", color=accent_color, fontsize=16, fontweight='bold')
plt.savefig(os.path.join(out_dir, "str_metrics_Bending_Stress.png"), dpi=150, bbox_inches='tight')
plt.close()

# ==========================================================
# 2. RRG
# ==========================================================
fig, ax = setup_fig("2. Relative Rotation Gradient (RRG)")
draw_tv_box(ax, 0.2, 0.4)
ax.text(0.5, 0.3, "Detects Local Wrinkling / Folding in Cushions or Box Panels", color='#7f8c8d', fontsize=12, ha='center')

# Two adjacent blocks in cushion
rect1 = patches.Rectangle((0.6, 0.6), 0.15, 0.1, angle=0, fill=True, color=blue_accent, alpha=0.5, ec='k', lw=1)
rect2 = patches.Rectangle((0.75, 0.6), 0.15, 0.1, angle=-25, fill=True, color=accent_color, alpha=0.5, ec='k', lw=1)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.text(0.675, 0.65, "Geom i\n($R_i$)", color='w', ha='center', va='center')
ax.text(0.825, 0.6, "Geom j\n($R_j$)", color='w', ha='center', va='center', rotation=-25)

ax.text(0.1, 0.15, r"$\mathbf{Eq 1.}$ Deformation matrices comparing adjacent blocks $i, j$", color=text_color, fontsize=14)
ax.text(0.1, 0.05, r"$\mathbf{Eq 2.} \ RRG_i = \max_{j \in N(i)} \left( \cos^{-1}\left( \frac{\mathrm{Tr}(R_i^T R_j) - 1}{2} \right) \right)$", color=accent_color, fontsize=16, fontweight='bold')
plt.savefig(os.path.join(out_dir, "str_metrics_RRG.png"), dpi=150, bbox_inches='tight')
plt.close()

# ==========================================================
# 3. PBA
# ==========================================================
fig, ax = setup_fig("3. Principal Bending Axis (PBA) & GTI")
# Draw a bent TV box
box = patches.FancyBboxPatch((0.2, 0.4), 0.6, 0.3, boxstyle="round,pad=0.02,rounding_size=0.05", mutation_aspect=0.5, fill=False, color='#34495e', lw=3)
ax.add_patch(box)
ax.plot([0.2, 0.8], [0.55, 0.55], 'r--', lw=3) # Major bending axis
ax.text(0.85, 0.55, r"Major Axis ($\mathbf{v}_{max}$)", color=accent_color, fontsize=14, va='center')

ax.text(0.5, 0.3, "Extracts global bending direction (PBA) from PCA of rotation vectors", color='#7f8c8d', fontsize=12, ha='center')
ax.text(0.1, 0.15, r"$\mathbf{Eq 1.}$ Covariance Matrix: $\mathbf{C} = \frac{1}{N} \sum_{i} \mathbf{u}_i \mathbf{u}_i^T$ (where $\mathbf{u}_i$ is block rotation vector)", color=text_color, fontsize=14)
ax.text(0.1, 0.05, r"$\mathbf{Eq 2.}$ PBA = Eigenvector of max eigenvalue ($\lambda_{max}$) from $\mathbf{C}$", color=accent_color, fontsize=16, fontweight='bold')
plt.savefig(os.path.join(out_dir, "str_metrics_PBA.png"), dpi=150, bbox_inches='tight')
plt.close()

# ==========================================================
# 4. Total Strain Energy
# ==========================================================
fig, ax = setup_fig("4. Total Strain Energy (TSE) & Specific Energy")
draw_tv_box(ax, 0.15, 0.5, angle=5)
# Show shockwave / energy absorption
ax.add_patch(patches.Circle((0.15, 0.5), 0.1, fill=True, color=accent_color, alpha=0.3))
ax.add_patch(patches.Circle((0.15, 0.5), 0.15, fill=False, color=accent_color, lw=2, ls='--'))

ax.text(0.5, 0.35, "Integrates all elastic deformation across chassis & cushions", color='#7f8c8d', fontsize=12, ha='center')
ax.text(0.1, 0.20, r"$\mathbf{Eq 1.}$ Axial Energy: $U_{axial} = \sum \frac{1}{2} k_{lin,i} (\Delta x_i)^2$", color=text_color, fontsize=14)
ax.text(0.1, 0.12, r"$\mathbf{Eq 2.}$ Bending Energy: $U_{bend} = \sum \frac{1}{2} K_{rot,i} (\theta_i)^2$", color=text_color, fontsize=14)
ax.text(0.1, 0.04, r"$\mathbf{Eq 3.} \ TSE = \sum (U_{axial} + U_{bend} + U_{torsion})$", color=accent_color, fontsize=16, fontweight='bold')
plt.savefig(os.path.join(out_dir, "str_metrics_Energy.png"), dpi=150, bbox_inches='tight')
plt.close()
