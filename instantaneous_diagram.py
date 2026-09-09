import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import math

# Constants
TAU = 2 * np.pi

# Cooling spiral (one cycle)
lambdas = np.linspace(0, 0.999, 800)
tau = TAU * (1 - lambdas)
x = tau * np.cos(tau)
y = tau * np.sin(tau)

# Late cooling points approaching origin
lam_end = np.array([0.90, 0.95, 0.98, 0.99, 0.995, 0.999])
tau_end = TAU * (1 - lam_end)
x_end = tau_end * np.cos(tau_end)
y_end = tau_end * np.sin(tau_end)

# Reset point on the circle of radius 2π (phase 0 for clarity)
x_reset = TAU
y_reset = 0.0

fig, ax = plt.subplots(figsize=(11, 11))

# Continuous cooling spiral
ax.plot(x, y, color='royalblue', linewidth=2.2, label='Continuous cooling spiral\nτ(λ) = 2π(1−λ)', zorder=2)

# Circle of radius 2π
circle = Circle((0, 0), TAU, fill=False, edgecolor='seagreen', linestyle='--', linewidth=2.0, label='Circle of radius 2π')
ax.add_patch(circle)

# Origin
ax.plot(0, 0, 'o', color='black', markersize=9, zorder=5, label='Origin (τ → 0)')

# Late points
ax.plot(x_end, y_end, 'o', color='darkorange', markersize=7, zorder=4, label='Approaching origin')

# Reset point
ax.plot(x_reset, y_reset, 'o', color='crimson', markersize=14, zorder=6, label='Reset point (τ = 2π)')

# Discontinuous radial jump arrow
arrow = FancyArrowPatch(
    (x_end[-1], y_end[-1]),
    (x_reset * 0.92, y_reset),  # slightly short of the point for clarity
    arrowstyle='->', mutation_scale=25,
    color='crimson', linewidth=2.8, linestyle='--',
    connectionstyle='arc3,rad=0.0', zorder=3
)
ax.add_patch(arrow)

# Annotation
ax.annotate('Instantaneous radial jump\nfrom ≈0 back to radius 2π',
            xy=(3.8, 1.8), fontsize=12, color='crimson',
            ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='crimson', alpha=0.9))


# Formatting
ax.set_aspect('equal')
ax.set_xlim(-7.8, 7.8)
ax.set_ylim(-7.8, 7.8)
ax.set_xlabel('x (temperature × cos)', fontsize=12)
ax.set_ylabel('y (temperature × sin)', fontsize=12)
ax.set_title('Instantaneous Radial Jump\nfrom the Origin back to the Circle of Radius $2\\pi$', fontsize=15, pad=12)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(True, linestyle=':', alpha=0.4)
ax.axhline(0, color='gray', linewidth=0.7)
ax.axvline(0, color='gray', linewidth=0.7)

plt.tight_layout()
plt.savefig('/home/workdir/plots/radial_jump_chart.png', dpi=160, bbox_inches='tight')
print("Chart saved to /home/workdir/plots/radial_jump_chart.png")
print(f"Reset radius: {math.hypot(x_reset, y_reset):.6f} (exactly 2π = {TAU:.6f})")
