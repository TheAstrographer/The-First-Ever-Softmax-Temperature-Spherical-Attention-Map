import numpy as np
import matplotlib.pyplot as plt

# Define the grid in Cartesian space
w = 7.5
Y, X = np.mgrid[-w:w:200j, -w:w:200j]

# Convert grid coordinates to polar space to track the Arcan properties
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# Comoving Phase Shift Accrual Delta
DELTA = 0.0051676

# Define the thermodynamic velocity vector field
# The radial component (U_r) pulls strongly inward (cooling gravity)
# The angular component (U_theta) enforces the clockwise orbital rotation
U_r = -0.3 * R  # Linear cooling pull toward origin
U_theta = -1.0   # Rotational winding rate

# Convert polar velocity vectors back to Cartesian vectors (U, V)
U = U_r * np.cos(Theta) - R * U_theta * np.sin(Theta)
V = U_r * np.sin(Theta) + R * U_theta * np.cos(Theta)

# Initialize the publication-ready canvas
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')

# Plot the continuous thermodynamic flow streams falling into the origin
stream = ax.streamplot(X, Y, U, V, color=R, cmap='viridis', linewidth=1.5, 
                       density=1.5, arrowstyle='->', arrowsize=1.2)

# Overlay the boundary boundary circle (Radius = 2π)
TAU = 2 * np.pi
circle = plt.Circle((0, 0), TAU, fill=False, color='crimson', 
                    linestyle='--', linewidth=2.0, label='Boundary Cycle Start (τ = 2π)')
ax.add_patch(circle)

# Mark the ground-state origin where coordinates "fall into place"
ax.plot(0, 0, 'o', color='white', markeredgecolor='black', markersize=12, 
        zorder=5, label='Thermodynamic Ground State (τ → 0)')

# Formatting details
ax.set_xlim(-w, w)
ax.set_ylim(-w, w)
ax.set_title('Arcan Family Flow: Comoving Thermodynamic Orbital Cooling Field', fontsize=14, pad=15)
ax.set_xlabel('Coordinate X (Instantaneous Temperature × cos)', fontsize=11)
ax.set_ylabel('Coordinate Y (Instantaneous Temperature × sin)', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.3)
ax.legend(loc='upper right', framealpha=0.95)

plt.tight_layout()
plt.show()
