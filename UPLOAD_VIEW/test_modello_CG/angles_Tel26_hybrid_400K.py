import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

run_tag = "5mvb_400K"
trj_path = f"{run_tag}/hybrid2_unf_400K_nowater_C3.xtc"
plot_angles_filename = f"{run_tag}/angles_time_{run_tag}.png"
plot_dihedrals_filename = f"{run_tag}/dihedrals_time_{run_tag}.png"

u = mda.Universe.empty(26, trajectory=True)
u.load_new(trj_path)

total_frames = len(u.trajectory)
print(f"Totale frame: {total_frames}")

# indici 0-based (sottrai 1)
angle_defs = [
    (4, 5, 9),    # 5-6-10
    (5, 9, 10),   # 6-10-11
    (10, 11, 15), # 11-12-16
    (11, 15, 16), # 12-16-17
    (16, 17, 21), # 17-18-22
    (17, 21, 22), # 18-22-23
]
angle_labels = ['5-6-10', '6-10-11', '11-12-16', '12-16-17', '17-18-22', '18-22-23']

dihedral_defs = [
    (4, 5, 9, 10),    # 5-6-10-11
    (10, 11, 15, 16), # 11-12-16-17
    (16, 17, 21, 22), # 17-18-22-23
]
dihedral_labels = ['5-6-10-11', '11-12-16-17', '17-18-22-23']

def calc_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def calc_dihedral(p1, p2, p3, p4):
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0
    n1 /= n1_norm
    n2 /= n2_norm
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.degrees(np.arctan2(y, x))

# accumula
angles_data = {label: [] for label in angle_labels}
dihedrals_data = {label: [] for label in dihedral_labels}
frames = []

for ts in u.trajectory:
    pos = u.atoms.positions
    frames.append(ts.frame)

    for (i, j, k), label in zip(angle_defs, angle_labels):
        angles_data[label].append(calc_angle(pos[i], pos[j], pos[k]))

    for (i, j, k, l), label in zip(dihedral_defs, dihedral_labels):
        dihedrals_data[label].append(calc_dihedral(pos[i], pos[j], pos[k], pos[l]))

    if ts.frame % 1000 == 0:
        print(f"calcolato frame {ts.frame}/{total_frames}")

frames = np.array(frames)

# --- plot angoli: 2 subplot da 3 angoli ciascuno ---
colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'purple', 'brown']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for (label, color) in zip(angle_labels[:3], colors[:3]):
    ax1.plot(frames, angles_data[label], linewidth=0.8, alpha=0.7,
             color=color, label=f'{label} (media={np.mean(angles_data[label]):.1f}°)')
ax1.set_ylabel('Angolo (°)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_title(f'Angoli nel tempo — Tel26 all-atom {run_tag}')

for (label, color) in zip(angle_labels[3:], colors[3:]):
    ax2.plot(frames, angles_data[label], linewidth=0.8, alpha=0.7,
             color=color, label=f'{label} (media={np.mean(angles_data[label]):.1f}°)')
ax2.set_ylabel('Angolo (°)')
ax2.set_xlabel('Frame')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plot_angles_filename, dpi=150)
plt.show()
print(f"Plot angoli salvato in {plot_angles_filename}")

# --- plot diedri ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for ax, label, color in zip(axes, dihedral_labels, colors):
    ax.plot(frames, dihedrals_data[label], linewidth=0.8, alpha=0.7, color=color)
    ax.axhline(np.mean(dihedrals_data[label]), color='black', linestyle='--', linewidth=1.5,
               label=f'Media = {np.mean(dihedrals_data[label]):.1f}°')
    ax.set_ylabel(f'Diedro {label} (°)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-180, 180)

axes[0].set_title(f'Diedri nel tempo — Tel26 all-atom {run_tag}')
axes[-1].set_xlabel('Frame')

plt.tight_layout()
plt.savefig(plot_dihedrals_filename, dpi=150)
plt.show()
print(f"Plot diedri salvato in {plot_dihedrals_filename}")