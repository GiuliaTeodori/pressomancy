import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

run_tag = "5mvb_400K"
trj_path = f"{run_tag}/hybrid2_unf_400K_nowater_C3.xtc"
plot_filename = f"{run_tag}/angles_groups_time_{run_tag}.png"

u = mda.Universe.empty(26, trajectory=True)
u.load_new(trj_path)

total_frames = len(u.trajectory)
print(f"Totale frame: {total_frames}")

# gruppi di angoli (indici 0-based)
angle_groups = [
    {
        'title': 'Gruppo 1',
        'defs': [(4,5,6), (5,6,7), (6,7,8), (7,8,9), (8,9,10)],
        'labels': ['5-6-7', '6-7-8', '7-8-9', '8-9-10', '9-10-11'],
    },
    {
        'title': 'Gruppo 2',
        'defs': [(10,11,12), (11,12,13), (12,13,14), (13,14,15), (14,15,16)],
        'labels': ['11-12-13', '12-13-14', '13-14-15', '14-15-16', '15-16-17'],
    },
    {
        'title': 'Gruppo 3',
        'defs': [(16,17,18), (17,18,19), (18,19,20), (19,20,21), (20,21,22)],
        'labels': ['17-18-19', '18-19-20', '19-20-21', '20-21-22', '21-22-23'],
    },
]

def calc_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# accumula
for group in angle_groups:
    group['data'] = {label: [] for label in group['labels']}

frames = []

for ts in u.trajectory:
    pos = u.atoms.positions
    frames.append(ts.frame)
    for group in angle_groups:
        for (i, j, k), label in zip(group['defs'], group['labels']):
            group['data'][label].append(calc_angle(pos[i], pos[j], pos[k]))
    if ts.frame % 1000 == 0:
        print(f"calcolato frame {ts.frame}/{total_frames}")

frames = np.array(frames)
colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'purple']

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

for ax, group in zip(axes, angle_groups):
    for label, color in zip(group['labels'], colors):
        data = np.array(group['data'][label])
        ax.plot(frames, data, linewidth=0.8, alpha=0.7, color=color,
                label=f'{label} (media={np.mean(data):.1f}°)')
    ax.set_ylabel('Angolo (°)')
    ax.set_title(group['title'])
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Frame')
fig.suptitle(f'Angoli nel tempo — Tel26 all-atom {run_tag}', fontsize=13)
plt.tight_layout()
plt.savefig(plot_filename, dpi=150)
plt.show()
print(f"Plot salvato in {plot_filename}")