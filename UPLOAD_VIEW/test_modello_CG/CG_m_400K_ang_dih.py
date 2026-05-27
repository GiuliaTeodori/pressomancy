import numpy as np
import h5py
import os
import argparse
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--RUN_TAG", required=True)
args = parser.parse_args()

# parametri fisici
N_avog = 6.02214076e23
rho_si = 0.6 * N_avog
no_obj = 3
N = int(no_obj / 3)
vol = N / rho_si
box_l = pow(vol, 1/3)
_box_l = box_l / 0.4e-09
box_dim = _box_l * np.ones(3)

context_string = "monomer"
base_root = "/home/stekajack/UPLOAD_VIEW"
sim_root = os.path.join(base_root, context_string)
run_dir = os.path.join(sim_root, args.RUN_TAG)

h5_path = os.path.join(run_dir, "data.h5")
angles_filename = os.path.join(run_dir, f"angles_{context_string}.txt")
dihedrals_filename = os.path.join(run_dir, f"dihedrals_{context_string}.txt")

# parametri temporali
SAVE_EVERY_STEPS = 100       # i dati sono salvati ogni 100 step
STEP_DT_NS = 0.003           # ogni step dura 0.003 ns
STEPS_TO_SKIP = 110000        # scarta i primi 110.000 step di equilibrazione
FRAMES_TO_SKIP = STEPS_TO_SKIP // SAVE_EVERY_STEPS  # = 100 frame

print(f"Leggendo da: {h5_path}")
print(f"Frame da saltare (equilibrazione): {FRAMES_TO_SKIP} "
      f"(= {STEPS_TO_SKIP} step = {STEPS_TO_SKIP * STEP_DT_NS:.1f} ns)")

order_path = os.path.join(run_dir, "corner_order.json")
with open(order_path, 'r') as f:
    corner_order = json.load(f)

with h5py.File(h5_path, 'r') as data_file:
    positions_all = data_file['particles/TelSeq/pos/value'][:]
    ids_all = data_file['particles/TelSeq/id/value'][:]

ids_frame0 = ids_all[0].flatten()
id_to_idx = {int(pid): idx for idx, pid in enumerate(ids_frame0)}

# =========================
# DEFINIZIONE ANGOLI E DIEDRI (USANDO ID)
# =========================

angle_defs_ids =  [
    (20, 70, 49),
    (70, 49, 24),
    (24, 74, 55),
    (74, 55, 5),
    (5, 30, 26),
    (30, 26, 1),
]
angle_labels = ['20,70,49', '70,49,24', '24,74,55', '74,55,5', '5,30,26', '30,26,1']

dihedral_defs_ids = [
    (20, 70, 51,  1),
    ( 1, 26, 30,  5),
    ( 5, 55, 49, 24),
]

# conversione ID -> indici
angle_defs    = [(id_to_idx[i], id_to_idx[j], id_to_idx[k])
                 for (i, j, k) in angle_defs_ids]
dihedral_defs = [(id_to_idx[i], id_to_idx[j], id_to_idx[k], id_to_idx[l])
                 for (i, j, k, l) in dihedral_defs_ids]

# =========================
# FUNZIONI
# =========================

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
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.degrees(np.arctan2(y, x))

# =========================
# LOOP SU TUTTI I FRAME (dopo equilibrazione)
# =========================

total_frames = positions_all.shape[0]
start_frame  = 110000

print(f"Frame totali nel file: {total_frames}")
print(f"Frame analizzati:      {total_frames - start_frame}")

angles_data    = []
dihedrals_data = []

for i in range(start_frame, total_frames):
    positions = positions_all[i]

    frame_angles = [calc_angle(positions[i1], positions[i2], positions[i3])
                    for (i1, i2, i3) in angle_defs]

    frame_dihedrals = [calc_dihedral(positions[i1], positions[i2],
                                     positions[i3], positions[i4])
                       for (i1, i2, i3, i4) in dihedral_defs]

    angles_data.append(frame_angles)
    dihedrals_data.append(frame_dihedrals)

    if i % 1000 == 0:
        print(f"  calcolato frame {i}/{total_frames}")

angles_data    = np.array(angles_data)
dihedrals_data = np.array(dihedrals_data)

# asse temporale in ns
time_ns = np.arange(start_frame, total_frames) * SAVE_EVERY_STEPS * STEP_DT_NS

# =========================
# SALVATAGGIO
# =========================

np.savetxt(angles_filename,    angles_data,    header="Angoli (deg)")
np.savetxt(dihedrals_filename, dihedrals_data, header="Diedri (deg)")

print(f"\nAngoli salvati in    {angles_filename}")
print(f"Diedri salvati in   {dihedrals_filename}")

# =========================
# STATISTICHE
# =========================

print("\n--- ANGOLI ---")
for i, ang in enumerate(angle_defs_ids):
    mean = np.mean(angles_data[:, i])
    std  = np.std(angles_data[:, i])
    print(f"  {ang}: {mean:.2f} ± {std:.2f} deg")

print("\n--- DIEDRI ---")
for i, dih in enumerate(dihedral_defs_ids):
    mean = np.mean(dihedrals_data[:, i])
    std  = np.std(dihedrals_data[:, i])
    print(f"  {dih}: {mean:.2f} ± {std:.2f} deg")

# =========================
# PLOT: ANDAMENTO TEMPORALE ANGOLI
# =========================

plt.figure()
for i, ang in enumerate(angle_defs_ids):
    plt.plot(time_ns, angles_data[:, i], label=str(ang))
plt.xlabel("Tempo (ns)")
plt.ylabel("Angolo (deg)")
plt.title("Evoluzione temporale angoli")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "angles_time.png"))
plt.show()

# =========================
# PLOT: ANDAMENTO TEMPORALE DIEDRI
# =========================

plt.figure()
for i, dih in enumerate(dihedral_defs_ids):
    plt.plot(time_ns, dihedrals_data[:, i], label=str(dih))
plt.xlabel("Tempo (ns)")
plt.ylabel("Diedro (deg)")
plt.title("Evoluzione temporale diedri")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "dihedrals_time.png"))
plt.show()

# =========================
# PLOT: ISTOGRAMMI ANGOLI
# =========================

plt.figure()
for i, ang in enumerate(angle_defs_ids):
    plt.hist(angles_data[:, i], bins=50, alpha=0.5, label=str(ang))
plt.xlabel("Angolo (deg)")
plt.ylabel("Frequenza")
plt.title("Distribuzione angoli")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "angles_hist.png"))
plt.show()

# =========================
# PLOT: ISTOGRAMMI DIEDRI
# =========================

plt.figure()
for i, dih in enumerate(dihedral_defs_ids):
    plt.hist(dihedrals_data[:, i], bins=50, alpha=0.5, label=str(dih))
plt.xlabel("Diedro (deg)")
plt.ylabel("Frequenza")
plt.title("Distribuzione diedri")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "dihedrals_hist.png"))
plt.show()