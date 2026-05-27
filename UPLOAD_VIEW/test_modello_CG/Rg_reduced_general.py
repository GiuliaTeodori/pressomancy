import numpy as np
import h5py
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--RUN_TAG", required=True)
args = parser.parse_args()

N_avog = 6.02214076e23
rho_si = 0.6 * N_avog
no_obj = 15
N = int(no_obj / 3)
vol = N / rho_si
box_l = pow(vol, 1/3)
_box_l = box_l / 0.4e-09
box_dim = _box_l * np.ones(3)

context_string = "pentam"
base_root = "/home/stekajack/UPLOAD_VIEW"
sim_root = os.path.join(base_root, context_string)
run_dir = os.path.join(sim_root, args.RUN_TAG)

h5_path = os.path.join(run_dir, "data.h5")
txt_filename = os.path.join(run_dir, f"Rg_quadriplex_central_{context_string}.txt")

print(f"Leggendo da: {h5_path}")

exclude_types = [200, 27, 100, 24, 25]

with h5py.File(h5_path, 'r') as data_file:
    positions_all = data_file['particles/TelSeq/pos/value'][:]
    types_all     = data_file['particles/TelSeq/type/value'][:]
    ids_all       = data_file['particles/TelSeq/id/value'][:]
    phq           = data_file['connectivity/TelSeq/ParticleHandle_to_Quadriplex'][:]


n_quadriplex = phq[:, 1].max() + 1
central_quadriplex_indices = set(range(1, n_quadriplex - 1))  # {1, 2}
print(f"Quadriplex centrali: {sorted(central_quadriplex_indices)}")

ids_quadriplex_central = set(phq[np.isin(phq[:, 1], list(central_quadriplex_indices)), 0].tolist())
print(f"Particelle nei quadriplex centrali: {len(ids_quadriplex_central)}")

# mappa id → indice posizione dal frame 0 (statica, gli ID non cambiano)
ids_frame0 = ids_all[0].flatten().astype(int)
id_to_idx = {pid: idx for idx, pid in enumerate(ids_frame0)}

total_frames = positions_all.shape[0]
start_frame = max(0, total_frames - 6000)
print(f"Totale frame: {total_frames}")
print(f"Analizzando frame {start_frame} → {total_frames}")

def calculate_CM(positions, box):
    CM = np.zeros(3)
    num_particles = len(positions)
    for axis in range(3):
        xi, zeta = 0.0, 0.0
        for pos in positions:
            pos_axis = pos[axis] + 0.5 * box[axis]
            theta = pos_axis * 2.0 * np.pi / box[axis]
            xi += np.cos(theta)
            zeta += np.sin(theta)
        xi /= num_particles
        zeta /= num_particles
        theta_mean = np.arctan2(-zeta, -xi) + np.pi
        CM[axis] = box[axis] * theta_mean / (2.0 * np.pi) - 0.5 * box[axis]
    return CM

def calculate_rg(positions, box):
    CM = calculate_CM(positions, box)
    pos_centered = np.array([pos - CM - box * np.round((pos - CM) / box)
                              for pos in positions])
    n = len(pos_centered)
    S = np.zeros((3, 3))
    for r in pos_centered:
        rv = r.reshape(3, 1)
        S += np.dot(rv, rv.T)
    S /= n
    evals = np.sort(np.linalg.eigh(S)[0])
    return np.sqrt(np.sum(evals)) * 0.4  # in nm

rg_list = []
for i in range(start_frame, total_frames):
    types_frame = types_all[i].flatten()
    ids_frame   = ids_all[i].flatten().astype(int)

    # maschera: nel quadriplex centrale E tipo non escluso
    mask = np.array([
        (pid in ids_quadriplex_central) and (types_frame[idx] not in exclude_types)
        for idx, pid in enumerate(ids_frame)
    ])

    positions_filtered = positions_all[i][mask]

    if len(positions_filtered) == 0:
        print(f"⚠️ frame {i} vuoto dopo filtro")
        continue

    rg_list.append(calculate_rg(positions_filtered, box_dim))

    if i % 1000 == 0:
        print(f"calcolato frame {i}/{total_frames} — N particelle usate: {mask.sum()}")

mean_rg = np.mean(rg_list)
std_rg  = np.std(rg_list)
print(f"\nRg medio quadriplex centrale (ultimi 3000 frame): {mean_rg:.6f} ± {std_rg:.6f} nm")

with open(txt_filename, "w") as f:
    f.write("mean_rg_nm\tstd_rg_nm\n")
    f.write(f"{mean_rg:.6f}\t{std_rg:.6f}\n")
print(f"Risultati salvati in {txt_filename}")