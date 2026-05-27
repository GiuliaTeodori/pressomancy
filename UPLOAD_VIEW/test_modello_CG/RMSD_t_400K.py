import numpy as np
import h5py
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--RUN_TAG", required=True)
args = parser.parse_args()

N_avog = 6.02214076e23
rho_si = 0.6 * N_avog
no_obj = 9
N = int(no_obj / 3)
vol = N / rho_si
box_l = pow(vol, 1/3)
_box_l = box_l / 0.4e-09
box_dim = _box_l * np.ones(3)

context_string = "trimer"
base_root = "/home/stekajack/UPLOAD_VIEW"
run_dir = os.path.join(base_root, context_string, args.RUN_TAG)

h5_path = os.path.join(run_dir, "data.h5")
txt_filename = os.path.join(run_dir, f"RMSD_{context_string}.txt")

print(f"Leggendo da: {h5_path}")
print(f"Salvando in: {txt_filename}")

exclude_types = [200, 27, 100, 101, 102]

def calculate_CM(positions, box):
    pos_mapped = positions + 0.5 * box
    theta = pos_mapped * 2.0 * np.pi / box
    xi = np.mean(np.cos(theta), axis=0)
    zeta = np.mean(np.sin(theta), axis=0)
    theta_mean = np.arctan2(-zeta, -xi) + np.pi
    return box * theta_mean / (2.0 * np.pi) - 0.5 * box

def align_to_ref(positions, ref_positions, box):
    """Centra sul CM e applica correzione PBC, poi allinea con Kabsch."""
    CM = calculate_CM(positions, box)
    pos = positions - CM - box * np.round((positions - CM) / box)
    CM_ref = calculate_CM(ref_positions, box)
    ref = ref_positions - CM_ref - box * np.round((ref_positions - CM_ref) / box)

    # Kabsch algorithm
    H = pos.T @ ref
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    pos_aligned = pos @ R.T
    return pos_aligned, ref

def calculate_rmsd(pos_aligned, ref):
    diff = pos_aligned - ref
    return np.sqrt(np.mean(np.sum(diff**2, axis=1))) * 0.4  # in nm

with h5py.File(h5_path, 'r') as data_file:
    positions_all = data_file['particles/TelSeq/pos/value'][:]
    types_all = data_file['particles/TelSeq/type/value'][:]

total_frames = positions_all.shape[0]
print(f"Totale frame: {total_frames}")

# costruisci il frame di riferimento (frame 0)
types_ref = types_all[0].flatten()
mask_ref = ~np.isin(types_ref, exclude_types)
ref_positions = positions_all[0][mask_ref]

rmsd_list = []
for i in range(total_frames):
    types = types_all[i].flatten()
    positions = positions_all[i]
    mask = ~np.isin(types, exclude_types)
    positions_filtered = positions[mask]

    if len(positions_filtered) == 0 or len(positions_filtered) != len(ref_positions):
        print(f"⚠️ frame {i} saltato")
        continue

    pos_aligned, ref = align_to_ref(positions_filtered, ref_positions, box_dim)
    rmsd = calculate_rmsd(pos_aligned, ref)
    rmsd_list.append(rmsd)

    if i % 1000 == 0:
        print(f"calcolato frame {i}/{total_frames}")

frames = np.arange(len(rmsd_list))
with open(txt_filename, "w") as f:
    f.write("Frame\tRMSD\n")
    for i in range(len(frames)):
        f.write(f"{frames[i]}\t{rmsd_list[i]:.6f}\n")

print(f"Results saved in {txt_filename}")