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
sim_root = os.path.join(base_root, context_string)
run_dir = os.path.join(sim_root, args.RUN_TAG)

h5_path = os.path.join(run_dir, "data.h5")

print(f"Leggendo da: {h5_path}")

with h5py.File(h5_path, 'r') as f:
    # albero completo
    print("\n=== STRUTTURA H5 ===")
    def print_tree(name, obj):
        print(name)
    f.visititems(print_tree)

    # primo frame
    types_frame0 = f['particles/TelSeq/type/value'][0].flatten()
    ids_frame0   = f['particles/TelSeq/id/value'][0].flatten()
    pos_frame0   = f['particles/TelSeq/pos/value'][0]

    print("\n=== FRAME 0 ===")
    print(f"N particelle: {len(ids_frame0)}")
    print(f"Tipi unici: {np.unique(types_frame0)}")
    print(f"Range ID: {ids_frame0.min()} → {ids_frame0.max()}")

    print("\n=== CONTEGGIO PER TIPO ===")
    for t in np.unique(types_frame0):
        mask = types_frame0 == t
        ids_of_type = ids_frame0[mask]
        print(f"  Tipo {int(t):4d}: {mask.sum():4d} particelle | ID {int(ids_of_type.min())} → {int(ids_of_type.max())}")
    print("\n=== CONNECTIVITY ===")
    for key in f['connectivity/TelSeq'].keys():
        data = f[f'connectivity/TelSeq/{key}'][:]
        print(f"  {key}: shape {data.shape}, dtype {data.dtype}")
        print(f"    primi valori: {data.flat[:10].tolist()}")
    print("\n=== DETTAGLIO CONNECTIVITY ===")
    
    # TelSeq_to_Quadriplex: shape (3,2) → 3 quadriplex, colonna 0 = TelSeq index, colonna 1 = Quadriplex index
    tsq = f['connectivity/TelSeq/TelSeq_to_Quadriplex'][:]
    print("TelSeq_to_Quadriplex (TelSeq_idx, Quadriplex_idx):")
    print(tsq)

    # Quadriplex_to_Quartet: shape (9,2) → 9 quartet, colonna 0 = Quadriplex index, colonna 1 = Quartet index
    qq = f['connectivity/TelSeq/Quadriplex_to_Quartet'][:]
    print("\nQuadriplex_to_Quartet (Quadriplex_idx, Quartet_idx):")
    print(qq)

    # ParticleHandle_to_Quadriplex: shape (441,2) → per ogni particella, a quale quadriplex appartiene
    phq = f['connectivity/TelSeq/ParticleHandle_to_Quadriplex'][:]
    print("\nParticleHandle_to_Quadriplex (ParticleHandle, Quadriplex_idx) — primi 20:")
    print(phq[:20])
    print("Quadriplex unici:", np.unique(phq[:, 1]))
    phq = f['connectivity/TelSeq/ParticleHandle_to_Quadriplex'][:]
    ids_quadriplex1 = phq[phq[:, 1] == 1, 0]
    print(f"\nQuadriplex 1: {len(ids_quadriplex1)} particelle")
    print(f"ID: {sorted(ids_quadriplex1)}")
    
    types_frame0 = f['particles/TelSeq/type/value'][0].flatten()
    ids_frame0 = f['particles/TelSeq/id/value'][0].flatten().astype(int)
    id_to_type = {pid: types_frame0[idx] for idx, pid in enumerate(ids_frame0)}
    
    print("\nTipi nel quadriplex 1:")
    types_in_q1 = [id_to_type[pid] for pid in ids_quadriplex1]
    for t in np.unique(types_in_q1):
        print(f"  Tipo {int(t):4d}: {types_in_q1.count(t):4d} particelle")