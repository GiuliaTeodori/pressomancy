# utils_sim.py
import numpy as np

def calc_v(pos_target, pos_ref, Lbox):
    """
    Calcola il vettore minimo considerando le condizioni al contorno periodiche.
    """
    v = np.array(pos_target) - np.array(pos_ref)
    for i in range(3):
        v[i] -= Lbox[i] * np.rint(v[i] / Lbox[i])
    return v

def find_by_id(ptc,pid,quartet_global_index):
        
    target_id = pid + 25 * quartet_global_index
        #print(target_id)
    for p in range(len(ptc)):

        if ptc[p].id == target_id:
            return ptc[p]



def fuoir(sim_inst, step, box_dim, filename, mode="load", bonds=None):
    """
    Salva le particelle della simulazione nel formato LAMMPS.
    Se bonds è passato, scrive anche ITEM: BONDS per OVITO.
    """
    if mode == "new" and step == 0:
        with open(filename, "w") as f:
            pass

    all_particles = [p for p in sim_inst.sys.part if hasattr(p, "pos")]

    with open(filename, "a") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(all_particles)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"0 {box_dim[0]}\n")
        f.write(f"0 {box_dim[1]}\n")
        f.write(f"0 {box_dim[2]}\n")
        f.write("ITEM: ATOMS id type x y z radius\n")

        for p in all_particles:
            x, y, z = p.pos
            radius = 0.25 if p.type in (100, 24, 25) else 0.5
            f.write(f"{p.id} {p.type} {x} {y} {z} {radius}\n")

        # legami per OVITO
        if bonds is not None:
            f.write(f"ITEM: BONDS\n")
            f.write(f"{len(bonds)}\n")
            for id1, id2 in bonds:
                f.write(f"{id1} {id2}\n")

def fuoir_patches_only(sim_inst, step, box_dim, filename):
    """Salva solo le particelle di tipo 24 e 25 (patches H-bond) — solo frame iniziale."""
    patch_particles = [p for p in sim_inst.sys.part if p.type in (24, 25)]
    with open(filename, "w") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(patch_particles)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"0 {box_dim[0]}\n")
        f.write(f"0 {box_dim[1]}\n")
        f.write(f"0 {box_dim[2]}\n")
        f.write("ITEM: ATOMS id type x y z radius\n")
        for p in patch_particles:
            x, y, z = p.pos
            f.write(f"{p.id} {p.type} {x} {y} {z} 0.25\n")

def salva_osservabili(sim_inst, timestep, file_path, mode='new'):
    if mode == 'new':
        file_mode = 'w'
        scrivi_header = True
    else:
        file_mode = 'a'
        scrivi_header = False

    energy = sim_inst.sys.analysis.energy()
    E_kin = energy["kinetic"]
    E_pot = energy["total"] - energy["kinetic"]
    E_tot = energy["total"]

    with open(file_path, file_mode) as file:
        if scrivi_header:
            file.write("Timestep\tE_kinetic\tE_potential\tE_total\n")
        file.write(f"{timestep}\t{E_kin:.6e}\t{E_pot:.6e}\t{E_tot:.6e}\n")
        