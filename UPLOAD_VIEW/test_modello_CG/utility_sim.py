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

def fuoir(sim_inst, step, box_dim, filename, mode="load"):
    """
    Salva le particelle della simulazione nel formato LAMMPS.
    
    Parameters:
    - sim_inst: istanza della simulazione
    - step: timestep corrente
    - box_dim: dimensioni del box (x, y, z)
    - filename: file di output
    - mode: "new" per sovrascrivere il file (solo all'inizio), "load" per aggiungere
    """
    # Se modalità è "new", svuota il file (solo la prima volta)
    if mode == "new" and step == 0:
        with open(filename, "w") as f:
            pass  # Svuota il file

    all_particles = [p for p in sim_inst.sys.part if hasattr(p, "pos")]

    with open(filename, "a") as f:  # Append sempre dopo il primo
        # HEADER
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(all_particles)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"0 {box_dim[0]}\n")
        f.write(f"0 {box_dim[1]}\n")
        f.write(f"0 {box_dim[2]}\n")
        f.write("ITEM: ATOMS id type x y z radius\n")

        # PARTICLE DATA
        for p in all_particles:
            x, y, z = p.pos
            #radius = 0.25 if p.type in (100, 101, 103, 104, 105) else 0.5
            radius = 0.25 if p.type in (100, 101, 102,103,104) else 0.5
            f.write(f"{p.id} {p.type} {x} {y} {z} {radius}\n")

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

    N = len(sim_inst.sys.part)

    T = (2 * E_kin) / (3 * N) if N > 0 else 0
    P = sim_inst.sys.analysis.pressure()
    P_total = P.get('total', None)

    with open(file_path, file_mode) as file:
        if scrivi_header:
            file.write("Timestep\tE_kinetic\tE_potential\tE_total\tTemperature\tPressure_total\n")

        file.write(f"{timestep}\t{E_kin:.6e}\t{E_pot:.6e}\t{E_tot:.6e}\t{T:.6e}\t{P_total:.6e}\n")

    #print(f"[✓] Dati salvati in {file_path} (mode='{mode}') al timestep {timestep}")