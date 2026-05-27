from pressomancy.simulation import Simulation, TelSeq, Quartet, Quadriplex
from pressomancy.helper_functions import BondWrapper
import espressomd
import numpy as np
import logging
import h5py
from pressomancy.analysis import H5DataSelector
import argparse
from espressomd import checkpointing
from utility_sim import fuoir, calc_v, find_by_id, salva_osservabili
import os
import json


context_string = "monomer"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{context_string}_logging.txt"),
        logging.StreamHandler()
    ]
)

parser = argparse.ArgumentParser()
parser.add_argument("--RUN_TAG", required=True)
parser.add_argument("--FROM_CHECKPOINT", default=None)
parser.add_argument("--KT", type=float, required=True)
parser.add_argument("--NEW_FILES", action="store_true", default=False)
args = parser.parse_args()

########################################
#FILE NAMING
########################################

base_root = "/home/stekajack/UPLOAD_VIEW"
sim_root = os.path.join(base_root, context_string)
run_dir = os.path.join(sim_root, args.RUN_TAG)

os.makedirs(run_dir, exist_ok=True)

h5_path = os.path.join(run_dir, "data.h5")
traj_path = os.path.join(run_dir, "traj.lammpstrj")
obs_path = os.path.join(run_dir, "observables.txt")
checkpoint_dir = os.path.join(run_dir, "checkpoint")

###############################################
#SIMULATION PARAMETERS
###############################################

Rf = 2e-10
etaw = 0.87e-3
t_ = 1e-9
d_ = 4e-10
mass_ = 2.5727522264874994e-20
gamma_T = 6 * np.pi * etaw * Rf * (t_ / mass_)
gamma_R = 8 * np.pi * etaw * pow(Rf, 3) * (t_ / (pow(d_, 2) * mass_))
print("gamma_T: ", gamma_T)
print("gamma_R: ", gamma_R)
N_avog = 6.02214076e23
sigma = 1.
rho_si = 0.6 * N_avog
no_obj = 3
N = int(no_obj / 3)
vol = N / rho_si
box_l = pow(vol, 1 / 3)
_box_l = box_l / 0.4e-09
box_dim = _box_l * np.ones(3)

sheets_per_quad = 3
part_per_filament = 1

md_timestep = 0.003
n_samples = 4000
steps_between_samples = 100
eq_steps = 2

###############################################
#SIMULATION SETUP — SISTEMA VUOTO
###############################################

sim_inst = Simulation(box_dim=box_dim)
sim_inst.periodicity = [True, True, True]
sim_inst.set_sys()

logging.info(f'box_dim: {sim_inst.sys.box_l}')

##################################################
#CHECKPOINTING — prima di aggiungere particelle
##################################################
if args.FROM_CHECKPOINT:
    load_dir = os.path.join(sim_root, args.FROM_CHECKPOINT, "checkpoint")

    print(f"🔍 Cercando checkpoint in: {load_dir}")
    print(f"🔍 Esiste: {os.path.exists(load_dir)}")
    if os.path.exists(load_dir):
        print(f"🔍 Contenuto: {os.listdir(load_dir)}")

    if os.path.exists(load_dir) and len(os.listdir(load_dir)) > 0:
        checkpoint = checkpointing.Checkpoint(
            checkpoint_id=f"{context_string}_{args.FROM_CHECKPOINT}",
            checkpoint_path=load_dir
        )
        try:
            checkpoint.load()
            print(f"🔁 Caricato checkpoint da {args.FROM_CHECKPOINT}")
            
            mode_io = "load"
            timestep = round(sim_inst.sys.time / md_timestep)
            current_step = timestep  # ← aggiungilo qui
        except Exception as e:
            print(f"⚠️ Errore nel caricamento checkpoint: {e}")
            print("🆕 Nuova simulazione")
            mode_io = "new"
            timestep = 0
    else:
        print(f"⚠️ Cartella checkpoint {load_dir} vuota o inesistente.")
        print("🆕 Nuova simulazione")
        mode_io = "new"
        timestep = 0

else:
    print("🆕 Nuova simulazione")
    mode_io = "new"
    timestep = 0

# Registra sempre il checkpoint per il run corrente (T2, T3, ecc.)
checkpoint = checkpointing.Checkpoint(
    checkpoint_id=f"{context_string}_{args.RUN_TAG}",
    checkpoint_path=checkpoint_dir
)
checkpoint.register("sim_inst.sys")
checkpoint.register("timestep")
##################################################
#COSTRUZIONE OGGETTI — solo se nuova simulazione
##################################################
quartet_config = Quartet.config.specify(type='broken', espresso_handle=sim_inst.sys)
quartets = [Quartet(config=quartet_config) for x in range(no_obj)]

grouped_quartets = [quartets[i:i+sheets_per_quad]
                    for i in range(0, len(quartets), sheets_per_quad)]
bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=300., r_0=2., d_r_max=2*1.2))
quadriplex_config_list = [Quadriplex.config.specify(associated_objects=elem, espresso_handle=sim_inst.sys, bonding_mode='ftf',bond_handle=bond_quad, size=np.sqrt(3)*5) for elem in grouped_quartets]
quadriplex = [Quadriplex(config=elem) for elem in quadriplex_config_list]
across_bond = BondWrapper(espressomd.interactions.FeneBond(k=3., r_0=5, d_r_max=5))
diag_bond = BondWrapper(espressomd.interactions.FeneBond(k=3., r_0=5, d_r_max=5))
grouped_quadriplexes = [quadriplex[i:i+part_per_filament:]
                        for i in range(0, len(quadriplex), part_per_filament)]
fold_types = ['hybrid'] * len(grouped_quadriplexes)
tel_config_list = [
    TelSeq.config.specify(
        n_parts=part_per_filament,
        espresso_handle=sim_inst.sys,
        associated_objects=grouped_quadriplexes[idx],
        size=quadriplex[0].params['size'] * part_per_filament + np.sqrt(3) * bond_quad.r_0 + (part_per_filament - 1),
        bond_handle=bond_quad,
        diag_bond_handle=diag_bond,
        across_bond_handle=across_bond,
        spacing=6.,
        type=fold_type,
    )
    for idx, fold_type in enumerate(fold_types)
]
telomeres = [TelSeq(config=elem) for elem in tel_config_list]
sim_inst.store_objects(telomeres)


if mode_io == "new":
    sim_inst.set_objects(telomeres)
    for telomere in telomeres:
        telomere.wrap_into_Tel()
    for quartet in quartets:
        quartet.add_h_bond_patches()
        quartet.patch_cation()
    

    for i, quad in enumerate(quadriplex):
        quad.add_dihedrals()
        quad.add_extra_bendings()


    particle_dict = {}
    for filo_id, filamento in enumerate(telomeres):  # telomeri = lista di filamenti
        #print(f"Filamento {filo_id}:")
        particle_dict[filo_id] = {}

        for quad_id, quadriplex in enumerate(filamento.associated_objects):
            #print(f"  Quadriplex {quad_id}:")
            particle_dict[filo_id][quad_id] = {}

            for quartet_id, quartet in enumerate(quadriplex.associated_objects):
                #print(f"    Quartet {quartet_id}:")
                particle_dict[filo_id][quad_id][quartet_id] = []

                for part_type, particles in quartet.type_part_dict.items():
                    #print(f"      Tipo particelle: {part_type}")

                    for p in particles:
                        #print(f"        ID: {p.id}, Posizione: {p.pos}, Oggetto: {p}")
                        particle_dict[filo_id][quad_id][quartet_id].append(p)
    quartet_global_index=0
    for fil_id, fil in enumerate(telomeres):
            #print(fil_id)
            for quad_id, quad in enumerate(fil.associated_objects):
                #print(quad_id)
                for quartet_id, quartet in enumerate(quad.associated_objects):
                    
                    particles = particle_dict[fil_id][quad_id][quartet_id]
                    cat = find_by_id(particles, 0, quartet_global_index)
                    ptcat_pos = np.array(cat.pos)
                    

                    pcat_next = None
                        # 1️⃣ prova quartet precedente, solo se esiste
                    if quartet_id == 0:
                        particles_prev = particle_dict[fil_id][quad_id][quartet_id + 1]
                        pcat_next = find_by_id(particles_prev, 0+25,quartet_global_index)
                        print('il',quartet_id,'ha trovato,',pcat_next.id)
                        
                        # 2️⃣ se non trovato, prova quartet successivo, solo se esiste
                    if quartet_id in [1, 2]:
                        print(quartet_id)
                        particles_next = particle_dict[fil_id][quad_id][0]
                        pcat_next = find_by_id(particles_next, 0 -25*quartet_id,quartet_global_index)
                        print('il',quartet_id,'ha trovato,',pcat_next.id)
                            
                    if pcat_next is None:
                            # fallback verticale se non esiste
                        print('attento')
                        v2 = np.array([0, 0, 1.0])
                    else:
                        v2 = np.array(pcat_next.pos) - ptcat_pos
                        v2 /= np.linalg.norm(v2)
                    delta = 1   # quanto vuoi spostarlo
                        
                                    # nuova posizione
                    if quartet_id==1:
                        cat.pos = np.array(cat.pos) + delta * v2
                    else:
                        cat.pos = np.array(cat.pos) - delta * v2
                    # se è l'ultimo quartet del quadriplex, elimina il catione
                    if quartet_id == 2:
                        cat.type = 200
                        print(f"Catione eliminato dall'ultimo quartet del quadriplex {quad_id}, quartet {quartet_id}")
    
                    quartet_global_index += 1 

    


##################################################
#STERIC — sempre, indipendentemente dal mode_io
##################################################



sim_inst.set_steric_custom(
    pairs=[
        ('cation', 'squareA'),
        ('cation', 'squareB'),
        ('cation', 'real'),
        ('cation','circ'),
        ('cation','charged'),

        ('squareA', 'real'),
        ('squareA', 'squareA'),

        ('squareB', 'squareB'),
        ('squareB', 'real'),

        ('real', 'real'),
        ('squareA', 'squareB'),

        ('circ', 'squareA'),
        ('circ', 'squareB'),
        ('circ', 'real'),
        ('circ', 'circ'),
        ('circ','charged')

    ],

    wca_eps=[1] * 16,

    sigma=[1*2**(-1/6)] * 16
)



dt=0.003
sim_inst.sys.time_step = dt
sim_inst.sys.integrator.run(0)

sim_inst.sys.integrator.run(0)



sim_inst.sys.non_bonded_inter[101,102].morse.set_params(eps=1.5,alpha=4.8,rmin =0.5*2**(-1/6), cutoff=1.5)

lp=np.sqrt(33/16-np.sqrt(2)/2)
sim_inst.sys.non_bonded_inter[100,27].morse.set_params(eps=3.5,alpha=4.8,rmin =lp*2**(-1/6), cutoff=17.0)

sim_inst.sys.non_bonded_inter[27, 27].wca.set_params(
    epsilon=1.0,
    sigma=1.5*2**(-1/6)
)

#for multimers
#sim_inst.sys.non_bonded_inter[26,26].lennard_jones.set_params(epsilon=2,sigma=2,cutoff=1.5*2,shift="auto")






############################################
#TEMPERATURE — sempre
############################################

sim_inst.sys.thermostat.set_langevin(
    kT=args.KT,
    gamma=gamma_T,
    gamma_rotation=gamma_R,
    seed=sim_inst.seed
)
sim_inst.sys.integrator.run(0)

############################################
#METADATA
############################################

metadata = {
    "run_tag": args.RUN_TAG,
    "from_checkpoint": args.FROM_CHECKPOINT,
    "kT": args.KT,
    "seed": sim_inst.seed,
    "box_dim": sim_inst.sys.box_l.tolist()
}

metadata_path = os.path.join(run_dir, "metadata.json")


with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

################################################
#SIMULATION SAVE
################################################

def save_all(timestep):
    sim_inst.write_part_group_to_h5(time_step=timestep)
    fuoir(sim_inst, step=timestep, box_dim=box_dim, filename=traj_path, mode="load")

############################################
#SIMULATION INITIALIZATION
############################################
if mode_io == "new":
    GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
        group_type=[TelSeq],
        h5_data_path=h5_path,
        mode='NEW'
    )
    fuoir(sim_inst, step=0, box_dim=box_dim, filename=traj_path, mode="new")

else:  # mode_io == "load"
    if args.RUN_TAG == args.FROM_CHECKPOINT:
        # prima ripara il file
        with h5py.File(h5_path, 'a') as f:
            props = f['particles/TelSeq']
            min_steps = min(props[p]['value'].shape[0] for p in props)
            print(f"🔧 Resize a {min_steps} timestep consistenti")
            for p in props:
                props[p]['value'].resize((min_steps, props[p]['value'].shape[1], props[p]['value'].shape[2]))
                props[p]['step'].resize((min_steps,))
                props[p]['time'].resize((min_steps,))
        # poi carica normalmente
        GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
            group_type=[TelSeq],
            h5_data_path=h5_path,
            mode='LOAD_NEW'
        )
        fuoir(sim_inst, step=current_step, box_dim=box_dim, filename=traj_path, mode="load")

    else:
        # run nuovo — copia file e crea nuovi
        import shutil
        h5_path_src = os.path.join(sim_root, args.FROM_CHECKPOINT, "data.h5")
        shutil.copy2(h5_path_src, h5_path)
        with h5py.File(h5_path, 'a') as f:
            props = f['particles/TelSeq']
            min_steps = min(props[p]['value'].shape[0] for p in props)
            print(f"🔧 Resize a {min_steps} timestep consistenti")
            for p in props:
                props[p]['value'].resize((min_steps, props[p]['value'].shape[1], props[p]['value'].shape[2]))
                props[p]['step'].resize((min_steps,))
                props[p]['time'].resize((min_steps,))
        GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
            group_type=[TelSeq],
            h5_data_path=h5_path,
            mode='LOAD_NEW'
        )
        fuoir(sim_inst, step=current_step, box_dim=box_dim, filename=traj_path, mode="new")
#SIMULATION RUN
############################################

current_step = round(sim_inst.sys.time / md_timestep)
total_steps = eq_steps + n_samples * steps_between_samples

# -------- EQUILIBRAZIONE --------
if current_step < eq_steps:
    print("⚖️ Equilibrazione")

    while current_step < eq_steps:
        sim_inst.sys.integrator.run(1)
        current_step = round(sim_inst.sys.time / md_timestep)

        save_all(current_step)
        pct = 100 * current_step / eq_steps
        print(f"⏱️ Equilibrazione: timestep {current_step}/{eq_steps} ({pct:.2f}%)")

# -------- PRODUZIONE --------
print("🚀 Produzione")

int_steps_completed = max(0, current_step - eq_steps)
current_sample = int(int_steps_completed / steps_between_samples)

for sample_i in range(current_sample, n_samples):
    print(f"Sample {sample_i + 1}/{n_samples}")

    sim_inst.sys.integrator.run(steps_between_samples)
    timestep = round(sim_inst.sys.time / md_timestep)

    save_all(timestep)
    pct = 100 * (timestep - eq_steps) / (n_samples * steps_between_samples)
    print(f"⏱️ Produzione: timestep {timestep} ({pct:.2f}%)")

    if sample_i % 200 == 0 and sample_i > 0:
        checkpoint.save()
        print(f"💾 Checkpoint salvato al sample {sample_i + 1} (timestep {timestep})")