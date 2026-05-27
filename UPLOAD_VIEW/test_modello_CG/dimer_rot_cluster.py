from pressomancy.simulation import Simulation, TelSeq, Quartet, Quadriplex
from pressomancy.helper_functions import BondWrapper
import espressomd
import numpy as np
import logging
import h5py
from pressomancy.analysis import H5DataSelector
import argparse
from espressomd import checkpointing
from utility_sim1 import fuoir, fuoirpatch, calc_v, find_by_id, salva_osservabili
import os
import json
import shutil

context_string = "dimer_cluster"

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
args = parser.parse_args()

########################################
# FILE NAMING
########################################

base_root = "/home/stekajack/UPLOAD_VIEW"
sim_root = os.path.join(base_root, context_string)
run_dir = os.path.join(sim_root, args.RUN_TAG)

os.makedirs(run_dir, exist_ok=True)

h5_path = os.path.join(run_dir, "data.h5")
traj_path = os.path.join(run_dir, "traj.lammpstrj")
trajpatch_path = os.path.join(run_dir, "trajpatch.lammpstrj")
obs_path = os.path.join(run_dir, "observables.txt")
checkpoint_dir = os.path.join(run_dir, "checkpoint")

###############################################
# SIMULATION PARAMETERS
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
no_obj = 6
N = int(no_obj / 3)
vol = N / rho_si
box_l = pow(vol, 1 / 3)
_box_l = box_l / 0.4e-09
box_dim = _box_l * np.ones(3)

sheets_per_quad = 3
part_per_filament = 2
fold_types = ['hybrid'] * int(no_obj / (sheets_per_quad * part_per_filament))

md_timestep = 0.003
n_samples = 8000
steps_between_samples = 100
eq_steps = 2

###############################################
# SIMULATION SETUP — SISTEMA VUOTO
###############################################

sim_inst = Simulation(box_dim=box_dim)
sim_inst.set_sys(timestep=0.003)
sim_inst.periodicity = [True, True, True]
sim_inst.set_sys()

logging.info(f'box_dim: {sim_inst.sys.box_l}')

##################################################
# CHECKPOINTING — prima di aggiungere particelle
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
            current_step = timestep
        except Exception as e:
            print(f"⚠️ Errore nel caricamento checkpoint: {e}")
            print("🆕 Nuova simulazione")
            mode_io = "new"
            timestep = 0
            current_step = 0
    else:
        print(f"⚠️ Cartella checkpoint {load_dir} vuota o inesistente.")
        print("🆕 Nuova simulazione")
        mode_io = "new"
        timestep = 0
        current_step = 0
else:
    print("🆕 Nuova simulazione")
    mode_io = "new"
    timestep = 0
    current_step = 0

# Registra sempre il checkpoint per il run corrente
checkpoint = checkpointing.Checkpoint(
    checkpoint_id=f"{context_string}_{args.RUN_TAG}",
    checkpoint_path=checkpoint_dir
)
checkpoint.register("sim_inst.sys")
checkpoint.register("timestep")

##################################################
# COSTRUZIONE OGGETTI
##################################################
logging.info(f'box_dim: {sim_inst.sys.box_l}')
bond_hndl = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=1.5))
bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=300., r_0=2., d_r_max=2 * 1.2))
quartets = []
quadriplex = []
for fold_type in fold_types:
    for _ in range(part_per_filament):
        if fold_type == 'antiparallel':
            quartet_types = ['brokenB', 'brokenA', 'brokenA']
        elif fold_type == 'hybrid':
            quartet_types = ['brokenA', 'brokenB', 'brokenA']
        else:
            quartet_types = ['brokenA', 'brokenA', 'brokenA']

        quartet_triplet = []
        for quartet_type in quartet_types:
            quartet_config = Quartet.config.specify(
                type=quartet_type,
                espresso_handle=sim_inst.sys,
            )
            quartet = Quartet(config=quartet_config)
            quartets.append(quartet)
            quartet_triplet.append(quartet)

        quadriplex_config = Quadriplex.config.specify(
            associated_objects=quartet_triplet,
            espresso_handle=sim_inst.sys,
            bonding_mode='ftf',
            bond_handle=bond_quad,
            size=np.sqrt(3) * 5,
        )
        quadriplex.append(Quadriplex(config=quadriplex_config))

        assert len(quartet_triplet) == 3
        if fold_type == 'antiparallel':
            expected_types = ['brokenB', 'brokenA', 'brokenA']
        elif fold_type == 'hybrid':
            expected_types = ['brokenA', 'brokenB', 'brokenA']
        else:
            expected_types = ['brokenA', 'brokenA', 'brokenA']
        observed_types = [q.params['type'] for q in quartet_triplet]
        assert observed_types == expected_types, f"quartet type mismatch for fold={fold_type}: expected={expected_types}, got={observed_types}"

assert len(quartets) == no_obj, f"unexpected quartet count: expected={no_obj}, got={len(quartets)}"

diag_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=np.sqrt(2) * 4.2, d_r_max=2 * 1.5))
across_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=4.2, d_r_max=2 * 1.5))
grouped_quadriplexes = [quadriplex[i:i + part_per_filament]
                        for i in range(0, len(quadriplex), part_per_filament)]

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

    angle_harmonic = espressomd.interactions.AngleHarmonic(bend=1000.0, phi0=np.pi)
    sim_inst.sys.bonded_inter.add(angle_harmonic)

    for quad in quadriplex:  # rinominato per evitare sovrascrittura
        quad.add_bending_potential(angle_harmonic)
        quad.add_dihedrals()
        quad.add_extra_bendings()

    particle_dict = {}
    for filo_id, filamento in enumerate(telomeres):
        particle_dict[filo_id] = {}
        for quad_id, quad in enumerate(filamento.associated_objects):
            particle_dict[filo_id][quad_id] = {}
            for quartet_id, quartet in enumerate(quad.associated_objects):
                particle_dict[filo_id][quad_id][quartet_id] = []
                for part_type, particles in quartet.type_part_dict.items():
                    for p in particles:
                        particle_dict[filo_id][quad_id][quartet_id].append(p)

    quartet_global_index = 0
    for fil_id, fil in enumerate(telomeres):
        for quad_id, quad in enumerate(fil.associated_objects):
            for quartet_id, quartet in enumerate(quad.associated_objects):
                particles = particle_dict[fil_id][quad_id][quartet_id]
                if quartet_id == 0:
                    charged_type = Quartet.part_types.get('charged')
                    if charged_type is not None:
                        for p in particles:
                            #print(f"Controllo particella {p.id} di tipo {p.type} (charged_type={charged_type})")
                            if p.type == charged_type:
                                print(f"  - Particella {p.id} è di tipo 'charged', cambiando tipo a 400")
                                p.type = 400   # o quello che vuoi

                

                quartet_global_index += 1

    corner_order = {}
    for quad in quadriplex:
        chain = quad.get_corner_chain()
        corner_order[quad.who_am_i] = [c.id for c in chain]

    order_path = os.path.join(run_dir, "corner_order.json")
    with open(order_path, 'w') as f:
        json.dump(corner_order, f)
    print(f"Corner order salvato in {order_path}")

##################################################
# STERIC — sempre, indipendentemente dal mode_io
##################################################

sim_inst.set_steric_custom(
    pairs=[
        ('real', 'real'),
        ('real', 'virt'),
        ('real', 'circ'),
        ('real', 'cation'),
        ('virt', 'circ'),
        ('virt', 'cation'),
        ('circ', 'circ'),
        ('cation', 'cation'),
        ('virt', 'virt')
    ],
    wca_eps=[1, 1, 1, 1, 1, 1, 1, 1, 1],
    sigma=[0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87]
)

dt = 0.003
sim_inst.sys.time_step = dt
sim_inst.sys.integrator.run(0)

sim_inst.sys.non_bonded_inter[24, 25].morse.set_params(eps=2.75, alpha=4.8, rmin=0.5 * 2 ** (-1 / 6), cutoff=1.5)

sim_inst.sys.non_bonded_inter[26,26].lennard_jones.set_params(epsilon=2,sigma=2,cutoff=1.5*2,shift="auto")


############################################
# TEMPERATURE — sempre
############################################

sim_inst.sys.thermostat.set_langevin(
    kT=args.KT,
    gamma=gamma_T,
    gamma_rotation=gamma_R,
    seed=sim_inst.seed
)
sim_inst.sys.integrator.run(0)

############################################
# METADATA — sempre
############################################

metadata = {
    "run_tag": args.RUN_TAG,
    "from_checkpoint": args.FROM_CHECKPOINT,
    "kT": args.KT,
    "seed": sim_inst.seed,
    "box_dim": sim_inst.sys.box_l.tolist()
}
with open(os.path.join(run_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

################################################
# SIMULATION SAVE
################################################

def save_all(timestep):
    sim_inst.write_part_group_to_h5(time_step=timestep)
    fuoir(sim_inst, step=timestep, box_dim=box_dim, filename=traj_path, mode="load")

def repair_h5(path):
    """Ripara il file h5 in caso di crash — tronca al minimo timestep consistente."""
    with h5py.File(path, 'a') as f:
        props = f['particles/TelSeq']
        min_steps = min(
            props[p]['value'].shape[0]
            for p in props
            if 'value' in props[p]
        )
        print(f"🔧 Resize a {min_steps} timestep consistenti")
        for p in props:
            if 'value' not in props[p]:
                continue
            props[p]['value'].resize((min_steps, props[p]['value'].shape[1], props[p]['value'].shape[2]))
            props[p]['step'].resize((min_steps,))
            props[p]['time'].resize((min_steps,))

############################################
# SIMULATION INITIALIZATION
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
        # stesso run — ripara e riprende
        repair_h5(h5_path)
        GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
            group_type=[TelSeq],
            h5_data_path=h5_path,
            mode='LOAD_NEW'
        )
        fuoir(sim_inst, step=current_step, box_dim=box_dim, filename=traj_path, mode="load")
    else:
        # run nuovo — copia file dal run precedente e crea nuovi
        h5_path_src = os.path.join(sim_root, args.FROM_CHECKPOINT, "data.h5")
        shutil.copy2(h5_path_src, h5_path)
        repair_h5(h5_path)
        GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
            group_type=[TelSeq],
            h5_data_path=h5_path,
            mode='LOAD_NEW'
        )
        fuoir(sim_inst, step=current_step, box_dim=box_dim, filename=traj_path, mode="new")

############################################
# SIMULATION RUN
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