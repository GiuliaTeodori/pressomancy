from pressomancy.simulation import Simulation, TelSeq, Quartet, Quadriplex
from pressomancy.helper_functions import BondWrapper
import espressomd
import numpy as np
import logging
import h5py
from pressomancy.analysis import H5DataSelector
from espressomd import electrostatics
import argparse
from espressomd import checkpointing
from utility_sim import fuoir, calc_v,find_by_id
# Imposta i parametri per il logging
prog="test modello CG"
context_string = "Tel22_300"
# Imposta i parametri per il logging
dir_path_log = f"/home/stekajack/UPLOAD_VIEW/logging/{context_string}"
dir_path_checkpoint = f"/home/stekajack/UPLOAD_VIEW//checkpoint/{context_string}"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{dir_path_log}_logging.txt"),  # Nome del file di log
        logging.StreamHandler()  # Mantiene anche l'output su console
    ]
)


N_avog = 6.02214076e23
sigma = 1.
rho_si = 0.6*N_avog
no_obj=3
N = int(no_obj/3)
vol = N/rho_si
box_l = pow(vol, 1/3)
_box_l = box_l/0.4e-09
box_dim = _box_l*np.ones(3)

sheets_per_quad = 3
part_per_filament = 1

# Aggiungi gli argomenti da riga di comando
parser = argparse.ArgumentParser()
parser.add_argument("--MODE", choices=["NEW", "LOAD"], default="NEW", help="Modalità di esecuzione")
args = parser.parse_args()

# Definisci i percorsi di salvataggio
# puoi metterci un nome univoco per il tuo esperimento

h5_path = f"/home/stekajack/DATA//H5/{context_string}.h5"
Rf=2e-10
etaw = 0.87e-3
t_=1e-9
d_=4e-10
mass_ = 2.5727522264874994e-20
gamma_T = 6*np.pi*etaw*Rf*(t_/mass_)
gamma_R = 8*np.pi*etaw*pow(Rf, 3)*(t_/(pow(d_, 2)*mass_))
print("gamma_T: ", gamma_T)
print("gamma_R: ", gamma_R)



sim_inst = Simulation(box_dim=box_dim)
sim_inst.periodicity = [True, True, True]
sim_inst.set_sys()


logging.info(f'box_dim: {sim_inst.sys.box_l}')
#bond_hndl=BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=1.2))
#bond_hndl=BondWrapper(espressomd.interactions.FeneBond(k=50., r_0=1., d_r_max=1.5))
#quartet_config = Quartet.config.specify(bond_handle=bond_hndl,type='broken', espresso_handle=sim_inst.sys)
quartet_config = Quartet.config.specify(type='broken', espresso_handle=sim_inst.sys)
quartets = [Quartet(config=quartet_config) for x in range(no_obj)]

grouped_quartets = [quartets[i:i+sheets_per_quad]
                    for i in range(0, len(quartets), sheets_per_quad)]
bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=300., r_0=2., d_r_max=2*1.2))
quadriplex_config_list = [Quadriplex.config.specify(associated_objects=elem, espresso_handle=sim_inst.sys, bonding_mode='ftf',bond_handle=bond_quad, size=np.sqrt(3)*5) for elem in grouped_quartets]
quadriplex = [Quadriplex(config=elem) for elem in quadriplex_config_list]
across_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=4.2, d_r_max=2*1.5))
diag_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=np.sqrt(2)*(2*2.), d_r_max=2*1.5))
grouped_quadriplexes = [quadriplex[i:i+part_per_filament:]
                        for i in range(0, len(quadriplex), part_per_filament)]
fold_types = ['parallel']
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
sim_inst.set_objects(telomeres)

for telomere in telomeres:
    telomere.wrap_into_Tel()
#si=2



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
#electrostatics_solver = DH(prefactor=1.0, kappa=0, r_cut=6)

#sim_inst.sys.actors.add(electrostatics_solver)



'''
electrostatics_solver = electrostatics.DH(
    prefactor=1.0,
    kappa=0.0,
    r_cut=8.0
)

sim_inst.sys.actors.add(electrostatics_solver)
'''
sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=gamma_T,
                               gamma_rotation=gamma_R, seed=sim_inst.seed)
sim_inst.sys.integrator.run(0)

#############################################################
#      add patches                                       #
#############################################################

for quartet in quartets:
    quartet.add_h_bond_patches()
    quartet.patch_cation()
    
    


angle_harmonic = espressomd.interactions.AngleHarmonic(bend=10.0, phi0=np.pi)
sim_inst.sys.bonded_inter.add(angle_harmonic)

for quadriplex in quadriplex:
    quadriplex.add_bending_potential(angle_harmonic)
    quadriplex.add_dihedrals()
    quadriplex.add_extra_bendings()

sim_inst.sys.integrator.run(0)



#sim_inst.sys.non_bonded_inter[101,102].morse.set_params(eps=1.5,alpha=4.8,rmin =0.5*2**(-1/6), cutoff=1.0)

lp=np.sqrt(33/16-np.sqrt(2)/2)
#sim_inst.sys.non_bonded_inter[100,27].morse.set_params(eps=4.0,alpha=4.8,rmin =lp*2**(-1/6), cutoff=15)

sim_inst.sys.non_bonded_inter[27, 27].wca.set_params(
    epsilon=1.0,
    sigma=1.5*2**(-1/6)
)

sim_inst.sys.non_bonded_inter[31,31].morse.set_params(eps=10.0,alpha=4.8,rmin =2, cutoff=2.5)
sim_inst.sys.non_bonded_inter[32,32].morse.set_params(eps=10.0,alpha=4.8,rmin =2, cutoff=2.5)
sim_inst.sys.non_bonded_inter[33,33].morse.set_params(eps=10.0,alpha=4.8,rmin =2, cutoff=2.5)
sim_inst.sys.non_bonded_inter[34,34].morse.set_params(eps=10.0,alpha=4.8,rmin =2, cutoff=2.5)

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
q0 = quartets[0]

parts, _ = q0.get_owned_part()

cation = [p for p in parts
          if p.type == q0.part_types['cation']][0]

circ_parts = [p for p in parts
              if p.type == q0.part_types['patch']]

print("\n--- Distances AFTER shifts ---")

for p in circ_parts:
    d = sim_inst.sys.distance(cation, p)
    print(f"Cation ID {cation.id} — patch ID {p.id} : {d}")

#sim_inst.sys.non_bonded_inter[101,102].lennard_jones.set_params(epsilon=5.0,sigma=0.5*2**(-1/6),cutoff=1.5,shift="auto")

#############################################################
#      Checkpoint                              #
#############################################################

# Checkpointing: carica o registra
checkpoint = checkpointing.Checkpoint(checkpoint_id="checkpoint_" + context_string, checkpoint_path=dir_path_checkpoint)
if args.MODE == 'LOAD':
    try:
        checkpoint.load()
        #vedi timestep

    except Exception as e:
        import sys
        logging.error(f"❌ Errore nel loading: {e}")
        sim_inst.sys.part.clear()
        checkpoint.load()



    logging.info("✅ Checkpoint caricato correttamente.")
elif args.MODE == 'NEW':
    timestep=0
    #electrostatics_solver = DH(prefactor=1.0, kappa=0, r_cut=6.)
    # Commentato: ELECTROSTATICS feature non compilata in ESPResSo
    #sim_inst.sys.actors.add(electrostatics_solver)
    checkpoint.register("sim_inst.sys")
    checkpoint.register("timestep")
   
#############################################################
#      funzione salva dati                                          #
#############################################################


# Funzione che salva i dati fisici in un file
import os

def salva_osservabili(timestep, mode='new'):
    # Verifica la modalità
    if mode == 'new':  # Se la modalità è 'new', sovrascrivi il file
        file_mode = 'w'  # Sovrascrivi il file
        scrivi_header = True
    else:  # Se la modalità è 'load', aggiungi i dati
        file_mode = 'a'  # Aggiungi i dati al file esistente
        scrivi_header = False
    energy=sim_inst.sys.analysis.energy()
    # Dati da scrivere nel file
    E_kin = energy["kinetic"]  # Energia cinetica
    E_pot = energy["total"] - energy["kinetic"]  # Energia potenziale
    E_tot = energy["total"]  # Energia totale
    N = 12

    T = (2 * E_kin) / (3 * N) if N > 0 else 0  # Temperatura in unità ridotte
    P = sim_inst.sys.analysis.pressure()  # Pressione
    P_total = P.get('total', None)  # Ottieni la pressione totale

    # Specifica il percorso del file
    file_path = f"/home/stekajack/DATA//osservabili/{context_string}.txt"
    
     # Salva nel file
    with open(file_path, file_mode) as file:
        # Scrivi l'header se modalità "new"
        if scrivi_header:
            file.write("Timestep\tE_kinetic\tE_potential\tE_total\tTemperature\tPressure_total\n")

        # Scrivi i valori come riga tabellare
        file.write(f"{timestep}\t{E_kin:.6e}\t{E_pot:.6e}\t{E_tot:.6e}\t{T:.6e}\t{P_total:.6e}\n")

    print(f"[✓] Dati salvati (mode='{mode}') al timestep {timestep}")


data_pathh_pat=f"/home/stekajack/DATA//traiettorie/{context_string}_pat.lammpstrj"

def fuoir_patches_only(sim_inst, step, box_dim, filename, mode="load"):
    """
    Scrive un dump LAMMPS contenente SOLO le patches.
    """

    # Svuota il file all'inizio
    if mode == "new" and step == 0:
        with open(filename, "w"):
            pass

    # FILTRO: solo patch
    patch_types = {100,101,102}   # <-- aggiungi altri se servono

    patch_particles = [
        p for p in sim_inst.sys.part
        if hasattr(p, "pos") and p.type in patch_types
    ]

    with open(filename, "a") as f:
        # HEADER
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")

        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(patch_particles)}\n")

        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"0 {box_dim[0]}\n")
        f.write(f"0 {box_dim[1]}\n")
        f.write(f"0 {box_dim[2]}\n")

        f.write("ITEM: ATOMS id type x y z radius\n")

        # DATI
        for p in patch_particles:
            x, y, z = p.pos

            # raggio patch
            radius = 0.25

            f.write(f"{p.id} {p.type} {x} {y} {z} {radius}\n")

fuoir_patches_only(
    sim_inst=sim_inst,
    step=0,
    box_dim=box_dim,
    filename=data_pathh_pat,
    mode="new"
)

data_pathh=f"/home/stekajack/DATA/traiettorie/{context_string}_HB.lammpstrj"


energy = sim_inst.sys.analysis.energy()
for keys,val in energy.items():
    if val!=0:
        print(keys,val)



#############################################################
#      Integration                                          #
#############################################################

checkpoint.save()
n_samples = 1000
 # Number of data saves
#steps_between_samples = 1e5  # 100,000 steps between each sample
steps_between_samples = 10
md_timestep=0.01
current_step= round(sim_inst.sys.time/md_timestep)
completed_steps=current_step
#current_sample=round(current_step/steps_between_samples)
print('Siamo allo step',current_step)
#print('current_sample',current_sample)
eq_steps=2
steps_per_block = 1         # ad esempio 100k per blocco
n_blocks = eq_steps // steps_per_block
int_steps=n_samples*steps_between_samples
GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[TelSeq], h5_data_path=h5_path, mode=args.MODE)
checkpoint.save()
if completed_steps==0:
    #GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[TelSeq], h5_data_path=h5_path, mode=args.MODE)
    timestep=sim_inst.sys.time
    #salva_osservabili(timestep,mode='new')
    # All'inizio della simulazione
    fuoir(sim_inst, step=0, box_dim=box_dim, filename = data_pathh,mode="new")

    with open(h5_path, 'a') as f:
        f.flush()
    print(f"Inizio equilibrazione: {eq_steps} passi, suddivisi in {n_blocks} blocchi")
    logging.info(f"Equilibrazione: {n_blocks} blocchi da {steps_per_block} passi")
    for i in range(n_blocks):
        print(f"  ➤ Blocco {i+1}/{n_blocks} ({steps_per_block} passi)")
        sim_inst.sys.integrator.run(steps_per_block)
        timestep=round(sim_inst.sys.time/md_timestep)
        f = h5py.File(h5_path, 'a')
        sim_inst.write_part_group_to_h5(time_step=timestep)
        f.flush()
        f.close()
        #salva_osservabili(timestep,mode='load')
        fuoir(sim_inst, step=timestep, box_dim=box_dim, filename = data_pathh, mode="load")
        #checkpoint.save()
        logging.info(f"💾 Checkpoint salvato dopo blocco {i + 1}")
        GLOBAL_COUNTER += 1
   
    logging.info("Equilibrazione completata.")
    # Esegui
    
 
    
    print('Equilibrazione appena completata. Inizio integrazione.')
    #GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[TelSeq], h5_data_path=h5_path, mode=args.MODE)
    timestep=round(sim_inst.sys.time/md_timestep)
    #salva_osservabili(timestep,mode='new')
    # Aggiungi il flush per i file
    with open(h5_path, 'a') as f:
        f.flush()
    for sample_i in range(n_samples):
        print('Inizio sample',sample_i,'/10')
        logging.info(f"📦 Sample {sample_i + 1}/{n_samples}")
        sim_inst.sys.integrator.run(steps_between_samples)
        f = h5py.File(h5_path, 'a')
        
        f.flush()  # Scrive i dati immediatamente su disco
        f.close()  # Chiudi il file esplicitamente  
        timestep=round(sim_inst.sys.time/md_timestep)
        sim_inst.write_part_group_to_h5(time_step=timestep)
        print('Tempo reale',timestep,'ps.')
        #salva_osservabili(timestep,mode='load')
        fuoir(sim_inst, step=timestep, box_dim=box_dim, filename = data_pathh, mode="load")
        if sample_i==15000:
            checkpoint.save()
        GLOBAL_COUNTER += 1

if completed_steps<eq_steps and completed_steps!=0:
    print('Riprendiamo da equilibration step:',completed_steps,'/2000000')
    current_block= round(completed_steps/steps_per_block)
    print('Riprendo dal blocco',current_block,'/',n_blocks)
    #GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[TelSeq], h5_data_path=h5_path, mode=args.MODE)
    for i in range(current_block,n_blocks):
        print(f"  ➤ Blocco {i+1}/{n_blocks} ({steps_per_block} passi)")
        sim_inst.sys.integrator.run(steps_per_block)
        timestep=round(sim_inst.sys.time/md_timestep)
        f = h5py.File(h5_path, 'a')
        sim_inst.write_part_group_to_h5(time_step=timestep)
        f.flush()
        f.close()
        salva_osservabili(timestep,mode='load')
        fuoir(sim_inst, step=timestep, box_dim=box_dim, mode="load")
        #checkpoint.save()
        logging.info(f"💾 Checkpoint salvato dopo blocco {i + 1}")
        GLOBAL_COUNTER += 1
    logging.info("Equilibrazione completata.")
    print('Equilibrazione appena completata. Inizio integrazione.')
    #GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[TelSeq], h5_data_path=h5_path, mode=args.MODE)
    with open(h5_path, 'a') as f:
        f.flush()
    for sample_i in range(n_samples):
       print('Inizio sample',sample_i,'/10')
       logging.info(f"📦 Sample {sample_i + 1}/{n_samples}")
       sim_inst.sys.integrator.run(steps_between_samples)
       timestep=round(sim_inst.sys.time/md_timestep)
       f = h5py.File(h5_path, 'a')
       sim_inst.write_part_group_to_h5(time_step=timestep)
       f.flush()  # Scrive i dati immediatamente su disco
       f.close()
       
       print('Tempo reale',timestep,'ps.')
       salva_osservabili(timestep,mode='load')
       fuoir(sim_inst, step=timestep, box_dim=box_dim, mode="load")
       #checkpoint.save()
       GLOBAL_COUNTER += 1


if completed_steps>=eq_steps:
    print('Equilibrazione già completata. Inizio integrazione.')
    #GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[TelSeq], h5_data_path=h5_path, mode=args.MODE)
    int_steps_completed = completed_steps-eq_steps
    if int_steps_completed<int_steps:
        current_sample=round(int_steps_completed/steps_between_samples)
        print('Riprendiamo dal sample:',current_sample,'/10')
        #GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[TelSeq], h5_data_path=h5_path, mode=args.MODE)
        with open(h5_path, 'a') as f:
            f.flush()
        for sample_i in range(current_sample,n_samples):
            print('Inizio sample',sample_i,'/10')
            logging.info(f"📦 Sample {sample_i + 1}/{n_samples}")
            sim_inst.sys.integrator.run(steps_between_samples)
            f = h5py.File(h5_path, 'a')
            timestep=round(sim_inst.sys.time/md_timestep)
            f = h5py.File(h5_path, 'a')
            sim_inst.write_part_group_to_h5(time_step=timestep)
            f.flush()  # Scrive i dati immediatamente su disco
            f.close()
            
            print('Tempo reale',timestep,'ps.')
            salva_osservabili(timestep,mode='load')
            fuoir(sim_inst, step=timestep, box_dim=box_dim, mode="load")
            #checkpoint.save()
            GLOBAL_COUNTER += 1
    if completed_steps==int_steps:
        print('Simulazione completata.')
# Salvataggio dell'ultimo checkpoint
checkpoint.save()
logging.info("🏁 Simulazione completata. Ultimo checkpoint salvato.")
from espressomd.io.writer import vtf
with open(f"/home/stekajack/DATA/traiettorie/folded_tel_compulsion.vtf", mode="w+t") as fp:
    vtf.writevsf(sim_inst.sys, fp)
    vtf.writevcf(sim_inst.sys, fp)
    for _ in range(2000):
        sim_inst.sys.integrator.run(1)
        vtf.writevcf(sim_inst.sys, fp)
        fp.flush()

print('simulazione completata')

