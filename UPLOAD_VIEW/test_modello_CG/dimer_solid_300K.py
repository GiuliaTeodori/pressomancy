from pressomancy.simulation import Simulation, Crowder, Filament, Quartet, Quadriplex
from pressomancy.helper_functions import BondWrapper
import espressomd
import numpy as np
import logging
#import ArgumentParser
from espressomd import checkpointing
import argparse
import math

import time
import h5py
from collections import defaultdict
from numpy import trace, sqrt

# Imposta i parametri per il logging
logging.basicConfig(level=logging.INFO)
 
# Costante di Boltzmann (nelle unità di ESPResSo)
k_B = 1.0  

N_avog = 6.02214076e23

sigma = 1.
rho_si = 0.6*N_avog #0.6= CONCENTRAZIONE mM
no_obj=6
N = int(no_obj/3)
vol = N/rho_si #volume sistema
box_l = pow(vol, 1/3) #lato box
_box_l = box_l/0.4e-09 #in unità ridotte
box_dim = _box_l*np.ones(3) #vettore dimensioni scatola
_rho = N/pow(_box_l, 3)#densità ridotta


sheets_per_quad = 3 #ogni quadruplex formato t da tre strati
part_per_filament = 2 #ogni filamento è composto da due quadruplex
no_crowders=10 #numero di particelle che simulano l'ambiente molecolare
part_per_ligand=2 #numero di crowder che formano un legame (Ligand)

# Aggiungi gli argomenti da riga di comando
parser = argparse.ArgumentParser()
parser.add_argument("--MODE", choices=["NEW", "LOAD"], default="NEW", help="Modalità di esecuzione")
args = parser.parse_args()
# Definisci i percorsi di salvataggio
context_string = "dimer_solid_300K"  # puoi metterci un nome univoco per il tuo esperimento
dir_path = f"/home/stekajack/DATA/{context_string}"
h5_path = f"/home/stekajack/DATA/{context_string}.h5"

#creazione istanza 
sim_inst = Simulation(box_dim=box_dim) #crea simulazione con una certa dimensione della box
sim_inst.set_sys() #inizializza il sistema
logging.info(f'box_dim: {sim_inst.sys.box_l}') #registra informazioni su dimesione box

#creazione dei quartet (sheets?)
quartet_configuration = Quartet.config.specify(espresso_handle=sim_inst.sys,type='solid') #i quartet sono di tipo  solid
quartets = [Quartet(config=quartet_configuration) for x in range(no_obj)]#lista dei quartet con no_object elementi (30) con la stessa configurzione
sim_inst.store_objects(quartets) # memorizzati nella simulazione
#legame Fene definito con k costante elastica, r lunghezza del legame, d distanza max
bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
#divide gli sheets in gruppi da 3 (G4)
grouped_quartets = [quartets[i:i+sheets_per_quad]
                    for i in range(0, len(quartets), sheets_per_quad)]
#configura i G4 con dimensione proporzionale a 5*rad(3) e li lega con FENE
quadriplex_configuration_list = [Quadriplex.config.specify(size=np.sqrt(3)*5., espresso_handle=sim_inst.sys, bond_handle=bond_quad, associated_objects=elem) for elem in grouped_quartets]
quadriplex = [Quadriplex(config=configuration) for configuration in quadriplex_configuration_list]
sim_inst.store_objects(quadriplex)  #li memorizza



#crea i filamenti dai quadruplex (nuovo legame FENE  per i filamenti)
bond_pass = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
#raggruppamento quadruplex per i filamenti
grouped_quadriplexes = [quadriplex[i:i+part_per_filament:]
                        for i in range(0, len(quadriplex), part_per_filament)]
#configura i filamenti con spacing (Lunghezza) 
filament_configuration_list = [Filament.config.specify(size=quadriplex[0].params['size']*part_per_filament+np.sqrt(3)*bond_pass.r_0+(part_per_filament-1), n_parts=part_per_filament, espresso_handle=sim_inst.sys, bond_handle=bond_pass, associated_objects=elem, spacing=6.) for elem in grouped_quadriplexes]
#li crea
filaments = [Filament(config=configuration) for configuration in filament_configuration_list]
sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments) #memorizza


#crea legami tra i G4 per il filmaneto (2)
for filament in filaments:        
    filament.bond_quadriplexes()

#avvia intergazione (non la simulazione, non c'è avanzamneto nel tempo, setup)
sim_inst.sys.integrator.run(0)
'''
#creazione configurazione per i crowders
crowder_configuration=Crowder.config.specify(sigma=6., size=6., espresso_handle=sim_inst.sys)
#lista dei crowders con la config sopra
crowders = [Crowder(config=crowder_configuration)
            for x in range(no_crowders)]
sim_inst.store_objects(crowders) #memorizzazione
#raggruppamento crowders in gruppi (part per ligand) -> 5 coppie di crowders
grouped_crowders = [crowders[i:i+part_per_ligand]
                for i in range(0, len(crowders), part_per_ligand)]
#legame FENE, wrappato per poter essere usato da pressomancy, facilità l'applicazione
bender_pass = BondWrapper(espressomd.interactions.FeneBond(
    k=10, r_0=6, d_r_max=6*1.5))
filament_configuration_list = [Filament.config.specify(sigma=6,size=6*part_per_ligand, n_parts=part_per_ligand, espresso_handle=sim_inst.sys, bond_handle=bender_pass, associated_objects=elem) for elem in grouped_crowders]
#crea la configurazione di filamneto
filaments = [Filament(config=elem) for elem in filament_configuration_list]
sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)
#collega i filamenti ai crowders con u legame tra i loro centri di massa
for filament in filaments:
    filament.bond_center_to_center(type_key='crowder')
   ''' 
#attiva le interazioni steriche per evitare che le particelle si sovrappongano (WCA)
sim_inst.set_steric(key=('real', 'virt'), wca_eps=1.)

#aggiunge dei punti di interazione ai G4, permettendo di creare interazioni specifiche tra quente unità
for el in quadriplex:
    el.add_patches_triples()
#imposta il VdW tra questi punti di interazione (patch)
sim_inst.set_vdW(key=('patch',), lj_eps=5, lj_size=2.)
#langevin e poi verifica che tutto sia configurato correttamente
sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=sim_inst.seed)


sim_inst.sys.integrator.run(0)

#############################################################
#      Checkpoint                              #
#############################################################

# Checkpointing: carica o registra
checkpoint = checkpointing.Checkpoint(checkpoint_id="checkpoint_" + context_string, checkpoint_path=dir_path)
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
    checkpoint.register("sim_inst.sys")
    checkpoint.register("timestep")
    logging.info("🆕 Registrato nuovo sistema per checkpoint.")


#############################################################
#      Equilibrazione modulare                             #
#############################################################
'''
#equilibration_total_steps = int(2e6)   # 2 milioni
#steps_per_block = 100000              # ad esempio 100k per blocco
equilibration_total_steps = int(2e5)   # 2 milioni
steps_per_block = 100              # ad esempio 100k per blocco
n_blocks = equilibration_total_steps // steps_per_block

print(f"Inizio equilibrazione: {equilibration_total_steps} passi, suddivisi in {n_blocks} blocchi")
logging.info(f"Equilibrazione: {n_blocks} blocchi da {steps_per_block} passi")


for i in range(n_blocks):
    print(f"  ➤ Blocco {i+1}/{n_blocks} ({steps_per_block} passi)")
    sim_inst.sys.integrator.run(steps_per_block)
    checkpoint.save()
    logging.info(f"💾 Checkpoint salvato dopo blocco {i + 1}")

    # Facoltativo: log energia, temperatura, ecc.
    energy = sim_inst.sys.analysis.energy()
    print(f"    ↪ Energia dopo blocco {i+1}: {energy['total']:.3f}")
    
    # Potresti anche salvare un checkpoint HDF5 qui se vuoi sicurezza

print("Equilibrazione completata.")
logging.info("Equilibrazione completata.")
'''
#############################################################
#      funzione salva dati                                          #
#############################################################
'''
def salva_osservabili(timestep):
    E_kin = compute_kinetic_energy()
    E_pot = compute_potential_energy()
    E_tot = E_kin + E_pot
    T = compute_temperature()
    P = compute_pressure()
    
    with open("output_dati.txt", "a") as f:
        f.write(f"{timestep}\t{E_kin:.5f}\t{E_pot:.5f}\t{E_tot:.5f}\t{T:.5f}\t{P:.5f}\n")
# Scrive solo l'intestazione all'inizio
with open("data_Tel48_kT=1.txt", "w") as f:
    f.write("timestep\tE_kin\tE_pot\tE_tot\tTemperature\tPressure\n")
'''

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
    N = len(sim_inst.sys.part)

    T = (2 * E_kin) / (3 * N) if N > 0 else 0  # Temperatura in unità ridotte
    P = sim_inst.sys.analysis.pressure()  # Pressione
    P_total = P.get('total', None)  # Ottieni la pressione totale

    # Specifica il percorso del file
    #file_path = '/home/stekajack/DATA/data_2JSL_kT_1.txt'
    file_path= f"/home/stekajack/DATA/data_{context_string}.txt"
    
     # Salva nel file
    with open(file_path, file_mode) as file:
        # Scrivi l'header se modalità "new"
        if scrivi_header:
            file.write("Timestep\tE_kinetic\tE_potential\tE_total\tTemperature\tPressure_total\n")

        # Scrivi i valori come riga tabellare
        file.write(f"{timestep}\t{E_kin:.6e}\t{E_pot:.6e}\t{E_tot:.6e}\t{T:.6e}\t{P_total:.6e}\n")

    print(f"[✓] Dati salvati (mode='{mode}') al timestep {timestep}")

#############################################################
#      Integration                                          #
#############################################################
# Ciclo di produzione (sampling)
GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[Filament], h5_data_path=h5_path, mode=args.MODE)

n_samples = 30000  # Number of data saves
#steps_between_samples = 1e5  # 100,000 steps between each sample
steps_between_samples = 100
md_timestep=0.01
current_step= round(sim_inst.sys.time/md_timestep)
completed_steps=current_step
#current_sample=round(current_step/steps_between_samples)
print('current_step',current_step)
#print('current_sample',current_sample)
eq_steps=2
steps_per_block = 1             # ad esempio 100k per blocco
n_blocks = eq_steps // steps_per_block
# Se necessario, arrotonda per eccesso per garantire che i passi siano sempre inclusi
n_blocks = math.ceil(n_blocks)
int_steps=n_samples*steps_between_samples


f = h5py.File(h5_path, 'a')
f.flush()  # Scrive i dati immediatamente su disco
f.close() 


if completed_steps==0:
    print(f"Inizio equilibrazione: {eq_steps} passi, suddivisi in {n_blocks} blocchi")
    logging.info(f"Equilibrazione: {n_blocks} blocchi da {steps_per_block} passi")
    fuoir(sim_inst, step=0, box_dim=box_dim, mode="new")
    for i in range(n_blocks):
        print(f"  ➤ Blocco {i+1}/{n_blocks} ({steps_per_block} passi)")
        sim_inst.sys.integrator.run(steps_per_block)
        checkpoint.save()
        fuoir(sim_inst, step=0, box_dim=box_dim, mode="load")
        logging.info(f"💾 Checkpoint salvato dopo blocco {i + 1}")
   
    logging.info("Equilibrazione completata.")
    print('Equilibrazione appena completata. Inizio integrazione.')
    timestep=sim_inst.sys.time
    salva_osservabili(timestep,mode='new')
    # Aggiungi il flush per i file
    with open(h5_path, 'a') as f:
        f.flush()
    for sample_i in range(n_samples):
        print('Inizio sample',sample_i,'/10')
        logging.info(f"📦 Sample {sample_i + 1}/{n_samples}")
        sim_inst.sys.integrator.run(steps_between_samples)
        f = h5py.File(h5_path, 'a')
        sim_inst.write_part_group_to_h5(time_step=sample_i)
        f.flush()  # Scrive i dati immediatamente su disco
        f.close()  # Chiudi il file esplicitamente  
        fuoir(sim_inst, step=0, box_dim=box_dim, mode="load")
        timestep=sim_inst.sys.time
        print('Tempo reale',timestep,'ps.')
        #salva_osservabili(timestep,mode='load')
        #checkpoint.save()
        GLOBAL_COUNTER += 1

if completed_steps<eq_steps and completed_steps!=0:
    print('Riprendiamo da equilibration step:',completed_steps,'/2000000')
    current_block= round(completed_steps/steps_per_block)
    print('Riprendo dal blocco',current_block,'/',n_blocks)
    for i in range(current_block,n_blocks):
        print(f"  ➤ Blocco {i+1}/{n_blocks} ({steps_per_block} passi)")
        sim_inst.sys.integrator.run(steps_per_block)
        checkpoint.save()
        logging.info(f"💾 Checkpoint salvato dopo blocco {i + 1}")
    logging.info("Equilibrazione completata.")
    print('Equilibrazione appena completata. Inizio integrazione.')
    timestep=sim_inst.sys.time
    salva_osservabili(timestep,mode='new')
    with open(h5_path, 'a') as f:
        f.flush()
    for sample_i in range(n_samples):
       print('Inizio sample',sample_i,'/10')
       logging.info(f"📦 Sample {sample_i + 1}/{n_samples}")
       sim_inst.sys.integrator.run(steps_between_samples)
       f = h5py.File(h5_path, 'a')
       sim_inst.write_part_group_to_h5(time_step=sample_i)
       f.flush()  # Scrive i dati immediatamente su disco
       f.close()
       timestep=sim_inst.sys.time
       print('Tempo reale',timestep,'ps.')
       salva_osservabili(timestep,mode='load')
       checkpoint.save()
       GLOBAL_COUNTER += 1


if completed_steps>=eq_steps:
    print('Equilibrazione già completata. Inizio integrazione.')
    int_steps_completed = completed_steps-eq_steps
    if int_steps_completed<int_steps:
        current_sample=round(int_steps_completed/steps_between_samples)
        print('Riprendiamo dal sample:',current_sample,'/10')
        with open(h5_path, 'a') as f:
            f.flush()
        for sample_i in range(current_sample,n_samples):
            print('Inizio sample',sample_i,'/10')
            logging.info(f"📦 Sample {sample_i + 1}/{n_samples}")
            sim_inst.sys.integrator.run(steps_between_samples)
            f = h5py.File(h5_path, 'a')
            sim_inst.write_part_group_to_h5(time_step=sample_i)
            f.flush()  # Scrive i dati immediatamente su disco
            f.close()
            timestep=sim_inst.sys.time
            print('Tempo reale',timestep,'ps.')
            salva_osservabili(timestep,mode='load')
            checkpoint.save()
            GLOBAL_COUNTER += 1
    if completed_steps==int_steps:
        print('Simulazione completata.')
# Salvataggio dell'ultimo checkpoint
checkpoint.save()
logging.info("🏁 Simulazione completata. Ultimo checkpoint salvato.")




#############################################################
#      visulization                                       #
#############################################################
# Estrazione delle posizioni delle particelle
'''
with h5py.File(h5_path, "r") as h5f:
    positions = h5f["particles/Filament/pos/value"][:]  # Le posizioni sono in un array (100, 780, 3)
    types = h5f["particles/Filament/type/value"][:]  # Tipi di particelle, array (100, 780, 1)

def save_to_xyz_vmd(positions, types, filename="output.xyz"):
    particle_type_names = {
        1: "Type1",     # tipo 1
        2: "Type2",     # tipo 2
        4: "Type4",     # tipo 2
        5: "Type5",     # tipo 2
    }

    with open(filename, "w") as file:
        num_atoms = positions.shape[1]  # Numero di particelle
        for step in range(positions.shape[0]):
            file.write(f"{num_atoms}\n")
            file.write(f"Step {step + 1}\n")
            for i in range(num_atoms):
                pos = positions[step, i]
                particle_type = types[step, i, 0]
                particle_name = particle_type_names.get(int(particle_type), "Unknown")
                file.write(f"{particle_name} {' '.join(map(str, pos))}\n")

save_to_xyz_vmd(positions, types, filename="/home/stekajack/DATA/G4/Tel48_kT=1.xyz")
'''
#salvataggio con doppioni
def save_to_xyz_vmd(h5_path, filename="output.xyz"):
    particle_type_names = {
        1: "Type1",
        2: "Type2",
        4: "Type4",
        5: "Type5",
    }

    with h5py.File(h5_path, "r") as h5f:
        pos_dataset = h5f["particles/Filament/pos/value"]
        type_dataset = h5f["particles/Filament/type/value"]

        n_frames = pos_dataset.shape[0]
        num_atoms = pos_dataset.shape[1]

        print(f"➡️  Salvo {n_frames} frame nel file XYZ '{filename}'")

        with open(filename, "w") as file:
            for step in range(n_frames):
                file.write(f"{num_atoms}\n")
                file.write(f"Step {step + 1}\n")
                for i in range(num_atoms):
                    pos = pos_dataset[step, i]
                    particle_type = type_dataset[step, i, 0]
                    particle_name = particle_type_names.get(int(particle_type), "Unknown")
                    file.write(f"{particle_name} {' '.join(map(str, pos))}\n")

    print(f"✅ File salvato correttamente: {filename}")

# Usa la funzione
save_to_xyz_vmd(h5_path, filename=f"/home/stekajack/DATA/{context_string}.xyz")
 #salvataggio senza doppioni
def save_to_xyz_vmd_unique_timesteps(h5_path, filename="output.xyz"):
    particle_type_names = {
        1: "Type1",
        2: "Type2",
        4: "Type4",
        5: "Type5",
    }

    with h5py.File(h5_path, "r") as h5f:
        pos_dataset = h5f["particles/Filament/pos/value"]
        type_dataset = h5f["particles/Filament/type/value"]
        
        # Qui supponiamo che i tempi siano salvati in questo dataset
        time_dataset = h5f["particles/Filament/pos/time"][:]  # shape (n_frames,)
        
        # Mappiamo i tempi all'indice del frame: se ci sono duplicati, l'ultimo vince
        time_to_index = {}
        for idx, t in enumerate(time_dataset):
            time_to_index[round(t, 6)] = idx  # round per sicurezza contro floating point noise

        # Ordinamento per tempo (opzionale, ma utile se servisse ordine temporale)
        unique_indices = sorted(time_to_index.values(), key=lambda i: time_dataset[i])
        num_atoms = pos_dataset.shape[1]

        print(f"➡️  Scrivo {len(unique_indices)} frame unici su {filename}")

        with open(filename, "w") as file:
            for step_count, idx in enumerate(unique_indices):
                file.write(f"{num_atoms}\n")
                file.write(f"Step {step_count + 1}  Time: {time_dataset[idx]} ps\n")
                for i in range(num_atoms):
                    pos = pos_dataset[idx, i]
                    particle_type = type_dataset[idx, i, 0]
                    particle_name = particle_type_names.get(int(particle_type), "Unknown")
                    file.write(f"{particle_name} {' '.join(map(str, pos))}\n")

    print(f"✅ File XYZ salvato: {filename}")

# Esegui
save_to_xyz_vmd_unique_timesteps(h5_path, filename=f"/home/stekajack/DATA/{context_string}_deduped.xyz")
 

