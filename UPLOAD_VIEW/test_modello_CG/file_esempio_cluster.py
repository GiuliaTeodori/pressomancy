from pressomancy.simulation import Simulation, TelSeq, Quartet, Quadriplex
from pressomancy.helper_functions import BondWrapper
import espressomd
import numpy as np
import logging
import h5py
from pressomancy.analysis import H5DataSelector
from espressomd.electrostatics import DH
import argparse
from espressomd import checkpointing
# Imposta i parametri per il logging
logging.basicConfig(level=logging.INFO)
 

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
context_string = "2JSL_300_T"  # puoi metterci un nome univoco per il tuo esperimento
dir_path = f"/home/stekajack/UPLOAD_VIEW/{context_string}"

h5_path = f"/home/stekajack/UPLOAD_VIEW/{context_string}.h5"



sim_inst = Simulation(box_dim=box_dim)
sim_inst.periodicity = [True, True, True]
sim_inst.set_sys()


logging.info(f'box_dim: {sim_inst.sys.box_l}')
#bond_hndl=BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=1.2))
bond_hndl=None
#quartet_config = Quartet.config.specify(bond_handle=bond_hndl,type='broken', espresso_handle=sim_inst.sys)
quartet_config = Quartet.config.specify(type='broken', espresso_handle=sim_inst.sys)
quartets = [Quartet(config=quartet_config) for x in range(no_obj)]

grouped_quartets = [quartets[i:i+sheets_per_quad]
                    for i in range(0, len(quartets), sheets_per_quad)]
bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
quadriplex_config_list = [Quadriplex.config.specify(associated_objects=elem, espresso_handle=sim_inst.sys, bonding_mode='ftf',bond_handle=bond_quad, size=np.sqrt(3)*5) for elem in grouped_quartets]
quadriplex = [Quadriplex(config=elem) for elem in quadriplex_config_list]

diag_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=np.sqrt(2)*(2*2.), d_r_max=2*1.5))
grouped_quadriplexes = [quadriplex[i:i+part_per_filament:]
                        for i in range(0, len(quadriplex), part_per_filament)]
tel_config_list = [TelSeq.config.specify(n_parts=part_per_filament, espresso_handle=sim_inst.sys, associated_objects=elem, size=quadriplex[0].params['size']*part_per_filament+np.sqrt(3)*bond_quad.r_0+(part_per_filament-1),bond_handle=bond_quad,diag_bond_handle=diag_bond,spacing=6.) for elem in grouped_quadriplexes]
telomeres = [TelSeq(config=elem) for elem in tel_config_list]
sim_inst.store_objects(telomeres)
sim_inst.set_objects(telomeres)

for telomere in telomeres:
    telomere.wrap_into_Tel()
#si=2

#steric
sim_inst.set_steric_custom(
    pairs=[('cation', 'circ'),('cation', 'squareA'),('cation', 'squareB'),('cation', 'charged'),('cation', 'real'),('cation', 'cation'),
     ('circ', 'squareA'), ('circ', 'squareB'),('circ', 'real'),('circ', 'circ'),
     ('squareA', 'charged'),('squareA', 'real'),('squareA', 'squareA'),('charged', 'real'),
     ('squareB', 'squareB'),('squareB', 'charged'),('squareB', 'real'),
     ('real','real'),('charged','circ'),('squareA','squareB'),('charged','charged')
     ], wca_eps=[1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1,1, 1,1,1, 1, 1, 1,1,1], sigma=[1,1, 1, 1,1, 1,1, 1, 1, 1, 1, 1,1, 1,1, 1, 1, 1, 1,1,1])#sigma=[ 0.95, 0.95, 0.95, 0.95, 0.95,0.95, 0.95, 0.95, 0.95, 0.95, 0.95,0.95, 0.95,0.95,0.95, 0.95, 0.95, 0.95, 0.95,0.95,0.95])
#('squareA','squareB'),('charged,)
electrostatics_solver = DH(prefactor=1.0, kappa=0, r_cut=6)

sim_inst.sys.actors.add(electrostatics_solver)

sim_inst.sys.thermostat.set_langevin(kT=0.68, gamma=1.0, seed=sim_inst.seed)
sim_inst.sys.integrator.run(0)

#############################################################
#      add patches                                       #
#############################################################
 # Struttura: filamenti[filo_id][quad_id][quartet_id] = lista di particelle (ParticleHandle)
particle_dict = {}

# Supponiamo di avere una lista di filamenti
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

def calc_v(pos_target, pos_ref, Lbox):
    v = np.array(pos_target) - np.array(pos_ref)
    for i in range(3):
        v[i] -= Lbox[i] * np.rint(v[i] / Lbox[i])
    
    return v

                         

mesh =  {
    'assoc': {
        1: [2, 3, 6, 7, 8],
        5: [4, 9, 10, 13, 14],
        20: [11, 12, 15, 16, 21],
        24: [17, 18, 19, 22, 23]
    }
}



def add_patches_to_particles(telomere_list, particle_dict, system, Lbox, sigma=1.0):
    patch_radius = 0.25
    
    ref_map = {
        3:  (2, False),
        4:  (5, True),
        6:  (1, False),
        7:  (2, False),
        8:  (7, False),
        9:  (10, False),
        11: (15, False),
        12: (16, False),
        13: (9, False),
        14: (10, False),
        16: (15, False),
        17: (18, True),
        18: (23, False),
        19: (24, False),
        21: (20, False),
        22: (23, True),
    }
    
    rep_map=[(2, 'rep1','rep2'), (7,'rep3','rep4' ), (9, 'rep1','rep2'), (10, 'rep3','rep4'), (15, 'rep1','rep2'), (16, 'rep3','rep4'), (18, 'rep1','rep2'), (23,'rep3','rep4')]

    '''
    ref_map = {
        3:  (1, False),
        4:  (5, True),
        6:  (1, False),
        11: (20, False),
        14: (5, False),
        19: (24, False),
        21: (20, False),
        22: (24, True),
    }
    '''
    
    pairs = [
        (3, 4, 100),
        (8, 9, 101),
        (14, 19, 102),
        (13, 18, 103),
        (16, 17, 104),
        (21, 22, 105),
        (6, 11, 106),
        (7, 12, 107)
    ]

    

    quartet_global_index = 0
    
    def find_by_id(ptc,pid):
        
        target_id = pid + 25 * quartet_global_index
        #print(target_id)
        for p in range(len(ptc)):

            if ptc[p].id == target_id:
                return ptc[p]
        
    def find_id_real(ptc,id_target):
    
        realpt = 0  # default: se non viene trovato nulla, restituisce id_target stesso
        objreal = None 
        for master_id, assoc_list in mesh['assoc'].items():
            if id_target in assoc_list:
                realpt = master_id + quartet_global_index * 25
        for p in range(len(ptc)):

            if ptc[p].id ==realpt:
                objreal=ptc[p]


        return realpt,objreal

        
    for fil_id, fil in enumerate(telomere_list):
        #print(fil_id)
        for quad_id, quad in enumerate(fil.associated_objects):
            #print(quad_id)
            for quartet_id, quartet in enumerate(quad.associated_objects):
                
                particles = particle_dict[fil_id][quad_id][quartet_id]
                #print('qratet',quartet_id)
                #print('global',quartet_global_index)
                '''
                for num, rep_a, rep_b in rep_map:
                    if quad_id % 2 != 0:
                       rep_a, rep_b = rep_b, rep_a  # inverti
                    targ=find_by_id(particles,num)
                    id_real,p_real=find_id_real(particles,num)
                    print('targid',targ.id,'num1',rep_a)
                    try:
                        # Se non siamo nell'ultimo quartet (max 2), usa il successivo
                        if quartet_id == 0:
                            coda = find_by_id(particle_dict[fil_id][quad_id][quartet_id + 1], num + 25)
                        if quartet_id == 1:
                            # Altrimenti siamo nell'ultimo: torna indietro
                            coda = find_by_id(particle_dict[fil_id][quad_id][quartet_id - 1], num - 25)
                        if quartet_id == 2:
                            rep_a, rep_b = rep_b, rep_a
                            # Altrimenti siamo nell'ultimo: torna indietro
                            coda = find_by_id(particle_dict[fil_id][quad_id][quartet_id - 1], num - 25)
                        print('coda', coda.id)
                    except (KeyError, IndexError, ValueError):
                        print(f"[!] Errore nel trovare coda per i={i}, quartet_id={quartet_id}")
                        continue
                    vect=calc_v(targ.pos, coda.pos, Lbox)
                    vect= vect/np.linalg.norm(vect)
                    patch_1 = np.array(targ.pos) + vect * (patch_radius)
                    patch_2 = np.array(targ.pos) - vect * (patch_radius)
                    ptch_1=quartet.add_particle(type_name=rep_a,pos=patch_1)
                    ptch_1.radius = patch_radius
                    #print('idrel',id_real)
                    ptch_1.vs_auto_relate_to(p_real)
                    ptch_2=quartet.add_particle(type_name=rep_b,pos=patch_2)
                    ptch_2.radius = patch_radius
                    ptch_2.vs_auto_relate_to(p_real)
                '''
                for (id1, id2, patch_type_base) in pairs:
                    #patch_type = patch_type_base + 10 * quartet_global_index  # offset tipo patch per quartetto

                    # Set parametri interazione SOLO per coppie patch_type e patch_type (uguale per entrambi i tipi nella coppia)
                    # Oppure estendere a coppie di tipi differenti (vedi sotto)
                    
                    p1 = find_by_id(particles,id1)
                    #print(p1.id)
                    p2 = find_by_id(particles,id2)
                    id_real_1,p_real_1=find_id_real(particles,id1)
                    #print(id_real_1)
                    #p_real_1=find_by_id(particles,id_real_1)

                    #print(p_real_1.id)
                    id_real_2,p_real_2=find_id_real(particles,id2)
                    #print('p1',p1.id,'idreal1',id_real_1,'preal1',p_real_1.id)
                    #print('p2',p2.id,'idreal1',id_real_2,'preal1',p_real_2.id)
                    if p1 is None or p2 is None:
                        print(f"Particella non trovata per coppia ({id1},{id2}) nel quartet {quartet_global_index}")
                        continue

                    # patch per p1
                    ref1_id, invert1 = ref_map.get(id1, (None, False))
                    if ref1_id is None:
                        print(f"No ref map for id {id1}")
                        continue
                    p1_ref = find_by_id(particles,ref1_id)
                    if p1_ref is None:
                        print(f"Particella di riferimento non trovata: {ref1_id} per id {id1}")
                        continue
                    v1 = calc_v(p1.pos, p1_ref.pos, Lbox)
                    patch_pos_1 = np.array(p1.pos) + v1 * (patch_radius)
                    #pv1 = system.part.add(pos=patch_pos_1, type=patch_type)
                    

                    pv1=quartet.add_particle(type_name='patch',pos=patch_pos_1)
                    pv1.radius = patch_radius
                    pv1.vs_auto_relate_to(p_real_1)
            
                    #p1.add_bond((rigid_bond, pv1))

                    # patch per p2
                    ref2_id, invert2 = ref_map.get(id2, (None, False))
                    if ref2_id is None:
                        print(f"No ref map per id {id2}")
                        continue
                    p2_ref = find_by_id(particles,ref2_id)
                    if p2_ref is None:
                        print(f"Particella di riferimento non trovata: {ref2_id} per id {id2}")
                        continue
                    v2 = calc_v(p2.pos, p2_ref.pos, Lbox)
                    patch_pos_2 = np.array(p2.pos) + v2 * (patch_radius) / 1
                    #pv2 = system.part.add(pos=patch_pos_2, type=patch_type)
                    pv2=quartet.add_particle(type_name='patch',pos=patch_pos_2)
                    
                    pv2.radius = patch_radius
                    pv2.vs_auto_relate_to(p_real_2)
                    #p2.add_bond((rigid_bond, pv2))
                    dist= np.linalg.norm(pv1.pos - pv2.pos)
                
                quartet_global_index += 1

add_patches_to_particles(telomeres, particle_dict, sim_inst.sys, box_dim)

sim_inst.sys.non_bonded_inter[100, 100].lennard_jones_cos.set_params(epsilon=2.5, sigma=0.5/(2**(1/6)), cutoff=1)

'''
sim_inst.sys.non_bonded_inter[101, 101].lennard_jones.set_params(
    epsilon=5,
    sigma=1.5 / (2 ** (1/6)),
    cutoff=1.6,
    shift='auto'
)
sim_inst.sys.non_bonded_inter[103, 103].lennard_jones.set_params(
    epsilon=5,
    sigma=1.5 / (2 ** (1/6)),
    cutoff=1.6,
    shift='auto'
)
sim_inst.sys.non_bonded_inter[104, 104].lennard_jones.set_params(
    epsilon=5,
    sigma=1.5 / (2 ** (1/6)),
    cutoff=1.6,
    shift='auto'
)
sim_inst.sys.non_bonded_inter[105, 105].lennard_jones.set_params(
    epsilon=5,
    sigma=1.5 / (2 ** (1/6)),
    cutoff=1.6,
    shift='auto'
)
'''



#############################################################
#      set forces                                       #
##############################################################angle harmonic
print('armonico')
mesh =  {
    'assoc': {
        1: [2, 3, 6, 7, 8],
        5: [4, 9, 10, 13, 14],
        20: [11, 12, 15, 16, 21],
        24: [17, 18, 19, 22, 23]
    }
}

#angle_harmonic = espressomd.interactions.AngleHarmonic(bend=50.0, phi0=np.pi / 2)
angle_harmonic = espressomd.interactions.AngleCosine(bend=10.0, phi0=np.pi/2 )
sim_inst.sys.bonded_inter.add(angle_harmonic)
quartet_global_index=0
listt=[(0,24,11),(1,23,10),(2,22,10),(3,21,9)]
for fil_id, fil in enumerate(telomeres):
    #print('fil',fil_id)
    for quad_id, quad in enumerate(fil.associated_objects):
        #print('quad_id',quad_id)
        for quartet_id, quartet in enumerate(quad.associated_objects):
                #if quartet_id==0:
            #print('quartet_id',quartet_id)
            particles = particle_dict[fil_id][quad_id][quartet_id]
                    #bond del centrale con sopra e sotto
            #print([i.id for i in particles])
            
            if quartet_id==0:
                for i,j,k in listt:
                    #print(i)
                    particlesprec = particle_dict[fil_id][quad_id][quartet_id+1]
                    sotto=particlesprec[i]
                    sotto_ad= particlesprec[j]
                    #print(sotto.id)
                    particlessucc = particle_dict[fil_id][quad_id][quartet_id+2]
                    sopra=particlessucc[i]
                    #print(sopra.id)
                    corrente=particle_dict[fil_id][quad_id][quartet_id][i]
                    #print(corrente.id)
                    adiac=particle_dict[fil_id][quad_id][quartet_id]
                    
                    #corrente.add_bond((angle_harmonic, sopra,sotto))
                    corrente.add_bond((angle_harmonic, sotto,adiac[j]))
                    corrente.add_bond((angle_harmonic, sotto,adiac[k]))
                    corrente.add_bond((angle_harmonic, sopra,adiac[j]))
                    corrente.add_bond((angle_harmonic, sopra,adiac[k]))
                    sotto.add_bond((angle_harmonic, corrente,particlesprec[j]))
                    sotto.add_bond((angle_harmonic, corrente,particlesprec[k]))
                    sopra.add_bond((angle_harmonic, corrente,particlessucc[j]))
                    sopra.add_bond((angle_harmonic, corrente,particlessucc[k]))

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
    N = len(sim_inst.sys.part)

    T = (2 * E_kin) / (3 * N) if N > 0 else 0  # Temperatura in unità ridotte
    P = sim_inst.sys.analysis.pressure()  # Pressione
    P_total = P.get('total', None)  # Ottieni la pressione totale

    # Specifica il percorso del file
    file_path = f"/home/stekajack/UPLOAD_VIEW/{context_string}.txt"
    
     # Salva nel file
    with open(file_path, file_mode) as file:
        # Scrivi l'header se modalità "new"
        if scrivi_header:
            file.write("Timestep\tE_kinetic\tE_potential\tE_total\tTemperature\tPressure_total\n")

        # Scrivi i valori come riga tabellare
        file.write(f"{timestep}\t{E_kin:.6e}\t{E_pot:.6e}\t{E_tot:.6e}\t{T:.6e}\t{P_total:.6e}\n")

    print(f"[✓] Dati salvati (mode='{mode}') al timestep {timestep}")


data_pathh=f"/home/stekajack/UPLOAD_VIEW/{context_string}.lammpstrj"

def fuoir(sim_inst, step, box_dim, filename=data_pathh, mode="load"):
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
            radius = 0.25 if p.type in (100, 101, 103, 104, 105) else 0.5
            f.write(f"{p.id} {p.type} {x} {y} {z} {radius}\n")

#############################################################
#      Integration                                          #
#############################################################

checkpoint.save()
n_samples = 30000
 # Number of data saves
#steps_between_samples = 1e5  # 100,000 steps between each sample
steps_between_samples = 100
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
    salva_osservabili(timestep,mode='new')
    # All'inizio della simulazione
    fuoir(sim_inst, step=0, box_dim=box_dim, mode="new")

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
        fuoir(sim_inst, step=timestep, box_dim=box_dim, mode="load")
        #checkpoint.save()
        logging.info(f"💾 Checkpoint salvato dopo blocco {i + 1}")
        GLOBAL_COUNTER += 1
   
    logging.info("Equilibrazione completata.")
    # Esegui
    
 
    
    print('Equilibrazione appena completata. Inizio integrazione.')
    #GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[TelSeq], h5_data_path=h5_path, mode=args.MODE)
    timestep=round(sim_inst.sys.time/md_timestep)
    salva_osservabili(timestep,mode='new')
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
        salva_osservabili(timestep,mode='load')
        fuoir(sim_inst, step=timestep, box_dim=box_dim, mode="load")
        #checkpoint.save()
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


print('simulazione completata')

