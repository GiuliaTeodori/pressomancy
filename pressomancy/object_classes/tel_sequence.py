import espressomd
import numpy as np
import random
from pressomancy.object_classes.quadriplex_class import *
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams 
from pressomancy.helper_functions import RoutineWithArgs, make_centered_rand_orient_point_array, PartDictSafe, SinglePairDict, BondWrapper, get_orientation_vec, get_perpendicular, align_vectors
import logging
import warnings

TELSEQ_RULES = {
    'quartet': {
        'block_size': 75,
        'top': [26, 45, 49, 30],
        'bottom': [51, 70, 74, 55],
    },
    'quartet_11x11': {
        'block_size': 363,
        'top': [122, 231, 241, 132],
        'bottom': [243, 352, 362, 253],
    },
}


def rule_maker(fold_type, choice_id, offset, n=3):
    length = 4
    choice_local = choice_id - offset

    for rule_params in TELSEQ_RULES.values():
        top = rule_params['top']
        bottom = rule_params['bottom']
        if choice_local in bottom:
            i = bottom.index(choice_local)
            start_on_top = False
            break
        if choice_local in top:
            i = top.index(choice_local)
            start_on_top = True
            break
    else:
        raise ValueError(
            f"choice_id={choice_id} (local={choice_local}) is not a valid TelSeq corner id for offset={offset}"
        )

    diag_pairs = []
    across_pairs = []
    free_end = 0

    if fold_type == 'parallel':
        for _ in range(n):
            next_index = (i + 1) % length
            if start_on_top:
                diag_pairs.append((bottom[i] + offset, top[next_index] + offset))
                free_end = bottom[next_index] + offset
            else:
                diag_pairs.append((top[i] + offset, bottom[next_index] + offset))
                free_end = top[next_index] + offset
            i = next_index

    elif fold_type in ['hybrid', 'hybrid1']:
        idx1 = (i + 1) % length
        idx2 = (i + 2) % length
        idx3 = (i + 3) % length
        idxm1 = (i - 1) % length
        if start_on_top:
            diag_pairs.append((bottom[i] + offset, top[idx1] + offset))
            across_pairs.append((bottom[idx1] + offset, bottom[idx2] + offset))
            across_pairs.append((top[idx2] + offset, top[idx3] + offset))
            free_end = bottom[idxm1] + offset
        else:
            diag_pairs.append((top[i] + offset, bottom[idx1] + offset))
            across_pairs.append((top[idx1] + offset, top[idx2] + offset))
            across_pairs.append((bottom[idx2] + offset, bottom[idx3] + offset))
            free_end = top[idxm1] + offset

    elif fold_type == 'hybrid2':
        idx1 = (i - 1) % length
        idx2 = (i - 2) % length
        idx3 = (i - 3) % length
        idxm1 = (i + 1) % length
        if start_on_top:
            diag_pairs.append((bottom[i] + offset, top[idx1] + offset))
            across_pairs.append((bottom[idx1] + offset, bottom[idx2] + offset))
            across_pairs.append((top[idx2] + offset, top[idx3] + offset))
            free_end = bottom[idxm1] + offset
        else:
            diag_pairs.append((top[i] + offset, bottom[idx1] + offset))
            across_pairs.append((top[idx1] + offset, top[idx2] + offset))
            across_pairs.append((bottom[idx2] + offset, bottom[idx3] + offset))
            free_end = top[idxm1] + offset

    elif fold_type == 'antiparallel':
        idxm1 = (i - 1) % length
        idx1 = (i + 1) % length
        idx2 = (i + 2) % length
        if start_on_top:
            across_pairs.append((bottom[i] + offset, bottom[idxm1] + offset))
            diag_pairs.append((top[idxm1] + offset, top[idx1] + offset))
            across_pairs.append((bottom[idx1] + offset, bottom[idx2] + offset))
            free_end = top[idx2] + offset
        else:
            across_pairs.append((top[i] + offset, top[idxm1] + offset))
            diag_pairs.append((bottom[idxm1] + offset, bottom[idx1] + offset))
            across_pairs.append((top[idx1] + offset, top[idx2] + offset))
            free_end = bottom[idx2] + offset

    else:
        raise ValueError(f"unknown fold_type: {fold_type}")
    return diag_pairs, across_pairs, free_end

class TelSeq(metaclass=Simulation_Object):
    '''
    Class that contains TelSeq relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a TelSeq. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=['MORSE']	
    numInstances = 0
    simulation_type=SinglePairDict('tel_seq', 37)
    part_types = PartDictSafe({'real': 1, 'virt': 2,'to_be_magnetized':3})
    config = ObjectConfigParams(
        bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0)),
        diag_bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0)),
        across_bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0)),
        spacing=None,
        type='parallel',
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a TelSeq object requires the specification of particle size, number of parts and a handle to the espresso system
        '''
        self.sys=config['espresso_handle']
        assert config['type'] in ['parallel', 'antiparallel', 'hybrid', 'hybrid1', 'hybrid2'], 'type must be either parallel, antiparallel, hybrid, hybrid1 or hybrid2!!!'
        self.params=config
        if self.params['associated_objects']==None:
            warnings.warn('no associated_objects have been passed explicity. Creating objects required to initialise object implicitly!')
            configuration=Quartet.config.specify(espresso_handle=self.sys,type='brokenA')
            quartets=[Quartet(config=configuration) for _ in range(3*self.params['n_parts'])]
            grouped_quartets = [quartets[i:i+3]
                    for i in range(0, len(quartets), 3)]
            quadriplex_config_list = [Quadriplex.config.specify(associated_objects=elem, espresso_handle=self.sys) for elem in grouped_quartets]
            self.params['associated_objects']= [Quadriplex(config=elem) for elem in quadriplex_config_list]
        self.associated_objects=self.params['associated_objects']

        self.build_function=RoutineWithArgs(func=make_centered_rand_orient_point_array,num_monomers=self.params['n_parts'],spacing=config['spacing'])  
        self.who_am_i = TelSeq.numInstances
        TelSeq.numInstances += 1
        self.orientor = np.empty(shape=3, dtype=float)
        self.type_part_dict=PartDictSafe({key: [] for key in TelSeq.part_types.keys()})

    def _choose_antiparallel_phi(self, chain_dir, n_phi=720):
        chain_dir = np.asarray(chain_dir, dtype=float)
        chain_dir /= np.linalg.norm(chain_dir)

        z_axis = np.array([0.0, 0.0, 1.0])
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])

        best_phi = 0.0
        best_score = -np.inf
        for idx in range(n_phi):
            phi = 2.0 * np.pi * idx / n_phi
            side_axis = get_perpendicular(chain_dir, phi=phi)
            rotation_matrix = align_vectors(z_axis, side_axis)
            x_world = rotation_matrix @ x_axis
            y_world = rotation_matrix @ y_axis
            score = max(np.abs(np.dot(x_world, chain_dir)), np.abs(np.dot(y_world, chain_dir)))
            if score > best_score + 1e-12:
                best_score = score
                best_phi = phi
        return best_phi

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of TelSeq stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        pos=np.atleast_2d(pos)
        assert len(
            pos) == self.params['n_parts'], 'there is a missmatch between the pos lenth and TelSeq n_parts'
        self.orientor = get_orientation_vec(pos)

        assert self.params['n_parts'] == len(
            self.associated_objects), " there doest seem to be enough monomers stored!!! "
        assert all([x.simulation_type==self.associated_objects[0].simulation_type for x in self.associated_objects[1:]]), 'all objects must have the same simulation type!'
        local_orientor = self.orientor
        if self.params['type'] == 'antiparallel':
            phi_opt = self._choose_antiparallel_phi(self.orientor)
            local_orientor = get_perpendicular(self.orientor, phi=phi_opt)
        for obj_el, pos_el in zip(self.associated_objects, pos):
            _=obj_el.set_object(pos_el, local_orientor)
        return self

    def wrap_into_Tel(self):
        '''
        associated_objects contains monomer objects (assume quadriplex). We add cormer particles in each quadriplex pair to a pool of candidate corners: candidate1 and candidate2. Finaly checks which corner pairs have a distance self.params['sigma']-2*fene_r0. Relies on np.isclose().
        :return: None

        '''
        pos_first = self.associated_objects[0].associated_objects[0].unperturbed_particles[0].pos
        pos_last = self.associated_objects[-1].associated_objects[0].unperturbed_particles[0].pos
        v_chain = pos_last - pos_first
        dot_prod = np.dot(v_chain, self.orientor)
        
        order = list(range(len(self.associated_objects)))
        if self.params['type'] == 'hybrid1':
            if dot_prod > 0:
                order = order[::-1]
        elif self.params['type'] == 'hybrid2':
            if dot_prod < 0:
                order = order[::-1]
            
        for loop_idx, iid in enumerate(order):
            monomer = self.associated_objects[iid]
            ########################
            #METTIAMO LE ESCLUSIONI
            circ_parts = []
            for q in monomer.associated_objects:  # tutti i quartet del monomero
                for p in q.unperturbed_particles:  # tutti i particles, non solo corner
                    if p.type == q.part_types['circ']:
                        circ_parts.append(p)
            # applica exclusion tra tutte le circ del monomero
            for i, p1 in enumerate(circ_parts):
                for p2 in circ_parts[i+1:]:
                    print(f"Applying exclusion between circ particles {p1.id} and {p2.id} in monomer {monomer.who_am_i}")
                    p1.add_exclusion(p2)

            #########################
            candidates1 = []
            if self.params['type'] == 'hybrid1':
                candidates1.extend(monomer.associated_objects[1].corner_particles)
            elif self.params['type'] == 'hybrid2':
                candidates1.extend(monomer.associated_objects[2].corner_particles)
            else:
                candidates1.extend(monomer.associated_objects[1].corner_particles)
                candidates1.extend(monomer.associated_objects[2].corner_particles)
            if loop_idx == 0:
                start_part_id = random.choice(candidates1).id
                self.start_part_id = start_part_id
            alias = monomer.associated_objects[0].params['alias']
            offset = monomer.who_am_i * TELSEQ_RULES[alias]['block_size']
            logging.debug(
                "wrap_into_Tel step=%s monomer_id=%s start_part_id=%s offset=%s",
                iid,
                monomer.who_am_i,
                start_part_id,
                offset,
            )
            diag_pairs, across_pairs, free_end = rule_maker(
                self.params['type'], start_part_id, offset
            )
            logging.debug(
                "rule_result fold_type=%s choice_local=%s diag_pairs=%s across_pairs=%s free_end=%s",
                self.params['type'],
                start_part_id - offset,
                diag_pairs,
                across_pairs,
                free_end,
            )
            for id1, id2 in diag_pairs:
                self.bond_owned_part_pair(self.sys.part.by_id(id1), self.sys.part.by_id(id2), bond_handle=self.params['diag_bond_handle'])
            for id1, id2 in across_pairs:
                self.bond_owned_part_pair(self.sys.part.by_id(id1), self.sys.part.by_id(id2), bond_handle=self.params['across_bond_handle'])

            candidates2 = []

            try:
                next_iid = order[loop_idx + 1]
                next_monomer = self.associated_objects[next_iid]
                candidates2.extend(
                    next_monomer.associated_objects[1].corner_particles)
                candidates2.extend(
                    next_monomer.associated_objects[2].corner_particles)
                candidate_pos = np.array([x.pos for x in candidates2])

                pair_distances_free = np.linalg.norm(
                    candidate_pos - self.sys.part.by_id(free_end).pos, axis=-1
                )
                pair_distances_start = np.linalg.norm(
                    candidate_pos - self.sys.part.by_id(start_part_id).pos, axis=-1
                )
                combined_distances = np.column_stack((pair_distances_free, pair_distances_start))
                index, ref_index = np.unravel_index(np.argmin(combined_distances), combined_distances.shape)
                ref_part_id = free_end if ref_index == 0 else start_part_id
                logging.debug(
                    "handoff_choice next_monomer_id=%s index=%s ref=%s min_dist=%s",
                    next_monomer.who_am_i,
                    index,
                    "free_end" if ref_index == 0 else "start_part_id",
                    combined_distances[index, ref_index],
                )
                self.bond_owned_part_pair(candidates2[index], self.sys.part.by_id(ref_part_id))

                start_part_id = candidates2[index].id
                logging.debug("new_start_part_id=%s", start_part_id)
            except IndexError:
                logging.info('end of chain reached')
                continue
    def get_full_chain(self):
        # Definiamo i tipi di particella che compongono la catena portante (corner e loop)
        # Tipo 1: corner reali del G-quadruplex
        # Tipi 103, 104, 105: sfere reali inserite al posto dei vecchi legami FENE
        target_types = {1, 103, 104, 105}
        
        # Recuperiamo tutte le particelle del filamento corrente di questi tipi
        owned_parts, _ = self.get_owned_part()
        owned_parts = [p for p in owned_parts if p.type in target_types]
        owned_ids = {p.id for p in owned_parts}
        
        # Costruiamo la lista delle adiacenze (chi è legato a chi tramite FENE)
        adj = {pid: set() for pid in owned_ids}
        for p in owned_parts:
            for bond in p.bonds:
                partner_id = bond[1]
                if partner_id in owned_ids:
                    adj[p.id].add(partner_id)
                    adj[partner_id].add(p.id)
                    
        # Identifichiamo i capi del filamento (le particelle che hanno solo 1 legame attivo)
        endpoints = [pid for pid, neighbors in adj.items() if len(neighbors) == 1]
        
        if not endpoints:
            # Fallback di sicurezza: se la catena è chiusa o non trova capi, prende un punto qualsiasi con vicini
            endpoints = [pid for pid, neighbors in adj.items() if len(neighbors) > 0]
            if not endpoints:
                return []
                
        # Ricostruiamo la catena camminando lungo i legami a partire dal capo che si trova sul piano Top
        start = endpoints[0]
        alias = self.associated_objects[0].associated_objects[0].params['alias']
        block_size = TELSEQ_RULES[alias]['block_size']
        top_list = TELSEQ_RULES[alias]['top']
        
        for ep in endpoints:
            local_id = ep % block_size
            if local_id in top_list:
                start = ep
                break
        chain = [start]
        visited = {start}
        
        curr = start
        while True:
            next_nodes = adj[curr] - visited
            if not next_nodes:
                break
            # Essendo una catena lineare, ogni nodo interno ha al massimo un vicino non visitato
            nxt = list(next_nodes)[0]
            chain.append(nxt)
            visited.add(nxt)
            curr = nxt
            
        # Ritorniamo la lista ordinata di oggetti ParticleHandle di ESPResSo
        return [self.sys.part.by_id(pid) for pid in chain]