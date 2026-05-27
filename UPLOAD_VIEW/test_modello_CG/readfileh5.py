def inscribe_part_group_to_h5(self, group_type=None, h5_data_path=None,mode='NEW',force_resize_to_size=None):
        """
        Inscribe one or more groups of simulation objects into an HDF5 file.

        This method creates (or opens) an HDF5 file and, for each `group_type`:
        - Builds a flat list of particle handles and their coordinating indices
        - Creates `/particles/<GroupName>` and corresponding property datasets
        - Creates `/connectivity/<GroupName>/ParticleHandle_to_<OwnerClass>` tables
        - Creates `/connectivity/<GroupName>/<Left>_to_<Right>` object–object tables

        Parameters
        ----------
        group_type : list of type
            A list of `SimulationObject` subclasses. All instances of each
            class in `self.objects` will be registered and inscribed.
        h5_data_path : str
            Path to the HDF5 file to write (mode='NEW') or append (mode='LOAD').
        mode : {'NEW', 'LOAD'}, optional
            - 'NEW' : create a fresh file structure (default).
            - 'LOAD': open existing file and resume writing.  

        Returns
        -------
        int
            The starting global counter for writing time steps. Always 0 in
            'NEW' mode; in 'LOAD' mode, the current number of already‑saved steps.

        Raises
        ------
        ValueError
            If `mode` is not one of 'NEW' or 'LOAD'.
        ValueError
            If `group_type` is not a list.
        ValueError
            In 'LOAD' mode, if different groups have mismatched saved step counts.
        """
        if not isinstance(group_type, list):
            raise ValueError("group_type must be a list of classes.")
        if mode not in ('NEW', 'LOAD', 'LOAD_NEW', 'INIT_SRC'):
            raise ValueError(f"Unknown mode: {mode}")
        if force_resize_to_size is not None:
            assert mode=='LOAD_NEW', 'force_resize_to_size can only be used in LOAD_NEW mode'
        self.io_dict['registered_group_type']=[grp_typ.__name__ for grp_typ in group_type]

        if mode in ['NEW', 'INIT_SRC']:
            self.io_dict['h5_file'] = h5py.File(h5_data_path, "w")
            par_grp = self.io_dict['h5_file'].require_group(f"particles")
            for grp_typ in group_type:
                data_grp = par_grp.require_group(grp_typ.__name__)
                box_grp = data_grp.require_group("box")
                box_grp.attrs["dimension"] = int(len(self.sys.box_l))
                box_grp.attrs["boundary"] = np.array(
                    ["periodic" if flag else "none" for flag in self.sys.periodicity],
                    dtype=h5py.string_dtype(encoding="ascii"),
                )
                if "edges" in box_grp:
                    del box_grp["edges"]
                box_grp.create_dataset(
                    "edges",
                    data=np.asarray(self.sys.box_l, dtype=np.float64),
                    dtype=np.float64,
                )
                connect_grp = self.io_dict['h5_file'].require_group(f"connectivity").require_group(grp_typ.__name__)
                logging.info(f"Inscribe: Creating group {grp_typ.__name__} in HDF5 file.")
                objects_to_register=[obj for obj in self.objects if isinstance(obj,grp_typ)]
            
                coordination_indices=[]
                for cr in objects_to_register:
                    part,coord=cr.get_owned_part()
                    self.io_dict['flat_part_view'][grp_typ.__name__].extend(part)
                    coordination_indices.extend(coord)

                total_part_num=len(self.io_dict['flat_part_view'][grp_typ.__name__])

                # Create the connectivity for ParticleHandle to objects that own them.
                grouped = defaultdict(list)
                for part, coords in zip(self.io_dict['flat_part_view'][grp_typ.__name__], coordination_indices):
                    for cls_name, idx in coords:
                        grouped[cls_name].append((part.id, idx))

                for cls_name in sorted(grouped):
                    arr = np.array(grouped[cls_name], dtype=np.int32)
                    connect_grp.create_dataset(
                        f"ParticleHandle_to_{cls_name}",
                        data=arr,
                        dtype=np.int32,
                        maxshape=(arr.shape)
                    )
                # Create the connectivity for objects that own each other
                pair_buckets = defaultdict(list)
                
                for obj in self.collect_instances_recursively(objects_to_register):
                    if not obj.associated_objects:
                        continue
                    left_name = obj.__class__.__name__
                    for sub in obj.associated_objects:
                        right_name = sub.__class__.__name__
                        pair_buckets[(left_name, right_name)].append((obj.who_am_i, sub.who_am_i))

                for (left_name, right_name) in sorted(pair_buckets):
                    arr = np.array(pair_buckets[(left_name, right_name)], dtype=np.int32)
                    ds = connect_grp.create_dataset(
                        f"{left_name}_to_{right_name}",
                        data=arr,
                        dtype=np.int32,
                        maxshape=(arr.shape)
                    )
                # Create the datasets for each property           
                for prop,dim in self.io_dict['properties']:
                    prop_group = data_grp.require_group(prop)
                    prop_group.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int32)
                    prop_group.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64)
                    prop_group.create_dataset(
                        "value",
                        shape=(0, total_part_num, dim),  # Store all particles in a single dataset
                        maxshape=(None, total_part_num, dim),
                        dtype=np.float32,
                        chunks=(1, total_part_num, dim),
                        compression="gzip",
                        compression_opts=4
                    )
            GLOBAL_COUNTER=0

        elif mode=='LOAD_NEW':

            self.io_dict['h5_file'] = h5py.File(h5_data_path, "a")
            particles_group = self.io_dict['h5_file']["particles"]
            candidate_lens=[]
            for grp_typ in group_type:
                data_view=H5DataSelector(self.io_dict['h5_file'], particle_group=grp_typ.__name__)
                ids=data_view.get_connectivity_values(grp_typ.__name__)
                part_ids=[]
                for iid in ids:
                    temp=data_view.select_particles_by_object(object_name=grp_typ.__name__,connectivity_value=iid)
                    part_ids+=temp.timestep[-1].id.flatten().tolist()
                part_ids=[int(x) for x in part_ids]
                self.io_dict['flat_part_view'][grp_typ.__name__].extend(self.sys.part.by_ids(part_ids))
                data_grp = particles_group[grp_typ.__name__]
                dataset_val = data_grp["pos/value"]
                candidate_lens.append(dataset_val.shape[0])
            if len(set(candidate_lens)) != 1:
                raise ValueError(
                    f"Inconsistent step counts across groups: {candidate_lens}"
                )
            GLOBAL_COUNTER=candidate_lens[0]

            if force_resize_to_size is not None:
                assert type(force_resize_to_size) is int, 'force_resize_to_size must be an integer'
                assert force_resize_to_size<=GLOBAL_COUNTER, 'force_resize_to_size must be smaller than or equal to the current number of timesteps saved in file'
                if force_resize_to_size==GLOBAL_COUNTER:
                    logging.info(f'force_resize_to_size is equal to the current number of timesteps saved in file. No resizing will be done.')
                else:
                    for grp_typ in group_type:
                        data_grp = particles_group[grp_typ.__name__]
                        for prop,_ in self.io_dict['properties']:
                            dataset_val = data_grp[f"{prop}/value"]
                            step_dataset = data_grp[f"{prop}/step"]
                            time_dataset = data_grp[f"{prop}/time"]
                            step_dataset.resize((force_resize_to_size,))
                            time_dataset.resize((force_resize_to_size,))
                            dataset_val.resize((force_resize_to_size, dataset_val.shape[1], dataset_val.shape[2]))
                    self.io_dict['h5_file'].flush()
                    logging.info(f'Force resized all datasets from {GLOBAL_COUNTER} to size {force_resize_to_size}')
                    GLOBAL_COUNTER=force_resize_to_size
            logging.info(f"Loaded h5 file with GLOBAL_COUNTER={GLOBAL_COUNTER} ")
            return GLOBAL_COUNTER
        
        elif mode=='LOAD':
            self.io_dict['h5_file'] = h5py.File(h5_data_path, "a")
            particles_group = self.io_dict['h5_file']["particles"]
            candidate_lens=[]
            for grp_typ in group_type:
                objects_to_register=[obj for obj in self.objects if isinstance(obj,grp_typ)]
                for cr in objects_to_register:
                    part,_=cr.get_owned_part()
                    self.io_dict['flat_part_view'][grp_typ.__name__].extend(part)
                data_grp = particles_group[grp_typ.__name__]
                dataset_val = data_grp["pos/value"]
                candidate_lens.append(dataset_val.shape[0])
            if len(set(candidate_lens)) != 1:
                raise ValueError(
                    f"Inconsistent step counts across groups: {candidate_lens}"
                )
            GLOBAL_COUNTER=candidate_lens[0]
            logging.info(f"Loading h5 file with GLOBAL_COUNTER={GLOBAL_COUNTER} ")

        return GLOBAL_COUNTER
        
    def write_part_group_to_h5(self, time_step=None):
        """Append one frame using an integer frame counter and current ESPResSo time."""
        assert self.io_dict['h5_file']!=None,'storage file has not been inscribed!'
        if not isinstance(time_step, Integral):
            raise TypeError("time_step must be provided as an integer frame counter.")
        physical_time = float(self.sys.time)
        for grp_typ in self.io_dict['registered_group_type']:
            particles_group = self.io_dict['h5_file']["particles"]
            data_grp = particles_group[grp_typ]
            for prop,_ in self.io_dict['properties']:
                dataset_val = data_grp[f"{prop}/value"]
                step_dataset = data_grp[f"{prop}/step"]
                time_dataset = data_grp[f"{prop}/time"]
                step_dataset.resize((dataset_val.shape[0] + 1,))
                time_dataset.resize((dataset_val.shape[0] + 1,))
                dataset_val.resize((dataset_val.shape[0] + 1, dataset_val.shape[1], dataset_val.shape[2]))
                step_dataset[-1] = time_step
                time_dataset[-1] = physical_time
                dataset_val[-1, :, :] = np.array([np.atleast_1d(getattr(part, prop)) for part in self.io_dict['flat_part_view'][grp_typ]], dtype=np.float32)

        logging.debug(f"Successfully wrote timestep for {self.io_dict['registered_group_type']}.")

    def inscribe_observable_group_to_h5(self, observable_defs=None, h5_data_path=None, mode='NEW', force_resize_to_size=None):
        """
        Create or reopen observables under ``/observables/<name>/{step,time,value}``.
        """
        if not isinstance(observable_defs, list) or not observable_defs:
            raise ValueError("observable_defs must be a non-empty list.")
        if mode not in ('NEW', 'LOAD', 'LOAD_NEW', 'INIT_SRC'):
            raise ValueError(f"Unknown mode: {mode}")
        if force_resize_to_size is not None:
            assert mode == 'LOAD_NEW', 'force_resize_to_size can only be used in LOAD_NEW mode'

        normalised_defs = []
        for obs_def in observable_defs:
            if len(obs_def) != 4:
                raise ValueError("Each observable definition must be (name, shape, dtype, observable_value_ref).")
            name, shape, dtype, observable_value_ref = obs_def
            if shape is None:
                shape = tuple()
            elif isinstance(shape, Integral):
                shape = (int(shape),)
            else:
                shape = tuple(shape)
            normalised_defs.append((str(name), shape, np.dtype(dtype), observable_value_ref))

        self.io_dict['registered_observables'] = {
            name: {'shape': shape, 'dtype': dtype, 'value': observable_value_ref}
            for name, shape, dtype, observable_value_ref in normalised_defs
        }

        if self.io_dict['h5_file'] is None:
            if h5_data_path is None:
                raise ValueError("h5_data_path must be provided when no HDF5 file is currently open.")
            file_mode = "w" if mode in ('NEW', 'INIT_SRC') else "a"
            self.io_dict['h5_file'] = h5py.File(h5_data_path, file_mode)

        observables_group = self.io_dict['h5_file'].require_group("observables")

        if mode in ('NEW', 'INIT_SRC'):
            for name, shape, dtype, _ in normalised_defs:
                obs_group = observables_group.require_group(name)
                if any(key in obs_group for key in ('step', 'time', 'value')):
                    raise ValueError(f"Observable '{name}' already exists in HDF5 file.")
                obs_group.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int32)
                obs_group.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64)
                obs_group.create_dataset(
                    "value",
                    shape=(0, *shape),
                    maxshape=(None, *shape),
                    dtype=dtype,
                    chunks=(1, *shape) if shape else (1,),
                    compression="gzip",
                    compression_opts=4,
                )
            return 0

        candidate_lens = []
        for name, shape, dtype, _ in normalised_defs:
            obs_group = observables_group.get(name)
            if obs_group is None:
                raise ValueError(f"Observable '{name}' was not found in HDF5 file during {mode}.")
            value_dataset = obs_group["value"]
            if tuple(value_dataset.shape[1:]) != shape:
                raise ValueError(
                    f"Observable '{name}' shape mismatch: file has {value_dataset.shape[1:]}, expected {shape}."
                )
            candidate_lens.append(value_dataset.shape[0])

        if len(set(candidate_lens)) != 1:
            raise ValueError(f"Inconsistent step counts across observables: {candidate_lens}")

        GLOBAL_COUNTER = candidate_lens[0]
        if force_resize_to_size is not None:
            assert type(force_resize_to_size) is int, 'force_resize_to_size must be an integer'
            assert force_resize_to_size <= GLOBAL_COUNTER, 'force_resize_to_size must be smaller than or equal to the current number of timesteps saved in file'
            if force_resize_to_size == GLOBAL_COUNTER:
                logging.info('force_resize_to_size is equal to the current number of timesteps saved in file. No resizing will be done.')
            else:
                for name, _, _, _ in normalised_defs:
                    obs_group = observables_group[name]
                    step_dataset = obs_group["step"]
                    time_dataset = obs_group["time"]
                    value_dataset = obs_group["value"]
                    step_dataset.resize((force_resize_to_size,))
                    time_dataset.resize((force_resize_to_size,))
                    value_dataset.resize((force_resize_to_size, *value_dataset.shape[1:]))
                self.io_dict['h5_file'].flush()
                logging.info(f'Force resized all observables from {GLOBAL_COUNTER} to size {force_resize_to_size}')
                GLOBAL_COUNTER = force_resize_to_size
        logging.info(f"Loaded h5 file with GLOBAL_COUNTER={GLOBAL_COUNTER} ")
        return GLOBAL_COUNTER

    def write_observable_group_to_h5(self, time_step=None):
        """Append one frame using an integer frame counter and current ESPResSo time."""
        assert self.io_dict['h5_file'] != None, 'storage file has not been inscribed!'
        if not isinstance(time_step, Integral):
            raise TypeError("time_step must be provided as an integer frame counter.")

        registered_observables = self.io_dict['registered_observables']
        if not registered_observables:
            raise ValueError("No observables have been inscribed in HDF5.")

        physical_time = float(self.sys.time)
        observables_group = self.io_dict['h5_file']["observables"]
        for name, obs_data in registered_observables.items():
            obs_group = observables_group[name]
            payload = obs_data['value']
            value_dataset = obs_group["value"]
            expected_shape = tuple(value_dataset.shape[1:])
            if hasattr(payload, 'shape'):
                payload_shape = tuple(payload.shape)
            else:
                payload_shape = tuple()
            if payload_shape != expected_shape:
                raise ValueError(
                    f"Observable '{name}' shape mismatch: payload has {payload_shape}, dataset expects {expected_shape}."
                )
            step_dataset = obs_group["step"]
            time_dataset = obs_group["time"]
            step_dataset.resize((value_dataset.shape[0] + 1,))
            time_dataset.resize((value_dataset.shape[0] + 1,))
            value_dataset.resize((value_dataset.shape[0] + 1, *value_dataset.shape[1:]))
            step_dataset[-1] = time_step
            time_dataset[-1] = physical_time
            value_dataset[-1] = payload

        logging.debug(f"Successfully wrote timestep for {list(registered_observables)}.")

    def write_registered_to_h5(self, time_step=None):
        """Append one synchronized frame for all currently inscribed HDF5 streams."""
        assert self.io_dict['h5_file'] is not None, 'storage file has not been inscribed!'
        if not isinstance(time_step, Integral):
            raise TypeError("time_step must be provided as an integer frame counter.")

        registered_groups = self.io_dict['registered_group_type'] or []
        registered_observables = self.io_dict['registered_observables']
        if not registered_groups and not registered_observables:
            raise ValueError("No particle groups or observables have been inscribed in HDF5.")

        if registered_groups:
            self.write_part_group_to_h5(time_step=time_step)
        if registered_observables:
            self.write_observable_group_to_h5(time_step=time_step)
