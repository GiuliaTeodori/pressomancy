#!/bin/bash

NUMBER_OF_CORES=32
RANKS_PER_PROCESS=16
PARALLEL_PARAMS="3 5"
WHO_AM_I=quadriplex_project/ligands_nobending
PATH_TO_DATA=$WORK/deniz/DATA/${WHO_AM_I}

chmod +x "$HOME/submission_scripts/make_rankfiles.sh"
$HOME/submission_scripts/make_rankfiles.sh $PATH_TO_DATA $NUMBER_OF_CORES $RANKS_PER_PROCESS $PARALLEL_PARAMS || { echo "make_rankfiles.sh failed" >&2; exit 1; }

for no_obj in 6120 ; do
for concentration in 0.6 ; do
for vdw in 5 ; do
for vdw_ligand in 5 10 15 20 ; do
for part_per_filament in 15 ; do
for no_crwd in 4080 ; do
for sim_dir in sim_1 sim_2 sim_3 sim_4 sim_5 sim_6 sim_7 sim_8 ; do

cat > ${PATH_TO_DATA}/placeholder_name_${vdw_ligand}_${sim_dir}.slurm<<EOF
#!/bin/bash

#SBATCH -p boost_usr_prod
#SBATCH -N 1                       # 1 node
#SBATCH --ntasks-per-node=$NUMBER_OF_CORES  # Total number of tasks
#SBATCH --gres=gpu:0               # Request 0 GPUs (update if needed)
#SBATCH --mem=64000                # Memory per node in MB
#SBATCH --job-name=quadriplex_job
#SBATCH --mail-user=avantardejack@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time 24:00:00              # Format: HH:MM:SS

EOF
cat >>  ${PATH_TO_DATA}/placeholder_name_${vdw_ligand}_${sim_dir}.slurm<<EOF
singularity exec --bind $WORK/deniz/DATA:$HOME/DATA_VIEW $HOME/preggomancy_runtime.sif bash -lc "
EOF

for pattern in $PARALLEL_PARAMS ; do

cat >>  ${PATH_TO_DATA}/placeholder_name_${vdw_ligand}_${sim_dir}.slurm<<EOF

mpirun -np $RANKS_PER_PROCESS --report-bindings --bind-to core --rankfile $HOME/DATA_VIEW/${WHO_AM_I}/rankfiles/rankfile_${pattern} /home/stekajack/espresso/build/pypresso $HOME/upload/qudriplex_project/poly_BRACO.py -no_obj $no_obj -concentration $concentration -no_per 25 -no_crowders $no_crwd -vdW $vdw -vdW_ligand $vdw_ligand -part_per_filament $part_per_filament -part_per_ligand $pattern -path_data $HOME/DATA_VIEW/${WHO_AM_I}/${sim_dir} -MODE LOAD -bonding_mode ftf &> $HOME/DATA_VIEW/${WHO_AM_I}/quadriplex_${sim_dir}_${no_obj}_${concentration}_${no_crwd}_${vdw}_${vdw_ligand}_${part_per_filament}_${pattern}.txt &
EOF
done

cat >>  ${PATH_TO_DATA}/placeholder_name_${vdw_ligand}_${sim_dir}.slurm<<EOF

wait"
EOF
# sbatch ${PATH_TO_DATA}/placeholder_name_${vdw_ligand}_${sim_dir}.slurm
# rm ${PATH_TO_DATA}/placeholder_name_${vdw_ligand}_${sim_dir}.slurm
done
done
done
done
done
done
done