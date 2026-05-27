#!/bin/bash
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=iscrc_gfold_0
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --mem=480000
#SBATCH --job-name=monomer_j1
#SBATCH --mail-user=giu.teodori1@stud.uniroma3.it
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/IscrC_GFOLD/GiuliaT/CG/sq/monomer_360K_2_4/slurm/monomer_j1_%j.out
#SBATCH --error=/leonardo_work/IscrC_GFOLD/GiuliaT/CG/sq/monomer_360K_2_4/slurm/monomer_j1_%j.err

WORKDIR=/leonardo_work/IscrC_GFOLD/GiuliaT/CG/sq/monomer_360K_2_4
SIF=/leonardo_work/IscrC_GFOLD/GiuliaT/CG/g4_twisted_nocat.sif
RANKFILE_DIR=${WORKDIR}/rankfiles

echo "=============================="
echo "JOB 1: T1_sim_1 T1_sim_2 T1_sim_3 T1_sim_4"
echo "Inizio: $(date)"
echo "=============================="

# Genera rankfile con i nodi reali assegnati da SLURM
chmod +x ${WORKDIR}/make_rankfiles.sh
bash ${WORKDIR}/make_rankfiles.sh \
    ${WORKDIR} \
    112 \
    28 \
    T1_sim_1 T1_sim_2 T1_sim_3 T1_sim_4

# Controlla checkpoint per ogni sim e prepara i flag
[ -d ${WORKDIR}/T1/sim_1/checkpoint ] && [ "$(ls -A ${WORKDIR}/T1/sim_1/checkpoint 2>/dev/null)" ] && FROM_1="--FROM_CHECKPOINT T1_sim_1" || FROM_1=""
[ -d ${WORKDIR}/T1/sim_2/checkpoint ] && [ "$(ls -A ${WORKDIR}/T1/sim_2/checkpoint 2>/dev/null)" ] && FROM_2="--FROM_CHECKPOINT T1_sim_2" || FROM_2=""
[ -d ${WORKDIR}/T1/sim_3/checkpoint ] && [ "$(ls -A ${WORKDIR}/T1/sim_3/checkpoint 2>/dev/null)" ] && FROM_3="--FROM_CHECKPOINT T1_sim_3" || FROM_3=""
[ -d ${WORKDIR}/T1/sim_4/checkpoint ] && [ "$(ls -A ${WORKDIR}/T1/sim_4/checkpoint 2>/dev/null)" ] && FROM_4="--FROM_CHECKPOINT T1_sim_4" || FROM_4=""

singularity exec \
    --bind /sys \
    --bind "${WORKDIR}:/home/stekajack/DATA" \
    ${SIF} bash -lc "
mpirun -np 28 \
    --bind-to core \
    --rankfile /home/stekajack/DATA/rankfiles/rankfile_T1_sim_1 \
    /home/stekajack/espresso/build/pypresso /home/stekajack/DATA/monomer.py \
        --RUN_TAG T1_sim_1 \
        ${FROM_1} \
        --KT 1.2 \
    &> /home/stekajack/DATA/T1/sim_1/run.txt &

mpirun -np 28 \
    --bind-to core \
    --rankfile /home/stekajack/DATA/rankfiles/rankfile_T1_sim_2 \
    /home/stekajack/espresso/build/pypresso /home/stekajack/DATA/monomer.py \
        --RUN_TAG T1_sim_2 \
        ${FROM_2} \
        --KT 1.2 \
    &> /home/stekajack/DATA/T1/sim_2/run.txt &

mpirun -np 28 \
    --bind-to core \
    --rankfile /home/stekajack/DATA/rankfiles/rankfile_T1_sim_3 \
    /home/stekajack/espresso/build/pypresso /home/stekajack/DATA/monomer.py \
        --RUN_TAG T1_sim_3 \
        ${FROM_3} \
        --KT 1.2 \
    &> /home/stekajack/DATA/T1/sim_3/run.txt &

mpirun -np 28 \
    --bind-to core \
    --rankfile /home/stekajack/DATA/rankfiles/rankfile_T1_sim_4 \
    /home/stekajack/espresso/build/pypresso /home/stekajack/DATA/monomer.py \
        --RUN_TAG T1_sim_4 \
        ${FROM_4} \
        --KT 1.2 \
    &> /home/stekajack/DATA/T1/sim_4/run.txt &

wait
"

echo "=============================="
echo "Fine: $(date)"
echo "=============================="

# Verifica completamento e resubmit automatico
echo ""
echo "--- STATO SIMULAZIONI ---"
NEEDS_RESUBMIT=0

for SIM in 1 2 3 4; do
    PCT=$(grep "Produzione: timestep" ${WORKDIR}/T1/sim_${SIM}/run.txt 2>/dev/null | tail -1 | grep -oP '\(\K[0-9.]+(?=%)' || echo "0")
    echo "T1_sim_${SIM}: ${PCT}% completato"
    if [ "$(echo "${PCT} < 99.9" | bc -l)" -eq 1 ]; then
        NEEDS_RESUBMIT=1
    fi
done

if [ $NEEDS_RESUBMIT -eq 1 ]; then
    echo ""
    echo "Alcune sim non hanno finito — resubmit automatico..."
    sbatch ${WORKDIR}/slurm/monomer_job1.slurm
else
    echo ""
    echo "Tutte le sim del job 1 completate!"
fi

