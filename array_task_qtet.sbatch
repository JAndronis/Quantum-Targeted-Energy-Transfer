#!/bin/bash

#SBATCH --job-name=qtet_trimer_data_generation
#SBATCH --output=sbatch_output/qtet_trimer_data_generation.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=sbatch_output/qtet_trimer_data_generation.%j.err # Stderr (%j expands to jobId)
#SBATCH --array=0-63%20
#SBATCH --time=05:00:00
#SBATCH --partition=serial
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ph4783@edu.physics.uoc.gr

# clean up old data
mv /home/ph4783/data/data_1* /home/ph4783/data/data_mpi_old

# create new data directory
DATA_PATH=/home/ph4783/data/data_${SLURM_JOBID}
mkdir ${DATA_PATH}

module purge
module load miniconda3/tf-2.9.1-cpu_py39
source activate qtet
srun python3 /home/ph4783/quantum_tet/qtet-repo/src/array_solver.py \
    --path ${DATA_PATH} --id $SLURM_ARRAY_TASK_ID --array-size $SLURM_ARRAY_TASK_COUNT
source deactivate
