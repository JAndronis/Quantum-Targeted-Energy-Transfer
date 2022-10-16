#!/bin/bash

#SBATCH --job-name=qtet_trimer_data_generation
#SBATCH --output=sbatch_output/qtet_trimer_data_generation.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=sbatch_output/qtet_trimer_data_generation.%j.err # Stderr (%j expands to jobId)
#SBATCH --nodes=1
#SBATCH --array=0-63%25
#SBATCH --mem=6G
#SBATCH --time=05:00:00
#SBATCH --partition=serial
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ph4783@edu.physics.uoc.gr

module purge
module load miniconda3/tf-2.9.1-cpu_py39

srun python3 /home/ph4783/quantum_tet/qtet-repo/src/tet/array_solver.py \
    --path /home/ph4783/data --id $SLURM_ARRAY_TASK_ID --array-size $SLURM_ARRAY_TASK_COUNT
