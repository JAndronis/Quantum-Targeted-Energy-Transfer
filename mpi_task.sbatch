#!/bin/bash

#SBATCH --job-name=qtet_trimer_data_generation_parallel
#SBATCH --output=sbatch_output/mpi_jobs/qtet_trimer_data_generation.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=sbatch_output/mpi_jobs/qtet_trimer_data_generation.%j.err # Stderr (%j expands to jobId)
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --partition=serial
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ph4783@edu.physics.uoc.gr

# clean up old data
#mv /home/ph4783/data/data_1* /home/ph4783/data/data_mpi_old

# create new data directory
DATA_PATH=/home/ph4783/data/data_${SLURM_JOBID}
#mkdir ${DATA_PATH}

CONSTANTS_PATH=/home/ph4783/quantum_tet/qtet-repo/src/constants.json

module purge
module load miniconda3/tf-2.9.1-cpu_py39
source activate qtet
srun --cpus-per-task=40 python3 /home/ph4783/quantum_tet/qtet-repo/src/mp_solver.py
source deactivate
