#!/bin/bash

#SBATCH --job-name=qtet_trimer_data_generation
#SBATCH --output=trimer_data.out
#SBATCH --error=trimer_data.err
#SBATCH --ntasks=32
#SBATCH --mem=4G
#SBATCH --partition=parallel
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ph4783@edu.physics.uoc.gr

module purge
module load miniconda3/tf-2.9.1-cpu_py39

srun python3 /home/ph4783/quantum_tet/qtet-repo/src/tet/mpi_solver.py /home/ph4783/data
