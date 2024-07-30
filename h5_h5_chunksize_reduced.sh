#!/bin/bash
#SBATCH --account=m4392
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J slurm_reduce_chunk_32
#SBATCH -t 06:30:00
#SBATCH --output=slurm_combine_original_%J.out
#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 2 --cpu_bind=cores python h5_h5_chunksize_reduced.py
