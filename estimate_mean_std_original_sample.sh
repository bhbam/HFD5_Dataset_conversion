#!/bin/bash
#SBATCH --account=m4392
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J slurm_original_combine_mean_std
#SBATCH -t 08:05:00
#SBATCH --output=slurm_original_combine_mean_std_%J.out

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 128 --cpu_bind=cores python estimate_mean_std_original_sample.py
