#!/bin/bash
#SBATCH --account=m4392
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J slurm_mea_after_outlier
#SBATCH -t 00:30:00
#SBATCH --output=slurm_combine_original_%J.out
#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 256 --cpu_bind=cores python estimate_mean_std_size_after_outlier.py
