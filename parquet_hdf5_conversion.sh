#!/bin/bash
#SBATCH --account=m4392
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J pq_h5_phy
#SBATCH --output=slurm_pq_h5_phy_%J.out
#SBATCH -t 22:10:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 10 --cpu_bind=cores python parquet_hdf5_conversion.py -i /pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300_unbiased -o /pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300_unbiased_h5
