#!/bin/bash
#SBATCH --account=m4392
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J pq_h5_phy
#SBATCH --output=slurm_pq_h5_phy_%J.out
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
# srun -n 1 -c 11 --cpu_bind=cores python parquet_hdf5_conversion.py -i /global/cfs/cdirs/m4392/bbbam/classifier_signal_background_Run2_parquet/signal -o /global/cfs/cdirs/m4392/bbbam/classifier_signal_background_Run2_hdf5/signal &
srun -n 1 -c 5 --cpu_bind=cores python parquet_hdf5_conversion.py -i /global/cfs/cdirs/m4392/bbbam/classifier_signal_background_Run2_valid_parquet -o /global/cfs/cdirs/m4392/bbbam/classifier_signal_background_Run2_valid_h5 &
wait
echo "All done"
