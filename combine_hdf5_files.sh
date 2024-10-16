#!/bin/bash
#SBATCH --account=m4392
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J slurm_combine_original
#SBATCH -t 03:05:00
#SBATCH --output=slurm_combine_original_%J.out
#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 64 --cpu_bind=cores python combine_hdf5_files.py --input_data_path=/pscratch/sd/b/bbbam/IMG_m3p6To14p8_normalized_nan_replaced_train_h5 --output_data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_2_unbaised_v2_normalised_nan_replaced_combined.h5 #&
# srun -n 1 -c 64 --cpu_bind=cores python combine_hdf5_files.py --input_data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_valid_h5 --output_data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_normalised_nan_replaced_combined.h5 &
# wait
