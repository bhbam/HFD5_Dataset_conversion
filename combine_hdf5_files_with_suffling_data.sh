#!/bin/bash
#SBATCH --account=m4392
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J combine_h5_v2
#SBATCH -t 5:05:00
#SBATCH --output=slurm_combine_h5_%J.out
#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 1 --cpu_bind=cores python combine_hdf5_files_with_suffling_data.py --input_data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_m1p2To18_pt30T0300_unbiased_train_h5 --output_data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_m1p2To18_pt30T0300_original_combined_unbiased_h5 --output_data_file=IMG_aToTauTau_Hadronic_m1p2To3p6_pt30T0300_unbiased_original_combined_train.h5 &
srun -n 1 -c 1 --cpu_bind=cores python combine_hdf5_files_with_suffling_data.py --input_data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_m1p2To18_pt30T0300_unbiased_valid_h5 --output_data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_m1p2To18_pt30T0300_original_combined_unbiased_h5 --output_data_file=IMG_aToTauTau_Hadronic_m1p2To3p6_pt30T0300_unbiased_original_combined_valid.h5 &
# srun -n 1 -c 1 --cpu_bind=cores python combine_hdf5_files_with_suffling_data.py --input_data_path=/global/cfs/cdirs/m4392/bbbam/IMG_AToTau_Hadronic_massregssion_samples_m1p8To3p6_m3p6To18_pt30To300_original_train_h5   --output_data_path=/global/cfs/cdirs/m4392/bbbam/IMG_AToTau_Hadronic_massregssion_samples_m1p8To18_pt30To300_original_unbiased_combined_h5    --output_data_file=IMG_aToTauTau_Hadronic_m1p8To18_pt30T0300_original_unbiased_combined_train.h5 &
# srun -n 1 -c 1 --cpu_bind=cores python combine_hdf5_files_with_suffling_data.py --input_data_path=/global/cfs/cdirs/m4392/bbbam/IMG_AToTau_Hadronic_massregssion_samples_m1p8To3p6_m3p6To18_pt30To300_original_valid_h5   --output_data_path=/global/cfs/cdirs/m4392/bbbam/IMG_AToTau_Hadronic_massregssion_samples_m1p8To18_pt30To300_original_unbiased_combined_h5    --output_data_file=IMG_aToTauTau_Hadronic_m1p8To18_pt30T0300_original_unbiased_combined_valid.h5 &

wait