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
srun -n 1 -c 1 --cpu_bind=cores python combine_hdf5_files_with_suffling_data.py --input_data_path=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To18_pt30To300_with_unphysical_AToTau_decay_original_unbiased_train_h5 --output_data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_h5 --output_data_file=IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_train.h5 &
srun -n 1 -c 1 --cpu_bind=cores python combine_hdf5_files_with_suffling_data.py --input_data_path=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To18_pt30To300_with_unphysical_AToTau_decay_original_unbiased_valid_h5 --output_data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_h5 --output_data_file=IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_valid.h5 &
srun -n 1 -c 1 --cpu_bind=cores python combine_hdf5_files_with_suffling_data.py --input_data_path=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To18_pt30To300_with_unphysical_Tau_decay_original_unbiased_train_h5    --output_data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_original_unbiased_combined_h5    --output_data_file=IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_original_unbiased_combined_train.h5 &
srun -n 1 -c 1 --cpu_bind=cores python combine_hdf5_files_with_suffling_data.py --input_data_path=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To18_pt30To300_with_unphysical_Tau_decay_original_unbiased_valid_h5    --output_data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_original_unbiased_combined_h5    --output_data_file=IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_original_unbiased_combined_valid.h5 &

wait