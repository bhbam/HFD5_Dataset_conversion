#!/bin/bash
#SBATCH --account=m4392
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J unbiased_h5
#SBATCH -t 10:30:00
#SBATCH --output=slurm_unbaised_h5_%J.out
#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 1 --cpu_bind=cores python h5_to_unbiased_h5.py --bin_count=1800 --input_file=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To3p6_pt30To300_with_unphysical_AToTau_decay_original_train_h5/IMG_AToTau_Hadronic_decay_massregssion_samples_m1p8To3p6_pt30To300_combined_original_train.h5 --output_data_path=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To18_pt30To300_with_unphysical_AToTau_decay_original_unbiased_train_h5 --output_file=IMG_massregssion_samples_m0To3p6_pt30To300_with_unphysical_AToTau_decay_original_unbiased_train.h5 &
srun -n 1 -c 1 --cpu_bind=cores python h5_to_unbiased_h5.py --bin_count=200 --input_file=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To3p6_pt30To300_with_unphysical_AToTau_decay_original_valid_h5/IMG_AToTau_Hadronic_decay_massregssion_samples_m1p8To3p6_pt30To300_combined_original_valid.h5 --output_data_path=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To18_pt30To300_with_unphysical_AToTau_decay_original_unbiased_valid_h5 --output_file=IMG_massregssion_samples_m0To3p6_pt30To300_with_unphysical_AToTau_decay_original_unbiased_valid.h5 &
srun -n 1 -c 1 --cpu_bind=cores python h5_to_unbiased_h5.py --bin_count=1800 --input_file=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To3p6_pt30To300_with_unphysical_Tau_decay_original_train_h5/IMG_Tau_Hadronic_decay_massregssion_samples_m1p8To3p6_pt30To300_combined_original_train.h5 --output_data_path=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To18_pt30To300_with_unphysical_Tau_decay_original_unbiased_train_h5 --output_file=IMG_massregssion_samples_m0To3p6_pt30To300_with_unphysical_Tau_decay_original_unbiased_train.h5 &
srun -n 1 -c 1 --cpu_bind=cores python h5_to_unbiased_h5.py --bin_count=200 --input_file=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To3p6_pt30To300_with_unphysical_Tau_decay_original_valid_h5/IMG_Tau_Hadronic_decay_massregssion_samples_m1p8To3p6_pt30To300_combined_original_valid.h5 --output_data_path=/pscratch/sd/b/bbbam/IMG_massregssion_samples_m0To18_pt30To300_with_unphysical_Tau_decay_original_unbiased_valid_h5 --output_file=IMG_massregssion_samples_m0To3p6_pt30To300_with_unphysical_Tau_decay_original_unbiased_valid.h5 &

wait
