#!/bin/bash
#SBATCH --account=m4392
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J slurm_h5_h5
#SBATCH -t 24:00:00
#SBATCH --output=slurm_normalized_to_h5_%J.out
#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 1 --cpu_bind=cores python h5_to_normalized_h5_conversion_dynamic_size.py       -p=AToTau --input_data_file=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_h5/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_train.h5 --output_data_path=/global/cfs/cdirs/m4392/bbbam/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_normalized_unbiased_combined_h5 &
srun -n 1 -c 1 --cpu_bind=cores python h5_to_normalized_h5_conversion_dynamic_size.py       -p=Tau    --input_data_file=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_original_unbiased_combined_h5/IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_original_unbiased_combined_train.h5       --output_data_path=/global/cfs/cdirs/m4392/bbbam/IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_normalized_unbiased_combined_h5 &
srun -n 1 -c 1 --cpu_bind=cores python h5_to_normalized_h5_conversion_dynamic_size_valid.py -p=AToTau --input_data_file=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_h5/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_valid.h5 --output_data_path=/global/cfs/cdirs/m4392/bbbam/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_normalized_unbiased_combined_h5 &
srun -n 1 -c 1 --cpu_bind=cores python h5_to_normalized_h5_conversion_dynamic_size_valid.py -p=Tau    --input_data_file=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_original_unbiased_combined_h5/IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_original_unbiased_combined_valid.h5       --output_data_path=/global/cfs/cdirs/m4392/bbbam/IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_normalized_unbiased_combined_h5 &
wait
