#!/bin/bash
#SBATCH --account=m4392
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J slurm_original_mean_std
#SBATCH -t 12:05:00
#SBATCH --output=slurm_original_mean_std_%J.out

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 1 --cpu_bind=cores python estimate_mean_std_original_sample.py --input_data_file=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_Tau_decay_v2_m0To18_pt30T0300_original_unbiased_combined_h5/IMG_aToTauTau_Hadronic_with_Tau_decay_v2_m0To18_pt30T0300_original_unbiased_combined_train.h5 --output_data_path=run_3_original_mean_std_record_with_Tau_decay_v2 &
srun -n 1 -c 1 --cpu_bind=cores python estimate_mean_std_original_sample.py --input_data_file=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_Tau_decay_v2_m0To18_pt30T0300_original_unbiased_combined_h5/IMG_aToTauTau_Hadronic_with_Tau_decay_v2_m0To18_pt30T0300_original_unbiased_combined_valid.h5 --output_data_path=run_3_original_mean_std_record_with_Tau_decay_v2 &
wait
