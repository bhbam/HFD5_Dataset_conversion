#!/bin/bash
#SBATCH --account=m4392
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J slurm_mea_std_after_outlier
#SBATCH -t 06:05:00
#SBATCH --output=slurm_mean_std_after_outlier_%J.out
#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 1 --cpu_bind=cores python estimate_mean_std_size_after_outlier.py -p=AToTau --input_data_file=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_h5/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_train.h5 --output_data_path=run_3_mean_std_record_with_AToTau_decay_after_outlier &
srun -n 1 -c 1 --cpu_bind=cores python estimate_mean_std_size_after_outlier.py -p=Tau    --input_data_file=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_original_unbiased_combined_h5/IMG_aToTauTau_Hadronic_with_Tau_decay_m0To18_pt30T0300_original_unbiased_combined_train.h5       --output_data_path=run_3_mean_std_record_with_Tau_decay_after_outlier &
wait