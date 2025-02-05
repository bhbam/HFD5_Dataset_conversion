import os, glob, re
import json
import numpy as np
import h5py
import time
from tqdm import tqdm
from multiprocessing import Pool

minimum_nonzero_pixels = 3

def estimate_population_parameters(all_sample_sizes, all_sample_means, all_sample_stds):
    population_means = []
    population_stds = []
    for j in range(len(all_sample_means)):
        sample_means = all_sample_means[j]
        sample_stds = all_sample_stds[j]
        sample_sizes = all_sample_sizes[j]
        sample_means = sample_means[sample_sizes != 0]
        sample_stds = sample_stds[sample_sizes != 0]
        sample_sizes = sample_sizes[sample_sizes != 0]
        weighted_sum_of_variances = sum((n - 1) * s**2 for n, s in zip(sample_sizes, sample_stds))
        total_degrees_of_freedom = sum(n - 1 for n in sample_sizes)
        combined_variance = weighted_sum_of_variances / total_degrees_of_freedom
        population_std = np.sqrt(combined_variance)
        weighted_sum_of_means = sum(n * mean for n, mean in zip(sample_sizes, sample_means))
        total_observations = sum(sample_sizes)
        population_mean = weighted_sum_of_means / total_observations
        population_stds.append(population_std)
        population_means.append(population_mean)

    return population_means, population_stds

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)',s)]


def mean_std_original(file="/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_original_combined_valid.h5",outdir = "run_3_mean_std_record_original", batch_size=7000, minimum_nonzero_pixels=3):
    print(f"processing file ---> {file}\n")
    # tag = file.split('_')[-2]
    tag = file.split('_')[-1].split('.')[0]
    data = h5py.File(file, 'r')
    num_images = data["all_jet"].shape[0]
    print(f"data size: {num_images}\n")
    size_ = []
    mean_ = []
    std_ = []
    batch_size = batch_size
    for start_idx in tqdm(range(0, num_images, batch_size)):
        if images[0, ...].max() > 500:
            continue
        end_idx = min(start_idx + batch_size, num_images)
        images_batch = data["all_jet"][start_idx:end_idx, :, :, :]
        images_batch[np.abs(images_batch) < 1.e-3] = 0
        non_zero_mask = images_batch != 0
        images_non_zero = np.where(non_zero_mask, images_batch, np.nan)
        size_channel = np.count_nonzero(non_zero_mask, axis=(2, 3))
        mean_channel = np.nanmean(images_non_zero, axis=(2, 3))
        std_channel = np.nanstd(images_non_zero, axis=(2, 3), ddof=1)
        non_empty_event = np.all(size_channel > minimum_nonzero_pixels, axis=1)
        mean_channel = mean_channel[non_empty_event]
        std_channel = std_channel[non_empty_event]
        size_channel = size_channel[non_empty_event]
        size_.append(size_channel)
        mean_.append(mean_channel)
        std_.append(std_channel)

    data.close()  
    size_ = np.concatenate(size_, axis=0).T
    mean_ = np.concatenate(mean_, axis=0).T
    std_ = np.concatenate(std_, axis=0).T
    orig_mean, orig_std = estimate_population_parameters(size_, mean_, std_)


    print(f'Means with outliers {tag}: {orig_mean}\n' )
    print(f'Stds with outliers {tag}: {orig_std}\n')
    print(f'number of jets  outliers {tag}: {num_images}\n')

    stat = {
            "original_mean":orig_mean,
            "original_std":orig_std,
            "number_of_jets":num_images
            }

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(f"{outdir}/{outdir}_{tag}.json", 'w') as fp:
        json.dump(stat, fp)
    print(f"-----Process Complete for file------{file}-----------")
    return orig_mean, orig_std

### Run only once to calculate original mean and std
def process_files(file):
    file_path = file[0]
    mean_std_original(file=file_path,outdir = "run_3_mean_std_record_original", batch_size=30000, minimum_nonzero_pixels=3)
start_time=time.time()   
file_list = glob.glob("/pscratch/sd/b/bbbam/IMG_aToTauTau_m1p2T018_combined_h5/*.h5")   
args = list(zip(file_list)) 
with Pool(len(file_list)) as p:
    p.map(process_files,args)
end_time=time.time()
print(f"-----All Processes Completed in {(end_time-start_time)/60} minutes-----------------")
