import os, glob, re
import json
import numpy as np
import h5py
import time
from tqdm import tqdm
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
plt.style.use([hep.style.ROOT, hep.style.firamath])
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


orig_mean = np.array([2.0213729049447173, -0.6212337778385966, 0.4101959111624071, 0.44232572099413325, 0.02601063666646348, 1.0307435733760915, 1.027161095232908, 1.0314644000120317, 1.0389142152234738, 1.6267729507544177, 1.682110558106714, 1.6807049908347997, 1.6856835949463678])
orig_std = np.array([41.89302377240654, 281.7486185723839, 27.530200528379964, 2.0918438486330366, 0.08356221974207988, 0.18326159702115774, 0.17086703465837277, 0.1850492889044042, 0.2091712532384953, 0.9963542872097075, 1.0910279320573346, 1.0783809219516163, 1.127029077328961])


def mean_std_after_outlier(file="/pscratch/sd/b/bbbam/MG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_original_combined/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_original_combined_valid.h5",outdir = "mean_std_record_after_outlier_combined", batch_size=7000, minimum_nonzero_pixels=3):
    print(f"processing file ---> {file}\n")
    # tag = file.split('_')[-2]
    tag = file.split('_')[-1].split('.')[0]
    data = h5py.File(file, 'r')
    num_images = data["all_jet"].shape[0]
    print(f"data size: {num_images}\n")
    size_ = []
    mean_ = []
    std_ = []


    for start_idx in tqdm(range(0, num_images, batch_size)):
        end_idx = min(start_idx + batch_size, num_images)
        images_batch = data["all_jet"][start_idx:end_idx, :, :, :]
        images_batch[np.abs(images_batch) < 1.e-5] = 0
        non_zero_mask = images_batch != 0
        images_non_zero = np.where(non_zero_mask, images_batch, np.nan)
        size_channel = np.count_nonzero(non_zero_mask, axis=(2, 3))
        mean_channel = np.nanmean(images_non_zero, axis=(2, 3))
        std_channel = np.nanstd(images_non_zero, axis=(2, 3), ddof=1)
        non_empty_event = np.all(size_channel > minimum_nonzero_pixels, axis=1)
        mean_channel = mean_channel[non_empty_event]
        std_channel = std_channel[non_empty_event]
        size_channel = size_channel[non_empty_event]
        non_outlier_event = np.all(np.logical_and(size_channel > 1, mean_channel < (orig_mean + 10 * orig_std)), axis=1)
        size_channel = size_channel[non_outlier_event]
        mean_channel = mean_channel[non_outlier_event]
        std_channel = std_channel[non_outlier_event]
        size_.append(size_channel)
        mean_.append(mean_channel)
        std_.append(std_channel)

    data.close()
    size_ = np.concatenate(size_, axis=0).T
    mean_ = np.concatenate(mean_, axis=0).T
    std_ = np.concatenate(std_, axis=0).T
    after_outlier_mean, after_outlier_std = estimate_population_parameters(size_, mean_, std_)

    print(f'Means with out outliers {tag}: {after_outlier_mean}\n' )
    print(f'Stds with out outliers {tag}: { after_outlier_std}\n')
    print(f'number_of_selected_jets_std {tag}: {std_.shape[1]}\n')


    stat = {
            "after_outlier_mean":after_outlier_mean,
            "after_outlier_std":after_outlier_std,
            "number_of_selected_jets":std_.shape[1]
            }

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(outdir +'/'+ f'after_outlier_mean_std_record_dataset_{tag}.json', 'w') as fp:
        json.dump(stat, fp)
    print(f"-----Processes Complete for file -------{file}----------")
    return after_outlier_mean, after_outlier_std

# ### Run only once to calculate the mean and std after outlier removed
def process_files(file):
    file_path = file[0]
    mean_std_after_outlier(file=file_path,outdir = "mean_std_record_after_outlier", batch_size=30000, minimum_nonzero_pixels=3)
start_time=time.time()
file_list = glob.glob("/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_original_combined_hd5/*")
args = list(zip(file_list))
with Pool(len(file_list)) as p:
    p.map(process_files,args)
end_time=time.time()
print(f"-----All Processes Completed in {(end_time-start_time)/60} minutes-----------------")
