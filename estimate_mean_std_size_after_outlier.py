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

orig_mean = np.array([4.645458183417245, 0.06504846315331393, 0.34378980642869167, 0.7972543466848049, 0.023563469460984632, 1.0280894253257178, 1.0346738050616027, 1.0358756760067354, 1.0445996914155187, 1.8379277268808463, 1.8892057488773362, 1.8500928664771055, 1.842420185120387])
orig_std = np.array([20033.623117109404, 328.84449319780407, 27.577214664898687, 3.566478596451319, 0.08592338693956458, 0.17582382685766373, 0.19342414362074953, 0.19792395671219273, 0.22449221492860144, 1.1686338166301715, 1.2592183032480533, 1.2164901235916066, 1.258055319930126])

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
    mean_std_after_outlier(file=file_path,outdir = "run_3_mean_std_record_after_outlier", batch_size=30000, minimum_nonzero_pixels=3)
start_time=time.time()
file_list = glob.glob("/pscratch/sd/b/bbbam/IMG_aToTauTau_m1p2T018_combined_h5/*.h5")
args = list(zip(file_list))
with Pool(len(file_list)) as p:
    p.map(process_files,args)
end_time=time.time()
print(f"-----All Processes Completed in {(end_time-start_time)/60} minutes-----------------")
