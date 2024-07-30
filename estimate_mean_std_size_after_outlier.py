import os, glob, re
import shutil
import random
import json
import numpy as np
import h5py
import math
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


orig_mean  = [ 1.98238779, -0.91442459,  0.41702185,  0.43513746,  0.02550795,  1.03056945,
  1.02679871,  1.03097381,  1.03844133,  1.62629969,  1.68150326,  1.6804281,
  1.68519913]

orig_std  = [1.59927372e+02, 2.85947917e+02, 2.79158669e+01, 2.07958307e+00,
 8.02803456e-02, 1.82661113e-01, 1.69144079e-01, 1.82877885e-01,
 2.07325503e-01, 9.95635565e-01, 1.09017288e+00, 1.07802983e+00,
 1.12664552e+00]

def mean_std_after_outlier(file="/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_2_unbaised_v2_train_hd5/IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_2_unbaised_v2_0000_train.h5",outdir = "mean_std_record_after_outlier", batch_size=7000, minimum_nonzero_pixels=3):
    print(f"processing file ---> {file}\n")
    tag = file.split('_')[-2]
    data = h5py.File(file, 'r')
    num_images = data["all_jet"].shape[0]
    print(f"data size: {num_images}\n")
    size_ = []
    mean_ = []
    std_ = []


    for start_idx in tqdm(range(0, num_images, batch_size)):
        end_idx = min(start_idx + batch_size, num_images)
        images_batch = data["all_jet"][start_idx:end_idx, :, :, :]
        # print("start_idx : ", start_idx)
        # print("end_idx : ", end_idx)
        # print("images_batch before :",images_batch.shape)
        images_batch[images_batch < 1.e-5] = 0
        non_zero_mask = images_batch != 0
        # print("non_zero_mask", non_zero_mask)
        images_non_zero = np.where(non_zero_mask, images_batch, np.nan)
        size_channel = np.count_nonzero(non_zero_mask, axis=(2, 3))
        mean_channel = np.nanmean(images_non_zero, axis=(2, 3))
        std_channel = np.nanstd(images_non_zero, axis=(2, 3), ddof=1)
        non_empty_event = np.all(size_channel > minimum_nonzero_pixels, axis=1)
        mean_channel = mean_channel[non_empty_event]
        std_channel = std_channel[non_empty_event]
        size_channel = size_channel[non_empty_event]
        # print("mean_channel   :", mean_channel.shape)

        # print("orig_mean   :", orig_mean.shape)
        # non_outlier_event = np.all(size_channel > 1 and mean_channel < (np.tile(orig_mean, (batch_size, 1)) + 10 * np.tile(orig_std, (batch_size, 1))), axis=1)
        non_outlier_event = np.all(np.logical_and(size_channel > 1, mean_channel < (orig_mean + 10 * orig_std)), axis=1)
        size_channel = size_channel[non_outlier_event]
        mean_channel = mean_channel[non_outlier_event]
        std_channel = std_channel[non_outlier_event]
        # print("size_channel  after :", size_channel.shape)
        size_.append(size_channel)
        mean_.append(mean_channel)
        std_.append(std_channel)

    data.close()
    size_ = np.concatenate(size_, axis=0).T
    mean_ = np.concatenate(mean_, axis=0).T
    std_ = np.concatenate(std_, axis=0).T
    after_outlier_mean, after_outlier_std = estimate_population_parameters(size_, mean_, std_)

    print(f'Means with out outliers: {after_outlier_mean}\n' )
    print(f'Stds with out outliers : { after_outlier_std}\n')
    print(f'number_of_selected_jets_std : {std_.shape[1]}\n')


    stat = {
            "after_outlier_mean":after_outlier_mean,
            "after_outlier_std":after_outlier_std,
            "number_of_selected_jets":std_.shape[1]
            }

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(outdir +'/'+ f'after_outlier_mean_std_record_dataset_{tag}.json', 'w') as fp:
        json.dump(stat, fp)

    return after_outlier_mean, after_outlier_std

# ### Run only once to calculate the mean and std after outlier removed
def process_files(file):
    file_path = file[0]
    mean_std_after_outlier(file=file_path,outdir = "mean_std_record_after_outlier", batch_size=14000, minimum_nonzero_pixels=3)

file_list = glob.glob("/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_2_unbaised_v2_train_hd5/*")
args = list(zip(file_list))
with Pool(10) as p:
    p.map(process_files,args)
print("-----Process Complete-----------------")
