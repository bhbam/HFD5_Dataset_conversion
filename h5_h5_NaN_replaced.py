import os, glob, re
import shutil
import json
import numpy as np
import h5py
import math
import time
from tqdm import tqdm
from multiprocessing import Pool
#------------------------ combined mean snd std after outlier -------------------
def combined_mean_std(size, mean, std):
    mean_ = np.dot(size, mean)/np.sum(size)
    std_ = np.sqrt((np.dot((np.array(size)-1), np.square(std)) + np.dot(size,np.square(mean-mean_)))/(np.sum(size)-1))
    return mean_, std_


mean_ = []
std_ = []
size_ = []
file_path = np.sort(glob.glob("/global/u1/b/bbbam/H_AA_CLuster_jupyter_notebooks/mean_std_record_after_outlier/*"))
for file in file_path:
    with open(file, 'r') as file:
        data = json.load(file)
    mean_.append(data['after_outlier_mean'])
    std_.append(data['after_outlier_std'])
    size_.append(data['number_of_selected_jets'])
mean = np.array(mean_)
std = np.array(std_)
size = np.array(size_)

# print("mean  :", mean)
# print("std  :", std)
# print("size  :", size)
# nan_mean = np.isnan(mean)
# print("nan_mean:  ",np.any(nan_mean))
# nan_std = np.isnan(std)
# print("nan_std:  ",np.any(nan_std))
# nan_size = np.isnan(size)
# print("nan_size:  ",np.any(nan_size))

after_outlier_mean, after_outlier_std = combined_mean_std(size, mean, std)
nan_replace = - after_outlier_mean/after_outlier_std
nan_after_outlier_mean = np.isnan(after_outlier_mean)
print("nan_after_outlier_mean:  ",np.any(nan_after_outlier_mean))
nan_after_outlier_std = np.isnan(after_outlier_std)
print("nan_after_outlier_std:  ",np.any(nan_after_outlier_std))

dim = (125, 125)

# Generate the desired array
nan_replace_array = np.array([np.full(dim, v) for v in nan_replace])
#------------------------------------------------------------------------


def repalce_NAN(file_, nan_replace_array):
    file = file_
    data = h5py.File(f'{file}', 'r')
    num_images = data["all_jet"].shape[0]
    # num_images = 5000  # Adjusted number of images for processing
    batch_size = 4000

    print(f"Processing file ---> {file}\n")
    tag = 'NAN_removed'
    outdir = '/pscratch/sd/b/bbbam/normalized_nan_replaced_h5'
    if not os.path.exists(outdir):
        # Create the directory if it doesn't exist
        os.makedirs(outdir)
    out_prefix = (file.split('/')[-1]).split('.')[:-1][0]
    outfile = f'{out_prefix}_{tag}_train.h5'

    with h5py.File(f'{outdir}/{outfile}', 'w') as proper_data:
        dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'm0']
        datasets = {
            name: proper_data.create_dataset(
                name,
                (num_images, 13, 125, 125) if 'jet' in name else (num_images, 1),
                dtype='float32',
                compression='lzf',
                chunks=(batch_size, 13, 125, 125) if 'jet' in name else (batch_size, 1),
            ) for name in dataset_names
        }

        for start_idx in tqdm(range(0, num_images, batch_size)):
            end_idx = min(start_idx + batch_size, num_images)
            images_batch = data["all_jet"][start_idx:end_idx, :, :, :]
            am_batch = data["am"][start_idx:end_idx, :]
            ieta_batch = data["ieta"][start_idx:end_idx, :]
            iphi_batch = data["iphi"][start_idx:end_idx, :]
            m0_batch = data["m0"][start_idx:end_idx, :]

            # Replace NaN values in images_batch with the specified transformation
            nan_mask = np.isnan(images_batch)
            images_batch[nan_mask] =  np.tile(nan_replace_array, (end_idx-start_idx, 1, 1, 1))[nan_mask]
            # Write the processed batch to the new HDF5 file
            proper_data['all_jet'][start_idx:end_idx, :, :, :] = images_batch
            proper_data['am'][start_idx:end_idx, :] = am_batch
            proper_data['ieta'][start_idx:end_idx, :] = ieta_batch
            proper_data['iphi'][start_idx:end_idx, :] = iphi_batch
            proper_data['m0'][start_idx:end_idx, :] = m0_batch
    data.close()
    print(f">>>>>>>>>>>>>>>{file} DONE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


def process_files(file):
    file_path = file[0]
    repalce_NAN(file_path, nan_replace_array)

file_list = glob.glob("/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_normalised_train_hd5/*/*")
print("Total files :", len(file_list))
args = list(zip(file_list))
with Pool(len(file_list)) as p:
    p.map(process_files,args)
