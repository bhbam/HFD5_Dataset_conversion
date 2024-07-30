import os, glob, re
import shutil
import json
import numpy as np
import h5py
import math
import time
from tqdm import tqdm
from multiprocessing import Pool


def chunksize_reduce(file_, new_size):
    file = file_
    data = h5py.File(f'{file}', 'r')
    num_images = data["all_jet"].shape[0]
    # num_images = 5000  # Adjusted number of images for processing
    batch_size = 4000
    # batch_size = 32

    print(f"Processing file ---> {file}\n")
    tag = f'chunksize_{new_size}'
    outdir = f'/pscratch/sd/b/bbbam/normalized_nan_replaced_m1p2To17p2_massreg_samples_{tag}_h5'
    os.makedirs(outdir, exist_ok=True)
    out_prefix = file.split('/')[-1]
    # print("out_prefix  ", out_prefix)
    outfile = f'{tag}_{out_prefix}'

    with h5py.File(f'{outdir}/{outfile}', 'w') as proper_data:
        dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'm0']
        datasets = {
            name: proper_data.create_dataset(
                name,
                (num_images, 13, 125, 125) if 'jet' in name else (num_images, 1),
                dtype='float32',
                compression='lzf',
                chunks=(new_size, 13, 125, 125) if 'jet' in name else (new_size, 1),
            ) for name in dataset_names
        }

        for start_idx in tqdm(range(0, num_images, batch_size)):
            end_idx = min(start_idx + batch_size, num_images)
            proper_data['all_jet'][start_idx:end_idx, :, :, :] = data["all_jet"][start_idx:end_idx, :, :, :]
            proper_data['am'][start_idx:end_idx, :] = data["am"][start_idx:end_idx, :]
            proper_data['ieta'][start_idx:end_idx, :] = data["ieta"][start_idx:end_idx, :]
            proper_data['iphi'][start_idx:end_idx, :] = data["iphi"][start_idx:end_idx, :]
            proper_data['m0'][start_idx:end_idx, :] = data["m0"][start_idx:end_idx, :]

    data.close()
    print(f">>>>>>>>>>>>>>>{file} DONE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


def process_files(file):
    file_path = file[0]
    chunksize_reduce(file_path, 32)

file_list = glob.glob("/pscratch/sd/b/bbbam/mass_regression_train_valid_data_H5/*")
print("Total files :", len(file_list))
args = list(zip(file_list))
with Pool(len(file_list)) as p:
    p.map(process_files,args)
