import os, time, random
import json
import numpy as np
import h5py
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', default='/pscratch/sd/b/bbbam/IMG_aToTauTau_m1p2T018_combined_normalized_h5/IMG_aToTauTau_Hadronic_m1p2To18_pt30T0300_unbiased_train_normalized.h5',
                    help='input data path')
parser.add_argument('--output_data_path', default='/pscratch/sd/b/bbbam/IMG_aToTauTau_m0T018_with_stretching_lower_bins_combined_normalized_h5',
                    help='output data path')
parser.add_argument('--batch_size', type=int, default=320,
                    help='input batch size for conversion')
parser.add_argument('--chunk_size', type=int, default=32,
                    help='chunk size')
parser.add_argument('--in_size', type=int, default=-1,
                    help='number of input to process')
args = parser.parse_args()

start_time=time.time()
chunk_size = args.chunk_size
infile = args.input_file
out_dir = args.output_data_path
in_size = args.in_size
batch_size = args.batch_size

unphy_bins = np.arange(2.4,3.6,0.4)

data = h5py.File(f'{infile}', 'r')
num_images = data["all_jet"].shape[0] if in_size == -1 else in_size
outdir = out_dir

if not os.path.exists(outdir):
    os.makedirs(outdir)

prefix = infile.split('/')[-1].split('.')[0]
outfile = f'{prefix}_with_stretching_lower_bins.h5'

with h5py.File(f'{outdir}/{outfile}', 'w') as proper_data:
    dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'apt']
    datasets = {
    name: proper_data.create_dataset(
        name,
        shape= (0,13, 125, 125) if 'all_jet' in name else (0,1),
        maxshape=(None, 13, 125, 125) if 'all_jet' in name else (None, 1),
        dtype='float32',
        compression='lzf',
        chunks=(chunk_size, 13, 125, 125) if 'all_jet' in name else (chunk_size, 1),
    ) for name in dataset_names
    }
    orig_num_am = 0
    for start_idx_ in tqdm(range(0, num_images, batch_size)):
        end_idx_ = min(start_idx_ + batch_size, num_images)
        images_batch = data["all_jet"][start_idx_:end_idx_, :, :, :]
        am_batch = data["am"][start_idx_:end_idx_, :]
        ieta_batch = data["ieta"][start_idx_:end_idx_, :]
        iphi_batch = data["iphi"][start_idx_:end_idx_, :]
        apt_batch = data["apt"][start_idx_:end_idx_, :]

        am_batch[am_batch < 3.6] -= 1.2  
        lowest_mass_mask = (am_batch >= 3.6) & (am_batch < 4.8)
        stretched_images = images_batch[lowest_mass_mask.flatten()]
        stretched_am = am_batch[lowest_mass_mask]
        stretched_ieta = ieta_batch[lowest_mass_mask]
        stretched_iphi = iphi_batch[lowest_mass_mask]
        stretched_apt = apt_batch[lowest_mass_mask]

        if len(stretched_am) > 0:
            new_am_batch = (stretched_am - 1.2).reshape(-1, 1)
            new_images_batch = stretched_images
            new_ieta_batch = stretched_ieta.reshape(-1, 1)
            new_iphi_batch = stretched_iphi.reshape(-1, 1)
            new_apt_batch = stretched_apt.reshape(-1, 1)

            # Shuffle indices
            shuffle_indices = np.random.permutation(len(new_images_batch))

            # Apply shuffle
            new_am_batch = new_am_batch[shuffle_indices]
            new_images_batch = new_images_batch[shuffle_indices]
            new_ieta_batch = new_ieta_batch[shuffle_indices]
            new_iphi_batch = new_iphi_batch[shuffle_indices]
            new_apt_batch = new_apt_batch[shuffle_indices]

            # Concatenate with original batch
            full_am_batch = np.concatenate([am_batch, new_am_batch], axis=0)
            full_images_batch = np.concatenate([images_batch, new_images_batch], axis=0)
            full_ieta_batch = np.concatenate([ieta_batch, new_ieta_batch], axis=0)
            full_iphi_batch = np.concatenate([iphi_batch, new_iphi_batch], axis=0)
            full_apt_batch = np.concatenate([apt_batch, new_apt_batch], axis=0)
        else:
            full_am_batch = am_batch
            full_images_batch = images_batch
            full_ieta_batch = ieta_batch
            full_iphi_batch = iphi_batch
            full_apt_batch = apt_batch

        orig_num_am = orig_num_am + full_images_batch.shape[0]
        for name, dataset in datasets.items():
            dataset.resize((orig_num_am,13, 125, 125) if 'all_jet' in name else (orig_num_am,1))
        
        proper_data['all_jet'][orig_num_am-full_images_batch.shape[0]:orig_num_am] = full_images_batch
        proper_data['am'][orig_num_am-full_images_batch.shape[0]:orig_num_am] = full_am_batch
        proper_data['ieta'][orig_num_am-full_images_batch.shape[0]:orig_num_am] = full_ieta_batch
        proper_data['iphi'][orig_num_am-full_images_batch.shape[0]:orig_num_am] = full_iphi_batch
        proper_data['apt'][orig_num_am-full_images_batch.shape[0]:orig_num_am] = full_apt_batch

data.close()
end_time=time.time()
print(f"-----All Processes Completed in {(end_time-start_time)/60} minutes-----------------")
