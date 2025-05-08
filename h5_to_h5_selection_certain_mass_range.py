import os, time, random
import json
import numpy as np
import h5py
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', default='/global/cfs/cdirs/m4392/bbbam/IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300_unbiased_combined_h5/IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300_unbiased_combined_valid.h5',
                    help='input data path')
parser.add_argument('--output_data_path', default='/global/cfs/cdirs/m4392/bbbam/IMG_aToTauTau_Hadronic_m3p6To14_pt30T0300_unbiased_combined_h5',
                    help='output data path')
parser.add_argument('--prefix', type=str, default='valid',
                    help='select train or valid')
parser.add_argument('--batch_size', type=int, default=320,
                    help='input batch size for conversion')
parser.add_argument('--chunk_size', type=int, default=32,
                    help='chunk size')
parser.add_argument('--in_size', type=int, default=-1,
                    help='number of input to process')
parser.add_argument('--lower_mass', type=float, default=3.6,
                    help='lower mass point')
parser.add_argument('--upper_mass', type=float, default=14,
                    help='upper mass point')
args = parser.parse_args()

start_time=time.time()
chunk_size = args.chunk_size
infile = args.input_file
out_dir = args.output_data_path
in_size = args.in_size
batch_size = args.batch_size

def mass_str(mass):
    return str(mass).replace('.', 'p')

data = h5py.File(f'{infile}', 'r')
num_images = data["all_jet"].shape[0] if in_size == -1 else in_size
outdir = out_dir

if not os.path.exists(outdir):
    os.makedirs(outdir)

outfile = f'IMG_aToTauTau_Hadronic_m{mass_str(args.lower_mass)}To{mass_str(args.upper_mass)}_pt30T0300_unbiased_combined_{args.prefix}.h5'


with h5py.File(f'{outdir}/{outfile}', 'w') as proper_data:
    dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'apt', 'jet_pt', 'jet_mass']

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
        if images_batch[:, 0, ...].max() > 1000:
            continue
        images_batch[np.abs(images_batch) < 1.e-3] = 0
        am_batch = data["am"][start_idx_:end_idx_, :]
        ieta_batch = data["ieta"][start_idx_:end_idx_, :]
        iphi_batch = data["iphi"][start_idx_:end_idx_, :]
        apt_batch = data["apt"][start_idx_:end_idx_, :]
        jetpt_batch = data["jetpt"][start_idx_:end_idx_, :]
        jetmass_batch = data["m0"][start_idx_:end_idx_, :]

       
        
        mass_mask = (am_batch >= args.lower_mass) & (am_batch <= 14)

        selected_images_batch = images_batch[mass_mask.flatten()]

        if (len(selected_images_batch)) > 0:
            selected_am_batch = am_batch[mass_mask]
            selected_ieta_batch = ieta_batch[mass_mask]
            selected_iphi_batch = iphi_batch[mass_mask]
            selected_apt_batch = apt_batch[mass_mask]
            selected_jetpt_batch = jetpt_batch[mass_mask]
            selected_jetmass_batch = jetmass_batch[mass_mask]
        
        
            new_am_batch = selected_am_batch.reshape(-1, 1)
            new_images_batch = selected_images_batch
            new_ieta_batch = selected_ieta_batch.reshape(-1, 1)
            new_iphi_batch = selected_iphi_batch.reshape(-1, 1)
            new_apt_batch = selected_apt_batch.reshape(-1, 1)
            new_jetpt_batch = selected_jetpt_batch.reshape(-1, 1)
            new_jetmass_batch = selected_jetmass_batch.reshape(-1, 1)

            orig_num_am = orig_num_am + new_images_batch.shape[0]
            for name, dataset in datasets.items():
                dataset.resize((orig_num_am,13, 125, 125) if 'all_jet' in name else (orig_num_am,1))
            
            proper_data['all_jet'][orig_num_am-new_images_batch.shape[0]:orig_num_am] = new_images_batch
            proper_data['am'][orig_num_am-new_images_batch.shape[0]:orig_num_am] = new_am_batch
            proper_data['ieta'][orig_num_am-new_images_batch.shape[0]:orig_num_am] = new_ieta_batch
            proper_data['iphi'][orig_num_am-new_images_batch.shape[0]:orig_num_am] = new_iphi_batch
            proper_data['apt'][orig_num_am-new_images_batch.shape[0]:orig_num_am] = new_apt_batch
            proper_data['jet_pt'][orig_num_am-new_images_batch.shape[0]:orig_num_am] = new_jetpt_batch
            proper_data['jet_mass'][orig_num_am-new_images_batch.shape[0]:orig_num_am] = new_jetmass_batch

data.close()
end_time=time.time()
print(f"-----All Processes Completed in {(end_time-start_time)/60} minutes-----------------")
