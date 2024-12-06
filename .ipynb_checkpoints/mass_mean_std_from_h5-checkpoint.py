import h5py, random
import numpy as np
import os, glob, json
from tqdm import tqdm


outdir_record = 'mass_mean_std_record'
file =glob.glob('/pscratch/sd/b/bbbam/IMG_aToTauTau_m1p2T018_combined_normalized_h5/*train*')
batch_size =6400
file_ = file[0]
data = h5py.File(f'{file_}', 'r')
num_images = len(data["am"])
num_images_select = num_images
prefix = file_.split('/')[-1].split('.')[0]
print("Total number----", num_images)

am = []
for start_idx in tqdm(range(0, num_images_select, batch_size)):
    end_idx = min(start_idx + batch_size, num_images)
    am_batch = data["am"][start_idx:end_idx, :]
    am.append(am_batch)
    
am = np.concatenate(am)
mass_mean = np.mean(am)
mass_std = np.std(am)
stat = {
    "mass_mean": float(mass_mean),  # Convert to Python float
    "mass_std": float(mass_std),    # Convert to Python float
    "number_of_selected_jets": len(am)
}


if not os.path.exists(outdir_record):
        os.makedirs(outdir_record)

with open(outdir_record +'/'+ f'mean_std_{prefix}.json', 'w') as fp:
    json.dump(stat, fp)

print("---------Done--------------")