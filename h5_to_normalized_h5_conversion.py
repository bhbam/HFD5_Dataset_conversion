import os
import json
import numpy as np
import h5py
from tqdm import tqdm
import argparse





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


def h5_h5_normalizes(in_file='', out_dir='', out_dir_mean_std_record='', in_size=-1, out_size=10000, batch_size=3200, chunk_size=32):
    
    
    file = in_file
    data = h5py.File(f'{file}', 'r')
    num_images = data["all_jet"].shape[0] if in_size == -1 else in_size

    print(f"processing file ---> {file}\n")
    outdir = out_dir
    outdir_record = out_dir_mean_std_record
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    prefix = file.split('/')[-1].split('.')[0]
    outfile = f'{prefix}_normalized.h5'

    with h5py.File(f'{outdir}/{outfile}', 'w') as proper_data:
        dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'm0', 'apt', 'jetpt', 'taudR']
        datasets = {
            name: proper_data.create_dataset(
                name,
                (out_size,13, 125, 125) if 'jet' in name else (out_size, 1),
                dtype='float32',  # Specify an appropriate data type
                compression='lzf',
                chunks=(chunk_size, 13, 125, 125) if 'jet' in name else (chunk_size, 1),
            ) for name in dataset_names
        }

        size_ = []
        mean_ = []
        std_ = []
        start_idx_, end_idx_ = 0, 0
        for start_idx in tqdm(range(0, num_images, batch_size)):
            end_idx = min(start_idx + batch_size, num_images)
            images_batch = data["all_jet"][start_idx:end_idx, :, :, :]
            am_batch = data["am"][start_idx:end_idx, :]
            ieta_batch = data["ieta"][start_idx:end_idx, :]
            iphi_batch = data["iphi"][start_idx:end_idx, :]
            m0_batch = data["m0"][start_idx:end_idx, :]
            apt_batch = data["apt"][start_idx:end_idx, :]
            jetpt_batch = data["jetpt"][start_idx:end_idx, :]
            taudR_batch = data["tausdR"][start_idx:end_idx, :]

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
            am_batch = am_batch[non_empty_event]
            ieta_batch = ieta_batch[non_empty_event]
            iphi_batch = iphi_batch[non_empty_event]
            m0_batch   = m0_batch[non_empty_event]
            apt_batch   = apt_batch[non_empty_event]
            jetpt_batch   = jetpt_batch[non_empty_event]
            taudR_batch   = taudR_batch[non_empty_event]
    
            non_outlier_event = np.all(np.logical_and(size_channel > 1, mean_channel < (orig_mean + 10 * orig_std)), axis=1)

        
            images_non_zero = images_non_zero[non_empty_event]

            images_non_zero = images_non_zero[non_outlier_event]
            am_batch = am_batch[non_outlier_event]
            ieta_batch = ieta_batch[non_outlier_event]
            iphi_batch = iphi_batch[non_outlier_event]
            m0_batch = m0_batch[non_outlier_event]
            apt_batch = apt_batch[non_outlier_event]
            jetpt_batch = jetpt_batch[non_outlier_event]
            taudR_batch = taudR_batch[non_outlier_event]


            mean_channel = mean_channel[non_outlier_event]
            std_channel = std_channel[non_outlier_event]
            size_channel = size_channel[non_outlier_event]

            images_non_zero[np.isnan(images_non_zero)] = 0
            images_non_zero = (images_non_zero - after_outlier_mean.reshape(1, 13, 1, 1)) / after_outlier_std.reshape(1, 13, 1, 1)
            
            mean_channel_ = np.nanmean(images_non_zero, axis=(2, 3))
            std_channel_ = np.nanstd(images_non_zero, axis=(2, 3), ddof=1)
            size_.append(size_channel)
            mean_.append(mean_channel_)
            std_.append(std_channel_)
            start_idx_ = min(start_idx, end_idx_)
            end_idx_   = min(start_idx_ + images_non_zero.shape[0], num_images)
            


            proper_data['all_jet'][start_idx_:end_idx_,:,:,:] = images_non_zero
            proper_data['am'][start_idx_:end_idx_] = am_batch
            proper_data['ieta'][start_idx_:end_idx_] = ieta_batch
            proper_data['iphi'][start_idx_:end_idx_] = iphi_batch
            proper_data['m0'][start_idx_:end_idx_] = m0_batch
            proper_data['apt'][start_idx_:end_idx_] = apt_batch
            proper_data['jetpt'][start_idx_:end_idx_] = jetpt_batch
            proper_data['taudR'][start_idx_:end_idx_] = taudR_batch
            # print("_____________________________________________________________")

    data.close()
    size_ = np.concatenate(size_, axis=0).T
    mean_ = np.concatenate(mean_, axis=0).T
    std_ = np.concatenate(std_, axis=0).T
    normalised_mean, normalised_std = estimate_population_parameters(size_, mean_, std_)

    print(f'Normalised mean: {normalised_mean}\n' )
    print(f'normalised std : {normalised_std}\n')
    print(f'number_of_selected_jets_std : {std_.shape[1]}\n')

    stat = {
            "normalised_mean":normalised_mean,
            "normalised_std":normalised_std,
            "number_of_selected_jets":std_.shape[1]
            }

    if not os.path.exists(outdir_record):
        os.makedirs(outdir_record)

    with open(outdir_record +'/'+ f'mean_std_{prefix}.json', 'w') as fp:
        json.dump(stat, fp)

    return normalised_mean, normalised_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', default='/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_original_combined/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_original_combined_valid.h5', 
                        help='input data path')
    parser.add_argument('--output_data_path', default='/pscratch/sd/b/bbbam/IMG_aToTauTau_masregression_samples_m1p2To17p2_combined_normalized',
                        help='output data path')
    parser.add_argument('--batch_size', type=int, default=5000, 
                        help='input batch size for conversion')
    parser.add_argument('--chunk_size', type=int, default=32, 
                        help='chunk size')
    parser.add_argument('--minumum_non_zero_pixels', type=int, default=3, 
                        help='number of minimun non zero pixel in image')
    parser.add_argument('--in_size', type=int, default=-1, 
                        help='number of input to process')
    parser.add_argument('--out_size', type=int, default=10000, 
                        help='number of output /size of new h5 file')
    args = parser.parse_args()

    minimum_nonzero_pixels = args.minumum_non_zero_pixels
    chunk_size = args.chunk_size
    infile = args.input_file
    out_dir = args.output_data_path
    in_size = args.in_size
    out_size = args.out_size
    batch_size = args.batch_size

    out_dir_record = 'mean_std_record_normalized_massregression_sample'
    #----------------------------chage these values for removing outlier and normalization -------------------------------------------------
    orig_mean  = np.array([1.98238779, -0.91442459,  0.41702185,  0.43513746,  0.02550795,  1.03056945,
    1.02679871,  1.03097381,  1.03844133,  1.62629969,  1.68150326,  1.6804281,  1.68519913])

    orig_std  = np.array([1.59927372e+02, 2.85947917e+02, 2.79158669e+01, 2.07958307e+00,
    8.02803456e-02, 1.82661113e-01, 1.69144079e-01, 1.82877885e-01,
    2.07325503e-01, 9.95635565e-01, 1.09017288e+00, 1.07802983e+00, 1.12664552e+00]) 

    after_outlier_mean = np.array([1.95973739, -0.91428634,  0.41695268,  0.4351373,   0.02550794,  1.03056946,
    1.02679871,  1.03097382,  1.03844135,  1.62629992,  1.6815035,   1.68042818, 1.68519924]) 

    after_outlier_std =np.array([2.64603079e+01, 2.85947850e+02, 2.78975093e+01, 2.07958377e+00,
    8.02803342e-02, 1.82661149e-01, 1.69144090e-01, 1.82877912e-01,
    2.07325558e-01, 9.95635728e-01, 1.09017309e+00, 1.07802985e+00, 1.12664562e+00])

    #-------------------------------------set for mass regression-------------------------------------------------------------------------
    
    h5_h5_normalizes(infile, out_dir, out_dir_record, in_size, out_size, batch_size, chunk_size)
    print("------------Process is complete-------------------")
