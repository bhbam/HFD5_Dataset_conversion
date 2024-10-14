import os, time
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


def h5_h5_normalizes(in_file='', out_dir='', out_dir_mean_std_record='', in_size=-1, batch_size=3200, chunk_size=32):
    
    
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
        dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'm0']
        # dataset_names = ['all_jet', 'am', 'ieta', 'iphi']
        datasets = {
            name: proper_data.create_dataset(
                name,
                shape= (0,13, 125, 125) if 'jet' in name else (0,1),
                maxshape=(None, 13, 125, 125) if 'jet' in name else (None, 1),
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
            am_batch = am_batch[non_empty_event]
            ieta_batch = ieta_batch[non_empty_event]
            iphi_batch = iphi_batch[non_empty_event]
            m0_batch   = m0_batch[non_empty_event]
    
            non_outlier_event = np.all(np.logical_and(size_channel > 1, mean_channel < (orig_mean + 10 * orig_std)), axis=1)

        
            images_non_zero = images_non_zero[non_empty_event]

            images_non_zero = images_non_zero[non_outlier_event]
            am_batch = am_batch[non_outlier_event]
            ieta_batch = ieta_batch[non_outlier_event]
            iphi_batch = iphi_batch[non_outlier_event]
            m0_batch = m0_batch[non_outlier_event]


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
            

            for name, dataset in datasets.items():
                dataset.resize((end_idx_,13, 125, 125) if 'jet' in name else (end_idx_,1))


            proper_data['all_jet'][start_idx_:end_idx_,:,:,:] = images_non_zero
            proper_data['am'][start_idx_:end_idx_] = am_batch
            proper_data['ieta'][start_idx_:end_idx_] = ieta_batch
            proper_data['iphi'][start_idx_:end_idx_] = iphi_batch
            proper_data['m0'][start_idx_:end_idx_] = m0_batch
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
    parser.add_argument('--batch_size', type=int, default=30000, 
                        help='input batch size for conversion')
    parser.add_argument('--chunk_size', type=int, default=32, 
                        help='chunk size')
    parser.add_argument('--minumum_non_zero_pixels', type=int, default=3, 
                        help='number of minimun non zero pixel in image')
    parser.add_argument('--in_size', type=int, default=-1, 
                        help='number of input to process')
    args = parser.parse_args()

    start_time=time.time()
    minimum_nonzero_pixels = args.minumum_non_zero_pixels
    chunk_size = args.chunk_size
    infile = args.input_file
    out_dir = args.output_data_path
    in_size = args.in_size
    batch_size = args.batch_size



    #--------------------------dir to store mean and std after normalized--------------------------------------------
    out_dir_record = 'mean_std_record_normalized_signal_sample'
    # mean and std not equal to 0 and 1 because we replace all non values with zero and normalized
    #----------------------------chage these values for removing outlier and normalization -------------------------------------------------
    
    orig_mean = np.array([2.0213729049447173, -0.6212337778385966, 0.4101959111624071, 0.44232572099413325, 0.02601063666646348, 1.0307435733760915, 1.027161095232908, 1.0314644000120317, 1.0389142152234738, 1.6267729507544177, 1.682110558106714, 1.6807049908347997, 1.6856835949463678])
    orig_std = np.array([41.89302377240654, 281.7486185723839, 27.530200528379964, 2.0918438486330366, 0.08356221974207988, 0.18326159702115774, 0.17086703465837277, 0.1850492889044042, 0.2091712532384953, 0.9963542872097075, 1.0910279320573346, 1.0783809219516163, 1.127029077328961])
    # number_of_orig_jets = 5299319

    after_outlier_mean = np.array([2.00186027194732, -0.9327667763785292, 0.4117189123851803, 0.4427790365727392, 0.026002430537122163, 1.0307441059908344, 1.0271344385922005, 1.0314362031273734, 1.0389161993823062, 1.6264642395105546, 1.6817442443921755, 1.6806286899903164, 1.6854234605783123])
    after_outlier_std = np.array([11.602097269944196, 281.72189885561073, 27.51383690353155, 2.097769151892147, 0.08358663446283951, 0.1832351861511353, 0.1707668627208025, 0.18496502097288314, 0.20920433520293422, 0.9961141821301918, 1.090744381124439, 1.078384463638479, 1.1269819423138625])
    # number_of_selected_jets =  5263004

    
    #----------------------------------set for mass regression----------------------------------------------------------------------------
    
    h5_h5_normalizes(infile, out_dir, out_dir_record, in_size, batch_size, chunk_size)
    end_time=time.time()
    print(f"-----All Processes Completed in {(end_time-start_time)/60} minutes-----------------")
