import os, time, random
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
    os.makedirs(outdir, exist_ok=True)

    prefix = file.split('/')[-1].split('.')[0]
    outfile = f'{prefix}_normalized.h5'

    with h5py.File(f'{outdir}/{outfile}', 'w') as proper_data:
        # dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'm0', 'apt', 'jetpt', 'taudR']
        dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'apt']
        datasets = {
        name: proper_data.create_dataset(
            name,
            shape= (0,13, 125, 125) if 'all_jet' in name else (0,1),
            maxshape=(None, 13, 125, 125) if 'all_jet' in name else (None, 1),
            dtype='float32',  # Specify an appropriate data type
            compression='lzf',
            chunks=(chunk_size, 13, 125, 125) if 'all_jet' in name else (chunk_size, 1),
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
            # m0_batch = data["m0"][start_idx:end_idx, :]
            apt_batch = data["apt"][start_idx:end_idx, :]
            # jetpt_batch = data["jetpt"][start_idx:end_idx, :]
            # taudR_batch = data["taudR"][start_idx:end_idx, :]

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
            # m0_batch   = m0_batch[non_empty_event]
            apt_batch   = apt_batch[non_empty_event]
            # jetpt_batch   = jetpt_batch[non_empty_event]
            # taudR_batch   = taudR_batch[non_empty_event]
    
            non_outlier_event = np.all(np.logical_and(size_channel > 1, mean_channel < (orig_mean + 10 * orig_std)), axis=1)

        
            images_non_zero = images_non_zero[non_empty_event]

            images_non_zero = images_non_zero[non_outlier_event]
            am_batch = am_batch[non_outlier_event]
            ieta_batch = ieta_batch[non_outlier_event]
            iphi_batch = iphi_batch[non_outlier_event]
            # m0_batch = m0_batch[non_outlier_event]
            apt_batch = apt_batch[non_outlier_event]
            # jetpt_batch = jetpt_batch[non_outlier_event]
            # taudR_batch = taudR_batch[non_outlier_event]


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
                dataset.resize((end_idx_,13, 125, 125) if 'all_jet' in name else (end_idx_,1))


            proper_data['all_jet'][start_idx_:end_idx_,:,:,:] = images_non_zero
            proper_data['am'][start_idx_:end_idx_] = am_batch
            proper_data['ieta'][start_idx_:end_idx_] = ieta_batch
            proper_data['iphi'][start_idx_:end_idx_] = iphi_batch
            # proper_data['m0'][start_idx_:end_idx_] = m0_batch
            proper_data['apt'][start_idx_:end_idx_] = apt_batch
            # proper_data['jetpt'][start_idx_:end_idx_] = jetpt_batch
            # proper_data['taudR'][start_idx_:end_idx_] = taudR_batch
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

    os.makedirs(outdir_record, exist_ok=True)
    with open(outdir_record +'/'+ f'mean_std_{prefix}.json', 'w') as fp:
        json.dump(stat, fp)

    return normalised_mean, normalised_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data_file', default='/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_original_combined/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_original_combined_valid.h5', 
                        help='input data path')
    parser.add_argument('--output_data_path', default='/pscratch/sd/b/bbbam/IMG_aToTauTau_masregression_samples_m1p2To17p2_combined_normalized',
                        help='output data path')
    parser.add_argument('-p', '--process',     default='Tau',    type=str, help='select signal or background or other')
    parser.add_argument('--batch_size', type=int, default=320, 
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
    infile = args.input_data_file
    out_dir = args.output_data_path
    in_size = args.in_size
    batch_size = args.batch_size



    #--------------------------dir to store mean and std after normalized--------------------------------------------
    out_dir_record = f'mean_std_record_normalized_massregression_sample_train_with_{args.process}'
    # mean and std not equal to 0 and 1 because we replace all non values with zero and normalized
    #----------------------------chage these values for removing outlier and normalization -------------------------------------------------
    
    # orig_mean = np.array([4.645458183417245, 0.06504846315331393, 0.34378980642869167, 0.7972543466848049, 0.023563469460984632, 1.0280894253257178, 1.0346738050616027, 1.0358756760067354, 1.0445996914155187, 1.8379277268808463, 1.8892057488773362, 1.8500928664771055, 1.842420185120387])
    # orig_std = np.array([20033.623117109404, 328.84449319780407, 27.577214664898687, 3.566478596451319, 0.08592338693956458, 0.17582382685766373, 0.19342414362074953, 0.19792395671219273, 0.22449221492860144, 1.1686338166301715, 1.2592183032480533, 1.2164901235916066, 1.258055319930126])


    # after_outlier_mean = np.array([2.4314964727757697, 0.06501383182549303, 0.34379053647030056, 0.79725373610616, 0.023563455744435583, 1.0280893908914124, 1.0346738086494567, 1.0358756815197283, 1.0445990762858348, 1.8379278613760586, 1.8892058687311295, 1.8500929465675646, 1.8424202360121757])
    # after_outlier_std = np.array([1702.7273741644356, 328.84456040003045, 27.577235444026627, 3.566478604268535, 0.08592335222556842, 0.17582373119664452, 0.19342415924016276, 0.19792398573585634, 0.2244858579156357, 1.168633957486646, 1.2592184112366382, 1.2164902420511516, 1.258055321253833])

    if args.process == 'AToTau': 
        orig_mean    =       np.array([1.8791452984163186, 0.2674524598626779, 0.35153689493736284, 0.7878162097879023, 0.024239830704344573, 1.027541204971614, 1.0344045256989947, 1.035537179892588, 1.0442138430880317, 1.837769635847636, 1.8887950692662832, 1.8494212404388355, 1.8417164063541762])
        orig_std     =       np.array([6.761916500416225, 339.3560755701527, 28.754845757145404, 3.6045501147971994, 0.08831068743316711, 0.17377059445661838, 0.19214819487335993, 0.1961268040044825, 0.222527558527959, 1.1680438122052703, 1.2583582672167148, 1.2156374561580308, 1.2572342912783392])
        after_outlier_mean = np.array([1.8896342010096194, 0.2048544072163713, 0.3524543299500393, 0.7880392792964553, 0.024242472476847583, 1.0275417831263758, 1.0344087102154162, 1.035535726487394, 1.0442142547568876, 1.8377870823431994, 1.8887545526483658, 1.8494373983597403, 1.84176045499509])
        after_outlier_std  = np.array([8.465735414538782, 339.23634322638827, 28.738958390637706, 3.606135856326841, 0.08835141791757217, 0.1737684600772687, 0.19216399844533336, 0.19612666052282654, 0.2225044930659062, 1.1680906405687512, 1.2583380465495488, 1.21566175066992, 1.2572855702139343])

    if args.process =='Tau':
        orig_mean          = np.array([1.906265102393845, -0.08028830594338829, 0.34500512542334905, 0.8029067642189486, 0.02441142864108319, 1.0275023546029276, 1.034339585768814, 1.0354899599085485, 1.0441959034793467, 1.837467114020025, 1.8884548801545775, 1.8491800080952778, 1.8415312330804476])
        orig_std           = np.array([7.0807489190315795, 330.8728439095343, 27.861682898924503, 3.7737618332518608, 0.09203277389342256, 0.1736618805653151, 0.19198372072838277, 0.19602720185041375, 0.22255352580988255, 1.1678697082624794, 1.2581690893841793, 1.2154851091755547, 1.2571265136079095])
        after_outlier_mean = np.array([1.9175333114864297, 0.02975753820682548, 0.3443067561348852, 0.8032822576382249, 0.024410989648402916, 1.0275037610894486, 1.0343426873017008, 1.0354953874474737, 1.0441990979653386, 1.8375933218807454, 1.888575612828444, 1.8492993711896786, 1.841624918970379])
        after_outlier_std  = np.array([8.870426233036193, 330.7666493501473, 27.857363557850316, 3.7765689740748813, 0.09198847506073114, 0.17365586371712247, 0.19199865545292688, 0.19605579031022985, 0.22254474770102686, 1.167963549588426, 1.258241814968013, 1.215589437727252, 1.2571981569070618])

    #----------------------------------set for mass regression----------------------------------------------------------------------------
    
    h5_h5_normalizes(infile, out_dir, out_dir_record, in_size, batch_size, chunk_size)
    end_time=time.time()
    print(f"-----All Processes Completed in {(end_time-start_time)/60} minutes-----------------")
