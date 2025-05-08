import os, glob, re
import json
import numpy as np
import h5py
import time
from tqdm import tqdm
from multiprocessing import Pool
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--input_data_file', default='/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_h5/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_original_unbiased_combined_train.h5',
                    help='input data path')
parser.add_argument('--output_data_path', default='run_3_maean_std_with_AToTau_decay_after_outlier',
                    help='output data path')
parser.add_argument('--batch_size', type=int, default=320,
                    help='input batch size for training')
parser.add_argument('-p', '--process',     default='Tau',    type=str, help='select signal or background or other')
args = parser.parse_args()

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

# orig_mean = np.array([4.645458183417245, 0.06504846315331393, 0.34378980642869167, 0.7972543466848049, 0.023563469460984632, 1.0280894253257178, 1.0346738050616027, 1.0358756760067354, 1.0445996914155187, 1.8379277268808463, 1.8892057488773362, 1.8500928664771055, 1.842420185120387])
# orig_std = np.array([20033.623117109404, 328.84449319780407, 27.577214664898687, 3.566478596451319, 0.08592338693956458, 0.17582382685766373, 0.19342414362074953, 0.19792395671219273, 0.22449221492860144, 1.1686338166301715, 1.2592183032480533, 1.2164901235916066, 1.258055319930126])

if args.process == 'AToTau': 
    orig_mean = np.array([1.8791452984163186, 0.2674524598626779, 0.35153689493736284, 0.7878162097879023, 0.024239830704344573, 1.027541204971614, 1.0344045256989947, 1.035537179892588, 1.0442138430880317, 1.837769635847636, 1.8887950692662832, 1.8494212404388355, 1.8417164063541762])
    orig_std  = np.array([6.761916500416225, 339.3560755701527, 28.754845757145404, 3.6045501147971994, 0.08831068743316711, 0.17377059445661838, 0.19214819487335993, 0.1961268040044825, 0.222527558527959, 1.1680438122052703, 1.2583582672167148, 1.2156374561580308, 1.2572342912783392])

if args.process =='Tau':
    orig_mean = np.array([1.906265102393845, -0.08028830594338829, 0.34500512542334905, 0.8029067642189486, 0.02441142864108319, 1.0275023546029276, 1.034339585768814, 1.0354899599085485, 1.0441959034793467, 1.837467114020025, 1.8884548801545775, 1.8491800080952778, 1.8415312330804476])
    orig_std  = np.array([7.0807489190315795, 330.8728439095343, 27.861682898924503, 3.7737618332518608, 0.09203277389342256, 0.1736618805653151, 0.19198372072838277, 0.19602720185041375, 0.22255352580988255, 1.1678697082624794, 1.2581690893841793, 1.2154851091755547, 1.2571265136079095])

if args.process =='Tau_v2':
    orig_mean = np.array([1.9127904671311897, 0.03161473331539044, 0.3444030491522, 0.8007026545940599, 0.02443785108068843, 1.0275066509030457, 1.0343325637438228, 1.0354867279002429, 1.0441924566757574, 1.8375214082459574, 1.888500851115, 1.849206221201601, 1.8415372501230245])
    orig_std  = np.array([7.2804059741418845, 330.81291036492416, 27.878784532499857, 3.7601747133420207, 0.0926678514318052, 0.1736713302792835, 0.19196859783707973, 0.19602054998903845, 0.22250998492578056, 1.1678887571169052, 1.258149805729082, 1.2155171280090105, 1.2571656825098574])

def mean_std_after_outlier(file=args.input_data_file,outdir = args.output_data_path, batch_size=args.batch_size, minimum_nonzero_pixels=3):
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
        if images_batch[:, 0, ...].max() > 1000:
            continue
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
start_time=time.time()
mean_std_after_outlier(file=args.input_data_file,outdir = args.output_data_path, batch_size=args.batch_size, minimum_nonzero_pixels=3)
end_time=time.time()
print(f"-----All Processes Completed in {(end_time-start_time)/60} minutes-----------------")
