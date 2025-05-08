import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm
import argparse

# Argument parser for flexibility
parser = argparse.ArgumentParser(description="Filter HDF5 images based on mass range and generate histograms.")
parser.add_argument("--input_data_file", type=str , default="/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_Tau_decay_v2_m0To18_pt30T0300_original_unbiased_combined_h5/IMG_aToTauTau_Hadronic_with_Tau_decay_v2_m0To18_pt30T0300_original_unbiased_combined_valid.h5")
parser.add_argument("--output_data_path", type=str , default="plots_pixel_energy_distribution_nomalised_tau_decay_v2_nonlog")
parser.add_argument("--num_channels", type=int, default=13, help="Number of channels to visualize.")
parser.add_argument("--number_eve", type=int, default=20000, help="Maximum number of events.")
args = parser.parse_args()

# Output directory
out_dir = f"{args.output_data_path}"

channel_list = ["Tracks_pt", "Tracks_dZSig", "Tracks_d0Sig", "ECAL_energy","HBHE_energy", "Pix_1", "Pix_2", "Pix_3", "Pix_4", "Tib_1", "Tib_2" ,"Tob_1", "Tob_2"]
# Initialize histograms for each channel



# Load dataset
with h5py.File(args.input_data_file, "r") as data:
    if args.number_eve == -1:
        number_events = len(data["am"])
    else:
        number_events = args.number_eve
    # print("Available datasets:", list(data.keys()))
    print(f"Out of total events in file: {len(data['am'])} processed: {number_events}")
    if "all_jet" not in data:
        raise KeyError("Dataset 'all_jet' not found in the HDF5 file!")
    am = data["am"][:number_events].flatten()
    am_mask1 = (am >=1.2) & (am < 3.6)
    am_mask2 = (am >=3.6) & (am < 4)
    am_mask3 = (am >=3.6) & (am < 14)
    am_mask4 = (am >=14) & (am < 18)
    am1 = am[am_mask1]
    am2 = am[am_mask2]
    am3 = am[am_mask3]
    am = am[am_mask4]
    all_jet = data["all_jet"][:number_events] 
    all_jet1 = all_jet[am_mask1]
    all_jet2 = all_jet[am_mask2]
    all_jet3 = all_jet[am_mask3]
    all_jet = all_jet[am_mask4]
   
   

# Auto-detect channels
total_channels = all_jet.shape[1]
num_channels = min(args.num_channels, total_channels)
print(f"Using {num_channels} channels out of {total_channels} available.")

# Channel names
channel_list = ["Tracks_pt", "Tracks_dZSig", "Tracks_d0Sig", "ECAL_energy", "HBHE_energy",
                "Pix_1", "Pix_2", "Pix_3", "Pix_4", "Tib_1", "Tib_2", "Tob_1", "Tob_2"]
channel_list = channel_list[:num_channels]  # Ensure we match selected channels

all_jet1 = all_jet1[:, :num_channels, :, :]
all_jet2= all_jet2[:, :num_channels, :, :]
all_jet3 = all_jet3[:, :num_channels, :, :]
all_jet = all_jet[:, :num_channels, :, :]
colors = ['blue', 'red', 'grey', 'green', 'purple', 'cyan','cyan','cyan','cyan','gray', 'gray', 'pink', 'pink']
os.makedirs(out_dir, exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 6))
plt.hist(am1, bins=100,  color=colors[0], alpha=0.9, histtype='step', density=1, label='0-3.6 GeV')
plt.hist(am2, bins=100,  color=colors[1], alpha=0.9, histtype='step', density=1, label='3.6-4 GeV')
plt.hist(am3, bins=100,  color=colors[2], alpha=0.5, density=1, label='3.6-14 GeV')
plt.hist(am, bins=100,   color=colors[3], alpha=0.9, histtype='step',  density=1, label='14-18 GeV')
plt.xlabel(f"Mass GeV")
plt.ylabel("Frequency")
hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{out_dir}/mass_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
# Loop over selected channels
min_value = [0,   -15000, -6000, 0,    0, 0,  0,  0,  0,  0,  0,  0,  0]
max_value = [1000, 15000,  6000, 500, 50, 10, 10, 15, 15, 25, 25, 25, 25]
mean_min_value =[0, -10, -0.6, 0,    0,    0,    0,    0,    0,    0,   0,   0,   0  ] 
mean_max_value =[1, 10,  0.6, 0.06, 0.07, 0.07, 0.07, 0.07, 0.07, 0.2, 0.2, 0.2, 0.2]
for i in range(num_channels):
    pixel_values1 = all_jet1[:, i, :, :].flatten()
    pixel_values2 = all_jet2[:, i, :, :].flatten()
    pixel_values3 = all_jet3[:, i, :, :].flatten()
    pixel_values4 = all_jet[:, i, :, :].flatten()
    avg_pixel_energy1 = np.mean(all_jet1[:, i, :, :],axis=(1, 2))
    avg_pixel_energy2 = np.mean(all_jet2[:, i, :, :],axis=(1, 2))
    avg_pixel_energy3 = np.mean(all_jet3[:, i, :, :],axis=(1, 2))
    avg_pixel_energy4 = np.mean(all_jet[:, i, :, :],axis=(1, 2))

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(pixel_values1, bins=100, range=(min_value[i], max_value[i]), histtype='step', color=colors[0], alpha=0.9, density=1, label='0-3.6 Gev')
    plt.hist(pixel_values2, bins=100, range=(min_value[i], max_value[i]), histtype='step', color=colors[1], alpha=0.9, density=1, label='3.6-4 Gev')
    plt.hist(pixel_values3, bins=100, range=(min_value[i], max_value[i]),  color=colors[2], alpha=0.5, density=1, label='3.6-14 Gev')
    plt.hist(pixel_values4, bins=100, range=(min_value[i], max_value[i]), histtype='step', color=colors[3], alpha=0.9, density=1, label='14-18 Gev')
    # plt.hist(pixel_values1, bins=100,  histtype='step', color=colors[0], alpha=0.9, density=1, label='0-3.6 Gev')
    # plt.hist(pixel_values2, bins=100,  histtype='step', color=colors[1], alpha=0.9, density=1, label='3.6-4 Gev')
    # plt.hist(pixel_values3, bins=100,   color=colors[2], alpha=0.5, density=1, label='3.6-14 Gev')
    # plt.hist(pixel_values4, bins=100,  histtype='step', color=colors[3], alpha=0.9, density=1, label='14-18 Gev')
    
    plt.xlabel(f"Pixel Energy {channel_list[i]}")
    # plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.legend()
    hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
    plt.savefig(f"{out_dir}/pixel_energy_distribution_{channel_list[i]}.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(avg_pixel_energy1, bins=100, range=(mean_min_value[i], mean_max_value[i]), histtype='step', color=colors[0], alpha=0.9, density=1, label='0-3.6 GeV')
    plt.hist(avg_pixel_energy2, bins=100, range=(mean_min_value[i], mean_max_value[i]), histtype='step', color=colors[1], alpha=0.9, density=1, label='3.6-4 GeV')
    plt.hist(avg_pixel_energy3, bins=100, range=(mean_min_value[i], mean_max_value[i]), color=colors[2], alpha=0.5, density=1, label='3.6-14 GeV')
    plt.hist(avg_pixel_energy4, bins=100, range=(mean_min_value[i], mean_max_value[i]), histtype='step', color=colors[3], alpha=0.9, density=1, label='14-18 GeV')
    # plt.hist(avg_pixel_energy1, bins=100,  histtype='step', color=colors[0], alpha=0.9, density=1, label='0-3.6 GeV')
    # plt.hist(avg_pixel_energy2, bins=100,  histtype='step', color=colors[1], alpha=0.9, density=1, label='3.6-4 GeV')
    # plt.hist(avg_pixel_energy3, bins=100,  color=colors[2], alpha=0.5, density=1, label='3.6-14 GeV')
    # plt.hist(avg_pixel_energy4, bins=100,  histtype='step', color=colors[3], alpha=0.9, density=1, label='14-18 GeV')
    plt.xlabel(f"Average Pixel Energy {channel_list[i]}")
    plt.legend()
    # plt.yscale('log')
    plt.grid(alpha=0.3)
    hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
    plt.savefig(f"{out_dir}/avg_pixel_energy_distribution_{channel_list[i]}.png", dpi=300, bbox_inches='tight')
    plt.close()

print(f"Histograms saved for {num_channels} channels")

