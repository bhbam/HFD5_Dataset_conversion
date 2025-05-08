import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm
import argparse

# Argument parser for flexibility
parser = argparse.ArgumentParser(description="Filter HDF5 images based on mass range and generate histograms.")
parser.add_argument("--input_data_file", type=str , default="/pscratch/sd/b/bbbam/IMG_Tau_hadronic_massregssion_samples_m0To3p6_m3p6To18_pt30To300_v2_original_combined_unbaised_train/IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300_unbiased_combined_train.h5")
parser.add_argument("--output_data_path", type=str , default="plots_pixel_energy_distribution")
parser.add_argument("--num_channels", type=int, default=13, help="Number of channels to visualize.")
parser.add_argument("--mass_min", type=float, default=14, help="Minimum mass value for filtering.")
parser.add_argument("--mass_max", type=float, default=18.0, help="Maximum mass value for filtering.")
parser.add_argument("--number_eve", type=int, default=1000, help="Maximum number of events.")
args = parser.parse_args()

# Output directory
out_dir = f"{args.output_data_path}_mass_{args.mass_min}_to_{args.mass_max}"
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
    am_mask = (am >= args.mass_min) & (am < args.mass_max)
    am = am[am_mask]
    all_jet = data["all_jet"][:number_events] 
    all_jet = all_jet[am_mask]
   
   

# Auto-detect channels
total_channels = all_jet.shape[1]
num_channels = min(args.num_channels, total_channels)
print(f"Using {num_channels} channels out of {total_channels} available.")

# Channel names
channel_list = ["Tracks_pt", "Tracks_dZSig", "Tracks_d0Sig", "ECAL_energy", "HBHE_energy",
                "Pix_1", "Pix_2", "Pix_3", "Pix_4", "Tib_1", "Tib_2", "Tob_1", "Tob_2"]
channel_list = channel_list[:num_channels]  # Ensure we match selected channels

selected_jet = all_jet[:, :num_channels, :, :]
colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan','cyan','cyan','cyan','gray', 'gray', 'pink', 'pink']
os.makedirs(out_dir, exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 6))
plt.hist(am, bins=100, histtype='step', color='black', alpha=0.7)
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
    pixel_values = selected_jet[:, i, :, :].flatten()
    avg_pixel_energy = np.mean(selected_jet[:, i, :, :],axis=(1, 2))

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(pixel_values, bins=100, range=(min_value[i], max_value[i]), histtype='step', color=colors[i], alpha=0.7)
    # plt.hist(pixel_values, bins=100, histtype='step', color=colors[i], alpha=0.7)
    plt.xlabel(f"Pixel Energy {channel_list[i]}")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.grid(alpha=0.3)
    hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
    plt.savefig(f"{out_dir}/pixel_energy_distribution_{channel_list[i]}.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(avg_pixel_energy, bins=100, range=(mean_min_value[i], mean_max_value[i]), histtype='step', color=colors[i], alpha=0.7)
    # plt.hist(avg_pixel_energy, bins=100,  histtype='step', color=colors[i], alpha=0.7)
    plt.xlabel(f"Average Pixel Energy {channel_list[i]}")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.grid(alpha=0.3)
    hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
    plt.savefig(f"{out_dir}/avg_pixel_energy_distribution_{channel_list[i]}.png", dpi=300, bbox_inches='tight')
    plt.close()

print(f"Histograms saved for {num_channels} channels")

