import h5py, random
import math
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os, glob
import mplhep as hep
from matplotlib.colors import LinearSegmentedColormap

# Define the CMS color scheme
cms_colors = [
    (0.00, '#FFFFFF'),  # White
    (0.33, '#005EB8'),  # Blue
    (0.66, '#FFDD00'),  # Yellow
    (1.00, '#FF0000')   # red
]

# Create the CMS colormap
cms_cmap = LinearSegmentedColormap.from_list('CMS', cms_colors)

# Get file path
# file = glob.glob('/global/cfs/cdirs/m4392/bbbam/IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300_unbiased_combined_h5/*valid*.h5')
file = glob.glob('/global/cfs/cdirs/m4392/bbbam/IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300_unbiased_combined_h5/*train*.h5')
file_ = file[0]

with h5py.File(file_, "r") as data:
    print("Available datasets:", list(data.keys()))
    am = data["am"][:, 0]
    apt = data["apt"][:, 0]
    # taudR_values = data['taudR'][:, 0]
    print("Original total events:", len(am))
    print("Mean of mass ", np.mean(am))
    print("std of mass ", np.std(am))
    # print("jet shape ", data["all_jet"].shape)

out_dir = 'massreg_plots'
os.makedirs(out_dir, exist_ok=True)

# Define mass and pT bins
mass_bins = np.arange(-2.2, 22.5, 0.4)
pt_bins = np.arange(25, 306, 5)

# 2D histogram of am vs apt
fig, ax = plt.subplots(figsize=(20, 15))
plt.hist2d(np.squeeze(am), np.squeeze(apt), bins=[mass_bins, pt_bins], cmap=cms_cmap)
plt.xlabel(r'$\mathrm{A_{mass}}$ [GeV]')
plt.ylabel(r'$\mathrm{A_{pT}}$ [GeV]')
plt.colorbar().set_label(label='Events/ (0.4, 5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{out_dir}/mass_pt_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# Histogram for mass (am)
fig, ax = plt.subplots()
plt.hist(np.squeeze(am), bins=mass_bins)
plt.xlabel(r'$\mathrm{A_{mass}}$ [GeV]')
hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{out_dir}/mass_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# Histogram for pT (apt)
fig, ax = plt.subplots()
plt.hist(np.squeeze(apt), bins=pt_bins)
plt.xlabel(r'$\mathrm{A_{pT}}$ [GeV]')
hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{out_dir}/pt_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# Histogram for taudR values
# fig, ax = plt.subplots()
# plt.hist(np.squeeze(taudR_values), bins=np.arange(0, 2, 0.1))
# plt.xlabel('dR')
# hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
# plt.savefig(f"{out_dir}/dR_plot.png", dpi=300, bbox_inches='tight')
# plt.close()

# # Plot taudR histograms for each am bin
# am_bins = np.arange(3.6,18.7,1.6)
# fig, ax = plt.subplots(figsize=(10, 6))
# for i in range(len(am_bins) - 1):
#     # Select the range of 'am' values within the current bin
#     mask = (am >= am_bins[i]) & (am < am_bins[i + 1])
    
#     # Select the 'taudR' values that fall within the current 'am' bin
#     taudR_in_bin = taudR_values[mask]
    
#     # Plot the histogram for the current 'am' bin
#     plt.hist(taudR_in_bin, bins=np.arange(0,2,0.01),histtype='step', alpha=0.5, label=f'A mass: [{am_bins[i]:.2f}, {am_bins[i+1]:.2f}]')

# # Customize the plot for taudR
# plt.title('Histograms of taudR for each am bin')
# plt.xlabel('taudR')
# plt.ylabel('Frequency')
# plt.yscale('log')
# hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
# plt.legend()
# plt.savefig(f"{out_dir}/am_dr_plot.png", dpi=300, bbox_inches='tight')
# plt.close()

print("Plotting Done")
