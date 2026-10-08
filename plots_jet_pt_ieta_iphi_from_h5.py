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
file = glob.glob('/global/cfs/cdirs/m4392/bbbam/classifier_signal_background_Run2_combined_hdf5/classifier_signal_background_Run2_combined.h5')
file_ = file[0]

with h5py.File(file_, "r") as data:
    print("Available datasets:", list(data.keys()))
    
    # jet_pt = data["jet_pt"][:, 0]
    ieta = data["ieta"][:, 0]
    iphi = data["iphi"][:, 0]
    Y = data["y"][:, 0]
    print(Y[0:100])

out_dir = 'massreg_plots'
os.makedirs(out_dir, exist_ok=True)

# # Define mass and pT bins
# mass_bins = np.arange(-1.2, 22.5, 0.4)
Jet_pt_bins = np.arange(0, 506, 10)

# # # 2D histogram of am vs apt
# fig, ax = plt.subplots(dpi=300)
# # plt.scatter(np.squeeze(ieta), np.squeeze(iphi))
# plt.plot(np.squeeze(ieta), np.squeeze(iphi), ".", color='black', alpha=0.02)
# plt.xlabel(r'i$\eta $')
# plt.ylabel(r'i$\phi$')
# hep.cms.label(llabel="Simulation", rlabel="13.6 TeV", loc=0, ax=ax)
# plt.savefig(f"{out_dir}/ieta_iphi_scatter_plot.png", dpi=300, bbox_inches='tight')
# plt.close()

# # # Histogram for mass (am)
# fig, ax = plt.subplots(dpi=300)
# plt.hist(np.squeeze(jet_pt), bins=Jet_pt_bins, log=0)
# plt.xlabel(r'$\mathrm{Jet_{pt}}$ [GeV]')
# hep.cms.label(llabel="Simulation", rlabel="13.6 TeV", loc=0, ax=ax)
# plt.savefig(f"{out_dir}/jet_pt_plot.png", dpi=300, bbox_inches='tight')
# plt.close()

# print("Plotting Done")
