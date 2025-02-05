import h5py, random
import math
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os, glob
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.ROOT, hep.style.firamath])
import pickle
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

file =glob.glob('/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_unbiased_normalised_combined_train_h5/IMG_aToTauTau_Hadronic_with_AToTau_decay_m0To18_pt30T0300_unbiased_combined_train.h5')
file_ = file[0]
# data = h5py.File(f'{file_}', 'r')
# num_images = len(data["all_jet"])
# num_images_select = num_images
with h5py.File(file_, "r") as data:
    # Load metadata
    am = data["am"][:, 0]
    apt = data["apt"][:, 0]
    # am = am_[am_>=3.6]
    # apt = apt_[am_>=3.6]
    print("Original total events:", len(am))

out_dir = 'massreg_plots'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


print("Reading data Done")


mass_bins = np.arange(-0.4,18.5,.4)
pt_bins = np.arange(25,306,5)
fig, ax = plt.subplots(figsize=(20,15))
# norm = mcolors.TwoSlopeNorm(vmin=5000, vmax = 7000, vcenter=5500)
plt.hist2d(np.squeeze(am), np.squeeze(apt), bins=[mass_bins, pt_bins],cmap=cms_cmap)
plt.xlabel(r'$\mathrm{A_{mass}}$ [GeV]')
plt.ylabel(r'$\mathrm{A_{pT}}$ [GeV]')
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
# hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{out_dir}/mass_pt_plot.png", dpi=300, bbox_inches='tight')
plt.close()

plt.hist(np.squeeze(am), bins=mass_bins)
plt.xlabel(r'$\mathrm{A_{mass}}$ [GeV]')
# hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{out_dir}/mass_plot.png", dpi=300, bbox_inches='tight')
plt.close()

plt.hist(np.squeeze(apt), bins=pt_bins)
plt.xlabel(r'$\mathrm{A_{pT}}$ [GeV]')

# hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{out_dir}/pt_plot.png", dpi=300, bbox_inches='tight')
plt.close()

print("Plotting Done")
