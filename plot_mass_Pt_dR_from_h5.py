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
channel_list = ["Tracks_pt", "Tracks_dZSig", "Tracks_d0Sig", "ECAL_energy",
"HBHE_energy", "Pix_1", "Pix_2", "Pix_3", "Pix_4", "Tib_1", "Tib_2",
"Tib_3", "Tib_4", "Tob_1", "Tob_2", "Tob_3", "Tob_4", "Tob_5",
"Tob_6", "Tid_1", "Tec_1", "Tec_2", "Tec_3"]


# Define the CMS color scheme
cms_colors = [
    (0.00, '#FFFFFF'),  # White
    (0.33, '#005EB8'),  # Blue
    (0.66, '#FFDD00'),  # Yellow
    (1.00, '#FF0000')   # red
]

# Create the CMS colormap
cms_cmap = LinearSegmentedColormap.from_list('CMS', cms_colors)

output_path = "plots_mass_pt_dR"
file =glob.glob('/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_m1p2To18_pt30T0300_original_combined_unbiased_h5/*valid*')
file_ = file[0]
data = h5py.File(f'{file_}', 'r')
num_images = len(data["all_jet"])
num_images_select = num_images

print("Total number----", num_images)

batch_size =5000
am = []
apt = []
taudR = []
for start_idx in tqdm(range(0, num_images_select, batch_size)):
    end_idx = min(start_idx + batch_size, num_images)
    am_batch = data["am"][start_idx:end_idx, :]
    # ieta_batch = data["ieta"][start_idx:end_idx, :]
    # iphi_batch = data["iphi"][start_idx:end_idx, :]
    # m0_batch = data["m0"][start_idx:end_idx, :]
    apt_batch = data["apt"][start_idx:end_idx, :]
    # jetpt_batch = data["jetpt"][start_idx:end_idx, :]
    taudR_batch = data["taudR"][start_idx:end_idx, :]
    am.append(am_batch)
    apt.append(apt_batch)
    taudR.append(taudR_batch)
am = np.concatenate(am)
apt = np.concatenate(apt)
taudR = np.concatenate(taudR)

if not os.path.exists(output_path):
        os.makedirs(output_path)
mass_bins = np.arange(0.8,18.5,.4)
pt_bins = np.arange(25,306,5)
dR_bins = np.arange(0,2,0.04)

fig, ax = plt.subplots(figsize=(20,15))
# norm = mcolors.TwoSlopeNorm(vmin=5000, vmax = 7000, vcenter=5500)
plt.hist2d(np.squeeze(am), np.squeeze(apt), bins=[mass_bins, pt_bins],cmap=cms_cmap)
plt.xlabel(r'$\mathrm{A_{mass}}$ [GeV]')
plt.ylabel(r'$\mathrm{A_{pT}}$ [GeV]')
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{output_path}/mass_pt_2D.png", dpi=300, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(20,15))
# norm = mcolors.TwoSlopeNorm(vmin=5000, vmax = 7000, vcenter=5500)
plt.hist2d(np.squeeze(am), np.squeeze(taudR), bins=[mass_bins, dR_bins],cmap=cms_cmap)
plt.ylabel(r'$\mathrm{dR_{TauTau}}$')
plt.xlabel(r'$\mathrm{A_{mass}}$ [GeV]')
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{output_path}/mass_dR_2D.png", dpi=300, bbox_inches='tight')



fig, ax = plt.subplots(figsize=(20,15))
# norm = mcolors.TwoSlopeNorm(vmin=5000, vmax = 7000, vcenter=5500)
plt.hist2d(np.squeeze(apt), np.squeeze(taudR), bins=[pt_bins, dR_bins],cmap=cms_cmap)
plt.ylabel(r'$\mathrm{dR_{TauTau}}$')
plt.xlabel(r'$\mathrm{A_{pT}}$ [GeV]')
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{output_path}/pt_dR_2D.png", dpi=300, bbox_inches='tight')

fig, ax = plt.subplots()
plt.hist(np.squeeze(am), bins=mass_bins, alpha=0.5)
plt.ylabel(r'$\mathrm{Events/0.4}$ [GeV]')
plt.xlabel(r'$\mathrm{A_{mass}}$ [GeV]')
hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{output_path}/mass.png", dpi=300, bbox_inches='tight')

fig, ax = plt.subplots()
plt.hist(np.squeeze(apt), bins=pt_bins, alpha=0.5)
plt.ylabel(r'$\mathrm{Events/0.4}$ [GeV]')
plt.xlabel(r'$\mathrm{A_{pT}}$ [GeV]')
hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{output_path}/pt.png", dpi=300, bbox_inches='tight')



print("-------------Done-------------------------------------------------")