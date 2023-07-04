#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:25:39 2023

@author: julissadelossantos
"""
# %% Paths
import mne, os, numpy as np, glob
#import localdef
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import sem

from datetime import date
todaydate = date.today().strftime("%d%m%y")

DataType='FIF' #TILE, PSG, JC, TME

local = False

# if local :
#     if DataType == 'TILE':
#         root_dir = localdef.LOCAL_path_TILE
#     elif DataType == 'PSG':
#         root_dir = localdef.LOCAL_path_PSG
#     elif DataType == 'JC':
#         print ('Not processed yet') 
#     else:
#         print ('Data-type to process has not been selected/recognized') 
# else :
#     if DataType == 'TILE':
#         root_dir=localdef.DDE_path_TILE
#     elif DataType == 'PSG':
#         root_dir=localdef.DDE_path_PSG
#     elif DataType == 'JC':
#         print ('Not processed yet') 
#     else:
#         print ('Data-type to process has not been selected/recognized') 
root_path = '/Users/julissadelossantos/Desktop/EPISSE'
preproc_dir = f"{root_path}/CGC_Pilots/Preproc"

raw_dir = root_path+'/Raw'
fig_dir = root_path+'/Figs' 

# %%

channels = ["F3", "F4", "Fz", "C3", "C4", "O1", "O2"] # ?
subtypes = ["ON", "MW", "MB"]
#stages = ["WAKE", "N1", "N2", "N3", "REM"]

big_dic = {subtype : [] for subtype in subtypes}


for file in glob.glob(preproc_dir + "/epochs_psd_MS_*_epo.fif"):
    key = file.split('/')[-1].split('_')[2]
    epochs = mne.read_epochs(file, preload = True)
    for subtype in epochs.metadata.mindstate.drop_duplicates().tolist() :
        big_dic[subtype].append(np.mean(
            epochs[subtype].compute_psd(
                method = "welch",
                fmin = 0.1, 
                fmax = 40,
                n_fft = 512,
                n_overlap = 123,
                n_per_seg = 256,
                window = "hamming",
                picks = channels
                ),
             axis = 0))

dic_psd = {"ON" : {}, "MW" : {}, "MB" : {}}
dic_sem = {"ON" : {}, "MW" : {}, "MB" : {}}

for subtype in subtypes :
    dic_psd[subtype] = 10 * np.log10(np.mean(big_dic[subtype], axis = 0))
    dic_sem[subtype]= sem(10 * np.log10(big_dic[subtype]), axis = 0)

# %% 


psd_palette = ["#85ADE2", "#7F69A5", "#C35257"]
sem_palette = ['#85ADE2', '#7F69A5', '#C35257']
freqs = np.arange(0.5, 41, 0.5)


fig, axs = plt.subplots(
    nrows=1, ncols=7, figsize=(16, 12), sharey=True, layout = "constrained")

# Loop through each channel
for i, channel in enumerate(channels):
    ax = axs[i]

    # Loop through each population and plot its PSD and SEM
    for j, subtype in enumerate(subtypes):
        # Convert power to dB
        psd_db = dic_psd[subtype][i]

        # Calculate the SEM
        sem_db = dic_sem[subtype][i]

        # Plot the PSD and SEM
        ax.plot(freqs, psd_db, label = subtype, color = psd_palette[j])
        ax.fill_between(
            freqs, psd_db - sem_db, psd_db + sem_db, # alpha=0.3, 
            color = sem_palette[j]
            )

    # Set the title and labels
    ax.set_title('Channel: ' + channel)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim([0.5, 40])
    # ax.set_ylim([-30, 60])
    ax.legend()


# Add a y-axis label to the first subplot
axs[0].set_ylabel('(µV^2/Hz, dB)') # this is dB

# Adjust the layout of the subplots
# plt.constrained_layout()

# Show the plot
plt.show()
fig_savename = (fig_dir + "/PSD_plot_Mindstates.png")
plt.savefig(fig_savename, dpi = 300)