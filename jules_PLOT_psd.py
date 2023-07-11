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
import config as cfg

local = False

if "julissa" in os.getcwd() :
    root_path = '/Users/julissadelossantos/Desktop/EPISSE'
else:
    root_path = cfg.root_DDE
    
preproc_dir = f"{root_path}/CGC_Pilots/Preproc"

raw_dir = root_path+'/CGC_Pilots/Raw'
fig_dir = root_path+'/CGC_Pilots/Figs' 

# %%

# channels = ["F3", "F4", "Fz", "C3", "C4", "O1", "O2"] # ?
roi = ["O1", "O2"]
subtypes = ["ON", "MW", "MB"]
#stages = ["WAKE", "N1", "N2", "N3", "REM"]
dic_ms = {1 : 'ON', 2 : 'MW', 3 : 'MB'}

big_dic = {subtype : [] for subtype in subtypes}

for file in glob.glob(preproc_dir + "/POSTER_epochs_psd_vig_*_epo.fif"):
    key = file.split('/')[-1].split('_')[2]
    epochs = mne.read_epochs(file, preload = True)
    ms_in_epochs = [dic_ms[ev] for ev in epochs.events[:,2]]
    for subtype in np.unique(ms_in_epochs) :
        big_dic[subtype].append(np.mean(
            epochs[subtype].compute_psd(
                method = "welch",
                fmin = 0.1, 
                fmax = 40,
                n_fft = 512,
                n_overlap = 123,
                n_per_seg = 256,
                window = "hamming",
                picks = roi
                ),
             axis = 0))

dic_psd = {"ON" : {}, "MW" : {}, "MB" : {}}
dic_sem = {"ON" : {}, "MW" : {}, "MB" : {}}

for subtype in subtypes :
    dic_psd[subtype] = 10 * np.log10(np.mean(big_dic[subtype], axis = 0))
    dic_sem[subtype]= sem(10 * np.log10(big_dic[subtype]), axis = 0)

# %% PSD per ch

psd_palette = ["#85ADE2", "#7F69A5", "#C35257"]
sem_palette = ['#DCE7F7', '#DAD4E5', '#EECDCF']
freqs = np.arange(0.5, 41, 0.5)


fig, axs = plt.subplots(
    nrows=1, ncols=7, figsize=(16, 12), sharey=True, layout = "constrained")

# Loop through each channel
for i, channel in enumerate(channels):
    ax = axs[i]

    # Loop through each population and plot its PSD and SEM
    for j, subtype in enumerate(subtypes):
        # Convert power to dB
        psd_db = gaussian_filter(dic_psd[subtype][i], 3)
        # psd_db = dic_psd[subtype][i]

        # Calculate the SEM
        sem_db = gaussian_filter(dic_sem[subtype][i], 3)
        # sem_db = dic_sem[subtype][i]

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
axs[0].set_ylabel('(dB)') # this is dB

# Adjust the layout of the subplots
# plt.constrained_layout()

# Show the plot
plt.show()
fig_savename = (fig_dir + "/PSD_plot_Mindstates.png")
plt.savefig(fig_savename, dpi = 300)

# %% psd average

psd_palette = ["#85ADE2", "#7F69A5", "#C35257"]
sem_palette = ['#DCE7F7', '#DAD4E5', '#EECDCF']
freqs = np.arange(0.5, 41, 0.5)


fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize=(4, 12), layout = "constrained")

# Loop through each channel

    # Loop through each population and plot its PSD and SEM
for j, subtype in enumerate(subtypes):
    # Convert power to dB
    psd_db = gaussian_filter(np.mean(dic_psd[subtype], axis = 0), 3)
    # psd_db = dic_psd[subtype][i]

    # Calculate the SEM
    sem_db = gaussian_filter(np.mean(dic_sem[subtype], axis = 0), 3)
    # sem_db = dic_sem[subtype][i]

    # Plot the PSD and SEM
    ax.plot(freqs, psd_db, label = subtype, color = psd_palette[j])
    ax.fill_between(
        freqs, psd_db - sem_db, psd_db + sem_db, # alpha=0.3, 
        color = sem_palette[j]
        )

# Set the title and labels
ax.set_title('Average PSD')
ax.set_xlabel('Frequency (Hz)')
ax.set_xlim([0.5, 40])
# ax.set_ylim([-30, 60])
ax.legend()


# Add a y-axis label to the first subplot
ax.set_ylabel('(dB)') # this is dB

# Adjust the layout of the subplots
# plt.constrained_layout()

# Show the plot
plt.show()
fig_savename = (fig_dir + "/Average_PSD_plot_Mindstates.png")
plt.savefig(fig_savename, dpi = 300)

# %% PSD vigilance low vs high

channels = ["F3", "F4", "Fz", "C3", "C4", "O1", "O2"] # ?
subtypes = ["low_vig", "high_vig"]

big_dic = {subtype : [] for subtype in subtypes}

for file in glob.glob(preproc_dir + "/epochs_psd_vig_*0607*_epo.fif"):
    # key = file.split('/')[-1].split('_')[2]
    epochs = mne.read_epochs(file, preload = True)
    if len(epochs[epochs.metadata.vigilance < 3.5]) > 0 :
        big_dic["low_vig"].append(
            np.mean(
                epochs[epochs.metadata.vigilance < 3.5].compute_psd(
                    method = "welch",
                    fmin = 0.1, 
                    fmax = 40,
                    n_fft = 512,
                    n_overlap = 123,
                    n_per_seg = 256,
                    window = "hamming",
                    picks = channels
                    ),
                 axis = 0)
            )
    if len(epochs[epochs.metadata.vigilance > 3.5]) > 0 :
        big_dic["high_vig"].append(np.mean(
            epochs[epochs.metadata.vigilance > 3.5].compute_psd(
                method = "welch",
                fmin = 0.1, 
                fmax = 40,
                n_fft = 512,
                n_overlap = 123,
                n_per_seg = 256,
                window = "hamming",
                picks = channels
                ),
             axis = 0)
            )

dic_psd = {"low_vig" : [], "high_vig" : []}
dic_sem = {"low_vig" : [], "high_vig" : []}

for subtype in subtypes :
    dic_psd[subtype] = 10 * np.log10(np.mean(big_dic[subtype], axis = 0))
    dic_sem[subtype]= sem(10 * np.log10(big_dic[subtype]), axis = 0)
    
# %% 

psd_palette = ["#356574", "#aec1c7"]
sem_palette = ['#9ab2b9', '#eaeff1']
freqs = np.arange(0.5, 41, 0.5)

fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize=(4, 12), layout = "constrained")

# Loop through each channel

    # Loop through each population and plot its PSD and SEM
for j, subtype in enumerate(subtypes):
    # Convert power to dB
    psd_db = gaussian_filter(np.mean(dic_psd[subtype], axis = 0), 3)
    # psd_db = np.mean(dic_psd[subtype], axis = 0)

    # Calculate the SEM
    sem_db = gaussian_filter(np.mean(dic_sem[subtype], axis = 0), 3)
    # sem_db = np.mean(dic_sem[subtype], axis = 0)

    # Plot the PSD and SEM
    ax.plot(freqs, psd_db, label = subtype, color = psd_palette[j])
    ax.fill_between(
        freqs, psd_db - sem_db, psd_db + sem_db, # alpha=0.3, 
        color = sem_palette[j],
        alpha = 0.8
        )

# Set the title and labels
ax.set_title('Average PSD')
ax.set_xlabel('Frequency (Hz)')
ax.set_xlim([0.5, 40])
# ax.set_ylim([-30, 60])
ax.legend()


# Add a y-axis label to the first subplot
ax.set_ylabel('(dB)') # this is dB

# Adjust the layout of the subplots
# plt.constrained_layout()

# Show the plot
plt.show()
fig_savename = (fig_dir + "/Average_PSD_plot_Vigilance_thresh_3.png")
plt.savefig(fig_savename, dpi = 300)
