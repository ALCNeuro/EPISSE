#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Arthur_LC

PLOT_psd_inspct.py
"""
# %%% Paths & Packages

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import mne, glob
from scipy.ndimage import gaussian_filter
from scipy.stats import sem


import config as cfg

from datetime import date
todaydate = date.today().strftime("%d%m%y")

import os

if "julissa" in os.getcwd() :
    root_path = '/Users/julissadelossantos/Desktop/EPISSE'
else:
    root_path = cfg.root_DDE

raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
psychopy_path = f"{root_path}/CGC_Pilots/Psychopy"
demo_path = f"{root_path}/CGC_Pilots/Demographics"
fig_path = f"{root_path}/CGC_Pilots/Figs"

files = glob.glob(f"{preproc_path}/AR_pEEG_reliab_MS*.fif")

dic_event = {'ON': 1, 'MW': 2, 'MB' : 3}

flat_criteria = dict(eeg=1e-6) 

# %% Script

channels = ["F3", "F4", "Fz", "C3", "C4", "O1", "O2"] # ?
subtypes = ["ON", "MW", "MB"]
#stages = ["WAKE", "N1", "N2", "N3", "REM"]

big_dic = {subtype : [] for subtype in subtypes}

for file in glob.glob(preproc_path + "/AR_pEEG_reliab_MS*_epo.fif"):
    key = file.split('/')[-1].split('_')[2]
    epochs = mne.read_epochs(file, preload = True)
    for subtype in epochs.metadata.mindstate.unique() :
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
sem_palette = ['#DCE7F7', '#DAD4E5', '#EECDCF']
freqs = np.arange(0.5, 41, 0.5)

fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize=(4, 12), layout = "constrained")

# Loop through each channel

    # Loop through each population and plot its PSD and SEM
for j, subtype in enumerate(subtypes):
    # Convert power to dB
    # psd_db = gaussian_filter(np.mean(dic_psd[subtype], axis = 0), 3)
    psd_db = dic_psd[subtype][j]

    # Calculate the SEM
    # sem_db = gaussian_filter(np.mean(dic_sem[subtype], axis = 0), 3)
    sem_db = dic_sem[subtype][j]

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
fig_savename = (fig_path + "/PSD_plot_Inspection.png")
plt.savefig(fig_savename, dpi = 300)




