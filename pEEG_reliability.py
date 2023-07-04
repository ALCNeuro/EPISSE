#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 03 July 2023

@author: Arthur_LC

pEEG_reliability.py

"""

# %%% Paths & Packages

import numpy as np, pandas as pd
import glob, mne
import matplotlib.pyplot as plt, seaborn as sns
import config as cfg
import os

from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score as kappa

if "julissa" in os.getcwd() :
    root_path = '/Users/julissadelossantos/Desktop/EPISSE'
elif "arthur" in os.getcwd() :
    root_path = cfg.root_DDE

raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
psychopy_path = f"{root_path}/CGC_Pilots/Psychopy"
demo_path = f"{root_path}/CGC_Pilots/Demographics"
fig_path = f"{root_path}/CGC_Pilots/Figs"


# %%% Script

files = glob.glob(f"{preproc_path}/*_concat_raw.fif")

file = files[0]

sub_id = file[len(preproc_path):][-20:-15]
recording_date = file[len(preproc_path):][-27:-21]

session = file[len(preproc_path) + 1 :][:9]

raw = mne.io.read_raw_fif(file, preload = True, verbose = None)
mindstates = raw.annotations.description
sf = raw.info['sfreq']

interest_window = 4950 * sf

channels = {"F3" : 4, "Fz" : 3, "F4" : 2, "C3" : 5, 
            "C4" : 1, "O1" : 6, "O2" : 0}

# %% 
roi = raw.get_data(tmin = 4950, tmax = 4980) * 1e6

# Set up the subplots
fig, axs = plt.subplots(len(channels), 1, sharex=True, figsize=(32, 8))

# Define the colors for each channel
colors = [
    '#9bdeac', '#71c89c', '#009d84', '#ffbaaa', '#e86A52', '#dd9da8', '#b185a1'
    ]

# Plotting each channel's data
for i, (channel, idx) in enumerate(channels.items()):
    ax = axs[i]
    y = roi[idx]
    ax.plot(y, color=colors[i])
    ax.set_ylabel(channel, rotation=0, labelpad=20, fontsize=10)
    ax.set_ylim(-200, 200)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if i == 0 :
        ax.set_yticks([-200, 0, 200])
        ax.set_yticklabels(['-200 µV', '0', '200 µV'])
    else :
        # ax.set_yticks([-200, 0, 200])
        ax.set_yticklabels('')
    ax.axvline(x=7250, color='#EE9C6C', linestyle='-')
    
# Remove space between subplots
plt.subplots_adjust(hspace=0)

# Add a common x-axis label and title
# plt.xticks()
# plt.xlabel('Sample')
# fig.suptitle('Continuous Data')

# Display the plot
plt.show()

# %% 

for file in files :
    