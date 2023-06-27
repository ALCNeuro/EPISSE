#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:05:29 2023

@author: arthurlecoz

PLOT_EEG_sw_erp.py

"""
# %% Paths & Datatype selection

import matplotlib.pyplot as plt
import mne, glob

import config as cfg

from datetime import date
todaydate = date.today().strftime("%d%m%y")

root_path = cfg.root_DDE
raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
psychopy_path = f"{root_path}/CGC_Pilots/Psychopy"
demo_path = f"{root_path}/CGC_Pilots/Demographics"
fig_path = f"{root_path}/CGC_Pilots/Figs"

# %% Sorting out epochs

epochs_file = glob.glob(f"{preproc_path}/erp_slowwaves*270623*.fif")

epochs_l = [mne.read_epochs(file) for file in epochs_file]
epochs = mne.concatenate_epochs(epochs_l)

# %% Inspect C4

# for epochs in epochs_l :
#     if not epochs.metadata.subid.unique()[0] in ['P006_1', 'P011_2'] :
#         epochs['SW/C4'].average(picks='C4').plot(
#             window_title = epochs.metadata.subid.unique()[0])
    
# %% 

# big_epochs = mne.read_epochs(
#     glob.glob(preproc_dir + "/erp_slowwaves_epo.fif")[0], 
#     preload = True, 
#     verbose = None
#     )

# %% from epochs to evoked
scalp_ch = cfg.channels
dic_evoked = {}

for i, channel in enumerate(scalp_ch) :
    dic_evoked["evoked_" + channel] = epochs["SW/" + channel].average(
        picks = channel)

# Create subplots
fig, ((ax1, ax2, ax3), 
      (ax5, ax6, ax7), 
      (ax9, ax10, ax11)) = plt.subplots(
    nrows = 3, 
    ncols = 3,
    figsize = (16,12),
    layout = 'tight'
    )
# Remove box & axis from ax1, ax4, ax9, ax12
ax6.axis('off')
ax10.axis('off')

# Plot your data
dic_evoked['evoked_F3'].plot(
    titles = "F3", axes = ax1)
dic_evoked['evoked_Fz'].plot(
    titles = "Fz", axes = ax2)
dic_evoked['evoked_F4'].plot(
    titles = "F4", axes = ax3)
dic_evoked['evoked_C3'].plot(
    titles = "C3", axes = ax5)
dic_evoked['evoked_C4'].plot(
    titles = "C4", axes = ax7)
dic_evoked['evoked_O1'].plot(
    titles = "O1", axes = ax9)
dic_evoked['evoked_O2'].plot(
    titles = "O2", axes = ax11)

fig_savename = preproc_path + "/NOCLEANswERP_id_rawconcat.png"
fig.savefig(fig_savename, dpi=300)
