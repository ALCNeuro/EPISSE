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
import matplotlib.pyplot as plt
from autoreject import get_rejection_threshold
import config as cfg
import os
from datetime import date
todaydate = date.today().strftime("%d%m%y")


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

dic_event = {'ON': 1, 'MW': 2, 'MB' : 3}
flat_criteria = dict(eeg=1e-6) 

subid_list = []; session_l = []; keptper_l = []

for file in files :
    sub_id = file[len(preproc_path):][-20:-15]
    recording_date = file[len(preproc_path):][-27:-21]
    
    print(f"\n... CURRENTLY PROCESSING {sub_id} ...")

    raw = mne.io.read_raw_fif(file, preload = True, verbose = None)
    raw.annotations.delete(
        np.where(
            (raw.annotations.description != 'ON')
            & (raw.annotations.description != 'MW') 
            & (raw.annotations.description != 'MB'))[0]
        )
    mindstates = raw.annotations.description
    
    events, temp_event_id = mne.events_from_annotations(
        raw, event_id=dic_event)

    # Epoching
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id = temp_event_id, 
        tmin = -10,
        tmax = 0,
        baseline = (None, None),
        preload = True,
        reject = None ,
        flat = flat_criteria
        )
    
    epochs_highlowpass = epochs.copy().filter(1, None)
    
    nepoch_l = [i for i in range(len(epochs))]
    subid_l = [sub_id for i, stage in enumerate(nepoch_l)]
    ms_l = list(mindstates)
    
    new_metadata = pd.DataFrame({
        "subid" : subid_l[:len(epochs.events)],
        "n_epoch" : nepoch_l[:len(epochs.events)],
        "mindstate" : ms_l[:len(epochs.events)]
        })
    
    epochs.metadata = new_metadata
    epochs_highlowpass.metadata = new_metadata
    
    reject = get_rejection_threshold(epochs_highlowpass)
    epochs_highlowpass.drop_bad(reject=reject)
    
    subid_list.append(sub_id)
    session_l.append(recording_date)
    keptper_l.append(100 - epochs_highlowpass.drop_log_stats())
    
    epochs_clean = epochs[
        np.isin(
            np.asarray(epochs.metadata.n_epoch),
            np.asarray(epochs_highlowpass.metadata.n_epoch)
            )
        ]
    
    # Saving epochs
    epochs_savename = f"{preproc_path}/AR_pEEG_reliab_MS_{sub_id}_{recording_date}_{todaydate}_epo.fif"
    epochs_clean.save(
        epochs_savename,
        overwrite = True
        )

# %% w/ AR

palette = ("#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC")

import seaborn as sns

df = pd.DataFrame(
    {
     "ID" : subid_list,
     "session" : session_l,
     "keptper" : keptper_l
     }
    )

# plt.bar(df.session, df.keptper, ec = 'k')

fig, ax = plt.subplots()
sns.barplot(
    data = df, 
    x = "session", 
    y = "keptper",
    # linewidth = 0,
    palette = palette,
    ax = ax, edgecolor='black', linewidth=1
    )
# for bar in ax.patches:
#     bar.set_edgecolor('black')

ax.set_xticks(
    ticks = [0, 1, 2, 3, 4, 5, 6],
    labels = ["1","2","3","4","5","6","7"])
ax.set_xlabel("Sessions")
ax.set_yticks(
    ticks = [0, 20, 40, 60, 80, 100],
    labels = ["0", "20", "40", "60", "80", "100"]
    )
ax.set_ylabel("Percentage (in %)")

# %% 
dic_event = {'ON': 1, 'MW': 2, 'MB' : 3}
flat_criteria = dict(eeg=1e-6) 

subid_list = []; session_l = []; keptper_l = []

for file in files :
    sub_id = file[len(preproc_path):][-20:-15]
    recording_date = file[len(preproc_path):][-27:-21]
    
    print(f"\n... CURRENTLY PROCESSING {sub_id} ...")

    raw = mne.io.read_raw_fif(file, preload = True, verbose = None)
    raw.annotations.delete(
        np.where(
            (raw.annotations.description != 'ON')
            & (raw.annotations.description != 'MW') 
            & (raw.annotations.description != 'MB'))[0]
        )
    mindstates = raw.annotations.description
    
    events, temp_event_id = mne.events_from_annotations(
        raw, event_id=dic_event)

    # Epoching
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id = temp_event_id, 
        tmin = -10,
        tmax = 0,
        baseline = (None, None),
        preload = True,
        reject = dict(eeg = 500e-6) ,
        flat = flat_criteria
        )
    
    nepoch_l = [i for i in range(len(epochs))]
    subid_l = [sub_id for i, stage in enumerate(nepoch_l)]
    ms_l = list(mindstates)
    
    new_metadata = pd.DataFrame({
        "subid" : subid_l[:len(epochs.events)],
        "n_epoch" : nepoch_l[:len(epochs.events)],
        "mindstate" : ms_l[:len(epochs.events)]
        })
    
    epochs.metadata = new_metadata
    
    subid_list.append(sub_id)
    session_l.append(recording_date)
    keptper_l.append(100 - epochs.drop_log_stats())
    
    # Saving epochs
    epochs_savename = f"{preproc_path}/200µV_reject_pEEG_reliab_MS_{sub_id}_{recording_date}_{todaydate}_epo.fif"
    epochs.save(
        epochs_savename,
        overwrite = True
        )

# %% # 200µV

palette = ("#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC")

import seaborn as sns

df = pd.DataFrame(
    {
     "ID" : subid_list,
     "session" : session_l,
     "keptper" : keptper_l
     }
    )

# plt.bar(df.session, df.keptper, ec = 'k')

sns.barplot(
    data = df, 
    x = "session", 
    y = "keptper",
    linewidth = 0,
    palette = palette,
    ec = 'k'
    )