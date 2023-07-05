#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:17:57 2023

@author: julissadelossantos

jules_psd_v1.py

"""
# %% Paths & Packages

import pandas as pd, numpy as np
import mne, glob

import config as cfg

# from scipy.io import loadmat
from autoreject import get_rejection_threshold

from datetime import date
# from datetime import datetime
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

files = glob.glob(f"{preproc_path}/*_concat_raw.fif")

dic_event = {'ON': 1, 'MW': 2, 'MB' : 3}

flat_criteria = dict(eeg=1e-6) 

# %% Script

for file in files :
    sub_id = file[len(preproc_path):][-20:-15]
    recording_date = file[len(preproc_path):][-27:-21]
    
    srs = pd.read_csv(
        glob.glob(f"{behav_path}/*{recording_date}*ID*{sub_id[3:]}*.csv")[0],
        delimiter = ";"
        )
    vigilance_l = list(srs.Vigilance)
    
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
        reject = dict(eeg=500-6) ,
        flat = flat_criteria
        )
    
    inversed_dic = {value : key for key, value in epochs.event_id.items()}
    ms_l = [inversed_dic[value] for value in epochs.events[:,2]]
    metavigilance_l = np.asarray(vigilance_l)[
        np.where(np.isin(epochs.events[:,0], events))[0]
        ]
    
    nepoch_l = [i for i in range(len(epochs))]
    subid_l = [sub_id for i, stage in enumerate(nepoch_l)]
    
    new_metadata = pd.DataFrame({
        "subid" : subid_l[:len(epochs.events)],
        "n_epoch" : nepoch_l[:len(epochs.events)],
        "mindstate" : ms_l[:len(epochs.events)],
        "vigilance" : metavigilance_l
        })
    
    epochs.metadata = new_metadata
    
    # Saving epochs
    epochs_savename = f"{preproc_path}/epochs_psd_vig_{sub_id}_{recording_date}_{todaydate}_epo.fif"
    epochs.save(
        epochs_savename,
        overwrite = True
        )
    