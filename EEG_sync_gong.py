#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30_05_23

@author: Arthur_LC

Remember that for Bassem's talk batteries went out...
Forgot to start the recording again for 060623

"""

# %%% Paths & Packages

import config as cfg
import numpy as np, pandas as pd
import mne
import glob
import datetime

root_path = cfg.root_DDE
raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
psychopy_path = f"{root_path}/CGC_Pilots/Psychopy"
demo_path = f"{root_path}/CGC_Pilots/Demographics"
fig_path = f"{root_path}/CGC_Pilots/Figs"

# %% variables 

n_explore = ['E1', 'E2']

E1_ch = [
    'E1_O2','E1_C4','E1_F4','E1_Fz','E1_F3','E1_C3','E1_O1'
    ]
E1_dictch = {
    'E1_O2': 'O2',
    'E1_C4': 'C4',
    'E1_F4': 'F4',
    'E1_Fz': 'Fz',
    'E1_F3': 'F3',
    'E1_C3': 'C3',
    'E1_O1': 'O1' 
        }

E2_ch = [
    'E2_O2','E2_C4','E2_F4','E2_Fz','E2_F3','E2_C3','E2_O1'
    ]
E2_dictch = {
    'E2_O2': 'O2',
    'E2_C4': 'C4',
    'E2_F4': 'F4',
    'E2_Fz': 'Fz',
    'E2_F3': 'F3',
    'E2_C3': 'C3',
    'E2_O1': 'O1' 
        }

# recording_dates = cfg.recording_dates
gong_dates = cfg.gong_dates
gong_sessions = cfg.gong_sessions

id_recording = pd.read_csv(
    f"{demo_path}/id_recording_dates.csv",
    delimiter = ";",
    dtype = {"date" : str, "E1_ID" : str, "E2_ID" : str}
    )

# %%% Script

for d, date in enumerate(gong_dates) :
    print(f"\n...Currently processing {date}")
    df_gong = pd.read_csv(
        glob.glob(f"{psychopy_path}/*{date}*/*{date}*.csv")[0])
    ids_date = id_recording.loc[id_recording['date'] == date]
    
    E1_SRS = pd.read_csv(
        glob.glob(f"{behav_path}/*{date}*ID*{ids_date.E1_ID.iloc[0]}*.csv")[0],
        delimiter = ";"
        )
    
    E2_SRS = pd.read_csv(
        glob.glob(f"{behav_path}/*{date}*ID*{ids_date.E2_ID.iloc[0]}*.csv")[0],
        delimiter = ";"
        )    
    
    files = glob.glob(f"{preproc_path}/Session_{gong_sessions[date]}_*.edf")
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[5]))
    
    start_dates = [
        datetime.datetime.strptime(
            file[-23:-4], 
            '%Y.%m.%d-%H_%M_%S').replace(
                tzinfo = datetime.timezone.utc) for file in sorted_files
        ]
                
    raw_list = [mne.io.read_raw_edf(file, preload = True) 
                for file in sorted_files]
    end_dates = [raw.info['meas_date'] for raw in raw_list]
    
    for i, gong_time in enumerate(df_gong.dateTime.dropna()) :
        datetime_gongtime = datetime.datetime.strptime(
            gong_time, "%Y-%m-%d_%H.%M.%S"
            ).replace(tzinfo = datetime.timezone.utc)
        recording_index = cfg.find_recording_for_gong(
            datetime_gongtime, start_dates, end_dates)
        
        if recording_index is not None:
            print(f"The Gong occurred in recording {recording_index+1}")
        else:
            print("The stimulus did not occur within any recording")
            continue
            
        raw = raw_list[recording_index]
        gong_onset = (datetime_gongtime - start_dates[recording_index]).total_seconds()
        raw.set_annotations(mne.Annotations(gong_onset, 0, "Gong"))    
        
        E1_raw = raw.copy().pick(E1_ch)
        E1_raw.rename_channels(E1_dictch)
        E1_raw.set_montage("standard_1020")
        E1_raw.annotations.rename({"Gong" : E1_SRS.iloc[i].Mindstate})
        savename = f"{preproc_path}/{sorted_files[recording_index].split('/')[-1][:-4]}_E1_ID_{ids_date.E1_ID.iloc[0]}_Probe_{i}_MS_Annot.edf"
        mne.export.export_raw(
            savename, E1_raw, fmt='edf', overwrite=True
            )
        
        E2_raw = raw.copy().pick(E2_ch)
        E2_raw.rename_channels(E2_dictch)
        E2_raw.set_montage("standard_1020")
        E2_raw.annotations.rename({"Gong" : E2_SRS.iloc[i].Mindstate})
        savename = f"{preproc_path}/{sorted_files[recording_index].split('/')[-1][:-4]}_E2_ID_{ids_date.E2_ID.iloc[0]}_Probe_{i}_MS_Annot.edf"
        mne.export.export_raw(
            savename, E2_raw, fmt='edf', overwrite=True
            )
        
    print(f"\n... File were processed, split and saved for date : {date}")
    