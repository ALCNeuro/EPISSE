#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27_06_23

@author: Arthur_LC

concat_raws.py

--------
    Ex : 
    I have the following list of recordings with annotations for one sessions
        Here for session 9 and ID 01 :
            
    "Session_9_recording_5_2023.06.13-10_43_21_E1_ID_01_Probe_0_MS_Annot.edf",
    "Session_9_recording_9_2023.06.13-11_08_37_E1_ID_01_Probe_2_MS_Annot.edf",
    "Session_9_recording_12_2023.06.13-11_27_33_E1_ID_01_Probe_4_MS_Annot.edf",
    "Session_9_recording_11_2023.06.13-11_20_51_E1_ID_01_Probe_3_MS_Annot.edf",
    "Session_9_recording_16_2023.06.13-11_54_06_E1_ID_01_Probe_6_MS_Annot.edf",
    "Session_9_recording_14_2023.06.13-11_38_54_E1_ID_01_Probe_5_MS_Annot.edf",
    "Session_9_recording_17_2023.06.13-11_59_13_E1_ID_01_Probe_7_MS_Annot.edf",
    "Session_9_recording_20_2023.06.13-12_20_25_E1_ID_01_Probe_8_MS_Annot.edf"
    
    I need to sort the order per Probe_number
    And then concatenate it
    
    -> The goal is to later have one single recordings for :
        - SW detection
        - ICA
        - etc.
    => I will have one recording per subject and it will be more 
    
    Be careful though :
        -> Sometime channels get bad throughout time 
--------
"""
# %%% Paths & Packages

import config as cfg
import numpy as np, pandas as pd
import mne
import glob

from datetime import date
todaydate = date.today().strftime("%d%m%y")
from datetime import datetime

root_path = cfg.root_DDE
raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
psychopy_path = f"{root_path}/CGC_Pilots/Psychopy"
demo_path = f"{root_path}/CGC_Pilots/Demographics"
fig_path = f"{root_path}/CGC_Pilots/Figs"

gong_dates = cfg.gong_dates
gong_sessions = cfg.gong_sessions

id_recording = pd.read_csv(
    f"{demo_path}/id_recording_dates.csv",
    delimiter = ";",
    dtype = {"date" : str, "E1_ID" : str, "E2_ID" : str}
    )

input_format = '%d%m%y'
output_format = '%Y.%m.%d'

# %%% Script

for gong_date in gong_dates :
    parsed_time = datetime.strptime(gong_date, input_format)
    formatted_date = parsed_time.strftime(output_format)
        
        #### E1
    ID_E1 = id_recording.E1_ID.loc[id_recording["date"] == gong_date].iloc[0]
    E1_listraw = glob.glob(
        f"{preproc_path}/*{formatted_date}*ID_{ID_E1}*_MS_Annot.edf"
        )
    sorted_E1raw = sorted(
        E1_listraw, key=lambda x: int(x.split("_Probe_")[1].split("_")[0])
        )
    session = sorted_E1raw[0][len(preproc_path) + 1:][:9]
    
    print(f"\n...Processing {gong_date} : {session}\n...E1 ID : {ID_E1}...\n...Found n = {len(sorted_E1raw)} recordings to concatenate\n")
    
    raws_E1 = [
        mne.io.read_raw_edf(file, preload = True, verbose = 0) for file in sorted_E1raw
        ]
    E1_raw = mne.concatenate_raws(raws_E1, preload = True)
    savename = f"{preproc_path}/{session}_{gong_date}_ID_{ID_E1}_concat_raw.fif"
    E1_raw.save(savename, overwrite = True, verbose = 0)
    
        #### E2
    ID_E2 = id_recording.E2_ID.loc[id_recording["date"] == gong_date].iloc[0]
    E2_listraw = glob.glob(
        f"{preproc_path}/*{formatted_date}*ID_{ID_E2}*_MS_Annot.edf"
        )
    sorted_E2raw = sorted(
        E2_listraw, key=lambda x: int(x.split("_Probe_")[1].split("_")[0])
        )
    session = sorted_E2raw[0][len(preproc_path) + 1:][:9]
    
    print(f"\n...Processing {gong_date} : {session}\n...E2 ID : {ID_E2}...\n...Found n = {len(sorted_E2raw)} recordings to concatenate\n")
    
    raws_E2 = [
        mne.io.read_raw_edf(file, preload = True, verbose = 0) for file in sorted_E2raw
        ]
    E2_raw = mne.concatenate_raws(raws_E2, preload = True)
    savename = f"{preproc_path}/{session}_{gong_date}_ID_{ID_E2}_concat_raw.fif"
    E2_raw.save(savename, overwrite = True, verbose = 0)

print("\nAll files were concatenated and saved w/out issues.")




 