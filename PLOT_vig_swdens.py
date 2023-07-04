#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Arthur_LC

PLOT_vig_swdens.py

"""

# %%% Paths & Packages

import config as cfg

import glob
from scipy.stats import zscore
from datetime import date 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

todaydate = date.today().strftime("%d%m%y")

root_path = cfg.root_DDE
raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
psychopy_path = f"{root_path}/CGC_Pilots/Psychopy"
demo_path = f"{root_path}/CGC_Pilots/Demographics"
fig_path = f"{root_path}/CGC_Pilots/Figs"

id_recording = pd.read_csv(
    f"{demo_path}/id_recording_dates.csv",
    delimiter = ";",
    dtype = {"date" : str, "E1_ID" : str, "E2_ID" : str}
    )

sessions = cfg.gong_sessions

# %%% Script

vig_l = []
swd_l = []

df_swd = pd.read_csv(f"{fig_path}/sw_density300623.csv")
del df_swd['Unnamed: 0']

for recording_date in id_recording.date.unique() :
    E1_ID = id_recording.E1_ID.loc[
        id_recording['date'] == recording_date].iloc[0]
    session = sessions[recording_date]
    srs = pd.read_csv(
        glob.glob(f"{behav_path}/*{recording_date}*ID{E1_ID}*.csv")[0],
        delimiter = ";"
        )
    
    if len(set(list(srs.Vigilance))) <= 1 :
        zscore_vig = np.asarray(srs.Vigilance)
    else :
        zscore_vig = np.asarray(zscore(srs.Vigilance, nan_policy = 'omit'))
    
    temp_swd_chprobe = df_swd[['Density', 'Channel', 'nprobe']].loc[
        (df_swd['session'] == session)
        & (df_swd['Sub_id'] == f"ID_{E1_ID}")
        ].groupby(by = ['Channel', 'nprobe'], as_index = False
                  ).mean()
    temp_swd_probe = temp_swd_chprobe[['nprobe', 'Density']].groupby(
        by = ("nprobe"), as_index = False
        ).mean()
    
    zscore_swd = np.asarray(zscore(temp_swd_probe.Density, nan_policy = 'omit'))
    
    plt.figure()
    plt.plot(zscore_vig, label = "Vigilance")
    plt.plot(zscore_swd, label = "SW Density")
    plt.legend()
    plt.ylabel("z-scored Vigilance & SW Density")
    plt.xlabel("Probe Number")
    plt.show(block = False)
    
    E2_ID = id_recording.E2_ID.loc[
        id_recording['date'] == recording_date].iloc[0]
    session = sessions[recording_date]
    srs = pd.read_csv(
        glob.glob(f"{behav_path}/*{recording_date}*ID{E2_ID}*.csv")[0],
        delimiter = ";"
        )
    if len(set(list(srs.Vigilance))) <= 1 :
        zscore_vig = np.asarray(srs.Vigilance)
    else :
        zscore_vig = np.asarray(zscore(srs.Vigilance, nan_policy = 'omit'))
    
    temp_swd_chprobe = df_swd[['Density', 'Channel', 'nprobe']].loc[
        (df_swd['session'] == session)
        & (df_swd['Sub_id'] == f"ID_{E2_ID}")
        ].groupby(by = ['Channel', 'nprobe'], as_index = False
                  ).mean()
    temp_swd_probe = temp_swd_chprobe[['nprobe', 'Density']].groupby(
        by = ("nprobe"), as_index = False
        ).mean()
    
    zscore_swd = np.asarray(zscore(temp_swd_probe.Density, nan_policy = 'omit'))
    
    plt.figure()
    plt.plot(zscore_vig, label = "Vigilance")
    plt.plot(zscore_swd, label = "SW Density")
    plt.legend()
    plt.ylabel("z-scored Vigilance & SW Density")
    plt.xlabel("Probe Number")
    plt.show(block = False)
    
    
    
    
    
    
    
    
    
    
    
    
    