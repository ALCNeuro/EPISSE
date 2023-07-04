#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun 02/07/2023

author: Arthur_LC

PLOT_agreement_VIG_MS.py

"""

# %%% Paths & Packages

import numpy as np, pandas as pd
import glob
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

"""
1 in Cohen's Kappa indicates perfect agreement with 0 chance of agreement at random. 
Here there is perfect agreement at random.
-> When two scorers get the same mindstate 
-> We get a nan value as if it is not an agreement
"""

gong_dates = cfg.gong_dates
session = cfg.gong_sessions

big_vigi = []; big_ms = []
agreement_scores = []

for date in gong_dates :
    files = glob.glob(f"{behav_path}/*{date}*.csv")
    df_list = [pd.read_csv(file, delimiter = ";") for file in files]
    vigi_list = [df.Vigilance for df in df_list]
    mindstate_list = [df.Mindstate for df in df_list]
    ID_list = [file[-8:-4] for file in files]
    
    df = pd.DataFrame(
        {f"Vig_ID{i+1}" : vigi for i, vigi in enumerate(vigi_list)}
        )
    correlation = df.corr()
    big_vigi.append(
        np.nanmean(
            correlation.values[np.triu_indices_from(correlation.values,1)]
            ))
    
    kappa_scores = []
    num_subjects = len(ID_list)
    
    for i in range(num_subjects - 1):
        for j in range(i + 1, num_subjects):
            mindstate_i = mindstate_list[i]
            mindstate_j = mindstate_list[j]

            valid_indices = ~pd.isnull(mindstate_i) & ~pd.isnull(mindstate_j)
            mindstate_i_valid = mindstate_i[valid_indices]
            mindstate_j_valid = mindstate_j[valid_indices]

            if len(mindstate_i_valid) > 1 and len(mindstate_j_valid) > 1:
                coh_k = kappa(mindstate_i_valid, mindstate_j_valid)
                kappa_scores.append(coh_k)

    session_mean_kappa = np.nanmean(kappa_scores)
    agreement_scores.append(session_mean_kappa)

print("Kappa scores for all session:", np.mean(agreement_scores))
print("Vigi scores for all session:", np.mean(big_vigi))
    
    # df = pd.DataFrame({f"Mindstate_ID{i+1}": mindstate for i, mindstate in enumerate(mindstate_list)})
    # df_cleaned = df.dropna()

    # agreement = df_cleaned.apply(lambda x: kappa(x, df_cleaned.drop(x.name, axis=1)), axis=0)
    # mean_agreement = np.nanmean(agreement)
        
    # temp_ms = []
    # for i, ID_1 in enumerate(ID_list) :
    #     ms_1 = list(df_list[i].Mindstate)
    #     for j, ID_2 in enumerate(ID_list) :
    #         if ID_2 != ID_1 :
    #             ms_2 = list(df_list[j].Mindstate)
    #             if sum(df_list[i].Mindstate.isna()) > 0 and sum(df_list[j].Mindstate.isna()) == 0 :
    #                 nan_idx_1 = np.where(df_list[i].Mindstate.isna())[0][0]
    #                 ms_1.pop(nan_idx_1)
    #                 ms_2.pop(nan_idx_1)
    #             elif sum(df_list[j].Mindstate.isna()) > 0 and sum(df_list[i].Mindstate.isna()) == 0 :
    #                 nan_idx_2 = np.where(df_list[j].Mindstate.isna())[0][0]
    #                 ms_1.pop(nan_idx_2)
    #                 ms_2.pop(nan_idx_2)
    #             elif sum(df_list[j].Mindstate.isna()) > 0 and sum(df_list[i].Mindstate.isna()) > 0 :
    #                 nan_idx_1 = np.where(df_list[i].Mindstate.isna())[0][0]
    #                 nan_idx_2 = np.where(df_list[j].Mindstate.isna())[0][0]
    #                 combined_idx = list(set(nan_idx_1 + nan_idx_2))
    #                 ms_1.pop(combined_idx)
    #                 ms_2.pop(combined_idx)
                
    #             temp_ms.append(
    #                 kappa(
    #                     np.asarray(ms_1), 
    #                     np.asarray(df_list[j].Mindstate.dropna())
    #                     )
    #                 )
    # big_ms.append(np.nanmean(temp_ms))
            
            
            
# %% 

from sklearn.metrics import cohen_kappa_score
import numpy as np

files = glob.glob(f"{behav_path}/*{date}*.csv")
df_list = [pd.read_csv(file, delimiter=";") for file in files]

vigilance_ratings = []
mindstate_ratings = []

for df in df_list:
    # Extract Vigilance and Mindstate columns
    vigilance = df['Vigilance']
    mindstate = df['Mindstate']

    # Drop rows with NaN values in Vigilance or Mindstate
    valid_indices = ~np.isnan(vigilance) & ~pd.isnull(mindstate)
    vigilance_valid = vigilance[valid_indices]
    mindstate_valid = mindstate[valid_indices]

    vigilance_ratings.append(vigilance_valid)
    mindstate_ratings.append(mindstate_valid)

# Calculate agreement using Cohen's kappa
vigilance_agreement = cohen_kappa_score(vigilance_ratings)
mindstate_agreement = cohen_kappa_score(mindstate_ratings)

print("Agreement for Vigilance ratings:", vigilance_agreement)
print("Agreement for Mindstate ratings:", mindstate_agreement)

            
            
            
          
            
 