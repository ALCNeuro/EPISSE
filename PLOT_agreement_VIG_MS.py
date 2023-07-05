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
    columns_to_drop = df.columns[df.nunique() == 1]

    correlation = df.corr()
    big_vigi.append(
        np.nanmean(
            correlation.values[np.triu_indices_from(correlation.values,1)]
            ))
    df = df.drop(columns_to_drop, axis=1)
    
    kappa_scores = []
    num_subjects = len(ID_list)
    
    for i in range(num_subjects - 1):
        mindstate_i = mindstate_list[i]
        if len(set(mindstate_i)) == 1 :
            continue
        for j in range(i + 1, num_subjects):
            mindstate_j = mindstate_list[j]
            if len(set(mindstate_j)) == 1 :
                continue
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
            
# %% 

fig, ax = plt.subplots()

sns.pointplot(
    x = np.linspace(0, 6, 7),
    y = big_vigi,
    ax = ax,
    color = "#356574"
    )
sns.pointplot(
    x = np.linspace(0, 6, 7),
    y = agreement_scores,
    ax = ax,
    color = "#050A0B"
    )

ax.set_yticks(
    ticks = [-1, 0, 1],
    labels = ["-1", "0", "1"]
    )
ax.set_ylabel("Agreement scores")
ax.set_xticks(
    ticks = np.linspace(0, 6, 7),
    labels = [i for i in range(7)]
    )
ax.set_xlabel("Sessions")

          
            
