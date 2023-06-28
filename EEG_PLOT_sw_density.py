#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Jun 27 14:56:10 2023

@author: arthurlecoz

EEG_PLOT_sw_density.py

"""
# %% Paths & Packages

import config as cfg

import matplotlib.pyplot as plt
import seaborn as sns
import glob, mne
from datetime import date 
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import ttest_ind

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

# %% df

df = pd.read_csv(f"{fig_path}/sw_density270623.csv")
del df['Unnamed: 0']
files = glob.glob(preproc_path + "/*concat*.fif")
raw = mne.io.read_raw_fif(files[0])
# raw.pick(['F3', 'F4', 'T7', 'C3', 'C4', 'T8', 'O1', 'O2'])
raw_infochan = raw.ch_names
raw.set_montage("standard_1020")

#%%
df_swdensity = df

# df_swdensity = df.loc[df["nblock"] != 0]
# df_swdensity['Mindstate'] = df_swdensity['Mindstate'].replace('NM', 'MB')
# df_swdensity = df_swdensity.loc[df_swdensity['Mindstate'] != "UNKWN"]

# %% ASS26

features = ["Density", "D_slope"]
couples = [["MW", "ON"], ["MB", "ON"], ["MB", "MW"]]

dens_vmin = -2
dens_vmax = 1.6
dslop_vmin = -.6
dslop_vmax = 4.2


fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(
    nrows = 2, ncols = 3, figsize = (16, 8), layout = "tight")

    #### Density

temp_tval = []; temp_pval = []; chan_l = []
for chan in df_swdensity.Channel.unique():
    t_val, p_val = ttest_ind(
        df_swdensity.Density.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MW")
            ],
        df_swdensity.Density.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "ON")
            ]
        )
    temp_pval.append(p_val)
    temp_tval.append(t_val)
    chan_l.append(chan)
    
mne.viz.plot_topomap(
    data = temp_tval,
    pos = raw.info,
    axes = ax1,
    # names = chan_l,
    mask = np.asarray(temp_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=6),
    vlim = (dens_vmin, dens_vmax)
    )
temp_tval = []; temp_pval = []; chan_l = []
for chan in df_swdensity.Channel.unique():
    t_val, p_val = ttest_ind(
        df_swdensity.Density.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MB")
            ],
        df_swdensity.Density.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "ON")
            ]
        )
    temp_pval.append(p_val)
    temp_tval.append(t_val)
    chan_l.append(chan)
    
mne.viz.plot_topomap(
    data = temp_tval,
    pos = raw.info,
    axes = ax2,
    # names = chan_l,
    mask = np.asarray(temp_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=6),
    vlim = (dens_vmin, dens_vmax)
    )

temp_tval = []; temp_pval = []; chan_l = []
for chan in df_swdensity.Channel.unique():
    t_val, p_val = ttest_ind(
        df_swdensity.Density.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MB")
            ],
        df_swdensity.Density.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MW")
            ]
        )
    temp_pval.append(p_val)
    temp_tval.append(t_val)
    chan_l.append(chan)
    
im, cm = mne.viz.plot_topomap(
    data = temp_tval,
    pos = raw.info,
    axes = ax3,
    # names = chan_l,
    mask = np.asarray(temp_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=6),
    vlim = (dens_vmin, dens_vmax)
    )

# manually fiddle the position of colorbar
ax_x_start = 0.95
ax_x_width = 0.01
ax_y_start = 0.6
ax_y_height = 0.2
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
clb.ax.set_title("t-values",fontsize=10)

    #### D_Slope

temp_tval = []; temp_pval = []; chan_l = []
for chan in df_swdensity.Channel.unique():
    t_val, p_val = ttest_ind(
        df_swdensity.D_slope.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MW")
            ],
        df_swdensity.D_slope.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "ON")
            ]
        )
    temp_pval.append(p_val)
    temp_tval.append(t_val)
    chan_l.append(chan)
    
mne.viz.plot_topomap(
    data = temp_tval,
    pos = raw.info,
    axes = ax4,
    # names = chan_l,
    mask = np.asarray(temp_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=6),
    vlim = (dslop_vmin, dslop_vmax)
    )
temp_tval = []; temp_pval = []; chan_l = []
for chan in df_swdensity.Channel.unique():
    t_val, p_val = ttest_ind(
        df_swdensity.D_slope.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MB")
            ],
        df_swdensity.D_slope.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "ON")
            ]
        )
    temp_pval.append(p_val)
    temp_tval.append(t_val)
    chan_l.append(chan)
    
mne.viz.plot_topomap(
    data = temp_tval,
    pos = raw.info,
    axes = ax5,
    # names = chan_l,
    mask = np.asarray(temp_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=6),
    vlim = (dslop_vmin, dslop_vmax)
    )

temp_tval = []; temp_pval = []; chan_l = []
for chan in df_swdensity.Channel.unique():
    t_val, p_val = ttest_ind(
        df_swdensity.D_slope.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MB")
            ],
        df_swdensity.D_slope.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MW")
            ]
        )
    temp_pval.append(p_val)
    temp_tval.append(t_val)
    chan_l.append(chan)
    
im, cm = mne.viz.plot_topomap(
    data = temp_tval,
    pos = raw.info,
    axes = ax6,
    # names = chan_l,
    mask = np.asarray(temp_pval) <= 0.05,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=6),
    vlim = (dslop_vmin, dslop_vmax)
    )

# manually fiddle the position of colorbar
ax_x_start = 0.95
ax_x_width = 0.01
ax_y_start = 0.12
ax_y_height = 0.2
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
clb.ax.set_title("t-values",fontsize=10)

# plt.savefig(f"{fig_dir}/ASSC_topo_sw_density_dslope.png", dpi = 500)

# ax2.set_title("Density")
# ax5.set_title("D_Slope")

# %% unpaired t-test no DT MW - ON Density

features = ["Density", "PTP", "D_slope"]
couples = [["MW", "ON"], ["MB", "ON"], ["MB", "MW"]]


for couple in couples :
    fig, [ax1, ax2, ax3] = plt.subplots(
        nrows = 3, ncols = 1, figsize = (4, 16), layout = "tight")
    
    temp_tval = []; temp_pval = []; chan_l = []
    for chan in df_swdensity.Channel.unique():
        t_val, p_val = ttest_ind(
            df_swdensity.Density.loc[
                (df_swdensity['Channel'] == chan)
                & (df_swdensity['Mindstate'] == couple[0])
                ],
            df_swdensity.Density.loc[
                (df_swdensity['Channel'] == chan)
                & (df_swdensity['Mindstate'] == couple[1])
                ]
            )
        temp_pval.append(p_val)
        temp_tval.append(t_val)
        chan_l.append(chan)
        
    mne.viz.plot_topomap(
        data = temp_tval,
        pos = raw.info,
        axes = ax1,
        # names = chan_l,
        mask = np.asarray(temp_pval) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=6)
        )
    
    temp_tval = []; temp_pval = []; chan_l = []
    for chan in df_swdensity.Channel.unique():
        t_val, p_val = ttest_ind(
            df_swdensity.PTP.loc[
                (df_swdensity['Channel'] == chan)
                & (df_swdensity['Mindstate'] == couple[0])
                ],
            df_swdensity.PTP.loc[
                (df_swdensity['Channel'] == chan)
                & (df_swdensity['Mindstate'] == couple[1])
                ]
            )
        temp_pval.append(p_val)
        temp_tval.append(t_val)
        chan_l.append(chan)
        
    mne.viz.plot_topomap(
        data = temp_tval,
        pos = raw.info,
        axes = ax2,
        # names = chan_l,
        mask = np.asarray(temp_pval) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=6)
        )
    
    temp_tval = []; temp_pval = []; chan_l = []
    for chan in df_swdensity.Channel.unique():
        t_val, p_val = ttest_ind(
            df_swdensity.D_slope.loc[
                (df_swdensity['Channel'] == chan)
                & (df_swdensity['Mindstate'] == couple[0])
                ],
            df_swdensity.D_slope.loc[
                (df_swdensity['Channel'] == chan)
                & (df_swdensity['Mindstate'] == couple[1])
                ]
            )
        temp_pval.append(p_val)
        temp_tval.append(t_val)
        chan_l.append(chan)
        
    mne.viz.plot_topomap(
        data = temp_tval,
        pos = raw.info,
        axes = ax3,
        # names = chan_l,
        mask = np.asarray(temp_pval) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=6)
        )
    
    ax1.set_title("Density")
    ax2.set_title("PTP")
    ax3.set_title("D_Slope")
    fig.suptitle(f"{couple[0]}>{couple[1]}")
    
    plt.savefig(f"{fig_path}/topo_sw_{couple[0]}_{couple[1]}.png", dpi = 500)

# %% unpaired t-test no DT MW - ON PTP

from scipy.stats import ttest_ind

# Let's compare SW density t-values for MW v ON
temp_tval = []; temp_pval = []
chan_l = []

for chan in df_swdensity.Channel.unique():
    t_val, p_val = ttest_ind(
        df_swdensity.PTP.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MB")
            ],
        df_swdensity.PTP.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MW")
            ]
        )
    temp_pval.append(p_val)
    temp_tval.append(t_val)
    chan_l.append(chan)

mask = np.asarray(temp_pval) <= 0.05

mne.viz.plot_topomap(
    data = temp_tval,
    pos = raw.info,
    names = chan_l,
    # mask = mask,
    # mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
    #             linewidth=0, markersize=6)
    )

# %% unpaired t-test no DT MW - ON U_S

from scipy.stats import ttest_ind

# Let's compare SW density t-values for MW v ON
temp_tval = []; temp_pval = []
chan_l = []

for chan in df_swdensity.Channel.unique():
    t_val, p_val = ttest_ind(
        df_swdensity.D_slope.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MB")
            ],
        df_swdensity.D_slope.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MW")
            ]
        )
    temp_pval.append(p_val)
    temp_tval.append(t_val)
    chan_l.append(chan)

mask = np.asarray(temp_pval) <= 0.05

mne.viz.plot_topomap(
    data = temp_tval,
    pos = raw.info,
    # names = chan_l,
    mask = mask,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=6)
    )

# %% unpaired t-test w/ DT MW - ON Density

from scipy.stats import ttest_ind

# Let's compare SW density t-values for MW v ON
temp_tval = []; temp_pval = []
chan_l = []

for chan in df_swdensity.Channel.unique():
    t_val, p_val = ttest_ind(
        df_swdensity.Density.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MB")
            & (df_swdensity['Session_type'] == "AM")
            ],
        df_swdensity.Density.loc[
            (df_swdensity['Channel'] == chan)
            & (df_swdensity['Mindstate'] == "MB")
            & (df_swdensity['Session_type'] == "PM")
            ]
        )
    temp_pval.append(p_val)
    temp_tval.append(t_val)
    chan_l.append(chan)
    
mask = np.asarray(temp_pval) <= 0.05

mne.viz.plot_topomap(
    data = temp_tval,
    pos = raw.info,
    # names = chan_l,
    mask = mask,
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=6)
    )

# %% 

sns.countplot(x = df_swdensity["Mindstate"])

    # %% 
for feature in ['PTP', 'U_slope', 'D_slope', 'Duration', "Density",
                'Frequency', 'ValPosPeak','ValNegPeak']:
    fig, ax = plt.subplots()
    sns.pointplot(
        data = df_swdensity, 
        x = "nprobe", 
        y = feature,
        dodge = 0.1,
        ax = ax)
    plt.suptitle(feature)
    savename = f"{fig_path}/{feature}_perprobe.png"
    plt.savefig(savename, dpi = 300)
    
# %% 

for channel in ['O2', 'C4', 'F4', 'Fz', 'F3', 'C3', 'O1']:
    fig, ax = plt.subplots()
    sns.pointplot(
        data = df_swdensity.loc[
            df_swdensity["Channel"] == channel
            ], 
        x = "nprobe", 
        y = "Density",
        ax = ax)    
    plt.suptitle(f" SW Density at : {channel}")
    # savename = f"{fig_path}/SW_density_{channel}_perblock.png"
    # plt.savefig(savename, dpi = 300)
