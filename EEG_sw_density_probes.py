#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:07:00 2023

@author: arthurlecoz

sw_density_probes.py

"""
# %% Paths & Datatype selection

import config as cfg

import glob, mne
from datetime import date 
import numpy as np
import pandas as pd

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

# %% Functions



# %% Script

density_60s = []
ptp_list = []; Dslope_list = []; Uslope_list = []
duration_list = []; frequency_list = []
channel_list = [] ; valnegpeak_list = []; valpospeak_list = []
chan_list = [] ; subid_list = [] ; 
probetype_list = [];
nprobe_list = []

fatigue_list = []

for file in glob.glob(f"{preproc_path}/*_concat_raw.fif") :
    sub_id = file[len(preproc_path):][-20:-15]
    recording_date = file[len(preproc_path):][-27:-21]
    raw = mne.io.read_raw_fif(file, preload = True, verbose = None)
    raw.annotations.delete(
        np.where(
            (raw.annotations.description != 'ON')
            & (raw.annotations.description != 'MW') 
            & (raw.annotations.description != 'MB'))[0]
        )
    sf = raw.info['sfreq']
    srs = pd.read_csv(
        glob.glob(f"{behav_path}/*{recording_date}*ID*{sub_id[3:]}*.csv")[0],
        delimiter = ";"
        )
    vigilance_l = list(srs.Vigilance)
    
    # own df slow waves start in seconds
    mat_sw = np.load(
        glob.glob(
            f"{fig_path}/slowwaves/*{sub_id}*{recording_date}*270623.npy")[0],
        allow_pickle = True
        )
    
    df_dic = {}
    dic_chan = {0 : "O2", 1 : "C4", 2 : "F4", 3 : "Fz", 
                  4 : "F3", 5 : "C3", 6 : "O1"}
    
    for channel_idx in range(len(mat_sw)) :
        df_dic[dic_chan[channel_idx]] = pd.DataFrame(
            mat_sw[channel_idx],
            columns = [
                "start", 
                "end", 
                "middle", 
                "neg_halfway", 
                "period", 
                "neg_amp_peak", 
                "neg_peak_pos", 
                "pos_amp_peak", 
                "pos_peak_pos", 
                "PTP", 
                "1st_negpeak_amp", 
                "1st_negpeak_amp_pos", 
                "last_negpeak_amp", 
                "last_negpeak_amp_pos", 
                "1st_pospeak_amp", 
                "1st_pospeak_amp_pos", 
                "mean_amp", 
                "n_negpeaks", 
                "pos_halfway_period", 
                "1peak_to_npeak_period", 
                "inst_neg_1st_segment_slope", 
                "max_pos_slope_2nd_segment"])
        df_dic[dic_chan[channel_idx]].insert(
            0, "Channel", 
            [dic_chan[channel_idx] for _ in range(
                len(df_dic[dic_chan[channel_idx]]))
                ]
            )
    
    df_sw = pd.concat([df for _, df in df_dic.items()])
    
    for i, time_probe in enumerate(raw.annotations.onset) :
        
        # time_probe = time_probe - raw.first_samp/sf
        time_window_60 = [time_probe - 60, time_probe]
        probetype = raw.annotations.description[i]
        fatigue = vigilance_l[i]
        
        n_probe = f"probe_{i}"
        
        sw_60s_window = df_sw.iloc[
            np.logical_and(
                np.asarray(df_sw.start/sf) < time_window_60[1], 
                np.asarray(df_sw.start/sf) > time_window_60[0]
                )
            ]
        
        if sw_60s_window.shape[0] > 0 :
            for chan in sw_60s_window.Channel.unique() :
                density_60s.append(sw_60s_window.loc[
                    (sw_60s_window["Channel"] == chan)
                    ].shape[0])
                ptp_list.append(
                    sw_60s_window.PTP.loc[
                        (sw_60s_window["Channel"] == chan)
                        ].mean()
                    )
                Dslope_list.append(
                    np.mean(sw_60s_window.neg_amp_peak.loc[
                        (sw_60s_window["Channel"] == chan)
                        ]/(sw_60s_window.neg_peak_pos.loc[
                            (sw_60s_window["Channel"] == chan)
                            ] - sw_60s_window.start.loc[
                                (sw_60s_window["Channel"] == chan)
                                ]))
                    )
                Uslope_list.append(
                    np.mean(sw_60s_window.neg_amp_peak.loc[
                        (sw_60s_window["Channel"] == chan)
                        ]/(sw_60s_window.middle.loc[
                            (sw_60s_window["Channel"] == chan)
                            ] - sw_60s_window.neg_peak_pos.loc[
                                (sw_60s_window["Channel"] == chan)
                                ]))
                    )
                duration_list.append(
                    sw_60s_window.period.loc[
                        (sw_60s_window["Channel"] == chan)
                        ].mean()
                    )
                frequency_list.append(
                    1/sw_60s_window.period.loc[
                        (sw_60s_window["Channel"] == chan)
                        ].mean()
                    )
                valnegpeak_list.append(
                    sw_60s_window.neg_amp_peak.loc[
                        (sw_60s_window["Channel"] == chan)
                        ].mean()
                    )
                valpospeak_list.append(
                    sw_60s_window.pos_amp_peak.loc[
                        (sw_60s_window["Channel"] == chan)
                        ].mean()
                    )
                channel_list.append(chan)
                
                probetype_list.append(probetype)
                nprobe_list.append(n_probe)
                subid_list.append(sub_id)
                
                fatigue_list.append(fatigue)
        
df_swdensity = pd.DataFrame(
    {
     "PTP" : ptp_list,
     "U_slope" : Uslope_list,
     "D_slope" : Dslope_list,
     "Duration" : duration_list,
     "Frequency" : frequency_list,
     "ValPosPeak" : valpospeak_list,
     "ValNegPeak" : valnegpeak_list,
     "Channel" : channel_list,
     "Sub_id" : subid_list,    
     "Density" : density_60s,
     "Mindstate" : probetype_list,
     "fatigue" : fatigue_list,
     "nprobe" : nprobe_list
     }
    )

# df_swdensity = df_swdensity.loc[df_swdensity["nblock"] != 0]
# df_swdensity['Mindstate'] = df_swdensity['Mindstate'].replace('NM', 'MB')
# df_swdensity = df_swdensity.loc[df_swdensity['Mindstate'] != "UNKNWN"]
       
df_savename = fig_path + "/sw_density" + todaydate + ".csv"
df_swdensity.to_csv(df_savename)
