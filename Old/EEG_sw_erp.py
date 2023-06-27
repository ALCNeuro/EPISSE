#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:21:46 2023

@author: arthurlecoz

EEG_sw_erp.py

"""
# %% Paths & Packages
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import mne, glob

import config as cfg

from scipy.io import loadmat
from autoreject import get_rejection_threshold

from datetime import date
from datetime import datetime
todaydate = date.today().strftime("%d%m%y")

root_path = cfg.root_DDE
raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
psychopy_path = f"{root_path}/CGC_Pilots/Psychopy"
demo_path = f"{root_path}/CGC_Pilots/Demographics"
fig_path = f"{root_path}/CGC_Pilots/Figs"

# %% Variables

input_format = '%Y.%m.%d-%H_%M_%S'
output_format = '%d%m%y'

col_sample = []
col_empty = []
col_event = []

# raw_list = []
epochs_list = []
droplog_l = []
subid_drop_l = []

flat_criteria = dict(eeg=1e-6) 

dict_event = {
    "SW/O2" : 1,
    "SW/C4" : 2,
    "SW/F4" : 3,
    "SW/Fz" : 4,
    "SW/F3" : 5,
    "SW/C3" : 6,
    "SW/O1" : 7
    }

inversed_dict_event = {int(value) : key for key, value in dict_event.items()}
str_inversed_dict_event = {str(value) : key for key, value in dict_event.items()}

dict_channel_number = {
    "O2" : "1",
    "C4" : "2",
    "F4" : "3",
    "Fz" : "4",
    "F3" : "5",
    "C3" : "6",
    "O1" : "7"
    }

fmt_eve = "%d", "%d", "%d"

channels = cfg.channels

files = glob.glob("/Volumes/DDE_ALC/PhD/EPISSE/CGC_Pilots/Preproc/*_Annot.edf")

id_recording = pd.read_csv(
    f"{demo_path}/id_recording_dates.csv",
    delimiter = ";",
    dtype = {"date" : str, "E1_ID" : str, "E2_ID" : str}
    )

# %% Script

for file in files :
    if len(file) < 120 :
        sub_id = file[len(preproc_path):][-26:-21]
        recording_datetime = file[len(preproc_path):][-49:-30]
    else :
        sub_id = file[len(preproc_path):][-27:-22]
        recording_datetime = file[len(preproc_path):][-50:-31]
    
    parsed_time = datetime.strptime(recording_datetime, input_format)
    formatted_date = parsed_time.strftime(output_format)
    
    session = file[len(preproc_path) + 1 :][:22]
    nprobe = file[len(preproc_path) + 1 :][-20:-13]
    
    srs = pd.read_csv(
        glob.glob(f"{behav_path}/*{formatted_date}*ID*{sub_id[3:]}*.csv")[0],
        delimiter = ";"
        )
    vigilance = srs.Vigilance.iloc[int(nprobe[-1])]
    
    classtype = id_recording.type.loc[
        id_recording['date'] == formatted_date
        ].iloc[0]
    
    print(f"\n... CURRENTLY PROCESSING {sub_id} ...")
    if len(glob.glob(
            f"{fig_path}/slowwaves/{sub_id}_{recording_datetime}_260623.npy")
            ) < 1 :
        print(f"No SW files found for {sub_id} at {recording_datetime}")
        continue
    
    slow_waves = np.load(
        glob.glob(
            f"{fig_path}/slowwaves/{sub_id}_{recording_datetime}_260623.npy"
            )[0],
        allow_pickle = True
        )

    raw = mne.io.read_raw_edf(file, preload = True, verbose = None)
    mindstate = raw.annotations.description[0]
    sf = raw.info['sfreq']

    ### SW EVENT w/ NAMES
    
    temp_col_sample = []
    temp_col_empty = []
    temp_col_event = []
    
    for i, channel in enumerate(channels):
        for k, start in enumerate(slow_waves[i][:,0]) :
            if np.isnan(start) :
                continue
            if round(start) > len(raw) :
                continue
            temp_col_sample.append(round(start))
            temp_col_empty.append(0)
            temp_col_event.append(int(dict_channel_number[channel]))
        
    onset = [sampleonset / sf for sampleonset in temp_col_sample]
    duration = temp_col_empty
    description = [inversed_dict_event[event] for event in temp_col_event]
    if len(onset) < 1 :
        continue
    
    raw.annotations.append(onset, duration, description)
    
    # C3_broad_bp = raw.copy().pick_channels(['C3'])
    # C3_broad_bp.filter(
    #     0.1, 30, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
    #         filter_length='auto', phase='zero')
    
    # erp1= np.zeros((slow_waves[4].shape[0],len(range(-500,500))))
    # c=0
    # for x in range(0, np.size(slow_waves[4],0)):
    #     onset = int(slow_waves[4][x,0])
    #     if onset - 500 < 1 :
    #         continue
    #     erp1[c,:] = C3_broad_bp[0,-500+onset:500+onset][0]
    #     erp1[c,:] = erp1[c,:] - np.mean(erp1[c,:])
    #     c=c+1
    # fig1 = plt.figure("ERP_C3")
    # plt.plot(range(-500,500),erp1.mean(0))
    
    events, temp_event_id = mne.events_from_annotations(
        raw, event_id = dict_event)

    # Epoching
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id = temp_event_id, 
        tmin = -1,
        tmax = 1,
        baseline = (-1, -0.5),
        preload = True,
        reject = None,
        flat = flat_criteria,
        event_repeated = 'drop'
        )
    # Low pass

    epochs_hlf = epochs.copy()
    epochs.filter(l_freq = None, h_freq = 30, n_jobs = -1)
    epochs_hlf.filter(l_freq = 1, h_freq = 30, n_jobs = -1)
    
    # Metadata
    
    nepoch_l = [i for i in range(len(epochs))]
    subid_l = [sub_id for i, stage in enumerate(nepoch_l)]
    mindstate_l = [mindstate for _, _ in enumerate(nepoch_l)]
    classtype_l = [classtype for _, _ in enumerate(nepoch_l)]
    nprobe_l = [nprobe for _, _ in enumerate(nepoch_l)]
    vigilance_l = [vigilance for _, _ in enumerate(nepoch_l)]
    
    new_metadata = pd.DataFrame({
        "subid" : subid_l[:len(epochs.events)],
        "n_epoch" : nepoch_l[:len(epochs.events)],
        "mindstate" : mindstate_l[:len(epochs.events)],
        "vigilance" : vigilance_l[:len(epochs.events)],
        "classtype" : classtype_l[:len(epochs.events)],
        "nprobe" : nprobe_l[:len(epochs.events)],
        })
    
    report = mne.Report(
        title=f"Epocheds SW of : {sub_id}_{recording_datetime}")
    
    event_ch = [key[3:] for key, _ in epochs.event_id.items()]

    # Plot your data
    fig = epochs.average().plot(
        titles = "SW - Across Channels", show = False)
    plt.close('all')
    report.add_figure(fig, title = "SW - Across Channels")
    
    epochs.metadata = new_metadata
    epochs_hlf.metadata = new_metadata
    
    # Compute rejection tershold and reject epochs
    reject = get_rejection_threshold(epochs_hlf)
    epochs_hlf.drop_bad(reject=reject)
    epochs_clean = epochs[
        np.isin(
            np.asarray(epochs.metadata.n_epoch),
            np.asarray(epochs_hlf.metadata.n_epoch)
            )
        ]
    
    fig = epochs_clean.average().plot(
        titles = "SW - Across Channels", show = False)
    plt.close('all')
    report.add_figure(fig, title = "SW - Across Channels CLEANED")
    
    subid_drop_l.append(sub_id)
    droplog_l.append(np.round(epochs_hlf.drop_log_stats(), 2))
    
    fig = epochs_hlf.plot_drop_log(show = False)
    report.add_figure(fig, title = "Drop Log")
    
    report.save(
        f"{fig_path}/Reports/SW_ERP_{sub_id}_{recording_datetime}_figure.html", 
        overwrite=True,
        open_browser = False)  
    
    print(f"\n{np.round(epochs_hlf.drop_log_stats(), 2)}% of the epochs cleaned were dropped")
    
    # Saving epochs
    epochs_savename = f"{preproc_path}/erp_slowwaves_{sub_id}_{recording_datetime}_{todaydate}_epo.fif"
    epochs_clean.save(
        epochs_savename,
        overwrite = True
        )
    # epochs.save(epochs_savename,overwrite = True)
    # annotations_savename = preproc_dir + "/erp_slowwaves_epo_annot.csv"
    # epochs.annotations.save(annotations_savename, overwrite = True)   
    
    epochs_list.append(epochs_clean)

df_drop = pd.DataFrame(
    {"sub_id" : subid_drop_l, "drop_per": droplog_l
      }
    )
df_drop_savename = (f"{fig_path}/dropped_epoched_SWERP_{todaydate}.csv")
df_drop.to_csv(df_drop_savename)

# big_epochs = mne.concatenate_epochs(epochs_list)

# big_epochs_savename = preproc_dir + "/erp_slowwaves_all_epo.fif"
# big_epochs.save(
#     big_epochs_savename,
#     overwrite = True
#     )
# annotations_savename = preproc_dir + "/erp_slowwaves_all_epo_annot.csv"
# big_epochs.annotations.save(annotations_savename, overwrite = True)
    

    
    
    