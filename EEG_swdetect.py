#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 23:18:17 2017

@author: Arthur
From @author: Thomas A.

EEG_swdetect.py
"""
# %% Script

import multiprocessing

def SW_detect(file_path):
    ### import libraries
    import numpy as np, pandas as pd
    import mne
    import config as cfg
    import warnings
    from scipy.stats import exponnorm
    from datetime import date
    
    todaydate = date.today().strftime("%d%m%y")
    inspect = 1
    # sfreq = 512
    channels = cfg.channels
    # bad_ids = cfg.bad_ids
    
    root_dir = cfg.root_DDE
    preproc_dir = f"{root_dir}/CGC_Pilots/Preproc"
    # raw_dir = root_dir + "/Raw"
    fig_dir = f"{root_dir}/CGC_Pilots/Figs"
    
    # if len(file_path) < 120 :
    sub_id = file_path[len(preproc_dir):][-20:-15]
    recording_datetime = file_path[len(preproc_dir):][-27:-21]
    
    ### Selection criteria for SWs (estimated visually)
    slope_range = [0.25, 1] # in uV/ms
    positive_amp = [75] # in uV
    amplitude_max = 150
    filt_range = [0.2, 7] # in Hz
    thr_value = 0.1
    
    eeg_raw_data = mne.io.read_raw_fif(file_path, preload = True)
    sfreq = eeg_raw_data.info['sfreq']
    
    ### Prepare EEG for pre-processing
    eeg_low_bp = eeg_raw_data.copy()

    ### Pre-processing of EEG data: Filtering
    eeg_low_bp.filter(
        filt_range[0], filt_range[1], 
        l_trans_bandwidth='auto', h_trans_bandwidth='auto',
        filter_length='auto', phase='zero'
        )
    nchan, nsamples=eeg_low_bp._data.shape

    ### Pre-processing of EEG data: Down-sample to 100Hz
    eeg_low_bp_re = eeg_low_bp.copy().resample(100, npad='auto')
    nchannew, nsamplesnew = eeg_low_bp_re._data.shape
    newsfreq = 100
        
    ### Loop across channels
    nchan, nsamples=eeg_low_bp_re._data.shape
    allWaves=[]
    slowWaves=[]
    
    tresh_ch = []
    tresh_endgauss = []
    tresh_ids = []
    report = mne.Report(title=f"PTP inspections of {sub_id}")
    
    for ch in range(0,len(channels)):
        print('Processing: ')
        print(channels[ch])
        this_eeg_chan = eeg_low_bp_re.copy().pick_channels([channels[ch]])
        this_eeg_chan = this_eeg_chan[0]
        this_eeg_chan = this_eeg_chan[0][0,:]
        
        if np.max(this_eeg_chan)<1: # probably in V and not uV
            this_eeg_chan=this_eeg_chan * 1000000
            print('converting to uV')

        ### Detection of SWs: Find 0-crossings
        zero_crossing = this_eeg_chan > 0
        zero_crossing = zero_crossing.astype(int)
    
        # Find positive zero-crossing (start of potential SW)
        pos_crossing_idx = np.where(np.diff(zero_crossing)>0)[0]
        pos_crossing_idx = [x+1 for x in pos_crossing_idx]
        # Find negative zero-crossing (end of potential SW)
        neg_crossing_idx = np.where(np.diff(zero_crossing)<0)[0]
        neg_crossing_idx = [x+1 for x in neg_crossing_idx]
        
        # Compute derivative and smooth (5-sample running average)
        der_eeg_chan = np.convolve(
            np.diff(this_eeg_chan), np.ones((5,))/5, mode='valid'
            )
        thr_crossing = der_eeg_chan > thr_value
        thr_crossing = thr_crossing.astype(int)
        difference = np.diff(thr_crossing)
        peaks = np.asarray(np.where(difference==-1))+1
        peaks = peaks[0]
        troughs = np.asarray(np.where(difference==1))+1
        troughs = troughs[0]
        
        # Rejects peaks below zero and troughs above zero
        peaks = peaks[this_eeg_chan[peaks]>0]
        troughs = troughs[this_eeg_chan[troughs]<0]
        if neg_crossing_idx[0]<pos_crossing_idx[0]:
            start = 1
        else:
            start = 2
        if start == 2:
            pos_crossing_idx = pos_crossing_idx[1:]
        
        ### Detection of SWs
        waves = np.empty((len(neg_crossing_idx)-start+1,22))
        waves[:] = np.nan
        lastpk = np.nan
        for wndx in range(start,len(neg_crossing_idx)-1):
            wavest = neg_crossing_idx[wndx]
            wavend = neg_crossing_idx[wndx+1]
            # matrix (27) determines instantaneous positive 1st segement slope on smoothed signal, (name not representative)
            mxdn = np.abs(
                np.min(der_eeg_chan[wavest:pos_crossing_idx[wndx]])
                ) * newsfreq;  
            # matrix (28) determines maximal negative slope for 2nd segement (name not representative)
            mxup = np.max(
                der_eeg_chan[wavest:pos_crossing_idx[wndx]]
                ) * newsfreq; 
            tp1 = np.where(troughs>wavest)
            tp2 = np.where(troughs<wavend)
            negpeaks = troughs[np.intersect1d(tp1,tp2)]
            
            # In case a peak is not detected for this wave (happens rarely)
            if np.size(negpeaks) == 0:
                waves[wndx, :] = np.nan
                # thisStage=newscoring[wavest];
                # waves[wndx,22] =thisStage;
                continue
            
            tp1 = np.where(peaks > wavest)
            tp2 = np.where(peaks <= wavend)
            pospeaks = peaks[np.intersect1d(tp1,tp2)]
            
            # if negpeaks is empty set negpeak to pos ZX
            if np.size(pospeaks) == 0 :
                pospeaks = np.append(pospeaks,wavend)
                
            period = wavend-wavest #matrix(11) /SR
            poszx = pos_crossing_idx[wndx] #matrix(10)
            b = [np.min(this_eeg_chan[negpeaks])][0] #matrix (12) most pos peak /abs for matrix
            #if len(b)>1:
            #    b=b[0]
            bx = negpeaks[np.where(this_eeg_chan[negpeaks]==b)][0] #matrix (13) max pos peak location in entire night
            c = [np.max(this_eeg_chan[pospeaks])][0] #matrix (14) most neg peak
            #if len(c)>1:
            #    c=c[0]
            cx = pospeaks[np.where(this_eeg_chan[pospeaks]==c)] #matrix (15) max neg peak location in entire night
            cx = cx[0]
            maxb2c = c-b #matrix (16) max peak to peak amp
            nump = len(negpeaks) #matrix(24) now number of positive peaks
            n1 = np.abs(this_eeg_chan[negpeaks[0]]) #matrix(17) 1st pos peak amp
            n1x = negpeaks[0] #matrix(18) 1st pos peak location
            nEnd = np.abs(this_eeg_chan[negpeaks[len(negpeaks)-1]]) #matrix(19) last pos peak amp
            nEndx = negpeaks[len(negpeaks)-1] #matrix(20) last pos peak location
            p1 = this_eeg_chan[pospeaks[0]] #matrix(21) 1st neg peak amp
            p1x = pospeaks[0] #matrix(22) 1st pos peak location
            meanAmp = np.abs(np.mean(this_eeg_chan[negpeaks])); #matrix(23)
            nperiod = poszx-wavest; #matrix (25)neghalfwave period
            mdpt = wavest+np.ceil(nperiod/2); #matrix(9)
            p2p = (cx-lastpk)/newsfreq; #matrix(26) 1st peak to last peak period
            lastpk = cx;
            
            # thisStage=newscoring[poszx];
            # Result Matrix
            #0:  wave beginning (sample)
            #1:  wave end (sample)
            #2:  wave middle point (sample)
            #3:  wave negative half-way (sample)
            #4:  period in seconds
            #5:  negative amplitude peak
            #6:  negative amplitude peak position (sample)
            #7:  positive amplitude peak
            #8:  positive amplitude peak position (sample)
            #9:  peak-to-peak amplitude
            #10: 1st neg peak amplitude
            #11: 1st neg peak amplitude position (sample)
            #12: Last neg peak amplitude
            #13: Last neg peak amplitude position (sample)
            #14: 1st pos peak amplitude
            #15: 1st pos peak amplitude position (sample)
            #16: mean wave amplitude
            #17: number of negative peaks
            #18: wave positive half-way period
            #19: 1st peak to last peak period
            #20: determines instantaneous negative 1st segement slope on smoothed signal
            #21: determines maximal positive slope for 2nd segement
            #22: stage (if scored data)
            
            waves[wndx, :] = (wavest, wavend, mdpt, poszx, period/newsfreq, 
                              np.abs(b), bx, c, cx, maxb2c, n1, n1x, nEnd, 
                              nEndx, p1, p1x, meanAmp, nump, nperiod/newsfreq, 
                              p2p, mxdn, mxup)#, thisStage)
            waves[wndx, (0,1,2,3,6,8,11,13,15)] = waves[
                wndx, (0,1,2,3,6,8,11,13,15)]*(sfreq/newsfreq);
        
        allWaves.append(waves)
        
        cond1 = np.where(waves[:, 18] < slope_range[1])
        cond2 = np.where(waves[:, 18] > slope_range[0])
        cond3 = np.where(waves[:, 7] < positive_amp)
        
        temp_sw = waves[
            np.intersect1d(cond1,np.intersect1d(cond2,cond3)),:]
        temp_sw = temp_sw[np.where(temp_sw[:, 9] < amplitude_max)]
        
        temp_p2p = temp_sw[:, 9]
        
        # Trying relative treshold per subject -
        if len(temp_p2p) < 100 :
            warnings.warn('not enough waves (<100) to compute a threshold!!! Sub %s Elec %s\n' 
                          % (sub_id, channels[ch]))
            slowWaves.append(np.full(temp_sw.shape, np.nan))
            """
            THIS IS A TEMPORARY SOLUTION - I MIGHT NEED A BETTER WAY
            TO DO IT 
            """
            continue
        
        params = exponnorm.fit(temp_p2p)#, floc=temp_sw[:,9].min())
        mu, sigma, lam = params
        bins = np.arange(0, temp_sw[:,9].max(), 0.1)
        y = exponnorm.pdf(bins, mu, sigma, lam)
        max_gaus = bins[np.where(y == max(y))][0] * 2
        
        if inspect :
            import matplotlib.pyplot as plt
            
            fig1 = plt.figure(f"GaussianFit_{ch}")
            plt.hist(
                temp_sw[:,9], bins = 100, density = True, 
                alpha = 0.5, label = "PTP SW"
                )
            plt.plot(bins, y, 'r-', label = "Ex-Gaussian Fit")
            plt.axvline(
                x = max_gaus, color = 'r', 
                label = "2 * Max Gauss", ls = '--')
            plt.xlabel('Values')
            plt.ylabel('Density')
            plt.title('Ex-Gaussian Fit')
            plt.legend()
            plt.show(block=False)
            report.add_figure(
                fig=fig1,
                title=f"Ex Gaussian Fitting & PTP distribution at channel {channels[ch]}",
                caption="Is the distribution really Ex_G looking?",
                image_format="PNG",
            )
            plt.close(fig1)
        
        temp_sw = temp_sw[np.where(temp_sw[:, 9] > max_gaus)]
        
        # Fz_broad_bp = eeg_raw_data.copy().pick_channels(['Fz'])
        # Fz_broad_bp.filter(
        #     0.1, 30, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
        #         filter_length='auto', phase='zero')
        
        # erp1= np.zeros((temp_sw.shape[0],len(range(-500,500))))
        # c=0
        # for x in range(0, np.size(temp_sw,0)):
        #     onset = int(temp_sw[x,0])
        #     if onset - 500 < 1 :
        #         continue
        #     erp1[c,:] = Fz_broad_bp[0,-500+onset:500+onset][0]
        #     erp1[c,:] = erp1[c,:] - np.mean(erp1[c,:])
        #     c=c+1
        # fig1 = plt.figure(f"ERP_{ch}")
        # plt.plot(range(-500,500),erp1.mean(0))
        
        tresh_ch.append(channels[ch])
        tresh_endgauss.append(max_gaus)
        tresh_ids.append(sub_id)
      
        # slowWaves.append(waves[np.intersect1d(cond1,np.intersect1d(cond2,cond3)),:])
        slowWaves.append(temp_sw)
        print(f"...For {sub_id} :\n - Number of all-waves : {np.size(allWaves[ch], 0)}\n - Number of slow-waves : {np.size(slowWaves[ch], 0)}...\n")
    
    report.save(
        f"{fig_dir}/Reports/PTP_report_{sub_id}_{recording_datetime}_figure.html", 
        overwrite=True,
        open_browser = False)    
    
    df_tresh = pd.DataFrame({
        "sub_id" : tresh_ids,
        "channel" : tresh_ch,
        "treshold" : tresh_endgauss
        })
    df_tresh_savename = (
        f"{fig_dir}/threshold/SW_{sub_id}_{recording_datetime}_{todaydate}.csv")
    df_tresh.to_csv(df_tresh_savename)

    allwaves_savename = (f"{fig_dir}/allwaves/{sub_id}_{recording_datetime}_{todaydate}.npy")
    slowwaves_savename = (f"{fig_dir}/slowwaves/{sub_id}_{recording_datetime}_{todaydate}.npy")
    np.save(allwaves_savename, np.asarray(allWaves, dtype = object))
    np.save(slowwaves_savename, np.asarray(slowWaves, dtype = object))
        
    return "...\n\n ALL FILES WERE COMPUTED"

if __name__ == '__main__':
    import glob
    # Get the list of EEG files
    eeg_files = glob.glob("/Volumes/DDE_ALC/PhD/EPISSE/CGC_Pilots/Preproc/*_concat_raw.fif")
    
    # Set up a pool of worker processes
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=8)
    
    # Process the EEG files in parallel
    pool.map(SW_detect, eeg_files)
    
    # Clean up the pool of worker processes
    pool.close()
    pool.join()

