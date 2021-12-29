#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is used to test all functions written in motor_imagery_BCI.py

Created on Mon Dec  6 11:07:58 2021

@author: Connor Harrigan, Dustin Pereslete, Haorui Sun, 
         Max Farrington, Nicole Donahue
"""

# %% Load data
# import packages
import numpy as np
# import script
import motor_imagery_BCI as miBCI
# declare eeg data file path
directory = 'CLA_EEG_Data'
# load eeg data
subject = 'C'
trial_number = '1'
label = f'CLA-Subject{subject}-Trial{trial_number}'
data = miBCI.load_MI_data(directory, subject=subject, trial_number=trial_number)



# %% Rereference EEG Data
# call function to rereference EEG data 
eeg_norm = miBCI.reref_MI_data(data)



# %% Epoch MI events
# extract the event times
marker = data['marker']     #event labels
fs = data['sampFreq']       #sample frequency
event_duration = 1.5        #event duration in seconds
eeg_epochs, event_type = miBCI.epoch_MI(eeg_norm, marker, event_duration, fs)



# %% Plot ERP
# extract channel names
chnames = data['chnames']
# plot the ERP for right and left hand motor imagery
miBCI.plot_erp(eeg_epochs, event_type, event_duration, fs, chnames, label=label)



# %% Abstract ML features
# epoch eeg data with 0.85 duration
event_duration = 0.85
eeg_epochs, event_type = miBCI.epoch_MI(eeg_norm, marker, event_duration, fs)
# define the frequency threshold
freq_threshold = 5
# extract the features of all samples
X = miBCI.abstract_ml_features(eeg_epochs, fs, freq_threshold)



# %% Train and validate svm classifier
# make a copy of event type as truth label
y = np.copy(event_type)
# run machine learning
downsample_size = 300
clf_svm = miBCI.machine_learning(X, y, downsample_size, event_duration, label)



# %% Motor Imagery BCI
# run motor imagery BCI
clf_svm = miBCI.motor_imagery_BCI()

