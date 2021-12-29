#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script loads the raw motor imagery EEG data and builds a classifier that
can predict motor imagery activities based on epochs extracted from the data.

Created on Wed Dec  1 17:12:02 2021

@author: Connor Harrigan, Dustin Pereslete, Haorui Sun, 
         Max Farrington, Nicole Donahue
"""

# %% Load data
# import packages
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
# import the loadmat function from loadmat.py 
from loadmat import loadmat 

def load_MI_data(directory, subject, trial_number):
    '''
    This function loads in the motor imagery data file that is on the user's 
    computer. Variable will need to be indexed in order to access the individual 
    datasets, which include EEG data, channel data, frequency data, event sample 
    info, and event type info.

    Parameters
    ----------
    directory : string
        DESCRIPTION. The relative path of data file folder. 
    subject : string
        DESCRIPTION. The subject letter for each participant.[e.g., B and C]
    trial_number : string
        DESCRIPTION. The trial number of each dataset. [e.g., 1, 2 and 3]

    Returns
    -------
    data : dict
        DESCRIPTION. This library has 8 different variables within it. These 
        variables include the names id, tag, nS, sampFreq, marker, data, 
        chnames, and binsuV. These can be accessed by indexing within the 
        data variable like so: eeg = data['data']
        This indexing example would allow for the user to access the eeg data 
        within this dictionary.

    '''
    # load the raw dataset     
    data_file = loadmat(f'{directory}/Subject{subject}_Trial{trial_number}.mat')
    # select the dictionary in which the data is located 
    data = data_file['o']
    # return eeg data dict
    return data


# %% Rereference EEG data
def reref_MI_data(data):
    '''
    This function rereferences the data by implementing a 'common average
    reference' strategy. Specifically, it takes the mean of the raw data across
    electrodes to get a single average EEG electrode time series, and then 
    subtract this average from each of the individual electrodes. 

    Parameters
    ----------
    data : dict
        DESCRIPTION. The EEG data struct that has 8 different variables within 
        it. These variables include the names id, tag, nS, sampFreq, marker, 
        data, chnames, and binsuV.

    Returns
    -------
    eeg_norm : samples x channels 2d float array 
        DESCRIPTION. The referenced EEG data from the raw data set. The
        voltages of brain signals were recorded in uV. 

    '''
    # extract raw EEG data from data struct
    eeg_data = data['data'][:,:-1]
    # calculate the average voltages across channels
    ch_mean = np.mean(eeg_data, axis=1)
    # create the matrix of eeg mean across channels
    eeg_mean = np.transpose(np.full(np.transpose(eeg_data).shape, ch_mean))
    # fill arrays of channel averages and rereferenced data  
    eeg_norm = eeg_data - eeg_mean
    # return rereferenced data
    return eeg_norm



# %% Epoch MI events
def epoch_MI(eeg_data, marker, event_duration, fs):
    '''
    This function extracts event epochs with selected duration from the raw 
    data set. The start point of each event is chosen based on the transition
    of event label in corresponding arrays.

    Parameters
    ----------
    eeg_data : samples x channels 2d float array 
        DESCRIPTION. The raw EEG data that will be epoched. 
    marker : 1d int array with length of sample count
        DESCRIPTION. The interaction record of the session (action performed 
        after prompt from GUI).
        marker code:
            0 -- No action-signal displayed
            1 -- Left hand
            2 -- Right hand
            3 -- Passive
    event_duration : int
        DESCRIPTION. The total duration time for each event in seconds. 1.5s
        is used in this study to plot the ERP while 0.85s is used to abstract
        machine learning features.
    fs : int
        DESCRIPTION. The sampling frequency in Hz.

    Returns
    -------
    eeg_epochs : epochs x samples x channels 3d float array
        DESCRIPTION. Extracted epochs of motor imagery EEG data. The nubmer of
        sample is calculated by multiplying event duration with sampling frequency.
    event_type : 1d int array with length of epoch count.
        DESCRIPTION. Labels for every event. The code is the same as it is in marker.

    '''
    # extract event times
    event_sample = np.where(np.diff(marker) > 0)[0] + 1
    # event counts
    event_count = len(event_sample)
    # calculate number of samples in one epoch
    samples_per_epoch = event_duration * fs
    # extract event types
    event_type = np.zeros(event_count)
    # create a 3d array to hold epoched data
    eeg_epochs = np.zeros((event_count, # pages dimension 0: epochs
                           int(samples_per_epoch), # rows dimension 1: samples
                           eeg_data.shape[1])) # channels dimension 2: channels
    
    # loop through the list of samples and extract eeg data
    for epoch_index, sample in enumerate(event_sample):
        # calculate the starting and ending samples for one epoch
        sample_start = sample
        sample_end = sample + samples_per_epoch
        # extract event type
        event_type[epoch_index] = marker[sample]
        # extract eeg data and feed into eeg_epochs
        eeg_epochs[epoch_index] = eeg_data[int(sample_start):int(sample_end)]
        
    # return values
    return eeg_epochs, event_type


# %% Plot ERP
def plot_erp(eeg_epochs, event_type, event_duration, fs, chnames, electrode='C3', 
             label='CLA-SubjectB-Trial1'):
    '''
    This function plots the ERP of left and right hand motor imageries with 
    corresponding 95 confidence interval. 

    Parameters
    ----------
    eeg_epochs : epochs x samples x channels 3d float array
        DESCRIPTION. Extracted epochs of motor imagery EEG data. The nubmer of
        sample is calculated by multiplying event duration with sampling frequency.
    event_type : 1d int array with length of epoch count.
        DESCRIPTION. Labels for every event. The code is the same as it is in marker.
    event_duration : int
        DESCRIPTION. The total duration time for each event in seconds.
    fs : int
        DESCRIPTION. The sampling frequency in Hz.
    chnames : string array
        DESCRIPTION. The names of channels used in this study. 
    electrode : string, optional
        DESCRIPTION. The channel on which ERP is plotted. 
        The default is 'C3'.
    label : string, optional
        DESCRIPTION. The label of dataset being used in plot title. 
        The default is 'CLA-SubjectB-Trial1'.

    Returns
    -------
    None.

    '''
    # calculate ERP times
    erp_times = np.arange(0, event_duration, 1/fs)
    # abstract left and right hand epochs on the target electrode
    ch_index = np.where(chnames==electrode)[0][0]     # the index of traget ch
    lh_epochs = eeg_epochs[np.where(event_type==1)[0], :, ch_index]
    rh_epochs = eeg_epochs[np.where(event_type==2)[0], :, ch_index]
    # compute the ERP
    lh_erp = lh_epochs.mean(0)      # the erp of left hand MI
    rh_erp = rh_epochs.mean(0)      # the erp of right hand MI
    # compute the std across trials
    lh_sd = lh_epochs.std(0)
    rh_sd = rh_epochs.std(0)
    # compute the std of the mean
    lh_sdmn = lh_sd / math.sqrt(lh_epochs.shape[0])
    rh_sdmn = rh_sd / math.sqrt(rh_epochs.shape[0])
    
    # plot the ERP with 95 CI
    plt.figure()
    # plot the left hand MI and its 95 CI
    plt.plot(erp_times, lh_erp, color='r', label='left-hand motor imagery')
    plt.fill_between(erp_times, lh_erp-2*lh_sdmn, lh_erp+2*lh_sdmn,
                     color='r', alpha=0.25, label='left-hand mi +/- 95% CI')
    # plot the right hand MI and its 95 CI
    plt.plot(erp_times, rh_erp, color='b', label='right-hand motor imagery')
    plt.fill_between(erp_times, rh_erp-2*rh_sdmn, rh_erp+2*rh_sdmn,
                     color='b', alpha=0.25, label='right-hand mi +/- 95% CI')
    # annotate
    plt.xlabel('time relative to the action signal onset (s)')
    plt.ylabel('voltage (uV)')
    plt.grid()
    plt.legend()
    plt.title(label)
    plt.savefig(f'{label}_erp.png')
    


# %% Abstract ML features
def abstract_ml_features(eeg_epochs, fs, freq_threshold):
    '''
    This function abstracts features for machine learning. It first takes the
    Fourier transform of each epoch, keeps ones representing frequencies less
    than the threshold, and splits each of those complex numbers into two
    numbers (the real part and the imaginary part). Note that the imaginary
    part of the 0Hz will be dropped automatically because it's constant zero.

    Parameters
    ----------
    eeg_epochs : epochs x samples x channels 3d float array
        DESCRIPTION. Extracted epochs of motor imagery EEG data. The number of
        samples is calculated by multiplying event duration with sampling frequency.
    fs : int
        DESCRIPTION. The sampling frequency in Hz.
    freq_threshold : int
        DESCRIPTION. The frequency threshold in Hz to limit target frequencies.

    Returns
    -------
    X : epochs x features 2d float array
        DESCRIPTION. The samples for machine learning with rows representing
        samples and columns representing features. The features contain both 
        real and imaginary numbers. 

    '''
    # take the FFT of eeg epochs
    eeg_epochs_FFT = np.fft.rfft(eeg_epochs, axis=1)
    # compute frequencies of FFT
    fft_frequencies = np.fft.rfftfreq(eeg_epochs.shape[1]) * fs
    # get the amplitudes of frequencies within freq threshold
    target_freq = eeg_epochs_FFT[:,np.where(fft_frequencies < freq_threshold)[0]]
    
    # get the sample count
    epoch_count = target_freq.shape[0]
    # get the feature count
    feature_count = (2*target_freq.shape[1]-1)*target_freq.shape[2]
    # declare the sample matrix
    X = np.zeros((epoch_count, feature_count))
    
    # feed in real and imaginary parts of the FFT from each channel
    # as sample features
    for epoch_index in np.arange(epoch_count):
        # extract the real part of the complex number across channels
        real_parts = np.real(target_freq[epoch_index]).flatten()
        # extract the imaginary part of the complex number across channels
        # note that imaginary part of 0Hz will be dropped as it's always 0
        imag_parts = np.imag(target_freq[epoch_index, 1:]).flatten()
        # feed in extracted features
        X[epoch_index] = np.concatenate((real_parts, imag_parts))
        
    # return extracted features
    return X
            

# %% Train and validate svm classifier
def machine_learning(X, y, downsample_size, event_duration, label):
    '''
    This function specifies a series of works of running machine learning on 
    motor imagery EEG data. During the data preprocessing part, it downsamples
    the dataset with given size. Then it splits the downsampled dataset into 
    training data and testing data. The training data is used to optimize the 
    parameters C and gamma in the ML model through cross-validation method. The
    model is built with optimal parameters and trained on the training data. 
    Finally, the classifier is tested on the testing data, and its' performances
    of accuracy score and confusion matrix is reported. 

    Parameters
    ----------
    X : epochs x features 2d float array
        DESCRIPTION. The samples for machine learning with rows representing
        samples and columns representing features. The features contain both 
        real and imaginary numbers. 
    y : 1d int array with length of epochs
        DESCRIPTION. The truth labels of corresponding samples in X.
    downsample_size : int
        DESCRIPTION. The number of samples for each class that the raw dataset 
        will be downsampled.
    label : string
        DESCRIPTION. The label of dataset being used in plot title. 

    Returns
    -------
    clf_svm : SVC object
        DESCRIPTION. Fitted estimator according to the given training data.

    '''
    # abstract the number of classes
    classes = np.unique(y)
    # downsample size
    print('Downsample...')
    X_downsampled_created = False
    y_downsampled = []
    for sample_class in classes:
        # extract all samples for certain class
        X_samp = X[np.where(y==sample_class)[0]]
        # downsample
        X_samp_downsampled = resample(X_samp, replace=False, n_samples=downsample_size)
        y_samp_downsampled = [sample_class for i in np.arange(downsample_size)]
        # add them up
        if not X_downsampled_created:
            X_downsampled = np.copy(X_samp_downsampled)
            X_downsampled_created = True
        else:
            X_downsampled = np.vstack((X_downsampled, X_samp_downsampled))
        y_downsampled += y_samp_downsampled
    # validate the downsampled dataset
    print('After downsampling, there are '+str(len(classes))+' classes with total length of '+str(len(X_downsampled)))
    
    # split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X_downsampled, y_downsampled)
    # scale datasets
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)
    
    # optimize parameters with cross validation
    # set up potential parameters
    print('Optimize parameters with cross validation...')
    param_grid = [
        {'C': [10**i for i in range(-6, 7)],
         'gamma': ['scale', 1e0, 1e-1, 1e-2, 1e-3, 1e-4],
         'kernel': ['rbf']}, 
        ]
    # set up cross validation
    optimal_params = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    # compute the optimal parameters
    optimal_params.fit(X_train_scaled, y_train)
    # print the optimal parameters
    print('The optimal parameters are:', optimal_params.best_params_)
    # lock the values for C and gamma
    C = optimal_params.best_params_['C']
    gamma = optimal_params.best_params_['gamma']
    
    # build the final svm
    clf_svm = SVC(C=C, gamma=gamma)
    # train the model
    print('Train the model...')
    clf_svm.fit(X_train_scaled, y_train)
    print('Now the classifier has been trained.')
    
    # generate the predicted labels
    print('Predict labels for tested dataset...')
    y_pred = clf_svm.predict(X_test_scaled)
    # calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    # get the number of classes, N
    N = len(classes)
    # calculate the ITR_trial
    p = accuracy  
    # ITR_trial = 1 if P = 1
    if p == 1:
        ITR_trial = 1
    else:
        ITR_trial = np.log2(N) + p*np.log2(p) + (1-p)*np.log2((1-p)/(N-1))
    # calcuate the ITR_time
    ITR_time = ITR_trial / event_duration
    # print the results
    print('Accuracy: ', accuracy)
    print('Information Transfer rate (ITR): ', ITR_time, ' bits/second')
    
    # plot the confusion matrix
    print('Plot the confusion matrix...')
    class_names = ['Default', 'Left Hand', 'Right Hand', 'Passive']
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names)
    plt.show()
    plt.text(4.3, 2.1, 'number of samples', rotation=90)
    plt.suptitle(f'{label} Multiclass Confusion Matrix')
    plt.title(f'accuracy={accuracy:.3f}    ITR={ITR_time:.3f} bits/second')
    plt.savefig(f'{label}_mul_c.png')
    # display the main classification metrics
    print('Display the main classification metrics:')
    print(classification_report(y_test, y_pred, target_names=class_names))
    # return trained classifier and confusion matrix
    return clf_svm


# %% Motor Imagery BCI
def motor_imagery_BCI():
    '''
    This function calls functions defined above to train and validate a 
    support-vector machine classifier on selected datasets. Users need to first
    input the path of data folder. Users can choose the datasets. Input 
    1 to load datasets from both subjects from a specified trail. Input 2 to
    load datasets from all trails from a specified subject. Input 3 to load a
    single dataset with specified subject and trial. Note that the dataset
    of SubjectCTrial3 does not have enough sample for each class to train the
    classifier. Users then need to input the value for event duration, 
    frequency threshold, and downsample size according to the prompt. After all
    valid paratmeters are entered, a classifier will be trained on the training
    dataset and validated on the tested dataset. A confusion matrix and the 
    main classification metrics will be displayed. 

    Returns
    -------
    clf_svm : SVC object
        DESCRIPTION. Fitted estimator according to the given training data.

    '''
    # user guide
    print(('Welcome to use our Motor Imagery BCI. Please follow the instructions '
           'to build the classifier that predicts labels for brain signals.'))
    # ask user for data directory
    user_directory = input("Please type in the data directory location (hit enter for default): ")
    if user_directory != "":
        directory = user_directory
    else:
        directory = 'CLA_EEG_Data' # default filepath
    # ask user for the classifier type
    print('What type of classifier are you looking for?')
    print('1--Classifier across subjects')
    print('2--Classifier across trials')
    print('3--Classifier on a single dataset')
    type_num = int(input('Please enter the type number: '))
    # validate the input
    while type_num not in [1, 2, 3]:
        type_num = int(input('Please enter the valid type number: '))
    
    # ask user for subject or trial
    if type_num != 1:
        subject = input('Subject B or C? ')
        while subject not in ['B', 'C']:
            subject = input('Please enter the valid subject letter: ')
    if type_num != 2:
        trial = input('Trial 1, 2, or 3? ')
        while trial not in ['1', '2', '3']:
            trial = input('Please enter the valid trial number: ')
            
    # generate the data_list and label
    data_list = []
    # classifier across subjects
    if type_num == 1:
        for s in ['B', 'C']:
            data_list.append(load_MI_data(directory, s, trial))
        label = f'CLA-AllSubjects-Trial{trial}'
    # classifier across trials
    elif type_num == 2:
        if subject == 'B':
            trials = ['1', '2', '3']
        else:
            trials = ['1', '2']
        for t in trials:
            data_list.append(load_MI_data(directory, subject, t))
                
        label = f'CLA-AllTrials-Subject{subject}'
    # classifier on single dataset
    else:
        data_list.append(load_MI_data(directory, subject, trial))
        label = f'CLA-Subject{subject}-Trial{trial}'
    
    # ask user for event duration and frequency threshold
    user_event_duration = input(('Please enter the event duration in seconds '
                                 '(hit enter for default, 0.85s): '))
    if user_event_duration != '':
        event_duration = int(user_event_duration)
    else:
        event_duration = 0.85       # default event duration
    
    user_freq_threshold = input(('Please enter the frequency threshold in Hz '
                                 '(hit enter for default, 5Hz): '))
    if user_freq_threshold != '':
        freq_threshold = int(user_freq_threshold)
    else:
        freq_threshold = 5         # default frequency threshold
        
    # declare a boolean value for sample matrix for machine learning
    isXCreated = False
    # loop through each eeg dataset to abstract machine learning features
    for data in data_list:
        # rereference EEG data 
        eeg_norm = reref_MI_data(data)
        # extract the event times
        marker = data['marker']     #event labels
        fs = data['sampFreq']       #sample frequency
        # epoch eeg data
        eeg_epochs, event_type = epoch_MI(eeg_norm, marker, event_duration, fs)
        if not isXCreated:
            X = abstract_ml_features(eeg_epochs, fs, freq_threshold)
            y = np.copy(event_type)
            isXCreated = True
        else:
            X = np.vstack((X, abstract_ml_features(eeg_epochs, fs, freq_threshold)))
            y = np.concatenate((y, event_type))
    
    # ask user for downsample size
    user_downsample_size = input(('Please enter the downsample size for each class '
                                  '(hit enter for default, 300): '))
    if user_downsample_size != '':
        downsample_size = user_downsample_size
    else:
        downsample_size = 300        # default frequency threshold
        
    # run machine learning to train and validate the classifier
    clf_svm = machine_learning(X, y, downsample_size, event_duration, label)
    
    # return trained classifier
    return clf_svm
    
