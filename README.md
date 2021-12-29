# BCI-Fall21
Brain Computer Interfaces Final Project README Fall 2021
Written 12/1/21

This dataset is a collection of electroencephalographic (EEG) motor imagery data for EEG brain-computer interfaces (BCI). Participants were seated in a reclined chair with an EEG cap placed on their head, and a computer screen positioned 200 cm in front, slightly above eye level. This computer screen displayed a Graphical User Interface (GUI), in which images of the left or right hand were presented. Participants remained motionless during the session, and kept their gaze at a gaze-fixation point in the center of the window. The paradigm utilized for this study is the classical motor imagery paradigm (CLA). Participants began by focusing on the gaze-fixation point at the center of the window. At the start of each trial, a 1s action signal was displayed, representing left hand, right hand, or circle (passive). With the display of an action signal, the participants were instructed to imagine closing and opening the respective fist once. For a passive display, participants remained inactive until the next action signal was displayed.

More information about the dataset can be found at: https://www.nature.com/articles/sdata2018211#Sec2

Loading the data:
The raw dataset can be downloaded from: https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698

To load the data into Python, the function ‘loadmat’ will be utilized from the Python script loadmat.py (supplied in the BME 296 Class Code GitHub Repository - https://github.com/djangraw/BME296-F21)

To load the data into a dictionary, the following code can be used:
# Import the loadmat function from loadmat.py
from loadmat import loadmat
# Import the raw dataset
data_file = loadmat(‘CLA-SubjectJ-170508-3St-LRHand-Inter.mat’)
# Select the dictionary in which the data is located
data = data_file[‘o’]

To extract the fields of the dictionary, the following code can be used:
# Extract variables from data
channel_names = data[‘chnames’]
eeg = data[‘data’]
truth_data = data[‘marker’]
etc…

Dictionary Fields:
binsUV = Trivial integer 1 (ignored)
chnames = The labels for each of the electrodes used in the study, in the same order as the EEG data.
data = The raw EEG data of the recording session in microvolts. Each row is a sample and each column is a channel. The columns include the 21 electrodes and the additional bipolar lead. The length of the EEG data varies between subjects and trials.
id = A unique alphanumeric identifier of the data; string
marker = The interaction record of the session (action performed after prompt from GUI); array of integers of same length as EEG data
marker_old = The interaction record of the session from an old nomenclature (ignored)
nS = Total number of samples in the dataset; integer
sampFreq = The sampling frequency in Hz; integer
tag = Empty array (ignored)


Marker Codes:
CLA
0 – No action-signal displayed
1 – Left Hand
2 – Right Hand
3 - Passive

All Paradigms
91 – Inter-session rest break period
92 – Experiment end
99 – Initial relaxation period

Electrode Order: 
Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, A1, A2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz, X3

**Notes:**
- This EEG data has been filtered by a band-pass filter of 0.53-70 Hz and a 50 Hz notch filter. 1
- The X3 “electrode” is a bipolar lead used for data synchronization. This bipolar input port contains spikes associated with the beginning and end of each action signal presentation period. 
- Approximately 300 trials were recorded during interaction segments, with each trial lasting for about 3 seconds (1s action signal followed by a pause for 1.5-2.5s)
- Each interaction segment lasted approximately 15 minutes. 3 segments were conducted with a total BCI interaction time of 45 min per recording session. 
