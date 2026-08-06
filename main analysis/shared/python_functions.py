from itertools import repeat
import os
import scipy.signal as sig
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import math
from os import listdir
from os.path import isfile, join



################################################
# Global variables
sub_ch = ['C3', 'C4', 'O1', 'O2', 'A1', 'A2', 'Cz', 'F3', 'F4', 'F7', 'F8',
'Fz', 'Fp1', 'Fp2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
################################################

######################################################################
# Functions
######################################################################

#####################
# Data Loading Functions
#####################
def get_edf_files(gene, base_folder ="/mnt/isilon/helbig_lab/Users/galerp/EEG/"):
    """ Gathers all EDF file paths from specified gene folder and places into a list

    Args:
        gene (string): gene name (capitalized)
        base_folder (str, optional): file path to EEG folders. 
        Defaults to "/Volumes/helbig_lab/Users/galerp/EEG/".
    """
    base_dir = base_folder + gene
    all_edf = []
    os.chdir(base_dir)

    all_dir = [os.path.abspath(name) for name in os.listdir(".") if os.path.isdir(name)]
    all_pat_dir = [dir for dir in all_dir if "/EG" in dir or "/ENG" in dir or "/P" in dir or "/ENDD" in dir or "/chop" in dir or "/patsz" in dir]

    for pat in all_pat_dir:

        cur_pat = pat.rsplit('/', 1)[1]
        # print(cur_pat)

        allfiles = [f for f in listdir(pat) if isfile(join(pat, f))]

        edffiles = [str for str in allfiles if
                    any("edf" in str for sub in "edf")]
        edffiles = [pat + "/" + edf for edf in edffiles]
        all_edf = np.concatenate((all_edf, edffiles))

    return(all_edf)


############################
# EEG Data Information
############################

def get_eeg_details(edf_file):
    """
    Analyzes an EEG file and obtains details such as the sampling frequency.

    Parameters:
    - edf_file (str): The path to the EEG file.

    Returns:
    - pat_sf (pandas.DataFrame): A DataFrame containing the patient, date, gene, and sampling frequency.

    Example:
    >>> edf_file = "/mnt/isilon/ENGIN_1234_CROPPED_manual/2022_01_01.edf"
    >>> get_eeg_details(edf_file)
        patient        date   gene    sf
    0  ENGIN_1234  2022/01/01  ENGIN  256
    """


    file_string = edf_file.replace("/mnt/isilon/", "/Volumes/")
    if "ENGIN_" in edf_file:
        file_string = file_string.replace('ENGIN_', 'ENGIN')
    if "_CROPPED" in edf_file:
        file_string = file_string.replace('_CROPPED', '')
    if "_manual" in edf_file:
        file_string = file_string.replace('_manual', '')

    parts = file_string.split('/')
    Gene =  parts[6]
    cur_pat =  parts[7]

    #Get date from file path
    # Find the index of the second "_" character
    first_underscore_index = file_string.find('_')
    second_underscore_index = file_string.find('_', first_underscore_index + 1)
    # Extract everything after the second "_" character
    extracted_part = file_string[second_underscore_index + 1:]
    # Remove ".edf"
    extracted_part = extracted_part.replace('.edf', '')
    # Replace "_" with "/"
    file_date = extracted_part.replace('_', '/')

    raw = mne.io.read_raw_edf(edf_file)
    # Get the sampling frequency of the raw data
    sf = raw.info['sfreq']


    col_names = ['patient', 'date', 'gene', 'sf']
    pat_sf = pd.DataFrame(index = range(1), columns = col_names)
    pat_sf['patient'] = cur_pat
    pat_sf['gene'] = Gene
    pat_sf['date'] = file_date
    pat_sf['sf'] = sf

    return pat_sf

#####################
# Filter Functions
#####################

# Function to make a butterworth notch filter 

def create_notch_filter(order, notch_freq, fs) :
  """
  Creates a butterworth notch filter with the desired parameters
  Input: 
    order (int): Order of the filter
    notch_freq (float): Critical frequency of the notch band, in Hz
    fs (float): Sampling frequency of the signal the filter will be used on, in Hz
  Returns: 
    b, a (ndarray, ndarray): b, a are the numerator and denominator polynomials of the filter, respectively
  """
  b, a = sig.iirnotch(notch_freq, order, fs)
  return b, a

# Function to make a butterworth band pass filter

def create_band_pass_filter(order,  low_cutoff, high_cutoff, fs) :
  """
  Creates butterworth high pass filter
  Input: 
    order (int): Order of the filter
    cutoff (float): Critical frequency of the filter, in Hz
    fs (float): Sampling frequency of the signal the filter will be used on, in Hz
  Returns: 
    b, a (ndarray, ndarray): b, a are the numerator and denominator polynomials of the filter, respectively
  """
  b, a = sig.butter(N=order, Wn=[low_cutoff, high_cutoff], btype='bandpass', fs=fs)
  return b, a

# Function to apply a certain filter to a signal

def apply_filter(data: np.ndarray, filter) -> np.ndarray:
  """
  Applies the given filter to the given data. A filtered version of the data is returned. The original 
  input data is not modified
  Input: 
    data (ndarray): Input data to be filtered. Can either have shape (num_channels, num_samples) or (num_samples)
    filter (tuple): A tuple (b, a) of the filter to be applied. b, a are ndarrays that are the numerator and denominator polynomials of the filter, respectively
  Returns: 
    ndarray of the filtered data with the same shape as the input data
  """
  b, a = filter[0], filter[1]
  return sig.filtfilt(b, a, data)

def generate_harmonics(base_freq, sampling_freq):
    """
    Generate harmonics of a base frequency up to the Nyquist frequency.

    Input:
        base_freq (int): The base frequency (e.g., 60Hz for power line noise)
        sampling_freq (int): The sampling frequency of the data
    Returns:
        harmonics: A tuple of harmonics frequencies up to the Nyquist frequency
    """
    nyquist_freq = sampling_freq / 2
    harmonics = []

    for n in range(1, int(nyquist_freq / base_freq) + 1):
        harmonic_freq = n * base_freq
        if harmonic_freq <= nyquist_freq:
            harmonics.append(harmonic_freq)

    return tuple(harmonics)

#####################
# Montages Functions
#####################
# Common average reference montage

def common_average_reference(data) :
  """
  Applies common average reference montage to the input data. The input data is not modified. 
  Input:
    data (ndarray): Input signal to be common average referenced. Has shape (num_channels, num_samples).
  Returns:
    ndarray of same shape as input that is the input data with the common average reference montage applied to it. 
  """
  return data - np.mean(data, axis=0)

# Laplacian Montage

def laplac(data, ch_dict):
  """
  Applies laplacian montage to the input data.
  Input:
    data (ndarray): Input EEG signal. Has shape (num_channels, num_samples).
  Returns:
    ndarray of same shape as input that is the input data with the laplacian montage applied to it. 
  """
  data_lap = np.copy(data)
  data_lap[ch_dict['Fp1']] = (((2*data[ch_dict['Fp1']]) + \
        data[ch_dict['F3']] + \
            (2*data[ch_dict['F7']]))/5) - \
                data[ch_dict['Fp1']]
  
  data_lap[ch_dict['F3']] = ((data[ch_dict['F3']] + \
        data[ch_dict['Fp1']] + \
            data[ch_dict['Fz']] + \
                data[ch_dict['C3']])/4) - \
                    data[ch_dict['F3']]
  
  data_lap[ch_dict['C3']] = ((data[ch_dict['T3']] + \
        data[ch_dict['C3']] + \
            data[ch_dict['Cz']] + \
                data[ch_dict['P3']])/4) - \
                    data[ch_dict['C3']]
  
  data_lap[ch_dict['P3']] = ((data[ch_dict['C3']] + \
        data[ch_dict['Pz']] + \
            data[ch_dict['O2']] + \
                data[ch_dict['T5']])/4) - \
                    data[ch_dict['P3']]
  
  data_lap[ch_dict['F7']] = (((2*data[ch_dict['Fp1']]) + \
        data[ch_dict['C3']] + \
            (2*data[ch_dict['T3']]))/5) - \
                data[ch_dict['F7']]
  
  data_lap[ch_dict['T3']] = (((2*data[ch_dict['F7']]) + \
        data[ch_dict['C3']] + \
            (2*data[ch_dict['T5']]))/5) - \
                data[ch_dict['T3']]
  
  data_lap[ch_dict['T5']] = (((2*data[ch_dict['T3']]) + \
        data[ch_dict['P3']] + \
            (2*data[ch_dict['O1']]))/5) - \
                data[ch_dict['T5']]
  
  data_lap[ch_dict['O1']] = ((data[ch_dict['P3']] + \
        (2*data[ch_dict['O2']]) + \
            (2*data[ch_dict['T5']]))/5) - \
                data[ch_dict['O1']]
  
  data_lap[ch_dict['Fp2']] = (((2*data[ch_dict['F8']]) + \
        data[ch_dict['F4']] + \
            (2*data[ch_dict['Fp1']]))/5) - \
                data[ch_dict['Fp2']]
  
  data_lap[ch_dict['F4']] = ((data[ch_dict['Fp2']] + \
        data[ch_dict['F8']] + \
            data[ch_dict['C4']] + \
                data[ch_dict['Fz']])/4) - \
                    data[ch_dict['F4']]
  
  data_lap[ch_dict['C4']] = ((data[ch_dict['F4']] + \
        data[ch_dict['T4']] + \
            data[ch_dict['P4']] + \
                data[ch_dict['Cz']])/4) - \
                    data[ch_dict['C4']]
  
  data_lap[ch_dict['P4']] = ((data[ch_dict['C4']] + \
        data[ch_dict['T6']] + \
            data[ch_dict['O2']] + \
                data[ch_dict['Pz']])/4) - \
                    data[ch_dict['P4']]
  
  data_lap[ch_dict['F8']] = (((2*data[ch_dict['Fp2']]) + \
        (2*data[ch_dict['T4']]) + \
            data[ch_dict['F4']])/5 ) - \
                data[ch_dict['F8']]
  
  data_lap[ch_dict['T4']] = (((2*data[ch_dict['F8']]) + \
        (2*data[ch_dict['T6']]) + \
            data[ch_dict['C4']])/5) - \
                data[ch_dict['T4']]
  
  data_lap[ch_dict['T6']] = (((2*data[ch_dict['T4']]) + \
        (2*data[ch_dict['O2']]) + \
            data[ch_dict['P4']])/5) - \
                data[ch_dict['T6']]
  
  data_lap[ch_dict['O2']] = ((data[ch_dict['P4']] + \
        (2*data[ch_dict['T6']]) + \
            (2*data[ch_dict['O1']]))/5) - \
                data[ch_dict['O2']]
  
  data_lap[ch_dict['Fz']] = ((data[ch_dict['Fp2']] + \
        data[ch_dict['F4']] + \
            data[ch_dict['Cz']] + \
                data[ch_dict['C3']] + \
                    data[ch_dict['Fp1']])/5) - \
                        data[ch_dict['Fz']]
  
  data_lap[ch_dict['Cz']] = ((data[ch_dict['Fz']] + \
        data[ch_dict['C4']] + \
            data[ch_dict['Pz']] + \
                data[ch_dict['C3']])/4) - \
                    data[ch_dict['Cz']]
  
  data_lap[ch_dict['Pz']] = ((data[ch_dict['Cz']] + \
        data[ch_dict['P4']] + \
            data[ch_dict['O2']] + \
                data[ch_dict['O1']] + \
                    data[ch_dict['P3']])/5) - \
                        data[ch_dict['Pz']]
  
  return data_lap

# Function to obtain bipolar montage for 21 electrode EEG

def create_bipolar_montage(data, ch_dict):
  """
  Creates a dictionary that is the bipolar montage of an input EEG signal
  Input:
    data (ndarray): Input EEG data from which the bipolar montage is to be created. Has shape (num_channels, num_samples). Can have extra channels, but must 
    have all channels that are required to create a bipolar montage.
    ch_names (dictionary): The names of the channels corresponding to the index of the channels in the data array. 
  Returns: 
    Dictionary that is the bipolar montage of the input EEG. The keys are strings representing the name of a channel in the montage (eg: 'fp1-fp7'), and the 
    values are ndarrays of shape (num_samples) that are the associated signal for that channel
  """

  bipolar_montage = {} # Dictionary containing the bipolar montage of the eeg data


  bipolar_montage['Fp1-F7'] = data[ch_dict['Fp1']] - data[ch_dict['F7']]
  bipolar_montage['F7-T3'] = data[ch_dict['F7']] - data[ch_dict['T3']]
  bipolar_montage['T3-T5'] = data[ch_dict['T3']] - data[ch_dict['T5']]
  bipolar_montage['T5-O1'] = data[ch_dict['T5']] - data[ch_dict['O1']]

  bipolar_montage['Fp1-F3'] = data[ch_dict['Fp1']] - data[ch_dict['F3']]
  bipolar_montage['F3-C3'] = data[ch_dict['F3']] - data[ch_dict['C3']]
  bipolar_montage['C3-P3'] = data[ch_dict['C3']] - data[ch_dict['P3']]
  bipolar_montage['P3-O1'] = data[ch_dict['P3']] - data[ch_dict['O1']]

  bipolar_montage['Fp2-F4'] = data[ch_dict['Fp2']] - data[ch_dict['F4']]
  bipolar_montage['F4-C4'] = data[ch_dict['F4']] - data[ch_dict['C4']]
  bipolar_montage['C4-P4'] = data[ch_dict['C4']] - data[ch_dict['P4']]
  bipolar_montage['P4-O2'] = data[ch_dict['P4']] - data[ch_dict['O2']]

  bipolar_montage['Fp2-F8'] = data[ch_dict['Fp2']] - data[ch_dict['F8']]
  bipolar_montage['F8-T4'] = data[ch_dict['F8']] - data[ch_dict['T4']]
  bipolar_montage['T4-T6'] = data[ch_dict['T4']] - data[ch_dict['T6']]
  bipolar_montage['T6-O2'] = data[ch_dict['T6']] - data[ch_dict['O2']]

  bipolar_montage['Fz-Cz'] = data[ch_dict['Fz']] - data[ch_dict['Cz']]
  bipolar_montage['Cz-Pz'] = data[ch_dict['Cz']] - data[ch_dict['Pz']]

  return bipolar_montage

#####################
# Binning Functions
#####################

# Function to compute number of windows 

def num_windows(fs, winLen, winDisp, xLen) :
  """
  Computes number of full sliding windows of size winLen in a certain signal
  Input:
    fs (float): Sampling frequency of the signal
    winLen (float): Length of window, in seconds
    winDisp (float): Stride of the window, in seconds
    xLen (int): Length of signal, in samples
    Returns:
      An int that is the number of possible full windows in the signal 
  """
  # winLen, winDisp = winLen * fs, winDisp * fs # Convert winLen, winDisp to samples
  return int(((xLen/fs - winLen + winDisp) - ((xLen/fs - winLen + winDisp)%winDisp))/winDisp)


# Function to compute values of a given function at each window in the signal

def moving_window_features(x, fs, winLen, winDisp, featFn) :
  """
  Computes an array containing the value of featFn for each possible window in the given signal
  Input: 
    x (ndarray): Input signal of either shape (num_channels, num_samples) or (num_samples)
    fs (float): Sampling frequency of the input signal, in Hz
    winLen (float): Length of window, in seconds
    winDisp (float): Stride of window, in seconds
    featFn (function): The function to apply on windows of the signal
  Returns:
    ndarray whose ith value is the value of featFn evaluated at the ith sliding window from the left. 
    If input shape is (num_channels, num_samples), output shape is (num_channels, num_windows).
    If input shape is (num_samples), output shape is (num_windows).
  """
  
  num_wins = num_windows(fs, winLen, winDisp, x.shape[-1]) # Number of complete windows in the signal
  winLen, winDisp = winLen * fs, winDisp * fs # Convert winLen, winDisp to samples

  if (len(x.shape) == 1) : # x has shape (num_samples)
    features = np.zeros((num_wins)) # Features array to be populated
    # For loop populates features array with the values of featFn evaluated at each of the windows
    for i in range(num_wins) :
      features[i] = featFn(x[int(i * winDisp) : int(i * winDisp + winLen)])
    return features
  # x has shape (num_channels, num_samples)
  num_channels = x.shape[0] 
  features = np.zeros((num_channels, num_wins)) # Features array to be populated
  # For loop populates features array with the values of featFn evaluated at each of the windows in each of the channels
  for i in range(num_channels) :
    for j in range(num_wins) :
      features[i][j] = featFn(x[i][int(j * winDisp) : int(j * winDisp + winLen)])
  return features


# Function to place signal into bins

def moving_window(x, fs, winLen, winDisp, num_wins) :
  """
  Computes an array containing the value of featFn for each possible window in the given signal
  Input: 
    x (ndarray): Input signal of either shape (num_channels, num_samples) or (num_samples)
    fs (float): Sampling frequency of the input signal, in Hz
    winLen (float): Length of window, in seconds
    winDisp (float): Stride of window, in seconds
    featFn (function): The function to apply on windows of the signal
  Returns:
    ndarray whose ith value is the value of featFn evaluated at the ith sliding window from the left. 
    If input shape is (num_channels, num_samples), output shape is (num_channels, num_windows).
    If input shape is (num_samples), output shape is (num_windows).
  """
  
  # num_wins = num_windows(fs, winLen, winDisp, x.shape[-1]) # Number of complete windows in the signal
  winLen, winDisp = int(winLen * fs), int(winDisp * fs) # Convert winLen, winDisp to samples

  # x has shape (num_channels, num_samples)
  num_channels = x.shape[0] 
  x_binned = np.zeros((num_wins, num_channels, winLen)) # Features array to be populated
  # For loop populates features array with the values of featFn evaluated at each of the windows in each of the channels
  for i in range(num_channels) :
    for j in range(num_wins) :
      x_binned[j,i,:] = x[i,int(j * winDisp) : int(j * winDisp + winLen)]
    #   features[i][j] = x[i][int(j * winDisp) : int(j * winDisp + winLen)]
  return x_binned

#####################
# Event Functions
#####################


def remove_events(edf_file, filtered_epochs):
    """
    Remove specific events from the filtered epochs based on annotations in the EDF file.

    Parameters:
    - edf_file (str): The path to the EDF file.
    - filtered_epochs (array-like): The array of filtered epochs.

    Returns:
    - filtered_epochs (array-like): The updated array of filtered epochs after removing specific events.
    """
    # Read the raw EDF file
    raw = mne.io.read_raw_edf(edf_file)

    # Convert annotations to a DataFrame
    annos = raw.annotations.to_data_frame()

    # Get the start time of the recording
    record_start_dt = raw.info['meas_date'].replace(tzinfo=None)

    # Remove photic stimulation epochs
    photo_start_df = annos[annos['description'].str.contains('Photic Begin', case=False, na=False)]
    photic_end_df = photo_end_dt = annos[annos['description'].str.contains('Photic End', case=False, na=False)]
    if not photo_start_df.empty:
        photo_start_dt = annos[annos['description'].str.contains('Photic Begin', case=False, na=False)].onset.iloc[0]
        photo_start_sec = (photo_start_dt - record_start_dt).total_seconds()
        #Sometimes there is a start and end listed, sometimes only a start (possible annotated after end of recording, so removed)
        if not photic_end_df.empty:
            photo_end_dt = annos[annos['description'].str.contains('Photic End', case=False, na=False)].onset.iloc[0]
            photo_end_sec = (photo_end_dt - record_start_dt).total_seconds()
            filtered_epochs = filtered_epochs[(filtered_epochs > photo_end_sec ) | (filtered_epochs < photo_start_sec)]
        else:
            filtered_epochs = filtered_epochs[ filtered_epochs < photo_start_sec]

    # Remove photic stimulation epochs listed in channel "Photic Stimulation"
    photo_event_df, photo_event_sec =  all_events(edf_file, event_ch_names = ["Photic"], sf=256.0)
    if not photo_event_df.empty:
        photo_start_sec = min(photo_event_sec)
        photo_end_sec = max(photo_event_sec) + 20
        filtered_epochs = filtered_epochs[(filtered_epochs > photo_end_sec ) | (filtered_epochs < photo_start_sec)]

    # Remove HV epochs
    hv_start_df = annos[annos['description'].str.contains('HV Begin', case=False, na=False)]
    hv_end_df = annos[annos['description'].str.contains('HV End', case=False, na=False)]

    if not hv_start_df.empty:
        hv_start_dt = annos[annos['description'].str.contains('HV Begin', case=False, na=False)].onset.iloc[0]
        hv_start_sec = (hv_start_dt - record_start_dt).total_seconds()

        if not hv_end_df.empty:
            hv_end_dt = annos[annos['description'].str.contains('HV End', case=False, na=False)].onset.iloc[0]
            hv_end_sec = (hv_end_dt - record_start_dt).total_seconds() + 120

            filtered_epochs = filtered_epochs[(filtered_epochs > hv_end_sec ) | (filtered_epochs < hv_start_sec)]
        else:
            filtered_epochs = filtered_epochs[filtered_epochs < hv_start_sec]

    # Remove Sleep epochs
    sleep_anno = annos[
        annos['description'].str.contains('sleep', case=False) &
        ~annos['description'].str.contains("not asleep|not sleep|isn't asleep|out of sleep|try|sleep deprive|promote sleep|help pt fall asleep|allow|report|not fallen asleep", case=False)
    ]
    if not sleep_anno.empty:
        sleep_start_dt = min(sleep_anno['onset'])
        sleep_start_sec = (sleep_start_dt - record_start_dt).total_seconds()

        # List of wake-related substrings
        wake_strings = ["wake", "woke", "not sleep", "not asleep", "isn't asleep",
                        "isnt asleep", "alert", "out of sleep", "awk"]

        # Join the wake_strings into a single regex pattern with case-insensitive matching
        pattern = '|'.join(wake_strings)

        # Filter rows where 'description' contains any of the wake-related substrings
        wake_anno = annos[annos['description'].str.contains(pattern, case=False, na=False)]
        if not wake_anno.empty:
            wake_start_dt = max(wake_anno['onset'])
            wake_start_sec = (wake_start_dt - record_start_dt).total_seconds()
            filtered_epochs = filtered_epochs[(filtered_epochs > wake_start_sec ) | (filtered_epochs < sleep_start_sec)]
        else:
            filtered_epochs = filtered_epochs[(filtered_epochs < sleep_start_sec)]

    # Remove seizures
    seizure_anno = annos[
        annos['description'].str.contains('seiz', case=False) &
        ~annos['description'].str.contains('no seiz', case=False)&
        ~annos['description'].str.contains('not seiz', case=False)&
        ~annos['description'].str.contains('not a seiz', case=False)&
        ~annos['description'].str.contains('no clera seizure', case=False)&
        ~annos['description'].str.contains('no clear seizure', case=False)&
        ~annos['description'].str.contains("don't look like seizure", case=False)&
        ~annos['description'].str.contains('persyst', case=False)
    ]

    if not seizure_anno.empty:
        seize_start_dt = min(seizure_anno['onset'])
        seiz_start_sec = (seize_start_dt - record_start_dt).total_seconds() - 60
        filtered_epochs = filtered_epochs[(filtered_epochs < seiz_start_sec)]

    return filtered_epochs



def find_events(raw_mne):
    """ Find first index of all patient events (e.g., photic stimulation)

    Args:
        raw_mne (mne): the raw mne file
        note, it is ok if it has been converted to Volts

    Returns:
        integer: the index of every patient events
    """
    chan_names = raw_mne.ch_names
    event_channels = [s for s in chan_names if "Event" in s]
    raw_events = raw_mne.copy().pick_channels(event_channels).get_data()
    # event_ch_dict = {i:ch for ch, i in enumerate(event_channels)}

    raw_events.shape
    all_events = []
    for i in range(raw_events.shape[0]):
        itemindex = np.where(raw_events[i,] != 0)
        if itemindex[0].size > 0:
            f_event = min(itemindex[0])
            # print(f_event)
            all_events.append(f_event)
    return all_events

def all_events(edf_file, event_ch_names = ["event", "Event"], sf=256.0):
    """ Find time of all events (e.g., photic stimulation)

    Args:
        raw_mne (mne): the raw mne file
        note, it is ok if it has been converted to Volts

    Returns:
        cur_event_df (pandas df): a dataframe of specfic information on all events
        itemsec (list): seconds of each event
    """
    cur_pat = edf_file.rsplit('/', 1)[1].split('_', 1)[0]
    raw = mne.io.read_raw_edf(edf_file)

    chan_names = raw.ch_names
    event_channels = [s for s in chan_names if any(x in s for x in event_ch_names)]

    raw_events = raw.copy().pick_channels(event_channels).get_data()
    itemindex = []
    itemsec = []
    e_type = []

    for e in range(0, len(event_channels)):
        cur_itemindex = np.where(raw_events[e,] != 0)[0]
        itemindex.extend(cur_itemindex)
        itemsec.extend([i/int(sf) for i in cur_itemindex])
        e_type.extend(np.repeat(event_channels[e],len(cur_itemindex)))
    pat = np.repeat(cur_pat,len(itemindex))
    gene = np.repeat(edf_file.split("/")[6], len(itemindex))
    cur_event_df = pd.DataFrame({'pat_id': pat, 'event_type': e_type, 'e_index': itemindex, 'e_sec': itemsec,
                                'gene': gene})
    return cur_event_df, itemsec


def get_annos_df(edf_file):


    file_string = edf_file.replace("/mnt/isilon/", "/Volumes/")

    if "ENGIN_" in edf_file:
        file_string = file_string.replace('ENGIN_', 'ENGIN')
    if "_CROPPED" in edf_file:
        file_string = file_string.replace('_CROPPED', '')
    if "_manual" in edf_file:
        file_string = file_string.replace('_manual', '')
    
    parts = file_string.split('/')
    gene =  parts[6]
    cur_pat =  parts[7]

    #Get date from file path
    # Find the index of the second "_" character
    first_underscore_index = file_string.find('_')
    second_underscore_index = file_string.find('_', first_underscore_index + 1)
    # Extract everything after the second "_" character
    extracted_part = file_string[second_underscore_index + 1:]
    # Remove ".edf"
    extracted_part = extracted_part.replace('.edf', '')
    # Replace "_" with "/"
    file_date = extracted_part.replace('_', '/')

    #For some reason this doesn't work with some files
    # may be due to annotations with weird characters
    # annos  = mne.read_annotations(edf_file).to_data_frame()
    raw = mne.io.read_raw_edf(edf_file)
    annos = raw.annotations.to_data_frame() 
    annos['gene'] = gene
    annos['pat_id'] = cur_pat
    annos['date'] = file_date

    return annos


#####################
# Get Clean Segments Functions
#####################

def get_annos_df(edf_file):
    """
    Retrieves annotations from an EDF file and returns them as a pandas DataFrame.

    Parameters:
        - edf_file (str): The path to the EDF file.

    Returns:
        - annos (pandas.DataFrame): The annotations extracted from the EDF file, including additional columns for gene, patient ID, and date.
    """

    parts = edf_file.split('/')
    cur_pat =  parts[7]

    #Get date from file path
    # Find the index of the second "_" character
    first_underscore_index = edf_file.find('_')
    second_underscore_index = edf_file.find('_', first_underscore_index + 1)
    # Extract everything after the second "_" character
    extracted_part = edf_file[second_underscore_index + 1:]
    # Remove ".edf"
    extracted_part = extracted_part.replace('.edf', '')
    # Replace "_" with "/"
    file_date = extracted_part.replace('_', '/')

    gene = edf_file.split("/")[6]
    cur_pat = edf_file.rsplit('/', 1)[1].split('_', 1)[0]
    #For some reason this doesn't work with some files
    # may be due to annotations with weird characters
    # annos  = mne.read_annotations(edf_file).to_data_frame()
    raw = mne.io.read_raw_edf(edf_file)
    annos = raw.annotations.to_data_frame() 
    annos['gene'] = gene
    annos['pat_id'] = cur_pat
    annos['date'] = file_date

    return annos
    

def extract_sequences(arr, seq_length):
    """
    Extracts non-overlapping sequential sequences of length seq_length from arr.

    Parameters:
        arr (numpy.ndarray): Array of epochs that met certain filter criteria.
        seq_length (int): Length of the sequential sequences to extract.

    Returns:
        list: A list of non-overlapping sequential sequences of length seq_length.
    """
    sequences = []
    current_sequence = []
    for value in arr:
        # Start a new sequence if current_sequence is empty
        if not current_sequence:
            current_sequence.append(value)
        # Append the current sequence to sequences if it has reached the required length
        elif len(current_sequence) == seq_length:
            sequences.append(current_sequence)
            current_sequence = [value]  # Start a new sequence with the current value
        # Add value to the current sequence if it is consecutive
        elif value == current_sequence[-1] + 1 and len(current_sequence) < seq_length:
            current_sequence.append(value)
        # Start a new sequence with the current value if it is not consecutive
        else:
            current_sequence = [value]

    # Check if the last sequence meets the required length and is not already added
    if len(current_sequence) == seq_length:
        sequences.append(current_sequence)

    return sequences


def epoch_rejection_saby(filt_data, fs, winDisp, winLen, max_amp=500):
    """
    Rejects epochs based on the RMS amplitude, the line length, and amplitude of the signal.
    Function is based off of that as described in Saby et al. 2022 (https://doi.org/10.1093/braincomms/fcac197)
    Input:
        filt_data (ndarray): Filtered EEG data of shape (num_channels, num_samples)
        fs (float): Sampling frequency of the EEG data
        winDisp (float): Stride of the window, in seconds
        winLen (float): Length of window, in seconds
    Returns:
        inlier_epoch (ndarray): Array of indices of epochs that are not rejected
    """
    bin_n = winLen * fs
    num_wins = math.floor((filt_data.shape[1]-1) / bin_n)

    epoch_ch = moving_window(filt_data, fs, winDisp, winLen, num_wins)

    # Step 1: Calculate the RMS amplitude for each channel in each epoch
    rms_amplitudes = np.sqrt(np.mean(epoch_ch**2, axis=2))
    ch_lln = np.apply_along_axis(LLFn, 2, epoch_ch)

    rms_amplitudes = rms_amplitudes.T  # Transpose the data
    ch_lln = ch_lln.T  # Transpose the data

    # Step 2: Calculate the mean across epochs for each channel
    mean_rms_amplitudes = np.mean(rms_amplitudes, axis=1)
    mean_ch_lln = np.mean(ch_lln, axis=1)

    mean_rms_amplitudes = np.repeat(mean_rms_amplitudes[ :, np.newaxis], rms_amplitudes.shape[1], axis=1)
    mean_mean_ch_lln = np.repeat(mean_ch_lln[ :, np.newaxis], rms_amplitudes.shape[1], axis=1)


    # Step 3: Calculate the standard deviation of RMS values across epochs for each channel
    std_rms_amplitudes = np.std(rms_amplitudes, axis=1)
    std_ch_lln = np.std(ch_lln, axis=1)

    std_rms_amplitudes = np.repeat(std_rms_amplitudes[ :, np.newaxis], rms_amplitudes.shape[1], axis=1)
    std_ch_lln = np.repeat(std_ch_lln[ :, np.newaxis], rms_amplitudes.shape[1], axis=1)

    # Step 5: Find epochs where all channels are less than or equal to 2 standard deviations away
    inlier_epochs_rms = np.where(np.all(np.abs(rms_amplitudes - mean_rms_amplitudes) <=  2* std_rms_amplitudes, axis=0))[0]
    inlier_epochs_lln = np.where(np.all(np.abs(ch_lln - mean_mean_ch_lln) <=  2* std_ch_lln, axis=0))[0]
    
    # orginal only had this one
    # inlier_epoch = np.intersect1d(inlier_epochs_rms, inlier_epochs_lln)

    inlier_epochs_amp = np.where(~np.any(np.abs(epoch_ch) > max_amp, axis=(1, 2)))[0]

    inlier_epoch = np.intersect1d(np.intersect1d(inlier_epochs_rms, inlier_epochs_lln), inlier_epochs_amp)


    return inlier_epoch

#########################
# Artifact Rejection
#########################

def remove_art_ica(raw, ica_comp=20):
    """ Remove articacts such as eye blinks and muscles using ICA

    Args:
        raw (MNE object): MNE object from raw edf file
    Returns:
        reconst_raw (MNE object): MNE object with artifacts removed
    """
    montage = mne.channels.make_standard_montage('standard_1020')
    # Create a list of channel names to exclude
    exclude_channels = [ch_name for ch_name in raw.info['ch_names'] if 'Event' not in ch_name]

    # Create a new info object without the excluded channels
    raw = raw.pick_channels(exclude_channels)

    # Define the desired channel types
    channel_types = {
    'LOC': 'eog',
    'ROC': 'eog',
    'EMG1': 'emg',
    'EMG1': 'emg',
    'EMG2': 'emg',
    'EKGL': 'ecg',
    'EKGR': 'ecg',
    'T1': 'misc',
    'T2': 'misc',
    'Nasion': 'misc',
    'CHIN1': 'emg',
    'CHIN2': 'emg',
    'ECGL': 'ecg',
    'ECGR': 'ecg',
    'LAT1': 'emg',
    'LAT2': 'emg',
    'C3': 'eeg', 
    'C4': 'eeg', 
    'Cz': 'eeg', 
    'F3': 'eeg', 
    'F4': 'eeg', 
    'F7': 'eeg', 
    'F8': 'eeg', 
    'Fz': 'eeg', 
    'Fp1': 'eeg',
    'Fp2': 'eeg',
    'Fpz': 'eeg',
    'A1': 'eeg',
    'A2': 'eeg',
    'O1': 'eeg',
    'O2': 'eeg',
    'Oz': 'eeg',
    'P3': 'eeg',
    'P4': 'eeg',
    'T5': 'eeg',
    'T6': 'eeg',
    'Pz': 'eeg',
    'T3': 'eeg',
    'T4': 'eeg' 
    }

    ########
    # Set channel types based on the dictionary
    for channel_name in raw.info['ch_names']:
        if channel_name in channel_types.keys():
            channel_type = channel_types[channel_name]
        else:
            channel_type = 'misc'  # Set to 'misc' if not found in the dictionary
        raw.set_channel_types({channel_name: channel_type})
    raw.set_montage(montage)

    # They recommend 1-100 hz filter, due to anti-aliasing filter setting to 95
    raw.filter(0.5, 95, method='iir')  # Adjust the filter parameters as needed

    ica = ICA(n_components=ica_comp, random_state=97, max_iter=800)
    ica.fit(raw)
    ic_labels = label_components(raw, ica, method="iclabel")
    labels = ic_labels['labels']
    y_pred_proba = ic_labels['y_pred_proba']
    exclude_idx = [
        idx
        for idx, (label, proba) in enumerate(zip(labels, y_pred_proba))
        if label not in ["brain", "other"] or (label == "other" and proba <= 0.30)
    ]
    reconst_raw = raw.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)

    return(reconst_raw)



#####################
# Features Functions
#####################

#Line Length
def LLFn(x):
  """
    Calculate the line length (sum of absolute differences) between consecutive elements of an array.
    Parameters:
        x (array-like): Input array.
    Returns:
        float: The sum of absolute differences between consecutive elements of the input array.
  """
  return np.sum(np.abs(np.ediff1d(x)),axis=-1)

# Function to compute area

def area(x) : 
  """
    Computes area of given input signal. 
    Area of a signal [x0,x1,...,xn] is given by |x0| + |x1| + ... + |xn|
    Input: 
        x (ndarray): Input signal of shape (num_channels, num_samples) or (num_samples)
    Returns:
        ndarray or float of the areas of the signals in each channel. For input shape of (num_channels, num_samples) output is 
        ndarray of shape (num_channels). For input shape (num_samples) output is a float
  """

  return np.sum(np.abs(x), axis=-1)

# Function to compute energy 

def energy(x) :
  """
    Computes energy of given input signal.
    Energy of a signal [x0,x1,...,xn] is given by (x0)^2 + (x1)^2 + ... + (xn)^2
    Input: 
        x (ndarray): Input signal of shape (num_channels, num_samples) or (num_samples)
    Returns:
        ndarray or float of the energies of the signals in each channel. For input shape of (num_channels, num_samples) output is 
        ndarray of shape (num_channels). For input shape (num_samples) output is a float
  """
  return np.sum(x**2, axis=-1)

# Function to compute number of zero crossings

def zero_crossings(x) :
  """
  Computes number of zero crossings of given input signal.
  Input: 
    x (ndarray): Input signal of shape (num_channels, num_samples) or (num_samples)
  Returns:
    ndarray or int of the number of zero crossings of the signals in each channel. For input shape of (num_channels, num_samples) 
    output is ndarray of shape (num_channels). For input shape (num_samples) output is an int
  """

  y = x - np.mean(x, axis=-1).reshape(-1, 1)
  return np.sum(np.abs(np.diff(np.where(y > 0, 1, 0), axis=-1)), axis=-1)


#####################
# Bandpower Functions
#####################




def find_spectral_peaks(frequencies, power_db_smooth, fs):
    """
    Identifies and characterizes significant peaks in smoothed power spectral density data.

    This function uses the `find_peaks` method from scipy.signal to locate peaks in a power spectrum that meet specified criteria for prominence and minimum distance between peaks. Additional properties of the peaks such as their widths and heights are also computed and returned.

    Parameters:
    - frequencies (array_like): Array of frequency values corresponding to the power spectrum.
    - power_db_smooth (array_like): Smoothed power spectral density values in decibels.
    - fs (float): Sampling frequency of the original time series data from which the spectrum was derived.

    Returns:
    - dict: A dictionary containing arrays of characteristics for each identified peak:
      - 'peak_freqs': Frequencies at which peaks occur.
      - 'peak_powers': Power values at the peaks in dB.
      - 'prominences': Prominence values of the peaks.
      - 'widths': The widths of the peaks.
      - 'width_heights': Heights of the peaks at their respective widths.
      - 'left_bases': Frequencies at the left bases of the peaks.
      - 'right_bases': Frequencies at the right bases of the peaks.
      - 'peak_heights': Absolute heights of the peaks.

    Details:
    - The `distance` parameter is calculated to ensure a minimum separation of 0.5 Hz between peaks.
    - The function pads properties arrays with NaN where necessary to ensure all are of equal length, aiding in alignment when constructing the result.

    Example:
    >>> frequencies = np.linspace(0, 50, 500)
    >>> power_db_smooth = np.random.random(500) * 100
    >>> fs = 1000
    >>> peak_info = find_spectral_peaks(frequencies, power_db_smooth, fs)
    >>> print(peak_info['peak_freqs'])  # frequencies of identified peaks
    """
    # distance = int(fs * 0.5 / 0.25)  # Convert 0.5 Hz minimum distance to index count
    # Define window length for 4 seconds
    win_length = int(4 * fs)
    # 50% overlap
    # Setting nfft to win_length for a resolution of 0.25Hz
    nfft = win_length

    distance = int(0.5 / (fs / nfft))  # This simplifies to 2 as calculated

    peaks, properties = sig.find_peaks(power_db_smooth, prominence=5, distance=distance, width=0, height=None)

    # Ensure all properties have the same length, pad with NaNs if necessary
    max_length = len(peaks)
    properties_keys = ['prominences', 'widths', 'width_heights', 'left_bases', 'right_bases', 'peak_heights']
    for key in properties_keys:
        if key not in properties or len(properties[key]) < max_length:
            properties[key] = np.full(max_length, np.nan)  # Fill with NaNs where data is missing

    results = {
        'peak_freqs': frequencies[peaks],
        'peak_powers': power_db_smooth[peaks],
        'prominences': properties['prominences'],
        'widths': properties['widths'],
        'width_heights': properties['width_heights'],
        'left_bases': properties['left_bases'],
        'right_bases': properties['right_bases'],
        'peak_heights': properties['peak_heights']
    }
    return results


def find_dom_freq_welch(eeg_signal, sf=256.0):
    """Get the domenant frequency using welch

    Args:
        eeg_signal (1D numpy array): _description_
        sf (float, optional): _description_. Defaults to 256.0.

    Returns:
        dom_freq (float): dominant frequency of signal
        dom_value (float): dominant value of signal
    """

    f, pxx = sig.welch(eeg_signal,fs = sf)
    dom_freq = f[np.argmax(pxx)]
    dom_value = np.amax(pxx)

    return(dom_freq, dom_value)


def get_savgol_pdr_peaks(edf_file, ica = True, target_sf = 200, nfft=0.25):    
    """
    Processes EEG data from an EDF file to identify and characterize the posterior dominant rhythm (PDR) across specified channels using Welch's method, Savitzky-Golay smoothing, and peak detection.

    Inputs:
        edf_file (str): Path to the EDF file containing EEG data.
        ica (bool, optional): Indicates whether Independent Component Analysis (ICA) should be applied to remove artifacts. Defaults to True.

    Steps:
    1. Reads EEG data from the specified EDF file.
    2. Optionally applies ICA to remove artifacts if 'ica' is set to True.
    3. Performs signal preprocessing including resampling, notch filtering, and band-pass filtering.
    4. Applies Welch's method to estimate the power spectral density of the EEG signals.
    5. Converts the power estimates to decibel scale and smooths them using a Savitzky-Golay filter.
    6. Identifies peaks in the smoothed power spectra which represent the PDR, using specific criteria for peak prominence and distance.
    7. Analyzes the epochs and channels of interest, particularly focusing on 'O1' and 'O2' channels for PDR.
    8. Collects and returns the spectral data and peak characteristics in a structured format.

    Returns:
    tuple of pandas.DataFrame: Two dataframes are returned:
        results_df: Contains the peaks' characteristics including epoch, channel, frequency, power, prominence, and width.
        spectrum_df: Contains detailed spectral data for each epoch and channel including frequency, raw power, power in dB, and smoothed power in dB.

    Example usage:
    >>> results_df, spectrum_df = get_savgol_pdr_peaks('/path/to/your/data.edf')
    """
    winLen = 4
    ica_comp = 20
    #Get date from file path
    file_string = edf_file.replace("/mnt/isilon/", "/Volumes/")
    file_string = edf_file.replace("controls_auto", "controls")

    parts = file_string.split('/')
    Gene =  parts[6]
    cur_pat =  parts[8]

    #Get date from file path
    # Find the index of the second "_" character

    first_underscore_index = file_string.find('_')
    second_underscore_index = file_string.find('_', first_underscore_index + 1)
    # Extract everything after the second "_" character
    extracted_part = file_string[second_underscore_index + 1:]
    # Remove ".edf"
    extracted_part = extracted_part.replace('.edf', '')
    # Replace "_" with "/"
    file_date = extracted_part.replace('_', '/')

    raw = mne.io.read_raw_edf(edf_file, verbose = False)

    #  Maximum duration in seconds (4 hours)
    max_duration = 4 * 3600 +0.1

    # Get the duration of the recording
    recording_duration = raw.times[-1]


    # Crop the recording if it is longer than 4 hours
    if recording_duration > max_duration:
        raw.crop(tmax=max_duration)

        
    if "ENDD" in edf_file:
        raw =  endd_edf_chan_map(raw)
        ica_comp = 18
        
    raw.load_data()

    sf = raw.info['sfreq']

    if sf != target_sf:
        if sf > target_sf:
            # Apply notch filter to remove power line noise and its harmonics
            notch_harmonics = generate_harmonics(60, sf)
            rawraw = raw.copy().notch_filter(freqs=notch_harmonics)

            # Define the low-pass filter cutoff as just below half the target sampling frequency (Nyquist frequency)
            low_pass = target_sf / 2 - 5  # 95 Hz if target is 200 Hz

            # Apply a low-pass filter to prevent aliasing
            rawraw.filter(l_freq=None, h_freq=low_pass, method='iir')

            # Resample the data to the target sampling frequency
            rawraw.resample(sfreq=target_sf)
    else:
        notch_harmonics = generate_harmonics(60, sf)
        rawraw = raw.copy().notch_filter(freqs=notch_harmonics)
    
    sf = target_sf
    
    alt_channels = ['T7', 'T8', 'P7', 'P8']

    raw_volts = rawraw.apply_function(lambda x: x * 1e6)
    ch_names = raw_volts.info['ch_names']
    # T3 is now T7
    # T4 is now T8
    # T5 is now P7
    # T6 is now P8
    if any(x in ch_names for x in alt_channels):
        raw_volts.rename_channels({'T7':'T3', 'T8':'T4', 'P7':'T5', 'P8':'T6'})
    if ica:
        raw_volts = remove_art_ica(raw_volts,ica_comp)
            
    raw_sub= raw_volts.copy().pick_channels(sub_ch)
    sub_ch_names = raw_sub.info['ch_names']
    ch_dict = {i:ch for ch, i in enumerate(sub_ch_names)}

    raw_np = raw_sub.copy().get_data()


    order = 2
    low_cutoff = 1
    high_cutoff = 70
    b,a  = create_band_pass_filter(order,  low_cutoff, high_cutoff, sf) 
    filt_data = apply_filter(raw_np, [b,a])
    # raw_lap = np.apply_along_axis(laplac, 0, filt_data, ch_dict)
    raw_com_avg = np.apply_along_axis(common_average_reference, 0, filt_data)

    # Removing bad segments
    seq_len = 4
    winDisp_filt = 1
    winLen_filt = 1
    winLen = 4
    bin_n = winLen_filt * sf

    scan_durn = raw_volts._data.shape[1] / raw_volts.info['sfreq']

    num_wins = math.floor((raw_volts._data.shape[1]-1) / bin_n)

    # Remove A1 and A2 channels for epoch rejection
    rm_elects = [ch_dict['A1'],ch_dict['A2']]
    raw_com_avg_sub = np.delete(raw_com_avg, rm_elects, axis=0)

    inlier_epoch = epoch_rejection_saby(raw_com_avg_sub, sf, winDisp_filt, winLen_filt)
    # sequences = extract_sequences(inlier_epoch, seq_len)

    # Main data in 1 second time bins
    epoch_ch_1sec = moving_window(raw_com_avg, sf, winDisp_filt, winLen_filt, num_wins)

    # Remove events
    inlier_epoch_filt = remove_events(edf_file, inlier_epoch)

    #Place into winLen bins
    sequences = extract_sequences(inlier_epoch_filt, winLen)

    first_elements = [subarray[0] for subarray in sequences]

    epoch_ch_full = np.array([np.concatenate(epoch_ch_1sec[epochs], axis=1) for epochs in sequences])
    pdr_elects = [ch_dict['O1'],ch_dict['O2']]
    epoch_ch = epoch_ch_full[:,pdr_elects,:]

    results = []

    results = []
    spectrum_data = []

    for epoch_idx in range(epoch_ch.shape[0]):
        for channel_idx in range(epoch_ch.shape[1]):
            channel_data = epoch_ch[epoch_idx, channel_idx, :]
            frequencies, power, power_db, power_db_smooth = spectral_estimation(channel_data, sf, nfft)
            
            # Save spectrum data for each epoch
            for freq, pwr, pwr_db, pwr_db_smooth in zip(frequencies, power, power_db, power_db_smooth):
                spectrum_data.append({
                    'patient' : cur_pat,
                    'epoch': epoch_idx,
                    'channel': channel_idx,
                    'frequency': freq,
                    'raw_power': pwr,  # Raw power from Welch's method
                    # 'power_db': pwr_db,  # Power in dB
                    'power_db_smooth': pwr_db_smooth  # Smoothed power in dB
                })

            peak_results = find_spectral_peaks(frequencies, power_db_smooth, sf)

            # Assuming that the necessary libraries are already imported and setup is done
            for i in range(len(peak_results['peak_freqs'])):
                results.append({
                    'patient': cur_pat,
                    'epoch': epoch_idx,
                    'channel': channel_idx,
                    'frequency': peak_results['peak_freqs'][i],
                    'power': peak_results['peak_powers'][i],
                    'prominence': peak_results['prominences'][i],
                    'width': peak_results['widths'][i],
                    'width_height': peak_results['width_heights'][i] if 'width_heights' in peak_results else np.nan,
                    'left_base': peak_results['left_bases'][i] if 'left_bases' in peak_results else np.nan,
                    'right_base': peak_results['right_bases'][i] if 'right_bases' in peak_results else np.nan,
                    'peak_height': peak_results['peak_heights'][i] if 'peak_heights' in peak_results else np.nan
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    spectrum_df = pd.DataFrame(spectrum_data)

    return results_df, spectrum_df



def calculate_median_psd(ch_np, sf, step_size_hz = 1.0):
    """
    Calculate the mean Power Spectral Density (PSD) for a given electrode across intervals\epochs.

    Parameters:
        ch_np (numpy.ndarray): EEG data of shape (num_intervals, num_samples).
        sf (float): The sampling frequency.
        step_size_hz (float): The desired step size in Hz.

    Returns:
        frequencies (numpy.ndarray): The frequencies.
        median_psd (numpy.ndarray): The median PSD values.
    """

    # Calculate the number of samples corresponding to the desired step size
    nperseg_samples = int(sf / step_size_hz)

    # Choose a nperseg based on  desired step size
    noverlap_samples =  nperseg_samples - 1  
    
    mean_psd_values = []
    frequencies = None
    num_intervals = ch_np.shape[0]
    for interval_idx in range(num_intervals):
        f, psd = sig.welch(ch_np[interval_idx], fs=sf, noverlap=noverlap_samples, nperseg=nperseg_samples)
        frequencies = f  # Frequencies are the same for all intervals
        mean_psd_values.append(psd)
    
    median_psd = np.median(mean_psd_values, axis=0)
    
    return frequencies, median_psd


#Main function
def get_np_psd_channels_med(edf_file, winLen=4, ica = True, target_sf = 200):
    """
    Cleans the scalp EEG data from an EDF file and calculates the median power spectral density (PSD) 
    for each electrode.

    Parameters:
    - edf_file (str): The path to the EDF file containing the scalp EEG data.
    - winLen (int, optional): The length of the time window in seconds for calculating the PSD. 
        Defaults to 4.
    - ica (bool, optional): Whether to apply independent component analysis (ICA) for artifact removal. 
        Defaults to True.
    - target_sf (int, optional): The target sampling frequency in Hz. Defaults to 200.

    Returns:
    - pat_epochs (pandas.DataFrame): A DataFrame containing information about the patient, date, gene, 
        number of epochs, epochs, and amplitude filter.
    - pat_eo (pandas.DataFrame): A DataFrame containing the patient, date, gene, frequency, power, 
        electrode, and number of epochs for each electrode.

    Note:
    - The function assumes that the EDF file is in a specific format and follows certain naming conventions.
    - The function performs various preprocessing steps such as cropping the recording, resampling the data, 
        applying notch and band-pass filters, removing bad segments, and removing events.
    - The function calculates the median PSD for each electrode using Welch's method.
    """

    file_string = edf_file.replace("/mnt/isilon/", "/Volumes/")
    if "ENGIN_" in edf_file:
        file_string = file_string.replace('ENGIN_', 'ENGIN')
    if "_CROPPED" in edf_file:
        file_string = file_string.replace('_CROPPED', '')
    if "_manual" in edf_file:
        file_string = file_string.replace('_manual', '')

    parts = file_string.split('/')
    Gene =  parts[6]
    cur_pat =  parts[7]

    #Get date from file path
    # Find the index of the second "_" character
    first_underscore_index = file_string.find('_')
    second_underscore_index = file_string.find('_', first_underscore_index + 1)
    # Extract everything after the second "_" character
    extracted_part = file_string[second_underscore_index + 1:]
    # Remove ".edf"
    extracted_part = extracted_part.replace('.edf', '')
    # Replace "_" with "/"
    file_date = extracted_part.replace('_', '/')

    raw = mne.io.read_raw_edf(edf_file)
    #  Maximum duration in seconds (4 hours)
    max_duration = 4 * 3600 + 0.1

    # Get the duration of the recording
    recording_duration = raw.times[-1]

    # Crop the recording if it is longer than 4 hours
    if recording_duration > max_duration:
        raw.crop(tmax=max_duration)


    raw.load_data()

    sf = raw.info['sfreq']

    if sf != target_sf:
        if sf > target_sf:
            # Apply notch filter to remove power line noise and its harmonics
            notch_harmonics = generate_harmonics(60, sf)
            rawraw = raw.copy().notch_filter(freqs=notch_harmonics)

            # Define the low-pass filter cutoff as just below half the target sampling frequency (Nyquist frequency)
            low_pass = target_sf / 2 - 5  # 95 Hz if target is 200 Hz

            # Apply a low-pass filter to prevent aliasing
            rawraw.filter(l_freq=None, h_freq=low_pass, method='iir')

            # Resample the data to the target sampling frequency
            rawraw.resample(sfreq=target_sf)
    else:
        notch_harmonics = generate_harmonics(60, sf)
        rawraw = raw.copy().notch_filter(freqs=notch_harmonics)

    # Set the new sampling frequency in the info structure
    sf = target_sf
    
    alt_channels = ['T7', 'T8', 'P7', 'P8']

    raw_volts = rawraw.apply_function(lambda x: x * 1e6)
    ch_names = raw_volts.info['ch_names']
    # T3 is now T7
    # T4 is now T8
    # T5 is now P7
    # T6 is now P8
    if any(x in ch_names for x in alt_channels):
        raw_volts.rename_channels({'T7':'T3', 'T8':'T4', 'P7':'T5', 'P8':'T6'})
    if ica:
        raw_volts = remove_art_ica(raw_volts)
            
    raw_sub= raw_volts.copy().pick_channels(sub_ch)
    sub_ch_names = raw_sub.info['ch_names']
    ch_dict = {i:ch for ch, i in enumerate(sub_ch_names)}


    raw_np = raw_sub.copy().get_data()

    order = 2
    low_cutoff = 0.5
    # low_cutoff = 1
    high_cutoff = 70
    b,a  = create_band_pass_filter(order,  low_cutoff, high_cutoff, sf) 
    filt_data = apply_filter(raw_np, [b,a])
    raw_lap = np.apply_along_axis(laplac, 0, filt_data, ch_dict)

    # Removing bad segments
    seq_len = 4
    winDisp_filt = 1
    winLen_filt = 1

    bin_n = winLen_filt * sf

    scan_durn = rawraw._data.shape[1] / sf

    num_wins = math.floor((rawraw._data.shape[1]-1) / bin_n)


    # Remove A1 and A2 channels for epoch rejection
    rm_elects = [ch_dict['A1'],ch_dict['A2']]
    raw_lap_sub = np.delete(raw_lap, rm_elects, axis=0)

    inlier_epoch = epoch_rejection_saby(raw_lap_sub, sf, winDisp_filt, winLen_filt)

    # Main data in 1 second time bins
    epoch_ch_1sec = moving_window(raw_lap, sf, winDisp_filt, winLen_filt, num_wins)

    # Remove events
    inlier_epoch_filt = remove_events(edf_file, inlier_epoch)

    #Place into winLen bins
    sequences = extract_sequences(inlier_epoch_filt, winLen)
    epoch_starts = [i[0] for i in sequences]

    epoch_ch = np.array([np.concatenate(epoch_ch_1sec[epochs], axis=1) for epochs in sequences])

    col_names = ['patient', 'date', 'gene', 'frequency', 'power', 'electrode', 'epochs']
    rows = len(sub_ch_names) * 70 #257 is the number of frequencies in the welch output
    pat_eo = pd.DataFrame(index = range(rows), columns = col_names)
    pat_eo.set_axis(col_names, axis = 1, inplace = True)
    pat_eo['patient'] = cur_pat
    pat_eo['gene'] = Gene
    pat_eo['date'] = file_date
    pat_eo['epochs'] = len(epoch_starts)

    col_names = ['patient', 'date', 'gene', 'n_epochs', 'epochs', 'amp_filt']
    pat_epochs = pd.DataFrame(index = range(1), columns = col_names)
    pat_epochs['patient'] = cur_pat
    pat_epochs['gene'] = Gene
    pat_epochs['date'] = file_date
    pat_epochs['n_epochs'] = len(epoch_starts)
    pat_epochs['epochs'] = ';'.join(str(i) for i in epoch_starts)
    pat_epochs['amp_filt'] = 500

    #Calculate dominant frequency in each channel and the corresponding peak value
    for ep in range(epoch_ch.shape[1]):
        s_index = ep*70 #257 is the number of frequencies in the welch output
        e_index = s_index + 69
        cur_ch = sub_ch_names[ep]
        pat_eo.loc[s_index:e_index, 'electrode'] = cur_ch
        ch_idx = ch_dict[cur_ch]
        ch_np = epoch_ch[:,ch_idx,:]
        frequencies, med_psd = calculate_median_psd(ch_np, sf)
        mask = np.logical_and(frequencies >= 1, frequencies <= 70)
        pxx = med_psd[mask[0:len(med_psd)]]
        frequencies = frequencies[mask]
        pat_eo.loc[s_index:e_index, 'frequency'] = frequencies
        pat_eo.loc[s_index:e_index, 'power'] = pxx


    return(pat_epochs, pat_eo)