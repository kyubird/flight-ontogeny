#!/usr/bin/env python
# coding: utf-8


import glob
import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
from scipy.fft import fft, fftfreq
from tqdm.notebook import  tqdm, tqdm_notebook





def compile_FT_CSVs(path, calibration=False, year=2022): 
    # Compiles paths to all CSV files found in path folder and subfolders. 
    # 
    # INPUTS - path: string. The parent folder of all CSV files
    #        - calibration: boolean (default False). Special case: set to True if the CSVs are calibration files, not flights.
    
    # OUTPUT - csv_df: Pandas dataframe. Each row corresponds to one CSV file, with filename and information from parent folders stored as columns 

    if calibration == True:
        # get a list of paths to all csv files in subfolderse
        all_csv_filenames = sorted(glob.glob(path + r"*\*.csv"))

        # make a pandas dataframe with all CSV file paths and associated information 
        csv_df = pd.dataframe(columns=["Path","Direction","Distance","Filename"])

        for i, path in enumerate(all_csv_filenames):
            direction = path.split("\\")[-2] 
            filename = path.split("\\")[-1]  
            distance = filename.split("_")[-1][:-4]
            csv_row = pd.dataframe({"Path": path,
                                    "Direction": direction,
                                    "Distance":distance,
                                    "Filename":filename}, 
                                   index=[0])
            csv_df = pd.concat([csv_df, csv_row], axis=0, ignore_index=True)
            
    else:
        if year==2023:
            print("ERROR: wrong helpers for 2023 data - use helpers23.py")
            return None

        # get a list of paths to all csv files in subfolders
        all_csv_filenames = sorted(glob.glob(path + r"*\*\*takeoff*.csv"))
        # changed landing -> takeoff 2025-10-23

        # make a pandas dataframe with all CSV file paths and associated directory information (bird, date and filename)
        csv_df = pd.dataframe(columns=["Path","Bird","Date","Filename"])

        for i, path in enumerate(all_csv_filenames):
            bird = path.split("\\")[-3] # "<ring colour>R-<ring colour>L_<PIT tag>" -> bird identifier (folder name)
            date = path.split("\\")[-2] # "YYMMDD_day<bird age>" 
            filename = path.split("\\")[-1] # "YYMMDD_day<bird age>_<Trial number>_takeoff<number>.csv"
            csv_row = pd.dataframe({"Path": path,"Bird": bird,"Date":date,"Filename":filename}, index=[0])
            csv_df = pd.concat([csv_df, csv_row], axis=0, ignore_index=True)


    ### FOR 2023 year, PATTERNS ARE AS FOLLOWS:
        # `bird`: "<birdID>_<ring colour>R-<ring colour>L" 
        # `date`: "YYMMDD_<birdID>_<birdname>_day<age>" 
        # `filename`: "YYMMDD_<birdID>_<birdname>_day<age>_<Trial number>_XX00.csv"
        # where XX can be either `TR`,`LN`,`NW`,`WA`/`WB`/`WC`,`NA`/`NB`, `MA`/`MB`/`MC`, `UA`/`UB`/`UC`
        # and `00` is a chronological identifier

    print("Found " + str(len(all_csv_filenames)) + " .csv files.")
    print("Compiled " + str(csv_df.shape[0]) + "/"+ str(len(all_csv_filenames))+" CSV filepaths in output dataframe.")

    return csv_df

#df = compile_FT_CSVs("D:\\Vicon Install\\FinchPerching\\nano43_manualCal\\", True)


def get_birdID(bird):
    bird_ID_dict = {'greyR-blackL': 12,
                    'pinkR-blackL': 13,
                    'pinkR-blueL': 14,
                    'purpleR-blackL': 16,
                    'blackR-orangeL': 17,
                    'greenR': 21,
                    'redR-blackL': 22,
                    'cyanR-blackL': 23,
                    'blackR-yellowL': 25,
                    'greenR-purpleL': 26,
                    'greenR-orangeL': 27,
                    'redR-yellowL': 28,
                    'redR-purpleL': 29, 
                    'greenR-whiteL': 30,
                    'yellowR-purpleL': 31, 
                    'greenR-blueL': 32,
                    'blackR-blackL': 33,
                    'yellowL': 34,
                    'yellowR-greyL': 35,
                    'blackRpurpleL': 36,
                    'greenR-greenL': 37,

                   }
    return bird_ID_dict[bird]


def CSVs_to_dataframe(csv_df, year, calibration=False, include_NaNlandings=False, save_as_csv=False, destination = ""):
    # makes pandas dataframe of voltages (Nano43 output) from csv files 
    # INPUTS
    # csv_df: pandas dataframe referencing all the csv files (inc. filepaths) to compile in output dataframe
    # year: int. specify 2022 or 2023 depending on which dataset is used, since the metadata is different for both
    # calibration: boolean. set to true if loading the manual calibration check files.
    # include_NaNlandings: boolean. set to true if want to keep the non-numbered landing flights in the data (e.g. fails)
    #                      NOTE - column 'flightID' is meaningless if include_NaNlandings = True, do not use for indexing
    # save_as_csv: boolean. set to true if want to save the output dataframe as a csv file
    # destination: str. path+filename to specify where to save data if save_as_csv = True
    
    # OUTPUT
    # voltages_df: pandas dataframe containing all metadata + 6 voltage columns of each flight/signal
    
    if calibration==True: # call different function to return a calibration df (different format from flight df)
        voltages_df = calibration_CSVs_to_dataframe(csv_df, save_as_csv, destination)
    
    if year==2023:
        print("ERROR: wrong helpers for 2023 data - use helpers23.py")
        return None

    voltages_df = CSVs_to_dataframe_2022(csv_df, include_NaNlandings, save_as_csv, destination)
    return voltages_df
        

#changed landing -> takeoff    
def CSVs_to_dataframe_2022(csv_df, include_NaNlandings, save_as_csv, destination):

    # make empty list of dictionaries for voltage data
    rows_list = []

    # iterate over all CSV files (in CSV dataframe)
    for index, row in tqdm(csv_df.iterrows(), total=csv_df.shape[0]):    

        filepath = csv_df.at[index, 'Path'] # get path to CSV file
        bird = csv_df.at[index, 'Bird'] # get bird ID
        bird_short = bird.split("_")[0] # get first part of bird ID (bird rings)
        date = csv_df.at[index, 'Date'] # get date in format YYMMDD_day<age>
        age = date[12:14] # get age from date string (NEED CONSISTENT DATE FORMAT)
        if age == '10': # for birds aged 100 days, above method only gets first 2 digits (10) so need to change to 100
            age = '100'
        filename = csv_df.at[index, 'Filename'] # get name of CSV file, format YYMMDD_day00_00_landing*.csv
        match = re.search('takeoff(\d+)', filename) # search for number after 'takeoff' in filename string
        if match:
            takeoff = match.group(1) # return number if found
        else:
            takeoff = None # else return None

        # finally, if filename doesn't end with "<digit>.csv", write the last bit of filename as a note
        if not filename[-5].isdigit(): 
            note = filename.split("_")[-1][:-4]
        else:
            note = None

        # if non-numbered landings should be included, then 'FlightID' is None for all (useless column)
        if include_NaNlandings:
            flightID = None

        else:
            if takeoff: 
                # flightID is a unique int for each flight, formatted as such:
                # BBAAALL where BB:bird ID, AAA: age, LL: landing
                # e.g. 3102010 => bird #31, aged 020 days, landing #10
                flightID = int(str(get_birdID(bird_short)) + str(age).zfill(3) + str(takeoff).zfill(2))
            else: 
                # skip any data row that isn't from a numbered landing
                continue

        # load the CSV data into a pandas dataframe
        df = pd.read_csv(filepath, skiprows=5,  header=None)

        
        
        # compute decimal Time for each datapoint (decimal Time = Time + subTime/8)
        Time = (df[df.columns[0]]).astype('float') + (df[df.columns[1]]).astype('float')/8.

        # make dictionary of new row of data 
        for i, row in enumerate(df.index):
            data_row = { 'FlightID': flightID,
                         'Bird': bird_short, 
                         'Age': age, 
                         'Takeoff': takeoff,
                         'Note': note,
                         'Time': Time[i].astype('float'),
                         'V1': df.iat[i,2].astype('float'),
                         'V2': df.iat[i,3].astype('float'),
                         'V3': df.iat[i,4].astype('float'),
                         'V4': df.iat[i,5].astype('float'),
                         'V5': df.iat[i,6].astype('float'),
                         'V6': df.iat[i,7].astype('float'),
                        }
            # add new row to end of list of dictionaries
            rows_list.append(data_row)

        #print(filepath)

    # convert list of dictionaries to pandas dataframe
    voltages_df = pd.dataframe(rows_list)

    if not include_NaNlandings:
        voltages_df.set_index('FlightID')

    # convert type of Age & Takeoff columns to int
    voltages_df['Age'] = voltages_df['Age'].astype(int)
    voltages_df['Takeoff'] = voltages_df['Takeoff'].astype(np.int64)

    if save_as_csv:
        # save data (write single csv file) to location below
        voltages_df.to_csv(destination, index=False)
        print("Output CSV file saved to: " + destination)


    return voltages_df 




def get_flight_info(flight_ID):
    bird_ID_reverse_dict = {12: 'greyR-blackL',
                            13: 'pinkR-blackL',
                            14: 'pinkR-blueL',
                            16: 'purpleR-blackL',
                            17: 'blackR-orangeL',
                            21: 'greenR',
                            23: 'cyanR-blackL',
                            25: 'blackR-yellowL',
                            26: 'greenR-purpleL',
                            27: 'greenR-orangeL',
                            28: 'redR-yellowL',
                            29: 'redR-purpleL', 
                            30: 'greenR-whiteL',
                            31: 'yellowR-purpleL', 
                            32: 'greenR-blueL',
                            33: 'blackR-blackL',
                            34: 'yellowL',
                            35: 'yellowR-greyL',
                            36: 'blackRpurpleL',
                            37: 'greenR-greenL',}
    ID_str = str(flight_ID)
    flight_info = {'Bird': bird_ID_reverse_dict.get(int(ID_str[:2])),
                   'Age': int(ID_str[2:5]),
                   'Takeoff': int(ID_str[5:]),
                  }
    return flight_info




def calibration_CSVs_to_dataframe(csv_df, save_as_csv, destination):  
    print("Loading " + str(csv_df.shape[0]) + " CSV files:")
    # make empty list of dictionaries for voltage data
    rows_list = []

    # iterate over all CSV files (in CSV dataframe)
    for index, row in tqdm(csv_df.iterrows(), total=csv_df.shape[0]):    

        filepath = csv_df.at[index, 'Path'] # get path to CSV file
        direction = csv_df.at[index, 'Direction'] 
        distance = csv_df.at[index, 'Distance']
        if direction == 'vertical2':
            direction = 'vertical'
        elif direction == 'empty':
            #direction = None
            distance = -1


        # load the CSV data into a pandas dataframe
        df = pd.read_csv(filepath, skiprows=5,  header=None)

        # compute decimal Time for each datapoint (decimal Time = Time + subTime/8)
        Time = (df[df.columns[0]]).astype('float') + (df[df.columns[1]]).astype('float')/8.
        print(df.shape)

        # 
        for i, row in enumerate(df.index):
            data_row = {'Direction': direction, 
                        'Distance': distance, 
                        'Time': Time[i].astype('float'),
                        'V1': df.iat[i,2].astype('float'),
                        'V2': df.iat[i,3].astype('float'),
                        'V3': df.iat[i,4].astype('float'),
                        'V4': df.iat[i,5].astype('float'),
                        'V5': df.iat[i,6].astype('float'),
                        'V6': df.iat[i,7].astype('float'),
                        }
            # add new row to end of list of dictionaries
            rows_list.append(data_row)

    # convert list of dictionaries to pandas dataframe
    voltages_df = pd.dataframe(rows_list)
    
    # convert type of Age & Landing columns to int
    voltages_df['Distance'] = voltages_df['Distance'].astype(int)


    if save_as_csv:
        voltages_df.to_csv(destination, index=False)
        print("Output CSV file saved to: " + destination)
    
    return voltages_df
        




def get_calibration_matrix(calibration):
    
    
    if calibration == 'small' or calibration == 'Small' or calibration == 'SI-9-0.125':
        filepath = r"C:\Users\kmh\Documents\Perching_ontogeny\force_balance\small_cal_T.csv"

    elif calibration == 'large' or calibration == 'Large' or calibration == 'SI-18-0.25':
        filepath = r"C:\Users\kmh\Documents\Perching_ontogeny\force_balance\large_cal_T.csv"

    else:
        print("ERROR: Could not find calibration matrix. Try <calibration = 'small'> or <calibration = 'large'>")
        return None

    cal_df = pd.read_csv(filepath, 
                         header=0,
                         dtype = {'Fx': float, 'Fy': float, 'Fz': float, 'Tx': float,'Ty': float, 'Tz': float})

    # return calibration dataframe
    return cal_df 




def voltage_to_FT(voltages, year, calibration = "SI-9-0.125", cal_data=False, save_as_csv=None):
    
    if isinstance(voltages, str):
        if cal_data:
            voltage_df = pd.read_csv(voltages,
                                     dtype = {'Direction': str, 'Distance': pd.Int64Dtype(), 'Time': float, 
                                              'V1': float,'V2': float,'V3': float,'V4': float,'V5': float,'V6': float})
        elif year==2022: #was Landing, changed to Takeoff 2025-10-23
            voltage_df = pd.read_csv(voltages,
                                     dtype = {'Bird': str, 'Age': int, 'Takeoff': pd.Int64Dtype(), 'Note': str,'Time': float, 
                                              'V1': float,'V2': float,'V3': float,'V4': float,'V5': float,'V6': float})
        elif year==2023:
            voltage_df = pd.read_csv(voltages,
                                     dtype = {'FlightID': str, 'Bird': str, 'BirdID':int, 'Age': int, 
                                              'Category': str, 'Iteration': int, 'Time': float, 
                                              'V1': float,'V2': float,'V3': float,'V4': float,'V5': float,'V6': float})
       
            
        
    elif isinstance(voltages, pd.dataframe):
        voltage_df = voltages
    
    else: 
        print("ERROR: voltages needs to be a pandas dataframe OR a path to a csv file which can be read as a dataframe.")
        return None

    voltages_arr = voltage_df[['V1','V2','V3','V4','V5','V6']].to_numpy()

    # compute forces and torques from voltages 
    cal_df = get_calibration_matrix(calibration)
    cal_matrix = cal_df.to_numpy()

    # multiply voltages by calibration matrix in numpy for faster result
    product = voltages_arr.dot(cal_matrix) # compute [V1 V2 V3 V4 V5 V6] * [calibration matrix] = [Fx Fy Fz Tx Ty Tz]   ///   (n x 6)*(6 x 6) = (n x 6)
    
    # make copy of initial dataframe and replace voltage columns by F/T columns
    FT_df = voltage_df.drop(columns=['V1','V2','V3','V4','V5','V6']).copy()

    FT_new_columns = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    
    for i, col in enumerate(FT_new_columns):
        FT_df[col] = product[:,i]
        
    if save_as_csv:
        # save data (write single csv file) to location below
        FT_df.to_csv(save_as_csv, index=False)
        print("Output CSV file saved to: " + save_as_csv)
        
    return FT_df





def butter_filt(original_signal, order, cutoff):
    # Design low-pass Butterworth filter
    sos = signal.butter(order/2, cutoff, 'lp', fs=960, output='sos')
    filtered_signal = signal.sosfiltfilt(sos, original_signal)
    return filtered_signal

def butter_filt_takeoff(original_signal, order, cutoff):
        sos = signal.butter(order // 2, cutoff, 'lp', fs=960, output='sos')
        padlen = 2 * (order // 2) + 1  # conservative estimate
        if len(original_signal) > padlen:
            return signal.sosfiltfilt(sos, original_signal)
        else:
            return None  # signal too short, skip filtering


def apply_butterfilt(FT_df, order, cutoff, columns=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'], calibration=False):
    
    # split data by individual flights/landings/signals
    if calibration:
        grouped_df = FT_df.groupby(['Direction', 'Distance'], dropna=False, as_index=False)
    
    else:
        grouped_df = FT_df.groupby(['Bird', 'Age', 'Takeoff'], dropna=False, as_index=False)

    # Make new DataFrame for filtered data    
    butter_df = pd.DataFrame()

    # apply filter to all F/T components of each individual signal
    for name, signal_df in tqdm(grouped_df, total=grouped_df.size().shape[0]):
        for col in columns:
            col_filt = col+'_filt'
            signal_df[col_filt] = butter_filt(signal_df[col], order=order, cutoff=cutoff)

        butter_df = pd.concat([butter_df, signal_df], ignore_index=True)

    
    return butter_df

def apply_butterfilt_takeoff(FT_df, order, cutoff, columns=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'], calibration=False):

    if calibration:
        grouped_df = FT_df.groupby(['Direction', 'Distance'], dropna=False, as_index=False)
    else:
        grouped_df = FT_df.groupby(['Bird', 'Age', 'Takeoff'], dropna=False, as_index=False)

    butter_df = pd.DataFrame()

    for name, signal_df in tqdm(grouped_df, total=grouped_df.size().shape[0]):
        for col in columns:
            col_filt = col + '_filt'
            filtered = butter_filt_takeoff(signal_df[col], order=order, cutoff=cutoff)
            if filtered is not None:
                signal_df[col_filt] = filtered
            else:
                # Optionally fill with NaNs or skip entirely
                signal_df[col_filt] = pd.Series([float('nan')] * len(signal_df[col]), index=signal_df.index)

        butter_df = pd.concat([butter_df, signal_df], ignore_index=True)

    return butter_df




def plot_fft(ax, df, col, start_Time=0, stop_Time=-1, xticks_interval=100, xticks_range=[-400,401], label='', color='#437e9f', title="Fast Fourier Transform"):    
    # For single figure, need to include lines below to call plot_fft:
    # f = plt.figure(figsize=(10,5))
    # plt.axes()
    # ax = f.get_axes()[0]
    # then call with ax = ax
    
    # Else for multiple subplots, need to include:
    # fig, axs = plt.subplots(n, m)
    # then call with ax = axs[i,j]
    
    if stop_Time == -1:
         stop_Time=int(len(df[col])/8)
            
    samples_per_Time = 8
    Times_per_second = 120.0
    sample_rate = 960
    duration = (stop_Time/Times_per_second - start_Time/Times_per_second) # get duration in seconds from Time indices
    N = int(sample_rate * duration) # calculate number of samples
    df_col = df[col][start_Time*samples_per_Time:stop_Time*samples_per_Time] # truncate the input data (column of df)
    signal = df_col.to_numpy() - df_col.mean() # shift data by mean value to centre around 0
    
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)
    
    if label == -1:
        label = 'empty'
        
    ax.plot(xf, np.abs(yf), alpha=0.3, color=color, label=label)
    ax.set_xticks(np.arange(xticks_range[0], xticks_range[1], xticks_interval))
    ax.tick_params(axis ='x', rotation = 30)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("abs fft")
    ax.set_title(title)
    
    return ax




def calibrate_time(df, Time_zero=241, Timerate=120):
    df['Time'] = (df['Time'] - Time_zero)/Timerate 
    return df 




def plot_flight_FT(df, flights, column, x='Time'):
    
    if x=='Time':
        df = calibrate_time(df)
    
    if not isinstance(flights, list):
        flights = [flights]

    flight_dfs = []
    for flight in flights:
        flight_df = get_flight(df, flight)
        flight_dfs.append(flight_df)
    
    fig = plt.figure(figsize = (10, 6))
    for df in flight_dfs:
        label = df.Bird.unique()[0] +' day'+ df.Age.unique()[0]+' takeoff ' + df.Takeoff.unique()[0]
        sns.lineplot(data=df, x=x, y=column, alpha = 0.6, label=label)
   
    #plt.show()
    return fig




def get_flight(df, flight):
    if isinstance(flight, dict):
        flight_df = df[(df['Bird']==flight['Bird']) & (df['Age']==str(flight['Age'])) & (df['Takeoff']==str(flight['Takeoff']))]
        
    elif isinstance(flight, int):
        flight_df = df[df['FlightID']==flight]
        
    return flight_df




def zero_centering(input_df, columns=['Fx','Fy','Fz','Tx','Ty','Tz'], avg_start=0, avg_stop=200):
    # uses mean of values between avg_start and avg_stop to compute new 0-value
    # returns new dateTime with 0-centered columns (specify which or defaults to all F/T) 
    df_list = [x for _, x in input_df.groupby(['FlightID'])]
    
    df_0centered_list = []
    for df in tqdm(df_list):
        before_landing = df[(df['Time'] >  avg_start) & (df['Time'] < avg_stop)]
        df_0centered = df.copy()
        for col in columns:
            df_0centered[col] = [i-before_landing[col].mean() for i in df[col]]
            
        df_0centered_list.append(df_0centered)
        
    output_df = pd.concat(df_0centered_list, ignore_index=True)
        
    return output_df

def zero_centering_takeoff(input_df, columns=['Fx','Fy','Fz','Tx','Ty','Tz'], endTime = 0.1):
    # uses the last 0.1 seconds to zero-center the data as this is when the bird would have left. 
    # returns new dateTime with 0-centered columns (specify which or defaults to all F/T) 
    df_list = [x for _, x in input_df.groupby(['FlightID'])]
    
    df_0centered_list = []
    for df in tqdm(df_list):
        lastTime = df['Time'].iloc[-1] #last recorded timepoint

        after_takeoff = df[(df['Time'] >  lastTime - endTime) & (df['Time'] < lastTime)]
        df_0centered = df.copy()
        for col in columns:
            df_0centered[col] = [i-after_takeoff[col].mean() for i in df[col]]
            
        df_0centered_list.append(df_0centered)
        
    output_df = pd.concat(df_0centered_list, ignore_index=True)
        
    return output_df






def scale_by_bw(df, columns=['Fx','Fy','Fz','Tx','Ty','Tz'], avg_start=300, avg_stop=480):
    # uses mean of values between avg_start and avg_stop to compute mean bodyweight
    # returns new dateTime with scaled columns (specify which or defaults to all F/T) 
    after_landing = df[(df['Time'] >  avg_start) & (df['Time'] < avg_stop)]
    df_scaled = df.copy()
    for col in columns:
        df_scaled[col] = [i/after_landing[col].mean() for i in df[col]]
        
    return df_scaled





def describe_FTs(df, start_Time=0, stop_Time=-1, zero_centered=True):
    
    # if data isn't zero-centered, then center it first
    if not zero_centered:
        df = zero_centering(df)
    
    # if no stop Time specified (=-1 by default) then use all the Times (up to the last)
    if stop_Time == -1:
        stop_Time = df['Time'].max()
        
    # keep only the "region of interest" b/w start Time & stop Time for each flight
    df_truncated = df[(df['Time'] > start_Time) & (df['Time'] < stop_Time)]
    
    # get summary/description of all 6 axes for the region of interest
    df_grouped = df_truncated.groupby(['Bird', 'Age', 'Takeoff', 'FlightID']).agg({'Fx': ['mean', 'std', 'min', 'max'], 
                                                                                   'Fy': ['mean', 'std', 'min', 'max'], 
                                                                                   'Fz': ['mean', 'std', 'min', 'max'], 
                                                                                   'Tx': ['mean', 'std', 'min', 'max'], 
                                                                                   'Ty': ['mean', 'std', 'min', 'max'], 
                                                                                   'Tz': ['mean', 'std', 'min', 'max'],})
    return df_grouped



#csv_df = compile_FT_CSVs("D:\\Vicon Install\\FinchPerching\\nano43_manualCal\\", calibration=True)
#voltage_df = CSVs_to_dataframe(csv_df, calibration=True, save_as_csv=False)
#FT_df = voltage_to_FT(voltage_df, cal_data=True)
#print(FT_df)

