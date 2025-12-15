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
import flight_ontogeny.helpers as helpers
#, helpers23
from scipy.fft import fft, fftfreq
from tqdm.notebook import  tqdm, tqdm_notebook




def make_trial_data_df(df, year=2022):

    trial_datas = df['trial_dataID'].unique()
    trial_datas_list = []
    for trial_data in tqdm(trial_datas):
        trial_data_df = df[df['trial_dataID'] == trial_data]
        
        if year==2022:
            info = helpers.get_trial_data_info(trial_data)
            trial_datas_list.append({"trial_dataID": trial_data, 
                            "Bird": info.get("Bird"),
                            "Age": info.get("Age"),
                            "Takeoff": info.get("Takeoff")})
        elif year==2023:
            info = helpers23.get_trial_data_info(trial_data)
            trial_datas_list.append({"trial_dataID": trial_data, 
                        "BirdID": info.get("BirdID"),
                        "Age": info.get("Age"),
                        "Trial": trial_data_df.Trial.unique()[0],
                        "Code": trial_data_df.Code.unique()[0],
                        "Category": info.get("Category"),
                        "Iteration":info.get("Iteration")})
            
                            
    return pd.DataFrame(trial_datas_list)



def get_bodyweight(df, trial_data_ID=0, start_frame=240, stop_frame=480, use_filt=True):
    
    if trial_data_ID==0:
        trial_data_df = df
    else:
        trial_data_df =  df[df['trial_dataID'] == trial_data_ID]
        
    after_landing = trial_data_df[(trial_data_df['Frame'] >  start_frame) & (trial_data_df['Frame'] < stop_frame)]
    if use_filt:
        bw = after_landing['Fx_filt'].mean()
    else:
        bw = after_landing['Fx'].mean()
    return abs(bw)


def make_BW_df(df, output_df): # not used, to delete eventually

    trial_datas = df['trial_dataID'].unique()
    bw_list = []
    for trial_data in trial_datas:
        info = helpers.get_trial_data_info(trial_data)
        trial_data_df = df[df['trial_dataID'] == trial_data]
        # need to figure out a way to reliably find good/best start_frame and stop_frame values here!
        bw = get_bodyweight(trial_data_df)
        bw_list.append({"trial_dataID": trial_data, 
                        "Bird": info.get("Bird"),
                        "Age": info.get("Age"),
                        "Takeoff": info.get("Takeoff"),
                        "BW (perch)": bw, 
                        "BW (manual)": 'NA'})

    return pd.DataFrame(bw_list)



def compute_Fxy(df, use_filt=True):
    new_col = 'Fxy'
    if use_filt:
        filt = '_filt'
    else: 
        filt = ''
    
    df[new_col+filt] = np.sqrt(df['Fx'+filt]**2+df['Fy'+filt]**2)
    
    return df



def compute_Ftotal(df, use_filt=True):
    new_col = 'Ftotal'
    if use_filt:
        filt = '_filt'
    else: 
        filt = ''
    df[new_col+filt] = np.sqrt(df['Fx'+filt]**2+df['Fy'+filt]**2+df['Fz'+filt]**2)
    
    return df


def find_max_force(df, trial_data_ID=0, col="Fx", use_filt=False, end=280, argmax=False):
    if trial_data_ID==0:
        trial_data_df = df
    else:
        trial_data_df =  df[df['trial_dataID'] == trial_data_ID]
    landing_window = trial_data_df[(trial_data_df['Frame'] >  220) & (trial_data_df['Frame'] < end)]
    if use_filt:
        col = col + "_filt"
    
    output = max(abs(landing_window[col]))
    
    if argmax:
        argmax = abs(landing_window[col]).argmax()
        output = landing_window.reset_index().at[argmax,'Time']
    #print(output)
    return output



def get_forces_at(time, df, trial_data_ID=0, relative=False):
    if trial_data_ID==0:
        trial_data_df = df
    else:
        trial_data_df =  df[df['trial_dataID'] == trial_data_ID]
    time_df = trial_data_df[trial_data_df.Time == float(time)].reset_index()
    r = ''
    if relative:
        r = ' (relative)'
    return {'Ftotal': time_df.at[0,'Ftotal'+r],
            'Fx': time_df.at[0,'Fx'+r],
            'Fy': time_df.at[0,'Fy'+r],
            'Fz': time_df.at[0,'Fz'+r],
            'Tx': time_df.at[0,'Tx'+r],
            'Ty': time_df.at[0,'Ty'+r],
            'Tz': time_df.at[0,'Tz'+r]}




def AUCimpact(df, trial_data_ID=0, relative=False):
    try:
        if trial_data_ID==0:
            trial_data_df = df.copy()
        else:
            trial_data_df =  df[df['trial_dataID'] == trial_data_ID].copy()

        # define starting point for computing AUC
        start_idx = trial_data_df[trial_data_df.Time >= 0].index[0] # starts integrating after trigger (default t=0)

        # define ending point for computing AUC
        # subset to keep only data after start_idx
        post_impact_df =  trial_data_df[trial_data_df.Time >= 0.01]
        # here the end point is defined as the first instance of a negative Fy value 
        end_idx = post_impact_df[post_impact_df.Fy <= 0].index[0]

        # create a subset of the dataframe from start to end points of AUC
        impact_slice = df.loc[start_idx:end_idx]


        r = ''
        if relative:
            r = ' (relative)'
        
        #compute the approximate integral via trapezoidal method for each F/T column
        out  = {'Ftotal': np.trapz(impact_slice['Ftotal'+r]),
                'Fx': np.trapz(impact_slice['Fx'+r]),
                'Fy': np.trapz(impact_slice['Fy'+r]),
                'Fz': np.trapz(impact_slice['Fz'+r]),
                'Tx': np.trapz(impact_slice['Tx'+r]),
                'Ty': np.trapz(impact_slice['Ty'+r]),
                'Tz': np.trapz(impact_slice['Tz'+r])}
    
    except:
        print("ERROR:"+trial_data_ID)
        out = {'Ftotal': None,'Fx': None,'Fy': None,'Fz': None,'Tx': None,'Ty': None,'Tz': None}
        
    return out



def find_ith_occurrence(df, i=0, cat='W', block='A'):
    df['Trial'] = df.Trial.astype('int64')
    subdf = df[df.Code.str.contains(cat) & 
               df.Code.str.contains(block)].sort_values(['Trial'])
    
    idx_list = subdf.index.values 
    if idx_list.size > abs(i):
        return idx_list[i]
    else:
        return None




def plot_impactVSbw(df, col):
    fig = plt.figure(figsize = (8, 4))
    #df['BW (perch)'].hist(bins=10,alpha=0.7, label='Bodyweight')
    col_name = "Impact " + col
    df[col_name].hist(bins=20,alpha=0.7, color="Grey", label=col_name)
    plt.xlim(xmin=0)
    if col == "Fx":
        plt.xlabel("Fx = downward force (N)")
        plt.title("Fx - force distribution")
    elif col=="Fy":
        plt.xlabel("Fy = forward force (N)")
        plt.title("Fy - force distribution")
    elif col=="Fxy":
        plt.xlabel("Fxy = total force (N)")
        plt.title("Fxy - force distribution")
    elif col=="Tz":
        plt.xlabel("Tz = torque (Nmm)")
        plt.title("Tz - torque distribution")
    plt.ylabel("Count")
    plt.legend()
    
    
def plot_polar_hist(degree_array):
    radians = np.deg2rad(degree_array)

    bin_size = 5
    a , b = np.histogram(degree_array, bins=np.arange(0, 360+bin_size, bin_size))
    centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='polar')
    ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='#A7C7E7', edgecolor='k')
    #ax.set_rticks([0, 100, 200, 300])  # Less radial ticks
    ax.set_rlabel_position(80)  # Move radial labels away from plotted line
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    plt.tight_layout()
    #plt.savefig(r"D:\python_output\force_figures\polar_histogram_landing_angle.png", dpi=300)
    plt.show()

def zbdetect_bw_Ftot(event_df, var_threshold=0.00005, interval=50):
    plateaus = []
    
    Ftotal = event_df['Ftotal_filt'].values
    Ftotal_peak_index = np.argmax(Ftotal)
    # difference with shed_tit. In shed tit Ftotal_firsthalf is used. 
    # In zebra finch, we use all the data before the Ftot peak. 

    Ftotal_before = Ftotal[:Ftotal_peak_index] 

    # Iterate through the data in chunks of frame_length
    for start in range(0, len(Ftotal_before), interval):
        end = min(start + interval, len(Ftotal_before))
        segment = Ftotal_before[start:end]
        
        # Calculate the mean difference and variance of the current segment
        variance = np.var(segment)
        mean = np.mean(segment)
        
        # Check if both the mean difference and variance are below their respective thresholds
        if variance <= var_threshold and mean > 0.075: #(0.075 N is approx 7.5 grams, to avoid detecting zero-force plateaus)
            plateaus.append((start, end-1))
    
    #initialize bodyweight with NaN
    bodyweight = np.nan
    
    # Calculate the overall mean of the values corresponding to the plateaus
    if plateaus:
        plateau_values = [Ftotal_before[start:end] for start, end in plateaus]

        # Flatten the list of arrays into a single array
        all_plateau_values = np.concatenate(plateau_values)
        # Calculate the overall mean
        bodyweight = np.mean(all_plateau_values)
            

    return plateaus, bodyweight

#same but only returns bodyweight without highlighting plateaus
def zbreturn_bw_Ftot(event_df, var_threshold=0.00005, interval=50): 
    plateaus = []
    
    Ftotal = event_df['Ftotal_filt'].values
    Ftotal_peak_index = np.argmax(Ftotal)
    # difference with shed_tit. In shed tit Ftotal_firsthalf is used. 
    # In zebra finch, we use all the data before the Ftot peak. 

    Ftotal_before = Ftotal[:Ftotal_peak_index] 

    # Iterate through the data in chunks of frame_length
    for start in range(0, len(Ftotal_before), interval):
        end = min(start + interval, len(Ftotal_before))
        segment = Ftotal_before[start:end]
        
        # Calculate the mean difference and variance of the current segment
        variance = np.var(segment)
        mean = np.mean(segment)
        
        # Check if both the mean difference and variance are below their respective thresholds
        if variance <= var_threshold and mean > 0.075: #(0.075 N is approx 7.5 grams, to avoid detecting zero-force plateaus)
            plateaus.append((start, end-1))
    
    #initialize bodyweight with NaN
    bodyweight = np.nan
    
    # Calculate the overall mean of the values corresponding to the plateaus
    if plateaus:
        plateau_values = [Ftotal_before[start:end] for start, end in plateaus]

        # Flatten the list of arrays into a single array
        all_plateau_values = np.concatenate(plateau_values)
        # Calculate the overall mean
        bodyweight = np.mean(all_plateau_values)
            

    return bodyweight

def plot_flight_data(flight_data_df1, df):

    # fligt_df1: subset dictionary data with trial_dataID, Bird, Age, Takeoff (trial number)
    # df: main dataframe with all the response variables e.g. Fx, Fy, Fz, Ftotal_filt, etc. 

    fig, axs = plt.subplots(nrows=25, ncols=4, figsize=(20, 120))

    axs = axs.flatten()

    for i, id in enumerate(flight_data_df1['FlightID'].unique()):
        trial_data = df[df['FlightID'] == id]

        if trial_data.empty:
            print(f"Skipping FlightID {id} — no data found.")
            continue

        age = trial_data['Age'].values[0]
        bird = trial_data['Bird'].values[0]
        takeoff = trial_data['Takeoff'].values[0]

        time = trial_data['Time'].values

        #detect plateaus and bodyweight  
        plateaus, bodyweight = zbdetect_bw_Ftot(trial_data)

        #overlay 'Fz' by 'Time' in yellow line
        axs[i].plot(time, trial_data['Fz_filt'], label='Fz_filt', color='yellow')
        
        #overlay 'Fy' by 'Time' in blue line
        axs[i].plot(time, trial_data['Fy_filt'], label='Fy_filt')

        #overlay 'Fx' by 'Time'in orange line
        axs[i].plot(time, trial_data['Fx_filt'], label='Fx_filt', color='orange')

        #overlay 'Ftotal_filt'by 'Time' in pink line
        axs[i].plot(time, trial_data['Ftotal_filt'], label='Ftotal_filt', color='pink')

        # Highlight plateau regions in yellow
        for start_idx, end_idx in plateaus:
            axs[i].axvspan(time[start_idx], time[end_idx], color='yellow', alpha=0.3)

        # Plot bodyweight as horizontal dashed line
        if not np.isnan(bodyweight):
            axs[i].axhline(y=bodyweight, color='grey', linestyle='--', label='Bodyweight')
        
        #axs[i].axhline(y=helpers_BW.get_bodyweight(df, trial_data, use_filt=True), color='r', linestyle='--', label='Bodyweight')
        # Show bird, age, takeoff in title
        axs[i].set_title(f'FlightID: {id}_{bird}_{age}_{takeoff}')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('F (N)')
        axs[i].legend()

    plt.tight_layout()
    plt.show()

# I might use the function that calculates all the necessary force elements at one-go.

def plot_flight_peaks(flight_df1, df, interval = 0.11):

    # fligt_df1: subset dictionary data with trial_dataID, Bird, Age, Takeoff (trial number)
    # df: main dataframe with all the response variables e.g. Fx, Fy, Fz, Ftotal_filt, etc.
    # also needs to have bodyweight  
    # interval (seconds): The time that will be displayed on the plot centered around the peak
    ## e.g. 0.55 sec before the peak + 0.55 after the peak. 

    fig, axs = plt.subplots(nrows=25, ncols=4, figsize=(20, 120))

    axs = axs.flatten()

    for i, id in enumerate(flight_df1['FlightID'].unique()):

        # trial data is force logs with certain flightID
        trial_full = df[df['FlightID'] == id]

        if trial_full.empty:
            print(f"Skipping FlightID {id} — no data found.")
            continue

        age = trial_full[trial_full['FlightID'] == id]['Age'].values[0]
        bird = trial_full[trial_full['FlightID'] == id]['Bird'].values[0]
        takeoff = trial_full[trial_full['FlightID'] == id]['Takeoff'].values[0]
        bodyweight = trial_full[trial_full['FlightID'] == id]['bodyweight'].values[0]

        #calculate Ftotal_filt - bw
        trial_full['Ftotal_filt_minus_bw'] = trial_full['Ftotal_filt'] - trial_full['bodyweight']

        # find the peak index by Ftotal_filt - bodyweight
        f = trial_full['Ftotal_filt_minus_bw'].values
        peak_index = np.argmax(f) #more like a position
        peak_time = trial_full.iloc[peak_index]['Time']
        
    # finding impulse calculation region
        # Initialize start and end at the peak
        start = peak_index
        end = peak_index

        # Expand backward while values are positive
        while start > 0 and f[start - 1] > bodyweight*0.05:
            start -= 1

        # Expand forward while values are positive
        while end < len(f) - 1 and f[end + 1] > bodyweight*0.05:
            end += 1

        # Extract the time range
        start_time = trial_full.iloc[start]['Time']
        end_time = trial_full.iloc[end]['Time']

        trial_interval = trial_full[(trial_full['Time'] >= (peak_time - interval/2)) & (trial_full['Time'] <= (peak_time + interval/2))]

        #overlay 'Fz' by 'Time' in yellow line
        axs[i].plot(trial_interval['Time'], trial_interval['Fz_filt'], label='Fz_filt', color='yellow')
        
        #overlay 'Fy' by 'Time' in blue line
        axs[i].plot(trial_interval['Time'], trial_interval['Fy_filt'], label='Fy_filt')

        #overlay 'Fx' by 'Time'in orange line
        axs[i].plot(trial_interval['Time'], trial_interval['Fx_filt'], label='Fx_filt', color='orange')

        #overlay 'Ftotal_filt'by 'Time' in pink line
        axs[i].plot(trial_interval['Time'], trial_interval['Ftotal_filt'], label='Ftotal_filt', color='pink')

        #bodyweight as grey line
        axs[i].axhline(y=bodyweight, color='grey', linestyle='--', label='Bodyweight')

        #mark peak Ftotal_filt as a red dot
        
        axs[i].plot(peak_time, trial_full.iloc[peak_index]['Ftotal_filt'], 'ro')

        #light green shade from start_time to end_time 
        axs[i].axvspan(start_time, end_time, color='lightgreen', alpha=0.3, label='impulse region')

        #axs[i].axhline(y=helpers_BW.get_bodyweight(df, trial_data, use_filt=True), color='r', linestyle='--', label='Bodyweight')
        # Show bird, age, takeoff in title
        axs[i].set_title(f'FlightID: {id}_{bird}_{age}_{takeoff}')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('F (N)')
        axs[i].legend()

    plt.tight_layout()
    plt.show()

def get_force(df):
    # Ftotal already computed  

    # DataFrame to return
    force_outcome = pd.DataFrame(columns=['FlightID','Bird', 'Age', 'Takeoff', 'max_Ftot', 'time_max_Ftot',  'Fx_maxFtot',
'Fy_maxFtot', 'Fz_maxFtot', 'angle', 'bodyweight'])

    # Group the DataFrame by 'trial_dataID'
    grouped = df.groupby(['FlightID'])
    # Iterate through each group with a progress bar
    for _, trial_data in tqdm(grouped, desc="Processing groups"):
        # Extract required values
        flightid = trial_data['FlightID'].iloc[0]
        bird = trial_data['Bird'].iloc[0]
        age = trial_data['Age'].iloc[0]
        takeoff = trial_data['Takeoff'].iloc[0]

        #Find max_Ftot, the peak force at take-off. 
        max_Ftot = trial_data['Ftotal_filt'].max()
        
        #find the index of max_Ftot

        idx_max_Ftot = trial_data['Ftotal_filt'].idxmax()

        # extract Fx, Fy, and Time at max_Ftot

        Fx_maxFtot = trial_data.loc[idx_max_Ftot, 'Fx_filt']
        Fy_maxFtot = trial_data.loc[idx_max_Ftot, 'Fy_filt']
        Fz_maxFtot = trial_data.loc[idx_max_Ftot, 'Fz_filt'] #added June11-2025
        time_max_Ftot = trial_data.loc[idx_max_Ftot, 'Time']

        # bodyweight
        bodyweight = zbreturn_bw_Ftot(trial_data)

        
        # take-off angle 
        angle = np.rad2deg(np.arctan2(-Fx_maxFtot, Fy_maxFtot)) # to reflect the downward direction of Fx. - because x is minus. 
        # Note the role reversal: the "y-coordinate" is the first function parameter, the "x-coordinate" is the second. 
        #In my code, the conventional "y-coordinate" is x-axis. and flipped the direction because downward force that produces upward motion is -X. 

        # Create a new row as a dictionary (faster than creating a DataFrame for each row)
        new_row = {

            'FlightID': flightid,
            'Bird': bird,
            'Age': age,
            'Takeoff': takeoff,
            'max_Ftot': max_Ftot,
            'time_max_Ftot': time_max_Ftot,
            'Fx_maxFtot': Fx_maxFtot,
            'Fy_maxFtot': Fy_maxFtot,
            'Fz_maxFtot': Fz_maxFtot,
            'angle': angle,
            'bodyweight': bodyweight
            
        }

        # Append the new row to the force_outcome DataFrame
        force_outcome = pd.concat([force_outcome, pd.DataFrame([new_row])], ignore_index=True)

    return force_outcome



def compute_impulse(df_bw):

    impulse_list = []

    for id in df_bw['FlightID'].unique():

        trial_full = df_bw[df_bw['FlightID'] == id]
        bodyweight = trial_full[trial_full['FlightID'] == id]['bodyweight'].values[0]

        #calculate Ftotal_filt - bw
        trial_full['Ftotal_filt_minus_bw'] = trial_full['Ftotal_filt'] - trial_full['bodyweight']

        # find the peak index by Ftotal_filt - bodyweight
        f = trial_full['Ftotal_filt_minus_bw'].values
        peak_index = np.argmax(f) #more like a position
        peak_time = trial_full.iloc[peak_index]['Time']
        
    # finding impulse calculation region
        # Initialize start and end at the peak
        start = peak_index
        end = peak_index

        # Expand backward while values are positive
        while start > 0 and f[start - 1] > bodyweight*0.05:
            start -= 1

        # Expand forward while values are positive
        while end < len(f) - 1 and f[end + 1] > bodyweight*0.05:
            end += 1

        # Extract the time range
        start_time = trial_full.iloc[start]['Time']
        end_time = trial_full.iloc[end]['Time']

        #trial_interval = trial_full[(trial_full['Time'] >= (peak_time - interval/2)) & (trial_full['Time'] <= (peak_time + interval/2))]
        impulse = trial_full.iloc[start:end]['Ftotal_filt_minus_bw'].sum() /1000 # convert to N*s
        duration = trial_full.iloc[end]['Time'] - trial_full.iloc[start]['Time']
        left_impulse = trial_full.iloc[start:peak_index]['Ftotal_filt_minus_bw'].sum()/1000
        left_duration = trial_full.iloc[peak_index]['Time'] - trial_full.iloc[start]['Time']
        right_impulse = trial_full.iloc[peak_index:end]['Ftotal_filt_minus_bw'].sum()/1000
        right_duration = trial_full.iloc[end]['Time'] - trial_full.iloc[peak_index]['Time']
        impulse_ratio = left_impulse/right_impulse if right_impulse !=0 else np.nan

        impulse_list.append({

            'FlightID': id,
            'impulse': impulse,
            'duration': duration,
            'left_impulse': left_impulse,
            'left_duration': left_duration,
            'right_impulse': right_impulse,
            'right_duration': right_duration,
            'impulse_ratio': impulse_ratio,
            'impulse_time_diff': left_duration - right_duration

        })

        impulse_df = pd.DataFrame(impulse_list)
    
    return impulse_df 

def make_flight_df(df):
    flight_list = []
    for flight_id in df['FlightID'].unique():
        flight_data = df[df['FlightID'] == flight_id].iloc[0]
        flight_list.append({
            'FlightID': flight_id,
            'Bird': flight_data['Bird'],
            'Age': flight_data['Age'],
            'Takeoff': flight_data['Takeoff']
        })
    flight_df = pd.DataFrame(flight_list)
    return flight_df


