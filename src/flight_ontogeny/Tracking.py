import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import cdist
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.signal import find_peaks

from scipy import stats
from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from tqdm.notebook import tqdm, tqdm_notebook

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import math


def label_stationary(marker_coordinates, frame_identifiers, threshold_proximity=0.001):

    """
    Labels markers as stationary or not based on their displacement over a range of frames.

    Args:
    marker_coordinates: A numpy array of marker coordinates in a three-dimensional space.
    frame_identifiers: A list or numpy array of frame identifiers corresponding to the marker coordinates.
    threshold_proximity: Float, the threshold for determining if a marker is stationary.

    Returns:
    stationary_labels: Series indicating whether each marker is stationary (1) or not (0).
    """
        
    # Calculate distances and indices over frame gaps from 1 to 10
    distances, start_indices, end_indices = compute_marker_distances(marker_coordinates,
                                                                    frame_identifiers,
                                                                    frame_gap=list(range(1, 11)))

    # Normalize distances by frame durations and check against the threshold
    frame_durations = frame_identifiers[end_indices] - frame_identifiers[start_indices]
    normalized_distances = distances / frame_durations
    is_stationary = normalized_distances < threshold_proximity
    #since the Nano43 is meant to save 2 secs before and after threshold, The last 2.2 seconds should be plenty
    #If for 2.2 seconds the marker has not moved 1 cm => stationary marker. 
    #120 frames for seconds --> 2.2 secs x 120 = 264 frames 
    #1cm in mm = 10mm 
    #normalized distance = 10 /264 = 0.04 

    # Initialize labels and set stationary markers
    stationary_labels = np.zeros(len(frame_identifiers), dtype=int)
    stationary_indices = np.unique(np.concatenate((start_indices[is_stationary], end_indices[is_stationary])))
    stationary_labels[stationary_indices] = 1

    if np.sum(stationary_labels) > 1:  # Ensure there are at least two points to cluster
        stationary_coords = marker_coordinates[stationary_labels == 1]
        num_stationary = len(stationary_coords)
        k_range = range(2, min(10, num_stationary))  # Ensure valid range for k (at least 2, at most num_stationary - 1)

        if len(k_range) > 1:
            scores = [calinski_harabasz_score(stationary_coords, KMeans(n_clusters=k).fit_predict(stationary_coords)) for k in k_range]
            optimal_k = k_range[np.argmax(scores)]
            kmeans = KMeans(n_clusters=optimal_k)
            cluster_labels = kmeans.fit_predict(stationary_coords) + 1  # offset by 1 to avoid zero label
            stationary_labels[stationary_labels == 1] = cluster_labels

    return stationary_labels




def compute_marker_distances(marker_coordinates, 
                             frame_identifiers, 
                             frame_gap=0, 
                             return_unit_vectors=False):
    """
    Compute distances between based on their coordinates, 
    frame identifiers, and specified frame gap.

    Args:
    marker_coordinates: A numpy array of marker coordinates in a three-dimensional space.
    frame_identifiers: A list or numpy array of frame identifiers corresponding to the marker coordinates.
    frame_gap: An integer representing the desired gap between frames for calculating distances.
    return_unit_vectors: A boolean flag to return unit vectors representing the direction and magnitude of displacement between markers.
    
    Returns:
    marker_distances: A numpy array of Euclidean distances between markers.
    start_frame_indices: Indices of the starting frames for the distances computed.
    end_frame_indices: Indices of the ending frames for the distances computed.
    unit_displacement_vectors: Unit vectors representing the direction and magnitude of displacement between markers.
    Optional:
    If return_unit_vectors is set to False, the function will return marker_distances, start_frame_indices, and end_frame_indices.
    If return_unit_vectors is set to True, the function will return marker_distances, start_frame_indices, end_frame_indices, and unit_displacement_vectors.
    """
    
    start_frame_indices, end_frame_indices = find_edges(frame_identifiers, frame_gap)
    
    displacement_vectors = marker_coordinates[end_frame_indices, :3] - marker_coordinates[start_frame_indices, :3]
    
    marker_distances = np.linalg.norm(displacement_vectors, axis=1)

    if return_unit_vectors:
        unit_displacement_vectors = displacement_vectors / marker_distances[:, np.newaxis]
        return marker_distances, start_frame_indices, end_frame_indices, unit_displacement_vectors
    return marker_distances, start_frame_indices, end_frame_indices


def find_edges(frame_identifiers, frame_gap):

    """
    Helper function to find the index pairs with a specified gap. 
    
    Args:
    frame_identifiers: A list or numpy array of frame identifiers.
    frame_gap: An integer representing the gap between frames.
    
    Returns:
    starting_frame_indices: Indices of the starting frames.
    ending_frame_indices: Indices of the ending frames.

    """

    num_frames = len(frame_identifiers)
    frame_diff = np.array(frame_identifiers).reshape(-1, 1) - np.array(frame_identifiers)

    # Check if frame_gap is a list or array, and handle accordingly
    frame_match_mask = np.isin(frame_diff, frame_gap)

    start_frame_indices, end_frame_indices = np.where(frame_match_mask)
    valid_indices = start_frame_indices < end_frame_indices
    start_frame_indices = start_frame_indices[valid_indices]
    ending_frame_indices = end_frame_indices[valid_indices]

    return start_frame_indices, ending_frame_indices


def plot_markers(data, bird, age, trial):
    
    # Shows raw vs stationary vs marker plots for X, Y, Z coordinates over time
    # Returns trial_data, the dataset used for graph. 
    
    
    # Validate required columns
    required_cols = ['Bird', 'Age', 'Takeoff', 'X', 'Y', 'Z', 'Time', 'Frame', 'Marker']
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        print(f"Missing columns in data: {missing}")
        return

    # Filter data
    trial_data = data[(data['Bird'] == bird) & 
                            (data['Age'] == age) & 
                            (data['Takeoff'] == trial)]
    if trial_data.empty:
        print("No matching data found for the given bird, age, and trial.")
        return

    # Compute stationary labels
    coords = trial_data[['X', 'Y', 'Z']].values
    frames = trial_data['Frame'].values
    stationary_labels = label_stationary(coords, frames, 0.001)

    if len(stationary_labels) != len(trial_data):
        print("Mismatch between stationary labels and data length.")
        return

    trial_data = trial_data.copy()
    trial_data['StationaryLabels'] = stationary_labels

    # Create figure and subplots
    fig, ax = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle(f'{bird}, age {age}, trial #{trial}', fontsize=15)

    # Raw plots
    ax[0, 0].plot(trial_data.Time, trial_data.X, '.', label='X')
    ax[1, 0].plot(trial_data.Time, trial_data.Y, '.', label='Y')
    ax[2, 0].plot(trial_data.Time, trial_data.Z, '.', label='Z')
    ax[0, 0].set_title("Time vs X raw")
    ax[1, 0].set_title("Time vs Y raw")
    ax[2, 0].set_title("Time vs Z raw")

    # Stationary label plots
    stat1 = ax[0, 1].scatter(trial_data.Time, trial_data.X, c=trial_data.StationaryLabels, cmap='viridis')
    stat2 = ax[1, 1].scatter(trial_data.Time, trial_data.Y, c=trial_data.StationaryLabels, cmap='viridis')
    stat3 = ax[2, 1].scatter(trial_data.Time, trial_data.Z, c=trial_data.StationaryLabels, cmap='viridis')
    fig.colorbar(stat1, ax=ax[0, 1])
    fig.colorbar(stat2, ax=ax[1, 1])
    fig.colorbar(stat3, ax=ax[2, 1])
    ax[0, 1].set_title("Time vs X stationary labels")
    ax[1, 1].set_title("Time vs Y stationary labels")
    ax[2, 1].set_title("Time vs Z stationary labels")

    # Marker plots
    distinct_colors = ['red', 'green', 'orange', 'purple', 'pink', 'cyan']
    marker_cmap = ListedColormap(distinct_colors)

    mark1 = ax[0, 2].scatter(trial_data.Time, trial_data.X, c=trial_data.Marker, cmap=marker_cmap)
    mark2 = ax[1, 2].scatter(trial_data.Time, trial_data.Y, c=trial_data.Marker, cmap=marker_cmap)
    mark3 = ax[2, 2].scatter(trial_data.Time, trial_data.Z, c=trial_data.Marker, cmap=marker_cmap)
    fig.colorbar(mark1, ax=ax[0, 2])
    fig.colorbar(mark2, ax=ax[1, 2])
    fig.colorbar(mark3, ax=ax[2, 2])
    ax[0, 2].set_title("Time vs X markers")
    ax[1, 2].set_title("Time vs Y markers")
    ax[2, 2].set_title("Time vs Z markers")

    # Stationary-only plots
    stationary_only = trial_data[trial_data.StationaryLabels == 1]
    ax[0, 3].plot(stationary_only.Time, stationary_only.X, '.', label='X stationary')
    ax[1, 3].plot(stationary_only.Time, stationary_only.Y, '.', label='Y stationary')
    ax[2, 3].plot(stationary_only.Time, stationary_only.Z, '.', label='Z stationary')
    ax[0, 3].set_title("X stationary == 1")
    ax[1, 3].set_title("Y stationary == 1")
    ax[2, 3].set_title("Z stationary == 1")
    ax[0, 3].legend()
    ax[1, 3].legend()
    ax[2, 3].legend()

    # Final layout
    plt.tight_layout()
    plt.show()
    return trial_data

def plot_single_marker(trial_data, marker_number):

    marker_data = trial_data[trial_data['Marker'] == marker_number]
    bird = marker_data['Bird'].iloc[0]
    age = marker_data['Age'].iloc[0]
    trial = marker_data['Takeoff'].iloc[0]
    
    fig, ax = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{bird}, age {age}, trial #{trial}, marker #{marker_number}', fontsize=15)
    # Raw plots
    ax[0, 0].plot(marker_data.Time, marker_data.X, '.', label='X')
    ax[1, 0].plot(marker_data.Time, marker_data.Y, '.', label='Y')
    ax[2, 0].plot(marker_data.Time, marker_data.Z, '.', label='Z')
    ax[0, 0].set_title("Time vs X raw")
    ax[1, 0].set_title("Time vs Y raw")
    ax[2, 0].set_title("Time vs Z raw")

    # Stationary label plots
    #stat1 = ax[0, 1].scatter(marker_data.Time, marker_data.X, c=marker_data.StationaryLabels, cmap='viridis')
    #stat2 = ax[1, 1].scatter(marker_data.Time, marker_data.Y, c=marker_data.StationaryLabels, cmap='viridis')
    #stat3 = ax[2, 1].scatter(marker_data.Time, marker_data.Z, c=marker_data.StationaryLabels, cmap='viridis')
    #fig.colorbar(stat1, ax=ax[0, 1])
    #fig.colorbar(stat2, ax=ax[1, 1])
    #fig.colorbar(stat3, ax=ax[2, 1])
    #ax[0, 1].set_title("Time vs X stationary labels")
    #ax[1, 1].set_title("Time vs Y stationary labels")
    #ax[2, 1].set_title("Time vs Z stationary labels")

    # Marker plots


    mark1 = ax[0, 1].scatter(marker_data.Time, marker_data.X, c=marker_data.Marker)
    mark2 = ax[1, 1].scatter(marker_data.Time, marker_data.Y, c=marker_data.Marker)
    mark3 = ax[2, 1].scatter(marker_data.Time, marker_data.Z, c=marker_data.Marker)
    fig.colorbar(mark1, ax=ax[0, 1])
    fig.colorbar(mark2, ax=ax[1, 1])
    fig.colorbar(mark3, ax=ax[2, 1])
    ax[0, 1].set_title("Time vs X markers")
    ax[1, 1].set_title("Time vs Y markers")
    ax[2, 1].set_title("Time vs Z markers")

    # Stationary-only plots
    #stationary_only = marker_data[marker_data.StationaryLabels == 1]
    #ax[0, 3].plot(stationary_only.Time, stationary_only.X, '.', label='X stationary')
    #ax[1, 3].plot(stationary_only.Time, stationary_only.Y, '.', label='Y stationary')
    #ax[2, 3].plot(stationary_only.Time, stationary_only.Z, '.', label='Z stationary')
    #ax[0, 3].set_title("X stationary == 1")
    #ax[1, 3].set_title("Y stationary == 1")
    #ax[2, 3].set_title("Z stationary == 1")
    #ax[0, 3].legend()
    #ax[1, 3].legend()
    #ax[2, 3].legend()

    # Final layout
    plt.tight_layout()
    plt.show()


# radical thought - can I just filter with Euclidean distance and Y min at the start?
def head_marker_test(trial_data):
    
    #trial_data
    #a dataframe with bird, age, and trial
    bird = trial_data['Bird'].iloc[0]
    age = trial_data['Age'].iloc[0]
    trial = trial_data['Takeoff'].iloc[0]


    #markers df
    markers_df = trial_data.groupby('Marker')['StationaryLabels'].nunique().reset_index()
    markers_df.columns = ['Marker', 'StationaryLabels'] 
    
    # Sort markers by stationary label values in descending order
    sorted_markers = markers_df.sort_values(by='StationaryLabels', ascending=False)
    
    # Select the top three markers. top_two_markers include the columns 'pinkR-blueL', 'Marker', 'StationaryLabels'
    top_three_markers_df = sorted_markers.head(3)
    top_three_markers = top_three_markers_df[['Marker']]
    
    # Check if the difference between the highest and the second highest stationary label values is more than 3
    if len(top_three_markers) == 3 and (top_three_markers_df.iloc[0]['StationaryLabels'] - top_three_markers_df.iloc[1]['StationaryLabels']) > 3:
        # Drop the second marker and the third marker
        top_three_markers = top_three_markers.head(1)
    
    # if the difference between the second highest and third highest is more than three,
    elif len(top_three_markers) == 3 and (top_three_markers_df.iloc[1]['StationaryLabels'] - top_three_markers_df.iloc[2]['StationaryLabels']) > 3:
        # Drop the third marker
        top_three_markers = top_three_markers.head(2)
    
    
    # If there are still more than one markers
    if len(top_three_markers) >= 2:
        # make the summary dataframe for the remaining markers 
        # for the first Y value, mean, min, std, and Euclidean distance mean between consecutive points.
        summary_stats = []
        selected_markers = top_three_markers['Marker'].tolist()
        for marker in selected_markers:
            marker_data = trial_data[trial_data['Marker'] == marker]
            y_values = marker_data['Y']
            stationary_values = marker_data['StationaryLabels']
            
            
            
            # Compute mean Euclidean distance between consecutive points
            coords = marker_data[['X', 'Y', 'Z']].values
            distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            mean_distance = distances.mean()
               
            summary_stats.append({
                'Marker': marker,
                'n_stationary': stationary_values.nunique(),
                'Start_Y': y_values.iloc[0],
                'Min_Y': y_values.min(),
                'Mean_Y': y_values.mean(),
                'Std_Y': y_values.std(),
                'Mean_Euc_Distance': mean_distance
            })
       # choose the marker with the largest Start_Y and lowest Mean_Euc_Distance as head marker
        summary_df = pd.DataFrame(summary_stats)
        # only keep the ones where 'Start_Y' is less than 0 (per position~ -500) 
        summary_df = summary_df[summary_df['Start_Y'] < 0]
        summary_df = summary_df.sort_values(by=['Start_Y', 'Mean_Euc_Distance'], ascending=[False, True])
        head_marker = summary_df.iloc[0]['Marker']
        
        # ends up prioritising Euclidean distance too much without a threshold.
        # when does it become not worth keeping head marker with higher Euclidean distance but higher Start_Y?
        if len(summary_df) > 1:
            if (summary_df.iloc[0]['Start_Y'] - summary_df.iloc[1]['Start_Y']) < 20:
                # head - neck distance 10 mm. If within 20 mm, check Euclidean distance
                if summary_df.iloc[0]['Mean_Euc_Distance']*0.70 > summary_df.iloc[1]['Mean_Euc_Distance']:
                    head_marker = summary_df.iloc[1]['Marker']
                # it doesn't really matter head or neck as long as  
        else:
            head_marker = summary_df.iloc[0]['Marker']  
                
        
    else: 
        head_marker = top_three_markers.iloc[0]['Marker']
    
    
    print(f"{bird}-{age}-{trial} summary_df:\n{summary_df}")
    print(f"{bird}-{age}-{trial} head_marker: {head_marker}")
    final = trial_data[trial_data['Marker'] == head_marker]
    
    return final

def head_marker_test2(trial_data):
    bird = trial_data['Bird'].iloc[0]
    age = trial_data['Age'].iloc[0]
    trial = trial_data['Takeoff'].iloc[0]

    # Count stationary labels per marker
    markers_df = trial_data.groupby('Marker')['StationaryLabels'].nunique().reset_index()
    markers_df.columns = ['Marker', 'StationaryLabels']
    sorted_markers = markers_df.sort_values(by='StationaryLabels', ascending=False)

    # Top three markers
    top_three_markers_df = sorted_markers.head(3)

    # Apply filtering rules
    if len(top_three_markers_df) == 3:
        if (top_three_markers_df.iloc[0]['StationaryLabels'] - top_three_markers_df.iloc[1]['StationaryLabels']) > 3:
            top_three_markers_df = top_three_markers_df.head(1)
        elif (top_three_markers_df.iloc[1]['StationaryLabels'] - top_three_markers_df.iloc[2]['StationaryLabels']) > 3:
            top_three_markers_df = top_three_markers_df.head(2)

    # Build summary stats if more than one candidate
    summary_df = None
    if len(top_three_markers_df) >= 2:
        summary_stats = []
        for marker in top_three_markers_df['Marker']:
            marker_data = trial_data[trial_data['Marker'] == marker]
            y_values = marker_data['Y']
            stationary_values = marker_data['StationaryLabels']

            coords = marker_data[['X', 'Y', 'Z']].values
            distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            mean_distance = distances.mean()

            summary_stats.append({
                'Marker': marker,
                'n_stationary': stationary_values.nunique(),
                'Start_Y': y_values.iloc[0],
                'Min_Y': y_values.min(),
                'Mean_Y': y_values.mean(),
                'Std_Y': y_values.std(),
                'Mean_Euc_Distance': mean_distance
            })

        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df[summary_df['Start_Y'] < 0]

        if not summary_df.empty:
            summary_df = summary_df.sort_values(by=['Start_Y', 'Mean_Euc_Distance'], ascending=[False, True])
            head_marker = summary_df.iloc[0]['Marker']

            # Threshold rule
            if len(summary_df) > 1:
                if (summary_df.iloc[0]['Start_Y'] - summary_df.iloc[1]['Start_Y']) < 20:
                    if summary_df.iloc[0]['Mean_Euc_Distance'] * 0.70 > summary_df.iloc[1]['Mean_Euc_Distance']:
                        head_marker = summary_df.iloc[1]['Marker']
        else:
            head_marker = top_three_markers_df.iloc[0]['Marker']
    else:
        head_marker = top_three_markers_df.iloc[0]['Marker']

    # Debug prints
    if summary_df is not None:
        print(f"{bird}-{age}-{trial} summary_df:\n{summary_df}")
    print(f"{bird}-{age}-{trial} head_marker: {head_marker}")

    final = trial_data[trial_data['Marker'] == head_marker]
    return final



## trial marker choosing function (2024)
def head_marker_xyz(trial_data):
    
    #trial_data
    #a dataframe with bird, age, and trial


    #markers df
    markers_df = trial_data.groupby('Marker')['StationaryLabels'].nunique().reset_index()
    markers_df.columns = ['Marker', 'StationaryLabels'] 
    
    # Sort markers by stationary label values in descending order
    sorted_markers = markers_df.sort_values(by='StationaryLabels', ascending=False)
    
    # Select the top two markers. top_two_markers include the columns 'pinkR-blueL', 'Marker', 'StationaryLabels'
    top_two_markers_df = sorted_markers.head(2)
    top_two_markers = top_two_markers_df[['Marker']]
    
    # Check if the difference between the highest and the second highest stationary label values is more than 3
    if len(top_two_markers) == 2 and (top_two_markers_df.iloc[0]['StationaryLabels'] - top_two_markers_df.iloc[1]['StationaryLabels']) > 3:
        # Drop the second marker
        top_two_markers = top_two_markers.head(1)
    
    # If there are still two markers
    if len(top_two_markers) == 2:
        # Filter the trial_data by each marker
        marker_1 = top_two_markers['Marker'].iloc[0]
        marker_2 = top_two_markers['Marker'].iloc[1]

        print(f"marker1: {marker_1}")
        print(f"marker2: {marker_2}")

        trial_data_marker_1 = trial_data[trial_data['Marker'] == marker_1]
        trial_data_marker_2 = trial_data[trial_data['Marker'] == marker_2]

        std_marker_1 = trial_data_marker_1['Y'].std()
        std_marker_2 = trial_data_marker_2['Y'].std()
        min_y_marker_1 = trial_data_marker_1['Y'].min()
        min_y_marker_2 = trial_data_marker_2['Y'].min()

        # Determine head marker based on std deviation and min Y value
        if abs(std_marker_1 - std_marker_2) < 5:
            if min_y_marker_2 > min_y_marker_1:
                head_marker = marker_2
            else:
                head_marker = marker_1
        else:
            head_marker = marker_1 if std_marker_1 < std_marker_2 else marker_2

        # add the part if the other marker than chosen head marker has higher min Y value and lower 
        # Euclidean distance mean between consecutive points, that other marker becomes head marker.
        

    else:
        # If there is only one marker, it is the head marker
        head_marker = top_two_markers.iloc[0]['Marker']
        
    print(f"head_marker: {head_marker}")
    final = trial_data[trial_data['Marker'] == head_marker]
    
    return final

#trial_data_marker_1
#trial_data_marker_2
#head_marker_xyz(trial_data) 

def remove_outliers_zscore(data, threshold):
    # Ensure that the input is a DataFrame and contains the necessary columns
    if isinstance(data, pd.DataFrame):
        if {'X', 'Y', 'Z'}.issubset(data.columns):
            # Calculate z-scores for the 'X', 'Y', 'Z' columns
            z_scores = np.abs(stats.zscore(data[['X', 'Y', 'Z']]))
            # Filter rows where all z-scores are below the threshold
            filtered_data = data[(z_scores < threshold).all(axis=1)]
            #print(f'z score filtered_data shape: {filtered_data.shape}')
            #print(f'filtered_data head:\n {filtered_data.head(5)}')
            return filtered_data
        
        else:
            raise ValueError("Input DataFrame must contain 'X', 'Y', 'Z' columns")
    else:
        raise ValueError("Input data should be a pandas DataFrame")
    
    #check that there are at least 20 first columns in the data survived to the filtered_data, and if not, do not append the trial to the final result. 

def process_all_combinations(df):
    results = []

    #df: takeoffs.csv where the trajectory data is filtered from the start of takeoff (max resultant force)

    # Get unique combinations of bird, age, and trial
    combinations = df[['Bird', 'Age', 'Takeoff']].drop_duplicates()

    # Loop through each combination
    for _, combo in tqdm(combinations.iterrows(), total=combinations.shape[0]):
        bird = combo['Bird']
        age = combo['Age']
        trial = combo['Takeoff']

        # Filter the dataframe for the current combination
        trial_data = df[(df['Bird'] == bird) & (df['Age'] == age) & (df['Takeoff'] == trial)].copy()
        # Apply the head_marker_xyz function to extract the head marker trajectory 
        coords = trial_data[['X', 'Y', 'Z']].values
        frames = trial_data['Frame'].values
        stationary_labels = label_stationary(coords, frames, 0.001)

        #add the stationary_markers array to trial_data
        trial_data.loc[:,'StationaryLabels'] = stationary_labels
        head_xyz = head_marker_test2(trial_data)

        # Check if result has at least 10 rows
        if len(head_xyz) < 10:
            continue

        # Get the first 10 rows of result
        head_xyz_10 = head_xyz.head(10)
        
        # Apply the remove_outliers_zscore function
        final = remove_outliers_zscore(head_xyz, threshold=2.4) #to get rid of ghost reading
        #print(f'shape of final: {final.shape}')
        
        # Check if the first 5 rows of result made it to the final
        merged = pd.merge(head_xyz_10, final, how='inner', indicator=True)
        #print(f'merged shape: {merged.shape}')
        #print(f'merged:\n {merged}')
        if len(merged) >= 5:
            # Append the result to the results list
            results.append(final)

    # Concatenate all results into a single DataFrame
    final_results = pd.concat(results, ignore_index=True)

    return final_results


def plot_flight_trajectory_grid(track_data_df1, df):
    fig = plt.figure(figsize=(24, 120))  # Adjust height for 25 rows
    flight_ids = track_data_df1['FlightID'].unique()

    rows, cols = 25, 4
    for i, flight_id in enumerate(flight_ids):
        trial_data = df[df['FlightID'] == flight_id]

        if trial_data.empty:
            print(f"Skipping FlightID {flight_id} — no data found.")
            continue

        age = trial_data['Age'].iloc[0]
        bird = trial_data['Bird'].iloc[0]
        takeoff = trial_data['Takeoff'].iloc[0]
        x = trial_data['X'].values
        y = trial_data['Y'].values
        z = trial_data['Z'].values
        time = trial_data['Time'].values

        # Normalize time for colormap
        norm = plt.Normalize(time.min(), time.max())
        colors = cm.plasma(norm(time))

        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        # Plot trajectory segment by segment with gradient colors
        for j in range(len(x) - 1):
            ax.plot(y[j:j+2], x[j:j+2], z[j:j+2],  color=colors[j], linewidth = 4)

        ax.set_title(f'{flight_id}_{bird}_{age}_{takeoff}', fontsize=20, y = 0.2)
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_zlabel('Z')
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()






def head_marker_prints_for_review(trial_data):
    
    #trial_data
    #a dataframe with bird, age, and trial
    bird = trial_data['Bird'].iloc[0]
    age = trial_data['Age'].iloc[0]
    trial = trial_data['Takeoff'].iloc[0]


    #markers df
    markers_df = trial_data.groupby('Marker')['StationaryLabels'].nunique().reset_index()
    markers_df.columns = ['Marker', 'StationaryLabels'] 
    
    # Sort markers by stationary label values in descending order
    sorted_markers = markers_df.sort_values(by='StationaryLabels', ascending=False)
    
    # Select the top three markers. top_two_markers include the columns 'pinkR-blueL', 'Marker', 'StationaryLabels'
    top_three_markers_df = sorted_markers.head(3)
    top_three_markers = top_three_markers_df[['Marker']]
    
    # Check if the difference between the highest and the second highest stationary label values is more than 3
    if len(top_three_markers) == 3 and (top_three_markers_df.iloc[0]['StationaryLabels'] - top_three_markers_df.iloc[1]['StationaryLabels']) > 3:
        # Drop the second marker and the third marker
        top_three_markers = top_three_markers.head(1)
    
    # if the difference between the second highest and third highest is more than three,
    elif len(top_three_markers) == 3 and (top_three_markers_df.iloc[1]['StationaryLabels'] - top_three_markers_df.iloc[2]['StationaryLabels']) > 3:
        # Drop the third marker
        top_three_markers = top_three_markers.head(2)
    
    
    # If there are still more than one markers
    if len(top_three_markers) >= 2:
        # make the summary dataframe for the remaining markers 
        # for the first Y value, mean, min, std, and Euclidean distance mean between consecutive points.
        summary_stats = []
        selected_markers = top_three_markers['Marker'].tolist()
        for marker in selected_markers:
            marker_data = trial_data[trial_data['Marker'] == marker]
            y_values = marker_data['Y']
            stationary_values = marker_data['StationaryLabels']
            
            
            
            # Compute mean Euclidean distance between consecutive points
            coords = marker_data[['X', 'Y', 'Z']].values
            distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            mean_distance = distances.mean()
               
            summary_stats.append({
                'Marker': marker,
                'n_stationary': stationary_values.nunique(),
                'Start_Y': y_values.iloc[0],
                'Min_Y': y_values.min(),
                'Mean_Y': y_values.mean(),
                'Std_Y': y_values.std(),
                'Mean_Euc_Distance': mean_distance
            })
       # choose the marker with the largest Start_Y and lowest Mean_Euc_Distance as head marker
        summary_df = pd.DataFrame(summary_stats)
        # only keep the ones where 'Start_Y' is less than 0 (per position~ -500) 
        summary_df = summary_df[summary_df['Start_Y'] < 0]
        summary_df = summary_df.sort_values(by=['Start_Y', 'Mean_Euc_Distance'], ascending=[False, True])
        head_marker = summary_df.iloc[0]['Marker']
        
        # ends up prioritising Euclidean distance too much without a threshold.
        # when does it become not worth keeping head marker with higher Euclidean distance but higher Start_Y?
        if len(summary_df) > 1:
            if (summary_df.iloc[0]['Start_Y'] - summary_df.iloc[1]['Start_Y']) < 20:
                # head - neck distance 10 mm. If within 20 mm, check Euclidean distance
                if summary_df.iloc[0]['Mean_Euc_Distance']*0.70 > summary_df.iloc[1]['Mean_Euc_Distance']:
                    head_marker = summary_df.iloc[1]['Marker']
                # it doesn't really matter head or neck as long as  
        else:
            head_marker = summary_df.iloc[0]['Marker']  
                
        
    else: 
        head_marker = top_three_markers.iloc[0]['Marker']
    
    
    print(f"{bird}-{age}-{trial} summary_df:\n{summary_df}")
    print(f"{bird}-{age}-{trial} head_marker: {head_marker}")
    final = trial_data[trial_data['Marker'] == head_marker]
    
    return final

def head_marker_prints_for_review(trial_data):


    # Count stationary labels per marker
    markers_df = trial_data.groupby('Marker')['StationaryLabels'].nunique().reset_index()
    markers_df.columns = ['Marker', 'StationaryLabels']
    sorted_markers = markers_df.sort_values(by='StationaryLabels', ascending=False)

    # Top three markers
    top_three_markers_df = sorted_markers.head(3)

    # Apply filtering rules
    if len(top_three_markers_df) == 3:
        if (top_three_markers_df.iloc[0]['StationaryLabels'] - top_three_markers_df.iloc[1]['StationaryLabels']) > 3:
            top_three_markers_df = top_three_markers_df.head(1)
        elif (top_three_markers_df.iloc[1]['StationaryLabels'] - top_three_markers_df.iloc[2]['StationaryLabels']) > 3:
            top_three_markers_df = top_three_markers_df.head(2)

    # Build summary stats if more than one candidate
    summary_df = None
    if len(top_three_markers_df) >= 2:
        summary_stats = []
        for marker in top_three_markers_df['Marker']:
            marker_data = trial_data[trial_data['Marker'] == marker]
            y_values = marker_data['Y']
            stationary_values = marker_data['StationaryLabels']

            coords = marker_data[['X', 'Y', 'Z']].values
            distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            mean_distance = distances.mean()

            summary_stats.append({
                'Marker': marker,
                'n_stationary': stationary_values.nunique(),
                'Start_Y': y_values.iloc[0],
                'Min_Y': y_values.min(),
                'Mean_Y': y_values.mean(),
                'Std_Y': y_values.std(),
                'Mean_Euc_Distance': mean_distance
            })

        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df[summary_df['Start_Y'] < 0]

        if not summary_df.empty:
            summary_df = summary_df.sort_values(by=['Start_Y', 'Mean_Euc_Distance'], ascending=[False, True])
            head_marker = summary_df.iloc[0]['Marker']

            # Threshold rule
            if len(summary_df) > 1:
                if (summary_df.iloc[0]['Start_Y'] - summary_df.iloc[1]['Start_Y']) < 20:
                    if summary_df.iloc[0]['Mean_Euc_Distance'] * 0.70 > summary_df.iloc[1]['Mean_Euc_Distance']:
                        head_marker = summary_df.iloc[1]['Marker']
        else:
            head_marker = top_three_markers_df.iloc[0]['Marker']
    else:
        head_marker = top_three_markers_df.iloc[0]['Marker']

    
    return {'summary_df': summary_df, 'head_marker': head_marker}



def review_flights(review_df, takeoffs, processed_df_final):

        """
        review_df: The .csv file with flights to review
        takeoffs: clipped flights with all the markers 
        processed_df_final: the outcome of process_all_combinations(takeoffs) and filtering
        
        Aim:
        show a 3D graph at processed_df (to re-check why I chose that flight to review)
        
        """
        
        summaries = []

        

        # for each row in review_df
        for idx, row in enumerate(review_df.itertuples(index = False), start = 1):
            bird, age, trial = row.Bird, row.Age, row.Trial


            # Filter data
            trial_data = takeoffs[(takeoffs['Bird'] == bird) & 
                                    (takeoffs['Age'] == age) & 
                                    (takeoffs['Takeoff'] == trial)]
            
            marker_data = processed_df_final[(processed_df_final['Bird'] == bird) & 
                                             (processed_df_final['Age'] == age) & 
                                             (processed_df_final['Takeoff'] == trial)]


            if trial_data.empty or marker_data.empty:
                    print(f"Skipping {bird}-{age}-{trial} — no data found.")
                    continue
            
            

            # 1. Compute stationary labels from trial_data with all the markers
            coords = trial_data[['X', 'Y', 'Z']].values
            frames = trial_data['Frame'].values
            stationary_labels = label_stationary(coords, frames, 0.001)

            if len(stationary_labels) != len(trial_data):
                print("Mismatch between stationary labels and data length.")
                continue

            trial_data = trial_data.copy()
            trial_data['StationaryLabels'] = stationary_labels

            #---------------------------------------
            # Figure 1: 3D Trajectory Plot
            #----------------------------------------

            
            if trial_data.empty or trial_data['Time'].dropna().empty or trial_data[['X', 'Y', 'Z']].dropna().empty:
                print(f"Skipping {idx}-{bird}-{age}-{trial} — missing valid data.")
                continue

            # Extract coordinates and time for 3D plot 
            x, y, z = marker_data['X'].values, marker_data['Y'].values, marker_data['Z'].values           
            time = marker_data['Time'].values


            # Create 3D plot
            fig3d = plt.figure(figsize=(8, 6))
            ax3d = fig3d.add_subplot(111, projection='3d')

            norm = plt.Normalize(time.min(), time.max())
            colors = cm.plasma(norm(time))


            # 1. 3D plot
            for j in range(len(x) - 1):
                    ax3d.plot(y[j:j+2], x[j:j+2], z[j:j+2],  color=colors[j], linewidth = 3)

            ax3d.set_title(f'{idx}-{bird}_{age}_{trial}', fontsize=16)
            ax3d.set_xlabel('Y')
            ax3d.set_ylabel('X')
            ax3d.set_zlabel('Z')
            ax3d.tick_params(labelsize=8)
            ax3d.legend(fontsize=9)

            plt.tight_layout()
            plt.show()

            #-----------------------------
            # Figure 2: Diagnostic Plots
            #-----------------------------
            fig_diag, axes = plt.subplots(3, 3, figsize=(14, 10))
            fig_diag.suptitle(f'{idx}-{bird}, age {age}, trial {trial}', fontsize = 15)


            # 1. Stationary label plots
            for i, col in enumerate(['X', 'Y', 'Z']):
                
                sc = axes[i, 0].scatter(
                trial_data.Time, trial_data[col],
                c=trial_data.StationaryLabels, cmap='viridis', s=10
                                        )
                axes[i, 0].set_title(f"Stat {col}", fontsize=10)
                axes[i, 0].tick_params(labelsize=8)
                fig_diag.colorbar(sc, ax=axes[i, 0])

            

            # 2. All marker plots
            distinct_colors = ['red', 'green', 'orange', 'purple', 'pink', 'cyan', 'blue','yellow','brown']
            marker_cmap = ListedColormap(distinct_colors)
            for i, col in enumerate(['X', 'Y', 'Z']):
                
                sc = axes[i, 1].scatter(
                    trial_data.Time, trial_data[col],
                    c=trial_data.Marker, cmap=marker_cmap, s=10
                )
                axes[i, 1].set_title(f"All {col} markers", fontsize=10)
                axes[i, 1].tick_params(labelsize=8)
                fig_diag.colorbar(sc, ax=axes[i, 1])

                

            # Summary
            summary = head_marker_prints_for_review(trial_data)
            summary_df = summary['summary_df']
            head_marker = summary['head_marker']



            # 3. Selected marker plots
            for i, col in enumerate(['X', 'Y', 'Z']):
                
                axes[i, 2].scatter(marker_data.Time, marker_data[col], color='blue', s=10)
                axes[i, 2].set_title(f"head marker: {head_marker}, {col}", fontsize=10)
                axes[i, 2].tick_params(labelsize=8)

            plt.tight_layout()
            plt.show()

            #----------------------------------------------
            # Bonus: Each Non-head marker plotted in its own independent row 
            #----------------------------------------------
            print(f'summary_df: {summary_df}')
            print(f'head_marker: {head_marker}')
            
            if not summary_df.empty:
                
                # Collect other markers excluding head marker
                other_markers_list = [m for m in summary_df['Marker'].unique() if (m != head_marker and pd.notna(m))]

                if len(other_markers_list) == 0:
                    print(f"No non-head markers to plot for {bird}-{age}-{trial}.")
                else:
                    # Dynamic figure: one row per other marker, 3 columns (X, Y, Z)
                    fig_bonus, axes_bonus = plt.subplots(len(other_markers_list), 3,
                                                        figsize=(12, 4 * len(other_markers_list)))
                    fig_bonus.suptitle(f"{idx}-Other Candidate Markers: {bird}-{age}-{trial}", fontsize=14)

                    # Normalize axes for single-row case
                    if len(other_markers_list) == 1:
                        axes_bonus = np.array([axes_bonus])

                    for row_idx, marker in enumerate(other_markers_list):
                        marker_subset = trial_data[trial_data['Marker'] == marker]

                        for col_idx, col in enumerate(['X', 'Y', 'Z']):
                            ax = axes_bonus[row_idx, col_idx]
                            ax.scatter(marker_subset['Time'], marker_subset[col],
                                    color='blue', alpha=0.7, label=f'Marker {marker}')
                            ax.set_title(f"{col} vs Time (Marker {marker})", fontsize=10)
                            ax.tick_params(labelsize=8)
                            ax.legend(fontsize=8)

                    plt.tight_layout()
                    plt.show()


                #summaries.append(summary)
            
            #summaries = pd.concat(summaries, ignore_index = True)


        return summaries

def review_individual_flight(flightID, takeoffs, processed_df_final):


    review_flight = takeoffs[takeoffs['FlightID'] == flightID]

    if review_flight.empty:
        print(f"Skipping {flightID} — no data found.")

    # save bird identity for plots
    bird = review_flight['Bird'].iloc[0]
    age = review_flight['Age'].iloc[0]
    trial = review_flight['Takeoff'].iloc[0]


    # 1. Compute stationary labels from trial_data with all the markers
    coords = review_flight[['X', 'Y', 'Z']].values
    frames = review_flight['Frame'].values
    stationary_labels = label_stationary(coords, frames, 0.001)


    if len(stationary_labels) != len(review_flight):
        print("Mismatch between stationary labels and data length.")

    # make stationary labels
    review_flight = review_flight.copy()
    review_flight['StationaryLabels'] = stationary_labels

    # configure Diagonistic plots
    fig_diag, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig_diag.suptitle(f'{bird}, age {age}, trial {trial}', fontsize = 15)

    # 1. Stationary Labels Plots
    for i, col in enumerate(['X', 'Y', 'Z']):
                
        sc = axes[i, 0].scatter(
        review_flight.Time, review_flight[col],
        c=review_flight.StationaryLabels, cmap='viridis', s=10
                                        )
        axes[i, 0].set_title(f"Stat {col}", fontsize=10)
        axes[i, 0].tick_params(labelsize=8)
        fig_diag.colorbar(sc, ax=axes[i, 0])

            

    # 2. All marker plots overlayed
    distinct_colors = ['red', 'green', 'orange', 'purple', 'pink', 'cyan', 'blue','yellow','brown']
    marker_cmap = ListedColormap(distinct_colors)
    for i, col in enumerate(['X', 'Y', 'Z']):
                
        sc = axes[i, 1].scatter(
        review_flight.Time, review_flight[col],
        c = review_flight.Marker, cmap=marker_cmap, s=10
                )
        axes[i, 1].set_title(f"All {col} markers", fontsize=10)
        axes[i, 1].tick_params(labelsize=8)
        fig_diag.colorbar(sc, ax=axes[i, 1])


    #----------------------------------------------
    # Bonus: Each Non-head marker plotted in its own independent row 
    #----------------------------------------------
    
    # Print summary stats

    # Build summary stats for all the markers
    summary_df = None
    markers_df = pd.DataFrame({'Marker' : review_flight['Marker'].unique()})

    head_marker = None
    summary_stats = []

    for marker in markers_df['Marker']:
        marker_data = review_flight[review_flight['Marker'] == marker]
        y_values = marker_data['Y']
        stationary_values = marker_data['StationaryLabels']

        coords = marker_data[['X', 'Y', 'Z']].values
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        mean_distance = distances.mean()

        summary_stats.append({
            'Marker': marker,
            'n_stationary': stationary_values.nunique(),
            'Start_Y': y_values.iloc[0],
            'Min_Y': y_values.min(),
            'Mean_Y': y_values.mean(),
            'Std_Y': y_values.std(),
            'Mean_Euc_Distance': mean_distance
            })

        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df[summary_df['Start_Y'] < 0] #Silencing this one beause some markers jump to the trajectory -> not silencing this one because the logic for head marker should be shown

        if not summary_df.empty:
            summary_df = summary_df.sort_values(by=['Start_Y', 'Mean_Euc_Distance'], ascending=[False, True])
            head_marker = summary_df.iloc[0]['Marker']

            # Threshold rule
            if len(summary_df) > 1:
                if (summary_df.iloc[0]['Start_Y'] - summary_df.iloc[1]['Start_Y']) < 20:
                    if summary_df.iloc[0]['Mean_Euc_Distance'] * 0.70 > summary_df.iloc[1]['Mean_Euc_Distance']:
                        head_marker = summary_df.iloc[1]['Marker']
                    else:
                        head_marker = summary_df.iloc[0]['Marker']
                else:
                    head_marker = summary_df.iloc[0]['Marker']

    # 3. Head Marker graph while showing head marker
    marker_data = processed_df_final[processed_df_final['FlightID'] == flightID]
    head_marker = marker_data['Marker'].iloc[0]

    for i, col in enumerate(['X', 'Y', 'Z']):
                
        axes[i, 2].scatter(marker_data.Time, marker_data[col], color='blue', s=10)
        axes[i, 2].set_title(f"head marker: {head_marker}, {col}", fontsize=10)
        axes[i, 2].tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()
    

    # print summary stats for all the markers
    print(f'summary_df: {summary_df}')
    print(f'head_marker: {head_marker}')
    # 3. Selected marker plots
            
    if not summary_df.empty:
                
        # Collect ALL other markers excluding head marker
        other_markers_list = [m for m in markers_df['Marker'].unique() if (m != head_marker and pd.notna(m))]

        if len(other_markers_list) == 0:
            print(f"No non-head markers to plot for {bird}-{age}-{trial}.")
        else:
            # Dynamic figure: one row per other marker, 3 columns (X, Y, Z)
            fig_bonus, axes_bonus = plt.subplots(len(other_markers_list), 3,
                                                        figsize=(12, 4 * len(other_markers_list)))
            fig_bonus.suptitle(f"Other Candidate Markers: {bird}-{age}-{trial}", fontsize=14)

            # Normalize axes for single-row case
            if len(other_markers_list) == 1:
                axes_bonus = np.array([axes_bonus])

            for row_idx, marker in enumerate(other_markers_list):
                marker_subset = review_flight[review_flight['Marker'] == marker]

                for col_idx, col in enumerate(['X', 'Y', 'Z']):
                    ax = axes_bonus[row_idx, col_idx]
                    ax.scatter(marker_subset['Time'], marker_subset[col],
                            color='blue', alpha=0.7, label=f'Marker {marker}')
                    ax.set_title(f"{col} vs Time (Marker {marker})", fontsize=10)
                    ax.tick_params(labelsize=8)
                    ax.legend(fontsize=8)

            plt.tight_layout()
            plt.show()

    
    return {'summary_df': summary_df, 'head_marker': head_marker}

def get_flight_performance(flights):

    # DataFrame to return
    flight_outcome = pd.DataFrame(columns=['FlightID','Bird', 'Age', 'Takeoff', 'mean_vx', 'mean_vy', 'mean_vz', 
                                          'max_vx', 'max_vy', 'max_vz', 'mean_acc_x', 'mean_acc_y', 'mean_acc_z',
                                          'max_acc_x', 'max_acc_y', 'max_acc_z',
                                          'mean_v_res', 'mean_acc_res', 'max_v_res', 'max_acc_res',
                                          'mean_turn_rate', 'mean_centri_acc', 'max_turn_rate', 'max_centri_acc'])

    # Group the dataframe by 'FlightID'
    grouped = flights.groupby(['FlightID'])


   # Iterate through each group with a progress bar
    for _, flight in tqdm(grouped, desc="Processing groups"):
        # Extract required values
        flightID = flight['FlightID'].iloc[0]
        bird = flight['Bird'].iloc[0]
        age = flight['Age'].iloc[0]
        takeoff = flight['Takeoff'].iloc[0]

        #Find max_Ftot, the peak force at take-off. 

        mean_vx = flight['inst_V_X'].mean()
        mean_vy = flight['inst_V_Y'].mean()
        mean_vz = flight['inst_V_Z'].mean()

        max_vx = flight['inst_V_X'].max()
        max_vy = flight['inst_V_Y'].max()
        max_vz = flight['inst_V_Z'].max()

        mean_acc_x = flight['inst_acc_X'].mean()
        mean_acc_y = flight['inst_acc_Y'].mean()
        mean_acc_z = flight['inst_acc_Z'].mean()

        max_acc_x = flight['inst_acc_X'].max()
        max_acc_y = flight['inst_acc_Y'].max()
        max_acc_z = flight['inst_acc_Z'].max()

        mean_v_res = flight['inst_V_res'].mean()
        mean_acc_res = flight['inst_acc_res'].mean()

        max_v_res = flight['inst_V_res'].max()
        max_acc_res = flight['inst_acc_res'].max()

        max_acc_mag = flight['inst_acc_mag'].max()
        mean_acc_mag = flight['inst_acc_mag'].mean()

        mean_turn_rate = flight['Turn_Rate_degrees'].mean()
        mean_centri_acc = flight['Centripetal_acc'].mean()

        max_turn_rate = flight['Turn_Rate_degrees'].max()
        max_centri_acc = flight['Centripetal_acc'].max()

        
        new_row = {

            'FlightID': flightID,
            'Bird': bird,
            'Age': age,
            'Takeoff': takeoff,
            'mean_vx': mean_vx, 
            'mean_vy': mean_vy, 
            'mean_vz': mean_vz, 

            'max_vx': max_vx, 
            'max_vy': max_vy, 
            'max_vz': max_vz, 

            'mean_acc_x': mean_acc_x, 
            'mean_acc_y': mean_acc_y, 
            'mean_acc_z': mean_acc_z,

            'max_acc_x': max_acc_x, 
            'max_acc_y': max_acc_y, 
            'max_acc_z': max_acc_z,

            'mean_v_res': mean_v_res, 
            'mean_acc_res': mean_acc_res, 
            'mean_acc_mag': mean_acc_mag,

            'max_v_res': max_v_res, 
            'max_acc_res': max_acc_res,
            'max_acc_mag': max_acc_mag,

            'mean_turn_rate': mean_turn_rate, 
            'mean_centri_acc': mean_centri_acc, 
            'max_turn_rate': max_turn_rate, 
            'max_centri_acc': max_centri_acc
            
        }

        # Append the new row to the force_outcome DataFrame
        flight_outcome = pd.concat([flight_outcome, pd.DataFrame([new_row])], ignore_index=True)

    return flight_outcome


def angle_between_vectors(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dot = x1*x2 + y1*y2 + z1*z2
    mag1 = math.sqrt(x1**2 + y1**2 + z1**2)
    mag2 = math.sqrt(x2**2 + y2**2 + z2**2)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_theta = max(min(dot/ (mag1 * mag2), 1.0), - 1.0)
    return math.degrees(math.acos(cos_theta))


def calculate_differences(copy):
    # Ensure the group is sorted by 'Time' or 'Frame' if necessary
    copy = copy.sort_values(by='Time').copy()

    #change mm -> m, Time is already in s
    copy['X'] = copy['X']/1000
    copy['Y'] = copy['Y']/1000
    copy['Z'] = copy['Z']/1000


    
    # Calculate the differences in X, Y, Z row with the next row
    copy['diff_X'] = copy['X'].diff()
    copy['diff_Y'] = copy['Y'].diff()
    copy['diff_Z'] = copy['Z'].diff()
    copy['diff_Time'] = copy['Time'].diff()

    # Instantaneous distance and velocity
    copy['inst_res_dist'] = np.sqrt(copy['diff_X']**2 + copy['diff_Y']**2 + copy['diff_Z']**2)

    copy['inst_V_res'] = copy['inst_res_dist'] / copy['diff_Time']  # diff_Time in s
    copy['inst_V_X'] = copy['diff_X'] / copy['diff_Time']  # diff_Time in s
    copy['inst_V_Y'] = copy['diff_Y'] / copy['diff_Time']  # diff_Time in s
    copy['inst_V_Z'] = copy['diff_Z'] / copy['diff_Time']  # diff_Time in s

    # Calculating Centripetal acceleration
    # previous segment: P[i] - P[i-1]
    turn_angles = [np.nan] * len(copy)

    for i in range(1, len(copy)-1):
       
        p1 = (copy.iloc[i]['X'] - copy.iloc[i-1]['X'],
              copy.iloc[i]['Y'] - copy.iloc[i-1]['Y'],
              copy.iloc[i]['Z'] - copy.iloc[i-1]['Z'])
        p2 = (copy.iloc[i+1]['X'] - copy.iloc[i]['X'],
              copy.iloc[i+1]['Y'] - copy.iloc[i]['Y'],
              copy.iloc[i+1]['Z'] - copy.iloc[i]['Z'])
        
        turn_angles[i] = angle_between_vectors(p1, p2) 

    copy['Turn_Angle'] = turn_angles 
    
    # turn rate and centripetal acceleration
    copy['Turn_Rate_degrees'] = copy['Turn_Angle'] / copy['diff_Time']
    copy['Turn_Rate_radians'] = copy['Turn_Rate_degrees'] * (math.pi/180.0)
    copy['Centripetal_acc'] = copy['Turn_Rate_radians'] * copy['inst_V_res']

    
    # Calculating instantaneous acceleration
    copy['inst_acc_res'] = copy['inst_V_res'].diff() / copy['diff_Time']
    copy['inst_acc_X'] = copy['inst_V_X'].diff() / copy['diff_Time']
    copy['inst_acc_Y'] = copy['inst_V_Y'].diff() / copy['diff_Time']
    copy['inst_acc_Z'] = copy['inst_V_Y'].diff() / copy['diff_Time']
    copy['inst_acc_mag'] = np.sqrt(copy['inst_acc_X']**2 + copy['inst_acc_Y']**2 + copy['inst_acc_Z']**2)

    # Cumulative metrics usually used for plots
    copy['cumul_moving_distance'] = copy['inst_res_dist'].cumsum()
    copy['cumul_time'] = copy['diff_Time'].cumsum()

    return copy



def compute_wingbeat_metrics(flights: pd.DataFrame, flight_outcome: pd.DataFrame) -> pd.DataFrame:
    """
    Compute wingbeat metrics for each FlightID and merge with flight outcome DataFrame.

    Parameters:
    flights (pd.DataFrame): DataFrame containing columns ['FlightID', 'cumul_time', 'inst_acc_Z']
    flight_outcome (pd.DataFrame): DataFrame with FlightID as a column

    Returns:
    pd.DataFrame: Updated DataFrame with wingbeat metrics added
    """

    results = []

    # Iterate through each FlightID
    for flight_id in flights['FlightID'].drop_duplicates():
        flight_data = flights[flights['FlightID'] == flight_id].sort_values('cumul_time')
        x = flight_data['cumul_time'].values
        y = flight_data['inst_acc_Z'].values

        # Find maxima and minima
        maxima, _ = find_peaks(y, prominence=1, distance=3)
        minima, _ = find_peaks(-y, prominence=1, distance=3)

        # Wingbeat frequency (Hz)
        if len(maxima) > 1:
            time_diffs = x[maxima][1:] - x[maxima][:-1]
            wingbeat_freq = 1 / time_diffs.mean()
        else:
            wingbeat_freq = float('nan')

        # upstroke, downstroke amplitudes
        paired_count = min(len(maxima), len(minima))
        if paired_count > 1:
            upstroke_amplitudes_acc = []
            downstroke_amplitudes_acc = []

            for i in range(paired_count - 1):
                upstroke_amplitudes_acc.append(abs(y[maxima[i+1]] - y[minima[i]]))
                downstroke_amplitudes_acc.append(abs(y[maxima[i]] - y[minima[i+1]]))

            upstroke_amp_mean_acc = np.mean(upstroke_amplitudes_acc)
            downstroke_amp_mean_acc = np.mean(downstroke_amplitudes_acc)
        else:
            upstroke_amp_mean_acc = downstroke_amp_mean_acc = float('nan')

        paired_count = min(len(maxima), len(minima))

        if paired_count > 1:
            upstroke_amplitudes_m = []
            downstroke_amplitudes_m = []

            for i in range(paired_count - 1):
                # Downstroke: minima[i] to maxima[i+1]
                if minima[i] < maxima[i+1]:
                    t_up = x[minima[i]:maxima[i+1]+1]
                    a_up = y[minima[i]:maxima[i+1]+1]
                    if len(t_up) > 1:
                        v_up = cumtrapz(a_up, t_up, initial=0)  # integrate acceleration -> velocity
                        disp_up = cumtrapz(v_up, t_up)[-1]  # integrate velocity -> displacement
                        downstroke_amplitudes_m.append(abs(disp_up))

                # Upstroke: maxima[i] to minima[i+1]
                if maxima[i] < minima[i+1]:
                    t_down = x[maxima[i]:minima[i+1]+1]
                    a_down = y[maxima[i]:minima[i+1]+1] 
                    if len(t_down) > 1:
                        v_down = cumtrapz(a_down, t_down, initial=0)
                        disp_down = cumtrapz(v_down, t_down)[-1]
                        upstroke_amplitudes_m.append(abs(disp_down))

            upstroke_amp_mean_m = np.mean(upstroke_amplitudes_m)
            downstroke_amp_mean_m = np.mean(downstroke_amplitudes_m)
        else:
            upstroke_amp_mean_m = downstroke_amp_mean_m = float('nan')


        results.append({
            'FlightID': flight_id,
            'wingbeat_frequency': wingbeat_freq,
            'upstroke_amplitude_acc': upstroke_amp_mean_acc,
            'downstroke_amplitude_acc': downstroke_amp_mean_acc,
            'upstroke_amplitude_m': upstroke_amp_mean_m,
            'downstroke_amplitude_m': downstroke_amp_mean_m
        })

    # Merge with flight_outcome
    metrics_df = pd.DataFrame(results).set_index('FlightID')
    flight_outcome_updated = pd.concat([flight_outcome.set_index('FlightID'), metrics_df], axis=1)

    return flight_outcome_updated

            


