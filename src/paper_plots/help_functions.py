import pandas as pd
import numpy as np
from pathlib import PurePath
import os


def load_release_time_summary():


    """
    Load the summary of release time from the true data.
    """
    path = PurePath("..", "..", "assets", "valid_true_data_pad_daily")
    files = [PurePath(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(f) for f in files])

    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df["date"] = pd.to_datetime(df["Datetime (UTC)"]).map(lambda x: x.date())
    number_of_date = len(df["date"].unique())
    number_of_release_hours = df.loc[(df["True Release Rate (kg/h)"] > 0) & (df["tag"] != 1), :].shape[0] / 3600
    number_of_comp_hours = df.loc[df["tag"] != 1, :].shape[0] / 3600
    percentage_of_release_hours = number_of_release_hours / number_of_comp_hours

    return number_of_date, number_of_comp_hours, number_of_release_hours, percentage_of_release_hours

def load_release_rate_summary():
    """
    Load the summary of release rate from the true data.
    """

    path = PurePath("..", "..", "assets", "valid_true_data_pad_daily")
    files = [PurePath(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df.loc[
        df["tag"] == 0, :
    ] # filter padding data, interpolate data, and non-formal testing data
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df[["Datetime (UTC)", "True Release Rate (kg/h)"]]

    min_rate = df.loc[df["True Release Rate (kg/h)"] > 0, "True Release Rate (kg/h)"].min()
    avg_rate = df.loc[df["True Release Rate (kg/h)"] > 0, "True Release Rate (kg/h)"].mean()
    max_rate = df.loc[df["True Release Rate (kg/h)"] > 0, "True Release Rate (kg/h)"].max()

    return {
        "min_rate": min_rate,
        "avg_rate": avg_rate,
        "max_rate": max_rate
    }

def load_greatest_variability_rate_of_true_events():
    path = PurePath("..", "..", "assets", "valid_true_data_pad_daily")
    files = [PurePath(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(f) for f in files])   
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"]) 

    path = PurePath("..", "..", "assets", "events_PN")
    events_file = PurePath(path, "true_events.csv")
    df_events = pd.read_csv(events_file)

    P_events = df_events.loc[df_events["Label"] == "P", :].reset_index(drop=True)
    drop_index = []

    mean_release_of_true_events = []
    for i in range(P_events.shape[0]):
        stime = P_events.loc[i, "ManualStartDateTime"]
        etime = P_events.loc[i, "ManualEndDateTime"]
        mean_release = get_event_release_rate(df, stime, etime)[1]
        if mean_release is np.nan:
            drop_index.append(i)
        else:
            mean_release_of_true_events.append(mean_release)
    P_events = P_events.drop(drop_index).reset_index(drop=True)
   
    mean_release_of_true_events = np.array(mean_release_of_true_events)
    greatest_variability = np.max(mean_release_of_true_events) - np.min(mean_release_of_true_events)
    min_idx = np.argmin(mean_release_of_true_events)
    print("min release event: ", P_events.loc[min_idx, "ManualStartDateTime"], P_events.loc[min_idx, "ManualEndDateTime"])
    max_idx = np.argmax(mean_release_of_true_events)
    print("max release event: ", P_events.loc[max_idx, "ManualStartDateTime"], P_events.loc[max_idx, "ManualEndDateTime"])
    
    return greatest_variability, np.min(mean_release_of_true_events), np.max(mean_release_of_true_events)

def get_event_release_rate(df, stime, etime):
    """
    Get the release rate of a given event.
    """
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df_copy = df.loc[
        (df["Datetime (UTC)"] >= stime) &
        (df["Datetime (UTC)"] <= etime) &
        (df["True Release Rate (kg/h)"] > 0),
        "True Release Rate (kg/h)"
    ]
    return df_copy.min(), df_copy.mean(), df_copy.max()

def get_valid_N_idx(df_events):
    n = len(df_events)
    if n == 0:
        return []
    
    valid_N_idx = []
    for i in range(n):
        if i == 0 or i == n - 1:
            continue
        else:
            if df_events.loc[i, "Label"] == "N" and df_events.loc[i-1, "Label"] == "P" and df_events.loc[i+1, "Label"] == "P":
                valid_N_idx.append(i)
    
    return valid_N_idx

def get_N_max_duration(df_release):
    """
    max duration of Non-release, it can be two or more N events
    """
    max_duration = 0
    df_release["Datetime (UTC)"] = pd.to_datetime(df_release["Datetime (UTC)"])
    df_release_copy = df_release.reset_index(drop=True)
    df_release_copy["date"] = df_release_copy["Datetime (UTC)"].map(lambda x: x.date())
    # for 2022-10-17~2022-10-22, set True Release Rate (kg/h) to NaN
    df_release_copy.loc[
                        ((df_release_copy["date"] >= pd.to_datetime("2022-10-17").date()) &
                        (df_release_copy["date"] <= pd.to_datetime("2022-10-22").date())) |
                        (df_release_copy["date"] == pd.to_datetime("2022-10-14").date()), "True Release Rate (kg/h)"] = np.nan
    start_second = 0
    end_second = 0
    for i in range(df_release_copy.shape[0]):
        end_second = i
        if df_release_copy.loc[i, "True Release Rate (kg/h)"] != 0:
            if end_second - start_second > max_duration:
                max_duration = max(max_duration, end_second - start_second)
            start_second = i
        else:
            continue
    if end_second - start_second > max_duration:
        max_duration = max(max_duration, end_second - start_second)
    return max_duration / 3600

def load_true_events_summary():
    path = PurePath("..", "..", "assets", "events_PN")
    # before wind transpose model
    file_before = PurePath(path, "candidates.csv")
    # after wind transpose model
    file_after = PurePath(path, "true_events.csv")
    df_before = pd.read_csv(file_before)
    df_after = pd.read_csv(file_after)
    df_before["ManualStartDateTime"] = pd.to_datetime(df_before["ManualStartDateTime"])
    df_before["ManualEndDateTime"] = pd.to_datetime(df_before["ManualEndDateTime"])
    df_after["ManualStartDateTime"] = pd.to_datetime(df_after["ManualStartDateTime"])
    df_after["ManualEndDateTime"] = pd.to_datetime(df_after["ManualEndDateTime"])

    df_before["date"] = df_before["ManualStartDateTime"].map(lambda x: x.date())
    df_after["date"] = df_after["ManualStartDateTime"].map(lambda x: x.date())

    # filter the events in 2022-10-17~2022-10-22 and filter 2022-10-14
    df_before = df_before.loc[
        ((df_before["date"] < pd.to_datetime("2022-10-17").date()) |
        (df_before["date"] > pd.to_datetime("2022-10-22").date())) &
        (df_before["date"] != pd.to_datetime("2022-10-14").date()), :
    ]
    df_before = df_before.reset_index(drop=True)
    df_after = df_after.loc[
        ((df_after["date"] < pd.to_datetime("2022-10-17").date()) |
        (df_after["date"] > pd.to_datetime("2022-10-22").date())) &
        (df_after["date"] != pd.to_datetime("2022-10-14").date()), :
    ]
    df_after = df_after.reset_index(drop=True)

    # load release rate
    path = PurePath("..", "..", "assets", "valid_true_data_pad_daily")
    files = [PurePath(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    df_release = pd.concat([pd.read_csv(f) for f in files])
    df_release["Datetime (UTC)"] = pd.to_datetime(df_release["Datetime (UTC)"])
    df_release = df_release.sort_values(by="Datetime (UTC)")


    before_results = {}
    after_results = {}

    # number of P events
    before_results["number_of_P_events"] = (df_before["Label"] == "P").sum()
    after_results["number_of_P_events"] = (df_after["Label"] == "P").sum()

    # number of N events
    before_results["number_of_N_events"] = (df_before["Label"] == "N").sum()
    after_results["number_of_N_events"] = (df_after["Label"] == "N").sum()

    # calculate durations of events, duration is defined as the time between the ManualStartDateTime and ManualEndDateTime
    df_before["ManualStartDateTime"] = pd.to_datetime(df_before["ManualStartDateTime"])
    df_before["ManualEndDateTime"] = pd.to_datetime(df_before["ManualEndDateTime"])
    df_before["duration"] = df_before["ManualEndDateTime"] - df_before["ManualStartDateTime"]
    df_before["duration"] = df_before["duration"].apply(lambda x: x.total_seconds() / 60)

    df_after["ManualStartDateTime"] = pd.to_datetime(df_after["ManualStartDateTime"])
    df_after["ManualEndDateTime"] = pd.to_datetime(df_after["ManualEndDateTime"])
    df_after["duration"] = df_after["ManualEndDateTime"] - df_after["ManualStartDateTime"]
    df_after["duration"] = df_after["duration"].apply(lambda x: x.total_seconds() / 60)

    # minumum release rate of P events
    before_results["min_release_rate_of_P_events"] = df_release.loc[df_release["True Release Rate (kg/h)"] > 0, "True Release Rate (kg/h)"].min()
    after_results["min_release_rate_of_P_events"] = df_release.loc[df_release["True Release Rate (kg/h)"] > 0, "True Release Rate (kg/h)"].min()

    # maximum release rate of P events
    before_results["max_release_rate_of_P_events"] = df_release.loc[df_release["True Release Rate (kg/h)"] > 0, "True Release Rate (kg/h)"].max()
    after_results["max_release_rate_of_P_events"] = df_release.loc[df_release["True Release Rate (kg/h)"] > 0, "True Release Rate (kg/h)"].max()

    # minumum duration of P events
    before_results["min_duration_of_P_events"] = df_before.loc[df_before["Label"] == "P", "duration"].min() / 60 # hours
    after_results["min_duration_of_P_events"] = df_after.loc[df_after["Label"] == "P", "duration"].min() / 60 # hours

    # maximum duration of P events
    before_results["max_duration_of_P_events"] = df_before.loc[df_before["Label"] == "P", "duration"].max() / 60 # hours
    after_results["max_duration_of_P_events"] = df_after.loc[df_after["Label"] == "P", "duration"].max() / 60 # hours

    # average duration of P events
    before_results["avg_duration_of_P_events"] = df_before.loc[df_before["Label"] == "P", "duration"].mean() / 60 # hours
    after_results["avg_duration_of_P_events"] = df_after.loc[df_after["Label"] == "P", "duration"].mean() / 60 # hours

    # minumum and maximum duration of N events
    before_results["min_duration_of_N_events"] = df_before.loc[df_before["Label"] == "N", "duration"].min()
    after_results["min_duration_of_N_events"] = df_after.loc[df_after["Label"] == "N", "duration"].min()

    before_results["max_duration_of_N_events"] = df_before.loc[df_before["Label"] == "N", "duration"].max()
    after_results["max_duration_of_N_events"] = df_after.loc[df_after["Label"] == "N", "duration"].max()

    # average duration of N events
    before_results["avg_duration_of_N_events"] = df_before.loc[df_before["Label"] == "N", "duration"].mean()
    after_results["avg_duration_of_N_events"] = df_after.loc[df_after["Label"] == "N", "duration"].mean()


    # mean release rate of min duration P event and max duration P event
    # 1. the min duration P event
    min_duration_P_event_idx = df_before.loc[df_before["duration"] / 60 == before_results["min_duration_of_P_events"], :].index[0]
    # 2. the release rate of min duration P event
    stime = df_before.loc[min_duration_P_event_idx, "ManualStartDateTime"]
    etime = df_before.loc[min_duration_P_event_idx, "ManualEndDateTime"]
    df_release["Datetime (UTC)"] = pd.to_datetime(df_release["Datetime (UTC)"])
    before_results["minP_duration"] = (etime - stime) / pd.Timedelta("1m")
    min_release, avg_release, max_release = get_event_release_rate(df_release, stime, etime)
    before_results["minP_min_rate"] = min_release
    before_results["minP_avg_rate"] = avg_release
    before_results["minP_max_rate"] = max_release

    # mean release rate of max duration P event
    max_duration_P_event_idx = df_before.loc[df_before["duration"] / 60 == before_results["max_duration_of_P_events"], :].index[0]
    # 2. the release rate of min duration P event
    stime = df_before.loc[max_duration_P_event_idx, "ManualStartDateTime"] 
    etime = df_before.loc[max_duration_P_event_idx, "ManualEndDateTime"]
    df_release["Datetime (UTC)"] = pd.to_datetime(df_release["Datetime (UTC)"])
    before_results["maxP_duration"] = (etime - stime) / pd.Timedelta("1m")
    min_release, avg_release, max_release = get_event_release_rate(df_release, stime, etime)
    before_results["maxP_min_rate"] = min_release
    before_results["maxP_avg_rate"] = avg_release
    before_results["maxP_max_rate"] = max_release

    min_duration_P_event_idx = df_after.loc[df_after["duration"] / 60 == after_results["min_duration_of_P_events"], :].index[0]
    # 2. the release rate of min duration P event
    stime = df_after.loc[min_duration_P_event_idx, "ManualStartDateTime"]
    etime = df_after.loc[min_duration_P_event_idx, "ManualEndDateTime"]
    df_release["Datetime (UTC)"] = pd.to_datetime(df_release["Datetime (UTC)"])
    after_results["minP_duration"] = (etime - stime) / pd.Timedelta("1m")
    min_release, avg_release, max_release = get_event_release_rate(df_release, stime, etime)
    after_results["minP_min_rate"] = min_release
    after_results["minP_avg_rate"] = avg_release
    after_results["minP_max_rate"] = max_release

    # mean release rate of max duration P event
    max_duration_P_event_idx = df_after.loc[df_after["duration"] / 60 == after_results["max_duration_of_P_events"], :].index[0]
    # 2. the release rate of min duration P event
    stime = df_after.loc[max_duration_P_event_idx, "ManualStartDateTime"]
    etime = df_after.loc[max_duration_P_event_idx, "ManualEndDateTime"]
    df_release["Datetime (UTC)"] = pd.to_datetime(df_release["Datetime (UTC)"])
    after_results["maxP_duration"] = (etime - stime) / pd.Timedelta("1m")
    min_release, avg_release, max_release = get_event_release_rate(df_release, stime, etime)
    after_results["maxP_min_rate"] = min_release
    after_results["maxP_avg_rate"] = avg_release
    after_results["maxP_max_rate"] = max_release


    # get valid N events, which means before of N event is P event and after of N event is P event

    # 1. get the before of N event is P event
    valid_N_idx = get_valid_N_idx(df_before)
    df_valid_N = df_before.iloc[valid_N_idx]
    # 2. min, avg, max duration of valid N events
    before_results["N_min_duration"] = df_valid_N["duration"].min() / 60 # hours
    before_results["N_max_duration"] = get_N_max_duration(df_release)
    before_results["N_avg_duration"] = df_valid_N["duration"].mean() / 60 # hours
    
    valid_N_idx = get_valid_N_idx(df_after)
    df_valid_N = df_after.iloc[valid_N_idx]
    # 2. min, avg, max duration of valid N events
    after_results["N_min_duration"] = df_valid_N["duration"].min() / 60 # hours
    after_results["N_max_duration"] = get_N_max_duration(df_release)
    after_results["N_avg_duration"] = df_valid_N["duration"].mean() / 60 # hours


    # merge before and after results

    return {
        "before": before_results,
        "after": after_results
    }

def get_longest_non_emission():
    path = PurePath("..", "..", "assets", "valid_true_data_pad_daily")
    files = [PurePath(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    df_release = pd.concat([pd.read_csv(f) for f in files])
    df_release["Datetime (UTC)"] = pd.to_datetime(df_release["Datetime (UTC)"])
    df_release = df_release.sort_values(by="Datetime (UTC)")

    df_release.reset_index(drop=True, inplace=True)

    max_duration = 0
    max_duration_start_time = ""
    max_duration_end_time = ""
    s_time = ""
    e_time = ""

    i = 0
    while(i < df_release.shape[0]):
        if df_release.loc[i, "True Release Rate (kg/h)"] == 0:
            s_time = df_release.loc[i, "Datetime (UTC)"]
            while(i < df_release.shape[0] and df_release.loc[i, "True Release Rate (kg/h)"] == 0):
                i += 1
            e_time = df_release.loc[i-1, "Datetime (UTC)"]
            duration = (e_time - s_time) / pd.Timedelta("1h")
            if duration > max_duration:
                max_duration = duration
                max_duration_start_time = s_time
                max_duration_end_time = e_time
        else:
            i += 1

    return max_duration, max_duration_start_time, max_duration_end_time


def load_time_based_samples_summary():
    path = PurePath("..", "..", "results", "03_DetectionAnalysis", "Test-case Matching Data", "Time-based Events")
    sensor_names = [
        "Andium", "Canary", "Ecoteco", "Kuva", "Oiler"
    ]
    sample_size = []
    for sensor in sensor_names:
        file_path = PurePath(path, f"2xradius_60xseconds_{sensor}_match_events.csv")
        df = pd.read_csv(file_path)
        sample_size.append(df["norm_class"].notna().sum())
    
    return {
        "samples": {
            "Andium": sample_size[0],
            "Canary": sample_size[1],
            "Ecoteco": sample_size[2],
            "Kuva": sample_size[3],
            "Oiler": sample_size[4]
        }
    }

def get_inconsistency_of_event_based_classification():
    sensors = ["Andium", "Canary", "Ecoteco", "Kuva", "Oiler", "Qube", "Sensirion"]
    stanford_defined_events_root = PurePath("../..", "results", "03_DetectionAnalysis", "Test-case Matching Data", "Stanford Defined Events")
    team_defeind_events_root = PurePath("../..", "results", "03_DetectionAnalysis", "Test-case Matching Data", "Team Defined Events")

    filename_pattern = "2xradius_60xseconds_{}_match_events.csv"
    columns = [
        "sensor",
        "Number of all Stanford Define Events",
        "Number of Stanford Defined Events that account the greatest overlap",
        "Number of all Team Defined Events",
        "Number of Team Defined Events that account the greatest overlap"
    ]
    df = pd.DataFrame(columns=columns)

    for sensor in sensors:
        filename = PurePath(stanford_defined_events_root, filename_pattern.format(sensor))
        tmp_events_df = pd.read_csv(filename)
        number_of_all_stanford_defined_events = tmp_events_df.shape[0]
        number_of_inconsistent_stanford_defined_events = (tmp_events_df["cutoff"] == True).sum()

        filename = PurePath(team_defeind_events_root, filename_pattern.format(sensor))
        tmp_events_df = pd.read_csv(filename)
        number_of_all_team_defined_events = tmp_events_df.shape[0]
        number_of_inconsistent_team_defined_events = (tmp_events_df["cutoff"] == True).sum()
        idx = df.shape[0]
        df.loc[idx, columns] = [sensor, 
                          number_of_all_stanford_defined_events, 
                          number_of_inconsistent_stanford_defined_events,
                          number_of_all_team_defined_events,
                          number_of_inconsistent_team_defined_events]
        
    return df

def plot_end_points(non_zero_release_start, 
                    non_zero_release_end, 
                    next_non_zero_release_start,
                    threshold=2):
    """
    Plot the end points of the release events.
    """
    non_zero_release_start = pd.to_datetime(non_zero_release_start)
    non_zero_release_end = pd.to_datetime(non_zero_release_end)
    next_non_zero_release_start = pd.to_datetime(next_non_zero_release_start)
    # load wind volume data
    # Get wind speed data, wind speed data stored in ../../assets/all_wind_data, contains 5 files
    # Read in wind speed data from a csv files
    # 1. set root path and file name list 
    root_path = PurePath("../../assets/all_wind_data/")
    file_name_list = [
        "all_wind_1.csv",
        "all_wind_2.csv",
        "all_wind_3.csv",
        "all_wind_4.csv",
        "all_wind_5.csv"
    ]

    # initialize a DataFrame to store all wind speed data, columns = [datetime, windspeed, winddirection, day, month, u_east, u_north]
    wind_speed = pd.DataFrame(columns=["datetime", "windspeed", "winddirection", "day", "month", "u_east", "u_north"])
    # 2. read in all wind speed data
    for file_name in file_name_list:
        tmp_df = pd.read_csv(PurePath(root_path, file_name), low_memory=False)
        tmp_df["datetime"] = pd.to_datetime(tmp_df["datetime"])
        # keep only the columns we need
        tmp_df = tmp_df[["datetime", "windspeed", "winddirection", "day", "month", "u_east", "u_north"]]
        # append to all_wind_speed
        wind_speed = pd.concat([wind_speed, tmp_df], ignore_index=True)

    # 3. sort by datetime
    wind_speed.sort_values(by="datetime", inplace=True)
    wind_speed = wind_speed.reset_index(drop=True)
    # Calculate the wind speed magnitude (u_norm) using the east and north components (u_east and u_north)
    wind_speed["u_norm"] = np.sqrt(wind_speed["u_east"] ** 2 + wind_speed["u_north"] ** 2)

    # Fill in any missing values using the previous value (pad method)
    wind_speed.fillna(method="pad", inplace=True)


    # set radius
    radius = 82 * threshold # meters

    # Calculate the travel distance of wind particles from starttime to targettime for each second
    internal_df_windspeed = wind_speed.loc[wind_speed['datetime'].between(non_zero_release_start, next_non_zero_release_start, inclusive="both"), :]
    internal_df_windspeed.reset_index(drop=True, inplace=True)
    internal_df_windspeed["sum_u_east"] = internal_df_windspeed["u_east"].cumsum()
    internal_df_windspeed["sum_u_north"] = internal_df_windspeed["u_north"].cumsum()


    plot_time = non_zero_release_start
    outside_experiment = True
    outside_experiment_travel_list_x = []
    outside_experiment_travel_list_y = []
    inside_experiment_travel_list_x = []
    inside_experiment_travel_list_y = []
    max_travel_dis = 0
    while(plot_time <= non_zero_release_end):
        # Calculate the travel distance of wind particles from plot_time to targettime
        # 1. get the index of plot_time
        plot_time_idx = internal_df_windspeed.loc[internal_df_windspeed["datetime"] == plot_time, :].index[0]
        # 2. calculate the wind volumn
        plot_time_east_volumn = internal_df_windspeed.loc[plot_time_idx, "sum_u_east"]
        plot_time_north_volumn = internal_df_windspeed.loc[plot_time_idx, "sum_u_north"]

        target_time_east_volumn = internal_df_windspeed.loc[internal_df_windspeed.shape[0] - 1, "sum_u_east"]
        target_time_north_volumn = internal_df_windspeed.loc[internal_df_windspeed.shape[0] - 1, "sum_u_north"]

        delta_x = target_time_east_volumn - plot_time_east_volumn
        delta_y = target_time_north_volumn - plot_time_north_volumn

        # 3. calculate the travel distance
        travel_distance = np.sqrt((target_time_east_volumn - plot_time_east_volumn) ** 2 + (target_time_north_volumn - plot_time_north_volumn) ** 2)
        max_travel_dis = max(max_travel_dis, travel_distance)

        if travel_distance >= radius and outside_experiment:
            outside_experiment_travel_list_x.append(delta_x)
            outside_experiment_travel_list_y.append(delta_y)
        elif travel_distance < radius:
            inside_experiment_travel_list_x.append(delta_x)
            inside_experiment_travel_list_y.append(delta_y)
            outside_experiment = False
        else:
            inside_experiment_travel_list_x.append(delta_x)
            inside_experiment_travel_list_y.append(delta_y)
        
        plot_time = plot_time + pd.Timedelta("1s")
    
    # plot the travel distance
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.figure(figsize=(12, 12)), plt.gca()
    # ax.circle((0, 0), radius, color="blue", fill=False, label="radius: {}(m)".format(radius))
    circle = Circle((0, 0), radius, edgecolor='blue', label="experiment range", facecolor='none')
    ax.add_patch(circle)
    ax.plot(outside_experiment_travel_list_x, outside_experiment_travel_list_y, label="end points outside \n experiment range", color="red", linewidth=2)
    ax.plot(inside_experiment_travel_list_x, inside_experiment_travel_list_y, label="end points inside \n experiment range", color="green", linewidth=2)    

    if len(outside_experiment_travel_list_x) == 0:
        ax.plot([inside_experiment_travel_list_x[0]], [inside_experiment_travel_list_y[0]], marker='o', markersize=6, color="red", label="start time of non-zero \n release period")
        ax.plot([inside_experiment_travel_list_x[-1]], [inside_experiment_travel_list_y[-1]], marker='o', markersize=6, color="green", label="end time of non-zero \n release period")
    elif len(inside_experiment_travel_list_x) == 0:
        ax.plot([outside_experiment_travel_list_x[0]], [outside_experiment_travel_list_y[0]], marker='o', markersize=6, color="red", label="start time of non-zero \n release period")
        ax.plot([outside_experiment_travel_list_x[-1]], [outside_experiment_travel_list_y[-1]], marker='o', markersize=6, color="green", label="end time of non-zero \n release period")
    else:
        ax.plot([outside_experiment_travel_list_x[0]], [outside_experiment_travel_list_y[0]], marker='o', markersize=6, color="red", label="start time of non-zero \n release period")
        ax.plot([inside_experiment_travel_list_x[-1]], [inside_experiment_travel_list_y[-1]], marker='o', markersize=6, color="green", label="end time of non-zero \n release period")

    ax.set_xlim(-max_travel_dis - 100, max_travel_dis + 100)
    ax.set_ylim(-max_travel_dis - 100, max_travel_dis + 100)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("End points of particle emitted starting from start time and end time of non-zero release period")

    ax.legend()

    return ax

def load_statistical_data_of_error_bars():
    data_path = PurePath("../..", "results", "04_QuantificationAnalysis", "Daily-based")
    data_pattern = "daily_metrics_for_{}.csv"
    sensors = ["Canary", "Oiler", "Qube", "Sensirion", "Soofie"]

    columns = [
        "sensor",
        "type",
        "count",
        "min",
        "max",
        "mean",
        "std"
    ]

    error_bars_df = pd.DataFrame(columns=columns)

    for sensor in sensors:

        tmp_df = pd.read_csv(PurePath(data_path, data_pattern.format(sensor)))
        true_uncertainty = tmp_df["true_uncertainty"]
        report_uncertainty = tmp_df["reported_uncertainty"]

        count = tmp_df.shape[0]

        true_min = true_uncertainty.min()
        true_max = true_uncertainty.max()
        true_mean = true_uncertainty.mean()
        true_std = true_uncertainty.std()

        report_min = report_uncertainty.min()
        report_max = report_uncertainty.max()
        report_mean = report_uncertainty.mean()
        report_std = report_uncertainty.std()


        idx = error_bars_df.shape[0]
        error_bars_df.loc[idx, columns] = [sensor, "Uncertainty of Release", count, true_min, true_max, true_mean, true_std]
        idx = error_bars_df.shape[0]
        error_bars_df.loc[idx, columns] = [sensor, "Uncertainty of Report", count, report_min, report_max, report_mean, report_std]
    
    return error_bars_df


    






