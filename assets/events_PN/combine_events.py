import pandas as pd
import numpy as np



candidate_P = pd.read_csv('./candidate_event_duration=60xseconds_P.csv')[["ManualStartDateTime", "ManualEndDateTime", "Label"]]
candidate_P["ManualStartDateTime"] = pd.to_datetime(candidate_P["ManualStartDateTime"])
candidate_P["ManualEndDateTime"] = pd.to_datetime(candidate_P["ManualEndDateTime"])

candidate_N = pd.read_csv('./candidate_event_duration=60xseconds_N.csv')[["ManualStartDateTime", "ManualEndDateTime", "Label"]]
candidate_N["ManualStartDateTime"] = pd.to_datetime(candidate_N["ManualStartDateTime"])
candidate_N["ManualEndDateTime"] = pd.to_datetime(candidate_N["ManualEndDateTime"])

candidate_D = pd.read_csv('./candidate_event_duration=60xseconds_NA.csv')[["ManualStartDateTime", "ManualEndDateTime", "Label"]]
candidate_D["ManualStartDateTime"] = pd.to_datetime(candidate_D["ManualStartDateTime"])
candidate_D["ManualEndDateTime"] = pd.to_datetime(candidate_D["ManualEndDateTime"])


candidates = pd.concat([candidate_P, candidate_N, candidate_D], ignore_index=True)
candidates.sort_values(by=['ManualStartDateTime'], inplace=True)
candidates.reset_index(drop=True, inplace=True)

time_delta = (candidates["ManualEndDateTime"] - candidates["ManualStartDateTime"])
time_delta = time_delta.dt.total_seconds()
candidates.loc[time_delta < 60, "Label"] = str("NA")
candidates.loc[(candidates["Label"] != "P") & (candidates["Label"] != "N"), "Label"] = str("NA")
candidates.to_csv("./candidates.csv", index=False)


true_event_P = pd.read_csv('./true_event_threshold=2xradius_duration=60xseconds_P.csv')[["ManualStartDateTime", "ManualEndDateTime", "Label"]]
true_event_P["ManualStartDateTime"] = pd.to_datetime(true_event_P["ManualStartDateTime"])
true_event_P["ManualEndDateTime"] = pd.to_datetime(true_event_P["ManualEndDateTime"])

true_event_N = pd.read_csv('./true_event_threshold=2xradius_duration=60xseconds_N.csv')[["ManualStartDateTime", "ManualEndDateTime", "Label"]]
true_event_N["ManualStartDateTime"] = pd.to_datetime(true_event_N["ManualStartDateTime"])
true_event_N["ManualEndDateTime"] = pd.to_datetime(true_event_N["ManualEndDateTime"])

true_event_D = pd.read_csv('./true_event_threshold=2xradius_duration=60xseconds_NA.csv')[["ManualStartDateTime", "ManualEndDateTime", "Label"]]
true_event_D["ManualStartDateTime"] = pd.to_datetime(true_event_D["ManualStartDateTime"])
true_event_D["ManualEndDateTime"] = pd.to_datetime(true_event_D["ManualEndDateTime"])

true_events = pd.concat([true_event_P, true_event_N, true_event_D], ignore_index=True)
true_events.sort_values(by=['ManualStartDateTime'], inplace=True)
true_events.reset_index(drop=True, inplace=True)

time_delta = (true_events["ManualEndDateTime"] - true_events["ManualStartDateTime"])
time_delta = time_delta.dt.total_seconds()

true_events.loc[time_delta < 60, "Label"] = str("NA")
true_events.loc[(true_events["Label"] != "P") & (true_events["Label"] != "N"), "Label"] = str("NA")
true_events.to_csv("./true_events.csv", index=False)
