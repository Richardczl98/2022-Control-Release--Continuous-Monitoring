This module functions as a data processing module for raw data, specifically addressing tasks such as handling missing values, ensuring data continuity, and extracting valid data. It can be further categorized into two types of data processing: one for true release data and another for report event data.

## Preprocessing for True Release Data

To facilitate subsequent analysis, the processing of real release dose data involves the following steps:

1) Missing value imputation: Missing values within each day are filled using linear interpolation, while missing values at the beginning and end of each day are padded with zeros.
2) Continuity: Restructure the raw data by each day beginning from 00:00:00 to 23:59:59 in UTC.
3) Tagging as 'internal testing period': An 'internal testing period' refers to a brief interval between Unofficial Test and Official Test. During this period, releases are identified as "NA" and are not filled in. 
    - Unofficial Test Time: This can be found in the Data from [Daily Testing Cycle File](../../assets/Daily%20Test%20Cycle.xlsx). 
    -  Official Test Time: This can be directly retrieved from the [Daily Testing Cycle File](../../assets/Daily%20Test%20Cycle.xlsx)
These steps ensure the completeness and continuity of the real release dose data for further analysis. 

The raw release dose data is sourced from the [Raw_Data_Per_Day](../../assets/Raw_Data_Per_Day/) directory, and the output is a cleaned true release data file named 'valid_true_data_pad.csv'.

## Preprocessing for Report Event Data

The preprocessing of report event data encompasses the following steps:

1) Extraction of required data: Relevant information, such as event start time, end time, and release dose, is extracted from the data.
2) Extraction of Missing Report Dates: Missing report dates are identified and extracted from the dataset.
3) Extraction of Report Dates: Report dates, including report start time and end time, are extracted.

These steps preprocess the report event data by extracting necessary information and ensuring dataset completeness and accuracy for further analysis. The raw reported event data is sourced from the [sensor_raw_data](../../assets/sensor_raw_data/) directory, and the output is a cleaned report data file named [sensor_data](../../assets/sensor_data/), containing preprocessed data for all sensors.