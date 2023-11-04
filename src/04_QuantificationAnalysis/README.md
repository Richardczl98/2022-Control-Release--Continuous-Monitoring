The Quantification Analysis module focuses on quantifying the reported events based on the true events obtained from [02_FilterEvent](../02_FilterEvent/README.md). It comprises two main directions:

## Daily-based Linear Regression
This direction primarily focuses on the quantitative analysis of daily emission and reported emission quantities. In other words, for each day, we calculate the difference between the average actual methane emission quantity and the average reported emission quantity. Additionally, we also calculate their uncertainties.

#### Data
The input data for this analysis includes the valid true release data and the reported event data for each sensor. The Missing Data Reports and offline dates for each sensor are taken into account during computation. Furthermore, we also utilized indicators such as 'meter used for release,' 'sigma of meter,' and 'Stanford data process QC flag' to calculate the uncertainty of daily methane release. Additionally, we used the sensor's 'Emission Rate Upper,' 'Emission Rate Lower,' and 'Uncertainty Type' to calculate the uncertainty of reported release for each day. 

For Canary, we only consider short stack scenario.


### Ouputs
The outputs of this analysis are multiple linear regression plots that illustrate the statistical analysis of the average release dose and average reported dose for every valid day. These plots provide insights into the relationship between the two variables and offer valuable information about their correlation, Moreover, these plots also provide error bars based on sensor and real data, which help us better analyze the differences between them. 
These plots can be divided into two types: Zoom-in and Uniform. The former shows magnified versions of current regression plots for each sensor, allowing us to examine the details. The latter displays linear regression plots with a uniform scale for all sensors. The plots are saved in the [Linear Regression Stanford Defined Events Uniform Size](../../results/04_QuantificationAnalysis/Daily-based/Uniform%20Size/) and [Linear Regression Stanford Defined Zoom-in Size](../../results/04_QuantificationAnalysis/Daily-based/Zoom-In%20Size/).


## Event-based Linear Regression
This direction primarily focuses on event-based quantitative analysis, which means that we analyze the discrepancy between true methane release rate and reported emission rate for an event (both true and reported events).


### Linear Regression Based on Team Defined Events

This analysis measures the difference between the release dosage of the team-defined events from each sensor and the average release dosage of the Stanford defined events.

#### Inputs

- **Short Stack**: Indicates whether to analyze results only for the short stack (0 for analyzing all experimental dates, 1 for analyzing only the short stack scenario).

#### Data

The input data for this analysis includes the valid true release data and the reported event data for each sensor. The Missing Data Reports and offline dates for each sensor are taken into account during computation.

#### Outputs

The outputs of this analysis are multiple linear regression plots that illustrate the statistical analysis of the average release dose and average reported dose. These plots show the relationship between the two variables and provide insights into their correlation. These plots can be divided into two types: Zoom-in and Uniform. The former shows magnified versions of current regression plots for each sensor, allowing us to examine the details. The latter displays linear regression plots with a uniform scale for all sensors. The plots are saved in the [Linear Regression Team Defined Events Uniform Size](../../results/04_QuantificationAnalysis/Event-based/Uniform%20Size/LinearRegression%20Stanford%20Defined%20Events/) and [Linear Regression Team Defined Events Zoom-in Size](../../results/04_QuantificationAnalysis/Event-based/Zoom-In%20Size/LinearRegression%20Stanford%20Defined%20Events/).

### Linear Regression on Stanford Defined Events

This analysis examines the difference between the reported release dosage and the average release dosage of the Stanford Defined Events when they occur.

#### Inputs

- **Threshold**: Represents the distance parameter used by the wind transpose model, with values of 1/2/3/4 (indicating 1/2/3/4 times the experimental area radius).
- **Ignore Duration**: Denotes the minimum event length in Stanford Defined Events that should be ignored, with values of 30/60/120 (indicating events shorter than 30/60/120 seconds should be ignored).
- **Short Stack**: Indicates whether to analyze results only for the short stack (0 for analyzing all experimental dates, 1 for analyzing only the short stack scenario).

#### Data

The data for this analysis includes the Stanford defined events calculated after the wind transpose model, the candidate events before the wind transpose model, and the team defined events for each sensor. The Missing Data Reports and offline dates for each sensor are ignored during computation.

#### Outputs

The outputs of this analysis are multiple linear regression plots that illustrate the statistical analysis of the average release dose and average reported dose. These plots provide insights into the relationship between the two variables and offer valuable information about their correlation. These plots can be divided into two types: Zoom-in and Uniform. The former shows magnified versions of current regression plots for each sensor, allowing us to examine the details. The latter displays linear regression plots with a uniform scale for all sensors. The plots are saved in the [Linear Regression Stanford Defined Events Uniform Size](../../results/04_QuantificationAnalysis/Event-based/Uniform%20Size/LinearRegression%20Team%20Defined/) and [Linear Regression Stanford Defined Zoom-in Size](../../results/04_QuantificationAnalysis/Event-based/Zoom-In%20Size/LinearRegression%20Team%20Defined/).