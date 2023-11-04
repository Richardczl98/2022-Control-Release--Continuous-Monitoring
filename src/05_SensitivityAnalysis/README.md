The Sensitivity Analysis module focuses on analyzing the results of the [03_DetectionAnalysis](../03_DetectionAnalysis/) module and consists of two main components:

## Event-based Confusion Matrix

The Event-based Confusion Matrix component analyzes the percentage variations of the True Positive Rate (TPR) (%) 、 precision、True Negative Rate (TNR) and Negative Predictive Value (NPV) for both Stanford Defined Events and Team Defined Events across different input parameters.

### Inputs

The module requires the following input variables:

- **Threshold**: The distance parameter used by the wind transpose model (1/2/4 times the experimental area radius).
- **Ignore Duration**: The minimum event length in Stanford Defined Events that should be ignored (30/60/120 seconds).
- **Short Stack**: Determines whether to analyze results only for the short stack scenario (0 for all experimental dates, 1 for the short stack scenario).

### Data

The output results from `event_based_confusion_matrix_stanford_defined_event.ipynb` and `event_based_confusion_matrix_team_defined_event.ipynb` are used, focusing on  TPR、precision、 TNR and NPV for different input parameters.

### Outputs

The percentage differences in TPR、precision、 TNR and NPV for each sensor under different input scenarios are saved in the [Event-based Difference](../../results/05_SensitivityAnalysis/Event-based%20Difference/) directory.

## Time-based Confusion Matrix

The Time-based Confusion Matrix component analyzes the variations in metrics of the Time-based Confusion Matrix under different input scenarios.

### Inputs

The module requires the following input variables:

- **Threshold**: The distance parameter used by the wind transpose model (1/2/4 times the experimental area radius).
- **Ignore Duration**: The minimum event length in Stanford Defined Events that should be ignored (30/60/120 seconds).
- **Short Stack**: Determines whether to analyze results only for the short stack scenario (0 for all experimental dates, 1 for the short stack scenario).
- **Overlap**: Determines whether to consider only the time when events reported by all sensors overlap (0 for no, 1 for yes).

### Data

The output results from `time_based_confusion_matrix.ipynb` are used, focusing on the metrics of the Confusion Matrix under different input parameters.

### Outputs

The differences in Confusion Matrix metrics under different input parameters are saved in the [Time-based Difference](../../results/05_SensitivityAnalysis/Time-based%20Difference/) directory.