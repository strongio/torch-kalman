# Kalman-Filters in Pytorch

A bare-bones implementation of kalman-filters for multivariate time-series in PyTorch.

## Big Picture:

Classes that extend `pytorch.nn`. These handle time-series that are multivariate. KalmanFilters take a 3D tensor: group, variable, timepoint. Parameters (e.g., process-noise, measurement-noise) are optimized across groups.

The approach here...
- ...naturally handles missing data.
- ...should especially shine when we have multiple groups (i.e., multiple time-series), and we want to estimate parameters globally (rather than from each group individually). This is a feature [surprisingly lacking from most forecasting tools](https://stats.stackexchange.com/questions/23036/estimating-same-model-over-multiple-time-series)).
- ...should especially shine for multivariate time-series (also a surprisingly lacking feature in other tools).

Combining these three, the tools here could be useful in contexts not usually ascribed to time-series/forecasting, such as customer lifecycle modeling -- i.e., modeling/forecasting KPIs for individual customers. While each individual customer might have somewhat sparse data (a) we can estimate global parameters by combining this data across many customers, (b) we can model the correlation across multiple KPIs and so improve estimates.
