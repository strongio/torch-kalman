from collections import OrderedDict
import numpy as np
import pandas as pd

from torch_kalman.utils.utils import product


class ForecastPreprocessor(object):
    def __init__(self, group_col, date_col, freq):
        """
        Create a preprocessor for multivariate time-series. This converts pandas dataframes into numpy arrays and
        vice-versa.

        :param group_col: The column in the pandas dataframe containing the group. Each group is a separate multivariate
        time-series
        :param date_col: The column in the pandas dataframe containing the date.
        :param freq: The frequency for the date. Currently pandas.Timedeltas (but not TimeOffsets) are supported. Can pass a
        string that can be interpreted by pandas.to_timedelta.
        """
        if not isinstance(freq, pd.Timedelta):
            freq = pd.to_timedelta(freq)
        self.freq = freq
        self.group_col = group_col
        self.date_col = date_col

    # Pandas to Numpy ------------------------------------------------------
    def pd_to_array(self, dataframe, value_cols, min_len_prop):
        """
        Convert a dataframe into an array (that's suitable for passing to KalmanFilters), additionally returning
        information about that array.

        :param dataframe: A pandas dataframe.
        :param min_len_prop: The longest group in the dataset will dictate the size of the array. If there are groups
        that have very little data and therefore will be mostly missing values in the array, they can be excluded. For
        example, if the longest group is 365, and you want to exclude any groups that are less than half of this length,
        Then set `min_len_prop = .50`.
        :return: A tuple with three elements:
            * A 3D numpy array. The first dimension is the group, the second is the timepoint, and the third is the
            measure. All groups will have their first element corresponding to their first date. If one or more of the
            measures for a group starts later than the first measure in that group, they will be nan-padded.
            * Information about the elements of each dimension: an ordered dictionary with (1) the group-column-name :
            the group of each slice along this dimension, (2) 'timesteps' : a generator for the timesteps, and (2) the
            measure-column name : the measure of each slice along this dimension.
            * The original start-date for each group.
        """

        # check date-col:
        date_cols = dataframe.select_dtypes(include=[np.datetime64]).columns.tolist()
        if self.date_col not in date_cols:
            raise Exception("The date column ('{}') is not of type np.datetime64".format(self.date_col))

        # subsequent methods will assume data are sorted:
        dataframe = dataframe.sort_values(by=[self.group_col, self.date_col])

        # get info per group:
        info_per_group = {g: self.get_group_info(df_g, value_cols) for g, df_g in dataframe.groupby(self.group_col)}

        # filter based on min_len_prop:
        longest_group_length = max(info['length'] for info in info_per_group.values())
        group_length_min = round(longest_group_length * min_len_prop)
        info_per_group = {g: info for g, info in info_per_group.items() if info['length'] >= group_length_min}

        # create the 'dim_info' dict with information about each dimension:
        dim_info = OrderedDict()
        dim_info[self.group_col] = sorted(info_per_group.keys())  # sorted so we always know order later
        dim_info['timestep'] = [self.freq * i for i in range(longest_group_length + 1)]
        dim_info['variable'] = value_cols

        # preallocate numpy array:
        x = np.empty(shape=[len(x) for x in dim_info.values()])
        x[:, :, :] = np.nan

        # fill array:
        start_dates = {}
        for g_idx, group in enumerate(dim_info[self.group_col]):
            # for each group...
            start_dates[group] = info_per_group[group]['start_date']
            for v_idx, variable in enumerate(value_cols):
                x[g_idx, info_per_group[group]['idx'], v_idx] = info_per_group[group]['values'][variable]

        return x, dim_info, start_dates

    def timedelta_int(self, t1, t2):
        """
        Subtract t1 from t2, to get difference in integers where the units are self.freq.
        :param t1: A datetime (or DateTimeIndex)
        :param t2: A datetime (or DateTimeIndex)
        :return: An integer.
        """
        diff = (t1 - t2) / self.freq
        if isinstance(diff, float):
            out = int(diff)
        else:
            out = diff.astype(int)
        if not np.isclose(diff, out).all():
            raise ValueError("Timedelta did not divide evenly into self.freq.")
        return out

    def get_group_info(self, dataframe, value_cols):
        """
        Helper function for `pd_to_array`. Given a dataframe with only a single group, get information about that groups
        start-date, length, and measures.
        :param dataframe: A slice of the original dataframe passed to `pd_to_array` corresponding to one of its groups.
        :return: A dictionary with start-date, length, and measures.
        """
        group_info = dict()
        group_info['start_date'] = dataframe[self.date_col].values.min()
        group_info['end_date'] = dataframe[self.date_col].values.max()
        group_info['length'] = self.timedelta_int(group_info['end_date'], group_info['start_date'])
        group_info['idx'] = self.timedelta_int(dataframe[self.date_col].values, group_info['start_date'])
        group_info['values'] = dict()
        for value_col in value_cols:
            group_info['values'][value_col] = dataframe[value_col].values

        return group_info

    # Numpy to Pandas ------------------------------------------------------
    def array_to_pd(self, array, dim_info, start_dates):
        """
        Convert the output of `pd_to_array` (or an array with the same shape as it) into a pandas dataframe. Typically
        this will be used on predictions that were generated from passing that original `pd_to_array` output to a model.

        :param array: The array output from `pd_to_array` (or an array with the same shape as it, e.g., predictions).
        :param dim_info: The dim_info dictionary output from `pd_to_array`.
        :param start_dates: A list of start_dates output from `pd_to_array`.
        :return: A pandas dataframe. The dataframe will be in a "long" format: a single value column contains the
        reshaped contents of `array`, a 'measure' column indicates which values belong to which measure within a
        group, and a 'group' column indicates which values belong to which group.
        """

        # make sure correct dtypes so joining to original can work:
        num_rows = len(dim_info['timestep']) * len(dim_info[self.group_col])
        group_dtype = np.array(dim_info[self.group_col]).dtype
        out = {self.group_col: np.empty(shape=(num_rows,), dtype=group_dtype),
               self.date_col: np.empty(shape=(num_rows,), dtype='datetime64[ns]')}
        for variable_col in dim_info['variable']:
            out[variable_col] = np.empty(shape=(num_rows,)) * np.nan

        row = 0
        for g_idx, group_name in enumerate(dim_info[self.group_col]):
            row2 = row + len(dim_info['timestep'])
            for v_idx, variable_col in enumerate(dim_info['variable']):
                values = array[g_idx, :, v_idx]
                out[variable_col][row:row2] = values
                out[self.group_col][row:row2] = group_name
                out[self.date_col][row:row2] = pd.date_range(start_dates[group_name],
                                                             periods=len(values), freq=self.freq)
                row = row2

        return pd.DataFrame(out)
