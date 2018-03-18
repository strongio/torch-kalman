from collections import OrderedDict
import numpy as np
import pandas as pd


class TSPreprocessor(object):
    def __init__(self, variables, group_col, variable_col, value_col, date_col, freq='D'):
        """
        Create a preprocessor for multivariate time-series. This converts pandas dataframes into numpy arrays and
        vice-versa.

        :param variables: A list naming the variables in the multivariate time-series.
        :param group_col: The column in the pandas dataframe containing the group. Each group is a separate multivariate
        time-series
        :param variable_col: The column in the pandas dataframe containing the variable.
        :param value_col: The column in the pandas dataframe containing the actual values of the time-series.
        :param date_col: The column in the pandas dataframe containing the date.
        :param freq: The frequency for the date. Currently only daily data are supported.
        """
        if freq != 'D':
            raise Exception("Only data with freq='D' (daily) currently supported.")
        self.freq = freq
        self.variables = sorted(variables)
        self.group_col = group_col
        self.variable_col = variable_col
        self.value_col = value_col
        self.date_col = date_col

    # Pandas to Numpy ------------------------------------------------------
    def pd_to_array(self, dataframe, min_len_prop):
        """
        Convert a dataframe into an array (that's suitable for passing to KalmanFilters), additionally returning
        information about that array.

        :param dataframe: A pandas dataframe. The dataframe should be in a "tidy" format: a single 'value' column
        contains the actual values of the time-series, a 'variable' column indicates which values belong to which
        variable within a group, and a 'group' column indicates which values belong to which group.
        :param min_len_prop: The longest group in the dataset will dictate the size of the array. If there are groups
        that have very little data and therefore will be mostly missing values in the array, they can be excluded. For
        example, if the longest group is 365, and you want to exclude any groups that are less than half of this length,
        Then set `min_len_prop = .50`.
        :return: A tuple with three elements:
            * A 3D numpy array. The first dimension is the group, the second is the variable, and the third is the
        timepoint. All groups will have their first element corresponding to their first date.
            * Information about the elements of each dimension: an ordered dictionary with (1) the group-column-name :
            the group of each slice along this dimension, (2) the variable-column name : the variable of each slice
            along this dimension, (3) 'timesteps' : a generator for the timesteps.
            * The original start-date for each group.
        """

        # check date-col:
        date_cols = dataframe.select_dtypes(include=[np.datetime64]).columns.tolist()
        if self.date_col not in date_cols:
            raise Exception("The date column (%s) is not of type np.datetime64" % self.date_col)

        # subsequent methods will assume data are sorted:
        dataframe = dataframe.sort_values(by=[self.group_col, self.variable_col, self.date_col])

        # get info per group:
        info_per_group = {g: self.get_group_info(df_g) for g, df_g in dataframe.groupby(self.group_col)}

        # filter based on min_len_prop:
        max_len = max([i['length'] for i in info_per_group.values()])
        min_len = round(max_len * min_len_prop)
        info_per_group = {g: i for g, i in info_per_group.iteritems() if i['length'] >= min_len}

        # create the 'dim_info' dict with information about each dimension:
        dim_info = OrderedDict()
        dim_info[self.group_col] = sorted(info_per_group.keys())  # sorted so we always know order later
        dim_info[self.variable_col] = self.variables  # sorted in __init__ so "
        dim_info['timesteps'] = xrange(max_len)

        # preallocate numpy array:
        x = np.empty(shape=tuple([len(x) for x in dim_info.values()]))
        x[:, :, :] = np.nan

        # fill array:
        start_dates = {}
        for g_idx, group_name in enumerate(dim_info[self.group_col]):
            # for each group...
            start_dates[group_name] = info_per_group[group_name]['start_date']
            for v_idx, var_name in enumerate(self.variables):
                # for each variable...
                this_var_info = info_per_group[group_name]['variable_info'][var_name]
                # whichever variable starts first determines the true start for this group. if any variables started
                # later, their start will be offset accordingly:
                start = this_var_info['offset']
                # after ts ends, no values will be filled, so nans will remain as padding:
                end = len(this_var_info['values'])
                # fill:
                x[g_idx, v_idx, start:end] = this_var_info['values']

        return x, dim_info, start_dates

    def get_group_info(self, dataframe):
        """
        Helper function for `pd_to_array`. Given a dataframe with only a single group, get information about that groups
        start-date, length, and variables.
        :param dataframe: A slice of the original dataframe passed to `pd_to_array` corresponding to one of its groups.
        :return: A dictionary with start-date, length, and variables.
        """
        variable_info = {var: self.get_variable_info(df_gd) for var, df_gd in dataframe.groupby(self.variable_col)}
        if not all([var in self.variables for var in variable_info.keys()]):
            raise Exception("Some variables in this dataframe are not in self.variables.")

        # offset from groups start-date so all variables have sycn'd seasonality,
        # also add nans for missing-variables
        start_date = min([vi['start_date'] for vi in variable_info.values()])
        for var_name in self.variables:
            if var_name in variable_info.keys():
                # TODO: divide by self.freq
                variable_info[var_name]['offset'] = (variable_info[var_name]['start_date'] - start_date).days
            else:
                variable_info[var_name] = {'values': np.array([np.nan]), 'start_date': start_date, 'offset': 0}

        group_info = {'start_date': start_date,
                      'length': max([len(vi['values']) for vi in variable_info.values()]),
                      'variable_info': variable_info}

        return group_info

    def get_variable_info(self, dataframe):
        """
        Helper function for `pd_to_array`. Given a dataframe with only a single variable within a single group, get
        the start-date and actual values.

        :param dataframe: A slice of the original dataframe passed to `pd_to_array` corresponding to one of the
        variables within one of the groups.
        :return: A dctionary with start-date and actual values.
        """
        # TODO: check w/self.freq (where self.freq is in days)
        return ({'start_date': dataframe[self.date_col].min(),
                 'values': dataframe[self.value_col].values})

    # Numpy to Pandas ------------------------------------------------------
    def array_to_pd(self, array, dim_info, start_dates, value_col=None):
        """
        Convert the output of `pd_to_array` (or an array with the same shape as it) into a pandas dataframe. Typically
        this will be used on predictions that were generated from passing that original `pd_to_array` output to a model.

        :param array: The array output from `pd_to_array` (or an array with the same shape as it, e.g., predictions).
        :param dim_info: The dim_info dictionary output from `pd_to_array`.
        :param start_dates: A list of start_dates output from `pd_to_array`.
        :param value_col: What should the column containing the actual values be named in the output pandas dataframe?
        By default this will be the original value-column, name, but you could rename it (e.g., to 'prediction').
        :return: A pandas dataframe. The dataframe will be in a "tidy" format: a single value column contains the
        reshaped contents of `array`, a 'variable' column indicates which values belong to which variable within a
        group, and a 'group' column indicates which values belong to which group.
        """
        if value_col is None:
            value_col = self.value_col

        # make sure correct dtypes so joining to original can work:
        num_rows = reduce(lambda x, y: x * y, [len(val) for val in dim_info.values()])
        group_dtype = np.array(dim_info[self.group_col]).dtype
        var_dtype = np.array(dim_info[self.variable_col]).dtype
        out = {self.group_col: np.empty(shape=(num_rows,), dtype=group_dtype),
               self.variable_col: np.empty(shape=(num_rows,), dtype=var_dtype),
               self.date_col: np.empty(shape=(num_rows,), dtype='datetime64[D]'),
               value_col: np.empty(shape=(num_rows,)) * np.nan}

        row = 0
        for g_idx, group_name in enumerate(dim_info[self.group_col]):
            for v_idx, var_name in enumerate(dim_info[self.variable_col]):
                values = array[g_idx, v_idx, :]
                row2 = row + len(values)
                out[self.group_col][row:row2] = group_name
                out[self.variable_col][row:row2] = var_name
                out[self.date_col][row:row2] = pd.date_range(start_dates[group_name],
                                                             periods=len(values), freq=self.freq)
                out[value_col][row:row2] = values
                row = row2

        return pd.DataFrame(out)
