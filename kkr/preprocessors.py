#     Python Implementation of the Algorithms presented in the publications
#     around the kernel Kalman rule.
#     Copyright (C) 2017  Gregor Gebhardt
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np

import scipy.spatial as spatial
import scipy.stats as stats


def _generate_data_windows_single_series(series: pd.Series or np.ndarray, window_size):
    df = pd.DataFrame(series)

    if isinstance(series.index, pd.MultiIndex):
        def get_window_item(s, i):
            return s.unstack(0).rename_axis(lambda x: x + i).stack().swaplevel().sort_index()
    else:
        def get_window_item(s, i):
            return s.rename_axis(lambda x: x + i)

    for i in range(1, window_size):
        # create a copy with renamed axis
        df_i = get_window_item(series, i)

        # concat copy to df
        df = pd.concat([df, df_i], axis=1, ignore_index=True)

    return df


def generate_data_windows(data: pd.DataFrame or pd.Series, window_size, drop_nan=True):
    if isinstance(data, pd.Series):
        windows = _generate_data_windows_single_series(data, window_size)
        # create new names with data.name
        if isinstance(data.name, str):
            # if the series has a name, we create a MultiIndex with the name on the upper layer and the index on the
            # lower
            new_columns = [(data.name, i) for i in range(window_size)]
            windows.columns = pd.MultiIndex.from_tuples(new_columns, names=[data.name, 'win'])
        else:
            # otherwise we just create a regular index for the columns
            new_columns = list(range(window_size))
            windows.columns = pd.Int64Index(new_columns)
    elif isinstance(data, pd.DataFrame):
        windows = pd.DataFrame()

        # handle each column individually
        for column in data.columns:
            windows_col = _generate_data_windows_single_series(data[column], window_size)
            # create new column names
            if isinstance(column, tuple):
                # if the columns come from a MultiIndex, we get tuples and extend them by the index
                new_columns = [column + (i,) for i in range(window_size)]
            else:
                # if the columns come from a regular index, we create a MultiIndex with the index on the lower layer
                new_columns = [(column, i) for i in range(window_size)]
            windows_col.columns = pd.MultiIndex.from_tuples(new_columns, names=data.columns.names + ['win'])

            # concatenate the column windows to the return DataFrame
            windows = pd.concat([windows, windows_col], axis=1)
    elif isinstance(data, np.ndarray):
        # TODO implement
        raise NotImplementedError("function generate_data_windows not yet implemented for type {}".format(type(data)))
    else:
        raise NotImplementedError("function generate_data_windows not implemented for type {}".format(type(data)))

    if drop_nan:
        return windows.dropna()
    else:
        return windows


def compute_median_bandwidth(data, quantile=.5, sample_size=1000):
    """Computes a bandwidth for the given data set using the median heuristic.
    Other quantiles can be chosen with the quantile keyword argument.

    Arguments:
    data -- a DataFrame with the variables in the columns
    quantile -- scalar or list of scalars from the range (0,1)
    sample_size -- maximum number of sample to compute the point-wise distances

    Returns:
    bandwidths -- an array with the bandwidth for each variable
    """
    if len(data.shape) > 1:
        num_variables = data.shape[1]
    else:
        num_variables = 1

    bandwidths = np.zeros(num_variables)

    num_data_points = data.shape[0]

    if sample_size > num_data_points:
        data_points = data.values
    else:
        data_points = data.sample(sample_size).values

    for i in range(num_variables):
        distances = spatial.distance.pdist(data_points[:, i:i+1])
        if quantile == .5:
            bandwidths[i] = np.median(distances)
        else:
            bandwidths[i] = pd.DataFrame(distances).quantile(quantile)

    return bandwidths


def add_noise(data, column, noise_generator=stats.norm(loc=.0, scale=.1)):
    # TODO add documentation
    """

    :param data:
    :param column:
    :param noise_generator:
    """

    # generate noise
    noise = noise_generator.rvs(size=data[column].shape)

    # add noise to data
    noisy_column = column + "Noisy"
    data[noisy_column] = data[column] + noise


def select_reference_set_randomly(data, size, consecutive_sets=1, group_by=None):
    """selects a random reference set from the given DataFrame. Consecutive sets are computed from the first random
    reference set, where it is assured that only data points are chosen for the random set that have the required
    number of successive data points. Using the group_by argument allows to ensure that all consecutive samples are
    from the same group.

    :param data: a pandas.DataFrame with the samples to choose from
    :param size: the number of samples in the reference set
    :param consecutive_sets: the number of consecutive sets returned by this function (default: 1)
    :param group_by: a group_by argument to ensure that the consecutive samples are from the same group as the first
    random sample
    :return: a tuple with the reference sets
    """
    weights = np.ones(data.shape[0])

    if group_by is not None:
        gb = data.groupby(level=group_by)
        last_windows_idx = [ix[-i] for _, ix in gb.indices.items() for i in range(1, consecutive_sets)]
        weights[last_windows_idx] = 0
    else:
        last_windows_idx = [data.index[-i] for i in range(1, consecutive_sets+1)]
        weights[last_windows_idx] = 0

    # select reference set
    if weights.sum() <= size:
        # if there is not enough data, we take all data points
        reference_set1 = data.loc[weights == 1].index.sort_values()
    else:
        # otherwise we chose a random reference set from the data
        reference_set1 = data.sample(n=size, weights=weights).index.sort_values()

    if consecutive_sets > 1:
        reference_set = [reference_set1]
        for i in range(1, consecutive_sets):
            if type(reference_set1) is pd.MultiIndex:
                reference_set_i = pd.MultiIndex.from_tuples([*map(lambda t: (*t[:-1], t[-1] + i),
                                                                  reference_set1.values)])
                reference_set_i.set_names(reference_set1.names, inplace=True)
                reference_set.append(reference_set_i)
            else:
                reference_set_i = pd.Index(data=reference_set1 + i, name=reference_set1.name)
                reference_set.append(reference_set_i)

    else:
        reference_set = reference_set1

    return tuple(reference_set)


def select_reference_set_by_kernel_activation(data: pd.DataFrame, size: int, kernel_function,
                                              consecutive_sets: int = 1, group_by: str = None) -> tuple:

    """
    Iteratively selects a subset from the given data by applying a heuristic that is based on the kernel activations of
    the data with the already selected data points. The returned If the consecutive_sets parameter is greater than 1,
    multiple

    :param data: a pandas.DataFrame with the data from which the subset should be selected
    :param size: the size of the subset (if data has less data points, all data points are selected into the subset.)
    :param kernel_function: the kernel function for computing the kernel activations
    :param consecutive_sets: number of consecutive sets, i.e. subsets that contain the consecutive points of the
    previous subset
    :param group_by: used to group the data before selecting the subsets to ensure that the points in the consecutive
    sets belong to the same group
    :return: a tuple of
    """
    logical_idx = np.ones(data.shape[0], dtype=bool)
    if group_by is not None:
        gb = data.groupby(level=group_by)
        last_windows_idx = [gb.indices[e][-i] for e in gb.indices for i in range(1, consecutive_sets+1)]
    else:
        last_windows_idx = [data.index[-i] for i in range(1, consecutive_sets+1)]

    logical_idx[last_windows_idx] = False

    # get data points that can be used for first data sets
    reference_data = data.iloc[logical_idx]
    num_reference_data_points = reference_data.shape[0]

    # if we have not enough data to select a reference set, we take all data points
    if num_reference_data_points <= size:
        reference_set1 = reference_data.index.sort_values()
    else:
        reference_set1 = [reference_data.sample(1).index[0]]

        kernel_matrix = np.zeros((size+1, num_reference_data_points))

        for i in range(size-1):
            # compute kernel activations for last chosen kernel sample
            kernel_matrix[i, :] = kernel_function(reference_data.loc[reference_set1[-1]].values,
                                                  reference_data)
            kernel_matrix[-1, reference_data.index.get_locs(reference_set1[-1])] = 1000

            max_kernel_activations = kernel_matrix.max(0)
            next_reference_point = np.argmin(max_kernel_activations)

            reference_set1.append(reference_data.index.values[next_reference_point])

        if type(reference_set1[0]) is tuple:
            reference_set1 = pd.MultiIndex.from_tuples(reference_set1).sort_values()
            reference_set1.set_names(reference_data.index.names, inplace=True)
        else:
            reference_set1 = pd.Index(data=reference_set1, name=reference_data.index.names)

    if consecutive_sets > 1:
        reference_set = [reference_set1]
        for i in range(1, consecutive_sets):
            if type(reference_set1) is pd.MultiIndex:
                reference_set_i = pd.MultiIndex.from_tuples([*map(lambda t: (*t[:-1], t[-1] + i),
                                                                  reference_set1.values)])
                reference_set_i.set_names(reference_set1.names, inplace=True)
                reference_set.append(reference_set_i)
            else:
                reference_set_i = pd.Index(data=reference_set1 + i, name=reference_set1.name)
                reference_set.append(reference_set_i)

        return tuple(reference_set)
    else:
        return reference_set1


def cut_at_impact(data: pd.DataFrame):
    df = pd.DataFrame()

    for i, group in data.groupby(level=0):
        impacts, = np.where(np.diff(np.sign(np.diff(group[('observation', 'z')].values, axis=0)), axis=0) == 2)

        if len(impacts) >= 3:
            final_impact = impacts[2] + 2
        else:
            final_impact = impacts[-1] + 2

        cutted_group = group.iloc[0:final_impact]
        df = pd.concat([df, cutted_group])

    # df.sort_index(level=0, inplace=True)
    return df
