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

import time

import numpy as np
import pandas as pd

from .kernels import ExponentialQuadraticKernel
from .filter import KernelKalmanFilter
from .smoother import KernelForwardBackwardSmoother

from typing import Dict


def parameter_transform(transform):
    """Decorates a function with a transformation of the passed keyword arguments. If a dictionary is passed to the
    function, each keyword argument is transformed with the transformation function found under the same keyword in
    the transform dictionary. Otherwise, it is assumed the the passed object is a function and all keyword arguments
    are transformed with that function.

    :param transform: dict or function for transforming all keyword arguments passed to the decorated function
    :return: the decorated function
    """

    def _parameter_transform(func):
        def _transformed_parameters_func(*args, **kwargs):
            if isinstance(transform, dict):
                transformed_kwargs = {k: transform[k](kwargs[k]) for k in kwargs
                                      if k in transform and transform[k] is not None}
                kwargs.update(transformed_kwargs)
            elif transform is not None:
                kwargs = {k: transform(kwargs[k]) for k in kwargs}

            return func(*args, **kwargs)

        return _transformed_parameters_func

    return _parameter_transform


def parameter_naming(names):
    """ Decorates a function by naming parameters passed as an array as the first positional argument of the function
    with the given list of names. The original function is then called with the named parameters as additional
    keyword arguments. The original keyword arguments are passed through, additional positional arguments are not
    supported.

    :param names: list of names as strings. If the list of names is shorter or longer than the list of parameters,
    a Warning is raised.
    :return: the decorated function
    """

    def _parameter_naming(func):
        def _named_parameters_func(parameters=None, **kwargs):
            if parameters is not None:
                if len(parameters) != len(names):
                    raise Warning("Number of parameters is not equal to number of parameter names.")

                named_parameters = dict(zip(names, parameters))
                kwargs.update(named_parameters)

            return func(**kwargs)

        return _named_parameters_func

    return _parameter_naming


def bandwidth_factor(**bandwidths):
    """ Decorates a function with an argument replacement intended for scaling bandwidths. The bandwidths to scale
    are expected as keyword arguments to the decorator. The decorated function takes the argument with the same
    keyword and scales the bandwidth with that. The scaled bandwidths are then passed to the original function.

    :return: the decorated function
    """

    def _bandwidth_factor(func):
        def _bandwidth_factor_func(*args, **kwargs):
            new_bandwidths = {k: kwargs[k] * bandwidths[k] for k in bandwidths}
            kwargs.update(new_bandwidths)

            return func(*args, **kwargs)

        return _bandwidth_factor_func

    return _bandwidth_factor


def window_bandwidth_factor(**bandwidths: Dict[str, np.ndarray]):
    """Decorates a function with an argument replacement intended for scaling the bandwidths of the elements of  data
    windows. The decorator accepts an arbitrary number of keyword arguments, where each kw argument contains the base
    bandwidths for one bandwidth array that should be created from the decoration. The base bandwidths are expected
    as a nxm matrix where n is the dimensionality of the data (without windowing) and m is the size of the data
    windows, i.e., one row for each window and one column for each element in the data windows

    :param bandwidths

    """
    def _window_bandwidth_factor(func):
        def _window_bandwidth_factor_func(*args, **kwargs):
            new_bandwidths = dict()
            for bw_key, bw_array in bandwidths.items():
                # get scaling vector from kwargs and broadcast it to the bandwidth array for the keyword
                scaled_bandwidths = bw_array * kwargs[bw_key]
                # unfold the scaled bandwidth matrix to a vector and store it in the dict
                new_bandwidths[bw_key] = scaled_bandwidths.flatten()

            kwargs.update(new_bandwidths)
            return func(*args, **kwargs)

        return _window_bandwidth_factor_func

    return _window_bandwidth_factor


def dimension_bandwidth_factor(**bandwidths: Dict[str, np.ndarray]):
    """Decorates a function with an argument replacement intended for scaling the bandwidths for each dimension
    individually when using data windows. I.e., the windows in each dimension get a single bandwidth factor.

    :param bandwidths

    """
    def _dimension_bandwidth_factor(func):
        def _dimension_bandwidth_factor_func(*args, **kwargs):
            new_bandwidths = dict()

            for bw_key, bw_array in bandwidths.items():
                # get scaling vector from kwargs and broadcast it ot the bandwidth array for the keyword
                # TODO switch broadcast directions such that each dimension gets a single factor...
                bw_factor = np.array(kwargs[bw_key])
                if bw_factor.ndim == 1:
                    bw_factor = bw_factor.reshape((-1, 1))
                scaled_bandwidths = bw_array * bw_factor
                # unfold the scaled bandwidth matrix to a vector and store it in the new_bandwidths dict
                new_bandwidths[bw_key] = scaled_bandwidths.flatten()

            kwargs.update(new_bandwidths)
            return func(*args, **kwargs)

        return _dimension_bandwidth_factor_func

    return _dimension_bandwidth_factor


def parameter_arrays(**prefixes):
    """ Decorates a function with an argument replacement intended for grouping parameters into lists. The keyword
    arguments to the decorator define prefixes for which the incoming arguments are filtered and grouped into a new
    list which is passed to the decorated function under the same keyword.

    :param prefixes: keyword arguments with the structure group_name=prefix, all incoming arguments whose name starts
    with prefix are put into a list which is passed as group_name to the original function.
    :return: the decorated function
    """

    def _parameter_arrays(func):
        def _parameter_arrays_func(*args, **kwargs):
            # create new parameter list
            param_groups = {group_name: np.array([value for name, value in kwargs.items() if name.startswith(prefix)])
                            for group_name, prefix in prefixes.items()}
            old_names = [name for name in kwargs for prefix in prefixes.values() if name.startswith(prefix)]
            for name in old_names:
                del kwargs[name]

            return func(*args, **param_groups, **kwargs)

        return _parameter_arrays_func

    return _parameter_arrays


def exception_catcher(exception_type, exception_score):
    """Decorates a function with a try-except-statement for a given type of exception. If the exception is caught,
    the decorated function returns the exception_score otherwise the original function's return values are returned.

    :param exception_type: the type of the exception to be caught
    :param exception_score: the return value for the case that an exception is caught.
    :return: the decorated function
    """

    def _exception_catcher(func):
        def _catched_exception_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type:
                return exception_score

        return _catched_exception_func

    return _exception_catcher


def mean_squared_error(groundtruth, mu, _):
    return ((groundtruth - mu) ** 2).mean()


def normalized_mean_square_error(groundtruth, mu, _):
    return np.abs(((groundtruth - mu) ** 2 / (mu.mean() * groundtruth.mean())).mean())


def neg_log_likelihood(groundtruth, mu, sigma):
    # TODO implement for multivariate data
    sigma = np.diagonal(sigma, axis1=1, axis2=2)
    if len(mu.shape) == 3:
        sigma = np.tile(np.expand_dims(sigma, 2), (1, 1, mu.shape[2]))
    return ((groundtruth - mu) ** 2 / sigma + np.log(2 * np.pi * sigma)).sum()


class FilterEvaluation(object):
    def __init__(self):
        super().__init__()

        self.__test_observations: np.ndarray or pd.DataFrame = None
        self.__test_groundtruth: np.ndarray or pd.DataFrame = None

        self.model_class = KernelKalmanFilter
        self.kernel_k_class = ExponentialQuadraticKernel
        self.kernel_g_class = ExponentialQuadraticKernel

        self.model: KernelKalmanFilter = None
        self._is_setup = False

    def setup_evaluation(self, train_data, test_observations, test_groundtruth):
        self.model = self.model_class(**train_data)

        self.model.kernel_k = self.kernel_k_class()
        self.model.kernel_g = self.kernel_g_class()

        self.__test_observations = test_observations
        self.__test_groundtruth = test_groundtruth

        self._is_setup = True

    def evaluate(self, eval_function=mean_squared_error, **kw_args) -> float:
        assert self._is_setup

        self.model.learn_model(**kw_args)
        mu, sigma = self._evaluate_model(self.__test_observations)

        return eval_function(self.__test_groundtruth, mu, sigma)

    def evaluate_loglikelihood(self, **kwargs) -> float:
        return self.evaluate(eval_function=neg_log_likelihood, **kwargs)

    def evaluate_nmse(self, **kwargs):
        return self.evaluate(eval_function=normalized_mean_square_error, **kwargs)

    def evaluate_groupby(self, level=0, eval_function=mean_squared_error, precompute_observations=None, **kwargs):
        assert self._is_setup

        self.model.learn_model(**kwargs)

        if precompute_observations is not None:
            self.model.precompute_Q_and_S(precompute_observations)

        score = 0

        gb_obs = self.__test_observations.groupby(level=level)
        gb_gth = self.__test_groundtruth.groupby(level=level)
        for (_, observations), (_, groundtruth) in zip(gb_obs, gb_gth):
            mu, sigma = self._evaluate_model(observations.values)
            score += eval_function(groundtruth.values, mu, sigma)

        score /= gb_gth.ngroups
        return score

    def evaluate_groupby_loglikelihood(self, **kwargs) -> float:
        return self.evaluate_groupby(eval_function=neg_log_likelihood, **kwargs)

    def evaluate_groupby_nmse(self, **kwargs) -> float:
        return self.evaluate_groupby(eval_function=normalized_mean_square_error, **kwargs)

    def _evaluate_model(self, *args, **kwargs):
        return self.model.filter(*args, **kwargs)


class SmootherEvaluation(FilterEvaluation):
    def __init__(self):
        super().__init__()

        self.model: KernelForwardBackwardSmoother = None
        self.model_class = KernelForwardBackwardSmoother

    def _evaluate_model(self, *args, **kwargs):
        return self.model.smooth(*args, **kwargs)


class BayesianUpdateEvaluation(object):
    def __init__(self):
        super().__init__()

        self.kernel_k = ExponentialQuadraticKernel()
        self.kernel_k.normalized = True
        self.kernel_g = ExponentialQuadraticKernel()
        self.kernel_g.normalized = True
        self._model = "kkr"
        self._data = None
        self._reference_set = None

        self._kernel_size = None
        self._data_size = None

        self._test_data = None

    @property
    def test_data(self):
        return self._test_data

    @test_data.setter
    def test_data(self, df):
        self._test_data = df

    def setup_evaluation(self, model, data, reference_set, test_data):
        self._test_data = test_data

        self._data = data
        self._reference_set = reference_set

        self._kernel_size = len(self._reference_set)
        self._data_size = len(self._data)

        # save model data
        self._model = model

    def evaluate(self, bandwidth_k, bandwidth_g, alpha_1, alpha_2, with_time=False):
        self.kernel_k.bandwidth = bandwidth_k
        self.kernel_g.bandwidth = bandwidth_g

        if self._model.startswith('kbr'):

            class _Model:
                def __init__(s):
                    # model matrices
                    s._K = self.kernel_k(self._reference_set[['context']].values)
                    s._G = self.kernel_g(self._reference_set[['samples']].values)
                    s._X = self._reference_set[['context']].values

                    # initialize embedding
                    s._C = np.linalg.solve(s._K + alpha_1 * np.eye(self._kernel_size), s._K)
                    s._K_all = self.kernel_k(self._reference_set[['context']].values,
                                             self._data[['context']].values)
                    s._C_0 = np.linalg.solve(s._K + alpha_1 * np.eye(s._K.shape[0]), s._K_all)
                    s.m_0 = np.mean(s._C_0, axis=1, keepdims=True)

                    s._L = np.empty((self._kernel_size, self._kernel_size), dtype=np.double)
                    s._D = np.empty((self._kernel_size, self._kernel_size), dtype=np.double)
                    s._DG = np.empty((self._kernel_size, self._kernel_size), dtype=np.double)
                    s._GDg = np.empty(self._kernel_size, dtype=np.double)

                @staticmethod
                def k_g(y):
                    return self.kernel_g(self._reference_set[['samples']].values, y)

                def transform(s, _m):
                    return s._X.T.dot(s._C.dot(_m))

            if self._model == 'kbr_a':
                class Model(_Model):
                    def update(s, _m, _, _g_y):
                        s._L = s._C.dot(np.diag(_m.flat))
                        s._D = np.diag(s._C.dot(_m).flat)
                        s._DG = s._D.dot(s._G)

                        s._GDg = s._G.dot(s._D).dot(_g_y)
                        _m = s._L.T.dot(np.linalg.solve(s._DG.dot(s._DG) + alpha_2 * np.eye(self._kernel_size), s._GDg))

                        _m = _m / _m.sum(axis=0)

                        return _m, None

            if self._model == 'kbr_b':
                class Model(_Model):
                    def update(s, _m, _, _g_y):
                        s._D = np.diag(s._C.dot(_m).flat)
                        s._DG = s._D.dot(s._G)
                        _m = s._DG.dot(np.linalg.solve(s._DG.dot(s._DG) + alpha_2 * np.eye(self._kernel_size), s._D)).dot(_g_y)

                        _m = _m / _m.sum(axis=0)

                        return _m, None

            if self._model == 'kbr_c':
                class Model(_Model):
                    def update(s, _m, _, _g_y):
                        s._D = np.diag(s._C.dot(_m).flat)
                        s._DG = s._D.dot(s._G)
                        _m = np.linalg.solve(s._DG + alpha_2 * np.eye(self._kernel_size), s._D).dot(_g_y)

                        _m = _m / _m.sum(axis=0)

                        return _m, None

        if self._model == 'sub_kbr':
            class Model:
                def __init__(s):
                    s._K_r = self.kernel_k(self._data[['context']].values,
                                           self._reference_set[['context']].values)
                    s._G = self.kernel_g(self._data[['samples']].values)
                    s._X = self._data[['context']].values

                    s.m_0 = np.ones(self._data_size) / self._data_size

                    s._C = np.linalg.solve(s._K_r.T.dot(s._K_r) + alpha_1 * np.eye(self._kernel_size), s._K_r.T)
                    s._E = s._K_r.T.dot(s._G).dot(s._K_r)
                    # s._E2 = s._G.dot(s._K_r)

                    s._L = np.empty((self._data_size, self._kernel_size), dtype=np.double)
                    s._D = np.empty((self._kernel_size, self._kernel_size), dtype=np.double)
                    s._DE = np.empty((self._kernel_size, self._kernel_size), dtype=np.double)
                    s._DKg = np.empty(self._kernel_size, dtype=np.double)

                @staticmethod
                def k_g(y):
                    return self.kernel_g(self._data[['samples']].values, y)

                def update(s, _m, _, _g_y):
                    s._L = np.diag(_m.flat).dot(s._C.T)
                    # s._L2 = np.diag(s._K_r.dot(s._C).dot(_m))
                    s._D = s._C.dot(s._L)
                    s._DE = s._D.dot(s._E)
                    s._DKg = s._D.dot(s._K_r.T.dot(g_y))

                    _m = s._L.dot(s._E).dot(np.linalg.solve(s._DE.dot(s._DE) + alpha_2 * np.eye(self._kernel_size), s._DKg))
                    # _m = s._L2.T.dot(s._E2).dot(np.linalg.solve(s._DE.dot(s._DE) + alpha_2 * np.eye(self.__kernel_size), s._DKg))

                    _m = _m / _m.sum(axis=0)

                    return _m, None

                def transform(s, _m):
                    return s._X.T.dot(s._K_r).dot(s._C.dot(_m))

        if self._model == 'kkr':
            class Model:
                def __init__(s):
                    s._K = self.kernel_k(self._reference_set[['context']].values)
                    s._G = self.kernel_g(self._reference_set[['samples']].values)
                    s._X = self._reference_set[['samples']].values

                    s._O = np.linalg.solve(s._K + alpha_1 * np.eye(self._kernel_size), s._K)
                    s._GO = s._G.dot(s._O)

                    s._K_all = self.kernel_k(self._reference_set[['context']].values,
                                             self._data[['context']].values)
                    s._C_0 = np.linalg.solve(s._K + alpha_1 * np.eye(s._K.shape[0]), s._K_all)

                    s._X = self._reference_set[['context']].values
                    s._XO = s._X.T.dot(s._O)

                    s.m_0 = np.mean(s._C_0, axis=1, keepdims=True)
                    s.cov_0 = np.cov(s._C_0)

                    # normalization
                    s.cov_0 = 0.5 * (s.cov_0 + s.cov_0.T)
                    # [eig_v, eig_r] = np.linalg.eigh(cov_0)
                    # eig_v[eig_v < 1e-16 * eig_v.max()] = 1e-16 * eig_v.max()
                    # cov_0 = eig_r.dot(np.diag(eig_v)).dot(eig_r.T)

                    s._Ocov = np.empty((self._kernel_size, self._kernel_size), dtype=np.double)
                    s._Q_denominator_T = np.empty((self._kernel_size, self._kernel_size), dtype=np.double)
                    s._Q = np.empty((self._kernel_size, self._kernel_size), dtype=np.double)

                @staticmethod
                def k_g(y):
                    return self.kernel_g(self._reference_set[['samples']].values, y)

                def update(s, _m, _cov, _g_y):
                    # compute kernel Kalman gain
                    s._Ocov = s._O.dot(_cov)
                    s._Q_denominator_T = s._Ocov.dot(s._GO.T) + alpha_2 * np.eye(self._kernel_size)
                    s._Q = np.linalg.solve(s._Q_denominator_T, s._Ocov).T

                    # perform Kalman update
                    _m = _m + s._Q.dot(_g_y - s._GO.dot(_m))
                    _cov = _cov - s._Q.dot(s._GO).dot(_cov)

                    # normalization
                    _cov = 0.5 * (_cov + _cov.T)
                    # [eig_v, eig_r] = np.linalg.eigh(_cov)
                    # eig_v[eig_v < 1e-16 * eig_v.max()] = 1e-16 * eig_v.max()
                    # _cov = eig_r.dot(np.diag(eig_v)).dot(eig_r.T)

                    return _m, _cov

                def transform(s, _m):
                    return s._XO.dot(_m)

        if self._model == 'sub_kkr':
            class Model:
                def __init__(s):
                    s._K_r = self.kernel_k(self._data[['context']].values,
                                           self._reference_set[['context']].values)
                    s._G = self.kernel_g(self._data[['samples']].values)

                    s._O = np.linalg.inv(s._K_r.T.dot(s._K_r) + alpha_1 * np.eye(self._kernel_size))
                    s._KO = s._K_r.dot(s._O)
                    s._GKO = s._G.dot(s._KO)
                    s._KGKO = s._K_r.T.dot(s._GKO)

                    s._X = self._data[['context']].values
                    s._XKO = s._X.T.dot(s._KO)

                    s._K_r0 = self.kernel_k(self._reference_set[['context']].values,
                                            self._data[['context']].values)

                    s.m_0 = np.mean(s._K_r0, axis=1, keepdims=True)
                    s.cov_0 = np.cov(s._K_r0)

                    s.cov_0 = 0.5 * (s.cov_0 + s.cov_0.T)

                    s._Ocov = np.empty((self._kernel_size, self._kernel_size), dtype=np.double)
                    s._Q_denominator_T = np.empty((self._kernel_size, self._kernel_size), dtype=np.double)
                    s._Q = np.empty((self._kernel_size, self._kernel_size), dtype=np.double)

                @staticmethod
                def k_g(y):
                    return self.kernel_g(self._data[['samples']].values, y)

                def update(s, _m, _cov, _g_y):
                    s._Ocov = s._O.dot(_cov)
                    s._Q_denominator_T = s._Ocov.dot(s._KGKO.T) + alpha_2 * np.eye(self._kernel_size)
                    s._Q = np.linalg.solve(s._Q_denominator_T, s._Ocov).T.dot(s._K_r.T)

                    # update mean and covariance
                    _m = _m + s._Q.dot(_g_y - s._GKO.dot(_m))
                    _cov = _cov - s._Q.dot(s._GKO).dot(_cov)

                    # normalization
                    _cov = 0.5 * (_cov + _cov.T)

                    return _m, _cov

                def transform(s, _m):
                    return s._XKO.dot(_m)

        model = Model()
        t_diff = 0

        # perform Bayesian updates
        if self._model in ['kkr', 'sub_kkr']:
            num_trials = len(self._test_data.index.remove_unused_levels().levels[0])

            m = np.tile(model.m_0, (1, num_trials))
            cov = model.cov_0.copy()

            for i, df_sample in self._test_data[['context', 'samples']].groupby(level=1):
                g_y = model.k_g(df_sample['samples'].values.reshape((-1, 1)))

                t0 = time.clock()
                m, cov = model.update(m, cov, g_y)
                t_diff += time.clock() - t0

                self._test_data.loc[(slice(None), i), 'mu'] = model.transform(m).T
        else:
            for i, df_trial in self._test_data[['context', 'samples']].groupby(level=0):
                m = model.m_0.copy()
                cov = None

                for (t_i, s_i), context, sample in df_trial.itertuples():
                    g_y = model.k_g(sample)

                    t0 = time.clock()
                    m, cov = model.update(m, cov, g_y)
                    t_diff += time.clock() - t0

                    # normalization
                    # m = m / m.sum(axis=0)

                    # project into state space
                    self._test_data.loc[(t_i, s_i), 'mu'] = model.transform(m)

        score = ((self._test_data['context'] - self._test_data['mu']) ** 2).groupby(level=1).mean().sum()

        if with_time:
            return score, t_diff
        else:
            return score
