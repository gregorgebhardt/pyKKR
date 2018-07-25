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

from .filter import KernelKalmanFilter, SubspaceKernelKalmanFilter
from .smoother import KernelForwardBackwardSmoother, SubspaceKernelForwardBackwardSmoother

from .environments.Pendulum import Pendulum

from .evaluation import FilterEvaluation, SmootherEvaluation, BayesianUpdateEvaluation, parameter_naming, \
    parameter_transform, bandwidth_factor, window_bandwidth_factor, dimension_bandwidth_factor, parameter_arrays, \
    exception_catcher, tile_bandwidth

from .kernels import ExponentialQuadraticKernel, LinearBandwidthKernel, linear_kernel

from .preprocessors import generate_data_windows, compute_median_bandwidth, add_noise, select_reference_set_randomly,\
    select_reference_set_by_kernel_activation, cut_at_impact

from .simulator import Simulator

__all__ = ['KernelKalmanFilter',
           'SubspaceKernelKalmanFilter',
           'KernelForwardBackwardSmoother',
           'SubspaceKernelForwardBackwardSmoother',
           'Pendulum',
           'FilterEvaluation',
           'SmootherEvaluation',
           'BayesianUpdateEvaluation',
           'parameter_transform',
           'parameter_naming',
           'bandwidth_factor',
           'window_bandwidth_factor',
           'dimension_bandwidth_factor',
           'tile_bandwidth',
           'parameter_arrays',
           'exception_catcher',
           'ExponentialQuadraticKernel',
           'LinearBandwidthKernel',
           'linear_kernel',
           'generate_data_windows',
           'compute_median_bandwidth',
           'add_noise',
           'select_reference_set_randomly',
           'select_reference_set_by_kernel_activation',
           'cut_at_impact',
           'Simulator']
