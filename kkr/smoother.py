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

import numpy as np

from kkr.filter import KernelKalmanFilter, SubspaceKernelKalmanFilter


class KernelForwardBackwardSmoother(object):
    def __init__(self, states_1: np.ndarray, states_2: np.ndarray, states_3: np.ndarray, observations: np.ndarray,
                 init_fw_states: np.ndarray, init_bw_states: np.ndarray, preimage_states: np.ndarray = None):

        self.states_1 = states_1
        self.states_2 = states_2
        self.states_3 = states_3
        self.observations_2 = observations
        self.init_fw_states = init_fw_states
        self.init_bw_states = init_bw_states
        self.preimage_states = preimage_states
        if self.preimage_states is None:
            self.preimage_states = observations

        self.kernel_size = states_1.shape[0]
        self.output_dimension = self.preimage_states.shape[1]

        self._fw_filter = KernelKalmanFilter(self.states_1, self.states_2, self.observations_2, self.init_fw_states,
                                             self.preimage_states)
        self._bw_filter = KernelKalmanFilter(self.states_3, self.states_2, self.observations_2, self.init_bw_states,
                                             self.preimage_states)

        self._K = None

        self._model_learned = False
        self._use_observation_cov = False
        self._fw_filter.use_observation_cov = False
        self._bw_filter.use_observation_cov = False

    @property
    def use_observation_cov(self):
        return self._use_observation_cov

    @use_observation_cov.setter
    def use_observation_cov(self, val: bool):
        self._use_observation_cov = val
        self._fw_filter.use_observation_cov = val
        self._bw_filter.use_observation_cov = val

    @property
    def kernel_k(self):
        return self._fw_filter.kernel_k

    @kernel_k.setter
    def kernel_k(self, kernel):
        self._fw_filter.kernel_k = kernel
        self._bw_filter.kernel_k = kernel

    @property
    def kernel_g(self):
        return self._fw_filter.kernel_g

    @kernel_g.setter
    def kernel_g(self, kernel):
        self._fw_filter.kernel_g = kernel
        self._bw_filter.kernel_g = kernel

    def learn_model(self, bandwidth_k=None, bandwidth_g=None,
                    alpha_t=None, alpha_o=None, alpha_q=None,
                    alpha_t_bw=None, alpha_o_bw=None, alpha_q_bw=None):
        if not alpha_t_bw:
            alpha_t_bw = alpha_t
        if not alpha_o_bw:
            alpha_o_bw = alpha_o
        if not alpha_q_bw:
            alpha_q_bw = alpha_q

        self._fw_filter.learn_model(bandwidth_k, bandwidth_g, alpha_t, alpha_o, alpha_q)
        self._bw_filter.learn_model(bandwidth_k, bandwidth_g, alpha_t_bw, alpha_o_bw, alpha_q_bw)

        self._K = self._fw_filter.kernel_k(self.states_2, self.states_2)

        self._model_learned = True

    def filter(self, observations, return_m=False, return_cov=False):
        return self._fw_filter.filter(observations, return_m, return_cov)

    def filter_bw(self, observations, return_m=False, return_cov=False):
        return self._bw_filter.filter(observations, return_m, return_cov)

    def smooth(self, observations, return_m=False, return_cov=False):
        assert self._model_learned

        if len(observations.shape) == 1:
            observations = observations.reshape(-1, 1, 1)
        elif len(observations.shape) == 2:
            observations = observations.reshape(observations.shape + (1,))

        num_observations, data_dimension, num_parallel_evaluations = observations.shape

        mu_x = np.zeros((observations.shape[0], self.output_dimension, num_parallel_evaluations))
        sigma_x = np.zeros((observations.shape[0], self.output_dimension, self.output_dimension))

        m, cov = self._fw_filter.initial_embeddings(num_parallel_evaluations)

        m_storage = np.zeros((num_observations, *m.shape)) if return_m else None
        cov_storage = np.zeros((num_observations, *cov.shape)) if return_cov else None

        # this could be implemented more memory efficient in that we combine the estimates already during the
        # backward pass, but for simplicity we will do it for now like this
        _, _, m_fw, cov_fw = self._fw_filter.filter(observations, return_m='post', return_cov='post')
        _, _, m_bw, cov_bw = self._bw_filter.filter(np.flipud(observations), return_m='prior', return_cov='prior')
        m_bw = np.flipud(m_bw)
        cov_bw = np.flipud(cov_bw)

        # combine forward and backward estimates
        for t in range(num_observations):
            m_s, cov_s = self.smoothing_update(m_fw[t, :, :], m_bw[t, :, :], cov_fw[t, :, :], cov_bw[t, :, :])

            # map to preimage space
            mu_x[t, :, :], sigma_x[t, :, :] = self._fw_filter.transform_outputs(m_s, cov_s)

            # store embeddings
            if return_m:
                m_storage[t, :] = m_s
            if return_cov:
                cov_storage[t, :, :] = cov_s

        if num_parallel_evaluations == 1:
            mu_x = mu_x.squeeze(axis=2)

        return_values = (mu_x, sigma_x)

        if return_m:
            return_values += (m_storage,)
        if return_cov:
            return_values += (cov_storage,)

        return return_values

    def smoothing_update(self, m_fw, m_bw, cov_fw, cov_bw):
        _Z = np.linalg.solve((cov_fw + cov_bw).dot(self._K), cov_bw).T
        m_s = m_bw + _Z.dot(self._K).dot(m_fw - m_bw)
        cov_s = _Z.dot(self._K).dot(cov_fw)

        return m_s, cov_s

    def precompute_Q_and_S(self, observations_fw, observations_bw=None):
        self._fw_filter.precompute_Q_and_S(observations_fw)
        if observations_bw:
            self._bw_filter.precompute_Q_and_S(observations_bw)
        else:
            self._bw_filter.precompute_Q_and_S(np.flipud(observations_fw))


class SubspaceKernelForwardBackwardSmoother(KernelForwardBackwardSmoother):
    def __init__(self, subspace_states: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subspace_states = subspace_states

        self._fw_filter = SubspaceKernelKalmanFilter(self.states_1, self.states_2, self.observations_2,
                                                    self.init_fw_states, self.preimage_states, self.subspace_states)
        self._bw_filter = SubspaceKernelKalmanFilter(self.states_3, self.states_2, self.observations_2,
                                                    self.init_bw_states, self.preimage_states, self.subspace_states)

    def smoothing_update(self, m_fw, m_bw, cov_fw, cov_bw):
        _Z = np.linalg.solve((cov_fw + cov_bw), cov_bw).T
        m_s = m_bw + _Z.dot(m_fw - m_bw)
        cov_s = _Z.dot(cov_fw)

        return m_s, cov_s
