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

from kkr.kernels import ExponentialQuadraticKernel


class KernelKalmanFilter(object):
    def __init__(self, states_1: np.ndarray, states_2: np.ndarray, observations: np.ndarray,
                 init_states: np.ndarray, preimage_states: np.ndarray = None):
        super().__init__()

        assert (states_1.shape == states_2.shape)
        assert (states_2.shape[0] == observations.shape[0])

        self.states_1 = states_1
        self.states_2 = states_2
        self.observations = observations
        self.init_states = init_states
        self.preimage_states = preimage_states
        if self.preimage_states is None:
            self.preimage_states = observations

        self.kernel_k = ExponentialQuadraticKernel()
        self.kernel_g = ExponentialQuadraticKernel()
        self.alpha_t = np.exp(-10)
        self.alpha_o = np.exp(-10)
        self.alpha_q = np.exp(-10)

        self.num_states = states_1.shape[0]
        self.output_dimension = self.preimage_states.shape[1]

        self._model_learned = False

        self._k_g = lambda y: self.kernel_g(self.observations, y)
        self._transition_model = None
        self._transition_cov = None
        self._observation_model = None
        self._observation_cov = None
        self._GO = None
        self._RG = None
        self._XO = None

        self.use_posterior_decoding = False
        self._K = None
        self._diagK = None

        self.use_observation_cov = False
        self.interpret_bandwidth_as_factor = True

        self._m_0 = None
        self._cov_0 = None

        self._precomputed_observations = None
        self._precomputed_Q = None
        self._precomputed_cov = None
        self._precomputed = False

    def learn_model(self, bandwidth_k=None, bandwidth_g=None,
                    alpha_t=None, alpha_o=None, alpha_q=None):
        # update model parameters
        if bandwidth_k is not None:
            self.kernel_k.bandwidth = bandwidth_k
        if bandwidth_g is not None:
            self.kernel_g.bandwidth = bandwidth_g
        if alpha_t is not None:
            self.alpha_t = alpha_t
        if alpha_o is not None:
            self.alpha_o = alpha_o
        if alpha_q is not None:
            self.alpha_q = alpha_q

        # compute kernel matrices
        _K_11 = self.kernel_k(self.states_1)
        _K_12 = self.kernel_k(self.states_1, self.states_2)
        _K_22 = self.kernel_k(self.states_2)
        _G_22 = self.kernel_g(self.observations)

        # compute model matrices and errors
        # transition model
        self._transition_model = np.linalg.solve(_K_11 + self.alpha_t * np.eye(self.num_states), _K_12)

        # covariance of the error of the transition model
        _v = np.linalg.solve(_K_11 + self.alpha_t * np.eye(self.num_states), _K_11) - np.eye(self.num_states)
        self._transition_cov = (_v.dot(_v.T)) / self.num_states

        # observation model
        self._observation_model = np.linalg.solve(_K_22 + self.alpha_o * np.eye(self.num_states), _K_22)
        self._GO = _G_22.dot(self._observation_model)

        # covariance of the error of the observation model
        _r = self._observation_model - np.eye(self.num_states)
        self._observation_cov = (_r.dot(_r.T)) / self.num_states
        self._RG = self._observation_cov.dot(_G_22)

        # projection into state space
        self._XO = self.preimage_states.T.dot(self._observation_model)

        # initial embedding
        _K_20 = self.kernel_k(self.states_2, self.init_states)
        _C_0 = np.linalg.solve(_K_22 + self.alpha_o * np.eye(self.num_states), _K_20)

        self._m_0 = np.mean(_C_0, axis=1, keepdims=True)
        self._cov_0 = np.cov(_C_0)

        # self.S_0 = 0.5 * (self.S_0 + self.S_0.T)
        # [eig_v, eig_r] = linalg.eigh(self.S_0)
        # eig_v[eig_v < 1e-16*eig_v.max()] = 1e-16*eig_v.max()
        # self.S_0 = eig_r.dot(np.diag(eig_v)).dot(eig_r.T)
        #
        # self.m_0 = self.m_0 / self.m_0.sum(axis=0)

        if self.use_posterior_decoding:
            self._K = _K_11
            self._diagK = self.kernel_k.get_gram_diag(self.states_1)

        self._precomputed = False
        self._model_learned = True

    def filter(self, observations, return_m=False, return_cov=False):
        assert self._model_learned
        assert (len(observations.shape) <= 3)

        if len(observations.shape) == 1:
            observations = observations.reshape(-1, 1, 1)
        elif len(observations.shape) == 2:
            observations = observations.reshape(observations.shape + (1,))

        num_observations, data_dimension, num_parallel_evaluations = observations.shape

        mu_x = np.zeros((observations.shape[0], self.output_dimension, num_parallel_evaluations))
        sigma_x = np.zeros((observations.shape[0], self.output_dimension, self.output_dimension))

        m, cov = self.initial_embeddings(num_parallel_evaluations)

        m_storage_prior = np.zeros((num_observations, *m.shape)) if return_m in ['prior', 'both'] else None
        cov_storage_prior = np.zeros((num_observations, *cov.shape)) if return_cov in ['prior', 'both'] else None
        m_storage_post = np.zeros((num_observations, *m.shape)) if return_m in ['post', 'posterior', 'both'] else None
        cov_storage_post = np.zeros((num_observations, *cov.shape)) if return_cov in ['post', 'posterior',
                                                                                      'both'] else None

        ignore_precomputed = False

        for i in range(num_observations):
            # store embeddings
            if return_m in ['prior', 'both']:
                m_storage_prior[i, :] = m
            if return_cov in ['prior', 'both']:
                cov_storage_prior[i, :, :] = cov

            # observation update
            if not np.isnan(observations[i, :]).any():
                # check for precomputed _Q
                # if we have more observations than precomputed or in the precomputed Qs there was no observation at
                # step i then ignore the precomputed Qs from now on
                if self._precomputed:
                    if i >= len(self._precomputed_observations) or not self._precomputed_observations[i]:
                        ignore_precomputed = True

                # if no precomputed Qs exist or if they are ignored, do the normal update
                if not self._precomputed or ignore_precomputed:
                    m, cov = self.observation_update(m, cov, observations[i, ...].T)
                # otherwise do the update with the precomputed Q
                else:
                    _Q = self._precomputed_Q[i]
                    cov = self._precomputed_cov[i]
                    m, _ = self.observation_update(m, cov, observations[i, ...].T, _Q)

            # output transform
            if self.use_posterior_decoding:
                mu_x[i, :, :], sigma_x[i, :, :] = self.max_aposteriori_output(m, cov)
            else:
                mu_x[i, :, :], sigma_x[i, :, :] = self.transform_outputs(m, cov)

            # store embeddings
            if return_m in ['post', 'posterior', 'both']:
                m_storage_post[i, :] = m
            if return_cov in ['post', 'posterior', 'both']:
                cov_storage_post[i, :, :] = cov

            # transition update
            m, cov = self.transition_update(m, cov)

        if num_parallel_evaluations == 1:
            mu_x = mu_x.squeeze(axis=2)

        return_values = (mu_x, sigma_x)

        if return_m in ['prior', 'both']:
            return_values += (m_storage_prior,)
        if return_cov in ['prior', 'both']:
            return_values += (cov_storage_prior,)
        if return_m in ['post', 'posterior', 'both']:
            return_values += (m_storage_post,)
        if return_cov in ['post', 'posterior', 'both']:
            return_values += (cov_storage_post,)

        return return_values

    def initial_embeddings(self, num_parallel_evaluations: int=1):
        m = np.tile(self._m_0, [num_parallel_evaluations])
        cov = self._cov_0.copy()

        return m, cov

    def observation_update(self, m, cov, y, _Q=None, return_Q=False):
        # embed observation
        g_y = self._k_g(y)

        # kernel Kalman gain
        if _Q is None:
            _O_cov = self._observation_model.dot(cov)
            if self.use_observation_cov:
                _Q_denominator_T = _O_cov.dot(self._GO.T) + self._RG + self.alpha_q * np.eye(self.num_states)
            else:
                _Q_denominator_T = _O_cov.dot(self._GO.T) + self.alpha_q * np.eye(self.num_states)
            _Q = np.linalg.solve(_Q_denominator_T, _O_cov).T

            # update covariance
            cov = cov - _Q.dot(self._GO).dot(cov)

            # normalization
            cov = 0.5 * (cov + cov.T)

        # update mean and covariance
        m = m + _Q.dot(g_y - self._GO.dot(m))

        m = m / m.sum(axis=0)

        if return_Q:
            return m, cov, _Q
        else:
            return m, cov

    def transition_update(self, m, cov):
        m = self._transition_model.dot(m)
        if cov is not None:
            cov = self._transition_model.dot(cov).dot(self._transition_model.T) + self._transition_cov

            # normalization
            cov = 0.5 * (cov + cov.T)

        # m = m / m.sum(axis=0)

        return m, cov

    def transform_outputs(self, m, cov):
        mu_x = self._XO.dot(m)
        sigma_x = self._XO.dot(cov).dot(self._XO.T)

        return mu_x, sigma_x

    def max_aposteriori_output(self, m, cov):
        w = -2 * m.T.dot(self._K) + self._diagK
        max_w = np.argmax(w, axis=1)
        max_x = self.preimage_states[max_w, :].T
        sigma_x = self._XO.dot(cov).dot(self._XO.T)

        return max_x, sigma_x

    def precompute_Q_and_S(self, observations):
        self._precomputed_observations = observations.copy()

        if self._precomputed_Q is None:
            kernel_size = self._transition_model.shape[0]
            self._precomputed_Q = np.empty((len(observations), kernel_size, kernel_size))
            self._precomputed_cov = np.empty((len(observations), kernel_size, kernel_size))

        m, cov = self.initial_embeddings(1)

        for i, obs in enumerate(observations):
            if obs:
                m, cov, _Q = self.observation_update(m, cov, np.ones((1, self.observations.shape[1])), return_Q=True)
                self._precomputed_Q[i] = _Q
                self._precomputed_cov[i] = cov
            else:
                self._precomputed_Q[i] = np.nan
                self._precomputed_cov[i] = np.nan

            m, cov = self.transition_update(m, cov)

        self._precomputed = True


class SubspaceKernelKalmanFilter(KernelKalmanFilter):
    def __init__(self,
                 states_1: np.ndarray,
                 states_2: np.ndarray,
                 observations: np.ndarray,
                 init_states: np.ndarray,
                 preimage_states: np.ndarray = None,
                 subspace_states: np.ndarray = None):
        super().__init__(states_1, states_2, observations, init_states, preimage_states)

        if subspace_states is None:
            subspace_states = states_1
            Warning("No subspace data set was given. Taking states_1 as subspace data.")

        assert (subspace_states.shape <= self.states_1.shape)

        self.subspace_states = subspace_states
        self.num_subspace_states = self.subspace_states.shape[0]

        self._GKO = None
        self._KGKO = None
        self._KO = None
        self._XKO = None
        self._K_1r = None

    def learn_model(self, bandwidth_k=None, bandwidth_g=None,
                    alpha_t=None, alpha_o=None, alpha_q=None):
        # update model parameters
        if bandwidth_k is not None:
            self.kernel_k.bandwidth = bandwidth_k
        if bandwidth_g is not None:
            self.kernel_g.bandwidth = bandwidth_g
        if alpha_t is not None:
            self.alpha_t = alpha_t
        if alpha_o is not None:
            self.alpha_o = alpha_o
        if alpha_q is not None:
            self.alpha_q = alpha_q

        # compute kernel matrices
        self._K_1r = self.kernel_k(self.states_1, self.subspace_states)
        _K_2r = self.kernel_k(self.states_2, self.subspace_states)

        _G = self.kernel_g(self.observations)

        # compute model matrices and errors
        # transition model
        self._transition_model = np.linalg.solve(
            self._K_1r.T.dot(self._K_1r) + self.alpha_t * np.eye(self.num_subspace_states),
            self._K_1r.T.dot(_K_2r)).T

        # covariance of the error of the transition model
        _v = self._transition_model.dot(self._K_1r.T) - _K_2r.T
        self._transition_cov = (_v.dot(_v.T)) / self.num_states

        # observation model
        # self._observation_model = np.linalg.solve(self._K_1r.T.dot(self._K_1r) +
        #                                           self.alpha_o * np.eye(self.num_subspace_states),
        #                                           np.eye(self.num_subspace_states))
        self._observation_model = np.linalg.inv(
            self._K_1r.T.dot(self._K_1r) + self.alpha_o * np.eye(self.num_subspace_states))
        self._KO = self._K_1r.dot(self._observation_model)
        self._GKO = _G.dot(self._KO)
        self._KGKO = self._K_1r.T.dot(self._GKO)

        if self.preimage_states is None:
            self.preimage_states = self.observations

        # projection into state space
        self._XKO = self.preimage_states.T.dot(self._KO)
        self.output_dimension = self.preimage_states.shape[1]

        # initial embedding
        _K_r0 = self.kernel_k(self.subspace_states, self.init_states)

        self._m_0 = np.mean(_K_r0, axis=1, keepdims=True)
        self._cov_0 = np.cov(_K_r0)

        self._cov_0 = 0.5 * (self._cov_0 + self._cov_0.T)
        # [eig_v, eig_r] = linalg.eigh(self.P_0)
        # eig_v[eig_v < 1e-16 * eig_v.max()] = 1e-16 * eig_v.max()
        # self._cov_0 = eig_r.dot(np.diag(eig_v)).dot(eig_r.T)
        #
        # self._m_0 = self._m_0 / self._m_0.sum(axis=0)

        if self.use_posterior_decoding:
            self._diagK = self.kernel_k.get_gram_diag(self.states_1)

        self._precomputed = False
        self._model_learned = True

    def observation_update(self, n, cov, y, _Q=None, return_Q=False):
        # embed observation
        g_y = self._k_g(y)

        # kernel Kalman gain
        if not _Q:
            _O_cov = self._observation_model.dot(cov)
            _Q_denominator_T = _O_cov.dot(self._KGKO.T) + self.alpha_q * np.eye(self.num_subspace_states)
            _Q = np.linalg.solve(_Q_denominator_T, _O_cov).T #.dot(self._K_1r.T)

            # update covariance
            cov = cov - _Q.dot(self._KGKO).dot(cov)

            # normalization
            cov = 0.5 * (cov + cov.T)
            # [eig_v, eig_r] = np.linalg.eigh(cov)
            # eig_v[eig_v < 1e-16 * eig_v.max()] = 1e-16 * eig_v.max()
            # cov = eig_r.dot(np.diag(eig_v)).dot(eig_r.T)

        # update mean and covariance
        n = n + _Q.dot(self._K_1r.T.dot(g_y) - self._KGKO.dot(n))

        # n[n < .0] = 1e-8
        # n = n / n.sum(axis=0)

        if return_Q:
            return n, cov, _Q
        else:
            return n, cov

    def transition_update(self, n, cov):
        n = self._transition_model.dot(n)
        if cov is not None:
            cov = self._transition_model.dot(cov).dot(self._transition_model.T) + self._transition_cov

            # normalization
            cov = 0.5 * (cov + cov.T)
            # [eig_v, eig_r] = np.linalg.eigh(cov)
            # eig_v[eig_v < 1e-16 * eig_v.max()] = 1e-16 * eig_v.max()
            # cov = eig_r.dot(np.diag(eig_v)).dot(eig_r.T)

        # n[n < .0] = 1e-8
        # n = n / n.sum(axis=0)

        return n, cov

    def transform_outputs(self, n, cov):
        # mu_x = self._XKO.dot(n)
        # sigma_x = self._XKO.dot(cov).dot(self._XKO.T)

        # normalize the weights in the RKHS of the full data set.
        m = self._KO.dot(n)
        m /= m.sum(axis=0)
        mu_x = self.preimage_states.T.dot(m)

        sigma_x = self._XKO.dot(cov).dot(self._XKO.T)

        return mu_x, sigma_x

    def max_aposteriori_output(self, m, cov):
        w = -2 * m.T.dot(self._K_1r.T) + self._diagK
        max_w = np.argmax(w, axis=1)
        max_x = self.preimage_states[max_w, :].T
        sigma_x = self._XKO.dot(cov).dot(self._XKO.T)

        return max_x, sigma_x


class KernelBayesFilter:
    def __init__(self,
                 states_1: np.ndarray,
                 states_2: np.ndarray,
                 observations: np.ndarray,
                 init_states: np.ndarray,
                 preimage_states: np.ndarray = None):

        self.states_1 = states_1
        self.states_2 = states_2
        self.observations = observations
        self.init_states = init_states
        self.preimage_states = preimage_states
        if self.preimage_states is None:
            self.preimage_states = observations

        self.kernel_k = ExponentialQuadraticKernel()
        self.kernel_g = ExponentialQuadraticKernel()
        self._k_g = lambda y: self.kernel_g(self.observations, y)

        self.num_states = states_1.shape[0]
        self.output_dimension = self.preimage_states.shape[1]

        self.alpha_t = np.exp(-10)
        self.alpha_o1 = np.exp(-10)
        self.alpha_o2 = np.exp(-10)

        self._transition_model = None

        self._G = None
        self._C = None
        self._XO = None
        self._cov_0 = None
        self._m_0 = None

        self._model_learned = False

    def learn_model(self, bandwidth_k=None, bandwidth_g=None, alpha_t=None, alpha_o1=None, alpha_o2=None):
        # update model parameters
        if bandwidth_k is not None:
            self.kernel_k.bandwidth = bandwidth_k
        if bandwidth_g is not None:
            self.kernel_g.bandwidth = bandwidth_g
        if alpha_t is not None:
            self.alpha_t = alpha_t
        if alpha_o1 is not None:
            self.alpha_o1 = alpha_o1
        if alpha_o2 is not None:
            self.alpha_o2 = alpha_o2

        # compute kernel matrices
        _K_11 = self.kernel_k(self.states_1)
        _K_12 = self.kernel_k(self.states_1, self.states_2)
        _K_22 = self.kernel_k(self.states_2, self.states_2)
        _K_20 = self.kernel_k(self.states_2, self.init_states)
        self._G = self.kernel_g(self.observations)

        # transition model
        self._transition_model = np.linalg.solve(_K_11 + self.alpha_t * np.eye(self.num_states), _K_12)

        # initialize embedding
        self._C = np.linalg.solve(_K_22 + self.alpha_o1 * np.eye(self.num_states), _K_22)

        self._cov_0 = np.linalg.solve(_K_22 + self.alpha_o1 * np.eye(self.num_states), _K_20)
        self._m_0 = np.mean(self._cov_0, axis=1, keepdims=True)

        # observation model
        _O = np.linalg.solve(_K_22 + self.alpha_t * np.eye(self.num_states), _K_22)

        # projection into state space
        self._XO = self.preimage_states.T.dot(_O)

        self._model_learned = True

    def filter(self, observations, return_m=False):
        assert self._model_learned
        assert (len(observations.shape) <= 3)

        if len(observations.shape) == 1:
            observations = observations.reshape(-1, 1, 1)
        elif len(observations.shape) == 2:
            observations = observations.reshape(observations.shape + (1,))

        num_observations, data_dimension, num_parallel_evaluations = observations.shape

        mu_x = np.zeros((observations.shape[0], self.output_dimension, num_parallel_evaluations))

        m = self.initial_embeddings(num_parallel_evaluations)

        m_storage_prior = np.zeros((num_observations, *m.shape)) if return_m in ['prior', 'both'] else None
        m_storage_post = np.zeros((num_observations, *m.shape)) if return_m in ['post', 'posterior', 'both'] else None

        for i in range(num_observations):
            # store embeddings
            if return_m in ['prior', 'both']:
                m_storage_prior[i, :] = m

            # observation update
            if not np.isnan(observations[i, :]).any():
                for j in range(num_parallel_evaluations):
                    m[:, [j]] = self.observation_update(m[:, [j]], observations[i, :, [j]])

            # output transform
            mu_x[i, :, :] = self.transform_outputs(m)

            # store embeddings
            if return_m in ['post', 'posterior', 'both']:
                m_storage_post[i, :] = m

            # transition update
            m = self.transition_update(m)

        if num_parallel_evaluations == 1:
            mu_x = mu_x.squeeze(axis=2)

        return_values = (mu_x, None)

        if return_m in ['prior', 'both']:
            return_values += (m_storage_prior,)
        if return_m in ['post', 'posterior', 'both']:
            return_values += (m_storage_post,)

        return return_values

    def initial_embeddings(self, num_parallel_evaluations: int = 1):
        m = np.tile(self._m_0, [num_parallel_evaluations])

        return m

    def observation_update(self, m, y):
        # embed observation
        g_y = self._k_g(y)

        _D = np.diag(self._C.dot(m).flat)
        _DG = _D.dot(self._G)
        m = np.linalg.solve(_DG + self.alpha_o2 * np.eye(self.num_states), _D).dot(g_y)

        # m = m / m.sum(axis=0)

        return m

    def transition_update(self, m):
        m = self._transition_model.dot(m)
        # m = m / m.sum(axis=0)

        return m

    def transform_outputs(self, m):
        mu_x = self._XO.dot(m)

        return mu_x


class SubspaceKernelBayesFilter(KernelBayesFilter):
    def __init__(self,
                 states_1: np.ndarray,
                 states_2: np.ndarray,
                 observations: np.ndarray,
                 init_states: np.ndarray = None,
                 preimage_states: np.ndarray = None,
                 subspace_states: np.ndarray = None):

        super().__init__(states_1, states_2, observations, init_states, preimage_states)

        if subspace_states is None:
            subspace_states = states_1
            Warning("No subspace data set was given. Taking states_1 as subspace data.")

        assert (subspace_states.shape <= self.states_1.shape)

        self._K_r = None
        self._E = None

        self.subspace_states = subspace_states
        self.num_subspace_states = self.subspace_states.shape[0]

    def learn_model(self, bandwidth_k=None, bandwidth_g=None, alpha_t=None, alpha_o1=None, alpha_o2=None):
        # update model parameters
        if bandwidth_k is not None:
            self.kernel_k.bandwidth = bandwidth_k
        if bandwidth_g is not None:
            self.kernel_g.bandwidth = bandwidth_g
        if alpha_t is not None:
            self.alpha_t = alpha_t
        if alpha_o1 is not None:
            self.alpha_o1 = alpha_o1
        if alpha_o2 is not None:
            self.alpha_o2 = alpha_o2

        # compute kernel matrices
        _K_11 = self.kernel_k(self.states_1)
        _K_12 = self.kernel_k(self.states_1, self.states_2)
        _K_22 = self.kernel_k(self.states_2, self.states_2)
        _K_20 = self.kernel_k(self.states_2, self.init_states)
        self._G = self.kernel_g(self.observations)
        self._K_r = self.kernel_k(self.states_2, self.subspace_states)

        # transition model
        self._transition_model = np.linalg.solve(_K_11 + self.alpha_t * np.eye(self.num_states), _K_12)

        # initialize embedding
        self._m_0 = np.ones((self.num_states, 1)) / self.num_states

        # observation model
        _O = np.linalg.solve(_K_22 + self.alpha_t * np.eye(self.num_states), _K_22)

        # projection into state space
        self._XO = self.preimage_states.T.dot(_O)

        self._C = np.linalg.solve(self._K_r.T.dot(self._K_r) + self.alpha_o1 * np.eye(self.num_subspace_states),
                                  self._K_r.T)
        self._E = self._K_r.T.dot(self._G).dot(self._K_r)

        self._model_learned = True

    def observation_update(self, n, y):
        # embed observation
        g_y = self._k_g(y)

        _L = n * self._C.T
        # _L = np.diag(n.flat).dot(self._C.T)
        _D = self._C.dot(_L)
        _DE = _D.dot(self._E)
        _DKg = _D.dot(self._K_r.T.dot(g_y))

        n = _L.dot(self._E).dot(np.linalg.solve(_DE.dot(_DE) + self.alpha_o2 * np.eye(self.num_subspace_states), _DKg))

        n = n / n.sum(axis=0)

        return n
