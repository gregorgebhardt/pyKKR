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


def linear_kernel(a, b=None):
    if b is None:
        return a.dot(a.T)
    else:
        return a.dot(b.T)


class Kernel(object):
    # Evaluates kernel for all elements
    # Compute gram matrix of the form
    #  -----------------------------
    #  | k(x₁,y₁) | k(x₁,y₂) | ... |
    #  -----------------------------
    #  | k(x₂,y₁) | k(x₂,y₂) | ... |
    #  -----------------------------
    #  | ...      | ...      | ... |
    #  -----------------------------
    # if y=None, K(x,x) is computed
    def get_gram_matrix(self, a, b=None):
        pass

    # Returns the diagonal of the gram matrix
    # which means the kernel is evaluated between every data point and itself
    def get_gram_diag(self, data):
        diag = np.zeros(data.shape[0])

        for i in range(data.shape[0]):
            diag[i] = self.get_gram_matrix(data[i, :], data[i, :])


class LinearBandwidthKernel(Kernel):
    def __init__(self, bandwidth=1.):
        self.bandwidth = bandwidth

    def get_gram_matrix(self, a, b=None):
        if len(a.shape) == 1:
            a = a.reshape((1, -1))

        if np.isscalar(self.bandwidth):
            q = np.eye(a.shape[1]) / (self.bandwidth ** 2)
        else:
            assert (a.shape[1] == self.bandwidth.shape[0])
            q = np.diag((np.ones(self.bandwidth.shape[0]) / (self.bandwidth ** 2)))

        aq = a.dot(q)
        if b is None:
            return aq.dot(a.T)
        else:
            return aq.dot(b.T)

    def __call__(self, a: np.ndarray, b: np.ndarray = None) -> np.ndarray:
        return self.get_gram_matrix(a, b)


class ExponentialQuadraticKernel(Kernel):
    def __init__(self, bandwidth=1., normalized=False, ard=False):
        self.ard = ard
        self.normalized = normalized
        self.bandwidth = bandwidth

    def get_gram_matrix(self, a: np.ndarray, b: np.ndarray = None) -> np.ndarray:
        if len(a.shape) == 1:
            a = a.reshape((1, -1))

        if np.isscalar(self.bandwidth):
            q = np.eye(a.shape[1]) / (self.bandwidth ** 2)
        else:
            assert (a.shape[1] == self.bandwidth.shape[0])
            q = np.diag((np.ones(self.bandwidth.shape[0]) / (self.bandwidth ** 2)))

        aq = a.dot(q)
        aq_a = np.sum(aq * a, axis=1)
        if b is None:
            sqdist = aq_a[:, np.newaxis] + aq_a - 2 * aq.dot(a.T)
        else:
            # catch lists and scalars
            b = np.array(b)

            # reshape if one-dimensional
            if len(b.shape) == 1:
                b = b.reshape(-1, 1)

            bq_b = np.sum(b.dot(q) * b, axis=1)
            # Equivalent to MATLAB bsxfun(@plus, ..)
            sqdist = aq_a[:, np.newaxis] + bq_b - 2 * aq.dot(b.T)
        K = np.exp(-0.5 * sqdist)

        if self.normalized:
            K = K / np.sqrt(np.prod(self.bandwidth) ** 2 * (2 * np.pi) ** a.shape[1])

        return K

    def __call__(self, a: np.ndarray, b: np.ndarray = None) -> np.ndarray:
        return self.get_gram_matrix(a, b)

    def get_gram_diag(self, data):
        diag = np.ones(data.shape[0])
        if self.normalized:
            return diag / np.sqrt(np.prod(self.bandwidth) ** 2 * (2 * np.pi) ** data.shape[1])
        else:
            return diag
