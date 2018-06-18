from cython.parallel import parallel, prange
import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64


cdef class MahaDist:
    def __init__(self, bandwidth=None):
        self.bandwidth = np.array([1.], dtype=DTYPE)
        self.preprocessor = None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef np.ndarray get_distance_matrix(self, double[:, :] a, double[:, :] b=None):
        maha_dist = self.get_distance_matrix_c(a, b)
        return np.asarray(maha_dist)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef double[:, :] get_distance_matrix_c(self, double[:, :] a, double[:, :] b=None):
        """computes the squared mahalanobis distance
        
        :param a: q x d matrix of kilobot positions
        :param b: r x d matrix of kilobot positions
        :param eval_gradient: a boolean whether the function should also return the gradients wrt the bandwidth
        :return:  q x r matrix if a and b are given, q x q matrix if only a is given
        """
        # assert a.dtype == DTYPE

        cdef Py_ssize_t q, r, d
        cdef Py_ssize_t i, j, k

        q = a.shape[0]
        d = a.shape[1]

        cdef double[:, :] maha_dist
        if b is None:
            maha_dist = np.empty((q, q))
        else:
            assert b.shape[1] == d
            r = b.shape[0]
            maha_dist = np.empty((q, r))

        cdef double[:] bw
        if np.isscalar(self.bandwidth):
            bw = np.ones(a.shape[1]) / (2 * self.bandwidth)
        else:
            bw = 1 / (2 * self.bandwidth)

        # assert d == len(bw)

        cdef double sq_dist_a = .0, sq_dist_ab = .0
        with nogil, parallel():
            if b is None:
                for i in prange(q, schedule='guided'):
                    for j in range(i, q):
                        maha_dist[i, j] = .0
                        maha_dist[j, i] = .0
                        sq_dist_a = .0
                        for k in range(d):
                            sq_dist_a += (a[i, k] - a[j, k])**2 * bw[k]

                        maha_dist[i, j] += sq_dist_a
                        if j != i:
                            maha_dist[j, i] += sq_dist_a
            else:
                for i in prange(q, schedule='guided'):
                    for j in range(r):
                        maha_dist[i, j] = .0
                        for k in range(d):
                            maha_dist[i, j] += (a[i, k] - b[j, k])**2 * bw[k]

        return maha_dist

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef np.ndarray get_distance_matrix_gradient(self, double[:, :] a, double[:, :] b=None):
        """computes the squared mahalanobis distance
        
        :param a: q x d matrix of kilobot positions
        :param b: r x d matrix of kilobot positions
        :param eval_gradient: a boolean whether the function should also return the gradients wrt the bandwidth
        :return:  q x r matrix if a and b are given, q x q matrix if only a is given
        """
        # assert a.dtype == DTYPE

        cdef Py_ssize_t q, r, d
        cdef Py_ssize_t i, j, k

        q = a.shape[0]
        d = a.shape[1]

        assert d == len(self.bandwidth)

        cdef double[:, :, :] d_maha_dist_d_bw
        if b is None:
            d_maha_dist_d_bw = np.empty((q, q, d))
        else:
            assert b.shape[1] == d
            r = b.shape[0]
            d_maha_dist_d_bw = np.empty((q, r, d))

        cdef double[:] bw = 1 / self.bandwidth

        with nogil, parallel():
            if b is None:
                for i in prange(q, schedule='guided'):
                    for j in range(i, q):
                        for k in range(d):
                            d_maha_dist_d_bw[i, j, k] = -(a[i, k] - a[j, k])**2 * bw[k]**2 / 2
                            d_maha_dist_d_bw[j, i, k] = -(a[i, k] - a[j, k])**2 * bw[k]**2 / 2
            else:
                for i in prange(q, schedule='guided'):
                    for j in range(r):
                        for k in range(d):
                            d_maha_dist_d_bw[i, j, k] = -(a[i, k] - b[j, k])**2 * bw[k]**2 / 2

        return np.asarray(d_maha_dist_d_bw)

    def diag(self, np.ndarray data):
        return np.zeros(data.shape[0])

    def __call__(self, X, Y=None):
        if self.preprocessor:
            X = self.preprocessor(X)
            if Y is not None:
                Y = self.preprocessor(Y)
        return self.get_distance_matrix(X, Y)

    def set_bandwidth(self, bandwidth):
        if np.isscalar(bandwidth):
            self.bandwidth = np.array([bandwidth])
        else:
            self.bandwidth = bandwidth

    def get_bandwidth(self):
        return self.bandwidth
