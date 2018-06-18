cimport numpy as np

cdef class MahaDist:
    cdef public np.ndarray bandwidth
    cdef public preprocessor
    cpdef np.ndarray get_distance_matrix(self, double[:, :] a, double[:, :] b=?)
    cdef double[:, :] get_distance_matrix_c(self, double[:, :] a, double[:, :] b=?)
    cpdef np.ndarray get_distance_matrix_gradient(self, double[:, :] a, double[:, :] b=?)
