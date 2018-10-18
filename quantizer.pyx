# cython: embedsignature=True
# cython: wraparound=False
# # cython: boundscheck=False
# turning off boundscheck is dangerous (you get segfault instead of IndexError) and only gives about 10% speedup

# Yibo Yang, 2018

cimport numpy as np
import numpy as np

# for the type of numeric data; float32 is generally fine and memory efficient
value_type = np.float32  # runtime type, for dynamically creating numpy arrays etc.
ctypedef np.float32_t value_type_t  # '_t' suffix means compile-time type; np.float32 ~ C float, np.float64 ~ C double

# for the cluster assignments; technically should be [0, K-1], unsigned, but we allow negative numbers for flexibility;
asm_type = np.int32
ctypedef np.int32_t asm_type_t

# for precision, e.g. during accumulation
ctypedef double prec_type

from libc.math cimport fabs
from numpy.math cimport INFINITY as inf
from libcpp.vector cimport vector
from quantizer cimport kmeans_cluster as cpp_kmeans_cluster
from quantizer cimport kmeans_cluster_lazy as cpp_kmeans_cluster_lazy
from builtins import property


def create(asm_type_t[:] cluster_assignments, value_type_t[:] cluster_centers):
    """
    Factory method that creates new Quantizer object given all its data attributes; used by copy.copy and pickle.restore
    :param cluster_assignments: 1d memoryview of ints of size num_data_points (C/C++ ints, corresponding to np.int32)
    :param cluster_centers: 1d memoryview of value_type of size num_clusters
    :return: a new Quantizer object with the provided data attributes
    """
    q = Quantizer(cluster_assignments.shape[0], cluster_centers.shape[0])
    q.cluster_assignments = cluster_assignments   # setting by reference
    q.cluster_centers = cluster_centers   # setting by reference
    return q


cdef class Quantizer:
    """
    A fast extension type performs clustering/quantization.
    """

    # Cython memoryviews to be bound to numpy arrays; 'self._cluster_centers' is preferred to 'self.cluster_centers' in
    # numerical code for least overhead and fastest access, unless np.ndarray semantics is desired (e.g. fancy indexing)
    # use np.asarray(self._cluster_assignments) to get a numpy array without copying, or
    # np.array(self._cluster_assignments) to get a copy
    cdef public asm_type_t[:] _cluster_assignments
    cdef public value_type_t[:] _cluster_centers


    @property
    def num_data_points(self):
        return self._cluster_assignments.shape[0]
    @property
    def num_clusters(self):
        return self._cluster_centers.shape[0]

    @property
    def cluster_assignments(self):
        """
        :return: a thin np.ndarray wrapper around underlying memoryview
        """
        return np.asarray(self._cluster_assignments)

    @cluster_assignments.setter
    def cluster_assignments(self, asm_type_t[:] cluster_assignments):
        """
        Bind the self._cluster_assignments memoryview to a new np array buffer; no validity checking!
        :param cluster_assignments:
        :return:
        """
        self._cluster_assignments = cluster_assignments

    @property
    def cluster_centers(self):
        """
        :return: a thin np.ndarray wrapper around underlying memoryview
        """
        return np.asarray(self._cluster_centers)

    @cluster_centers.setter
    def cluster_centers(self, value_type_t[:] cluster_centers):
        """
        Bind the self._cluster_centers memoryview to a new np array buffer
        :param cluster_centers:
        :return:
        """
        self._cluster_centers = cluster_centers

    def __cinit__(self, size_t num_data_points, size_t num_clusters):
        # here we really just want to allocate two dumb arrays, so self.num_clusters and self.num_data_points are valid
        if num_data_points <= 0 or num_clusters <= 0:
            raise ValueError('arguments must be positive')
        self._cluster_assignments = np.empty(num_data_points, dtype=asm_type)
        self._cluster_centers = np.empty(num_clusters, dtype=value_type)


    def __init__(self, size_t num_data_points, size_t num_clusters):
        """
        Construct a Quantizer; this only sets up empty data attributes and performs no actual cluster initialization.
        :param num_data_points: positive int
        :param num_clusters: positive int
        """
        pass    # real work is done in __cinit__, which is guaranteed to be called once before Python level init


    def __reduce__(self):
        """
        Return a special tuple that contains the necessary info/data to reconstruct self; used by copy.copy and
        pickle.dump; more details see https://docs.python.org/2/library/pickle.html#object.__reduce__
        :return: tuple for re-creating this object
        """
        return (create, (self.cluster_assignments, self.cluster_centers)) # note that numpy representations are returned


    def random_init_means(self, value_type_t[:] data):
        """
        Initialize clusters by randomly assigning data to [0, num_clusters), then updating cluster centers by cluster means.
        :param data: 1d memoryview of value_type
        :return: None
        """
        self.assign_random()
        self.update_cluster_centers_means(data)


    def evenly_init_means(self, value_type_t[:] data):
        """
        Initialize cluster centers to evenly span the range of data; then assign to closest centers.
        :param data: 1d memoryview of value_type
        :return: None
        """
        self._cluster_centers = np.linspace(np.min(data), np.max(data), num=self.num_clusters, dtype=value_type)


    def assign_random(self):
        """
        Assigns data to [0, num_clusters) using uniform discrete distribution; only modifies cluster_assignments.
        :return:
        """
        self._cluster_assignments = np.random.randint(low=0, high=self.num_clusters, size=self.num_data_points, dtype=asm_type)


    def assign_closest(self, value_type_t[:] data, str distance='Euclidean'):
        """
        Primitive method that changes assignments to closest cluster centers; by default l1 distance is used
        for efficiency.
        :param data: 1d memoryview of value_type
        :param distance: str option
        :return: None
        """
        cdef size_t N=data.shape[0], i, K=self._cluster_centers.shape[0], argmin=0
        cdef value_type_t d, min_dist
        if distance=='Euclidean':
            for i in range(N):
                d = data[i]
                min_dist = inf
                for k in range(K):
                    dist = fabs(d - self._cluster_centers[k])
                    if dist < min_dist:
                        min_dist = dist
                        argmin = k
                self._cluster_assignments[i] = argmin
        else:
            raise NotImplementedError('may only use Euclidean distance for now')

    # condense_clusters?

    def update_cluster_centers_sums(self, value_type_t[:] data):
        """
        Update cluster centers to be cluster sums, based on current cluster_assignments; no index bounds checking!
        :param data: 1d memoryview of value_type
        :return: None
        """
        cdef:
            size_t N=data.shape[0], i, a, K=self._cluster_centers.shape[0]
            vector[prec_type] cluster_sums = vector[prec_type](K)

        for i in range(N):
            a = self._cluster_assignments[i]
            cluster_sums[a] += data[i]

        for k in range(K):
            self._cluster_centers[k] = cluster_sums[k]


    def update_cluster_centers_means(self, value_type_t[:] data):
        """
        Update cluster centers to be cluster means, based on current cluster_assignments; no index bounds checking!
        :param data: 1d memoryview of value_type
        :return: None
        """
        cdef size_t N=data.shape[0], i, K=self._cluster_centers.shape[0], k, a, c
        cdef vector[size_t] cluster_sizes = vector[size_t](K)   # default init to all zeros; stack allocated placeholder
        cdef vector[prec_type] cluster_sums = vector[prec_type](K)
        self._cluster_centers[:] = 0

        for i in range(N):
            a = self._cluster_assignments[i]
            cluster_sizes[a] += 1
            cluster_sums[a] += data[i]

        for k in range(K):
            c = cluster_sizes[k]
            if c == 0:
                self._cluster_centers[k] = float('inf')
            else:
                self._cluster_centers[k] = cluster_sums[k] / cluster_sizes[k]


    def update_quantization(self, value_type_t[:] quantized_data):
        """
        Assign elements in quantized_data to their respective cluster centers based on cluster_assignments of data.
        Basically np.take(cluster_centers, cluster_assignments, out=quantized_data) but a few times faster.
        :param quantized_data: 1d memoryview of value_type, same length as num_data_points
        :return: None
        """
        cdef size_t N=self._cluster_assignments.shape[0], i, a
        for i in range(N):
            a = self._cluster_assignments[i]
            quantized_data[i] = self._cluster_centers[a]


    def kmeans_cluster(self, value_type_t[:] data, size_t max_iterations=100):
        """
        Run 1d iterative k-means algorithm for at most max_iterations or till convergence. This method is typically
        at least an order of magnitude faster than scipy.cluster.vq.kmeans(data, K, iter=max_iterations, check_finite=False)
        and uses less working memory. Does its own cluster initialization. Strict EM steps, so it may end up with empty clusters
        with largest cluster ids, whose centers are NaN; this can be fixed by calling condense() to remove empty clusters.
        :param data: 1d memoryview of value_type
        :param max_iterations:
        :return: actual number of iterations run
        """
        return cpp_kmeans_cluster(&data[0], &self._cluster_assignments[0], self.num_data_points, &self._cluster_centers[0], self.num_clusters, max_iterations)


    def kmeans_cluster_lazy(self, value_type_t[:] data, size_t max_iterations=100):
        """
        Run 1d iterative k-means algorithm for at most max_iterations or till convergence, using current centers. This method is typically
        at least an order of magnitude faster than scipy.cluster.vq.kmeans(data, K, iter=max_iterations, check_finite=False)
        and uses less working memory. Does its own cluster initialization. Strict EM steps, so it may end up with empty clusters
        with largest cluster ids, whose centers are NaN; this can be fixed by calling condense() to remove empty clusters.
        :param data: 1d memoryview of value_type
        :param max_iterations:
        :return: actual number of iterations run
        """
        return cpp_kmeans_cluster_lazy(&data[0], &self._cluster_assignments[0], self.num_data_points, &self._cluster_centers[0], self.num_clusters, max_iterations)


    def get_cluster_sizes(self):
        """
        Get cluster sizes corresponding to cluster_centers.
        :return: numpy array that is the same length as num_clusters
        """
        return np.bincount(self.cluster_assignments, minlength=self.num_clusters)


    @property
    def num_nonempty_clusters(self):
        """
        :return: number of clusters with > 0 elements according to cluster_assignments; no side effects
        """
        return np.unique(self.cluster_assignments).size
