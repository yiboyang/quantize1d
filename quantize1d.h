#include <iostream>
#include <cstdlib>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <parallel/algorithm>
#include <unordered_set>

/* Fast 1d quantization methods
 * Currently supports k-means. dp-means would require std::vector for cluster_centers (see original C++ implementation)
 * k-means uses multithreaded (openMP) sort, as it tends to dominate the runtime.
 * https://arxiv.org/abs/1806.05355
 * Yibo Yang, Vibhav Gogate
 */

typedef double prec_type;   // high precision type e.g., for std::accumulate; may need long double if input is huge (1e8)

// utility methods
// return a vector of indices in sorted order, based on an array of values
template<typename T>
std::vector<size_t> sort_index(const T *v, size_t v_size) {

    // initialize index vector
    std::vector<size_t> idx(v_size);
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    __gnu_parallel::sort(idx.begin(), idx.end(),
                         [&v](size_t i, size_t j) -> bool { return v[i] < v[j]; });

    // uncomment below for single-threaded sort
    // std::sort(idx.begin(), idx.end(),
    //                      [&v](size_t i, size_t j) -> bool { return v[i] < v[j]; });
    //
    return idx;
}

// overloaded version of the above accepting vectors
template<typename T>
std::vector<size_t> sort_index(const std::vector<T> &v) {
    return sort_index(v.data(), v.size());
}

template<typename T>
void print_arr(T *a, const unsigned size) {
    for (unsigned i = 0; i < size; ++i)
        std::cout << *(a + i) << ' ';
    std::cout << '\n';
}

template<typename T>
void print_vec(const std::vector<T> &v) {
    for (auto &i: v) {
        std::cout << i << ' ';
    }
    std::cout << '\n';
}

// Fast iterative 1d kmeans:
// Strictly executes the E and M steps of kmeans, but operates on entire partitions of data points in 1D
// for O(K logN) E steps and O(N) M-steps; may end up with some empty clusters and have NaN as cluster centers;
// return the number of iterations that were actually run.
// The clusters are heuristically initialized to have equal cluster sizes.
// Always outputs cluster centers that are in sorted order.
template<typename asm_type, typename value_type>
size_t kmeans_cluster(value_type *data, asm_type *cluster_assignments, size_t num_data_points,
                      value_type *cluster_centers, size_t num_clusters, size_t max_iterations) {
    if (num_clusters == 1) {
        std::fill(cluster_assignments, cluster_assignments + num_data_points, 0);
        prec_type data_sum = std::accumulate(data, data + num_data_points, 0.0);
        value_type mean = static_cast<value_type>(data_sum / num_data_points);
        cluster_centers[0] = mean;
        return 0;
    }

    // need high precision for cluster_centers; will copy back later
    auto cluster_centers_v = std::vector<prec_type>(num_clusters, 0.0);

    // setup sorted data
    std::vector<size_t> idx = sort_index(data, num_data_points); // idx addresses the original data in sorted order
    std::vector<value_type> sorted_data(num_data_points);   // copy of data that is sorted
    for (unsigned i = 0; i < num_data_points; ++i)
        sorted_data[i] = data[idx[i]];

    // initialize by partitioning sorted data into K roughly equally-sized partitions; cluster_sizes
    // implicitly determines cluster_assignments; each cluster roughly the same size
    auto cluster_sizes = std::vector<size_t>(num_clusters, num_data_points / num_clusters);
    for (size_t i = 0; i < (num_data_points % num_clusters); i++)   // divvy up the remainder
        ++cluster_sizes[i];

    // initialize the boundaries based on partition sizes;
    // the kth boundary element marks the beginning of the kth partition (cluster); e.g. the 0th sorted data point
    // trivially marks the beginning of the 0th cluster/partition;
    // for convenience, I use an extra boundary at the end (whose position is equal to num_data_point) to mark
    // the end of the last cluster
    std::vector<size_t> boundaries(num_clusters + 1);
    size_t position = 0;
    for (size_t k = 0; k < num_clusters + 1; ++k) { // boundaries[num_clusters] == num_data_points
        boundaries[k] = position;
        position += cluster_sizes[k];
    }

    // update centers/sums based on assignment boundaries
    auto cluster_sums = std::vector<prec_type>(num_clusters, 0.0);
    for (size_t k = 0; k < num_clusters; ++k) {
        size_t beg = boundaries[k], end = boundaries[k + 1];
        for (size_t j = beg; j < end; ++j)
            cluster_sums[k] += sorted_data[j];

        cluster_centers_v[k] = cluster_sums[k] / (end - beg);
    }


    // assume Euclidean (geometric) distance is used
    unsigned t;
    const auto sorted_data_iter = sorted_data.begin();  // base iterator
    auto new_boundaries = boundaries;
    for (t = 0; t < max_iterations; t++) {
        bool assignments_changed = false;
        // reassign clusters (by redrawing boundaries); E-step
        // in the kth iter, redraw the boundary (beginning) of the (k+1)th partition
        for (size_t k = 0; k < (num_clusters - 1); k++) {
            auto left_center = cluster_centers_v[k];
            auto right_center = cluster_centers_v[k + 1];
            size_t left_bound = boundaries[k];  // beginning of the kth cluster
            size_t right_bound = boundaries[k + 2];   // beginning of the (k+2)th cluster (one past the end of (k+1)th)

            // points lying in between left_center and right_center belongs to whichever is closer;
            // simply find the mid-point, and assign points in [left_center, mid) to cluster k, and
            // points in [mid, right_center) cluster k+1, i.e. the mid point marks the beginning of the (k+1)th
            // cluster; note there's no need to adjust points either to the left of the smallest center or to the
            // right of the largest center
            auto mid = (left_center + right_center) / 2.0;

            // binary-search between left_bound and right_bound for efficiency
            auto mid_iter = lower_bound(sorted_data_iter + left_bound, sorted_data_iter + right_bound, mid);
            auto mid_idx = static_cast<unsigned long>(mid_iter - sorted_data_iter); // guaranteed to be non-negative

            if (boundaries[k + 1] != mid_idx) {
                new_boundaries[k + 1] = mid_idx;
                assignments_changed = true;
            }
        }

        if (!assignments_changed)   // converged; more iterations won't help
            break;

        // M-step
        // update cluster sums based on changes in partitioning
        for (size_t k = 0; k < num_clusters - 1; ++k) {
            size_t prev_end = boundaries[k + 1], new_end = new_boundaries[k + 1];
            value_type inc_sum; // how much the kth cluster expanded (and how much the (k+1)th shrunk)
            if (new_end > prev_end) {
                inc_sum = std::accumulate(sorted_data_iter + prev_end, sorted_data_iter + new_end, 0.0);
            } else {    // note the negative sign
                inc_sum = -std::accumulate(sorted_data_iter + new_end, sorted_data_iter + prev_end, 0.0);
            }
            cluster_sums[k] += inc_sum;
            cluster_sums[k + 1] -= inc_sum;
        }
        // update cluster centers
        for (size_t k = 0; k < num_clusters; ++k) {
            size_t cluster_size = new_boundaries[k + 1] - new_boundaries[k];
            cluster_centers_v[k] = cluster_sums[k] / cluster_size;
        }

        boundaries = new_boundaries;
    }

    // map the partitioning scheme back to cluster_assignments
    for (size_t k = 0; k < num_clusters; ++k) {
        size_t beg = boundaries[k], end = boundaries[k + 1];
        for (size_t j = beg; j < end; ++j)
            cluster_assignments[idx[j]] = static_cast<asm_type>(k);
    }

    // copy to original cluster_centers
    std::copy(cluster_centers_v.begin(), cluster_centers_v.end(), cluster_centers);

    return t;
}


// Fast iterative 1d kmeans with warm start:
// Strictly executes the E and M steps of kmeans, but operates on entire partitions of data points in 1D
// for O(K logN) E steps and O(N) M-steps; may end up with some empty clusters and have NaN as cluster centers;
// return the number of iterations that were actually run.
// The clusters are initialized based on given values of cluster centers and are assumed valid (hence "lazy").
// Always outputs cluster centers that are in sorted order.
template<typename asm_type, typename value_type>
size_t kmeans_cluster_lazy(value_type *data, asm_type *cluster_assignments, size_t num_data_points,
                           value_type *cluster_centers, size_t num_clusters, size_t max_iterations) {
    if (num_clusters == 1) {
        std::fill(cluster_assignments, cluster_assignments + num_data_points, 0);
        double data_sum = std::accumulate(data, data + num_data_points, 0.0);
        value_type mean = static_cast<value_type>(data_sum / num_data_points);
        cluster_centers[0] = mean;
        return 0;
    }

    // need high precision for cluster_centers; will copy back later
    std::vector<prec_type> cluster_centers_v;
    cluster_centers_v.insert(cluster_centers_v.end(), cluster_centers,
                             cluster_centers + num_clusters); // copy over to vec

    // keep the current values of cluster centers, but ensure cluster_centers is sorted (this may cause some data points
    // to switch cluster/assignment labels if the cluster centers have evolved to no longer being sorted
    std::sort(cluster_centers_v.begin(), cluster_centers_v.end());    // recall that raw pointers are also iterators

    // set up sorted data
    std::vector<size_t> idx = sort_index(data, num_data_points); // idx addresses the original data in sorted order
    std::vector<value_type> sorted_data(num_data_points);   // copy of data that is sorted
    for (unsigned i = 0; i < num_data_points; ++i)
        sorted_data[i] = data[idx[i]];



    // initialize the cluster boundaries (therefore, assignments) with a E-step, based on existing cluster center values
    // the kth boundary element marks the beginning of the kth partition (cluster); e.g. the 0th sorted data point
    // trivially marks the beginning of the 0th cluster/partition;
    // for convenience, I use an extra boundary at the end (whose position is equal to num_data_point) to mark
    // the end of the last cluster
    std::vector<size_t> boundaries(num_clusters + 1);
    // in kth iter, determine boundary (beginning idx) of k+1th cluster
    for (size_t k = 0; k < (num_clusters - 1); k++) {
        auto left_center = cluster_centers_v[k];
        auto right_center = cluster_centers_v[k + 1];

        auto mid = (left_center + right_center) / 2.0;

        // search entire data array for correctness
        auto mid_iter = lower_bound(sorted_data.begin(), sorted_data.end(), mid);
        auto mid_idx = static_cast<unsigned long>(mid_iter - sorted_data.begin()); // guaranteed to be non-negative

        boundaries[k + 1] = mid_idx;
    }
    boundaries[num_clusters] = num_data_points; // important

    // update centers/sums based on assignment boundaries
    auto cluster_sums = std::vector<prec_type>(num_clusters, 0.0);
    std::fill(cluster_centers_v.begin(), cluster_centers_v.end(), 0.0);
    for (size_t k = 0; k < num_clusters; ++k) {
        size_t beg = boundaries[k], end = boundaries[k + 1];
        for (size_t j = beg; j < end; ++j)
            cluster_sums[k] += sorted_data[j];

        cluster_centers_v[k] = cluster_sums[k] / (end - beg);
    }


    // assume Euclidean (geometric) distance is used
    unsigned t;
    const auto sorted_data_iter = sorted_data.begin();  // base iterator
    auto new_boundaries = boundaries;
    for (t = 0; t < max_iterations; t++) {
        bool assignments_changed = false;
        // reassign clusters (by redrawing boundaries); E-step
        // in the kth iter, redraw the boundary (beginning) of the (k+1)th partition
        for (size_t k = 0; k < (num_clusters - 1); k++) {
            auto left_center = cluster_centers_v[k];
            auto right_center = cluster_centers_v[k + 1];
            size_t left_bound = boundaries[k];  // beginning of the kth cluster
            size_t right_bound = boundaries[k + 2];   // beginning of the (k+2)th cluster (one past the end of (k+1)th)

            // points lying in between left_center and right_center belongs to whichever is closer;
            // simply find the mid-point, and assign points in [left_center, mid) to cluster k, and
            // points in [mid, right_center) cluster k+1, i.e. the mid point marks the beginning of the (k+1)th
            // cluster; note there's no need to adjust points either to the left of the smallest center or to the
            // right of the largest center
            auto mid = (left_center + right_center) / 2.0;

            // binary-search between left_bound and right_bound for efficiency
            auto mid_iter = lower_bound(sorted_data_iter + left_bound, sorted_data_iter + right_bound, mid);
            auto mid_idx = static_cast<unsigned long>(mid_iter - sorted_data_iter); // guaranteed to be non-negative

            if (boundaries[k + 1] != mid_idx) {
                new_boundaries[k + 1] = mid_idx;
                assignments_changed = true;
            }
        }

        if (!assignments_changed)   // converged; more iterations won't help
            break;

        // M-step
        // update cluster sums based on changes in partitioning
        for (size_t k = 0; k < num_clusters - 1; ++k) {
            size_t prev_end = boundaries[k + 1], new_end = new_boundaries[k + 1];
            value_type inc_sum; // how much the kth cluster expanded (and how much the (k+1)th shrunk)
            if (new_end > prev_end) {
                inc_sum = std::accumulate(sorted_data_iter + prev_end, sorted_data_iter + new_end, 0.0);
            } else {    // note the negative sign
                inc_sum = -std::accumulate(sorted_data_iter + new_end, sorted_data_iter + prev_end, 0.0);
            }
            cluster_sums[k] += inc_sum;
            cluster_sums[k + 1] -= inc_sum;
        }
        // update cluster centers
        for (size_t k = 0; k < num_clusters; ++k) {
            size_t cluster_size = new_boundaries[k + 1] - new_boundaries[k];
            cluster_centers_v[k] = cluster_sums[k] / cluster_size;
        }

        boundaries = new_boundaries;
    }

    // map the partitioning scheme back to cluster_assignments
    for (size_t k = 0; k < num_clusters; ++k) {
        size_t beg = boundaries[k], end = boundaries[k + 1];
        for (size_t j = beg; j < end; ++j)
            cluster_assignments[idx[j]] = static_cast<asm_type>(k);
    }

    // copy to original cluster_centers
    std::copy(cluster_centers_v.begin(), cluster_centers_v.end(), cluster_centers);

    return t;
}

