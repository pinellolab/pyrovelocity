import time

import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import memory_usage
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import pairwise_distances


def generate_data_with_correlation(n_cells, n_genes, correlation):
    if correlation not in [-1, 0, 1]:
        raise ValueError("Correlation must be -1, 0, or 1")

    if correlation == 0:
        expression_vectors = np.random.randint(0, 100, size=(n_cells, n_genes))
        temporal_coordinates = np.random.rand(n_cells)
    else:
        expression_vectors = np.zeros((n_cells, n_genes))

        for i in range(1, n_cells):
            noise = np.random.normal(0, 1, n_genes)
            # noise = 0
            expression_vectors[i, :] = expression_vectors[i - 1, :] + 1 + noise

        if correlation == -1:
            temporal_coordinates = np.arange(n_cells - 1, -1, -1)
        else:
            temporal_coordinates = np.arange(n_cells)

    return expression_vectors, temporal_coordinates


def compute_distances_and_correlation(n_cells, n_genes, desired_corr=0):
    start_time_total = time.time()
    expression_vectors, temporal_coordinates = generate_data_with_correlation(
        n_cells, n_genes, desired_corr
    )

    start_time = time.time()
    expression_distances_matrix = pairwise_distances(
        expression_vectors, metric="cityblock", n_jobs=-1
    )
    expression_distances = expression_distances_matrix[
        np.triu_indices(n_cells, k=1)
    ]
    time_dist = time.time() - start_time

    start_time = time.time()
    temporal_differences_matrix = (
        temporal_coordinates - temporal_coordinates.reshape(-1, 1)
    )
    temporal_differences = temporal_differences_matrix[
        np.triu_indices(n_cells, k=1)
    ]
    time_temporal = time.time() - start_time

    start_time = time.time()
    correlation, p_value = spearmanr(expression_distances, temporal_differences)
    time_corr = time.time() - start_time
    total_time = time.time() - start_time_total
    return correlation, p_value, time_dist, time_temporal, time_corr, total_time


correlations = []
p_values = []
memory_usages = []
n_genes_list = []
n_cells_list = []
desired_corr_list = []
total_times = []

for desired_corr in [-1, 0, 1]:
    for n_genes in [200, 2000]:
        for n_cells in [100, 1000, 10000]:
            (
                mem_usage,
                (
                    correlation,
                    p_value,
                    time_dist,
                    time_temporal,
                    time_corr,
                    total_time,
                ),
            ) = memory_usage(
                (
                    compute_distances_and_correlation,
                    (n_cells, n_genes, desired_corr),
                ),
                interval=0.1,
                max_usage=True,
                retval=True,
            )
            print(f"For {n_cells} cells and {n_genes} genes:")
            print(f"Memory usage: {mem_usage} MB")
            print(f"Time for distance: {time_dist} s")
            print(f"Time for temporal differences: {time_temporal} s")
            print(f"Time for correlation: {time_corr} s")
            print(f"Total time: {total_time} s")
            print(f"Correlation: {correlation}")
            print(f"P-value: {p_value:.2e}")
            print()

            correlations.append(correlation)
            p_values.append(p_value)
            memory_usages.append(mem_usage)
            n_genes_list.append(n_genes)
            n_cells_list.append(n_cells)
            desired_corr_list.append(desired_corr)
            total_times.append(total_time)

data = np.column_stack(
    (
        n_genes_list,
        n_cells_list,
        correlations,
        p_values,
        memory_usages,
        total_times,
    )
)
np.savetxt(
    "distance_time_correlation_benchmark.csv",
    data,
    delimiter=",",
    header="n_genes,n_cells,correlations,p_values,memory_usages,total_time",
    comments="",
)

plt.figure(figsize=(10, 10))
plt.subplot(411)
plt.plot(correlations, "o-", color="black", label="Correlations")
plt.xticks(
    range(len(n_genes_list)),
    [
        f"{x}\n{y}\n{z}"
        for x, y, z in zip(n_genes_list, n_cells_list, desired_corr_list)
    ],
)
plt.legend()

plt.subplot(412)
plt.plot(p_values, "o-", color="black", label="P-values")
plt.xticks(
    range(len(n_genes_list)),
    [
        f"{x}\n{y}\n{z}"
        for x, y, z in zip(n_genes_list, n_cells_list, desired_corr_list)
    ],
)
plt.legend()

plt.subplot(413)
plt.plot(memory_usages, "o-", color="black", label="Memory Usage (MB)")
plt.xticks(
    range(len(n_genes_list)),
    [
        f"{x}\n{y}\n{z}"
        for x, y, z in zip(n_genes_list, n_cells_list, desired_corr_list)
    ],
)
plt.legend()

plt.subplot(414)
plt.plot(total_times, "o-", color="black", label="Total Time (s)")
plt.xticks(
    range(len(n_genes_list)),
    [
        f"{x}\n{y}\n{z}"
        for x, y, z in zip(n_genes_list, n_cells_list, desired_corr_list)
    ],
)
plt.legend()

plt.tight_layout()
plt.savefig("distance_time_correlation_benchmark.png")
