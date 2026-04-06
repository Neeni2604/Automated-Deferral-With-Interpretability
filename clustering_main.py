"""
Cluster error instances by SAE features, then evaluate cluster membership
as a deferral signal. Compares against Phase 3 classifier results.
"""

import numpy as np
from sklearn.preprocessing import normalize
from clustering import (
    load_all_data,
    reduce_dimensions,
    find_optimal_k,
    cluster_errors,
    profile_clusters,
    compute_cluster_distance_scores,
    evaluate_cluster_deferral,
    print_cluster_profiles,
    print_deferral_comparison,
    save_phase4_results,
)


DATA_DIR = "data/"
PCA_COMPONENTS = 50
K_RANGE = range(2, 10)
K = 3   # set manually after looking at elbow output, or leave None
COVERAGES = [0.2, 0.3, 0.4]
N_TOP_FEATURES = 10


def main():
    print("=" * 60)
    print("STEP 1 - Load data")
    print("=" * 60)
    data = load_all_data(DATA_DIR)

    feature_matrix = data["feature_matrix"]
    labels = data["labels"]
    instance_ids = data["instance_ids"]

    print("\n" + "=" * 60)
    print("STEP 2 - Separate error instances")
    print("=" * 60)
    error_mask = ~labels
    error_features = feature_matrix[error_mask]

    print(f" Error instances : {error_mask.sum()}")
    print(f" Correct instances : {labels.sum()}")

    print("\n" + "=" * 60)
    print("STEP 3 - Normalize binary activation patterns")
    print("=" * 60)

    X_error_binary = (error_features > 0).astype(float)
    X_error_norm = normalize(X_error_binary, norm='l2')

    X_all_binary = (feature_matrix > 0).astype(float)
    X_all_norm = normalize(X_all_binary, norm='l2')

    print(f"  Avg active features per instance: {X_error_binary.sum(axis=1).mean():.1f}")

    print("\n" + "=" * 60)
    print("STEP 4 - Find optimal k")
    print("=" * 60)
    best_k = find_optimal_k(X_error_norm, k_range=K_RANGE)
    k = K if K is not None else best_k
    print(f"\nUsing k={k}")

    print("\n" + "=" * 60)
    print(f"STEP 5 - Cluster error instances (k={k})")
    print("=" * 60)
    assignments, centroids = cluster_errors(X_error_norm, k=k)

    print("\n" + "=" * 60)
    print("STEP 6 - Profile clusters")
    print("=" * 60)

    correct_features = feature_matrix[labels]

    profiles = profile_clusters(
        assignments = assignments,
        error_features = error_features,
        correct_features = correct_features,
        n_total_errors = error_mask.sum(),
        n_top_features = N_TOP_FEATURES,
    )
    for profile in profiles:
        profile.centroid = centroids[profile.cluster_id]

    

    print("\n" + "=" * 60)
    print("STEP 7 - Cluster membership deferral")
    print("=" * 60)
    min_distances = compute_cluster_distance_scores(X_all_norm, centroids)
    cluster_results, profiles = evaluate_cluster_deferral(
        min_distances  = min_distances,
        labels = labels,
        profiles = profiles,
        assignments = assignments,
        X_all_reduced = X_all_norm,
        centroids = centroids,
        coverages = COVERAGES,
    )

    print_cluster_profiles(profiles)

    print("\n" + "=" * 60)
    print("STEP 8 - Compare against Phase 3")
    print("=" * 60)
    print_deferral_comparison(
        cluster_results = cluster_results,
        phase3_results_path = DATA_DIR + "phase3_results.json",
    )

    print("\n" + "=" * 60)
    print("STEP 9 - Save")
    print("=" * 60)
    save_phase4_results(
        profiles = profiles,
        cluster_results = cluster_results,
        assignments = assignments,
        centroids = centroids,
        data_dir = DATA_DIR,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()