"""
Cluster SAE features of wrong instances, then use cluster membership
as a deferral signal.
"""

import json
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterProfile:
    cluster_id: int
    size: int
    coverage: float   # fraction of total errors in this cluster
    error_ratio: float   # out of nearby instances (correct+wrong), how many are wrong
    top_features: list[tuple[int, float]]  # (feature_idx, mean_activation)
    centroid: np.ndarray


def load_all_data(data_dir: str = "data/") -> dict:
    feature_matrix = np.load(data_dir + "feature_matrix.npy")
    labels = np.load(data_dir + "labels.npy")

    with open(data_dir + "instance_ids.json") as f:
        instance_ids = json.load(f)

    def _load_ids(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(d["id"] for d in data)

    train_ids = _load_ids(data_dir + "train.json")
    val_ids = _load_ids(data_dir + "val.json")
    test_ids = _load_ids(data_dir + "test.json")

    print(f"Loaded: feature_matrix={feature_matrix.shape}, labels={labels.shape}")
    print(f" Correct: {labels.sum()} / {len(labels)} ({100*labels.mean():.1f}%)")
    print(f" Errors: {(~labels).sum()} / {len(labels)} ({100*(~labels).mean():.1f}%)")

    return {
        "feature_matrix": feature_matrix,
        "labels": labels,
        "instance_ids": instance_ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }


def reduce_dimensions(
    feature_matrix: np.ndarray,
    n_components: int = 50,
    scaler: StandardScaler = None,
    pca: PCA = None,
) -> tuple[np.ndarray, StandardScaler, PCA]:
    # 16k dimensions is too high for kmeans to work well so we reduce first
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)
    else:
        X_scaled = scaler.transform(feature_matrix)

    if pca is None:
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_.sum()
        print(f"PCA: {n_components} components explain {100 * explained:.1f}% of variance")
    else:
        X_reduced = pca.transform(X_scaled)

    return X_reduced, scaler, pca


def find_optimal_k(
    X_error_reduced: np.ndarray,
    k_range: range = range(2, 10),
) -> int:
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_error_reduced)
        inertias.append(km.inertia_)
        print(f" k={k}: inertia={km.inertia_:.2f}")

    drops = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    best_k = list(k_range)[np.argmax(drops) + 1]
    print(f" Suggested k: {best_k}")
    return best_k


def cluster_errors(
    X_error_reduced: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    assignments = km.fit_predict(X_error_reduced)
    centroids = km.cluster_centers_

    for c in range(k):
        n = (assignments == c).sum()
        print(f" Cluster {c}: {n} error instances")

    return assignments, centroids


def profile_clusters(
    assignments: np.ndarray,
    error_features: np.ndarray,
    correct_features: np.ndarray,
    n_total_errors: int,
    n_top_features: int = 10,
) -> list[ClusterProfile]:
    k = assignments.max() + 1
    profiles = []
    correct_mean_global = correct_features.mean(axis=0)

    for c in range(k):
        mask = assignments == c
        cluster_feats = error_features[mask]
        cluster_size = mask.sum()

        mean_activations = cluster_feats.mean(axis=0)

        differential = mean_activations - correct_mean_global

        top_idxs = np.argsort(differential)[::-1][:n_top_features]
        top_features = [(int(idx), float(differential[idx])) for idx in top_idxs]

        profiles.append(ClusterProfile(
            cluster_id = c,
            size = int(cluster_size),
            coverage = float(cluster_size / n_total_errors),
            error_ratio = 0.0,
            top_features = top_features,
            centroid = np.zeros(1),
        ))

    return profiles


def compute_cluster_distance_scores(
    X_all_reduced: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    # small distance = close to a failure cluster = likely an error
    distances = np.array([
        np.linalg.norm(X_all_reduced - centroid, axis=1)
        for centroid in centroids
    ]).T
    return distances.min(axis=1)


def evaluate_cluster_deferral(
    min_distances:  np.ndarray,
    labels: np.ndarray,
    profiles: list[ClusterProfile],
    assignments: np.ndarray,
    X_all_reduced: np.ndarray,
    centroids: np.ndarray,
    coverages: list[float] = [0.2, 0.3, 0.4],
    threshold_pct: float = 0.5,
) -> tuple[list[dict], list[ClusterProfile]]:
    # negate distance: closer to a failure cluster = higher error score
    error_scores = -min_distances
    error_labels = (~labels).astype(int)

    auroc = roc_auc_score(error_labels, error_scores)
    print(f"\nCluster-based deferral AUROC: {auroc:.4f}")

    results = []
    for cov in coverages:
        n = len(labels)
        n_defer = max(1, int(n * cov))
        ranked = np.argsort(error_scores)[::-1]
        defer_idx = set(ranked[:n_defer])
        keep_idx = [i for i in range(n) if i not in defer_idx]

        deferred_labels = labels[list(defer_idx)]
        precision = float((~deferred_labels).mean())
        accuracy_remaining = float(labels[keep_idx].mean()) if keep_idx else 0.0

        threshold = np.sort(error_scores)[::-1][n_defer - 1]
        predictions = (error_scores >= threshold).astype(int)
        f1 = f1_score(error_labels, predictions, zero_division=0)

        results.append({
            "name": "Cluster Membership",
            "coverage": cov,
            "auroc": auroc,
            "f1_defer": f1,
            "precision": precision,
            "accuracy_remaining": accuracy_remaining,
            "n_deferred": n_defer,
        })

    for profile in profiles:
        c = profile.cluster_id
        centroid = centroids[c]
        dists = np.linalg.norm(X_all_reduced - centroid, axis=1)
        # use median distance among error instances in this cluster as radius
        error_dists = dists[~labels]
        radius = np.median(error_dists)
        near = dists <= radius
        if near.sum() > 0:
            profile.error_ratio = float((~labels[near]).mean())
        else:
            profile.error_ratio = 0.0

    return results, profiles


def print_cluster_profiles(profiles: list[ClusterProfile]) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  FAILURE CLUSTER PROFILES\n{bar}")
    for p in profiles:
        print(f"\n  Cluster {p.cluster_id}:")
        print(f" Size : {p.size} error instances")
        print(f" Coverage : {p.coverage:.1%} of all errors")
        print(f" Error ratio : {p.error_ratio:.1%} of nearby instances are errors")
        print(f" Top SAE features:")
        for feat_idx, mean_act in p.top_features[:5]:
            print(f" Feature {feat_idx:>6}  mean_activation={mean_act:.4f}")
    print(f"\n{bar}\n")


def print_deferral_comparison(
    cluster_results: list[dict],
    phase3_results_path: str = "data/phase3_results.json",
) -> None:
    with open(phase3_results_path) as f:
        phase3 = json.load(f)

    p3_lookup = {(r["name"], r["coverage"]): r for r in phase3}

    bar = "=" * 78
    print(f"\n{bar}")
    print(f" CLUSTER DEFERRAL vs PHASE 3 CLASSIFIER COMPARISON")
    print(f"{bar}")
    print(f"  {'Coverage':>8}  {'System':<22}  {'AUROC':>6}  "
          f"{'F1':>6}  {'Prec@k':>7}  {'Acc Rem':>8}")
    print(f"  {'-'*8}  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}")

    coverages = sorted(set(r["coverage"] for r in cluster_results))
    for cov in coverages:
        cr = next(r for r in cluster_results if r["coverage"] == cov)
        print(f"  {cov:>8.0%}  {'Cluster Membership':<22}  "
              f"{cr['auroc']:>6.4f}  {cr['f1_defer']:>6.4f}  "
              f"{cr['precision']:>7.4f}  {cr['accuracy_remaining']:>8.4f}")

        for name in ["SAE MLP", "SAE LogReg", "SAE RF", "Confidence", "Random"]:
            key = (name, cov)
            if key in p3_lookup:
                r = p3_lookup[key]
                print(f" {cov:>8.0%}  {name:<22}  "
                      f"{r['auroc']:>6.4f}  {r['f1_defer']:>6.4f}  "
                      f"{r['precision']:>7.4f}  {r['accuracy_remaining']:>8.4f}")
        print()
    print(bar)


def save_phase4_results(
    profiles: list[ClusterProfile],
    cluster_results: list[dict],
    assignments: np.ndarray,
    centroids: np.ndarray,
    data_dir: str = "data/",
) -> None:
    import os, dataclasses
    os.makedirs(data_dir, exist_ok=True)

    np.savez(
        data_dir + "clusters.npz",
        centroids = centroids,
        assignments = assignments,
    )

    def _profile_to_dict(p):
        d = dataclasses.asdict(p)
        d["centroid"] = p.centroid.tolist()
        return d

    with open(data_dir + "phase4_results.json", "w") as f:
        json.dump({
            "cluster_profiles": [_profile_to_dict(p) for p in profiles],
            "deferral_results": cluster_results,
        }, f, indent=2)

    print(f"Saved clusters -> {data_dir}clusters.npz")
    print(f"Saved results -> {data_dir}phase4_results.json")
