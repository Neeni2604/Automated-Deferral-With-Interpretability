"""
Trains classifiers over SAE feature vectors to predict model errors,
then evaluates them as deferral systems against confidence-score baselines.
"""

import json
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, classification_report,)
from sklearn.pipeline import Pipeline


@dataclass
class DeferralResult:
    """
    Holds evaluation results for one deferral system at one coverage level
    """
    name: str
    coverage: float
    precision: float
    accuracy_remaining: float
    n_deferred: int
    auroc: float
    f1_defer: float


def load_phase2_outputs(data_dir: str = "data/") -> tuple:
    """
    Load feature matrix, labels, and instance ids from Phase 2 outputs.
    Also loads log_probs from labeled_instances.json for the confidence baseline.
    """
    feature_matrix = np.load(data_dir + "feature_matrix.npy")
    labels = np.load(data_dir + "labels.npy")

    with open(data_dir + "instance_ids.json") as f:
        instance_ids = json.load(f)

    with open(data_dir + "labeled_instances.json", encoding="utf-8") as f:
        instances_raw = json.load(f)

    id_to_logprob = {
        inst["id"]: float(np.mean(inst["log_probs"]))
        for inst in instances_raw
        if inst.get("log_probs")
    }

    log_probs = np.array([
        id_to_logprob.get(id_, 0.0) for id_ in instance_ids
    ])

    print(f"Loaded: feature_matrix={feature_matrix.shape}, " f"labels={labels.shape}, log_probs={log_probs.shape}")
    print(f"  Correct: {labels.sum()} / {len(labels)} " f"({100 * labels.mean():.1f}%)")

    return feature_matrix, labels, log_probs, instance_ids


def split_by_ids(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    log_probs: np.ndarray,
    instance_ids: list[str],
    train_ids_path: str = "data/train.json",
    val_ids_path: str = "data/val.json",
    test_ids_path: str = "data/test.json",
) -> dict:
    """
    Split feature matrix into train/val/test using the same splits saved
    in Phase 1, matching by instance id so splits are consistent.
    """
    def _load_ids(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(d["id"] for d in data)

    train_ids = _load_ids(train_ids_path)
    val_ids = _load_ids(val_ids_path)
    test_ids = _load_ids(test_ids_path)

    id_to_idx = {id_: i for i, id_ in enumerate(instance_ids)}

    splits = {}
    for name, id_set in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        idxs = [id_to_idx[id_] for id_ in id_set if id_ in id_to_idx]
        idxs = sorted(idxs)
        splits[name] = {
            "X": feature_matrix[idxs],
            "y": labels[idxs],
            "logprobs": log_probs[idxs],
            "ids": [instance_ids[i] for i in idxs],
        }
        n_correct = splits[name]["y"].sum()
        print(f"  {name}: {len(idxs)} instances, "
              f"correct={n_correct} ({100 * n_correct / max(len(idxs), 1):.1f}%)")

    return splits


# Classifiers
def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Train a logistic regression classifier on SAE features
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",  # important: errors are the minority class
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)
    print("Logistic Regression trained.")
    return pipeline


def train_mlp(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    from sklearn.utils import resample

    X_correct   = X_train[y_train == True]
    y_correct   = y_train[y_train == True]
    X_incorrect = X_train[y_train == False]
    y_incorrect = y_train[y_train == False]

    # Oversample the minority class to balance
    n_majority = max(len(X_correct), len(X_incorrect))
    if len(X_correct) < len(X_incorrect):
        X_correct, y_correct = resample(
            X_correct, y_correct, n_samples=n_majority, random_state=42
        )
    else:
        X_incorrect, y_incorrect = resample(
            X_incorrect, y_incorrect, n_samples=n_majority, random_state=42
        )

    X_balanced = np.vstack([X_correct, X_incorrect])
    y_balanced = np.concatenate([y_correct, y_incorrect])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 64),
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )),
    ])

    pipeline.fit(X_balanced, y_balanced)
    print("MLP trained.")
    return pipeline


# Evaluation

def get_error_scores_from_classifier(
    pipeline,
    X: np.ndarray,
) -> np.ndarray:
    """
    Get P(incorrect) scores from a fitted sklearn classifier pipeline.
    Higher score = model more likely wrong = stronger case to defer.
    """
    proba = pipeline.predict_proba(X)

    classes = list(pipeline.named_steps["clf"].classes_)
    incorrect_class_idx = classes.index(False)

    return proba[:, incorrect_class_idx]


def get_error_scores_from_logprobs(log_probs: np.ndarray) -> np.ndarray:
    """
    Convert log-probs to error scores for the confidence baseline.
    Lower log-prob -> less confident -> more likely to defer.
    Negate so higher score = more likely wrong
    """
    return -log_probs


def evaluate_deferral(
    name: str,
    error_scores: np.ndarray,
    labels: np.ndarray,
    coverage: float = 0.3,
) -> DeferralResult:
    """
    Evaluate a deferral system at a given coverage level
    """
    n = len(labels)
    n_defer = max(1, int(n * coverage))

    # Rank by error score descending, defer the top n_defer
    ranked_idx = np.argsort(error_scores)[::-1]
    defer_idx = set(ranked_idx[:n_defer])
    keep_idx = [i for i in range(n) if i not in defer_idx]

    # Precision of deferral: fraction of deferred that are actually wrong
    deferred_labels = labels[list(defer_idx)]
    precision = float((~deferred_labels).mean()) if len(deferred_labels) > 0 else 0.0

    # Accuracy on kept instances (ones automated-deferral handles)
    kept_labels = labels[keep_idx]
    accuracy_remaining = float(kept_labels.mean()) if len(kept_labels) > 0 else 0.0

    # AUROC: predict errors (label=False → class 1 in error prediction)
    error_labels = (~labels).astype(int)
    auroc = roc_auc_score(error_labels, error_scores)

    # F1 on defer class
    threshold = np.sort(error_scores)[::-1][n_defer - 1]
    predictions = (error_scores >= threshold).astype(int)
    f1 = f1_score(error_labels, predictions, zero_division=0)

    return DeferralResult(
        name = name,
        coverage = coverage,
        precision = precision,
        accuracy_remaining = accuracy_remaining,
        n_deferred = n_defer,
        auroc = auroc,
        f1_defer = f1,
    )


def evaluate_all_systems(
    splits: dict,
    lr_pipeline,
    mlp_pipeline,
    coverages: list[float] = [0.2, 0.3, 0.4],
) -> list[DeferralResult]:
    """
    Evaluate all deferral systems at multiple coverage levels on the test set.
    Systems evaluated: SAE LR, SAE MLP, Confidence baseline (logprob), Random baseline
    """
    X_test = splits["test"]["X"]
    y_test = splits["test"]["y"]
    lp_test = splits["test"]["logprobs"]

    # Error scores for each system
    lr_scores = get_error_scores_from_classifier(lr_pipeline, X_test)
    mlp_scores = get_error_scores_from_classifier(mlp_pipeline, X_test)
    conf_scores = get_error_scores_from_logprobs(lp_test)
    random_scores = np.random.default_rng(42).random(len(y_test))

    results = []
    for cov in coverages:
        results.append(evaluate_deferral("SAE LogReg", lr_scores, y_test, cov))
        results.append(evaluate_deferral("SAE MLP", mlp_scores, y_test, cov))
        results.append(evaluate_deferral("Confidence", conf_scores, y_test, cov))
        results.append(evaluate_deferral("Random", random_scores, y_test, cov))

    return results


# Printing
def print_results_table(results: list[DeferralResult]) -> None:
    """
    Print a formatted comparison table of all deferral systems
    """
    bar = "=" * 78
    print(f"\n{bar}")
    print(f"  DEFERRAL EVALUATION RESULTS")
    print(f"{bar}")
    print(f"  {'Coverage':>8}  {'System':<18}  {'AUROC':>6}  {'F1':>6}  "
          f"{'Prec@k':>7}  {'Acc Rem':>8}")
    print(f"  {'-'*8}  {'-'*18}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}")

    coverages = sorted(set(r.coverage for r in results))
    for cov in coverages:
        cov_results = [r for r in results if r.coverage == cov]
        for r in cov_results:
            print(
                f"  {r.coverage:>8.0%}  {r.name:<18}  "
                f"{r.auroc:>6.4f}  {r.f1_defer:>6.4f}  "
                f"{r.precision:>7.4f}  {r.accuracy_remaining:>8.4f}"
            )
        print()
    print(bar)


def print_top_features(
    lr_pipeline,
    n_top: int = 20,
) -> None:
    """
    Print the SAE feature indices with the highest logistic regression
    coefficients for predicting errors. 
    """
    clf = lr_pipeline.named_steps["clf"]
    classes = list(clf.classes_)
    incorrect_idx = classes.index(False)
    coefs = clf.coef_[incorrect_idx]   # (sae_dim,)

    top_positive = np.argsort(coefs)[::-1][:n_top]   # features -> predicts error
    top_negative = np.argsort(coefs)[:n_top]          # features -> predicts correct

    print(f"\n{'='*58}")
    print(f"  TOP {n_top} SAE FEATURES PREDICTING ERRORS")
    print(f"{'='*58}")
    print(f"  {'Feature idx':>12}  {'Coefficient':>12}")
    for idx in top_positive:
        print(f"  {idx:>12}  {coefs[idx]:>12.4f}")

    print(f"\n  TOP {n_top} SAE FEATURES PREDICTING CORRECT ANSWERS")
    print(f"  {'Feature idx':>12}  {'Coefficient':>12}")
    for idx in top_negative:
        print(f"  {idx:>12}  {coefs[idx]:>12.4f}")
    print(f"{'='*58}\n")


def save_results(results: list[DeferralResult], path: str = "data/phase3_results.json") -> None:
    """Save deferral results to JSON for later reference."""
    import os, dataclasses
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump([dataclasses.asdict(r) for r in results], f, indent=2)
    print(f"Results saved → {path}")