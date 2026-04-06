"""
1. Load Phase 2 feature matrix and labels
2. Split into train/val/test (matching Phase 1 splits)
3. Train logistic regression and MLP classifiers on SAE features
4. Evaluate all deferral systems against confidence-score baseline
5. Print results table and top predictive features
6. Save results to disk
"""

import numpy as np
from phase3_classifier import (
    load_phase2_outputs,
    split_by_ids,
    train_logistic_regression,
    train_mlp,
    evaluate_all_systems,
    print_results_table,
    print_top_features,
    save_results,
)


DATA_DIR = "data/"
COVERAGES = [0.2, 0.3, 0.4]   # evaluate at 20%, 30%, 40% deferral rates
RESULTS_PATH = "data/phase3_results.json"


def main():
    print("=" * 60)
    print("STEP 1: Load Phase 2 outputs")
    print("=" * 60)
    feature_matrix, labels, log_probs, instance_ids = load_phase2_outputs(DATA_DIR)


    print("\n" + "=" * 60)
    print("STEP 2: Train / val / test split")
    print("=" * 60)
    splits = split_by_ids(
        feature_matrix = feature_matrix,
        labels = labels,
        log_probs = log_probs,
        instance_ids = instance_ids,
        train_ids_path = DATA_DIR + "train.json",
        val_ids_path = DATA_DIR + "val.json",
        test_ids_path = DATA_DIR + "test.json",
    )

    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val,   y_val   = splits["val"]["X"],   splits["val"]["y"]

    print(f"\n  Training on {len(X_train)} instances, "
          f"validating on {len(X_val)}, "
          f"testing on {len(splits['test']['X'])}")


    print("\n" + "=" * 60)
    print("STEP 3: Train classifiers")
    print("=" * 60)

    print("\nTraining Logistic Regression...")
    lr_pipeline = train_logistic_regression(X_train, y_train)

    print("Training MLP...")
    mlp_pipeline = train_mlp(X_train, y_train)

    # Quick val-set check before full evaluation
    print("\nValidation set sanity check:")
    for name, pipeline in [("LogReg", lr_pipeline), ("MLP", mlp_pipeline)]:
        val_preds = pipeline.predict(X_val)
        val_acc = (val_preds == y_val).mean()
        print(f"  {name} val accuracy: {val_acc:.4f}")


    print("\n" + "=" * 60)
    print("STEP 4: Evaluate deferral systems")
    print("=" * 60)
    results = evaluate_all_systems(
        splits = splits,
        lr_pipeline = lr_pipeline,
        mlp_pipeline = mlp_pipeline,
        coverages = COVERAGES,
    )


    print_results_table(results)

    print_top_features(lr_pipeline, n_top=20)


    save_results(results, RESULTS_PATH)
    print(f"\nPhase 3 complete. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()