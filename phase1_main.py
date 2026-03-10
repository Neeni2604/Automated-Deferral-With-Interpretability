"""
Entry point for Phase 1. Run this file to:
  1. Load ContractNLI dataset
  2. Load Gemma 3 and confirm the GemmaScope 2 SAE loads
  3. Label all instances with robustness-aware correctness labels
     (majority-vote across 5 sampling temperatures)
  4. Collect per-instance confidence scores and log-probs (baseline deferral signals)
  5. Save results and train/val/test splits to disk

Usage:
  python phase1_main.py

Prerequisites:
  pip install -r requirements.txt
  huggingface-cli login   # Gemma 3 is a gated model
"""

from phase1_dataset import (
    load_dataset,
    RobustnessLabeler,
    save_instances,
    load_instances,
    print_label_statistics,
    train_val_test_split,
)
from phase1_models import (
    load_gemma3,
    load_sae,
    load_available_layers,
    MODEL_ID,
    SAE_REPO_ID,
    TARGET_LAYER,
)


# ---------------------------------------------------------------------------
# Config — edit these as needed
# ---------------------------------------------------------------------------

# ContractNLI: 3-way NLI over contract clauses
# HF identifier: "kiddothe2b/contract-nli"
# Splits: "train" (7191) / "validation" (1010) / "test" (1571)
DATASET_NAME  = "kiddothe2b/contract-nli"
DATASET_SPLIT = "dev"     # use validation for dev; swap to "test" for final eval
MAX_INSTANCES = 5               # start small; increase to 1000+ for full runs

# 5 temperatures → professor's robustness requirement
# Each instance is run once per temperature; majority vote determines is_correct
TEMPERATURES       = [0.3, 0.7, 1.0, 1.3, 1.7]
MAJORITY_THRESHOLD = 0.5     # must be correct on >50% of runs

OUTPUT_PATH = "data/labeled_instances.json"
SPLITS_DIR  = "data/"

# Set SKIP_LABELING = True after a full run to reload from disk and skip re-labeling
SKIP_LABELING = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Step 1: Inspect available SAE layers — run once, then set TARGET_LAYER
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 — Available SAE layers")
    print("=" * 60)
    load_available_layers(SAE_REPO_ID)
    # After seeing the output, set TARGET_LAYER in phase1_models.py.
    # Default is 26 (late-middle layer for Gemma 3 4B).
    # Try 20, 24, 28, 32, 36 in later experiments.

    # ------------------------------------------------------------------
    # Step 2: Load Gemma 3
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Load Gemma 3")
    print("=" * 60)
    model, tokenizer = load_gemma3(MODEL_ID)

    # ------------------------------------------------------------------
    # Step 3: Load GemmaScope 2 SAE (smoke-test; used heavily in Phase 2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 3 — Load GemmaScope 2 SAE (layer {TARGET_LAYER})")
    print("=" * 60)
    sae_weights = load_sae(sae_repo_id=SAE_REPO_ID, layer=TARGET_LAYER)
    print("SAE loaded successfully.")

    # ------------------------------------------------------------------
    # Step 4: Load dataset
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Load ContractNLI dataset")
    print("=" * 60)
    instances = load_dataset(DATASET_NAME, DATASET_SPLIT, MAX_INSTANCES)
    print(f"Loaded {len(instances)} instances.")

    # ------------------------------------------------------------------
    # Step 5: Robustness labeling (or reload from disk)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 — Robustness labeling")
    print("=" * 60)

    if SKIP_LABELING:
        print(f"SKIP_LABELING=True — loading from {OUTPUT_PATH}")
        labeled_instances = load_instances(OUTPUT_PATH)
    else:
        labeler = RobustnessLabeler(
            model=model,
            tokenizer=tokenizer,
            temperatures=TEMPERATURES,
            majority_threshold=MAJORITY_THRESHOLD,
        )
        labeled_instances = labeler.label_dataset(instances)

    # ------------------------------------------------------------------
    # Step 6: Print statistics (sanity-check labels and log-probs)
    # ------------------------------------------------------------------
    print_label_statistics(labeled_instances)

    # ------------------------------------------------------------------
    # Step 7: Save labeled instances — always checkpoint before splitting
    # ------------------------------------------------------------------
    save_instances(labeled_instances, OUTPUT_PATH)

    # ------------------------------------------------------------------
    # Step 8: Train / val / test split (stratified by is_correct)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8 — Train / val / test split")
    print("=" * 60)
    train, val, test = train_val_test_split(labeled_instances)
    save_instances(train, SPLITS_DIR + "train.json")
    save_instances(val,   SPLITS_DIR + "val.json")
    save_instances(test,  SPLITS_DIR + "test.json")

    print(f"\nDone. Splits: {len(train)} train / {len(val)} val / {len(test)} test")
    print(f"All outputs in: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
