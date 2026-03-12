"""
Phase 1 entry point.

Runs the full data collection pipeline:
  1. Sanity-check available SAE layers
  2. Load Gemma 3
  3. Load GemmaScope 2 SAE
  4. Load ContractNLI dataset
  5. Label each instance correct/incorrect via majority vote across 5 temperatures
  6. Print statistics
  7. Save labeled instances to disk
  8. Save train/val/test splits

Usage:
  python phase1_main.py

Prerequisites:
  pip install -r requirements.txt
  huggingface-cli login   # Gemma 3 is gated
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
# Config
# ---------------------------------------------------------------------------

DATASET_NAME  = "kiddothe2b/contract-nli"
DATASET_SPLIT = "dev"        # dev = validation split in this dataset
MAX_INSTANCES = 500

TEMPERATURES       = [0.3, 0.7, 1.0, 1.3, 1.7]
MAJORITY_THRESHOLD = 0.5     # correct on >50% of runs → is_correct = True

OUTPUT_PATH = "data/labeled_instances.json"
SPLITS_DIR  = "data/"

# Set True after the first full run to skip labeling and reload from disk
SKIP_LABELING = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Step 1 — check what SAE layers are available
    print("=" * 60)
    print("STEP 1 — Available SAE layers")
    print("=" * 60)
    load_available_layers(SAE_REPO_ID)

    # Step 2 — load Gemma 3
    print("\n" + "=" * 60)
    print("STEP 2 — Load Gemma 3")
    print("=" * 60)
    model, tokenizer = load_gemma3(MODEL_ID)

    # Step 3 — load SAE (smoke test; SAE is used heavily in Phase 2)
    print("\n" + "=" * 60)
    print(f"STEP 3 — Load GemmaScope 2 SAE (layer {TARGET_LAYER})")
    print("=" * 60)
    sae_weights = load_sae(sae_repo_id=SAE_REPO_ID, layer=TARGET_LAYER)
    print("SAE loaded successfully.")

    # Step 4 — load dataset
    print("\n" + "=" * 60)
    print("STEP 4 — Load ContractNLI dataset")
    print("=" * 60)
    instances = load_dataset(DATASET_NAME, DATASET_SPLIT, MAX_INSTANCES)
    print(f"Loaded {len(instances)} instances.")

    # Step 5 — robustness labeling
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

    # Step 6 — print statistics
    print_label_statistics(labeled_instances)

    # Step 7 — save full labeled dataset
    save_instances(labeled_instances, OUTPUT_PATH)

    # Step 8 — stratified train/val/test split
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
