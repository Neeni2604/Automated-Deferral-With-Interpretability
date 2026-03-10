"""
Entry point for Phase 1. Run this file to:
1. Load the dataset
2. Load Gemma 3 and the GemmaScope2 SAE
3. Label all instances with robustness-aware correctness labels
4. Save results to disk
"""

from phase1_dataset import (
    load_dataset,
    RobustnessLabeler,
    save_instances,
    print_label_statistics,
    train_val_test_split,
)
from phase1_models import (
    load_gemma3,
    load_sae,
    MODEL_ID,
    SAE_REPO_ID,
    TARGET_LAYER,
)


# ---------------------------------------------------------------------------
# Config — change these as needed
# ---------------------------------------------------------------------------

DATASET_NAME = "cais/mmlu"       # HuggingFace dataset identifier
DATASET_SPLIT = "test"
DATASET_SUBJECT = "all"          # MMLU has subjects; "all" or pick one to start small
MAX_INSTANCES = 500              # start small during development

TEMPERATURES = [0.3, 0.7, 1.0, 1.3, 1.7]   # 5 runs per instance
MAJORITY_THRESHOLD = 0.5         # wrong on >50% of runs → is_correct=False

OUTPUT_PATH = "data/labeled_instances.json"
SPLITS_DIR = "data/"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Step 1: Inspect available SAE layers (run this first, then set TARGET_LAYER)
    # print("Available SAE layers:")
    # load_available_layers(SAE_REPO_ID)
    # Once you've seen the output, set TARGET_LAYER in phase1_models.py and continue.

    # Step 2: Load Gemma 3
    print(f"Loading Gemma 3 from {MODEL_ID}...")
    model, tokenizer = load_gemma3()

    # Step 3: Load the GemmaScope2 SAE for the chosen layer
    # (not used for labeling, but load it now to confirm everything works
    #  before you get to Phase 2)
    print(f"Loading SAE for layer {TARGET_LAYER}...")
    sae_weights = load_sae(layer=TARGET_LAYER)
    print("SAE loaded successfully.")

    # Step 4: Load dataset
    print(f"Loading dataset: {DATASET_NAME}...")
    instances = load_dataset(DATASET_NAME, DATASET_SPLIT, MAX_INSTANCES)
    print(f"Loaded {len(instances)} instances.")

    # Step 5: Run robustness labeling
    print("Running robustness labeling across temperatures...")
    labeler = RobustnessLabeler(
        model=model,
        tokenizer=tokenizer,
        temperatures=TEMPERATURES,
        majority_threshold=MAJORITY_THRESHOLD,
    )
    labeled_instances = labeler.label_dataset(instances)

    # Step 6: Print statistics and sanity-check the labels
    print_label_statistics(labeled_instances)

    # Step 7: Save labeled instances to disk
    # Do this before splitting — treat the full labeled set as your source of truth
    save_instances(labeled_instances, OUTPUT_PATH)
    print(f"Saved labeled instances to {OUTPUT_PATH}")

    # Step 8: Split into train / val / test
    train, val, test = train_val_test_split(labeled_instances)
    save_instances(train, SPLITS_DIR + "train.json")
    save_instances(val,   SPLITS_DIR + "val.json")
    save_instances(test,  SPLITS_DIR + "test.json")
    print(f"Splits saved: {len(train)} train / {len(val)} val / {len(test)} test")


if __name__ == "__main__":
    main()