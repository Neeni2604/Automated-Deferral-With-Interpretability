"""
Entry point for Phase 2. Run this file to:
- Load Gemma 3 and GemmaScope2 SAE
- Load labeled instances from Phase 1 output
- Extract SAE feature vectors for every labeled instance
- Save feature matrix and labels to disk for Phase 3
"""

import torch
from phase1_models import (
    load_gemma3,
    load_sae,
    MODEL_ID,
    SAE_REPO_ID,
    TARGET_LAYER,
    SAE_TYPE,
    SAE_WIDTH,
    SAE_L0,
    DEVICE,
)
from phase1_dataset import load_instances
from phase2_extraction import (
    extract_all,
    save_features,
    load_features,
    print_feature_statistics,
)


INSTANCES_PATH   = "data/labeled_instances.json"
OUTPUT_DIR       = "data/"
CHECKPOINT_EVERY = 50
SKIP_EXTRACTION  = False        # Set True after a completed run to skip re-extraction



def main():
    print("=" * 60)
    print("STEP 1: Load Gemma 3")
    print("=" * 60)
    model, tokenizer = load_gemma3(MODEL_ID)


    print("\n" + "=" * 60)
    print(f"STEP 2: Load GemmaScope2 SAE (layer {TARGET_LAYER})")
    print("=" * 60)
    sae_weights = load_sae(
        sae_repo_id=SAE_REPO_ID,
        layer=TARGET_LAYER,
        sae_type=SAE_TYPE,
        width=SAE_WIDTH,
        l0=SAE_L0,
    )


    print("\n" + "=" * 60)
    print("STEP 3: Load labeled instances")
    print("=" * 60)
    instances = load_instances(INSTANCES_PATH)


    print("\n" + "=" * 60)
    print("STEP 4: SAE feature extraction")
    print("=" * 60)

    if SKIP_EXTRACTION:
        print("SKIP_EXTRACTION=True - loading from disk...")
        feature_matrix, labels, instance_ids = load_features(OUTPUT_DIR)
    else:
        feature_matrix, labels, instance_ids = extract_all(
            instances=instances,
            model=model,
            tokenizer=tokenizer,
            sae_weights=sae_weights,
            layer_index=TARGET_LAYER,
            device=DEVICE,
            checkpoint_every=CHECKPOINT_EVERY,
            checkpoint_path=OUTPUT_DIR + "features_checkpoint.npz",
        )


    print("\n" + "=" * 60)
    print("STEP 5: Feature statistics")
    print("=" * 60)
    print_feature_statistics(feature_matrix, labels)


    print("\n" + "=" * 60)
    print("STEP 6: Save features")
    print("=" * 60)
    save_features(feature_matrix, labels, instance_ids, OUTPUT_DIR)


    print(f"\nDone. Feature matrix shape: {feature_matrix.shape}")
    print(f"Correct: {labels.sum()} / {len(labels)} instances")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()