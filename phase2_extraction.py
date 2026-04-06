"""
SAE feature extraction for every labeled instance.

For each instance:
- Tokenize the input_text
- Register activation hook on TARGET_LAYER
- Run a forward pass (don't generate anything cuz we just want the hidden state)
- Extract hidden state at the last token position
- Pass through SAE encoder → sparse feature vector of shape (16384,)
- Store the feature vector
"""

import json
import numpy as np
import torch

from phase1_dataset import load_instances, Instance
from phase1_models import (
    load_gemma3,
    load_sae,
    register_activation_hook,
    extract_answer_token_hidden_state,
    encode_with_sae,
    MODEL_ID,
    SAE_REPO_ID,
    TARGET_LAYER,
    SAE_TYPE,
    SAE_WIDTH,
    SAE_L0,
)


# Extracting features for all instances

def extract_sae_features_for_instance(
    instance: Instance,
    model,
    tokenizer,
    sae_weights: dict,
    layer_index: int,
    device: str,
) -> torch.Tensor:
    """
    Run one forward pass for a single instance and return its SAE feature vector.
    """
    messages = [{"role": "user", "content": instance.input_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    buffer, hook_handle = register_activation_hook(model, layer_index)

    with torch.no_grad():
        model(**inputs)

    hook_handle.remove()

    hidden_state = buffer[0]
    buffer.clear()

    h = extract_answer_token_hidden_state(hidden_state, answer_token_position=-1)

    features = encode_with_sae(h, sae_weights)

    return features.cpu()


def extract_all(
    instances: list[Instance],
    model,
    tokenizer,
    sae_weights: dict,
    layer_index: int,
    device: str,
    checkpoint_every: int = 50,
    checkpoint_path: str = "data/features_checkpoint.npz",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract SAE feature vectors for all instances.
    """
    feature_list = []
    label_list = []
    id_list = []

    labeled = [inst for inst in instances if inst.is_correct is not None]
    print(f"Extracting features for {len(labeled)} labeled instances "
          f"(skipped {len(instances) - len(labeled)} unlabeled)...")

    for i, instance in enumerate(labeled):
        features = extract_sae_features_for_instance(
            instance, model, tokenizer, sae_weights, layer_index, device
        )

        feature_list.append(features.numpy())
        label_list.append(instance.is_correct)
        id_list.append(instance.id)


        if (i + 1) % 50 == 0 or (i + 1) == len(labeled):
            n_done = i + 1
            n_correct = sum(label_list)
            print(
                f"  [{n_done}/{len(labeled)}]  "
                f"correct: {n_correct}/{n_done} ({100 * n_correct / n_done:.1f}%)"
            )

        if (i + 1) % checkpoint_every == 0:
            _save_checkpoint(feature_list, label_list, id_list, checkpoint_path)
            print(f"  Checkpoint saved → {checkpoint_path}")

    feature_matrix = np.stack(feature_list, axis=0)   # (n_instances, sae_dim)
    labels = np.array(label_list, dtype=bool)  # (n_instances,)

    return feature_matrix, labels, id_list



# Helper functions

def save_features(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    instance_ids: list[str],
    output_dir: str = "data/",
) -> None:
    """
    Save feature matrix, labels, and instance ids to disk.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    np.save(output_dir + "feature_matrix.npy", feature_matrix.astype(np.float32))
    np.save(output_dir + "labels.npy", labels)

    with open(output_dir + "instance_ids.json", "w") as f:
        json.dump(instance_ids, f, indent=2)

    print(f"Saved feature_matrix {feature_matrix.shape} → {output_dir}feature_matrix.npy")
    print(f"Saved labels {labels.shape} → {output_dir}labels.npy")
    print(f"Saved instance_ids → {output_dir}instance_ids.json")


def load_features(
    output_dir: str = "data/",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load previously saved feature matrix, labels, and instance ids
    """
    feature_matrix = np.load(output_dir + "feature_matrix.npy")
    labels = np.load(output_dir + "labels.npy")

    with open(output_dir + "instance_ids.json") as f:
        instance_ids = json.load(f)

    print(f"Loaded feature_matrix {feature_matrix.shape} ← {output_dir}feature_matrix.npy")
    print(f"Loaded labels {labels.shape} ← {output_dir}labels.npy")

    return feature_matrix, labels, instance_ids


def print_feature_statistics(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
) -> None:
    """
    Prints:
    - Average sparsity (fraction of zero entries per vector)
    - Average number of active features per instance
    - Mean activation value of non-zero features
    - Whether correct vs incorrect instances have different average activations (do the features carry signal?)
    """
    n_instances, sae_dim = feature_matrix.shape
    nonzero_counts = (feature_matrix > 0).sum(axis=1)  # per instance
    sparsity = 1.0 - (nonzero_counts / sae_dim)

    correct_mean = feature_matrix[labels].mean()
    incorrect_mean = feature_matrix[~labels].mean()

    bar = "=" * 58
    print(f"\n{bar}\n  FEATURE STATISTICS\n{bar}")
    print(f"  Instances : {n_instances}")
    print(f"  SAE dim : {sae_dim}")
    print(f"  Avg active features / instance : {nonzero_counts.mean():.1f}  "
          f"(out of {sae_dim})")
    print(f"  Avg sparsity : {sparsity.mean():.4f}")
    print(f"  Mean activation - correct : {correct_mean:.6f}")
    print(f"  Mean activation - incorrect : {incorrect_mean:.6f}")
    print(f"{bar}\n")


def _save_checkpoint(
    feature_list: list,
    label_list: list,
    id_list: list,
    path: str,
) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        features=np.stack(feature_list, axis=0).astype(np.float32),
        labels=np.array(label_list, dtype=bool),
        ids=np.array(id_list),
    )