"""
Dataset loading and robustness labeling for Phase 1.

We run Gemma 3 on each ContractNLI instance at 5 different temperatures and
take a majority vote to decide whether the model "knows" the answer. Instances
where the model consistently gets it wrong become deferral candidates in Phase 2.

Labels: Entailment → A, Contradiction → B, Not Mentioned → C
"""

import json
import random
from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    """One ContractNLI example, plus everything we collect about it."""

    id: str           # e.g. "dev_42"
    input_text: str   # formatted prompt sent to Gemma 3
    gold_label: str   # correct answer: "A", "B", or "C"

    # filled in by RobustnessLabeler
    predictions: list[str] = field(default_factory=list)  # one per temperature
    log_probs: list[float] = field(default_factory=list)  # log-prob of predicted token

    # True  = model usually gets this right (keep)
    # False = model usually gets this wrong (defer)
    is_correct: Optional[bool] = None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _format_prompt(premise: str, hypothesis: str) -> str:
    """Build the multiple-choice prompt we send to Gemma 3."""
    # truncate long contracts so we don't blow the context window
    if len(premise) > 1500:
        premise = premise[:1500] + "..."

    return (
        "Read the contract excerpt and the hypothesis, then choose the correct label.\n\n"
        f"Contract excerpt:\n{premise}\n\n"
        f"Hypothesis: {hypothesis}\n\n"
        "A) Entailment   — the contract supports the hypothesis\n"
        "B) Contradiction — the contract contradicts the hypothesis\n"
        "C) Not Mentioned — the contract does not address the hypothesis\n\n"
        "Answer (A, B, or C):"
    )


def load_dataset(dataset_name: str, split: str, max_instances: int) -> list[Instance]:
    """
    Load ContractNLI from HuggingFace.
    split must be one of: "train", "dev", "test"
    """
    import zipfile
    import json as _json
    from huggingface_hub import hf_hub_download

    label_map = {
        "entailment":    "A",
        "contradiction": "B",
        "neutral":       "C",
        "not_mentioned": "C",
    }

    print(f"Loading ContractNLI (kiddothe2b/contract-nli) [{split}] ...")
    zip_path = hf_hub_download("kiddothe2b/contract-nli", "contract_nli.zip", repo_type="dataset")

    instances = []
    global_idx = 0  # used for stable IDs across the full file

    with zipfile.ZipFile(zip_path) as z:
        with z.open("contract_nli_v1.jsonl") as f:
            for line in f:
                row = _json.loads(line)
                global_idx += 1

                if row.get("subset") != split:
                    continue

                gold = label_map.get(str(row.get("label", "")).lower().strip(), "")
                if not gold:
                    continue

                instances.append(Instance(
                    id=f"{split}_{global_idx}",
                    input_text=_format_prompt(
                        str(row.get("premise", "")),
                        str(row.get("hypothesis", "")),
                    ),
                    gold_label=gold,
                ))

                if max_instances and len(instances) >= max_instances:
                    break

    print(f"  -> {len(instances)} instances loaded")
    return instances


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def train_val_test_split(
    instances: list[Instance],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Instance], list[Instance], list[Instance]]:
    """
    Stratified split that keeps the correct/incorrect ratio balanced across
    train, val, and test. 
    """
    rng = random.Random(seed)

    correct   = [i for i in instances if i.is_correct is True]
    incorrect = [i for i in instances if i.is_correct is False]
    unlabeled = [i for i in instances if i.is_correct is None]

    def _split(items):
        items = items[:]
        rng.shuffle(items)
        n_train = int(len(items) * train_ratio)
        n_val   = int(len(items) * val_ratio)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    tr_c, va_c, te_c = _split(correct)
    tr_i, va_i, te_i = _split(incorrect)
    tr_u, va_u, te_u = _split(unlabeled)

    train = tr_c + tr_i + tr_u
    val   = va_c + va_i + va_u
    test  = te_c + te_i + te_u

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    print(f"Split -> train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ---------------------------------------------------------------------------
# Robustness labeling
# ---------------------------------------------------------------------------

class RobustnessLabeler:
    """
    Runs Gemma 3 at multiple temperatures per instance and labels each one
    correct/incorrect by majority vote.
    """

    def __init__(self, model, tokenizer, temperatures, majority_threshold=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.temperatures = temperatures
        self.majority_threshold = majority_threshold
        self.device = next(model.parameters()).device

    def run_single(self, input_text: str, temperature: float) -> tuple[str, float]:
        """
        One forward pass at a given temperature. Returns the predicted letter
        (A/B/C or ? if unrecognized) and the log-prob of the first generated token.
        """
        messages = [{"role": "user", "content": input_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=3,   # a few tokens in case of leading whitespace
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output.sequences[0, prompt_len:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()
        predicted = next((c for c in raw if c in "ABC"), "?")

        first_token_id = output.sequences[0, prompt_len]
        log_probs = torch.log_softmax(output.scores[0][0], dim=-1)
        log_prob  = log_probs[first_token_id].item()

        return predicted, log_prob

    def label_instance(self, instance: Instance) -> Instance:
        """Run all temperatures on one instance and set is_correct by majority vote."""
        for temp in self.temperatures:
            pred, lp = self.run_single(instance.input_text, temp)
            instance.predictions.append(pred)
            instance.log_probs.append(lp)

        n_correct = sum(p == instance.gold_label for p in instance.predictions)
        instance.is_correct = (n_correct / len(self.temperatures)) > self.majority_threshold
        return instance

    def label_dataset(self, instances: list[Instance]) -> list[Instance]:
        """Label all instances; prints progress every 50."""
        labeled = []
        for i, instance in enumerate(instances):
            labeled.append(self.label_instance(instance))

            if (i + 1) % 50 == 0 or (i + 1) == len(instances):
                n_done    = i + 1
                n_correct = sum(1 for inst in labeled if inst.is_correct)
                print(
                    f"  [{n_done:>4}/{len(instances)}]  "
                    f"correct so far: {n_correct}/{n_done} "
                    f"({100 * n_correct / n_done:.1f}%)"
                )

        return labeled


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _to_dict(inst: Instance) -> dict:
    return {
        "id":          inst.id,
        "input_text":  inst.input_text,
        "gold_label":  inst.gold_label,
        "predictions": inst.predictions,
        "log_probs":   inst.log_probs,
        "is_correct":  inst.is_correct,
    }


def _from_dict(d: dict) -> Instance:
    return Instance(
        id          = d["id"],
        input_text  = d["input_text"],
        gold_label  = d["gold_label"],
        predictions = d.get("predictions", []),
        log_probs   = d.get("log_probs", []),
        is_correct  = d.get("is_correct"),
    )


def save_instances(instances: list[Instance], path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([_to_dict(i) for i in instances], f, indent=2, ensure_ascii=False)
    print(f"Saved {len(instances)} instances -> {path}")


def load_instances(path: str) -> list[Instance]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    instances = [_from_dict(d) for d in data]
    print(f"Loaded {len(instances)} instances <- {path}")
    return instances


def print_label_statistics(instances: list[Instance]) -> None:
    """Quick sanity check: correct instances should have higher avg log-prob."""
    import numpy as np

    total     = len(instances)
    n_correct = sum(1 for i in instances if i.is_correct is True)
    n_wrong   = sum(1 for i in instances if i.is_correct is False)
    n_none    = total - n_correct - n_wrong

    correct_lps = [lp for i in instances if i.is_correct is True  for lp in i.log_probs]
    wrong_lps   = [lp for i in instances if i.is_correct is False for lp in i.log_probs]

    bar = "=" * 58
    print(f"\n{bar}\n  LABEL STATISTICS\n{bar}")
    print(f"  Total instances   : {total}")
    print(f"  Correct  (True)   : {n_correct:>5}  ({100 * n_correct / total:.1f}%)")
    print(f"  Incorrect (False) : {n_wrong:>5}  ({100 * n_wrong / total:.1f}%)")
    if n_none:
        print(f"  Unlabeled (None)  : {n_none:>5}")
    print()
    if correct_lps:
        print(f"  Avg log-prob — correct   : {np.mean(correct_lps):.4f}  (std {np.std(correct_lps):.4f})")
    if wrong_lps:
        print(f"  Avg log-prob — incorrect : {np.mean(wrong_lps):.4f}  (std {np.std(wrong_lps):.4f})")
    print("  (higher = more confident)")
    print(f"{bar}\n")
