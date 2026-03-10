"""
Handles dataset loading and robustness-aware correctness labeling.
Each instance is labeled correct/incorrect based on majority vote
across multiple sampling temperatures (per professor's instructions).

Dataset: ContractNLI (stanfordnlp/contract-nli)
Task: 3-way NLI classification over contract clauses
Labels: Entailment → A, Contradiction → B, Not Mentioned → C
"""

import json
import random
from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Label mappings for ContractNLI
# ---------------------------------------------------------------------------

# ContractNLI exposes integer labels; map to single letters for generation.
LABEL_INT_TO_LETTER = {0: "A", 1: "B", 2: "C"}   # Entailment / Contradiction / Not Mentioned
LABEL_STR_TO_LETTER = {
    "entailment":    "A",
    "contradiction": "B",
    "not_mentioned": "C",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    """
    Represents a single dataset example and everything we collect about it.
    Populated incrementally across phases.
    """
    id: str                          # unique identifier for the instance
    input_text: str                  # the prompt/question fed to Gemma 3
    gold_label: str                  # ground-truth answer: "A", "B", or "C"

    # Filled in during robustness labeling (Phase 1)
    predictions: list[str] = field(default_factory=list)
    # One prediction per temperature run, e.g. ["A", "A", "B", "A", "C"]

    log_probs: list[float] = field(default_factory=list)
    # Log-probability of the predicted answer token for each run

    is_correct: Optional[bool] = None
    # Final majority-vote correctness label:
    #   True  → model handles this correctly most of the time
    #   False → model fails consistently (candidate for deferral)


# ---------------------------------------------------------------------------
# Dataset loading — ContractNLI
# ---------------------------------------------------------------------------

def _format_contractnli_prompt(premise: str, hypothesis: str) -> str:
    """
    Format a ContractNLI row as a multiple-choice prompt.
    Gemma 3 is expected to emit a single letter: A, B, or C.
    Premise is truncated to avoid blowing the context window.
    """
    max_chars = 1500
    if len(premise) > max_chars:
        premise = premise[:max_chars] + "..."

    return (
        "Read the contract excerpt and the hypothesis, then choose the correct label.\n\n"
        f"Contract excerpt:\n{premise}\n\n"
        f"Hypothesis: {hypothesis}\n\n"
        "A) Entailment   — the contract supports the hypothesis\n"
        "B) Contradiction — the contract contradicts the hypothesis\n"
        "C) Not Mentioned — the contract does not address the hypothesis\n\n"
        "Answer (A, B, or C):"
    )


def _normalize_label(raw_label) -> str:
    """Convert int or str label from HF dataset into A / B / C."""
    if isinstance(raw_label, int):
        return LABEL_INT_TO_LETTER.get(raw_label, "")
    if isinstance(raw_label, str):
        return LABEL_STR_TO_LETTER.get(raw_label.lower().replace(" ", "_"), "")
    return ""


def load_dataset(dataset_name: str, split: str, max_instances: int) -> list[Instance]:
    """
    Load ContractNLI from the kiddothe2b/contract-nli zip file on HuggingFace.

    The dataset uses a legacy loading script incompatible with modern HF datasets,
    so we download the raw zip and parse the JSONL directly.

    Source  : kiddothe2b/contract-nli  (contract_nli.zip → contract_nli_v1.jsonl)
    Splits  : "train" / "validation" / "test"
    Columns : premise, hypothesis, label (str: entailment/neutral/contradiction), subset
    """
    import zipfile
    import json as _json
    from huggingface_hub import hf_hub_download

    print(f"Loading ContractNLI (kiddothe2b/contract-nli) [{split}] ...")
    zip_path = hf_hub_download(
        'kiddothe2b/contract-nli',
        'contract_nli.zip',
        repo_type='dataset',
    )

    instances: list[Instance] = []
    global_idx = 0  # index across all rows (for stable IDs)

    with zipfile.ZipFile(zip_path) as z:
        with z.open('contract_nli_v1.jsonl') as f:
            for line in f:
                row = _json.loads(line)
                global_idx += 1

                if row.get('subset') != split:
                    continue

                premise    = str(row.get('premise', ''))
                hypothesis = str(row.get('hypothesis', ''))
                raw_label  = row.get('label', '')

                # Map "neutral" → C (the dataset uses "neutral" not "not_mentioned")
                label_map = {
                    'entailment':    'A',
                    'contradiction': 'B',
                    'neutral':       'C',
                    'not_mentioned': 'C',
                }
                gold_letter = label_map.get(str(raw_label).lower().strip(), '')
                if not gold_letter:
                    continue

                instances.append(Instance(
                    id=f"{split}_{global_idx}",
                    input_text=_format_contractnli_prompt(premise, hypothesis),
                    gold_label=gold_letter,
                ))

                if max_instances and len(instances) >= max_instances:
                    break

    print(f"  -> {len(instances)} instances loaded from split='{split}'")
    return instances


def train_val_test_split(
    instances: list[Instance],
    train_ratio: float = 0.7,
    val_ratio: float   = 0.1,
    seed: int = 42,
) -> tuple[list[Instance], list[Instance], list[Instance]]:
    """
    Stratified split by is_correct so the error rate is balanced across
    train / val / test.  Uses a fixed seed for reproducibility.
    """
    rng = random.Random(seed)

    correct   = [i for i in instances if i.is_correct is True]
    incorrect = [i for i in instances if i.is_correct is False]
    unlabeled = [i for i in instances if i.is_correct is None]

    def _split(items: list) -> tuple[list, list, list]:
        items = items[:]
        rng.shuffle(items)
        n       = len(items)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    tr_c, va_c, te_c = _split(correct)
    tr_i, va_i, te_i = _split(incorrect)
    tr_u, va_u, te_u = _split(unlabeled)

    train = tr_c + tr_i + tr_u
    val   = va_c + va_i + va_u
    test  = te_c + te_i + te_u

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)

    print(f"Split → train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ---------------------------------------------------------------------------
# Robustness labeling
# ---------------------------------------------------------------------------

class RobustnessLabeler:
    """
    Runs Gemma 3 multiple times per instance at different sampling temperatures
    and assigns a majority-vote correctness label to each instance.
    """

    def __init__(
        self,
        model,
        tokenizer,
        temperatures: list[float],
        majority_threshold: float = 0.5,
    ):
        self.model             = model
        self.tokenizer         = tokenizer
        self.temperatures      = temperatures
        self.majority_threshold = majority_threshold
        self.device            = next(model.parameters()).device

    # ------------------------------------------------------------------

    def run_single(self, input_text: str, temperature: float) -> tuple[str, float]:
        """
        Run one forward pass at the given temperature using the Gemma 3 chat template.

        Gemma 3 IT requires the chat template to be applied — without it the model
        does not follow instructions and generates garbage tokens instead of A/B/C.

        Returns:
          predicted_answer : single letter string ("A", "B", "C", or "?")
          log_prob         : log-prob of the first generated token
        """
        # Apply Gemma 3 IT chat template
        messages = [{"role": "user", "content": input_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=3,             # allow a few tokens; model may emit leading whitespace
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,           # logit distributions per step
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude prompt)
        new_tokens = output.sequences[0, prompt_len:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()

        # Extract first A/B/C from the decoded output
        predicted_answer = next((c for c in raw if c in "ABC"), "?")

        # Log-prob of the very first generated token
        first_token_id    = output.sequences[0, prompt_len]
        log_probs_all     = torch.log_softmax(output.scores[0][0], dim=-1)
        log_prob          = log_probs_all[first_token_id].item()

        return predicted_answer, log_prob

    # ------------------------------------------------------------------

    def label_instance(self, instance: Instance) -> Instance:
        """
        Run all temperatures on one Instance; compute majority-vote is_correct.
        """
        for temp in self.temperatures:
            pred, lp = self.run_single(instance.input_text, temp)
            instance.predictions.append(pred)
            instance.log_probs.append(lp)

        total         = len(self.temperatures)
        n_correct     = sum(p == instance.gold_label for p in instance.predictions)
        instance.is_correct = (n_correct / total) > self.majority_threshold

        return instance

    # ------------------------------------------------------------------

    def label_dataset(self, instances: list[Instance]) -> list[Instance]:
        """
        Label every instance; prints progress every 50 items.
        """
        labeled: list[Instance] = []

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
# Helper functions
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
    """Serialize to JSON (preferred over pickle for readability)."""
    import os
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump([_to_dict(inst) for inst in instances], f, indent=2, ensure_ascii=False)
    print(f"Saved {len(instances)} instances → {path}")


def load_instances(path: str) -> list[Instance]:
    """Deserialize instances previously saved with save_instances()."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    instances = [_from_dict(d) for d in data]
    print(f"Loaded {len(instances)} instances ← {path}")
    return instances


def print_label_statistics(instances: list[Instance]) -> None:
    """
    Summary of the labeled dataset.
    Sanity check: correct instances should have higher (less-negative) avg log_prob.
    """
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
    print(f"  Incorrect (False) : {n_wrong:>5}  ({100 * n_wrong  / total:.1f}%)")
    if n_none:
        print(f"  Unlabeled (None)  : {n_none:>5}")
    print()

    if correct_lps:
        print(f"  Avg log-prob — correct   : {np.mean(correct_lps):.4f}  "
              f"(std {np.std(correct_lps):.4f})")
    if wrong_lps:
        print(f"  Avg log-prob — incorrect : {np.mean(wrong_lps):.4f}  "
              f"(std {np.std(wrong_lps):.4f})")

    print("  (higher / less-negative log-prob → model was more confident)")
    print(f"{bar}\n")
