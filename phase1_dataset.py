"""
Handles dataset loading and robustness-aware correctness labeling.
Each instance is labeled correct/incorrect based on majority vote
across multiple sampling temperatures (per professor's instructions).
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


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
    gold_label: str                  # ground-truth answer from the dataset

    # Filled in during robustness labeling (Phase 1)
    predictions: list[str] = field(default_factory=list)
    # One prediction per (temperature, sample) run
    # e.g. ["A", "A", "B", "A", "C"] for 5 runs

    log_probs: list[float] = field(default_factory=list)
    # Log-probability of the predicted answer token for each run

    is_correct: Optional[bool] = None
    # Final majority-vote correctness label:
    # True  → model handles this correctly most of the time
    # False → model fails consistently (this is what we want to defer)
    # Set by RobustnessLabeler.label_instance()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_name: str, split: str, max_instances: int) -> list[Instance]:
    """
    Load a dataset from HuggingFace datasets and return a list of Instance objects.

    - dataset_name: HuggingFace dataset identifier, e.g. "cais/mmlu"
    - split: which split to use, e.g. "test" or "validation"
    - max_instances: cap on number of instances to load (use a small number
      during development to avoid long runtimes)

    Steps:
    1. Load the dataset using datasets.load_dataset()
    2. For each row, construct an Instance with:
       - id: some unique string (e.g. str(index))
       - input_text: the formatted prompt (question + answer choices)
       - gold_label: the correct answer string (e.g. "A", "B", "C", "D")
    3. Return the list of Instance objects (predictions/labels not yet filled)

    NOTE: for MMLU, the prompt format matters. Format as:
    "Question: {question}\nA) {choice_A}\nB) {choice_B}\n...\nAnswer:"
    so that Gemma 3 is expected to generate a single answer token.
    """
    raise NotImplementedError


def train_val_test_split(
    instances: list[Instance],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple[list[Instance], list[Instance], list[Instance]]:
    """
    Split the labeled instances into train / val / test sets.

    - Use a fixed random seed for reproducibility.
    - Return (train_instances, val_instances, test_instances).
    - Stratify by is_correct so that the error rate is balanced across splits.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Robustness labeling
# ---------------------------------------------------------------------------

class RobustnessLabeler:
    """
    Runs Gemma 3 multiple times per instance at different sampling temperatures
    and assigns a majority-vote correctness label to each instance.

    This addresses the professor's concern: we only label an instance as
    'correct' if the model gets it right most of the time, not just once by luck.
    """

    def __init__(
        self,
        model,
        tokenizer,
        temperatures: list[float],
        majority_threshold: float = 0.5,
    ):
        """
        - model: loaded Gemma 3 HuggingFace model (already on device)
        - tokenizer: corresponding Gemma 3 tokenizer
        - temperatures: list of sampling temperatures to run, e.g. [0.3, 0.7, 1.0, 1.3, 1.7]
        - majority_threshold: fraction of runs that must be correct for
          is_correct=True. e.g. 0.5 means "correct on more than half the runs"
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperatures = temperatures
        self.majority_threshold = majority_threshold

    def run_single(
        self, input_text: str, temperature: float
    ) -> tuple[str, float]:
        """
        Run Gemma 3 on a single input at a given temperature.

        Returns:
        - predicted_answer: the model's answer string (e.g. "A")
        - log_prob: log-probability of the predicted answer token

        Steps:
        1. Tokenize input_text
        2. Run model.generate() with the given temperature and do_sample=True
        3. Decode the first generated token to get the predicted answer
        4. Extract the log-probability of that token from the model's scores
           (pass return_dict_in_generate=True, output_scores=True)
        5. Return (predicted_answer, log_prob)

        NOTE: we only want the first generated token since our tasks are
        multiple-choice (single letter answer).
        """
        raise NotImplementedError

    def label_instance(self, instance: Instance) -> Instance:
        """
        Run all temperatures on a single Instance and assign is_correct.

        Steps:
        1. For each temperature in self.temperatures, call self.run_single()
        2. Append each prediction to instance.predictions
        3. Append each log_prob to instance.log_probs
        4. Count how many predictions match instance.gold_label
        5. Set instance.is_correct = True if (correct_count / total_runs) > majority_threshold
        6. Return the updated instance
        """
        raise NotImplementedError

    def label_dataset(self, instances: list[Instance]) -> list[Instance]:
        """
        Apply label_instance() to every instance in the list.

        - Print progress every 50 instances so you can monitor runtime.
        - Return the fully labeled list.

        NOTE: this will be slow. Consider saving results to disk with
        save_instances() after this step so you don't have to re-run it.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_instances(instances: list[Instance], path: str) -> None:
    """
    Serialize the list of Instance objects to disk.

    Use either pickle or JSON (JSON preferred for readability).
    Save to `path`.

    This is important — robustness labeling is expensive, so you want
    to checkpoint after it completes.
    """
    raise NotImplementedError


def load_instances(path: str) -> list[Instance]:
    """
    Load a previously saved list of Instance objects from `path`.
    Returns the list of Instance objects.
    """
    raise NotImplementedError


def print_label_statistics(instances: list[Instance]) -> None:
    """
    Print a summary of the labeled dataset:
    - Total number of instances
    - Number and percentage labeled is_correct=True
    - Number and percentage labeled is_correct=False
    - Average log_prob for correct vs incorrect instances
      (sanity check: correct instances should have higher log_probs on average)
    """
    raise NotImplementedError