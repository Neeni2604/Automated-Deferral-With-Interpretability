"""
Handles loading Gemma 3 and the GemmaScope2 SAE from HuggingFace.
Also defines which transformer layer to extract SAE features from.

Relevant HuggingFace repos:
- Gemma 3:      "google/gemma-3-<size>-it"  (e.g. gemma-3-4b-it)
- GemmaScope 2: "google/gemma-scope-2-<size>-pt"
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-3-4b-pt"        # swap size as needed (1b, 4b, 12b, 27b)
SAE_REPO_ID = "google/gemma-scope-2-4b-pt"  # must match the model size above
TARGET_LAYER = 22                         # int - the transformer layer whose SAE
                                          # you will use for feature extraction.
                                          # Decide this before running Phase 2.
SAE_TYPE = "resid_post"
SAE_WIDTH = "16k"
SAE_L0 = "medium"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Gemma 3 loading
# ---------------------------------------------------------------------------

def load_gemma3(model_id: str = MODEL_ID):
    """
    Load the Gemma 3 model and tokenizer from HuggingFace.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()
    return (model, tokenizer)


# ---------------------------------------------------------------------------
# GemmaScope2 SAE loading
# ---------------------------------------------------------------------------

def load_sae(sae_repo_id: str = SAE_REPO_ID, 
             layer: int = TARGET_LAYER, 
             sae_type: str = SAE_TYPE,
             width: str = SAE_WIDTH,
             l0: str = SAE_L0,
             ) -> dict:
    """
    Load the pre-trained GemmaScope2 SAE for a specific transformer layer.
    """
    folder = f"{sae_type}/layer_{layer}_width_{width}_l0_{l0}"
    
    local_path = hf_hub_download(
        repo_id=sae_repo_id,
        filename=f"{folder}/params.safetensors",
    )

    params = load_file(local_path, device=DEVICE)
    
    print("Keys in params.safetensors:", list(params.keys()))
    print(f"SAE loaded from {folder}")
    
    return params



# ---------------------------------------------------------------------------
# Activation hooks
# ---------------------------------------------------------------------------

def register_activation_hook(model, layer_index: int) -> tuple[list, object]:
    """
    Register a forward hook on the transformer layer at `layer_index`
    to capture the residual stream hidden state during a forward pass.

    Steps:
    1. Identify the correct submodule path for layer `layer_index` in Gemma 3.
       In HuggingFace Gemma 3, layers are at model.model.layers[layer_index]
    2. Define a hook function that appends the layer's output hidden state
       to a buffer list
    3. Register the hook with .register_forward_hook()
    4. Return (buffer, hook_handle)
       - buffer: the list that will be populated with hidden states
       - hook_handle: used to remove the hook later with hook_handle.remove()

    Usage pattern:
        buffer, handle = register_activation_hook(model, TARGET_LAYER)
        with torch.no_grad():
            model(**inputs)
        hidden_state = buffer[0]   # shape: (batch, seq_len, hidden_dim)
        handle.remove()
        buffer.clear()
    """
    raise NotImplementedError


def extract_answer_token_hidden_state(
    hidden_state: torch.Tensor,
    answer_token_position: int = -1
) -> torch.Tensor:
    """
    Extract the hidden state at a specific token position from a full
    sequence hidden state tensor.

    - hidden_state: shape (batch, seq_len, hidden_dim)
    - answer_token_position: which token position to extract.
      Default -1 (last token) is a reasonable starting point for
      decoder-only models on multiple-choice tasks.

    Returns a tensor of shape (hidden_dim,) — the activation vector
    for a single instance at the chosen position.

    NOTE: The choice of token position is a design decision.
    Discuss with your professor or experiment with a few options
    (last token, answer token, etc.).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# SAE encoding
# ---------------------------------------------------------------------------

def encode_with_sae(hidden_state: torch.Tensor, sae_weights: dict) -> torch.Tensor:
    """
    Pass a hidden state vector through the SAE encoder to get the
    sparse feature representation.

    Steps:
    1. Apply the linear transformation: pre_activation = W_enc @ hidden_state + b_enc
    2. Apply ReLU to get the sparse feature vector: features = ReLU(pre_activation)
    3. Return the sparse feature vector — shape (sae_hidden_dim,)

    The resulting vector is sparse: most entries will be zero.
    The non-zero entries correspond to the interpretable SAE features
    that are "active" for this input.

    This is the core feature representation you will use for deferral.
    """
    raise NotImplementedError