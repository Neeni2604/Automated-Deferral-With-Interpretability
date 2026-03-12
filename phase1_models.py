"""
Model loading utilities for Phase 1.

Handles Gemma 3 and the GemmaScope 2 SAE. The SAE lets us decompose
Gemma's internal activations into sparse, interpretable features — these
become the input to our deferral classifier in Phase 2.

NOTE: MODEL_ID and SAE_REPO_ID below are set for the 1B model for local
development. Switch both to the 4B variants when running on CHPC.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Config — swap these when moving from local dev → CHPC
# ---------------------------------------------------------------------------

MODEL_ID     = "google/gemma-3-1b-it"       # change to gemma-3-4b-it on CHPC
SAE_REPO_ID  = "google/gemma-scope-2-1b-it" # change to gemma-scope-2-4b-it on CHPC
TARGET_LAYER = 17                           # change to 22 for 4B model
SAE_TYPE     = "resid_post"
SAE_WIDTH    = "16k"
SAE_L0       = "medium"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Gemma 3
# ---------------------------------------------------------------------------

def load_gemma3(model_id: str = MODEL_ID):
    """Load Gemma 3 IT in bfloat16. Returns (model, tokenizer)."""
    print(f"Loading tokenizer: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id} (bfloat16, device_map=auto) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  -> {n_params:.2f}B parameters on {DEVICE}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# GemmaScope 2 SAE
# ---------------------------------------------------------------------------

def load_available_layers(sae_repo_id: str = SAE_REPO_ID) -> list[str]:
    """Print the resid_post layers available in this SAE repo."""
    # 1B IT has SAEs at layers 9, 17, 22, 29
    # 4B IT has SAEs at every layer via gemma-scope-2-4b-it-res-all
    layers = ["layer_9", "layer_17", "layer_22", "layer_29"]
    print(f"Available resid_post SAE layers in {sae_repo_id}:")
    for l in layers:
        print(f"  {l}")
    return layers


def load_sae(
    sae_repo_id: str = SAE_REPO_ID,
    layer: int       = TARGET_LAYER,
    sae_type: str    = SAE_TYPE,
    width: str       = SAE_WIDTH,
    l0: str          = SAE_L0,
) -> dict:
    """
    Download and load SAE weights for a specific layer from HuggingFace.
    Returns a dict of tensors with keys: w_enc, w_dec, b_enc, b_dec, threshold.
    """
    filename   = f"{sae_type}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"
    local_path = hf_hub_download(repo_id=sae_repo_id, filename=filename)
    params     = load_file(local_path, device=DEVICE)

    # w_enc: (hidden_dim, sae_dim)  w_dec: (sae_dim, hidden_dim)
    print(f"SAE loaded - layer {layer}, sae_dim={params['w_enc'].shape[1]}, "
          f"hidden_dim={params['w_enc'].shape[0]}")
    return params


# ---------------------------------------------------------------------------
# Activation hooks
# ---------------------------------------------------------------------------

def register_activation_hook(model, layer_index: int) -> tuple[list, object]:
    """
    Hook into a transformer layer to capture the residual stream.

    Usage:
        buffer, handle = register_activation_hook(model, TARGET_LAYER)
        with torch.no_grad():
            model(**inputs)
        hidden = buffer[0]  # (batch, seq_len, hidden_dim)
        handle.remove()
        buffer.clear()
    """
    buffer = []

    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        buffer.append(hidden.detach().cpu())

    handle = model.model.layers[layer_index].register_forward_hook(_hook)
    return buffer, handle


def extract_answer_token_hidden_state(
    hidden_state: torch.Tensor,
    answer_token_position: int = -1,
) -> torch.Tensor:
    """
    Pull out the activation vector at one token position.
    Default is -1 (last token), which is the standard choice for
    decoder-only multiple-choice — the model has seen the full prompt here.

    Returns: (hidden_dim,) tensor
    """
    return hidden_state[0, answer_token_position, :]


# ---------------------------------------------------------------------------
# SAE encoding
# ---------------------------------------------------------------------------

def encode_with_sae(hidden_state: torch.Tensor, sae_weights: dict) -> torch.Tensor:
    """
    Run a hidden state through the SAE encoder to get a sparse feature vector.

    Uses JumpReLU (threshold-based activation) rather than plain ReLU —
    this is what GemmaScope 2 uses.

    Returns: (sae_dim,) tensor — sparse, most values are 0.
    """
    w_enc     = sae_weights["w_enc"]      # (hidden_dim, sae_dim)
    b_enc     = sae_weights["b_enc"]      # (sae_dim,)
    threshold = sae_weights["threshold"]  # (sae_dim,)

    h = hidden_state.to(w_enc.device).to(w_enc.dtype)
    pre_activation = h @ w_enc + b_enc
    features = pre_activation * (pre_activation > threshold).float()

    return features
