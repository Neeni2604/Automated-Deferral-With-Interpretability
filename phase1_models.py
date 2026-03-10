"""
Handles loading Gemma 3 and the GemmaScope 2 SAE from HuggingFace.
Also defines which transformer layer to extract SAE features from.

Relevant HuggingFace repos:
  Gemma 3       : "google/gemma-3-4b-it"
  GemmaScope 2  : loaded via sae_lens release "gemma-scope-2-4b-it-resid_post"
                  (SAEs trained on Gemma 3 4B IT residual stream, every layer)

Available resid_post layers: 9, 17, 22, 29
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID    = "google/gemma-3-4b-it"

# sae_lens release name for Gemma Scope 2 — 4B instruction-tuned, residual stream
# Full list: https://huggingface.co/google/gemma-scope-2-4b-it
SAE_REPO_ID = "google/gemma-scope-2-4b-it"

# Late-middle layer of Gemma 3 4B (0-indexed, 35 layers total).
TARGET_LAYER = 22
SAE_TYPE     = "resid_post"
SAE_WIDTH    = "16k"
SAE_L0       = "medium"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Gemma 3 loading
# ---------------------------------------------------------------------------

def load_gemma3(model_id: str = MODEL_ID):
    """
    Load the Gemma 3 instruction-tuned model and tokenizer from HuggingFace.

    Uses bfloat16 to halve memory vs float32.
    device_map="auto" handles single-GPU, multi-GPU, or CPU-offload automatically.

    Returns: (model, tokenizer)
    """
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
    print(f"  -> {n_params:.2f}B parameters loaded on {DEVICE}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# GemmaScope 2 SAE loading — via sae_lens
# ---------------------------------------------------------------------------

def load_available_layers(sae_repo_id: str = SAE_REPO_ID) -> list[str]:
    """
    Print the SAE release and target layer being used.
    With sae_lens we don't need to enumerate files manually.
    """
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
    Load the GemmaScope 2 SAE for a specific layer.
    """
    filename   = f"{sae_type}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"
    local_path = hf_hub_download(repo_id=sae_repo_id, filename=filename)
    params     = load_file(local_path, device=DEVICE)

    # Keys: w_enc (hidden_dim, sae_dim), w_dec (sae_dim, hidden_dim),
    #       b_enc (sae_dim,), b_dec (hidden_dim,), threshold (sae_dim,)
    print(f"SAE loaded - layer {layer}, sae_dim={params['w_enc'].shape[1]}, "
          f"hidden_dim={params['w_enc'].shape[0]}")

    return params


# ---------------------------------------------------------------------------
# Activation hooks
# ---------------------------------------------------------------------------

def register_activation_hook(model, layer_index: int) -> tuple[list, object]:
    """
    Register a forward hook on transformer layer `layer_index` to capture
    the residual-stream hidden state during a forward pass.

    In HuggingFace Gemma 3, transformer blocks live at:
        model.model.layers[layer_index]

    Usage:
        buffer, handle = register_activation_hook(model, TARGET_LAYER)
        with torch.no_grad():
            model(**inputs)
        hidden_state = buffer[0]   # shape: (batch, seq_len, hidden_dim)
        handle.remove()
        buffer.clear()

    Returns:
        buffer      : list populated with hidden state tensors during forward pass
        hook_handle : call hook_handle.remove() to de-register after use
    """
    buffer: list[torch.Tensor] = []

    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        buffer.append(hidden.detach().cpu())   # CPU to avoid GPU OOM

    target_module = model.model.layers[layer_index]
    hook_handle   = target_module.register_forward_hook(_hook)

    return buffer, hook_handle


def extract_answer_token_hidden_state(
    hidden_state: torch.Tensor,
    answer_token_position: int = -1,
) -> torch.Tensor:
    """
    Extract the activation vector at a specific token position.

    Args:
        hidden_state          : (batch, seq_len, hidden_dim)
        answer_token_position : -1 (last token) is standard for decoder-only
                                multiple-choice tasks.

    Returns: (hidden_dim,) tensor
    """
    return hidden_state[0, answer_token_position, :]


# ---------------------------------------------------------------------------
# SAE encoding
# ---------------------------------------------------------------------------

def encode_with_sae(hidden_state: torch.Tensor, sae_weights: dict) -> torch.Tensor:
    """
    Pass one hidden-state vector through the SAE encoder.

    Equation: features = ReLU(W_enc @ (scaling_factor * h) + b_enc)

    The result is sparse — most entries are zero.
    Non-zero entries are the active interpretable features for this input.

    Args:
        hidden_state : (hidden_dim,)
        sae_weights  : dict from load_sae()

    Returns: (sae_dim,) sparse feature vector
    """
    w_enc     = sae_weights["w_enc"]      # (hidden_dim, sae_dim)
    b_enc     = sae_weights["b_enc"]      # (sae_dim,)
    threshold = sae_weights["threshold"]  # (sae_dim,)

    h = hidden_state.to(w_enc.device).to(w_enc.dtype)

    pre_activation = h @ w_enc + b_enc           # (sae_dim,)
    features = pre_activation * (pre_activation > threshold).float()

    return features
