"""
Handles loading Gemma 3 and the GemmaScope 2 SAE from HuggingFace.
Also defines which transformer layer to extract SAE features from.

Relevant HuggingFace repos:
  Gemma 3       : "google/gemma-3-4b-it"
  GemmaScope 2  : loaded via sae_lens release "gemma-scope-2-4b-it-resid_post"
                  (SAEs trained on Gemma 3 4B IT residual stream, every layer)

Gemma 3 4B has 46 transformer layers (0-45).
TARGET_LAYER = 26 is a late-middle layer; experiment with 20, 24, 28, 32, 36.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID    = "google/gemma-3-4b-it"

# sae_lens release name for Gemma Scope 2 — 4B instruction-tuned, residual stream
# Full list: https://huggingface.co/google/gemma-scope-2-4b-it
SAE_REPO_ID = "gemma-scope-2-4b-it-res-all"

# Late-middle layer of Gemma 3 4B (0-indexed, 46 layers total).
# Good starting point; sweep 20, 24, 28, 32, 36 in later experiments.
TARGET_LAYER = 26

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
        dtype=torch.bfloat16,    # 'dtype' is the correct kwarg in transformers >= 4.40
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
    print(f"SAE release  : {sae_repo_id}")
    print(f"Target layer : {TARGET_LAYER}")
    print(f"SAE id : resid_post_all/layer_{TARGET_LAYER}_width_16k_l0_small")
    print("  (widths available: 16k, 64k, 256k, 1m)")
    print("  (L0 options: small ~10-20, medium ~30-60, large ~60-150)")
    return [f"layer_{TARGET_LAYER}"]


def load_sae(sae_repo_id: str = SAE_REPO_ID, layer: int = TARGET_LAYER) -> dict:
    """
    Load the GemmaScope 2 SAE for a specific layer using sae_lens.

    sae_lens handles downloading, caching, and weight extraction automatically.

    SAE encoder : f(x) = ReLU(W_enc @ x + b_enc)
    SAE decoder : x_hat = W_dec @ f(x) + b_dec

    Returns a dict with keys:
      W_enc          : (sae_dim, hidden_dim)
      b_enc          : (sae_dim,)
      W_dec          : (hidden_dim, sae_dim)
      b_dec          : (hidden_dim,)
      scaling_factor : float
      layer          : int
      sae_dim        : int
      hidden_dim     : int
    """
    from sae_lens import SAE

    if layer is None:
        raise ValueError("TARGET_LAYER is None — set it in phase1_models.py first.")

    sae_id = f"layer_{layer}_width_16k_l0_small"
    print(f"Loading SAE  : release='{sae_repo_id}', id='{sae_id}' ...")

    sae, cfg_dict, _ = SAE.from_pretrained(
        release=sae_repo_id,
        sae_id=sae_id,
    )
    sae = sae.to(DEVICE)

    W_enc = sae.W_enc.detach()        # (sae_dim, hidden_dim)
    b_enc = sae.b_enc.detach()        # (sae_dim,)
    W_dec = sae.W_dec.detach()        # (sae_dim, hidden_dim) in sae_lens convention
    # sae_lens stores W_dec as (sae_dim, hidden_dim); transpose to (hidden_dim, sae_dim)
    W_dec_T = W_dec.T                 # (hidden_dim, sae_dim)

    if hasattr(sae, 'b_dec') and sae.b_dec is not None:
        b_dec = sae.b_dec.detach()    # (hidden_dim,)
    else:
        b_dec = torch.zeros(W_dec_T.shape[0], dtype=W_enc.dtype, device=DEVICE)

    sae_weights = {
        "W_enc":          W_enc,
        "b_enc":          b_enc,
        "W_dec":          W_dec_T,
        "b_dec":          b_dec,
        "scaling_factor": 1.0,
        "layer":          layer,
        "sae_dim":        W_enc.shape[0],
        "hidden_dim":     W_enc.shape[1],
    }

    print(
        f"  SAE loaded — hidden_dim={sae_weights['hidden_dim']}, "
        f"sae_dim={sae_weights['sae_dim']}"
    )
    return sae_weights


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
    W_enc          = sae_weights["W_enc"]            # (sae_dim, hidden_dim)
    b_enc          = sae_weights["b_enc"]            # (sae_dim,)
    scaling_factor = sae_weights["scaling_factor"]   # float

    h = hidden_state.to(W_enc.device).to(W_enc.dtype)
    h = h * scaling_factor

    pre_activation = W_enc @ h + b_enc
    features       = torch.relu(pre_activation)

    return features
