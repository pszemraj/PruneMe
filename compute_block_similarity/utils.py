import logging
from typing import List

import torch

logging.basicConfig(level=logging.INFO)


def angular_distance(x_l, x_l_plus_n) -> torch.Tensor:
    """Compute the angular distance between layer output tokens."""
    x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True)
    x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True)
    cosine_similarity = (x_l_norm * x_l_plus_n_norm).sum(-1)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi


def compute_block_distances(
    hidden_states: List[torch.Tensor], layers_to_skip: int
) -> List[float]:
    """Compute and return angular distances for each block of layers."""
    distances = []
    num_layers = len(hidden_states)
    for l in range(num_layers - layers_to_skip):
        block_distance = (
            angular_distance(hidden_states[l], hidden_states[l + layers_to_skip])
            .mean()
            .item()
        )
        distances.append(block_distance)
    return distances


def get_last_non_padded_tokens(hidden_states, attention_mask) -> List[torch.Tensor]:
    """Get last non-padded tokens for each layer."""
    last_non_padded_hidden_states = []
    for layer in hidden_states:
        batch_size, _, _ = layer.size()
        batch_last_tokens = []
        for batch in range(batch_size):
            last_non_pad_index = attention_mask[batch].nonzero(as_tuple=True)[0].max()
            last_token = layer[batch, last_non_pad_index, :]
            batch_last_tokens.append(last_token.unsqueeze(0))
        last_non_padded_hidden_states.append(torch.cat(batch_last_tokens, dim=0))
    return last_non_padded_hidden_states


def check_ampere_gpu():
    """
    Check if the GPU supports NVIDIA Ampere or later and enable FP32 in PyTorch if it does.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.warning("No GPU detected, running on CPU.")
        return

    try:
        # Get the compute capability of the GPU
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability

        # Check if the GPU is Ampere or newer (compute capability >= 8.0)
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name(device)
            logging.info(
                f"{gpu_name} (compute capability {major}.{minor}) supports NVIDIA Ampere or later, "
                "enabled TF32 in PyTorch."
            )
        else:
            gpu_name = torch.cuda.get_device_name(device)
            logging.info(
                f"{gpu_name} (compute capability {major}.{minor}) does not support NVIDIA Ampere or later."
            )

    except Exception as e:
        logging.warning(f"Error occurred while checking GPU: {e}")
