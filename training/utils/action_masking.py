"""Action masking utilities for RL agents."""

import numpy as np
import torch
from typing import Union


def apply_action_mask(
    logits: Union[torch.Tensor, np.ndarray],
    action_mask: np.ndarray,
    mask_value: float = -1e9,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply action mask to logits, setting invalid actions to mask_value.
    
    Args:
        logits: Action logits (can be torch.Tensor or np.ndarray)
        action_mask: Boolean mask (True = valid, False = invalid)
        mask_value: Value to set for invalid actions (default: -1e9)
    
    Returns:
        Masked logits
    """
    if isinstance(logits, torch.Tensor):
        mask_tensor = torch.from_numpy(action_mask).to(logits.device).float()
        masked_logits = logits.clone()
        masked_logits[~mask_tensor.bool()] = mask_value
        return masked_logits
    else:
        masked_logits = logits.copy()
        masked_logits[~action_mask] = mask_value
        return masked_logits


def validate_action_mask(action_mask: np.ndarray, action_space_size: int) -> bool:
    """
    Validate that action mask is correct shape and has at least one valid action.
    
    Args:
        action_mask: Boolean mask
        action_space_size: Size of action space
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    if action_mask.shape != (action_space_size,):
        raise ValueError(
            f"Action mask shape {action_mask.shape} does not match "
            f"action space size {action_space_size}"
        )
    
    if not np.any(action_mask):
        raise ValueError("Action mask has no valid actions")
    
    return True


def get_valid_actions(action_mask: np.ndarray) -> np.ndarray:
    """
    Get array of valid action indices.
    
    Args:
        action_mask: Boolean mask
    
    Returns:
        Array of valid action indices
    """
    return np.where(action_mask)[0]


def sample_masked_action(
    logits: Union[torch.Tensor, np.ndarray],
    action_mask: np.ndarray,
    temperature: float = 1.0,
) -> int:
    """
    Sample action from masked logits.
    
    Args:
        logits: Action logits
        action_mask: Boolean mask
        temperature: Sampling temperature
    
    Returns:
        Sampled action index
    """
    masked_logits = apply_action_mask(logits, action_mask)
    
    if isinstance(masked_logits, torch.Tensor):
        masked_logits = masked_logits.cpu().numpy()
    
    # Apply temperature
    if temperature != 1.0:
        masked_logits = masked_logits / temperature
    
    # Softmax and sample
    probs = np.exp(masked_logits - np.max(masked_logits))
    probs = probs / probs.sum()
    action = np.random.choice(len(probs), p=probs)
    
    return int(action)
