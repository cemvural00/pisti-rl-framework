"""Network architecture definitions for RL agents."""

from typing import List, Dict, Optional, Any
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import create_mlp


def get_network_arch(config: Dict[str, Any], arch_type: str = "deep") -> Dict[str, List[int]]:
    """
    Get network architecture from config.
    
    Args:
        config: Training config dict
        arch_type: "deep", "medium", "shallow", "recurrent", "nfsp", "deep_cfr"
    
    Returns:
        Dict with network architecture specifications
    """
    arch_configs = config.get("network_architectures", {})
    
    if arch_type == "deep":
        default = {
            "pi": [256, 256, 128],
            "vf": [256, 256, 128],
            "qf": [256, 256, 128],
        }
        return arch_configs.get("deep", default)
    elif arch_type == "medium":
        default = {
            "pi": [128, 128],
            "vf": [128, 128],
            "qf": [128, 128],
        }
        return arch_configs.get("medium", default)
    elif arch_type == "shallow":
        default = {
            "pi": [64, 64],
            "vf": [64, 64],
            "qf": [64, 64],
        }
        return arch_configs.get("shallow", default)
    elif arch_type == "recurrent":
        default = {
            "lstm_hidden_size": 256,
            "lstm_layers": 2,
            "mlp_layers": [128, 128],
        }
        return arch_configs.get("recurrent", default)
    elif arch_type == "nfsp":
        default = {
            "average_strategy": [256, 256, 128],
            "best_response": [256, 256, 128],
        }
        return arch_configs.get("nfsp", default)
    elif arch_type == "deep_cfr":
        default = {
            "counterfactual_value": [256, 256, 256, 128],
            "strategy": [256, 256, 128],
        }
        return arch_configs.get("deep_cfr", default)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


def create_deep_policy_network(
    input_dim: int,
    output_dim: int,
    net_arch: List[int] = [256, 256, 128],
    activation_fn: type = nn.ReLU,
) -> nn.Module:
    """
    Create deep policy network.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension (action space size)
        net_arch: List of hidden layer sizes
        activation_fn: Activation function class
    
    Returns:
        PyTorch module
    """
    layers = create_mlp(input_dim, output_dim, net_arch, activation_fn)
    return nn.Sequential(*layers)


def create_recurrent_policy_network(
    input_dim: int,
    output_dim: int,
    lstm_hidden_size: int = 256,
    lstm_layers: int = 2,
    mlp_layers: List[int] = [128, 128],
    activation_fn: type = nn.ReLU,
) -> nn.Module:
    """
    Create recurrent policy network with LSTM.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension (action space size)
        lstm_hidden_size: LSTM hidden state size
        lstm_layers: Number of LSTM layers
        mlp_layers: MLP layers after LSTM
        activation_fn: Activation function class
    
    Returns:
        PyTorch module
    """
    class RecurrentPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                lstm_hidden_size,
                lstm_layers,
                batch_first=True,
            )
            lstm_output_dim = lstm_hidden_size
            mlp = create_mlp(lstm_output_dim, output_dim, mlp_layers, activation_fn)
            self.mlp = nn.Sequential(*mlp)
            self.hidden = None
        
        def forward(self, x, hidden=None):
            # x shape: (batch, seq_len, input_dim) or (batch, input_dim)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            lstm_out, hidden = self.lstm(x, hidden)
            # Take last output
            lstm_out = lstm_out[:, -1, :]
            output = self.mlp(lstm_out)
            return output, hidden
    
    return RecurrentPolicy()


def create_value_network(
    input_dim: int,
    net_arch: List[int] = [256, 256, 128],
    activation_fn: type = nn.ReLU,
) -> nn.Module:
    """
    Create value network.
    
    Args:
        input_dim: Input dimension
        net_arch: List of hidden layer sizes
        activation_fn: Activation function class
    
    Returns:
        PyTorch module (outputs scalar value)
    """
    layers = create_mlp(input_dim, 1, net_arch, activation_fn)
    return nn.Sequential(*layers)


def create_q_network(
    input_dim: int,
    output_dim: int,
    net_arch: List[int] = [256, 256, 128],
    activation_fn: type = nn.ReLU,
) -> nn.Module:
    """
    Create Q-network.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension (action space size)
        net_arch: List of hidden layer sizes
        activation_fn: Activation function class
    
    Returns:
        PyTorch module
    """
    return create_deep_policy_network(input_dim, output_dim, net_arch, activation_fn)
