"""Model metadata management for checkpoints and reproducibility."""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from datetime import datetime
import sys
import platform
import subprocess


@dataclass
class ModelMetadata:
    """Metadata for a trained model checkpoint."""

    # Training configuration
    config_path: str
    training_config: Dict[str, Any]
    algorithm: str
    hyperparameters: Dict[str, Any]
    
    # Environment configuration
    encoder_type: str
    encoder_config: Dict[str, Any]
    reward_config: Dict[str, Any]
    game_config: Dict[str, Any]
    
    # Training statistics
    total_timesteps: int
    training_start_time: str
    training_end_time: Optional[str] = None
    best_eval_score: Optional[float] = None
    best_eval_step: Optional[int] = None
    
    # Reproducibility
    git_commit_hash: Optional[str] = None
    python_version: str = sys.version
    platform_info: str = platform.platform()
    package_versions: Dict[str, str] = None
    
    # Model architecture info
    policy_type: Optional[str] = None
    observation_space: Optional[Dict[str, Any]] = None
    action_space: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.package_versions is None:
            self.package_versions = {}
        if self.training_end_time is None:
            self.training_end_time = datetime.now().isoformat()


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = [
        "numpy",
        "gymnasium",
        "pettingzoo",
        "stable_baselines3",
        "torch",
    ]
    versions = {}
    for package in packages:
        try:
            mod = __import__(package)
            if hasattr(mod, "__version__"):
                versions[package] = mod.__version__
        except ImportError:
            pass
    return versions


def get_model_info(model) -> Dict[str, Any]:
    """
    Extract model architecture information.
    
    Args:
        model: SB3 model instance
    
    Returns:
        Dict with model architecture info
    """
    info = {
        "policy_type": None,
        "observation_space": None,
        "action_space": None,
    }
    
    if hasattr(model, "policy"):
        info["policy_type"] = str(type(model.policy).__name__)
    
    if hasattr(model, "observation_space"):
        try:
            obs_space = model.observation_space
            if hasattr(obs_space, "spaces"):
                # Dict space
                info["observation_space"] = {
                    key: {
                        "shape": list(space.shape) if hasattr(space, "shape") else None,
                        "dtype": str(space.dtype) if hasattr(space, "dtype") else None,
                    }
                    for key, space in obs_space.spaces.items()
                }
            else:
                info["observation_space"] = {
                    "shape": list(obs_space.shape) if hasattr(obs_space, "shape") else None,
                    "dtype": str(obs_space.dtype) if hasattr(obs_space, "dtype") else None,
                }
        except Exception:
            pass
    
    if hasattr(model, "action_space"):
        try:
            action_space = model.action_space
            info["action_space"] = {
                "n": action_space.n if hasattr(action_space, "n") else None,
                "dtype": str(action_space.dtype) if hasattr(action_space, "dtype") else None,
            }
        except Exception:
            pass
    
    return info


def create_metadata(
    config_path: str,
    config: Dict[str, Any],
    algorithm: str,
    model,
    total_timesteps: int,
    training_start_time: Optional[str] = None,
) -> ModelMetadata:
    """
    Create ModelMetadata from training configuration and model.
    
    Args:
        config_path: Path to config YAML file
        config: Loaded config dict
        algorithm: Algorithm name (e.g., "PPO", "DQN")
        model: SB3 model instance
        total_timesteps: Total training timesteps
        training_start_time: Training start time (ISO format)
    
    Returns:
        ModelMetadata instance
    """
    if training_start_time is None:
        training_start_time = datetime.now().isoformat()
    
    training_config = config.get("training", {})
    encoding_config = config.get("encoding", {})
    reward_config = config.get("reward", {})
    game_config = config.get("game", {})
    
    # Extract hyperparameters
    hyperparameters = {}
    if algorithm == "PPO" or algorithm == "MaskablePPO":
        hyperparameters = training_config.get("ppo", {})
    elif algorithm == "DQN":
        hyperparameters = training_config.get("dqn", {})
    
    # Get model info
    model_info = get_model_info(model)
    
    metadata = ModelMetadata(
        config_path=config_path,
        training_config=training_config,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        encoder_type=encoding_config.get("encoder_type", "MultiHotEncoder"),
        encoder_config=encoding_config,
        reward_config=reward_config,
        game_config=game_config,
        total_timesteps=total_timesteps,
        training_start_time=training_start_time,
        git_commit_hash=get_git_commit_hash(),
        package_versions=get_package_versions(),
        policy_type=model_info.get("policy_type"),
        observation_space=model_info.get("observation_space"),
        action_space=model_info.get("action_space"),
    )
    
    return metadata


def save_metadata(metadata: ModelMetadata, checkpoint_dir: str, checkpoint_name: str):
    """
    Save metadata as JSON file alongside checkpoint.
    
    Args:
        metadata: ModelMetadata instance
        checkpoint_dir: Directory where checkpoint is saved
        checkpoint_name: Name of checkpoint (without extension)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    metadata_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_metadata.json")
    
    # Convert to dict, handling datetime and other non-serializable types
    metadata_dict = asdict(metadata)
    
    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2, default=str)
    
    return metadata_path


def load_metadata(checkpoint_dir: str, checkpoint_name: str) -> Optional[ModelMetadata]:
    """
    Load metadata from checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint
        checkpoint_name: Name of checkpoint (without extension)
    
    Returns:
        ModelMetadata instance or None if not found
    """
    metadata_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_metadata.json")
    
    if not os.path.exists(metadata_path):
        return None
    
    with open(metadata_path, "r") as f:
        metadata_dict = json.load(f)
    
    # Reconstruct ModelMetadata
    return ModelMetadata(**metadata_dict)


def update_metadata(
    metadata: ModelMetadata,
    best_eval_score: Optional[float] = None,
    best_eval_step: Optional[int] = None,
    training_end_time: Optional[str] = None,
) -> ModelMetadata:
    """
    Update metadata with new information.
    
    Args:
        metadata: Existing ModelMetadata
        best_eval_score: Best evaluation score achieved
        best_eval_step: Step at which best score was achieved
        training_end_time: Training end time
    
    Returns:
        Updated ModelMetadata
    """
    if best_eval_score is not None:
        metadata.best_eval_score = best_eval_score
    if best_eval_step is not None:
        metadata.best_eval_step = best_eval_step
    if training_end_time is not None:
        metadata.training_end_time = training_end_time
    else:
        metadata.training_end_time = datetime.now().isoformat()
    
    return metadata
