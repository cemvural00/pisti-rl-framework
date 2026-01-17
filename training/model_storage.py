"""Model storage utilities for organizing checkpoints and final models."""

import os
from typing import Optional, Dict, Any
from datetime import datetime


def get_model_directories(base_dir: str = ".", algorithm: Optional[str] = None) -> Dict[str, str]:
    """
    Get standardized model directory structure.
    
    Structure:
    models/
    ├── {algorithm}/          # Algorithm-specific folder
    │   ├── checkpoints/       # Intermediate checkpoints during training
    │   ├── final/            # Final trained models
    │   └── snapshots/         # Snapshots for self-play/league
    └── shared/               # Shared checkpoints (for opponent pool)
        └── checkpoints/
    
    Args:
        base_dir: Base directory (default: current directory)
        algorithm: Algorithm name (e.g., "PPO", "NFSP", "DeepCFR")
    
    Returns:
        Dict with directory paths:
        - models_dir: Base models directory
        - checkpoints_dir: For intermediate checkpoints
        - final_dir: For final models
        - snapshots_dir: For snapshots (self-play)
        - shared_checkpoints_dir: Shared checkpoints
    """
    models_dir = os.path.join(base_dir, "models")
    
    if algorithm:
        algorithm_dir = os.path.join(models_dir, algorithm.lower())
        checkpoints_dir = os.path.join(algorithm_dir, "checkpoints")
        final_dir = os.path.join(algorithm_dir, "final")
        snapshots_dir = os.path.join(algorithm_dir, "snapshots")
    else:
        checkpoints_dir = os.path.join(models_dir, "checkpoints")
        final_dir = os.path.join(models_dir, "final")
        snapshots_dir = os.path.join(models_dir, "snapshots")
    
    shared_checkpoints_dir = os.path.join(models_dir, "shared", "checkpoints")
    
    return {
        "models_dir": models_dir,
        "checkpoints_dir": checkpoints_dir,
        "final_dir": final_dir,
        "snapshots_dir": snapshots_dir,
        "shared_checkpoints_dir": shared_checkpoints_dir,
    }


def ensure_model_directories(base_dir: str = ".", algorithm: Optional[str] = None) -> Dict[str, str]:
    """
    Ensure all model directories exist.
    
    Args:
        base_dir: Base directory
        algorithm: Algorithm name
    
    Returns:
        Dict with directory paths (same as get_model_directories)
    """
    dirs = get_model_directories(base_dir, algorithm)
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def get_checkpoint_path(
    algorithm: str,
    timestep: Optional[int] = None,
    name_prefix: str = "model",
    is_final: bool = False,
    base_dir: str = ".",
) -> str:
    """
    Get standardized checkpoint path.
    
    Args:
        algorithm: Algorithm name
        timestep: Training timestep (for intermediate checkpoints)
        name_prefix: Name prefix
        is_final: Whether this is a final model
        base_dir: Base directory
    
    Returns:
        Full path to checkpoint
    """
    dirs = ensure_model_directories(base_dir, algorithm)
    
    if is_final:
        save_dir = dirs["final_dir"]
        filename = f"{name_prefix}_final"
    else:
        save_dir = dirs["checkpoints_dir"]
        if timestep is not None:
            filename = f"{name_prefix}_{timestep}_steps"
        else:
            filename = f"{name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return os.path.join(save_dir, filename)


def get_snapshot_path(
    algorithm: str,
    timestep: int,
    base_dir: str = ".",
) -> str:
    """
    Get snapshot path for self-play/league training.
    
    Args:
        algorithm: Algorithm name
        timestep: Training timestep
        base_dir: Base directory
    
    Returns:
        Full path to snapshot
    """
    dirs = ensure_model_directories(base_dir, algorithm)
    filename = f"snapshot_{timestep}_steps"
    return os.path.join(dirs["snapshots_dir"], filename)


def get_shared_checkpoint_path(
    checkpoint_name: str,
    base_dir: str = ".",
) -> str:
    """
    Get path for shared checkpoint (opponent pool).
    
    Args:
        checkpoint_name: Name of checkpoint
        base_dir: Base directory
    
    Returns:
        Full path to shared checkpoint
    """
    dirs = ensure_model_directories(base_dir)
    return os.path.join(dirs["shared_checkpoints_dir"], checkpoint_name)


def find_model(
    algorithm: str,
    model_name: str = "final",
    base_dir: str = ".",
) -> Optional[str]:
    """
    Find a model file.
    
    Args:
        algorithm: Algorithm name
        model_name: Model name (e.g., "final", "pisti_model_final", or specific checkpoint name)
        base_dir: Base directory
    
    Returns:
        Full path to model file, or None if not found
    """
    dirs = get_model_directories(base_dir, algorithm)
    
    # Try final directory first
    if model_name == "final" or "final" in model_name.lower():
        final_dir = dirs["final_dir"]
        if os.path.exists(final_dir):
            for f in os.listdir(final_dir):
                if model_name in f and not f.endswith("_metadata.json"):
                    return os.path.join(final_dir, f)
    
    # Try checkpoints directory
    checkpoints_dir = dirs["checkpoints_dir"]
    if os.path.exists(checkpoints_dir):
        for f in os.listdir(checkpoints_dir):
            if model_name in f and not f.endswith("_metadata.json"):
                return os.path.join(checkpoints_dir, f)
    
    # Try snapshots directory
    snapshots_dir = dirs["snapshots_dir"]
    if os.path.exists(snapshots_dir):
        for f in os.listdir(snapshots_dir):
            if model_name in f and not f.endswith("_metadata.json"):
                return os.path.join(snapshots_dir, f)
    
    return None


def list_models(algorithm: Optional[str] = None, base_dir: str = ".") -> Dict[str, list]:
    """
    List all saved models.
    
    Args:
        algorithm: Algorithm name (None for all)
        base_dir: Base directory
    
    Returns:
        Dict with lists of model paths:
        - checkpoints: Intermediate checkpoints
        - final: Final models
        - snapshots: Snapshots
    """
    dirs = get_model_directories(base_dir, algorithm)
    
    models = {
        "checkpoints": [],
        "final": [],
        "snapshots": [],
    }
    
    # List checkpoints
    if os.path.exists(dirs["checkpoints_dir"]):
        models["checkpoints"] = [
            os.path.join(dirs["checkpoints_dir"], f)
            for f in os.listdir(dirs["checkpoints_dir"])
            if os.path.isfile(os.path.join(dirs["checkpoints_dir"], f))
        ]
    
    # List final models
    if os.path.exists(dirs["final_dir"]):
        models["final"] = [
            os.path.join(dirs["final_dir"], f)
            for f in os.listdir(dirs["final_dir"])
            if os.path.isfile(os.path.join(dirs["final_dir"], f))
        ]
    
    # List snapshots
    if os.path.exists(dirs["snapshots_dir"]):
        models["snapshots"] = [
            os.path.join(dirs["snapshots_dir"], f)
            for f in os.listdir(dirs["snapshots_dir"])
            if os.path.isfile(os.path.join(dirs["snapshots_dir"], f))
        ]
    
    return models
