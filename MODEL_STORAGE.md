# Model Storage Guide

## Overview

The framework uses a structured `models/` directory to organize all model parameters, checkpoints, and metadata. This provides clear separation between intermediate checkpoints, final models, and snapshots for self-play.

## Directory Structure

```
models/
├── {algorithm}/              # Algorithm-specific folder (e.g., ppo, nfsp, deep_cfr)
│   ├── checkpoints/          # Intermediate checkpoints during training
│   │   ├── pisti_model_50000_steps.zip
│   │   ├── pisti_model_50000_steps_metadata.json
│   │   ├── pisti_model_100000_steps.zip
│   │   └── ...
│   ├── final/                # Final trained models
│   │   ├── pisti_model_final.zip
│   │   ├── pisti_model_final_metadata.json
│   │   └── ...
│   └── snapshots/            # Snapshots for self-play/league training
│       ├── snapshot_50000_steps.zip
│       ├── snapshot_100000_steps.zip
│       └── ...
└── shared/                   # Shared checkpoints (opponent pool)
    └── checkpoints/
        └── ...
```

## Algorithm Folders

Each algorithm gets its own folder:
- `models/ppo/` - PPO models
- `models/maskableppo/` - MaskablePPO models
- `models/recurrentppo/` - RecurrentPPO models
- `models/dqn/` - DQN models
- `models/rainbowdqn/` - RainbowDQN models
- `models/nfsp/` - NFSP models
- `models/deep_cfr/` - Deep CFR models
- `models/r2d2/` - R2D2 models

## File Types

### SB3 Models (PPO, DQN, etc.)
- **Checkpoints**: `.zip` files (SB3 format)
- **Metadata**: `{checkpoint_name}_metadata.json`

### Custom RL Agents (NFSP, Deep CFR, R2D2)
- **Checkpoints**: `.pt` files (PyTorch format)
- **Metadata**: `{checkpoint_name}_metadata.json` (if implemented)

## Configuration

In `configs/default.yaml`:

```yaml
logging:
  models_dir: "./models"  # Base directory for models
  save_freq: 50000        # Save checkpoint every N steps
```

The framework automatically:
1. Creates algorithm-specific subdirectories
2. Saves checkpoints to `{algorithm}/checkpoints/`
3. Saves final models to `{algorithm}/final/`
4. Saves snapshots to `{algorithm}/snapshots/`

## Using Model Storage

### Training

Models are automatically saved to the correct directories:

```bash
# Training automatically creates:
# models/ppo/checkpoints/pisti_model_50000_steps.zip
# models/ppo/final/pisti_model_final.zip
python -m training.train_sb3 --config configs/default.yaml
```

### Finding Models

Use the model storage utilities:

```python
from training.model_storage import find_model, list_models

# Find final model
final_model = find_model("ppo", "final")
# Returns: "models/ppo/final/pisti_model_final.zip"

# List all models
all_models = list_models("ppo")
# Returns: {
#   "checkpoints": [...],
#   "final": [...],
#   "snapshots": [...]
# }
```

### Evaluation

Evaluation scripts automatically look in the models directory:

```bash
# Can use full path or just model name
python -m training.evaluate_comprehensive \
    --checkpoint models/ppo/final/pisti_model_final \
    --config configs/default.yaml
```

## Migration from Old Structure

If you have existing models in `checkpoints/`, you can:

1. **Keep using old structure**: Set `save_path` in config (deprecated but still works)
2. **Migrate manually**: Move files to new structure
3. **Use both**: The code supports both `models_dir` and `save_path` for backward compatibility

## Benefits

1. **Organization**: Clear separation by algorithm
2. **Scalability**: Easy to manage many experiments
3. **Clarity**: Final models separate from intermediate checkpoints
4. **Self-play**: Snapshots organized separately
5. **Metadata**: Always alongside model files
