# Testing Guide for Pişti RL Project

## Comprehensive Project Tests

To verify the entire project works correctly:

**Option 1: Pytest (Recommended)**
```bash
# Run full project test suite (tests all components)
pytest tests/test_full_project.py -v
```

**Option 2: Standalone Script**
```bash
# Run standalone test script
python scripts/test_full_project.py
```

This tests:
- ✓ Core engine (cards, rules, state)
- ✓ Observation encoding (MultiHot, CNN, Feature encoders)
- ✓ Environments (Gymnasium & PettingZoo)
- ✓ Baseline agents (Random, Greedy, PistiHunter)
- ✓ Probabilistic agent
- ✓ Reward functions (sparse and shaped)
- ✓ Integration between components

## Quick Tests (Lightweight)

To verify the probabilistic agent works correctly without taxing your computer, use these lightweight tests:

### Option 1: Run Quick Pytest Tests

```bash
# Run quick tests with minimal samples (fast)
pytest tests/test_probabilistic_quick.py -v

# Or run specific test
pytest tests/test_probabilistic_quick.py::test_probabilistic_agent_basic -v
```

### Option 2: Run Simple Test Script

```bash
# Simple standalone test
python scripts/test_probabilistic_simple.py
```

### Option 3: Manual Quick Test

```python
from agents.probabilistic_agent import ProbabilisticOptimalAgent
from agents.baselines import RandomValidAgent
from envs.pisti_gym import PistiGymEnv
import numpy as np

# Create agent with minimal config (fast)
agent = ProbabilisticOptimalAgent(max_samples=5, depth=1, seed=42)

# Test prediction
obs = {
    "hand": np.zeros(52),
    "table_top": np.zeros(52),
    "seen_cards": np.zeros(52),
    "action_mask": np.zeros(52, dtype=bool),
    "table_count": np.array([0]),
    "opp_captured_count": np.array([0]),
    "my_captured_count": np.array([0]),
    "stock_remaining": np.array([44]),
}
obs["hand"][0] = 1.0
obs["hand"][1] = 1.0
obs["action_mask"][0] = True
obs["action_mask"][1] = True

action = agent.predict(obs, obs["action_mask"])
print(f"Agent predicted action: {action}")  # Should be 0 or 1

# Test in environment (just 3 steps)
env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
obs, _ = env.reset(seed=42)
for _ in range(3):
    action_mask = obs["action_mask"]
    legal_actions = np.where(action_mask)[0]
    if len(legal_actions) == 0:
        break
    action = int(legal_actions[0])
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print("Environment test passed!")
```

## Performance Parameters

For lightweight testing, use these parameters:

- `max_samples=5`: Only sample 5 opponent hands (default is 50)
- `depth=1`: Single-step lookahead (default is 1)
- Limit game steps: Test with 3-5 steps instead of full games

## What to Check

1. **No crashes**: Agent should make predictions without errors
2. **Valid actions**: Actions should be in legal action set
3. **Reasonable behavior**: Agent should prefer captures when possible
4. **State tracking**: Belief tracker should update correctly

## Full Test Suite

For comprehensive testing (more computationally intensive):

```bash
# Run all probabilistic agent tests
pytest tests/test_probabilistic_agent.py -v

# Run with coverage
pytest tests/test_probabilistic_agent.py --cov=agents.probabilistic_agent
```

## Troubleshooting

If you encounter segfaults or numpy issues:
- Try updating numpy: `pip install --upgrade numpy`
- Use a fresh virtual environment
- Test with minimal samples first (`max_samples=3`)
- Check Python version compatibility (requires Python 3.8+)
