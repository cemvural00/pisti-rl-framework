"""Gymnasium wrapper for single-agent training with pluggable opponent."""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from gymnasium import Env, spaces
from gymnasium.core import ActType, ObsType

from envs.base import PistiGameEngine
from encoding.encoders import ObservationEncoder, MultiHotEncoder
from agents.baselines import RandomValidAgent


class PistiGymEnv(Env):
    """Gymnasium environment for single-agent Pişti training."""

    metadata = {"render_modes": ["human"], "name": "Pisti-v0"}

    def __init__(
        self,
        encoder: Optional[ObservationEncoder] = None,
        reward_config: Optional[Dict] = None,
        game_config: Optional[Dict] = None,
        opponent=None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Gymnasium environment.
        
        Args:
            encoder: Observation encoder
            reward_config: Reward configuration
            game_config: Game configuration
            opponent: Opponent policy (must have predict(obs, action_mask) method)
                     Default: RandomValidAgent
            render_mode: Render mode
            seed: Random seed
        """
        super().__init__()
        
        self.engine = PistiGameEngine(
            encoder=encoder or MultiHotEncoder(),
            reward_config=reward_config,
            game_config=game_config,
            seed=seed,
        )
        
        self.opponent = opponent or RandomValidAgent()
        self.render_mode = render_mode
        
        # Set up observation and action spaces
        obs_space_dict = self.engine.encoder.get_observation_space_dict()
        self.observation_space = spaces.Dict(obs_space_dict)
        self.action_space = spaces.Discrete(52)
        
        self._last_obs = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Reset environment.
        
        Args:
            seed: Random seed
            options: Optional reset options
        
        Returns:
            (observation, info)
        """
        self.engine.reset(seed=seed)
        
        # Get observation for learning agent (player_0)
        obs = self.engine.get_observation(0)
        self._last_obs = obs
        
        info = {}
        
        # If opponent starts, make opponent move first
        if self.engine.state.current_player == 1:
            obs, _, _, info = self._opponent_step()
        
        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Step environment with action from learning agent.
        
        Args:
            action: Card ID (0-51)
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.engine.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        # Convert action to Python int if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item() if action.size == 1 else action[0])
        else:
            action = int(action)
        
        # Learning agent (player_0) makes move
        new_state, reward, done, info = self.engine.step(action)
        
        # Note: State is already updated in engine, so we pass the current state
        # Update agent state if it has the method (for probabilistic agents)
        if hasattr(self.opponent, "update_state"):
            self.opponent.update_state(self.engine.state)
        
        # If not done, opponent (player_1) makes move
        if not done and self.engine.state.current_player == 1:
            obs, opp_reward, done, opp_info = self._opponent_step()
            # Opponent's reward is negative of what we'd give them
            # But we only care about our reward, so ignore opp_reward
            info.update(opp_info)
        else:
            obs = self.engine.get_observation(0)
        
        self._last_obs = obs
        
        # CRITICAL: Always check if state is terminal, regardless of done flag
        # This ensures terminal state is detected even if done flag wasn't set correctly
        if self.engine.state is not None:
            is_terminal = self.engine.state.is_terminal()
            if is_terminal:
                done = True  # Override done flag if state is terminal
                # Ensure info contains final scores when terminal
                if "score_diff" not in info and "final_scores" not in info:
                    try:
                        scores = self.engine.state.get_final_scores()
                        info["final_scores"] = scores
                        info["score_diff"] = scores[0] - scores[1]
                    except Exception:
                        pass
        
        # Gymnasium uses terminated and truncated separately
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info

    def _opponent_step(self) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Make opponent move.
        
        Returns:
            (observation, reward, done, info)
        """
        # Get observation for opponent
        opp_obs = self.engine.get_observation(1)
        action_mask = opp_obs["action_mask"]
        
        # If opponent has update_state method, provide current state
        if hasattr(self.opponent, "update_state"):
            self.opponent.update_state(self.engine.state)
        
        # Opponent predicts action
        opp_action = self.opponent.predict(opp_obs, action_mask)
        
        # Apply opponent action
        new_state, opp_reward, done, info = self.engine.step(opp_action)
        
        # Get observation for learning agent (after opponent move)
        if not done:
            obs = self.engine.get_observation(0)
        else:
            obs = self._last_obs  # Use last observation if done
        
        return obs, opp_reward, done, info

    def render(self) -> None:
        """Render environment."""
        if self.render_mode == "human":
            if self.engine.state:
                print(f"Current player: {self.engine.state.current_player}")
                print(f"Table pile size: {len(self.engine.state.table_pile)}")
                if self.engine.state.table_pile:
                    print(f"Top card: {self.engine.state.table_pile[-1]}")
                print(f"Player 0 hand size: {len(self.engine.state.hands[0])}")
                print(f"Player 1 hand size: {len(self.engine.state.hands[1])}")

    def close(self) -> None:
        """Close environment."""
        pass
    
    def action_masks(self) -> np.ndarray:
        """
        Return action mask for MaskablePPO compatibility.
        
        Returns:
            (52,) boolean array where True = valid action, False = invalid
        """
        if self.engine.state is None:
            return np.ones(52, dtype=bool)  # All actions valid if not initialized
        
        # Get action mask for current player (player_0 is the learning agent)
        obs = self.engine.get_observation(0)
        return obs.get("action_mask", np.ones(52, dtype=bool))
