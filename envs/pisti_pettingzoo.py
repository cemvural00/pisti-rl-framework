"""PettingZoo AEC environment for Pişti."""

from typing import Optional, Dict, Any
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
try:
    # Try importing AgentSelector class (newer versions)
    from pettingzoo.utils.agent_selector import AgentSelector
    agent_selector = AgentSelector
except ImportError:
    try:
        # Try importing as function (older versions)
        from pettingzoo.utils.agent_selector import agent_selector
    except ImportError:
        # Fallback
        from pettingzoo.utils import agent_selector

from envs.base import PistiGameEngine
from encoding.encoders import ObservationEncoder, MultiHotEncoder


class PistiPettingZooEnv(AECEnv):
    """PettingZoo AEC environment for Pişti card game."""

    metadata = {"render_modes": ["human"], "name": "pisti_v0"}

    def __init__(
        self,
        encoder: Optional[ObservationEncoder] = None,
        reward_config: Optional[Dict] = None,
        game_config: Optional[Dict] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize PettingZoo environment.
        
        Args:
            encoder: Observation encoder
            reward_config: Reward configuration
            game_config: Game configuration
            render_mode: Render mode (currently only "human" supported)
            seed: Random seed
        """
        super().__init__()
        
        self.engine = PistiGameEngine(
            encoder=encoder or MultiHotEncoder(),
            reward_config=reward_config,
            game_config=game_config,
            seed=seed,
        )
        
        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents.copy()
        self.render_mode = render_mode
        
        # Set up observation and action spaces
        obs_space_dict = self.engine.encoder.get_observation_space_dict()
        self.observation_spaces = {
            agent: spaces.Dict(obs_space_dict) for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(52) for agent in self.agents
        }
        
        self._agent_selector = None
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for agent."""
        return self.action_spaces[agent]

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        """
        Get observation for agent.
        
        Args:
            agent: Agent name ("player_0" or "player_1")
        
        Returns:
            Observation dict
        """
        player_id = int(agent.split("_")[1])
        return self.engine.get_observation(player_id)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Reset environment.
        
        Args:
            seed: Random seed
            options: Optional reset options
        
        Returns:
            Dict with observations for each agent
        """
        self.engine.reset(seed=seed)
        self.agents = self.possible_agents.copy()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        # Initialize rewards, terminations, truncations, infos
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        observations = {
            agent: self.observe(agent) for agent in self.agents
        }
        
        return observations, self.infos

    def step(self, action: int) -> None:
        """
        Step environment with action.
        
        Args:
            action: Card ID (0-51)
        """
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return
        
        # Get current player ID
        player_id = int(self.agent_selection.split("_")[1])
        
        # Apply action
        new_state, reward, done, info = self.engine.step(action)
        
        # Store rewards and terminations
        self.rewards[self.agent_selection] = reward
        self.terminations[self.agent_selection] = done
        
        # If done, terminate all agents
        if done:
            self.terminations = {agent: True for agent in self.agents}
            # Give final reward to both players based on score differential
            scores = info.get("final_scores", {0: 0, 1: 0})
            score_diff = scores[0] - scores[1]
            self.rewards["player_0"] = float(score_diff)
            self.rewards["player_1"] = float(-score_diff)
        
        # Move to next agent
        self.agent_selection = self._agent_selector.next()

    def render(self) -> None:
        """Render environment (placeholder)."""
        if self.render_mode == "human":
            # Simple text rendering
            if self.engine.state:
                print(f"Current player: {self.engine.state.current_player}")
                print(f"Table pile size: {len(self.engine.state.table_pile)}")
                if self.engine.state.table_pile:
                    print(f"Top card: {self.engine.state.table_pile[-1]}")
