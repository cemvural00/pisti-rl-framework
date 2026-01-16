"""Modular observation encoders for Pişti RL environment."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
from engine.state import GameState
from encoding.obs_builder import ObsBuilder


class ObservationEncoder(ABC):
    """Abstract base class for all observation encoders."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize encoder.
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self.obs_builder = ObsBuilder()

    @abstractmethod
    def encode(self, state: GameState, player_id: int) -> Dict[str, np.ndarray]:
        """
        Encode game state into observation for a player.
        
        Args:
            state: Current game state
            player_id: Player to encode observation for
        
        Returns:
            Dict mapping feature names to numpy arrays
        """
        pass

    def get_observation_space_dict(self) -> Dict:
        """
        Get observation space as dict (for Gymnasium spaces.Dict).
        
        Returns:
            Dict mapping feature names to space objects
        """
        from gymnasium import spaces
        
        # Base features
        obs_space = {
            "hand": spaces.Box(low=0, high=1, shape=(52,), dtype=np.float32),
            "table_top": spaces.Box(low=0, high=1, shape=(52,), dtype=np.float32),
            "seen_cards": spaces.Box(low=0, high=1, shape=(52,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(52),
            "score_breakdown": spaces.Box(
                low=0, high=100, shape=(6,), dtype=np.float32
            ),
            "table_count": spaces.Box(low=0, high=52, shape=(1,), dtype=np.float32),
            "my_captured_count": spaces.Box(
                low=0, high=52, shape=(1,), dtype=np.float32
            ),
            "opp_captured_count": spaces.Box(
                low=0, high=52, shape=(1,), dtype=np.float32
            ),
            "stock_remaining": spaces.Box(low=0, high=52, shape=(1,), dtype=np.float32),
            "hand_size": spaces.Box(low=0, high=52, shape=(1,), dtype=np.float32),
            "last_capture_by": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "last_move_card": spaces.Box(low=0, high=1, shape=(52,), dtype=np.float32),
            "running_score_estimate": spaces.Box(
                low=-100, high=100, shape=(1,), dtype=np.float32
            ),
        }
        return obs_space


class MultiHotEncoder(ObservationEncoder):
    """
    Default encoder: returns dict with 52-length multi-hot vectors.
    
    Never uses raw integer IDs as NN inputs.
    """

    def encode(self, state: GameState, player_id: int) -> Dict[str, np.ndarray]:
        """Encode state using multi-hot vectors."""
        obs = {
            "hand": self.obs_builder.build_hand(state, player_id),
            "table_top": self.obs_builder.build_table_top(state),
            "seen_cards": self.obs_builder.build_seen_cards(state, player_id),
            "action_mask": self.obs_builder.build_action_mask(state, player_id),
            "score_breakdown": self.obs_builder.build_score_breakdown(state, player_id),
        }
        
        # Add scalar features
        scalar_features = self.obs_builder.build_scalar_features(state, player_id)
        obs.update(scalar_features)
        
        return obs


class CNNEncoder(ObservationEncoder):
    """
    Encoder with (4,13) reshaped tensor views for CNN experiments.
    
    Extends MultiHotEncoder and adds reshaped views.
    """

    def encode(self, state: GameState, player_id: int) -> Dict[str, np.ndarray]:
        """Encode state with CNN-friendly reshaped views."""
        # Get base multi-hot vectors
        base_encoder = MultiHotEncoder(self.config)
        base_obs = base_encoder.encode(state, player_id)
        
        # Add reshaped views: (4, 13) = (suit, rank)
        hand_multi_hot = base_obs["hand"]
        table_top_multi_hot = base_obs["table_top"]
        seen_cards_multi_hot = base_obs["seen_cards"]
        
        # Reshape: card_id = suit_id * 13 + rank_id
        # So we can reshape (52,) -> (4, 13) where first dim is suit, second is rank
        base_obs["hand_cnn"] = hand_multi_hot.reshape(4, 13)
        base_obs["table_top_cnn"] = table_top_multi_hot.reshape(4, 13)
        base_obs["seen_cards_cnn"] = seen_cards_multi_hot.reshape(4, 13)
        
        return base_obs

    def get_observation_space_dict(self) -> Dict:
        """Get observation space including CNN views."""
        from gymnasium import spaces
        
        obs_space = super().get_observation_space_dict()
        obs_space.update(
            {
                "hand_cnn": spaces.Box(low=0, high=1, shape=(4, 13), dtype=np.float32),
                "table_top_cnn": spaces.Box(
                    low=0, high=1, shape=(4, 13), dtype=np.float32
                ),
                "seen_cards_cnn": spaces.Box(
                    low=0, high=1, shape=(4, 13), dtype=np.float32
                ),
            }
        )
        return obs_space


class FeatureEncoder(ObservationEncoder):
    """
    Flattens observation dict to single vector for MLP policies.
    
    Concatenates all features into one flat vector.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature encoder."""
        super().__init__(config)
        self.include_features = (config or {}).get("include_features", "all")
        if self.include_features == "all":
            self.include_features = [
                "hand",
                "table_top",
                "seen_cards",
                "score_breakdown",
                "table_count",
                "my_captured_count",
                "opp_captured_count",
                "stock_remaining",
                "hand_size",
                "last_capture_by",
                "last_move_card",
                "running_score_estimate",
            ]

    def encode(self, state: GameState, player_id: int) -> Dict[str, np.ndarray]:
        """Encode state as flattened feature vector."""
        # Get base observation
        base_encoder = MultiHotEncoder(self.config)
        base_obs = base_encoder.encode(state, player_id)
        
        # Flatten selected features
        feature_list = []
        for feat_name in self.include_features:
            if feat_name in base_obs:
                feat = base_obs[feat_name]
                feature_list.append(feat.flatten())
        
        # Concatenate
        flat_features = np.concatenate(feature_list).astype(np.float32)
        
        return {"features": flat_features, "action_mask": base_obs["action_mask"]}

    def get_observation_space_dict(self) -> Dict:
        """Get observation space as flat vector."""
        from gymnasium import spaces
        
        # Calculate size
        base_encoder = MultiHotEncoder(self.config)
        base_obs_space = base_encoder.get_observation_space_dict()
        
        size = 0
        for feat_name in self.include_features:
            if feat_name in base_obs_space:
                space = base_obs_space[feat_name]
                if hasattr(space, "shape"):
                    size += np.prod(space.shape)
        
        return {
            "features": spaces.Box(
                low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32
            ),
            "action_mask": spaces.MultiBinary(52),
        }


class SequenceEncoder(ObservationEncoder):
    """
    Encoder for recurrent policies (LSTM/RNN).
    
    Returns dict with features (flat) + history (sequence).
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize sequence encoder."""
        super().__init__(config)
        self.history_length = config.get("history_length", 10)
        self.feature_encoder = FeatureEncoder(config)

    def encode(self, state: GameState, player_id: int) -> Dict[str, np.ndarray]:
        """Encode state with move history sequence."""
        # Get flat features
        flat_obs = self.feature_encoder.encode(state, player_id)
        
        # Build history sequence from move_history
        history = []
        move_history = state.move_history[-self.history_length :]
        
        for player, card, captured in move_history:
            # Encode move as: [player_id, card_id (one-hot 52), captured (0/1)]
            move_vec = np.zeros(54, dtype=np.float32)  # 1 + 52 + 1
            move_vec[0] = float(player)
            card_id = card_to_id(card)
            move_vec[1 + card_id] = 1.0
            move_vec[53] = float(captured)
            history.append(move_vec)
        
        # Pad if needed
        while len(history) < self.history_length:
            history.insert(0, np.zeros(54, dtype=np.float32))
        
        history_array = np.array(history, dtype=np.float32)
        
        return {
            "features": flat_obs["features"],
            "history": history_array,
            "action_mask": flat_obs["action_mask"],
        }

    def get_observation_space_dict(self) -> Dict:
        """Get observation space with history."""
        from gymnasium import spaces
        
        base_space = self.feature_encoder.get_observation_space_dict()
        base_space["history"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.history_length, 54),
            dtype=np.float32,
        )
        return base_space
