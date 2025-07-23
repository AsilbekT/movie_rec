import os
import pickle
import random
import numpy as np
from collections import defaultdict
from typing import Optional


class QLearningAgent:
    def __init__(
        self,
        action_space: list[int],
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        min_epsilon: float = 0.01,
        decay_rate: float = 0.995
    ):
        self.q_table = defaultdict(float)  # (state_key, action) → Q-value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.actions = action_space

        # Memory to track last action for feedback updates
        self.last_action: dict[str, int] = {}

    def choose_action(self, state_key: str) -> int:
        """Choose an action using ε-greedy policy."""
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = [self.q_table[(state_key, a)] for a in self.actions]
            action = self.actions[np.argmax(q_values)]

        self.last_action[state_key] = action
        return action

    def update(self, state_key: str, action: int, reward: float, next_state_key: str):
        """Update Q-value using the Q-learning formula."""
        best_next = max([self.q_table[(next_state_key, a)] for a in self.actions], default=0)
        current = self.q_table[(state_key, action)]
        new_value = current + self.alpha * (reward + self.gamma * best_next - current)
        self.q_table[(state_key, action)] = new_value

        print(f"[Q-Learning] Updated: S={state_key}, A={action}, R={reward}, S'={next_state_key}, Q={new_value:.4f}")

    def decay_epsilon(self):
        """Decay exploration rate over time."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def save(self, path: str):
        """Save Q-table to file."""
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"[Q-Learning] Q-table saved to {path}")

    def load(self, path: str):
        """Load Q-table from file."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.q_table = defaultdict(float, pickle.load(f))
            print(f"[Q-Learning] Q-table loaded from {path}")
        else:
            print(f"[Q-Learning] No saved Q-table at {path}; starting fresh.")

    def get_q_value(self, state_key: str, action: int) -> float:
        return self.q_table.get((state_key, action), 0.0)

    def reset(self):
        """Reset Q-table and internal memory."""
        self.q_table.clear()
        self.last_action.clear()
        print("[Q-Learning] Agent reset.")
