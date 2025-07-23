import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, action_space: list[int], alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = defaultdict(float)  # (state, action) → Q-value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = action_space

    def choose_action(self, state_key: str) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.q_table[(state_key, a)] for a in self.actions]
        return self.actions[np.argmax(q_values)]

    def update(self, state_key: str, action: int, reward: float, next_state_key: str):
        best_next = max([self.q_table[(next_state_key, a)] for a in self.actions], default=0)
        current = self.q_table[(state_key, action)]
        self.q_table[(state_key, action)] = current + self.alpha * (reward + self.gamma * best_next - current)
