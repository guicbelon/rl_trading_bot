import numpy as np

class PVM:
    def __init__(self, environment):
        """Initializes portfolio vector memory.

        Args:
          capacity: Max capacity of memory.
        """
        # initially, memory will have the same actions
        self.capacity = environment.episode_length
        self.portfolio_size = environment.portfolio_size
        self.reset()

    def reset(self):
        self.memory = [np.array([1] + [0] * (self.portfolio_size), dtype=np.float32)] * (self.capacity + 1)
        self.index = 0 # initial index to retrieve data

    def retrieve(self):
        last_action = self.memory[self.index]
        self.index = 0 if self.index == self.capacity else self.index + 1
        return last_action

    def add(self, action):
        self.memory[self.index] = action
