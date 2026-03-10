import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from simulator.workloadSimulator import workloadSimulator

class RLautoscalingProto(gym.Env):
    def __init__(self):
        super().__init__()
        self.workload = workloadSimulator()

        #scaling actions
        self.action_space = spaces.Discrete(3)
        #state
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.state = np.array([0.5, 0.2, 0.5], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ingestedWorkLoad = self.workload.nextLoad()
        self.state = np.array([ingestedWorkLoad, 0.5, 0.2, 0.5], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        ingestedWorkLoad, cpu, queue, instances = self.state

        if action == 0:
          instances -= 0.1
        if action == 2:
          instances += 0.1

        instances = np.clip(instances, 0, 1)
        cpu = np.clip(cpu + np.random.normal(0, 0.05), 0, 1)
        queue = np.clip(queue + np.random.normal(0, 0.05), 0, 1)
        ingestedWorkLoad = self.workload.nextLoad()

        self.state = np.array([ingestedWorkLoad, cpu, queue, instances], dtype=np.float32)

        reward = -(queue + 0.3 * instances)

        return self.state, reward, False, False, {}

env = RLautoscalingProto()

state, info = env.reset()

for _ in range(5):
  action = env.action_space.sample()
  state, reward, done, truncated, info = env.step(action)
  print("State:", state, "Reward:", reward)