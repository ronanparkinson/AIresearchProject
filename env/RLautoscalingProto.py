import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from simulator.workloadSimulator import workloadSimulator

class RLautoscalingProto(gym.Env):
    def __init__(self, datapath="data/borg_traces_data.csv", workloadData="average_usage", rewardVersion="v1"):
        super().__init__()
        self.workload = workloadSimulator(datapath, workloadData)
        self.rewardVersion = rewardVersion

        #scaling actions
        self.action_space = spaces.Discrete(3)
        #state
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.state = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.workload.reset()
        ingestedWorkLoad = self.workload.nextLoad()

        cpu = ingestedWorkLoad["cpu"]
        memory = ingestedWorkLoad["memory"]
        queue = 0.0
        instances = 0.5

        self.state = np.array([cpu, memory, queue, instances], dtype=np.float32)
        return self.state, {}

    def calcReward(self, queue, instances, scalingActions):
        if self.rewardVersion == "v1":
            return -(queue + 0.3 * instances)
        elif self.rewardVersion == "v2":
            return -(2.0 * queue + 0.3 * instances)
        elif self.rewardVersion == "v3":
            return -(queue + 0.3 * instances + 0.1 * scalingActions)
        else:
            raise ValueError("Unknown reward value: ", {self.rewardVersion})

    def step(self, action):
        cpu, memory, queue, instances = self.state

        prevInstances = instances

        if action == 0:
          instances -= 0.1
        if action == 2:
          instances += 0.1

        instances = np.clip(instances, 0.1, 1)

        scalingAction = abs(instances - prevInstances)

        ingestedWorkLoad = self.workload.nextLoad()

        cpu = 5 * ingestedWorkLoad["cpu"]
        memory = 5 * ingestedWorkLoad["memory"]

        cap = 0.8 * instances

        if cpu > cap:
            queue += 2.0 * (cpu - cap)
        else:
            queue -= 0.02

        queue = np.clip(queue, 0.0, 1.0)

        #cpu = np.clip(cpu + np.random.normal(0, 0.05), 0, 1)
        #queue = np.clip(queue + np.random.normal(0, 0.05), 0, 1)
        #ingestedWorkLoad = self.workload.nextLoad()

        self.state = np.array([cpu, memory, queue, instances], dtype=np.float32)

        reward = self.calcReward(queue, instances, scalingAction)

        return self.state, reward, False, False, {}

    #-----------testing below, remove after----------------

print("RLautoscalingProto testing starting")

#import numpy as np
#from env.RLautoscalingProto import RLautoscalingProto

#env = RLautoscalingProto()

#state, _ = env.reset()
#print("Initial state:", state)

#for step in range(20):
 #   action = np.random.choice([0, 1, 2])
 #   state, reward, done, truncated, info = env.step(action)
 #   print(f"Step {step}: action={action}, state={state}, reward={reward:.4f}")

#print("RLautoscalingProto testing finishing")
