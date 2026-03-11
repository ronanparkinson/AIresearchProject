from stable_baselines3 import PPO
from env.RLautoscalingProto import RLautoscalingProto

class ppoEvaluation:
    def __init__(self, modelPath="../training/PPOscaler"):
        self.model = PPO.load(modelPath)
        self.env = RLautoscalingProto()

    def run(self, steps=200):
        state, _ = self.env.reset()

        totalReward = 0
        totalQueue = 0
        totalCpu = 0
        totalInstances = 0
        scalingActions = 0

        for _ in range(steps):
            action, _ = self.model.predict(state, deterministic=True)

            if action != 1:
                scalingActions += 1

            state, reward, done, truncated, info = self.env.step(action)
            workload, cpu, queue, instances = state

            totalReward += reward
            totalQueue += queue
            totalCpu += cpu
            totalInstances += instances

        return {
            "reward": totalReward,
            "average queue": totalQueue / steps,
            "average cpu": totalCpu / steps,
            "average instances": totalInstances / steps,
            "scaling actions": scalingActions
        }
