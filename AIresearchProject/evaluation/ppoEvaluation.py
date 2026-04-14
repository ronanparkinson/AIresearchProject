from stable_baselines3 import PPO
from env.RLautoscalingProto import RLautoscalingProto

class ppoEvaluation:
    def __init__(self, modelPath="../training/PPOscaler", rewardVersion="v1"):
        self.model = PPO.load(modelPath)
        self.env = RLautoscalingProto(rewardVersion=rewardVersion)

    def run(self, steps=200, giveHistory=False):
        state, _ = self.env.reset()

        totalReward = 0
        totalQueue = 0
        totalCpu = 0
        totalInstances = 0
        scalingActions = 0

        past = {
            "rewardPast": [],
            "QueuePast": [],
            "CpuPast": [],
            "instancePast": []
        }

        for _ in range(steps):
            action, _ = self.model.predict(state, deterministic=True)

            if action != 1:
                scalingActions += 1

            state, reward, done, truncated, info = self.env.step(int(action))
            cpu, workload, queue, instances = state

            totalReward += reward
            totalQueue += queue
            totalCpu += cpu
            totalInstances += instances

            past["rewardPast"].append(float(reward))
            past["QueuePast"].append(float(queue))
            past["CpuPast"].append(float(cpu))
            past["instancePast"].append(float(instances))

            if done or truncated:
                break

        stats = {
            "reward": float(totalReward),
            "average queue": float(totalQueue / steps),
            "average cpu": float(totalCpu / steps),
            "average instances": float(totalInstances / steps),
            "scaling actions": int(scalingActions)
        }
        return stats, past