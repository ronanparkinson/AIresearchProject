from env.RLautoscalingProto import RLautoscalingProto
from agents.RuleBaseAutoScaling import RuleBaseAutoScaling

def evaluationAgents(steps=200, rewardVersion="v1"):
    env = RLautoscalingProto(rewardVersion=rewardVersion)
    agent = RuleBaseAutoScaling()

    state, _ = env.reset()
    totalReward = 0
    totalQueue = 0
    totalCpu = 0
    totalInstances = 0
    scalingActions = 0

    for _ in range(steps):
        action = agent.requiredAction(state)

        if action != 1:
            scalingActions += 1

        state, reward, done, truncated, info = env.step(action)

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

