from stable_baselines3 import PPO
from env.RLautoscalingProto import RLautoscalingProto
from agents.RuleBaseAutoScaling import RuleBaseAutoScaling

def evaluationAgents(steps=200):
    env = RLautoscalingProto()
    agent = RuleBaseAutoScaling()
