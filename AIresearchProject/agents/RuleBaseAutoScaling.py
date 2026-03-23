from env.RLautoscalingProto import RLautoscalingProto

class RuleBaseAutoScaling:
  def __init__(self, scaleUpThreshold=0.7, scaleDownThreshold=0.3, queueTh=0.6):
    self.scaleUpThreshold = scaleUpThreshold
    self.scaleDownThreshold = scaleDownThreshold
    self.queueTh = queueTh

  def requiredAction(self, state):
    ingestedWorkLoad, cpu, queue, instances = state

    if cpu > self.scaleUpThreshold:
      return 2
    elif queue > self.queueTh:
        return 2
    elif cpu < self.scaleDownThreshold and queue < 0.2:
      return 0
    else:
      return 1

env = RLautoscalingProto()
agent = RuleBaseAutoScaling()

state, info = env.reset()

for step in range(20):

    action = agent.requiredAction(state)
    state, reward, done, truncated, info = env.step(action)

    print("Step:", step, "Action:", action, "State:", state, "Reward:", reward)