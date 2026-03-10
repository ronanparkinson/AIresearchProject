from stable_baselines3 import PPO
from env.RLautoscalingProto import RLautoscalingProto

env = RLautoscalingProto()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("PPOscaler")