import os
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

import gymnasium as gym
from stable_baselines3 import SAC

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

def record_video(env_id, model, video_length=500, prefix="", video_folder="videos/"):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make("Walker2d-v5", render_mode="rgb_array")])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()


# Create the environment for rendering
env = gym.make('Walker2d-v5', render_mode="human")

# Load the trained SAC model
model = SAC.load("Models/best_model_5360ep")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


record_video("Walker2d-v5", model, video_length=5000, prefix="SAC-hyperparameter-tuning")

# Reset the environment and extract the observation
obs, _ = env.reset()

# Run inference
for i in range(1000):  
    action, _states = model.predict(obs, deterministic=True) 
    obs, reward, done, truncated, info = env.step(action)     
    env.render()                                              
    if done or truncated:                                     
        obs, _ = env.reset()                                  

env.close()
