import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
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

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model at {} timesteps".format(x[-1]))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)

        return True

# Hyperparameters to tune
tau_values = [0.1, 0.01, 0.001, 0.0001]
ent_coef_values = [1.0, 0.3333, 0.1, 0.03333, 0.01]
learning_rate_values = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]

env_id = 'Walker2d-v5'
total_timesteps = 300_000

# Iterate over tau values
for tau in tau_values:
    log_dir = f"./logs/SAC_tau_{tau}/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = gym.make(env_id)
    env = Monitor(env, log_dir)

    print(f"Starting training process with tau={tau}")

    # Create callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # Train model
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                tau=tau)
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"SAC_tau_{tau}", callback=callback)

    print(f"Training complete for tau={tau}\n")

    env.close()

for ent_coef in ent_coef_values:
    log_dir = f"./logs/SAC_entcoef_{ent_coef}/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = gym.make(env_id)
    env = Monitor(env, log_dir)

    print(f"Starting training process with ent_coef={ent_coef}")

    # Create callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # Train model
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                ent_coef=ent_coef)
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"SAC_entcoef_{ent_coef}", callback=callback)

    print(f"Training complete for ent_coef={ent_coef}\n")

    env.close()

for learning_rate in learning_rate_values:
    log_dir = f"./logs/SAC_lr_{learning_rate}/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = gym.make(env_id)
    env = Monitor(env, log_dir)

    print(f"Starting training process with learning_rate={learning_rate}")

    # Create callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # Train model
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                learning_rate=learning_rate)
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"SAC_lr_{learning_rate}", callback=callback)

    print(f"Training complete for learning_rate={learning_rate}\n")

    env.close()