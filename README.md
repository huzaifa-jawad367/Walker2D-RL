# Walker2D-RL

A reinforcement learning project that solves the **Walker2D** control problem using the **Soft Actor Critic (SAC)** algorithm in the **MuJoCo** simulation environment.

## 🧠 Project Overview

This project implements an off-policy, model-free reinforcement learning algorithm to train a bipedal robot (Walker2D) to walk effectively using torque control on its joints. The Walker2D environment is an extension of the Hopper environment, adding another leg to enable more natural bipedal movement.

## 🔬 Environment: Walker2D (MuJoCo)

- **Action Space**: 6 (torque values for 6 joints)
- **Observation Space**: 17
  - 8 position values of the robot’s body parts
  - 9 velocity values of body parts
- **Goal**: Walk forward by applying torque to 6 hinges connected to 7 body parts

## 🧮 Algorithm: Soft Actor Critic (SAC)

- **Key Features**:
  - Off-policy and model-free
  - Uses replay buffer for sampling states and actions
  - Employs the maximum entropy framework to balance exploration and exploitation
  - Computes soft Q-values using Bellman backups
  - Stable performance across different random seeds

### 🔍 Why SAC?
- Efficient with sample usage due to off-policy learning
- Handles complex dynamics (like balancing and falling)
- Resilient to hyperparameter variations
- Widely cited and benchmarked in literature

## ⚙️ Training Details

- Framework: [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3)
- Environment: `Walker2d-v4` (from OpenAI Gym with MuJoCo backend)
- Training Duration: Up to 1,000,000 timesteps

### 🧪 Hyperparameter Tuning

- Changed `tau` from 0.005 → 0.01
- Changed `ent-coefficient` to 0.1
- Conditional tuning due to limited compute resources (grid search not applied)
- Tuned parameters: Reward scale, target smoothing coefficient, learning rate

## 📊 Results

| Metric                     | Default SAC       | After Tuning     |
|---------------------------|-------------------|------------------|
| Episodes Trained          | 1254              | 1260             |
| Mean Episode Length       | 976               | 972              |
| Mean Reward per Episode   | 4140.84           | 4948.95          |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `mujoco` and `mujoco-py`
- `stable-baselines3`
- `gym`

```bash
pip install -r requirements.txt
