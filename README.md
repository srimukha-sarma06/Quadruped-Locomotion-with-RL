# Quadruped Locomotion with Reinforcement Learning

A reinforcement learning project focused on training a quadruped robot to achieve stable and efficient locomotion in simulation using PPO and MuJoCo.

The environment is custom-built using Gymnasium and trained using Stable-Baselines3.

---

## Features

- Custom MuJoCo quadruped environment
- PPO-based locomotion training
- PD torque controller
- Curriculum learning
- Domain randomization
- Contact-aware reward shaping
- Multi-environment parallel training
- Observation and reward normalization
- TensorBoard logging support

---

## Reward Design

The reward function combines multiple locomotion objectives:

- Forward velocity tracking
- Angular velocity tracking
- Orientation stabilization
- Height stabilization
- Foot clearance rewards
- Foot slip penalties
- Energy efficiency penalties
- Action smoothness penalties
- Pose regularization

---

## Domain Randomization

To improve robustness and sim-to-real transfer:

- Friction randomization
- Body mass randomization
- PD gain randomization

---

## Observation Space

The observation vector contains:

- Joint positions
- Joint velocities
- Base angular velocity
- Base linear velocity
- Gravity projection
- Previous action history
- Foot contact states
- Velocity commands

---

## Action Space

Continuous 12-dimensional action space controlling target joint positions through a PD controller.

---

## Tech Stack

- Python
- Gymnasium
- MuJoCo
- Stable-Baselines3
- NumPy
- PyTorch

---

## Project Structure

```bash
.
├── custom_env_new.py
├── custom_env_final.py
├── train_pd_ppo.py
├── checkpoints/
├── logs/
├── best_model/
└── README.md
```

---

## Training Setup

Training uses PPO from Stable-Baselines3 with:

- 8 parallel environments
- VecNormalize for observation and reward normalization
- Curriculum learning
- Checkpoint callbacks
- Evaluation callbacks
- TensorBoard logging

---

## Future Improvements

- Terrain adaptation
- Sim-to-real transfer
- Vision-based navigation
- Real robot deployment

---

## Acknowledgements

Built using:

- Gymnasium
- MuJoCo
- Stable-Baselines3
- PyTorch
- Numpy
---

## Author

**Srimukha Sarma**
