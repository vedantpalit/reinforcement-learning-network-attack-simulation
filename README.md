# reinforcement-learning-network-attack-simulation
# Reinforcement Learning Network Attack Simulation

## Overview

This repository contains the implementation of my Bachelor Thesis project focused on benchmarking multiple reinforcement learning agents and assessing their performance on different computer network attack scenarios. The project utilizes the NASim (Network Attack Simulator) environment to evaluate various RL algorithms in cybersecurity contexts.

## Project Structure

```
reinforcement-learning-network-attack-simulation/
├── agents/                     # RL agent implementations
│   ├── rl_agent.py            # Traditional RL agents (Q-Learning, SARSA, etc.)
│   └── dqn_agent.py           # Deep Q-Network implementation
├── benchmarked_scenarios/      # Network scenario configurations
├── main.py                    # Main execution script
└── README.md                  # This file
```

## Supported Agents

The benchmark includes the following reinforcement learning agents:

- **Random Agent**: Baseline agent that selects actions randomly
- **Brute Force Agent**: Systematic exploration agent
- **Q-Learning Agent**: Model-free temporal difference learning
- **SARSA(λ) Agent**: On-policy TD learning with eligibility traces
- **DQN Agent**: Deep Q-Network with experience replay

## Prerequisites

### Required Dependencies

```bash
pip install nasim
pip install matplotlib
pip install numpy
pip install torch  # For DQN agent
pip install gym
```

### System Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, for faster DQN training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vedantpalit/reinforcement-learning-network-attack-simulation.git
cd reinforcement-learning-network-attack-simulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Command Structure

```bash
python main.py --scenario_path <path_to_scenario> --num_episodes <episodes> --seed_value <seed> --agent <agent_type>
```

### Command Line Arguments

- `--scenario_path`: Path to the YAML scenario file (required)
- `--num_episodes`: Number of episodes to run (required)
- `--seed_value`: Random seed for reproducibility (required)
- `--agent`: Agent type to run (required)

### Available Agent Types

- `random` - Random action selection
- `bruteforce` - Systematic exploration
- `qlearning` - Q-Learning algorithm
- `sarsa` - SARSA(λ) algorithm
- `dqn` - Deep Q-Network

### Example Commands

#### Running Q-Learning Agent
```bash
python main.py --scenario_path benchmarked_scenarios/scenario_tiny.yaml --num_episodes 100 --seed_value 42 --agent qlearning
```

#### Running DQN Agent
```bash
python main.py --scenario_path benchmarked_scenarios/scenario_tiny.yaml --num_episodes 50 --seed_value 123 --agent dqn
```

#### Running Random Baseline
```bash
python main.py --scenario_path benchmarked_scenarios/scenario_tiny.yaml --num_episodes 100 --seed_value 42 --agent random
```

#### Running SARSA Agent
```bash
python main.py --scenario_path benchmarked_scenarios/scenario_tiny.yaml --num_episodes 100 --seed_value 42 --agent sarsa
```

#### Running Brute Force Agent
```bash
python main.py --scenario_path benchmarked_scenarios/scenario_tiny.yaml --num_episodes 100 --seed_value 42 --agent bruteforce
```

## Hyperparameters

The following hyperparameters are configured in the main script:

- **Learning Rate (α)**: 0.5
- **Discount Factor (γ)**: 0.99
- **Initial Exploration (ε)**: 1.0
- **Epsilon Decay**: 0.995
- **Minimum Epsilon**: 0.1
- **Step Limit**: 200 steps per episode
- **SARSA λ**: 0.99 (eligibility trace decay)
- **Replay Buffer Size**: 10,000 (for SARSA)
- **Batch Size**: 64 (for experience replay)

## Output Metrics

The benchmark reports the following performance metrics for each agent:

- **Average Reward**: Mean cumulative reward across episodes
- **Average Steps**: Mean number of steps taken per episode
- **Success Rate**: Percentage of episodes where the goal was reached
- **Network Visualization**: Graphical representation of the network topology

## Scenario Files

Network scenarios are defined in YAML format and stored in the `benchmarked_scenarios/` directory. Each scenario specifies:

- Network topology
- Host configurations
- Vulnerability distributions
- Attack objectives

### Example Scenario Structure
```yaml
# scenario_tiny.yaml
hosts: 3
services: 2
topology: [2, 1]
step_limit: 200
# ... additional configuration
```

## Benchmarking Workflow

1. **Environment Setup**: Load scenario and initialize environment
2. **Agent Training**: Train the selected agent (if applicable)
3. **Evaluation**: Run multiple episodes with the trained/configured agent
4. **Metrics Collection**: Record performance statistics
5. **Visualization**: Generate network graphs and performance summaries


## Contact

For questions regarding this Bachelor Thesis project, please contact:
- **Author**: Vedant Palit
- **Email**: vedantpalit@kgpian.iitkgp.ac.in
- **Institution**: IIT Kharagpur

---
