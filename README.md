# Reinforcement Learning Project for Robot Cleaner Coordination

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Environment](#environment)
- [Implemented Approaches](#implemented-approaches)
  - [Single Agent Model-Based (Value Iteration)](#single-agent-model-based-value-iteration)
  - [Single Agent Model-Free](#single-agent-model-free)
    - [Q-Learning](#q-learning)
    - [Double Q-Learning](#double-q-learning)
    - [Monte Carlo](#monte-carlo)
  - [Multi-Agent Model-Based](#multi-agent-model-based)
- [Comparative Analysis](#comparative-analysis)
- [Challenges and Solutions](#challenges-and-solutions)
- [Possible Improvements](#possible-improvements)
- [Conclusion](#conclusion)
- [Annexes](#annexes)
- [Installation](#installation)
- [Usage](#usage)
- [Contributions](#contributions)
- [License](#license)

## Introduction
This project aims to implement and compare various Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL) approaches to solve a cleaning problem in environments hazardous to humans. The robots navigate three distinct environments: a hangar (H), a warehouse (E), and a garage (G). Every day, the robots must collect materials from the hangar, clean the warehouse, and recharge in the garage. The positions of the materials and charging stations change daily, requiring continuous adaptation from the agents.

## Objectives
- Implement and compare different RL and MARL approaches.
- Develop agents capable of navigating and coordinating actions in dynamic environments.
- Analyze and compare the performance of the different approaches.

## Environment
The simulated environment consists of three grid worlds of potentially different sizes:
- **Hangar (H)**: A simple environment where cells can be obstacles, empty, or contain materials. One cell allows exiting the environment.
- **Warehouse (E)**: A more complex environment where cells can be obstacles, empty, or contain dirt. One cell allows exiting the environment.
- **Garage (G)**: A simple environment where cells can be obstacles, empty, or contain charging stations.

## Implemented Approaches
### Single Agent Model-Based (Value Iteration)
- **Principle**: Solving Bellman's equation to find the optimal policy.
- **Results**: Agents achieve their goals optimally.

### Single Agent Model-Free
#### Q-Learning
- **Implementation**: Estimating action values with a Q-table.
- **Results**: Agents learn to reach their goals after a number of episodes.

#### Double Q-Learning
- **Motivation**: Reducing overestimation bias.
- **Comparison**: Faster and more stable convergence than Q-learning.

#### Monte Carlo
- **Implementation**: Updating Q-values using average returns from complete episodes.
- **Advantages and Limitations**: Simple to implement and fast convergence, but requires complete episodes.

### Multi-Agent Model-Based
- **Adaptation**: Adapting Value Iteration for two agents coordinating to maximize cumulative reward.
- **Performance Analysis**: Coordination improves overall performance.

## Comparative Analysis
- **Value Iteration**: Guaranteed optimality but requires complete knowledge of the environment.
- **Q-learning**: Flexibility but may need many episodes to converge.
- **Double Q-learning**: Reduces overestimation bias but increases complexity.
- **Monte Carlo**: Simplicity but requires complete episodes.
- **Multi-Agent Value Iteration**: Optimal coordination but high computational complexity.

## Challenges and Solutions
- **Computational Complexity**: Using parallelization and dimensionality reduction techniques.
- **Slow Convergence**: ε-greedy and Boltzmann exploration techniques.

## Possible Improvements
- **Neural Networks**: Implementing Deep Q-Networks (DQN).
- **More Realistic Environments**: Testing algorithms in dynamic environments.

## Conclusion
This project demonstrates that agents can effectively learn to complete tasks even in changing environments.

## Annexes
- **Commented Code**: Available in the repository with detailed comments.
- **Experimentation Details**: Parameters and results available in the repository.
- **Additional Graphs**: Convergence curves and agent performance available in the repository.

## Installation
To install the necessary dependencies:
```bash
pip install numpy matplotlib torch
```

## Usage
To execute the different approaches, follow the instructions in the corresponding code files. The code files are organized as follows:
- `single_agent_model_based.py`: Implementation of Value Iteration for a single agent.
- `single_agent_model_free.py`: Implementation of Q-learning, Double Q-learning, and Monte Carlo for a single agent.
- `multi_agent_model_based.py`: Implementation of Value Iteration for two agents.

## Contributions
Contributions are welcome! If you wish to contribute to this project, please open an issue or submit a pull request.

