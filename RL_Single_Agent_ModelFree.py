import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Seed for reproducibility
np.random.seed(42)

###############################################################################
# Environment: Single-agent, Model-Free Setup
###############################################################################
class GridEnvironment:
    """
    A simple grid environment for either hangar, warehouse, or garage tasks.

    Attributes:
    -----------
    grid_size : tuple
        Dimensions of the grid (rows, columns).

    objectives : list of tuples
        Cells that represent the final objective(s).

    obstacles : list of tuples
        Cells that are blocked (agent will receive a penalty if it moves here).

    materials : list of tuples
        Cells containing materials (hangar-specific) that agent can pick up for a reward.

    dirts : list of tuples
        Cells containing dirt (warehouse-specific) that agent can clean for a reward.

    env_type : str
        One of {"hangar", "warehouse", "garage"}. Determines task logic.

    visited_materials : set
        Keeps track of which materials are collected (for hangar only).

    cleaned_cells : set
        Keeps track of which dirts are cleaned (for warehouse only).

    loop_memory : set
        Used to detect loops and apply additional penalty if the agent cycles.

    Methods:
    --------
    reset(start_position=(0, 0)):
        Resets the environment to a start position (or a custom one).

    step(action):
        Executes the action in the environment, returns (next_state, reward, done).

    """
    def __init__(self, grid_size, objectives=None, obstacles=None, 
                 materials=None, dirts=None, env_type="general"):
        self.grid_size = grid_size
        self.state = (0, 0)
        self.objectives = objectives if objectives else []
        self.obstacles = obstacles if obstacles else []
        self.materials = materials if materials else []
        self.dirts = dirts if dirts else []
        self.visited_materials = set()
        self.cleaned_cells = set()
        self.loop_memory = set()
        self.env_type = env_type

    def reset(self, start_position=(0, 0)):
        """
        Reset the environment to the start position. 
        Also clears visited/cleaned sets and loop_memory.

        For 'garage', the default start is (2,0).
        """
        if self.env_type == "garage":
            self.state = (2, 0)
        else:
            self.state = start_position
            self.visited_materials = set()
            self.cleaned_cells = set()
            self.loop_memory = set()

        print(f"Environment reset: {self.env_type}. Starting position: {self.state}")
        return self.state

    def step(self, action):
        """
        Takes an action in {0=Up,1=Down,2=Left,3=Right}, updates the agent's state,
        and returns (next_state, reward, done).
        """
        x, y = self.state
        rows, cols = self.grid_size

        # Compute candidate next state
        if action == 0:  # Up
            next_state = (max(x - 1, 0), y)
        elif action == 1:  # Down
            next_state = (min(x + 1, rows - 1), y)
        elif action == 2:  # Left
            next_state = (x, max(y - 1, 0))
        elif action == 3:  # Right
            next_state = (x, min(y + 1, cols - 1))
        else:
            next_state = self.state

        reward = -0.5  # Default penalty
        done = False

        # Check if next state is an obstacle
        if next_state in self.obstacles:
            reward = -25
            print(f"[{self.env_type}] Hit obstacle at {next_state}. Penalty applied.")

        # Hangar logic (collecting materials)
        elif next_state in self.materials and self.env_type == "hangar":
            if next_state not in self.visited_materials:
                reward = 20
                self.visited_materials.add(next_state)
                print(f"[{self.env_type}] Collected material at {next_state}. Reward gained.")
            else:
                # Already collected that material
                reward = -5

        # Warehouse logic (cleaning dirt spots)
        elif next_state in self.dirts and self.env_type == "warehouse":
            if next_state not in self.cleaned_cells:
                reward = 20
                self.cleaned_cells.add(next_state)
                print(f"[{self.env_type}] Cleaned dirty spot at {next_state}. Reward gained.")
            else:
                # Already cleaned that spot
                reward = -5

        # Check if next_state is an objective
        elif next_state in self.objectives:
            # Hangar objective (all materials must be collected)
            if self.env_type == "hangar" and len(self.visited_materials) == len(self.materials):
                reward = 100
                done = True
            # Warehouse objective (all dirt must be cleaned)
            elif self.env_type == "warehouse" and len(self.cleaned_cells) == len(self.dirts):
                reward = 100
                done = True
            # Garage objective (simply arrive at the objective)
            elif self.env_type == "garage":
                reward = 100
                done = True
            else:
                # Reached the objective prematurely
                reward = -20

        else:
            # Optional loop detection penalty
            if (next_state, action) in self.loop_memory:
                reward -= 10
                # Attempt to break loops by adding random action
                self.loop_memory.add((next_state, np.random.randint(4)))
            else:
                self.loop_memory.add((next_state, action))

        # Update state
        self.state = next_state
        return next_state, reward, done


###############################################################################
# Simple Q-Table Implementation
###############################################################################
def q_learning(env, episodes=1000, max_steps=200, alpha=0.1, gamma=0.9, epsilon=0.2):
    """
    Basic Q-learning with a single Q-table for each environment.

    Parameters:
    -----------
    env : GridEnvironment
        The environment instance.
    episodes : int
        Number of training episodes.
    max_steps : int
        Maximum steps allowed per episode.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration rate.

    Returns:
    --------
    q_table : np.ndarray
        A Q-table of shape (rows, cols, 4).
    """
    q_table = np.zeros((*env.grid_size, 4))

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            x, y = state

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(4)  # Explore
            else:
                action = np.argmax(q_table[x, y])  # Exploit

            next_state, reward, done = env.step(action)
            total_reward += reward

            nx, ny = next_state
            # Q-learning update
            q_table[x, y, action] += alpha * (
                reward + gamma * np.max(q_table[nx, ny]) - q_table[x, y, action]
            )

            state = next_state
            if done:
                break

        # Decay epsilon after each episode
        epsilon = max(0.01, epsilon * 0.995)
        print(f"[{env.env_type}] Episode {episode + 1}: Total Reward = {total_reward}")

    return q_table


def double_q_learning(env, episodes=1000, max_steps=200, alpha=0.1, gamma=0.9, epsilon=0.2):
    """
    Double Q-learning to reduce overestimation bias.

    Returns the sum of two Q-tables as the final Q-table.
    """
    q_table_1 = np.zeros((*env.grid_size, 4))
    q_table_2 = np.zeros((*env.grid_size, 4))

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            x, y = state

            # Epsilon-greedy action selection based on sum of Q-values
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                combined_q = q_table_1[x, y] + q_table_2[x, y]
                action = np.argmax(combined_q)

            next_state, reward, done = env.step(action)
            total_reward += reward

            nx, ny = next_state

            # Randomly choose one Q-table to update
            if np.random.rand() < 0.5:
                best_action = np.argmax(q_table_1[nx, ny])
                q_table_1[x, y, action] += alpha * (
                    reward + gamma * q_table_2[nx, ny, best_action] - q_table_1[x, y, action]
                )
            else:
                best_action = np.argmax(q_table_2[nx, ny])
                q_table_2[x, y, action] += alpha * (
                    reward + gamma * q_table_1[nx, ny, best_action] - q_table_2[x, y, action]
                )

            state = next_state
            if done:
                break

        epsilon = max(0.01, epsilon * 0.995)
        print(f"[{env.env_type}] Episode {episode + 1}: Total Reward = {total_reward}")

    return q_table_1 + q_table_2


def monte_carlo(env, episodes=1000, gamma=0.9, epsilon=0.2):
    """
    Monte Carlo control algorithm (exploring starts) for policy improvement.
    """
    q_table = np.zeros((*env.grid_size, 4))
    returns_dict = {}

    # Initialize returns dictionary
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            for act in range(4):
                returns_dict[(i, j, act)] = []

    for episode in range(episodes):
        state = env.reset()
        episode_data = []
        total_reward = 0

        # Generate an episode
        for _ in range(100):  # Step limit
            x, y = state
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(q_table[x, y])

            next_state, reward, done = env.step(action)
            episode_data.append((state, action, reward))
            total_reward += reward

            state = next_state
            if done:
                break

        # Monte Carlo returns update (backwards)
        G = 0
        for t in reversed(range(len(episode_data))):
            s, a, r = episode_data[t]
            x, y = s
            G = r + gamma * G

            # Check if this state-action didn't appear before in the episode
            if (s, a) not in [(ep[0], ep[1]) for ep in episode_data[:t]]:
                returns_dict[(x, y, a)].append(G)
                q_table[x, y, a] = np.mean(returns_dict[(x, y, a)])

        epsilon = max(0.01, epsilon * 0.995)
        print(f"[{env.env_type}] Episode {episode + 1}: Total Reward = {total_reward}")

    return q_table

###############################################################################
# DQN Implementation
###############################################################################
class DQN(nn.Module):
    """
    A simple feedforward network with two hidden layers,
    used for Deep Q-Learning.
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


def train_dqn(env, episodes=1000, max_steps=200, gamma=0.99, epsilon=1.0,
              epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001,
              batch_size=64):
    """
    Deep Q-Network training loop with replay buffer and target network.
    """
    input_dim = np.prod(env.grid_size)  # Flatten the grid into a single dimension
    output_dim = 4  # 4 possible actions

    q_network = DQN(input_dim, output_dim)
    target_network = DQN(input_dim, output_dim)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    replay_buffer = deque(maxlen=2000)

    def state_to_input(state):
        """One-hot encode the (x, y) position into a flat array."""
        x, y = state
        input_tensor = np.zeros(env.grid_size)
        input_tensor[x, y] = 1
        return input_tensor.flatten()

    def select_action(state, eps):
        """
        Epsilon-greedy action selection using the current Q-network.
        """
        if np.random.rand() < eps:
            return np.random.randint(4)
        else:
            state_tensor = torch.FloatTensor(state_to_input(state)).unsqueeze(0)
            q_values = q_network(state_tensor)
            return torch.argmax(q_values).item()

    def update_network():
        """
        Sample a batch from the replay buffer and perform one gradient update.
        """
        if len(replay_buffer) < batch_size:
            return

        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert batch data to tensors
        states_t = torch.FloatTensor([state_to_input(s) for s in states])
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor([state_to_input(s) for s in next_states])
        dones_t = torch.FloatTensor(dones)

        # Calculate targets
        with torch.no_grad():
            max_next_q = target_network(next_states_t).max(dim=1)[0]
            targets = rewards_t + (1 - dones_t) * gamma * max_next_q

        # Current Q-values for the taken actions
        current_q_vals = q_network(states_t).gather(1, actions_t).squeeze()

        # Update
        loss = loss_fn(current_q_vals, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Main training loop
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Store transition
            replay_buffer.append((state, action, reward, next_state, float(done)))
            state = next_state

            # Learn
            update_network()

            if done:
                break

        # Decay epsilon and update the target network occasionally
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    return q_network

###############################################################################
# Simulation Function
###############################################################################
def simulate(env, q_source, max_steps=100, use_q_table=True):
    """
    Run the environment with the given Q-source (Q-table or DQN model).
    """
    def state_to_input(state):
        x, y = state
        input_tensor = np.zeros(env.grid_size)
        input_tensor[x, y] = 1
        return input_tensor.flatten()

    state = env.reset()
    trajectory = [state]
    actions = []
    total_reward = 0

    for step in range(max_steps):
        x, y = state
        if use_q_table:
            # Use Q-table
            action = np.argmax(q_source[x, y])
        else:
            # Use DQN
            state_tensor = torch.FloatTensor(state_to_input(state)).unsqueeze(0)
            with torch.no_grad():
                q_values = q_source(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)
        total_reward += reward
        trajectory.append(next_state)
        actions.append(action)

        if done:
            print(f"[{env.env_type}] Task complete at step {step + 1} with total reward {total_reward}.")
            break

        state = next_state

    print(f"[{env.env_type}] Final Total Reward: {total_reward}")
    return trajectory, actions

###############################################################################
# Plotting Function
###############################################################################
def plot_grid(grid, title, trajectory=None, arrows=None):
    """
    Visualize the grid with special color-coding and optional agent trajectory.
    """
    custom_colors = ["grey", "black", "blue", "red", "green", "purple"]
    cmap = ListedColormap(custom_colors)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, origin="upper")
    plt.colorbar(ticks=[0, 1, 2, 3, 4, 5],
                 label="Legend: 0=Grey, 1=Black, 2=Blue, 3=Red, 4=Green, 5=Purple")

    if trajectory and arrows:
        for idx, (pos, act) in enumerate(zip(trajectory, arrows)):
            x, y = pos
            dx, dy = 0, 0
            if act == 0: 
                dx = -0.3
            elif act == 1: 
                dx = 0.3
            elif act == 2: 
                dy = -0.3
            elif act == 3: 
                dy = 0.3  

            # Special arrow for the starting point
            if idx == 0:
                plt.arrow(y, x, dy, dx, head_width=0.3, head_length=0.3, fc="red", ec="black")
            else:
                plt.arrow(y, x, dy, dx, head_width=0.2, head_length=0.2, fc="yellow", ec="black")

    plt.title(title)
    plt.show()

"""    Training logic in functions :    """

# Training Q-learning with different episodes
def Train_with_q_learning():
    print("Training Hangar Environment...")
    q_hangar = q_learning(hangar, episodes=1000)

    print("Training Warehouse Environment...")
    q_warehouse = q_learning(warehouse, episodes=1000)

    print("Training Garage Environment...")
    q_garage = q_learning(garage, episodes=1000)

    return q_hangar,q_warehouse,q_garage



#Training Double Q-learning with different episodes
def Train_with_double_q_learning():
    print("Training Hangar Environment...")
    q_hangar = double_q_learning(hangar, episodes=1000)

    print("Training Warehouse Environment...")
    q_warehouse = double_q_learning(warehouse, episodes=1000)

    print("Training Garage Environment...")
    q_garage = double_q_learning(garage, episodes=1000)
    
    return q_hangar,q_warehouse,q_garage


#Training Monte carlo with different episodes
def Train_with_MonteCarlo():
    print("Training Hangar Environment...")
    q_hangar = monte_carlo(hangar, episodes=1000)

    print("Training Warehouse Environment...")
    q_warehouse = monte_carlo(warehouse, episodes=2000)

    print("Training Garage Environment...")
    q_garage = monte_carlo(garage, episodes=2000)

    return q_hangar,q_warehouse,q_garage


""" Here i am defining the environments, you can make the env random or just simply change cell's positions to change the env structure, 
    in condition that there is a path from the start to the exit"""


#Hangar environement
hangar = GridEnvironment(grid_size=(4, 4),
                         objectives=[(3, 3)],
                         obstacles=[(0, 1), (2, 1)],
                         materials=[(1, 1), (3, 2)],
                         env_type="hangar")
                         
#Warehouse environement
warehouse = GridEnvironment(grid_size=(4, 5),
                            objectives=[(0, 4)],
                            obstacles=[(0, 3), (0, 2) , (1, 1) ],
                            dirts=[  (1, 0), (1, 4),
                                    (2, 3), (2, 1),
                                    (3, 2), (3, 4) ],
                            env_type="warehouse")

#garage environement
garage = GridEnvironment(grid_size=(3, 5),
                         objectives=[(2, 4), (0, 3)],
                         obstacles=[(1, 1), (2, 3)],
                         env_type="garage")




""" Uncomment to use different approaches for training.

         Training start for the three algos
                                                """


"""Storing the q_values for each environement after training the agent:  """

#Uncomment the algorithm you want and comment the others, check always if the DQN training is also commented  :

q_hangar,q_warehouse,q_garage = Train_with_q_learning()
#q_hangar,q_warehouse,q_garage = Train_with_double_q_learning()
#q_hangar,q_warehouse,q_garage = Train_with_MonteCarlo()



"""Saving Q-tables for consistent evaluation """
# ------------Uncomment all below to use , comment all the lines to use DQN -----------------

np.save('q_hangar.npy', q_hangar)
np.save('q_warehouse.npy', q_warehouse)
np.save('q_garage.npy', q_garage)


""" Loading Q-tables for consistent evaluation"""
# ------------Uncomment all below to use , comment all the lines to use DQN -----------------

q_hangar = np.load('q_hangar.npy')
q_warehouse = np.load('q_warehouse.npy')
q_garage = np.load('q_garage.npy')




"""             Training the DQN network: 

    Uncomment all bellow when you want to use it and check if you commented before everything : """

#print("Training Hangar Environment with DQN...")
#q_hangar = train_dqn(hangar, episodes=100)

#print("Training Warehouse Environment with DQN...")
#q_warehouse = train_dqn(warehouse, episodes=500)

#print("Training Garage Environment with DQN...")
#q_garage = train_dqn(garage, episodes=100)

#torch.save(q_hangar.state_dict(), 'q_hangar_dqn.pth')
#torch.save(q_warehouse.state_dict(), 'q_warehouse_dqn.pth')
#torch.save(q_garage.state_dict(), 'q_garage_dqn.pth')



"""                               Simulation logic 

        Uncomment to use the following algos, skip if you want to use DQN (bellow) :    """

"""   If you're using Q-learning, Monte_Carlo or Double_Q-learnig , Keep the use_q_table=True else change to False to use simulate DQN """

print("Simulating Hangar Environment...")
trajectory_hangar, actions_hangar = simulate(hangar, q_hangar, use_q_table=True) #False for DQN only

print("Simulating Warehouse Environment...")
cleaning_trajectory, cleaning_actions = simulate(warehouse, q_warehouse, use_q_table=True) #False for DQN only

print("Simulating Garage Environment...")
trajectory_garage, actions_garage  = simulate(garage, q_garage, use_q_table=True) #False for DQN only


"""     Visualisation logic and functions 
                                             """
# Create grid displays
def create_grid(env):
    grid = np.zeros(env.grid_size)
    for obj in env.objectives:
        grid[obj] = 5

    for obs in env.obstacles:
        grid[obs] = 1

    for mat in env.materials:
        grid[mat] = 2

    for drt in env.dirts:
        grid[drt] = 3

    for cell in env.cleaned_cells:
        grid[cell] = 4
    return grid

hangar_grid = create_grid(hangar)
warehouse_grid = create_grid(warehouse)
garage_grid = create_grid(garage)


# Plot final states for : 

plot_grid(hangar_grid, "Hangar Final State", trajectory_hangar, actions_hangar)
plot_grid(warehouse_grid, "Warehouse Final State",cleaning_trajectory, cleaning_actions)
plot_grid(garage_grid, "Garage Final State", trajectory_garage, actions_garage)




