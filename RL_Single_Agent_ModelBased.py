import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Seed for reproducibility
np.random.seed(42)

###############################################################################
# Helper functions for materials/dirts tracking via "task_status"
###############################################################################
def find_material_index(cell, all_materials):
    """Return the index of 'cell' in the materials list, or -1 if not found."""
    try:
        return all_materials.index(cell)
    except ValueError:
        return -1

def find_dirt_index(cell, all_dirts):
    """Return the index of 'cell' in the dirts list, or -1 if not found."""
    try:
        return all_dirts.index(cell)
    except ValueError:
        return -1

def is_material_done(task_status, material_index):
    """Check if the material_index-th bit is set (already picked up)."""
    return (task_status & (1 << material_index)) != 0

def mark_material_done(task_status, material_index):
    """Set the material_index-th bit to 1 in task_status (mark collected)."""
    return task_status | (1 << material_index)

def is_dirt_done(task_status, dirt_index, number_of_materials):
    """
    Check if the dirt_index-th bit is set, offset by number_of_materials
    (i.e., after the materials bits).
    """
    return (task_status & (1 << (number_of_materials + dirt_index))) != 0

def mark_dirt_done(task_status, dirt_index, number_of_materials):
    """
    Set the bit for dirt_index (offset by number_of_materials) in task_status.
    """
    return task_status | (1 << (number_of_materials + dirt_index))


###############################################################################
# Main Environment Class
###############################################################################
class GridEnvironment:
    def __init__(self, grid_size, objectives=None, obstacles=None, 
                 materials=None, dirts=None, env_type="general"):
        self.grid_size = grid_size
        self.objectives = objectives if objectives else []
        self.obstacles = obstacles if obstacles else []
        self.materials = materials if materials else []
        self.dirts = dirts if dirts else []
        self.env_type = env_type
        self.cleaned = []  # Track cleaned dirt cells

        # Count how many items (materials + dirts) for the expanded state representation
        self.number_of_materials = len(self.materials)
        self.number_of_dirts = len(self.dirts)
        self.total_task_items = self.number_of_materials + self.number_of_dirts

        # Our state is now (row, col, task_status)
        # but we will store a "current_state" for the agent and reset in reset()
        self.current_state = (0, 0, 0)  # (x, y, task_status=0)

        # Transition and reward dictionaries
        self.transition_probabilities = {}
        self.rewards = {}

        # Build the transition and reward models for all (x,y,task_status)
        self.set_rewards_and_transitions()

    def generate_states(self):
        """
        Generate all possible (row, col, task_status) states, 
        skipping positions that are obstacles.
        """
        states = []
        number_of_states_for_tasks = 1 << self.total_task_items  # 2^(materials+dirts)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if (x, y) in self.obstacles:
                    continue
                for task_status in range(number_of_states_for_tasks):
                    states.append((x, y, task_status))
        return states

    def set_rewards_and_transitions(self):
        """
        Populate self.transition_probabilities and self.rewards
        for all possible (state, action, next_state) tuples.
        """
        all_states = self.generate_states()

        # Initialize each triple with default probabilities=0 and reward=-0.5
        for state in all_states:
            for action in range(4):
                for new_state in all_states:
                    self.transition_probabilities[(state, action, new_state)] = 0.0
                    self.rewards[(state, action, new_state)] = -0.5

        # Now set the transitions and dynamic rewards
        for state in all_states:
            (row, col, task_status) = state
            for action in range(4):
                next_row, next_col = self.get_next_position(row, col, action)

                # If next position is an obstacle, we stay in place
                if (next_row, next_col) in self.obstacles:
                    next_row, next_col = row, col

                new_task_status = task_status
                move_penalty = -1  # default penalty for step

                # Hangar logic: if we step onto a new material
                if self.env_type == "hangar":
                    possible_material_index = find_material_index((next_row, next_col), self.materials)
                    if (possible_material_index != -1 and 
                        not is_material_done(task_status, possible_material_index)):
                        new_task_status = mark_material_done(task_status, possible_material_index)
                        move_penalty = 10

                # Warehouse logic: if we step onto a new dirt
                elif self.env_type == "warehouse":
                    possible_dirt_index = find_dirt_index((next_row, next_col), self.dirts)
                    if (possible_dirt_index != -1 and 
                        not is_dirt_done(task_status, possible_dirt_index, self.number_of_materials)):
                        new_task_status = mark_dirt_done(task_status, possible_dirt_index, self.number_of_materials)
                        move_penalty = 10

                # Now check if next cell is an objective
                if (next_row, next_col) in self.objectives:
                    if self.env_type == "hangar":
                        # Must have collected all materials
                        all_collected = True
                        for m_index in range(self.number_of_materials):
                            if not is_material_done(new_task_status, m_index):
                                all_collected = False
                                break
                        if all_collected:
                            move_penalty = 20
                        else:
                            move_penalty = -5

                    elif self.env_type == "warehouse":
                        # Must have cleaned all dirts
                        all_cleaned = True
                        for d_index in range(self.number_of_dirts):
                            if not is_dirt_done(new_task_status, d_index, self.number_of_materials):
                                all_cleaned = False
                                break
                        if all_cleaned:
                            move_penalty = 20
                        else:
                            move_penalty = -5

                    elif self.env_type == "garage":
                        move_penalty = 20

                # Fill in transitions (prob=1 for one next state)
                final_state = (next_row, next_col, new_task_status)
                self.transition_probabilities[(state, action, final_state)] = 1.0
                self.rewards[(state, action, final_state)] = move_penalty

    def get_next_position(self, row, col, action):
        """
        A helper for movement: returns the (row, col) after taking action 
        (0=up,1=down,2=left,3=right).
        """
        if action == 0:  # up
            return max(row - 1, 0), col
        elif action == 1:  # down
            return min(row + 1, self.grid_size[0] - 1), col
        elif action == 2:  # left
            return row, max(col - 1, 0)
        elif action == 3:  # right
            return row, min(col + 1, self.grid_size[1] - 1)
        return row, col

    def get_next_state(self, state, action):
        """
        For consistency with your existing code: looks up the next state in the transitions.
        'state' is now (row, col, task_status).
        """
        possible_next_states = []
        for s_next in self.generate_states():
            prob = self.transition_probabilities.get((state, action, s_next), 0.0)
            if prob > 0:
                possible_next_states.append(s_next)

        if len(possible_next_states) == 0:
            # No valid next state found, remain in place with a small penalty
            return state
        # In this setup, we typically have exactly one next_state with probability=1
        return possible_next_states[0]

    def reset(self, start_position=(0, 0)):
        """
        Reset environment. 
        For 'garage', we place the agent at row=2, col=0, task_status=0, just like in your code.
        """
        if self.env_type == "garage":
            self.current_state = (2, 0, 0)
        else:
            self.current_state = (start_position[0], start_position[1], 0)

        print(f"Environment reset: {self.env_type}. Starting state: {self.current_state}")
        return self.current_state

    def step(self, action):
        """
        Step from self.current_state using the transition and reward dictionaries.
        Updates `cleaned` for warehouse environments dynamically.
        Returns (next_state, reward, done).
        """
        state = self.current_state
        next_state = self.get_next_state(state, action)
        reward = self.rewards.get((state, action, next_state), -0.1)

        self.current_state = next_state
        (row, col, task_status) = next_state

        # If in a warehouse environment and on a dirt cell, mark it as cleaned
        if self.env_type == "warehouse" and (row, col) in self.dirts:
            dirt_index = find_dirt_index((row, col), self.dirts)
            if not is_dirt_done(task_status, dirt_index, self.number_of_materials):
                self.cleaned.append((row, col))  # Add to cleaned list

        # Check if done
        done = False
        if (row, col) in self.objectives:
            if self.env_type == "hangar":
                # Check if all materials are done
                all_collected = True
                for m_index in range(self.number_of_materials):
                    if not is_material_done(task_status, m_index):
                        all_collected = False
                        break
                done = all_collected

            elif self.env_type == "warehouse":
                all_cleaned = True
                for d_index in range(self.number_of_dirts):
                    if not is_dirt_done(task_status, d_index, self.number_of_materials):
                        all_cleaned = False
                        break
                done = all_cleaned

            elif self.env_type == "garage":
                done = True

        return next_state, reward, done


###############################################################################
# Value Iteration (same function name, but now it works in 3D)
###############################################################################
def value_iteration(env, max_iterations=1000, GAMMA=0.9):
    # Create a 3D array for the value function: shape = [rows, cols, 2^(materials+dirts)]
    rows, cols = env.grid_size
    number_of_states_for_tasks = 1 << env.total_task_items
    V = np.zeros((rows, cols, number_of_states_for_tasks))

    for _ in range(max_iterations):
        V_prev = V.copy()
        delta = 0

        all_states = env.generate_states()
        for (x, y, task_status) in all_states:
            best_value = float('-inf')

            # We skip obstacles if desired:
            if (x, y) in env.obstacles:
                V[x, y, task_status] = 0
                continue

            # Check each possible action
            for action in range(4):
                # Find the next state(s)
                next_states_list = []
                for possible_next in all_states:
                    prob = env.transition_probabilities.get(((x, y, task_status), action, possible_next), 0.0)
                    if prob > 0:
                        next_states_list.append(possible_next)

                if len(next_states_list) == 0:
                    continue

                # Typically exactly one next_state with prob=1
                next_s = next_states_list[0]
                reward_here = env.rewards[((x, y, task_status), action, next_s)]
                # Bellman backup
                candidate_value = reward_here + GAMMA * V_prev[next_s[0], next_s[1], next_s[2]]

                if candidate_value > best_value:
                    best_value = candidate_value

            if best_value == float('-inf'):
                best_value = 0

            # Update
            delta = max(delta, abs(best_value - V[x, y, task_status]))
            V[x, y, task_status] = best_value

        if delta < 1e-5:
            break

    return V

###############################################################################
# Derive Policy from Value Function
###############################################################################
def get_policy_from_value_function(env, value_function, GAMMA=0.9):
    policy = {}
    all_states = env.generate_states()

    for (x, y, task_status) in all_states:
        if (x, y) in env.obstacles:
            continue

        best_action = None
        best_value = float('-inf')

        # Check all actions
        for action in range(4):
            possible_next_states = []
            for candidate_next_state in all_states:
                prob = env.transition_probabilities.get(((x, y, task_status), action, candidate_next_state), 0.0)
                if prob > 0:
                    possible_next_states.append(candidate_next_state)

            if not possible_next_states:
                continue

            next_st = possible_next_states[0]
            r = env.rewards[((x, y, task_status), action, next_st)]
            new_value = r + GAMMA * value_function[next_st[0], next_st[1], next_st[2]]

            if new_value > best_value:
                best_value = new_value
                best_action = action

        if best_action is not None:
            policy[(x, y, task_status)] = best_action

    return policy


###############################################################################
# Simulate Policy (same function name)
###############################################################################
def simulate_policy(env, policy, max_steps=100):
    """
    Runs the agent in the environment for up to max_steps, returning the 
    visited states, chosen actions, and total reward.
    """
    current = env.reset()
    trajectory = [current]
    actions_done = []
    total_reward = 0

    for _ in range(max_steps):
        if current not in policy:
            print(f"No action defined for state {current}.")
            break

        chosen_action = policy[current]
        next_state, gained_reward, done = env.step(chosen_action)
        total_reward += gained_reward

        trajectory.append(next_state)
        actions_done.append(chosen_action)

        if done:
            print(f"Task completed with total reward: {total_reward}")
            break

        current = next_state

    return trajectory, actions_done

###############################################################################
# Plot Grid (unchanged)
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
                 label="Legend: 0=empty, 1=Obstacle, 2=Material, 3=Dirt, 4=Cleaned_spot, 5=Objective or recharge station")

    if trajectory and arrows:
        for idx, (pos, act) in enumerate(zip(trajectory, arrows)):
            x, y = pos[:2]
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


###############################################################################
# Create Grid (unchanged, but note we do not track cleaned cells in the same way)
###############################################################################
def create_grid(env):
    """
    Creates the grid and marks cleaned dirts as green (value=4).
    """
    grid = np.zeros(env.grid_size)
    for obj in env.objectives:
        grid[obj] = 5  # Objective
    for obs in env.obstacles:
        grid[obs] = 1  # Obstacle
    for mat in env.materials:
        grid[mat] = 2  # Material
    for dr in env.dirts:
        grid[dr] = 3  # Dirt
    for cl in env.cleaned:  # Mark cleaned dirt cells as green
        grid[cl] = 4
    return grid



###############################################################################
# Example usage with three environments
###############################################################################

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


# Solve each environment
print("Solving Hangar Environment...")
hangar_values = value_iteration(hangar)
print("\nOptimal Value Function for Hangar:")
print(hangar_values)

hangar_policy = get_policy_from_value_function(hangar, hangar_values)
print("\nSimulating Hangar Environment with Optimal Policy...")
trajectory_hangar, actions_hangar = simulate_policy(hangar, hangar_policy)
hangar_grid = create_grid(hangar)
plot_grid(hangar_grid, "Hangar Final State", trajectory_hangar, actions_hangar)

print("\nSolving Warehouse Environment...")
warehouse_values = value_iteration(warehouse)
print("\nOptimal Value Function for Warehouse:")
print(warehouse_values)

warehouse_policy = get_policy_from_value_function(warehouse, warehouse_values)
print("\nSimulating Warehouse Environment with Optimal Policy...")
trajectory_warehouse, actions_warehouse = simulate_policy(warehouse, warehouse_policy)
warehouse_grid = create_grid(warehouse)
plot_grid(warehouse_grid, "Warehouse Final State", trajectory_warehouse, actions_warehouse)

print("\nSolving Garage Environment...")
garage_values = value_iteration(garage)
print("\nOptimal Value Function for Garage :")
print(garage_values)

garage_policy = get_policy_from_value_function(garage, garage_values)
print("\nSimulating Garage Environment with Optimal Policy...")
trajectory_garage, actions_garage = simulate_policy(garage, garage_policy)

garage_grid = create_grid(garage)
plot_grid(garage_grid, "Garage Final State", trajectory_garage, actions_garage)

