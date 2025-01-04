import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

###############################################################################
# Utility Functions for Materials/Dirts :CHECKED
###############################################################################
def material_index(cell_position, list_of_material_positions):
    """Return the 0-based index of 'cell_position' in list_of_material_positions, or -1 if not present."""
    try:
        return list_of_material_positions.index(cell_position)
    except ValueError:
        return -1

def dirt_index(cell_position, list_of_dirt_positions):
    """Return the 0-based index of 'cell_position' in list_of_dirt_positions, or -1 if not present."""
    try:
        return list_of_dirt_positions.index(cell_position)
    except ValueError:
        return -1

def check_collected(material_tracker, material_index_position):
    """Check if the bit 'material_index_position' is set in 'material_tracker'."""
    return (material_tracker & (1 << material_index_position)) != 0

def set_collected(material_tracker, material_index_position):
    """Set the bit 'material_index_position' in 'material_tracker'."""
    return material_tracker | (1 << material_index_position)

def check_cleaned(global_tracker, dirt_index_position, total_material_count):
    """For a warehouse scenario, check if the bit for 'dirt_index_position' is set (offset by total_material_count)."""
    return (global_tracker & (1 << (total_material_count + dirt_index_position))) != 0

def set_cleaned(global_tracker, dirt_index_position, total_material_count):
    """Set the bit for 'dirt_index_position' (offset by total_material_count) in 'global_tracker'."""
    return global_tracker | (1 << (total_material_count + dirt_index_position))


###############################################################################
# Environment Definition (Grid + Agents + MDP Setup)
###############################################################################
class TwoAgentGridMDP:
    """
    A two-agent grid environment (hangar/warehouse/garage) with a bitmask 
    for materials/dirts. States: ((rowAgent1,colAgent1),(rowAgent2,colAgent2), progressMask).
    Joint actions: ((actionAgent1),(actionAgent2)).
    """
    def __init__(self, 
                 grid_shape, 
                 environment_type, 
                 obstacles=None, 
                 materials=None, 
                 dirts=None, 
                 objectives=None):
        """
        grid_shape: (rows, cols)
        environment_type: 'hangar','warehouse','garage'
        obstacles: list of (r,c)
        materials: list of (r,c) cells containing materials
        dirts: list of (r,c) cells that are dirty
        objectives: list of special cells (exit or stations)
        """
        self.grid_rows, self.grid_cols = grid_shape
        self.environment_type = environment_type  # 'hangar','warehouse','garage'
        self.obstacles = obstacles if obstacles else []
        self.material_list = materials if materials else []
        self.dirt_list = dirts if dirts else []
        self.cleaned_dirts = set()
        self.objectives = objectives if objectives else []

        # Count them
        self.num_materials = len(self.material_list)
        self.num_dirts = len(self.dirt_list)
        self.total_flags = self.num_materials + self.num_dirts

        # For 'garage'
        if self.environment_type == "garage":
            self.total_flags = 0

        # Single-agent moves: Up=0, Right=1, Down=2, Left=3 (or chosen ordering)
        self.actions = [(-1,0),(0,1),(1,0),(0,-1)]

    def valid_cell(self, row_check, col_check):
        """Check if (row_check,col_check) is inside the grid and not an obstacle."""
        if not (0 <= row_check < self.grid_rows and 0 <= col_check < self.grid_cols):
            return False
        return (row_check, col_check) not in self.obstacles

    def terminal_condition(self, rowAgent1, colAgent1, rowAgent2, colAgent2, progressMask):
        """
        Decide if we are in a terminal state for the scenario.
        """
        # HANGAR
        if self.environment_type == "hangar":
            # 2 materials => need progressMask == 3 => both on single exit
            all_collected = (progressMask == (1 << self.num_materials) - 1)
            if all_collected and len(self.objectives) == 1:
                exit_cell = self.objectives[0]
                if (rowAgent1,colAgent1) == exit_cell and (rowAgent2,colAgent2) == exit_cell:
                    return True
            return False

        # WAREHOUSE
        elif self.environment_type == "warehouse":
            needed_tracker = (1 << (self.num_materials + self.num_dirts)) - 1
            all_cleaned = (progressMask == needed_tracker)
            if all_cleaned and len(self.objectives) == 1:
                exit_position = self.objectives[0]
                return (rowAgent1,colAgent1) == exit_position and (rowAgent2,colAgent2) == exit_position
            return False

        # GARAGE
        elif self.environment_type == "garage":
            if len(self.objectives) == 2:
                station_set = set(self.objectives)
                agent_positions_set = {(rowAgent1,colAgent1),(rowAgent2,colAgent2)}
                return (agent_positions_set == station_set)
            return False

        else:
            return False

    def step_function(self, rowAgent1, colAgent1, rowAgent2, colAgent2, progressMask, actionAgent1, actionAgent2, 
                      reward_step=-1, 
                      reward_collect=10,
                      penalty_revisit=-5,
                      final_reward=20,
                      station_reward=100):
        """
        Single step for the environment:
        - Move each agent if valid
        - Check for material/dirt collection
        - If terminal, add final/station reward
        - Return (nextRowAgent1, nextColAgent1, nextRowAgent2, nextColAgent2, newProgressMask, immediate_reward)
        """
        # Move agent1
        nextRowAgent1 = rowAgent1 + self.actions[actionAgent1][0]
        nextColAgent1 = colAgent1 + self.actions[actionAgent1][1]
        if not self.valid_cell(nextRowAgent1, nextColAgent1):
            nextRowAgent1, nextColAgent1 = rowAgent1, colAgent1

        # Move agent2
        nextRowAgent2 = rowAgent2 + self.actions[actionAgent2][0]
        nextColAgent2 = colAgent2 + self.actions[actionAgent2][1]
        if not self.valid_cell(nextRowAgent2, nextColAgent2):
            nextRowAgent2, nextColAgent2 = rowAgent2, colAgent2

        # Base step cost
        immediate_reward = reward_step
        newProgressMask = progressMask

        if self.environment_type == "hangar":
            # collect materials if any
            if (nextRowAgent1,nextColAgent1) in self.material_list:
                materialIndexPosition1 = material_index((nextRowAgent1,nextColAgent1), self.material_list)
                if not check_collected(progressMask, materialIndexPosition1):
                    newProgressMask = set_collected(progressMask, materialIndexPosition1)
                    immediate_reward += reward_collect

            if (nextRowAgent2,nextColAgent2) in self.material_list:
                materialIndexPosition2 = material_index((nextRowAgent2,nextColAgent2), self.material_list)
                if not check_collected(newProgressMask, materialIndexPosition2):
                    newProgressMask = set_collected(newProgressMask, materialIndexPosition2)
                    immediate_reward += reward_collect

            if self.terminal_condition(nextRowAgent1,nextColAgent1,nextRowAgent2,nextColAgent2,newProgressMask):
                immediate_reward += final_reward

        elif self.environment_type == "warehouse":
            # cleaning logic
            if (nextRowAgent1,nextColAgent1) in self.dirt_list:
                dirtIndexPosition1 = dirt_index((nextRowAgent1,nextColAgent1), self.dirt_list)
                if not check_cleaned(progressMask, dirtIndexPosition1, self.num_materials):
                    newProgressMask = set_cleaned(progressMask, dirtIndexPosition1, self.num_materials)
                    immediate_reward += reward_collect
                    self.cleaned_dirts.add((nextRowAgent1,nextRowAgent1))
                else:
                    immediate_reward += penalty_revisit

            if (nextRowAgent2,nextColAgent2) in self.dirt_list:
                dirtIndexPosition2 = dirt_index((nextRowAgent2,nextColAgent2), self.dirt_list)
                if not check_cleaned(newProgressMask, dirtIndexPosition2, self.num_materials):
                    newProgressMask = set_cleaned(newProgressMask, dirtIndexPosition2, self.num_materials)
                    immediate_reward += reward_collect
                    self.cleaned_dirts.add((nextRowAgent2, nextColAgent2))
                else:
                    immediate_reward += penalty_revisit

            if self.terminal_condition(nextRowAgent1,nextColAgent1,nextRowAgent2,nextColAgent2,newProgressMask):
                immediate_reward += final_reward

        elif self.environment_type == "garage":
            # distinct stations => final reward
            if self.terminal_condition(nextRowAgent1,nextColAgent1,nextRowAgent2,nextColAgent2,newProgressMask):
                immediate_reward += station_reward

        return nextRowAgent1, nextColAgent1, nextRowAgent2, nextColAgent2, newProgressMask, immediate_reward


###############################################################################
# Value Iteration + Simulation
###############################################################################
class TwoAgentValueIteration:
    def __init__(self, 
                 environment: TwoAgentGridMDP, 
                 discount_factor=0.9, 
                 tolerance=1e-4,
                 step_cost=-1, 
                 collect_reward=10, 
                 revisit_penalty=-5, 
                 end_reward=20,
                 station_reward=100):
        self.environment = environment
        self.gamma = discount_factor
        self.theta = tolerance
        self.reward_step = step_cost
        self.reward_collect = collect_reward
        self.reward_revisit = revisit_penalty
        self.reward_final = end_reward
        self.reward_station = station_reward

        self.num_rows = environment.grid_rows
        self.num_cols = environment.grid_cols
        self.num_bits = environment.total_flags if environment.environment_type != "garage" else 0

        shape_mask = max(1, (1 << self.num_bits))
        self.value_function = np.zeros((self.num_rows, self.num_cols, 
                                        self.num_rows, self.num_cols, shape_mask))

        # Single-agent actions [Up=0, Right=1, Down=2, Left=3], or chosen ordering
        self.all_actions = [0,1,2,3]

    def is_end_state(self, rowAgent1, colAgent1, rowAgent2, colAgent2, progressMask):
        return self.environment.terminal_condition(rowAgent1,colAgent1,rowAgent2,colAgent2, progressMask)

    def do_step(self, rowAgent1, colAgent1, rowAgent2, colAgent2, progressMask, actionAgent1, actionAgent2):
        return self.environment.step_function(
            rowAgent1, colAgent1, rowAgent2, colAgent2, progressMask,
            actionAgent1, actionAgent2,
            reward_step=self.reward_step,
            reward_collect=self.reward_collect,
            penalty_revisit=self.reward_revisit,
            final_reward=self.reward_final,
            station_reward=self.reward_station
        )

    def run_value_iteration(self, max_iterations=1000):
        joint_actions = list(itertools.product(self.all_actions, self.all_actions))
        shape_mask = max(1, (1 << self.num_bits))

        for iteration_index in range(max_iterations):
            max_difference = 0.0
            new_value_function = np.copy(self.value_function)

            for rowA1 in range(self.num_rows):
                for colA1 in range(self.num_cols):
                    for rowA2 in range(self.num_rows):
                        for colA2 in range(self.num_cols):
                            for bitmask in range(shape_mask):
                                if self.is_end_state(rowA1,colA1,rowA2,colA2,bitmask):
                                    continue
                                if (rowA1,colA1) in self.environment.obstacles or (rowA2,colA2) in self.environment.obstacles:
                                    continue

                                best_value = float('-inf')
                                for (action1, action2) in joint_actions:
                                    (nextRowA1, nextColA1,
                                     nextRowA2, nextColA2,
                                     newProgressMask, immediateReward
                                    ) = self.do_step(rowA1,colA1,rowA2,colA2,bitmask,action1,action2)
                                    
                                    candidate_value = immediateReward + self.gamma * self.value_function[nextRowA1,
                                                                                                          nextColA1,
                                                                                                          nextRowA2,
                                                                                                          nextColA2,
                                                                                                          newProgressMask]
                                    if candidate_value > best_value:
                                        best_value = candidate_value

                                old_value = self.value_function[rowA1,colA1,rowA2,colA2,bitmask]
                                new_value_function[rowA1,colA1,rowA2,colA2,bitmask] = best_value
                                max_difference = max(max_difference, abs(best_value - old_value))

            self.value_function = new_value_function
            if max_difference < self.theta:
                print(f"Converged after {iteration_index} iterations.")
                break

    def extract_policy(self):
        shape_mask = max(1, (1 << self.num_bits))
        policy_array = np.full((self.num_rows, self.num_cols, self.num_rows, self.num_cols, shape_mask), -1, dtype=int)
        joint_actions = list(itertools.product(self.all_actions, self.all_actions))

        for rowA1 in range(self.num_rows):
            for colA1 in range(self.num_cols):
                for rowA2 in range(self.num_rows):
                    for colA2 in range(self.num_cols):
                        for bitmask_val in range(shape_mask):
                            if self.is_end_state(rowA1,colA1,rowA2,colA2,bitmask_val):
                                continue
                            if (rowA1,colA1) in self.environment.obstacles or (rowA2,colA2) in self.environment.obstacles:
                                continue

                            best_value = float('-inf')
                            best_index = -1
                            for index_joint, (action1, action2) in enumerate(joint_actions):
                                (nRowA1, nColA1, nRowA2, nColA2,
                                 newProgMask, rewardImmediate
                                ) = self.do_step(rowA1,colA1,rowA2,colA2,bitmask_val,action1,action2)
                                next_val = rewardImmediate + self.gamma * self.value_function[nRowA1,
                                                                                              nColA1,
                                                                                              nRowA2,
                                                                                              nColA2,
                                                                                              newProgMask]
                                if next_val > best_value:
                                    best_value = next_val
                                    best_index = index_joint
                            policy_array[rowA1,colA1,rowA2,colA2,bitmask_val] = best_index
        return policy_array

    def print_console_grid(self, rowA1, colA1, rowA2, colA2, progressMask):
        if self.environment.environment_type == "hangar":
            collectedMaterialIndices = []
            for materialIdx in range(self.environment.num_materials):
                if check_collected(progressMask, materialIdx):
                    collectedMaterialIndices.append(materialIdx)
            print(f"Materials collected so far: {collectedMaterialIndices}")

        elif self.environment.environment_type == "warehouse":
            cleanedIndices = []
            for dirtIdx in range(self.environment.num_dirts):
                combinedBit = dirtIdx + self.environment.num_materials
                if (progressMask & (1 << combinedBit)) != 0:
                    cleanedIndices.append(dirtIdx)
            print(f"Dirty cells cleaned so far: {cleanedIndices}")
        else:
            print("Garage scenario.")

        for rowCheck in range(self.num_rows):
            row_string = ""
            for colCheck in range(self.num_cols):
                if (rowCheck,colCheck) == (rowA1,colA1) and (rowCheck,colCheck) == (rowA2,colA2):
                    row_string += "X "
                elif (rowCheck,colCheck) == (rowA1,colA1):
                    row_string += "A1"
                elif (rowCheck,colCheck) == (rowA2,colA2):
                    row_string += "A2"
                else:
                    if (rowCheck,colCheck) in self.environment.obstacles:
                        row_string += "X "
                    elif (rowCheck,colCheck) in self.environment.material_list:
                        matIndex = material_index((rowCheck,colCheck), self.environment.material_list)
                        if check_collected(progressMask, matIndex):
                            row_string += ". "
                        else:
                            row_string += "M "
                    elif (rowCheck,colCheck) in self.environment.dirt_list:
                        dIndex = dirt_index((rowCheck,colCheck), self.environment.dirt_list)
                        if check_cleaned(progressMask, dIndex, self.environment.num_materials):
                            row_string += "C "
                        else:
                            row_string += "D "
                    elif (rowCheck,colCheck) in self.environment.objectives:
                        row_string += "O "
                    else:
                        row_string += "- "
            print(row_string)

    def simulate_policy_in_console(self, policy_array, start_positions, max_steps=50):
        """
        We now store the trajectory for each agent (positions) + the actions each agent took.
        We'll also keep the existing console prints.
        Returns:
          agent1_positions, agent1_actions, agent2_positions, agent2_actions
        """
        (rowAgent1, colAgent1), (rowAgent2, colAgent2) = start_positions
        progressMask = 0
        current_step = 0
        totalReward = 0.0

        # We'll store each agent's positions + actions
        agent1_positions_list = []
        agent2_positions_list = []
        agent1_actions_list = []
        agent2_actions_list = []

        # Recreate the joint action space
        all_joint_actions = list(itertools.product(self.all_actions, self.all_actions))

        while current_step < max_steps:
            # Add current positions to the trajectory
            agent1_positions_list.append((rowAgent1,colAgent1))
            agent2_positions_list.append((rowAgent2,colAgent2))

            # Check if terminal
            if self.is_end_state(rowAgent1, colAgent1, rowAgent2, colAgent2, progressMask):
                print(f"\nReached terminal state at step {current_step}.")
                break

            # Print the grid
            print(f"\nStep {current_step}:")
            self.print_console_grid(rowAgent1, colAgent1, rowAgent2, colAgent2, progressMask)

            best_joint_index = policy_array[rowAgent1,colAgent1,rowAgent2,colAgent2,progressMask]
            if best_joint_index < 0:
                print("No policy for this state. Stopping.")
                break

            (actionAg1, actionAg2) = all_joint_actions[best_joint_index]
            agent1_actions_list.append(actionAg1)
            agent2_actions_list.append(actionAg2)

            (nextRowA1, nextColA1,
             nextRowA2, nextColA2,
             newProgMask, immediateRew
            ) = self.do_step(rowAgent1, colAgent1, rowAgent2, colAgent2,
                             progressMask, actionAg1, actionAg2)
            totalReward += immediateRew
            rowAgent1, colAgent1 = nextRowA1, nextColA1
            rowAgent2, colAgent2 = nextRowA2, nextColA2
            progressMask = newProgMask
            current_step += 1

        # final position after loop
        agent1_positions_list.append((rowAgent1,colAgent1))
        agent2_positions_list.append((rowAgent2,colAgent2))

        print(f"\nFinal Step {current_step}:")
        self.print_console_grid(rowAgent1, colAgent1, rowAgent2, colAgent2, progressMask)
        print(f"Stopped after {current_step} steps. Total reward = {totalReward}")

        # Return the recorded trajectories & actions
        return agent1_positions_list, agent1_actions_list, agent2_positions_list, agent2_actions_list



def plot_grid_multi_agent(grid_array, title, agent1_positions, agent1_actions,
                          agent2_positions, agent2_actions):
    """
    grid_array: 2D array of shape [rows, cols] with numeric codes for empty, obstacle, etc.
    agent*_positions: list of (row,col) positions at each step
    agent*_actions:   list of action ints (0=Up,1=Right,2=Down,3=Left), or your chosen ordering

    We’ll draw an arrow from each position in positions to indicate the action.
    """
    cmap = ListedColormap(["grey", "black", "blue", "red", "green", "purple"])
    plt.figure(figsize=(6, 6))
    plt.imshow(grid_array, cmap=cmap, origin="upper")
    plt.colorbar(ticks=[0, 1, 2, 3, 4, 5],
                 label="0=Empty, 1=Obstacle, 2=Material, 3=Dirt, 4=Unused, 5=Objective")

    # Define small deltas for each action index
    # (assuming your code uses Up=0, Right=1, Down=2, Left=3)
    action_deltas = {
        0: (-0.3, 0.0),   # Up => move in negative row direction
        1: (0.0, 0.3),    # Right => move in positive col direction
        2: (0.3, 0.0),    # Down => move in positive row direction
        3: (0.0, -0.3)    # Left => move in negative col direction
    }

    colors = ["yellow", "cyan"]  # one color for each agent
    
    # For each agent, we draw arrows
    for agent_index, (positions_list, actions_list) in enumerate([
        (agent1_positions, agent1_actions),
        (agent2_positions, agent2_actions)
    ]):
        color_this_agent = colors[agent_index % len(colors)]
        for step_idx in range(len(actions_list)):
            (rowCoord, colCoord) = positions_list[step_idx]
            actNow = actions_list[step_idx]
            (deltaRow, deltaCol) = action_deltas[actNow]

            # In matplotlib, x=column, y=row => arrow(x,y,dx,dy)
            # But positions are (row, col), so x=colCoord, y=rowCoord
            if step_idx == 0:
                plt.arrow(colCoord, rowCoord, deltaCol, deltaRow, head_width=0.2, head_length=0.2,
                          fc="red", ec="black")  # highlight start
            else:
                if agent_index == 0:
                    plt.arrow(colCoord, rowCoord, deltaCol, deltaRow, head_width=0.2, head_length=0.2,
                              fc=color_this_agent, ec="black")
                else:
                    plt.arrow(colCoord, rowCoord, deltaCol, deltaRow, head_width=0.1, head_length=0.1,
                              fc=color_this_agent, ec="black")

    plt.title(title)
    plt.show()

def create_plotting_grid(environment_instance):
    """
    Convert your TwoAgentGridMDP environment into a 2D array for plotting:
      0=empty, 1=obstacle, 2=material, 3=dirt, 4=cleaned, 5=objective
    """
    grid_array = np.zeros((environment_instance.grid_rows, environment_instance.grid_cols), dtype=int)
    
    # Mark obstacles as 1
    for (obsR,obsC) in environment_instance.obstacles:
        grid_array[obsR,obsC] = 1
    
    # Mark materials as 2
    for (matR,matC) in environment_instance.material_list:
        grid_array[matR,matC] = 2
    
    # Mark dirts as 3
    for (dR,dC) in environment_instance.dirt_list:
        grid_array[dR,dC] = 3
    
    # Mark objectives as 5
    for (objR,objC) in environment_instance.objectives:
        grid_array[objR,objC] = 5
        
    # Mark cleaned as 4
    for (cR,cC) in environment_instance.cleaned_dirts:
        grid_array[cR,cC] = 4
    return grid_array




if __name__ == "__main__":

    ###############################################################################
    # HANGAR EXAMPLE
    ###############################################################################
    hangar_environment = TwoAgentGridMDP(
        grid_shape=(4,4),
        environment_type="hangar",
        obstacles=[(0,1),(2,1)],
        materials=[(1,1),(3,2)],
        dirts=[],
        objectives=[(3,3)]
    )
    hangar_solver = TwoAgentValueIteration(
        environment=hangar_environment,
        discount_factor=0.9,
        tolerance=1e-4,
        step_cost=-1,
        collect_reward=10,
        revisit_penalty=-5,  
        end_reward=20,
        station_reward=100   
    )
    hangar_solver.run_value_iteration(max_iterations=100)
    hangar_policy = hangar_solver.extract_policy()

    (agent1_positions_h, agent1_actions_h,
     agent2_positions_h, agent2_actions_h
    ) = hangar_solver.simulate_policy_in_console(
        hangar_policy,
        start_positions=((0,0),(0,2)),
        max_steps=100
    )

    # Optional: Plot the final path
    hangar_grid = create_plotting_grid(hangar_environment)
    plot_grid_multi_agent(
        hangar_grid, 
        "Hangar Environment Path",
        agent1_positions=agent1_positions_h,
        agent1_actions=agent1_actions_h,
        agent2_positions=agent2_positions_h,
        agent2_actions=agent2_actions_h
    )

    ###############################################################################
    # WAREHOUSE EXAMPLE
    ###############################################################################
    warehouse_environment = TwoAgentGridMDP(
        grid_shape=(4,5),
        environment_type="warehouse",
        obstacles=[(0,3),(0,2),(1,1)],
        materials=[],
        dirts=[(1,0),(1,4),(2,3),(2,1),(3,2),(3,4)],
        objectives=[(0,4)]
    )
    warehouse_solver = TwoAgentValueIteration(
        environment=warehouse_environment,
        discount_factor=0.9,
        tolerance=1e-4,
        step_cost=-1,
        collect_reward=10,    # cleaning reward
        revisit_penalty=-5,
        end_reward=50,
        station_reward=0      # not used here
    )
    warehouse_solver.run_value_iteration(max_iterations=100)
    warehouse_policy = warehouse_solver.extract_policy()

    (agent1_positions_w, agent1_actions_w,
     agent2_positions_w, agent2_actions_w
    ) = warehouse_solver.simulate_policy_in_console(
        warehouse_policy,
        start_positions=((1,2),(3,0)),  # example
        max_steps=100
    )

    # Optional: Plot the final path
    warehouse_grid = create_plotting_grid(warehouse_environment)
    plot_grid_multi_agent(
        warehouse_grid, 
        "Warehouse Environment Path",
        agent1_positions=agent1_positions_w,
        agent1_actions=agent1_actions_w,
        agent2_positions=agent2_positions_w,
        agent2_actions=agent2_actions_w
    )

    ###############################################################################
    # GARAGE EXAMPLE
    ###############################################################################
    garage_environment = TwoAgentGridMDP(
        grid_shape=(3,5),
        environment_type="garage",
        obstacles=[(1,1),(2,3)],
        materials=[],
        dirts=[],
        objectives=[(2,4),(0,3)]
    )
    garage_solver = TwoAgentValueIteration(
        environment=garage_environment,
        discount_factor=0.9,
        tolerance=1e-4,
        step_cost=-1,
        collect_reward=0,     
        revisit_penalty=0,    
        end_reward=0,         
        station_reward=100
    )
    garage_solver.run_value_iteration(max_iterations=100)
    garage_policy = garage_solver.extract_policy()

    (agent1_positions_g, agent1_actions_g,
     agent2_positions_g, agent2_actions_g
    ) = garage_solver.simulate_policy_in_console(
        garage_policy,
        start_positions=((2,0),(2,1)),  # both agents at bottom-left
        max_steps=50
    )

    # Optional: Plot the final path
    garage_grid = create_plotting_grid(garage_environment)
    plot_grid_multi_agent(
        garage_grid, 
        "Garage Environment Path",
        agent1_positions=agent1_positions_g,
        agent1_actions=agent1_actions_g,
        agent2_positions=agent2_positions_g,
        agent2_actions=agent2_actions_g
    )
