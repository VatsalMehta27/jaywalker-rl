import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np


class JaywalkEnv(gym.Env):
    def __init__(
        self,
        max_vehicles=5,
        p_vehicle_spawn=0.4,
        p_vehicle_stop_pedestrian=0.8,
        mean_vehicle_speed=1.5,
        use_traffic_light=True,
    ):
        super(JaywalkEnv, self).__init__()

        # Define grid dimensions
        self.grid_shape = (4, 11)
        self.crosswalk_column = self.grid_shape[1] // 2
        self.vehicle_lanes = [1, 2]

        # Start agent in the middle of the bottom row
        self.agent_start_position = (0, self.crosswalk_column)  # Row 3, Column 5
        self.agent_position = list(self.agent_start_position)  # Make it mutable

        self.goal_position = (3, self.crosswalk_column)

        self.actions = {1: "forward", -1: "backward", 0: "wait"}

        # Action space: using discrete integers
        self.action_space = spaces.Discrete(len(self.actions), start=-1)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "agent_position": spaces.Box(low=0, high=3, shape=(2,), dtype=np.int32),
                "vehicles": spaces.Dict(
                    {
                        lane: spaces.Box(
                            low=0, high=10, shape=(max_vehicles, 2), dtype=np.int32
                        )
                        for lane in self.vehicle_lanes
                    }
                ),
            }
        )

        # Vehicle parameters
        self.num_vehicles = 0
        self.max_vehicles = max_vehicles
        self.p_vehicle_spawn = p_vehicle_spawn
        self.p_vehicle_stop_pedestrian = p_vehicle_stop_pedestrian
        self.mean_vehicle_speed = mean_vehicle_speed
        self.road = [list() for _ in range(len(self.vehicle_lanes))]

        self.use_traffic_light = use_traffic_light
        self.light_color = "GREEN"
        self.light_timer = 0
        self.light_durations = {
            "GREEN": 10,
            "YELLOW": 3,
            "RED": 5,
        }

    def reset(self):
        """Resets the environment to the initial state and returns the initial observation."""
        self.agent_position = list(self.agent_start_position)
        self.road = [list() for _ in range(len(self.vehicle_lanes))]
        self.num_vehicles = 0
        self.light_color = "GREEN"
        self.light_timer = 0

        return self._get_observation()

    def step(self, action):
        """Executes one time step in the environment based on the given action."""
        if action == 1:  # Move forward
            self.agent_position[0] = max(self.agent_position[0] + 1, 0)
        elif action == -1:  # Move backward
            self.agent_position[0] = min(self.agent_position[0] - 1, 3)
        elif action == 0:  # Wait
            pass
        else:
            raise Exception("Invalid action.")

        pedestrian_collision = self._advance_vehicles()

        reward = 0
        done = False

        if pedestrian_collision:
            reward = -10
            done = True
        else:
            self._spawn_vehicle()
            self._update_traffic_light()

            # Define reward structure, terminal conditions, etc.
            if (
                self.agent_position[0] == self.goal_position[0]
                and self.agent_position[1] == self.goal_position[1]
            ):
                reward = 1
                done = True

        # Return observation, reward, done, info
        return self._get_observation(), reward, done, {}

    def _update_traffic_light(self):
        if self.use_traffic_light:
            self.light_timer += 1

            if self.light_timer >= self.light_durations[self.light_color]:
                match self.light_color:
                    case "GREEN":
                        self.light_color = "YELLOW"
                    case "YELLOW":
                        self.light_color = "RED"
                    case "RED":
                        self.light_color = "GREEN"

                self.light_timer = 0

    def _spawn_vehicle(self):
        """Spawns a new vehicle based on the defined probability and rules."""
        if (
            self.num_vehicles < self.max_vehicles
            and np.random.rand() < self.p_vehicle_spawn
        ):
            # Randomly select a lane (1 or 2) where the first column is unoccupied
            spawn_lanes = [
                lane_idx
                for lane_idx in range(len(self.vehicle_lanes))
                if not self._is_first_column_occupied(lane_idx)
            ]

            if spawn_lanes:
                lane_idx = np.random.choice(
                    spawn_lanes
                )  # Select a random unoccupied lane index
                speed = max(1, round(np.random.normal(self.mean_vehicle_speed, 1)))
                vehicle = (speed, 0)  # Vehicle starts at column 0 in its lane

                self.road[lane_idx].append(vehicle)
                self.num_vehicles += 1

    def _is_first_column_occupied(self, lane):
        """Checks if the first column (column index 0) of the specified lane is occupied by a vehicle."""
        return any(vehicle[1] == 0 for vehicle in self.road[lane])

    def _advance_vehicles(self):
        """
        Advances all vehicles in the grid by their speed, adjusting speeds to avoid collisions.
        """
        pedestrian_collision = False
        updated_road = []

        for lane_idx, lane in enumerate(self.road):
            sorted_lane = sorted(lane, key=lambda v: -1 * v[1])

            for i, vehicle in enumerate(sorted_lane):
                speed, position = vehicle

                new_speed = speed
                new_position = position + speed

                # Check for vehicle collision
                if i > 0 and sorted_lane[i - 1][1] <= new_position:
                    new_position = sorted_lane[i - 1][1] - 1
                    new_speed = sorted_lane[i - 1][0]
                elif (
                    position < self.crosswalk_column <= new_position
                ):  # Crossing the crosswalk
                    match self.light_color:
                        case "RED":
                            new_position = self.crosswalk_column - 1
                        case "YELLOW":
                            vehicle_stop = np.random.rand()

                            # TODO: Update this with some yellow light based prob of stopping?
                            if vehicle_stop < self.p_vehicle_stop_pedestrian:
                                new_position = self.crosswalk_column - 1
                        case "GREEN":
                            if self.vehicle_lanes[lane_idx] == self.agent_position[0]:
                                vehicle_stop = np.random.rand()

                                if vehicle_stop < self.p_vehicle_stop_pedestrian:
                                    new_position = self.crosswalk_column - 1
                                else:
                                    pedestrian_collision = True

                sorted_lane[i] = (new_speed, new_position)

            filtered_lane = list(
                filter(lambda v: v[1] < self.grid_shape[1], sorted_lane)
            )

            updated_road.append(filtered_lane)

        self.road = updated_road

        return pedestrian_collision

    def _get_observation(self):
        """Constructs the current observation as the agent's position and vehicles grouped by lane."""
        # Prepare a dictionary to hold vehicles per lane
        vehicles_dict = {
            lane: np.zeros((self.max_vehicles, 2), dtype=np.int32)
            for lane in self.vehicle_lanes
        }

        # Fill in each lane's vehicles based on self.road, ensuring consistent shape
        for lane_idx, lane_vehicles in enumerate(self.road):
            for i, (speed, position) in enumerate(lane_vehicles):
                if i < self.max_vehicles:
                    vehicles_dict[self.vehicle_lanes[lane_idx]][i] = [speed, position]

        # Construct and return the observation dictionary
        return {
            "agent_position": np.array(self.agent_position, dtype=np.int32),
            "vehicles": vehicles_dict,
        }

    @staticmethod
    def _plot_grid(grid):
        # Convert the grid into a numerical array
        color_array = np.zeros((len(grid), len(grid[0])), dtype=int)

        # Map 'V' to 1 (red), 'A' to 2 (green), and '.' to 0 (white)
        for i, row in enumerate(grid):
            for j, char in enumerate(row):
                if char == "V":
                    color_array[i, j] = 1
                elif char == "A":
                    color_array[i, j] = 2
                else:
                    color_array[i, j] = 0

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Create a colormap: 0 -> white, 1 -> red, 2 -> green
        cmap = plt.cm.colors.ListedColormap(["white", "red", "green"])

        # Plot the grid
        ax.imshow(color_array, cmap=cmap, aspect="equal")

        # Set the limits of the plot to match the grid size
        ax.set_xlim(-0.5, color_array.shape[1] - 0.5)
        ax.set_ylim(color_array.shape[0] - 0.5, -0.5)

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, color_array.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, color_array.shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

        # Remove the major ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Display the plot
        plt.show()

    def render(self, mode="ansi"):
        """Optional: Render the current state of the environment for debugging."""
        grid = np.full(self.grid_shape, ".", dtype=str)
        grid[self.agent_position[0], self.agent_position[1]] = "A"  # Place agent

        for lane_idx, lane in enumerate(self.road):
            for vehicle in lane:
                _, position = vehicle

                if position < self.grid_shape[1]:  # Ensure position is within bounds
                    grid[self.vehicle_lanes[lane_idx], position] = "V"

        grid_lanes = ["".join(row) for row in grid]

        if mode == "ansi":
            print("\n".join(grid_lanes))
            return

        if mode == "human":
            self._plot_grid(grid_lanes)
            return


# Example of how to use the environment
if __name__ == "__main__":
    env = JaywalkEnv()
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action = int(
            input("Choose an action (0: forward, 1: backward, 2: wait): ")
        )  # 0: forward, 1: backward, 2: wait
        obs, reward, done, info = env.step(action)
