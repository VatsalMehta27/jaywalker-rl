from dataclasses import dataclass, field
from itertools import cycle
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np


@dataclass
class EnvParams:
    max_vehicles: int = 5
    p_vehicle_spawn: float = 0.4
    p_vehicle_stop_pedestrian: float = 0.8
    mean_vehicle_speed: float = 1.5
    use_traffic_light: bool = True
    num_consecutive_roads: int = 2
    num_columns: int = 11
    light_durations: dict[str, int] = field(
        default_factory=lambda: {"GREEN": 10, "YELLOW": 3, "RED": 5}
    )


class JaywalkEnv(gym.Env):
    def __init__(self, params: EnvParams):
        super(JaywalkEnv, self).__init__()

        self.num_consecutive_roads = params.num_consecutive_roads
        # TODO: Remove hardcoded 3
        num_rows = 3 + self.num_consecutive_roads * 2
        self.grid_shape = (num_rows, params.num_columns)
        self.crosswalk_column = self.grid_shape[1] // 2

        self.agent_start_position = (0, self.crosswalk_column)
        self.goal_position = (self.grid_shape[0] - 1, self.crosswalk_column)

        self.agent_position = list(self.agent_start_position)
        self.agent_representation = 100

        self.actions = {-1: "backward", 0: "wait", 1: "forward"}

        # Action space: using discrete integers
        self.action_space = spaces.Discrete(len(self.actions), start=-1)
        self.observation_space = spaces.Dict(
            {
                "traffic_light": spaces.Discrete(3),  # 0: RED, 1: YELLOW, 2: GREEN
                "world_grid": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.grid_shape,
                    dtype=np.int32,
                ),
            }
        )

        self.num_vehicles = 0
        self.max_vehicles = params.max_vehicles
        self.p_vehicle_spawn = params.p_vehicle_spawn
        self.p_vehicle_stop_pedestrian = params.p_vehicle_stop_pedestrian
        self.mean_vehicle_speed = params.mean_vehicle_speed

        # TODO: Remove hardcoded 2
        # TODO: Introduce "direction" aspect
        self.lanes = {i: [] for i in range(self.grid_shape[0]) if i % 3 != 0}

        self.use_traffic_light = params.use_traffic_light
        self.light_durations = params.light_durations
        self.light_cycle = cycle(self.light_durations.keys())
        self.light_color = next(self.light_cycle)
        self.light_index = 0
        self.light_timer = 0

        self.time_steps = 0

    def reset(self):
        # Reset agent position
        self.agent_position = list(self.agent_start_position)

        # Reset all vehicles
        self.num_vehicles = 0
        self.lanes = {i: [] for i in range(self.grid_shape[0]) if i % 3 != 0}

        # Reset traffic light
        self.light_cycle = cycle(self.light_durations.keys())
        self.light_color = next(self.light_cycle)
        self.light_index = 0
        self.light_timer = 0

        self.time_steps = 0

        return self._get_observation()

    def step(self, action):
        match action:
            case 1:
                self.agent_position[0] = max(self.agent_position[0] + 1, 0)
            case -1:
                self.agent_position[0] = min(self.agent_position[0] - 1, 3)
            case 0:
                pass
            case _:
                raise Exception("Invalid action.")

        pedestrian_collion = self._advance_vehicles()
        print("Pedestrian Collision:", pedestrian_collion)

        reward = 0
        done = False

        if pedestrian_collion:
            reward = -10
            done = True
        else:
            self._spawn_vehicle()
            self._update_traffic_light()

            # Reward for reaching goal position
            if (
                self.agent_position[0] == self.goal_position[0]
                and self.agent_position[1] == self.goal_position[1]
            ):
                reward = 1
                done = True

        self.time_steps += 1

        return self._get_observation(), reward, done, {}

    def _advance_vehicles(self):
        pedestrian_collision = False

        for lane_idx, lane in self.lanes.items():
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
                            if lane_idx == self.agent_position[0]:
                                vehicle_stop = np.random.rand()

                                if vehicle_stop < self.p_vehicle_stop_pedestrian:
                                    new_position = self.crosswalk_column - 1
                                else:
                                    pedestrian_collision = True

                sorted_lane[i] = (new_speed, new_position)

            filtered_lane = list(
                filter(lambda v: v[1] < self.grid_shape[1], sorted_lane)
            )

            self.lanes[lane_idx] = filtered_lane

        return pedestrian_collision

    def _spawn_vehicle(self):
        if (
            self.num_vehicles < self.max_vehicles
            and np.random.rand() < self.p_vehicle_spawn
        ):
            spawn_lanes = [
                lane_idx
                for lane_idx in self.lanes.keys()
                if not self._is_first_column_occupied(lane_idx)
            ]
            print("Spawn Lanes:", spawn_lanes)

            if spawn_lanes:
                lane_idx = np.random.choice(
                    spawn_lanes
                )  # Select a random unoccupied lane index
                speed = max(1, round(np.random.normal(self.mean_vehicle_speed, 1)))
                vehicle = (speed, 0)  # Vehicle starts at column 0 in its lane

                self.lanes[lane_idx].append(vehicle)
                self.num_vehicles += 1

    def _is_first_column_occupied(self, lane_idx):
        """Checks if the first column (column index 0) of the specified lane is occupied by a vehicle."""
        return any(vehicle[1] == 0 for vehicle in self.lanes[lane_idx])

    def _update_traffic_light(self):
        if self.use_traffic_light:
            self.light_timer += 1

            if self.light_timer >= self.light_durations[self.light_color]:
                self.light_color = next(self.light_cycle)
                self.light_timer = 0

                self.light_index += 1
                if self.light_index >= len(self.light_durations):
                    self.light_index = 0

    def _get_observation(self):
        grid = np.zeros(self.grid_shape, dtype=np.int32)
        grid[self.agent_position[0], self.agent_position[1]] = self.agent_representation

        for lane_idx, lane_vehicles in self.lanes.items():
            for velocity, position in lane_vehicles:
                grid[lane_idx, position] = velocity

        return {"traffic_light": self.light_index, "world_grid": grid}

    def _plot_grid(self, grid):
        # Convert the grid into a numerical array for easy visualization
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

        # Set up the figure with two subplots: one for the grid and one for the traffic light
        fig, (ax, traffic_light_ax) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [4, 1]}, figsize=(10, 6)
        )

        # Plot the grid on the left subplot
        cmap = plt.cm.colors.ListedColormap(["white", "red", "green"])
        ax.imshow(color_array, cmap=cmap, aspect="equal")

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, color_array.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, color_array.shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])

        # Set up the traffic light circles on the right subplot
        light_colors = {color: color.lower() for color in self.light_durations.keys()}
        inactive_color = "gray"

        # Draw the three traffic light circles
        for idx, light in enumerate(self.light_durations.keys()):
            color = light_colors[light] if light == self.light_color else inactive_color
            circle = patches.Circle(
                (0.5, 2.5 - idx), 0.4, edgecolor="black", facecolor=color
            )
            traffic_light_ax.add_patch(circle)

        # Remove the axes for the traffic light subplot for a cleaner look
        traffic_light_ax.set_xlim(0, 1)
        traffic_light_ax.set_ylim(0, 3)
        traffic_light_ax.axis("off")

        # Display the plot
        plt.tight_layout()
        plt.show()

    def render(self, mode="ansi"):
        """Optional: Render the current state of the environment for debugging."""
        grid = self._get_observation()["world_grid"]

        grid_visualization = np.full(grid.shape, ".", dtype=str)

        # Replace values based on conditions
        grid_visualization[grid == self.agent_representation] = "A"
        grid_visualization[(grid != 0) & (grid != self.agent_representation)] = "V"

        grid_lanes = ["".join(row) for row in grid_visualization]

        if mode == "ansi":
            print("\n".join(grid_lanes) + f"\nTraffic Light Color: {self.light_color}")
            return

        if mode == "human":
            self._plot_grid(grid_visualization)
            return
