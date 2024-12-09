from dataclasses import dataclass, field
from itertools import cycle
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np


@dataclass
class EnvParams:
    max_vehicles: int = 10
    p_vehicle_spawn: float = 0.4
    p_vehicle_stop: float = 0.8
    mean_vehicle_speed: float = 1.5
    use_traffic_light: bool = True
    num_consecutive_roads: int = 2
    num_lane_groups: int = 2
    num_columns: int = 11
    light_durations: dict[str, int] = field(
        default_factory=lambda: {"GREEN": 10, "YELLOW": 3, "RED": 5}
    )
    max_reward: int = 100
    death_reward: int = -500
    wait_reward: int = -2


class JaywalkEnv(gym.Env):
    def __init__(self, params: EnvParams):
        super(JaywalkEnv, self).__init__()

        self.num_consecutive_roads = params.num_consecutive_roads
        num_rows = (
            params.num_lane_groups
            + 1
            + self.num_consecutive_roads * params.num_lane_groups
        )
        self.grid_shape = (num_rows, params.num_columns)
        self.crosswalk_column = self.grid_shape[1] // 2

        self.agent_start_position = (0, self.crosswalk_column)
        self.goal_position = (self.grid_shape[0] - 1, self.crosswalk_column)

        self.agent_position = list(self.agent_start_position)
        self.agent_representation = 100

        self.actions = {0: "backward", 1: "wait", 2: "forward"}

        # Action space: using discrete integers
        self.action_space = spaces.Discrete(len(self.actions), start=0)
        self.observation_space = spaces.Dict(
            {
                "traffic_light": spaces.Box(
                    low=0,
                    high=len(params.light_durations),
                    shape=(len(params.light_durations),),
                    dtype=np.int32,
                ),  # 0: RED, 1: YELLOW, 2: GREEN
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
        self.p_vehicle_stop = params.p_vehicle_stop
        self.mean_vehicle_speed = params.mean_vehicle_speed

        self.lanes = self._initialize_lanes()

        self.use_traffic_light = params.use_traffic_light
        self.light_durations = params.light_durations
        self.num_lights = len(self.light_durations)
        self.light_cycle = cycle(self.light_durations.keys())
        self.light_color = next(self.light_cycle)
        self.light_index = 0
        self.light_timer = 0

        self.max_reward = params.max_reward
        self.death_reward = params.death_reward
        self.wait_reward = params.wait_reward

        self.time_steps = 0

        self.received_bonus = {
            i: False for i in range(1, self.grid_shape[0] - 1) if i % 3 == 0
        }

    def reset(self, seed=None):
        # Reset agent position
        self.agent_position = list(self.agent_start_position)

        # Reset all vehicles
        self.num_vehicles = 0
        self.lanes = self._initialize_lanes()

        # Reset traffic light
        self.light_cycle = cycle(self.light_durations.keys())
        self.light_color = next(self.light_cycle)
        self.light_index = 0
        self.light_timer = 0

        self.time_steps = 0

        self.received_bonus = {
            i: False for i in range(1, self.grid_shape[0] - 1) if i % 3 == 0
        }

        for _ in range(10):
            self._environment_movement()

        return self._get_observation(), {}

    def _initialize_lanes(self):
        return {
            i: {"direction": -1 if (i // 3) % 2 != 0 else 1, "vehicles": []}
            for i in range(self.grid_shape[0])
            if i % 3 != 0
        }

    def step(self, action):
        match action:
            case 2:
                self.agent_position[0] = min(
                    self.agent_position[0] + 1, self.grid_shape[0]
                )
            case 0:
                self.agent_position[0] = max(self.agent_position[0] - 1, 0)
            case 1:
                pass
            case _:
                print(action)
                raise Exception("Invalid action.")

        reward, done = self._environment_movement()

        # if not self.received_bonus.get(self.agent_position[0]):
        #     reward += 1
        #     self.received_bonus[self.agent_position[0]] = True

        self.time_steps += 1

        return self._get_observation(), reward, done, False, {}

    def _environment_movement(self):
        pedestrian_collision = self._advance_vehicles()

        reward = self.wait_reward
        done = False

        if pedestrian_collision:
            reward = self.death_reward
            done = True
        else:
            self._spawn_vehicle()
            self._update_traffic_light()

            # Reward for reaching goal position
            if (
                self.agent_position[0] == self.goal_position[0]
                and self.agent_position[1] == self.goal_position[1]
            ):
                reward = self.max_reward
                done = True

        return reward, done

    def _advance_vehicles(self):
        pedestrian_collision = False
        num_vehicles = 0

        for lane_idx, lane_info in self.lanes.items():
            lane = lane_info["vehicles"]
            direction = lane_info["direction"]
            sorted_lane = sorted(lane, key=lambda v: -1 * direction * v[1])

            for i, vehicle in enumerate(sorted_lane):
                speed, position = vehicle

                new_speed = speed
                new_position = position + speed

                # Check for vehicle collision
                # TODO: fix red light skipping for direction = 1
                if (
                    i > 0
                    and direction * sorted_lane[i - 1][1] <= direction * new_position
                ):
                    new_position = sorted_lane[i - 1][1] - 1 * direction
                    new_speed = sorted_lane[i - 1][0]
                elif (position < self.crosswalk_column <= new_position) or (
                    new_position <= self.crosswalk_column < position
                ):  # Crossing the crosswalk
                    match self.light_color:
                        case "RED":
                            new_position = self.crosswalk_column - 1 * direction
                        case "YELLOW":
                            vehicle_stop = np.random.rand()

                            # TODO: Update this with some yellow light based prob of stopping?
                            if vehicle_stop < self.p_vehicle_stop:
                                new_position = self.crosswalk_column - 1 * direction
                        case "GREEN":
                            if lane_idx == self.agent_position[0]:
                                vehicle_stop = np.random.rand()

                                if vehicle_stop < self.p_vehicle_stop:
                                    new_position = self.crosswalk_column - 1 * direction
                                else:
                                    pedestrian_collision = True

                sorted_lane[i] = (new_speed, new_position)

            filtered_lane = list(
                filter(lambda v: 0 <= v[1] < self.grid_shape[1], sorted_lane)
            )

            self.lanes[lane_idx]["vehicles"] = filtered_lane
            num_vehicles += len(filtered_lane)

        self.num_vehicles = num_vehicles

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

            if spawn_lanes:
                # Select a random unoccupied lane index
                lane_idx = np.random.choice(spawn_lanes)
                direction = self.lanes[lane_idx]["direction"]
                speed = (
                    max(1, round(np.random.normal(self.mean_vehicle_speed, 1)))
                    * direction
                )

                if direction == 1:
                    start_pos = 0
                else:
                    start_pos = self.grid_shape[1] - 1

                vehicle = (speed, start_pos)  # Vehicle starts at column 0 in its lane

                self.lanes[lane_idx]["vehicles"].append(vehicle)
                self.num_vehicles += 1

    def _is_first_column_occupied(self, lane_idx):
        """Checks if the first column (column index 0) of the specified lane is occupied by a vehicle."""
        return any(vehicle[1] == 0 for vehicle in self.lanes[lane_idx]["vehicles"])

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

        for lane_idx, lane_info in self.lanes.items():
            for velocity, position in lane_info["vehicles"]:
                grid[lane_idx, position] = velocity

        light_one_hot = np.zeros(self.num_lights)
        light_one_hot[self.light_index] = 1

        return {"traffic_light": light_one_hot, "world_grid": grid}  # grid

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
                elif char == "X":
                    color_array[i, j] = 3
                else:
                    color_array[i, j] = 0

        # Set up the figure with two subplots: one for the grid and one for the traffic light
        fig, (ax, traffic_light_ax) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [4, 1]}, figsize=(10, 6)
        )

        # Plot the grid on the left subplot
        cmap = plt.cm.colors.ListedColormap(["white", "red", "green", "grey"])
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
        for idx, light in enumerate(list(self.light_durations.keys())[::-1]):
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

        # Initialize a grid visualization array with '.' for empty cells
        grid_visualization = np.full(grid.shape, ".", dtype=str)

        # Mark non-lane rows with 'X'
        for row in range(grid.shape[0]):
            if row not in self.lanes.keys():
                grid_visualization[row, :] = "X"

        # Mark the agent's position with 'A'
        grid_visualization[grid == self.agent_representation] = "A"

        # Mark vehicles with 'V'
        grid_visualization[(grid != 0) & (grid != self.agent_representation)] = "V"

        grid_lanes = ["".join(row) for row in grid_visualization]

        if mode == "ansi":
            print("\n".join(grid_lanes) + f"\nTraffic Light Color: {self.light_color}")
            return

        if mode == "human":
            self._plot_grid(grid_visualization)
            return
