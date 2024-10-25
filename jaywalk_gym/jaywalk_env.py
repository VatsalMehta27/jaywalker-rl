import gymnasium as gym
from gymnasium import spaces
import numpy as np


class JaywalkEnv(gym.Env):
    def __init__(
        self,
        max_vehicles=5,
        p_vehicle_spawn=0.4,
        p_vehicle_stop=0.8,
        mean_vehicle_speed=1.5,
    ):
        super(JaywalkEnv, self).__init__()

        # Define grid dimensions
        self.grid_shape = (4, 11)
        self.crosswalk_column = self.grid_shape[1] // 2

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
                "agent_position": spaces.Box(
                    low=0, high=3, shape=(2,), dtype=np.int32
                ),  # Row indices: 0-3
                "vehicles": spaces.Box(
                    low=0, high=10, shape=(max_vehicles, 3), dtype=np.int32
                ),
            }
        )

        # Vehicle parameters
        self.max_vehicles = max_vehicles
        self.p_vehicle_spawn = p_vehicle_spawn
        self.p_vehicle_stop = p_vehicle_stop
        self.mean_vehicle_speed = mean_vehicle_speed
        self.vehicles = []

    def reset(self):
        """Resets the environment to the initial state and returns the initial observation."""
        self.agent_position = list(self.agent_start_position)
        self.vehicles = []

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
            raise Exception()

        collision = self._advance_vehicles()

        reward = 0  # Define your reward structure
        done = False  # Define your termination condition

        if collision:
            reward = -10
            done = True
        else:
            self._spawn_vehicle()

            # Define reward structure, terminal conditions, etc.
            if (
                self.agent_position[0] == self.goal_position[0]
                and self.agent_position[1] == self.goal_position[1]
            ):
                reward = 1
                done = True

        # Return observation, reward, done, info
        return self._get_observation(), reward, done, {}

    def _spawn_vehicle(self):
        """Spawns a new vehicle based on the defined probability and rules."""
        if (
            len(self.vehicles) < self.max_vehicles
            and np.random.rand() < self.p_vehicle_spawn
        ):
            # Randomly select a lane (1 or 2) where the first column is unoccupied
            spawn_lanes = [
                lane for lane in range(1, 3) if not self._is_first_column_occupied(lane)
            ]
            if spawn_lanes:
                lane = np.random.choice(spawn_lanes)
                # Sample speed from a normal distribution (mean=2, std=1)
                speed = max(1, round(np.random.normal(self.mean_vehicle_speed, 1)))
                # Create vehicle and append to the vehicles list, starting at the first column
                vehicle = (lane, speed, 0)  # Starting at the first column (0)
                self.vehicles.append(vehicle)

    def _is_first_column_occupied(self, lane):
        """Checks if the first column (column index 0) of the specified lane is occupied by a vehicle."""
        return any(vehicle[0] == lane and vehicle[2] == 0 for vehicle in self.vehicles)

    def _get_front_vehicle(self, lane):
        """
        Retrieves the front vehicle in the specified lane.
        Returns None if no vehicles are in the lane.
        """
        # Filter vehicles in the specified lane and sort them by position
        lane_vehicles = [v for v in self.vehicles if v[0] == lane]

        if lane_vehicles:
            # Sort by position (column index) to find the front vehicle
            front_vehicle = sorted(lane_vehicles, key=lambda v: v[2])[0]
            return front_vehicle  # Return the front vehicle

        return None  # No vehicles in the lane

    def _advance_vehicles(self):
        """
        Advances all vehicles in the grid by their speed, adjusting speeds to avoid collisions.
        """
        # Sort vehicles by their position to ensure we check collisions in the correct order
        self.vehicles.sort(key=lambda v: v[2])  # Sort by position

        updated_vehicles = []
        collision = False

        for i, vehicle in enumerate(self.vehicles):
            lane, speed, position = vehicle

            # Determine the intended new position
            new_position = position + speed

            # Check if the new position would collide with the next vehicle in the lane
            if i < len(self.vehicles) - 1 and self.vehicles[i + 1][0] == lane:
                next_vehicle = self.vehicles[i + 1]
                # If the new position would collide with the next vehicle, stop before it
                if new_position >= next_vehicle[2]:  # Collision detected
                    new_position = next_vehicle[2] - 1  # Stop before the next vehicle
                    speed = next_vehicle[1]

            # Ensure the vehicle does not move beyond the right edge of the grid
            if new_position < self.grid_shape[1]:
                if (
                    lane == self.agent_position[0]
                    and position < self.agent_position[1] <= new_position
                ):
                    # Check the stopping condition
                    stop = np.random.rand()
                    print(stop)
                    if stop < self.p_vehicle_stop:
                        # Stop the vehicle at the square before the agent
                        new_position = self.agent_position[1] - 1
                    else:
                        collision = True

                updated_vehicles.append((lane, speed, new_position))  # Update position

        self.vehicles = updated_vehicles

        return collision

    def _get_observation(self):
        """Constructs the current observation as the agent's position and vehicle list."""
        # Pad vehicle list to ensure consistent shape
        vehicles_array = np.zeros((self.max_vehicles, 3), dtype=np.int32)
        for i, vehicle in enumerate(self.vehicles):
            vehicles_array[i] = vehicle

        return {
            "agent_position": np.array(self.agent_position, dtype=np.int32),
            "vehicles": vehicles_array,
        }

    def render(self, mode="human"):
        """Optional: Render the current state of the environment for debugging."""
        grid = np.full(self.grid_shape, ".", dtype=str)
        grid[self.agent_position[0], self.agent_position[1]] = "A"  # Place agent

        for vehicle in self.vehicles:
            lane, _, position = vehicle
            if position < self.grid_shape[1]:  # Ensure position is within bounds
                grid[lane, position] = "V"  # Place vehicle

        print("\n".join("".join(row) for row in grid))


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
