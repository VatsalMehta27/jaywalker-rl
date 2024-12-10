from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

from gymnasium import Env
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2


@dataclass
class TrainingResult:
    returns: np.ndarray[int]
    timesteps: np.ndarray[int]
    loss: np.ndarray[int]


class Agent(ABC):
    def __init__(self, env: Env, params: dict):
        self.env = env
        self.params = params

        self.action_dim = params["action_dim"]
        self.state_dim = params["state_dim"]

    @staticmethod
    def argmax(values: list[int]) -> int:
        """Argmax that breaks ties randomly"""
        values = np.asarray(values)
        max_indices = np.flatnonzero(values == values.max())

        return np.random.choice(max_indices)

    @staticmethod
    def transform_state(state: dict) -> np.ndarray:
        combined_state = np.concat(
            [state["traffic_light"], state["world_grid"].flatten()]
        )

        return combined_state

    @abstractmethod
    def get_action(self, state: np.ndarray) -> int:
        pass

    @abstractmethod
    def get_action_probs(self, state: np.ndarray) -> list[float]:
        pass

    @abstractmethod
    def get_greedy_action(self, state: np.ndarray) -> int:
        pass

    @abstractmethod
    def train(self, episodes) -> TrainingResult:
        pass

    def eval(
        self,
        render_mode: Literal["ascii", "human"] = "human",
        video_filename: Optional[str] = "policy_rollout.mp4",
        timeout: int = 250,
    ):
        matplotlib_backend = matplotlib.get_backend()

        state, _ = self.env.reset()
        done = False

        save_video = render_mode == "human" and video_filename
        frames = []

        if save_video:
            # Temporarily switch to a non-interactive backend to prevent plots from showing
            matplotlib.use("Agg")

        while not done and self.env.time_steps < timeout:
            self.env.render(mode=render_mode)
            action = self.get_greedy_action(state)  # ["world_grid"])
            state, reward, done, _, _ = self.env.step(action)

            if save_video:
                frames.append(self._capture_frame())

        self.env.render(mode=render_mode)

        if save_video:
            # Capture the current plot rendered by the environment
            frames.append(self._capture_frame())
            self._save_video(frames, video_filename)
            # Restore the original backend
            matplotlib.use(matplotlib_backend)

    def _capture_frame(self) -> np.ndarray:
        # Capture the current plot rendered by the environment
        fig = plt.gcf()
        fig.canvas.draw()
        # Save the current figure into a buffer
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)

        plt.close(fig)  # Close the figure to free memory

        return img

    def _save_video(self, frames: list[np.ndarray], filename: str) -> None:
        """Save the frames as a video using OpenCV"""
        if not frames:
            print("No frames to save!")
            return

        frames = [
            cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2BGR)
            for frame in frames
        ]

        # Get the frame size from the first frame
        height, width, _ = frames[0].shape

        for frame in frames:
            assert frame.shape == (height, width, 3), "Inconsistent frame size"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for mp4 format
        video_writer = cv2.VideoWriter(filename, fourcc, 3, (width, height))

        # Write each frame to the video file
        for frame in frames:
            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved as {filename}!")

    @abstractmethod
    def save(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        pass

    @staticmethod
    def _moving_average(data: np.ndarray, window_size: int = 50):
        """Smooths 1-D data array using a moving average.

        Args:
            data: 1-D numpy.array
            window_size: Size of the smoothing window

        Returns:
            smooth_data: A 1-d numpy.array with the same size as data
        """
        if len(data) == 0:
            # Return an empty array if the input data is empty
            return np.array([])

        assert data.ndim == 1

        kernel = np.ones(window_size)
        smooth_data = np.convolve(data, kernel) / np.convolve(
            np.ones_like(data), kernel
        )

        return smooth_data[: -window_size + 1]

    @staticmethod
    def plot_training_result(training_result: TrainingResult) -> None:
        smooth_returns = Agent._moving_average(training_result.returns)
        smooth_timesteps = Agent._moving_average(training_result.timesteps)
        smooth_loss = Agent._moving_average(training_result.loss)

        # Create a figure with 3 subplots: one for returns, one for timesteps, and one for loss
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # Plot returns in the first subplot
        axs[0].plot(training_result.returns, label="Returns")
        axs[0].plot(smooth_returns, label="Smoothed Returns", linestyle="--")
        axs[0].set_title("Training Returns")
        axs[0].set_xlabel("Episodes")
        axs[0].set_ylabel("Return")
        axs[0].legend()

        # Plot timesteps in the second subplot
        axs[1].plot(training_result.timesteps, label="Timesteps")
        axs[1].plot(smooth_timesteps, label="Smoothed Timesteps", linestyle="--")
        axs[1].set_title("Timesteps")
        axs[1].set_xlabel("Episodes")
        axs[1].set_ylabel("Timesteps")
        axs[1].legend()

        # Plot loss in the third subplot
        axs[2].plot(training_result.loss, label="Loss")
        axs[2].plot(smooth_loss, label="Smoothed Loss", linestyle="--")
        axs[2].set_title("Loss")
        axs[2].set_xlabel("Episodes")
        axs[2].set_ylabel("Loss")
        axs[2].legend()

        # Adjust layout for better spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()

    @staticmethod
    def plot_multiple_training_result(
        training_results: list[TrainingResult], hyperparams: list[str]
    ) -> None:
        assert len(training_results) == len(hyperparams)

        # Define a custom colormap using several base colors
        colors = [
            "#264653",  # Deep teal
            "#2a9d8f",  # Emerald green
            "#e9c46a",  # Golden yellow
            "#f4a261",  # Soft orange
            "#e76f51",  # Terracotta red
            "#a855f7",  # Vibrant purple
            "#1d3557",  # Dark blue
            "#457b9d",  # Steel blue
            "#81b29a",  # Sage green
            "#ef476f",  # Watermelon pink
            "#118ab2",  # Cerulean
            "#073b4c",  # Midnight blue
            "#06d6a0",  # Aquamarine
            "#ffd166",  # Mustard yellow
            "#e63946",  # Crimson red
            "#8338ec",  # Electric purple
            "#3a86ff",  # Sky blue
            "#ff006e",  # Magenta
            "#8d99ae",  # Dusty lavender
            "#ffbe0b",  # Bright sunflower
            "#6a4c93",  # Royal purple
            "#00a676",  # Tropical green
            "#d62828",  # Scarlet red
        ]

        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=30)

        # Create a figure with 3 subplots: one for returns, one for timesteps, and one for loss
        fig, axis = plt.subplots(3, 1, figsize=(10, 12))

        # Plot returns in the first subplot for each hyperparameter
        for i in range(len(training_results)):
            smooth_returns = Agent._moving_average(training_results[i].returns)
            axis[0].plot(
                smooth_returns,
                color=custom_cmap(i / len(training_results)),  # Use the custom colormap
                label=f"Smoothed Returns: {hyperparams[i]}",
                linestyle="--",
            )

        axis[0].set_title("Training Returns")
        axis[0].set_xlabel("Episodes")
        axis[0].set_ylabel("Return")

        # Plot timesteps in the second subplot for each hyperparameter
        for i in range(len(training_results)):
            smooth_timesteps = Agent._moving_average(training_results[i].timesteps)
            axis[1].plot(
                smooth_timesteps,
                color=custom_cmap(i / len(training_results)),  # Use the custom colormap
                label=f"Smoothed Timesteps: {hyperparams[i]}",
                linestyle="--",
            )

        axis[1].set_title("Timesteps")
        axis[1].set_xlabel("Episodes")
        axis[1].set_ylabel("Timesteps")

        # Plot loss in the third subplot for each hyperparameter
        for i in range(len(training_results)):
            smooth_loss = Agent._moving_average(training_results[i].loss)
            axis[2].plot(
                smooth_loss,
                color=custom_cmap(i / len(training_results)),  # Use the custom colormap
                label=f"Smoothed Loss: {hyperparams[i]}",
                linestyle="--",
            )

        axis[2].set_title("Loss")
        axis[2].set_xlabel("Episodes")
        axis[2].set_ylabel("Loss")

        # Add a shared legend for all subplots
        handles, labels = axis[2].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=(1.45, 1),
            ncol=1,
        )

        # Adjust layout for better spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()
