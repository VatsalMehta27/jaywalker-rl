from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

from gymnasium import Env
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import cv2


@dataclass
class TrainingResult:
    returns: list[int]
    timesteps: list[int]
    loss: list[int]


class Agent(ABC):
    def __init__(self, env: Env, params: dict):
        self.env = env
        self.params = params

    @staticmethod
    def argmax(values: list[int]) -> int:
        """Argmax that breaks ties randomly"""
        values = np.asarray(values)
        max_indices = np.flatnonzero(values == values.max())

        return np.random.choice(max_indices)

    @abstractmethod
    def get_action(self, state: np.ndarray) -> int:
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
    ):
        matplotlib_backend = matplotlib.get_backend()

        state, _ = self.env.reset()
        done = False

        save_video = render_mode == "human" and video_filename
        frames = []

        if save_video:
            # Temporarily switch to a non-interactive backend to prevent plots from showing
            matplotlib.use("Agg")

        while not done:
            self.env.render(mode=render_mode)
            action = self.get_greedy_action(state["world_grid"])
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

        for i, frame in enumerate(frames):
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
