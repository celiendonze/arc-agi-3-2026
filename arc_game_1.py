from pathlib import Path

import arc_agi
import cv2
import matplotlib.pyplot as plt
import numpy as np
from arcengine import FrameDataRaw, GameAction
from loguru import logger
from matplotlib.axes import Axes
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ollama_model = OpenAIChatModel(
    model_name="gemma4:e2b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)


class GameActionOutput(BaseModel):
    action: int


agent = Agent(model=ollama_model, output_type=GameActionOutput)


STEPS = 20
square_size = int(STEPS**0.5)
if square_size**2 < STEPS:
    square_size += 1
fig, axs = plt.subplots(ncols=square_size, nrows=square_size, figsize=(10, 10))
axs = axs.flatten()


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
    return frame.astype(np.uint8)


def frame_to_bytes(frame_data: FrameDataRaw) -> bytes:
    return cv2.imencode(
        ".png",
        normalize_frame(frame_data.frame[0]),
    )[1].tobytes()


def matplotlib_renderer(steps: int, frame_data: FrameDataRaw) -> None:
    ax: Axes = axs[steps - 1]
    ax.imshow(normalize_frame(frame_data.frame[0]))


def save_frame(frame_data: FrameDataRaw, step: int) -> None:
    image_path = IMAGES_DIR / f"frame_{step}.png"
    cv2.imwrite(str(image_path), normalize_frame(frame_data.frame[0]))


logger.info("Creating environment...")
arc = arc_agi.Arcade()
env = arc.make("ls20", renderer=matplotlib_renderer)
if env is None:
    raise ValueError("Failed to create environment")

agent_actions_history = []
for step in range(STEPS):
    action_space = env.action_space

    frame_data = env.observation_space
    if frame_data is None:
        raise ValueError("Failed to get observation space")
    save_frame(frame_data, step=step)
    binary_frame = frame_to_bytes(frame_data)
    if action_space is None or frame_data is None:
        raise ValueError("Failed to get action space or observation space")

    logger.info("Agent is deciding on the next action...")
    result = agent.run_sync(
        user_prompt=[
            "Given the current frame, decide on the next action to take in the environment.",
            f"The action space is: {action_space}",
            f"Here is your last action history: {agent_actions_history}",
            BinaryContent(
                data=frame_to_bytes(frame_data),
                media_type="image/png",
            ),
        ],
    )

    action = result.output
    agent_actions_history.append(action)
    action_taken = action_space[action.action - 1]
    logger.info(f"Agent decided to take action: {action_taken}")

    env.step(action_taken)


for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()
