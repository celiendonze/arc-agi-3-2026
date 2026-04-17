from pathlib import Path

import arc_agi
import cv2
import matplotlib.pyplot as plt
import numpy as np
from arcengine import FrameDataRaw, GameAction
from loguru import logger
from matplotlib.axes import Axes
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent, RunContext, ToolReturn, UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
IMAGES_DIR = DATA_DIR / "images"
THINKING_DIR = DATA_DIR / "thinking"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
THINKING_DIR.mkdir(parents=True, exist_ok=True)

# clean image_dir
for file in IMAGES_DIR.glob("*.png"):
    file.unlink()

ollama_model = OpenAIChatModel(
    # model_name="qwen3.6:latest",
    # model_name="gemma4:e2b",
    model_name="gemma4:e4b",
    # model_name="gemma4:31b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        min_val, max_val = float(frame.min()), float(frame.max())
        if max_val == min_val:
            return np.zeros_like(frame, dtype=np.uint8)
        frame = (frame - min_val) / (max_val - min_val) * 255
    return frame.astype(np.uint8)


def frame_to_bytes(frame_data: FrameDataRaw) -> bytes:
    return cv2.imencode(".png", normalize_frame(frame_data.frame[0]))[1].tobytes()


def save_frame(frame_data: FrameDataRaw, step: int) -> None:
    path = IMAGES_DIR / f"frame_{step}.png"
    cv2.imwrite(str(path), normalize_frame(frame_data.frame[0]))


def render_frame_as_text(frame_data: FrameDataRaw) -> str:
    frame = frame_data.frame[0]
    ascii_art = ""
    for row in frame:
        ascii_art += "".join(str(pixel) for pixel in row) + "\n"
    return ascii_art


logger.info("Creating environment...")

arc = arc_agi.Arcade()
env = arc.make("ls20")
if env is None:
    raise ValueError("Failed to create environment")
env.reset()

initial_frame = env.observation_space
if initial_frame is None:
    raise ValueError("Failed to get observation space")
save_frame(initial_frame, step=0)  # Save the initial frame for visualization

agent = Agent(
    model=ollama_model,
    instructions="\n".join(
        [
            "Your task is to interact with the game environment by taking actions and observing the resulting frames.",
            "Use the provided tools to list available actions, render the current frame as ASCII art, and take steps in the environment.",
            # "Your goal is to explore the environment and find a way to win the game.",
            # "Your goal is also to find the rules of the game",
            # "After taking an action, analyze the resulting frame to inform your next decision.",
            # "Use the tools multiple times as needed to gather information and progress in the game.",
            # "Do not stop before experimenting with at least 100 different actions and analyzing the resulting frames.",
        ]
    ),
)


@agent.tool_plain
def list_available_actions() -> str:
    """Tool to list available actions in the environment."""
    logger.info("Agent is listing available actions...")
    if env is None:
        return "Environment is not initialized"
    return "\n".join(f"{i}: {action}" for i, action in enumerate(env.action_space))


@agent.tool_plain
def render_frame() -> ToolReturn:
    """Tool to render the current frame as ASCII art."""
    logger.info("Agent is rendering the current frame...")
    if env is None:
        return ToolReturn(return_value="Environment is not initialized")
    frame_data = env.observation_space
    if frame_data is None:
        return ToolReturn(return_value="Failed to get observation space")
    # ascii_art = render_frame_as_text(frame_data)
    image = frame_to_bytes(frame_data)
    return ToolReturn(
        return_value="Current frame rendered as image",
        content=[BinaryContent(data=image, media_type="image/png")],
    )


current_step = 1


@agent.tool_plain
def step(action: int) -> ToolReturn:
    """Tool to take a step in the environment with the given action.

    Args:
        action (int): The index of the action to take from the action space.
    """
    logger.info(f"Agent is taking action: {action}")
    if env is None:
        return ToolReturn(return_value="Environment is not initialized")
    game_action = env.action_space[action]
    new_frame_data = env.step(game_action)
    if new_frame_data is None:
        return ToolReturn(return_value="Failed to step in the environment")

    global current_step
    save_frame(
        new_frame_data, step=current_step
    )  # Save the new frame for visualization
    current_step += 1
    # ascii_art = render_frame_as_text(new_frame_data)
    image_bytes = frame_to_bytes(new_frame_data)
    return ToolReturn(
        return_value="Resulting frame after taking action",
        content=[BinaryContent(data=image_bytes, media_type="image/png")],
    )


app = agent.to_web()
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
