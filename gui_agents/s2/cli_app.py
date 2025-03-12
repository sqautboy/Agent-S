import argparse
import datetime
import io
import logging
import os
import platform
import pyautogui
import sys
import time

from PIL import Image

if platform.system() == "Darwin":
    current_platform = "macos"
elif platform.system() == "Linux":
    current_platform = "ubuntu"
elif platform.system() == "Windows":
    current_platform = "windows"
else:
    raise ValueError("Unsupported platform")

from gui_agents.s2.agents.grounding import OSWorldACI
from gui_agents.s2.agents.agent_s import GraphSearchAgent

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)

platform_os = platform.system()


def show_permission_dialog(code: str, action_description: str):
    """Show a platform-specific permission dialog and return True if approved."""
    if platform.system() == "Darwin":
        result = os.system(
            f'osascript -e \'display dialog "Do you want to execute this action?\n\n{code} which will try to {action_description}" with title "Action Permission" buttons {{"Cancel", "OK"}} default button "OK" cancel button "Cancel"\''
        )
        return result == 0
    elif platform.system() == "Linux":
        result = os.system(
            f'zenity --question --title="Action Permission" --text="Do you want to execute this action?\n\n{code}" --width=400 --height=200'
        )
        return result == 0
    return False


def scale_screen_dimensions(width: int, height: int):
    MAX_DIMENSION_SIZE = 2400
    scale_factor = min(MAX_DIMENSION_SIZE / width, MAX_DIMENSION_SIZE / height, 1)
    safe_width = int(width * scale_factor)
    safe_height = int(height * scale_factor)
    return safe_width, safe_height


def run_agent(agent, instruction: str, safe_width: int, safe_height: int):
    obs = {}
    traj = "Task:\n" + instruction
    subtask_traj = ""
    for _ in range(15):
        # Get screen shot using pyautogui.
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((safe_width, safe_height), Image.LANCZOS)

        # Save the screenshot to a BytesIO object
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")

        # Get the byte value of the screenshot
        screenshot_bytes = buffered.getvalue()
        # Convert to base64 string.
        obs["screenshot"] = screenshot_bytes

        # Get next action code from the agent
        info, code = agent.predict(instruction=instruction, observation=obs)

        if "done" in code[0].lower() or "fail" in code[0].lower():
            if platform.system() == "Darwin":
                os.system(
                    f'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                )
            elif platform.system() == "Linux":
                os.system(
                    f'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                )

            agent.update_narrative_memory(traj)
            break

        if "next" in code[0].lower():
            continue

        if "wait" in code[0].lower():
            time.sleep(5)
            continue

        else:
            time.sleep(1.0)
            print("EXECUTING CODE:", code[0])

            # Ask for permission before executing
            exec(code[0])
            time.sleep(1.0)

            # Update task and subtask trajectories and optionally the episodic memory
            traj += (
                "\n\nReflection:\n"
                + str(info["reflection"])
                + "\n\n----------------------\n\nPlan:\n"
                + info["executor_plan"]
            )
            subtask_traj = agent.update_episodic_memory(info, subtask_traj)


def main():
    parser = argparse.ArgumentParser(
        description="Run GraphSearchAgent with specified model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Specify the model to use (e.g., gpt-4o)",
    )

    # Grounding model config option 1: API based
    parser.add_argument(
        "--grounding_model",
        type=str,
        default="",
        help="Specify the grounding model to use (e.g., claude-3-7-sonnet)",
    )

    # Grounding model config option 2: Self-hosted endpoint based
    parser.add_argument(
        "--endpoint_provider",
        type=str,
        default="huggingface",
        help="Specify the endpoint provider for your grounding model, only HuggingFace TGI support for now",
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        default="",
        help="Specify the endpoint URL for your grounding model",
    )

    args = parser.parse_args()
    assert (
        args.grounding_model or args.endpoint_url
    ), "Error: No grounding model was provided. Either provide an API based model, or a self-hosted HuggingFace endpoint"

    # Load the general engine params
    if args.model.startswith("claude"):
        engine_params = {"engine_type": "anthropic", "model": args.model}
    elif args.model.startswith("gpt"):
        engine_params = {"engine_type": "openai", "model": args.model}
    else:
        raise ValueError(
            "Invalid model specficiation. Please provide a supported model type"
        )

    # Load the grounding model engine params
    if args.endpoint_url:
        engine_params_for_grounding = {
            "engine_type": args.endpoint_provider,
            "endpoint_url": args.endpoint_url,
        }
    elif args.grounding_model.startswith("claude"):
        engine_params_for_grounding = {
            "engine_type": "anthropic",
            "model": args.grounding_model,
        }
    elif args.grounding_model.startswith("gpt"):
        engine_params_for_grounding = {
            "engine_type": "openai",
            "model": args.grounding_model,
        }
    else:
        raise ValueError(
            "Invalid grounding model specficiation. Please provide a supported model type"
        )

    # Re-scales screenshot size to ensure it fits in UI-TARS context limit
    screen_width, screen_height = pyautogui.size()
    safe_width, safe_height = scale_screen_dimensions(screen_width, screen_height)

    grounding_agent = OSWorldACI(
        platform=current_platform,
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=safe_width,
        height=safe_height,
    )

    agent = GraphSearchAgent(
        engine_params,
        grounding_agent,
        platform=current_platform,
        action_space="pyautogui",
        observation_type="mixed",
        search_engine=None,
        # search_engine="Perplexica",
    )

    while True:
        query = input("Query: ")

        agent.reset()

        # Run the agent on your own device
        run_agent(agent, query, safe_width, safe_height)

        response = input("Would you like to provide another query? (y/n): ")
        if response.lower() != "y":
            break


if __name__ == "__main__":
    main()
