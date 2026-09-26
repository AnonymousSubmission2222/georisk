from .constants import DEFAULT_IMAGE_TOKEN, DEFAULT_STOP_TOKEN, DEFAULT_TRAJ_TOKEN

STOP_REPLY = f"Stop: {DEFAULT_STOP_TOKEN}"
TRAJECTORY_REPLY = f"Future trajectory points: {DEFAULT_TRAJ_TOKEN}"


def observation_prompt(instruction, image_num=2):
    if image_num != 2:
        raise ValueError("GeoRisk requires current front/down RGB observations")
    images = "\n".join([DEFAULT_IMAGE_TOKEN] * image_num)
    return (f"Current multi-view images:\n{images}\n\nInstruction:{instruction}\n\n"
            "Please verify if the target has been reached based on the visual information.")


def trajectory_prompt(stage, delta, cur_pos):
    return (f"Telemetry updated:\nStage:{stage}\nPrevious displacement:{delta}\n"
            f"Current position:{cur_pos}\n\n"
            "Please predict the future trajectory points in 3D coordinates.")
