import pickle
import numpy as np
import zmq

from polaris.client.abstract_client import InferenceClient
from polaris.config import PolicyArgs


@InferenceClient.register(client_name="LeRobotDiffusionZMQ")
class LeRobotDiffusionZMQClient(InferenceClient):
    """
    Client that talks to lerobot_diffusion_server.py over ZMQ.
    The server must be running in the robodiff conda env before eval starts.
    """

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        host = args.host if args.host is not None else "localhost"
        port = args.port if args.port is not None else 5555

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"Connected to policy server at {host}:{port}")
        self._rerender = True

    @property
    def rerender(self) -> bool:
        return self._rerender

    def reset(self):
        self._rerender = True
        self.socket.send(pickle.dumps({"reset": True}))
        self.socket.recv()

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        curr_obs = self._extract_observation(obs)

        request = {
            "cam1":        curr_obs["cam1"],
            "wrist_cam":   curr_obs["wrist_image"],
            "state":       curr_obs["state"],
            "instruction": instruction,
        }
        self.socket.send(pickle.dumps(request))
        response = pickle.loads(self.socket.recv())
        action = response["action"]
        self._rerender = response.get("rerender", True)

        viz = None
        if return_viz:
            import cv2
            ext_small   = cv2.resize(curr_obs["cam1"], (224, 224))
            wrist_small = cv2.resize(curr_obs["wrist_image"], (224, 224))
            viz = np.concatenate([ext_small, wrist_small], axis=1)

        return action, viz

    def _extract_observation(self, obs_dict: dict) -> dict:
        exterior_image = obs_dict["splat"]["cam1"]
        wrist_image    = obs_dict["splat"]["wrist_cam"]

        robot_state = obs_dict["policy"]
        joint_pos   = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()[0]
        gripper_pos = robot_state["gripper_pos"].clone().detach().cpu().numpy()[0]
        state       = np.concatenate([joint_pos, gripper_pos], axis=0)

        return {
            "cam1":        exterior_image,
            "wrist_image": wrist_image,
            "state":       state,
        }
