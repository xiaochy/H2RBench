import torch
import cv2
from pathlib import Path
import numpy as np

from isaaclab.sensors.camera.camera import Camera
import isaaclab.utils.math as math
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaacsim.core.prims import GeometryPrim
from isaacsim.core.utils.stage import get_current_stage
from pxr import Semantics

from polaris.splat_renderer import SplatRenderer3DGS, SplatRenderer
from polaris.environments.rubrics import Rubric
from scipy.spatial.transform import Rotation

ISAAC_TO_CUROBO_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 12]

class ManagerBasedRLSplatEnv(ManagerBasedRLEnv):
    rubric: Rubric | None = None
    _task_name: str | None = None

    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        *args,
        rubric: Rubric | None = None,
        usd_file: str | None = None,
        **kwargs,
    ):
        # do dynamic setup here maybe
        if usd_file is not None:
            self.usd_file = usd_file
            cfg.dynamic_setup(usd_file)

        super().__init__(cfg=cfg, *args, **kwargs)
        self.setup_splat_world_and_robot_views()
        self.setup_splat_robot()
        self.rubric = rubric
        self._setup_grasp_frame_offset()

    def _evaluate_rubric(self) -> dict:
        """Evaluate rubric and return results for info dict."""
        if self.rubric is None:
            return {
                "rubric": {
                    "success": False,
                    "progress": -1.0,
                    "metrics": {},
                }
            }

        result = self.rubric.evaluate(self)
        return {
            "rubric": {
                "success": result.success,
                "progress": result.progress,
                "metrics": result.metrics,
            }
        }

    def reset(self, object_positions: dict = {}, expensive=True, *args, **kwargs):
        """
        Reset the environment

        Parameters
        ----------
        object_positions : dict
            A dictionary mapping object names to their desired poses (position and orientation).
        expensive : bool
            Whether to perform expensive (splat) rendering operations.
        """
        obs, info = super().reset(*args, **kwargs)

        # Reset rubric state
        if self.rubric:
            self.rubric.reset()

        # Following predefined initial conditions
        for obj, pose in object_positions.items():
            print(f"Setting initial condition for {obj} to {pose}")
            pose = torch.tensor(pose)[None]
            self.scene[obj].write_root_pose_to_sim(pose)
        self.sim.render()
        self.scene.update(0)
        obs = (
            self.observation_manager.compute()
        )  # update observation after setting ICs if needed
        obs["splat"] = self.custom_render(expensive, transform_static=True)
        # obs["object_pcd"] = self.get_object_pcds_from_mesh(4500)
        obs["joint_pos"] = self.scene["robot"].data.joint_pos.squeeze(0)[ISAAC_TO_CUROBO_IDX]

        # Evaluate rubric and add to info
        info.update(self._evaluate_rubric())

        return obs, info
    
    def get_object_pcds_from_mesh(self, n_points: int = 4500) -> np.ndarray:
        from pxr import UsdGeom
        from scipy.spatial.transform import Rotation

        stage = get_current_stage()
        all_vertices = []

        def get_all_vertices(p):
            verts = []
            m = UsdGeom.Mesh(p)
            if m:
                pts = m.GetPointsAttr().Get()
                if pts:
                    verts.extend(pts)
            for child in p.GetAllChildren():
                verts.extend(get_all_vertices(child))
            return verts

        for obj_name in self.scene.rigid_objects:
            if "static" in obj_name:
                continue

            # current world pose from Isaac
            pos  = self.scene[obj_name].data.root_state_w[0, :3].detach().cpu().numpy()
            quat = self.scene[obj_name].data.root_state_w[0, 3:7].detach().cpu().numpy()  # wxyz
            R    = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

            mesh_path = f"/World/envs/env_0/scene/{obj_name}"
            prim      = stage.GetPrimAtPath(mesh_path)

            if not prim.IsValid():
                continue

            # Read scale from USD xformOp
            xformable = UsdGeom.Xformable(prim)
            scale = np.array([1.0, 1.0, 1.0])
            for op in xformable.GetOrderedXformOps():
                if "scale" in op.GetName():
                    s = op.Get()
                    scale = np.array([s[0], s[1], s[2]], dtype=np.float32)
                    break

            vertices = get_all_vertices(prim)

            if not vertices:
                print(f"  [warn] no vertices found for {obj_name}")
                continue

            vertices = np.array(vertices, dtype=np.float32)  # (N, 3)

            vertices = vertices * 0.01 * scale

            # TODO: I didn't know why the rotation_90 is needed
            R_correction = Rotation.from_euler('x', 90, degrees=True).as_matrix()
            vertices = (R_correction @ vertices.T).T

            points_world = (R @ vertices.T).T + pos
            all_vertices.append(points_world)

        if not all_vertices:
            return np.zeros((n_points, 3), dtype=np.float32)

        all_vertices = np.concatenate(all_vertices, axis=0)  # (N_total, 3)

        # Sample n_points from combined pool
        if len(all_vertices) >= n_points:
            idx = np.random.choice(len(all_vertices), n_points, replace=False)
        else:
            idx = np.random.choice(len(all_vertices), n_points, replace=True)

        return all_vertices[idx].astype(np.float32)  # (n_points, 3)

    def _setup_grasp_frame_offset(self):
        yaml_wxyz = np.array([0.0, 0.7071067811865476, 0.0, 0.7071067811865476])  # wxyz
        yaml_xyzw = yaml_wxyz[[1, 2, 3, 0]]  # → xyzw for scipy

        self._grasp_offset_pos = np.array([0.15335, 0.0, 0.0], dtype=np.float32)
        self._grasp_offset_rot = Rotation.from_quat(yaml_xyzw)

        self._base_link_idx = self.scene["robot"].find_bodies("base_link")[0][0]    

    def step(self, action, expensive=True):
        """
        Steps the environment

        Parameters
        ----------
        action: torch.Tensor
            The action to take in the environment.
        expensive : bool
            Whether to perform expensive (splat) rendering operations.
        """
        obs, rew, done, trunc, info = super().step(action)
        obs["splat"] = self.custom_render(expensive)
        # obs["object_pcd"] = self.get_object_pcds_from_mesh(4500)
        obs["joint_pos"] = self.scene["robot"].data.joint_pos.squeeze(0)[ISAAC_TO_CUROBO_IDX]

        # Evaluate rubric and add to info
        info.update(self._evaluate_rubric())

        return obs, rew, done, trunc, info

    def custom_render(self, expensive: bool, transform_static: bool = False):
        """
        Render the environment
        """
        if expensive:
            self.transform_sim_to_splat(transform_static=transform_static)
            rgb = self.render_splat()
            mask_and_rgb = self.get_robot_from_sim()
            for cam in mask_and_rgb:
                og_img = (
                    rgb[cam] if cam in rgb else np.zeros_like(mask_and_rgb[cam]["rgb"])
                )
                mask = mask_and_rgb[cam]["mask"]
                sim_img = mask_and_rgb[cam]["rgb"]

                if "wrist" in cam:
                    rgb[cam] = sim_img
                else:
                    new_img = np.where(mask, sim_img, og_img)
                    rgb[cam] = new_img
                rgb[f"{cam}_depth"] = mask_and_rgb[cam]["depth"]
        else:
            rgb = {}
            for cam in self.scene.sensors:
                if isinstance(self.scene.sensors[cam], Camera):
                    rgb[cam] = (
                        self.scene[cam].data.output["rgb"][0].detach().cpu().numpy()
                    )
                    rgb[f"{cam}_depth"] = self.scene[cam].data.output["distance_to_image_plane"][0].detach().cpu().numpy()
        return rgb

    def setup_splat_world_and_robot_views(self):
        splats = {}
        self.views = {}
        stage = get_current_stage()

        # Allocate splats for all rigid objects in the scene and raytrace semantic tags
        for name in self.scene.rigid_objects:
            path = Path(self.usd_file).parent / "assets" / name / "splat.ply"
            if path.exists():
                splats[name] = path
            else:
                # apply semantic tags
                prim = stage.GetPrimAtPath(f"/World/envs/env_0/scene/{name}")
                semantic_type = "class"
                semantic_value = "raytraced"
                instance_name = f"{semantic_type}_{semantic_value}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)

        # Setup splat cameras with intrinsics and resolution from sim cameras
        camera_cfg = {}
        for name in self.scene.sensors:
            if not isinstance(self.scene.sensors[name], Camera):
                continue
            resolution = self.scene.sensors[name].image_shape
            h_aperture = (
                self.scene[name]._sensor_prims[0].GetHorizontalApertureAttr().Get()
            )
            v_aperture = (
                self.scene[name]._sensor_prims[0].GetVerticalApertureAttr().Get()
            )
            f = self.scene[name]._sensor_prims[0].GetFocalLengthAttr().Get()
            fovx = 2 * np.arctan(h_aperture / (2 * f))
            fovy = 2 * np.arctan(v_aperture / (2 * f))
            camera_cfg[name] = {
                "res": resolution,
                "fovx": fovx,
                "fovy": fovy,
            }
        self.splat_renderer = SplatRenderer3DGS(splats=splats, device=self.device)
        self.splat_renderer.init_cameras(camera_cfg)

    def setup_splat_robot(self):
        # Allocate robot splats and views on robot links to track
        more_splats = {}
        robot_asset_path = Path(self.cfg.scene.robot.spawn.usd_path).parent
        for ply in sorted(list(robot_asset_path.glob("SEGMENTED/*.ply"))):
            more_splats[ply.stem] = ply
            sim_path = ply.stem.replace("-", "/")
            view = GeometryPrim(
                prim_paths_expr=f"/World/envs/env_0/robot/{sim_path}",
                reset_xform_properties=False,
            )
            print(f"/World/envs/env_0/robot/{sim_path}")
            self.views[ply.stem] = view
        self.splat_renderer.add_splats(more_splats)

    def get_robot_from_sim(self):
        # TODO: comment this. does this get only robot? objects too?
        ret = {}
        for cam in self.scene.sensors:
            if not isinstance(self.scene.sensors[cam], Camera):
                continue
            base_cam = self.scene[cam]
            mask = (
                base_cam.data.output["semantic_segmentation"][0].detach().cpu().numpy()
            )
            img = base_cam.data.output["rgb"][0].detach().cpu().numpy()
            mask = np.where(mask >= 2, 1, 0)
            depth = base_cam.data.output["distance_to_image_plane"][0].detach().cpu().numpy()  # (H, W, 1)
            ret[cam] = {"rgb": img, "mask": mask, "depth": depth}

        return ret

    def transform_sim_to_splat(self, transform_static=False):
        """
        Update splat renderer transforms from simulation

        Parameters
        ----------
        transform_static : bool
            Whether to also transform static objects (like environment).
        """
        all_transforms = {}

        # rigid bodies
        for name in self.scene.rigid_objects:
            path = Path(self.usd_file).parent / "assets" / name / "splat.ply"
            if (
                "static" not in name or transform_static
            ) and path.exists():  # splat exists
                pos = self.scene[name].data.root_state_w[0, :3]
                quat = self.scene[name].data.root_state_w[0, 3:7]
                all_transforms[name] = (pos, quat)

        #  robot - this will only fire if setup_splat_robot has been called otherwise views will be empty
        for v_name in self.views:
            view = self.views[v_name]
            pos, quat = view.get_world_poses(usd=False)
            pos, quat = pos.squeeze(), quat.squeeze()
            all_transforms[v_name] = (pos, quat)

        if len(all_transforms) > 0:  # only transform if there is something to transform
            self.splat_renderer.transform_many(all_transforms)

        # set all cameras so that static cameras are set
        if transform_static:
            cam_extrinsics_dict = {}
            for name in self.splat_renderer.cameras:
                pos = self.scene[name].data.pos_w[0].detach().cpu().numpy()
                quat = self.scene[name].data.quat_w_world[0]

                rot = math.matrix_from_quat(quat).detach().cpu().numpy()
                cam_extrinsics_dict[name] = {"pos": pos, "rot": rot}

            # if len(self.splat_renderer.pcds) > 0:
            self.splat_renderer.render(cam_extrinsics_dict)

    def render_splat(self):
        # get camera extrinsics
        cam_extrinsics_dict = {}
        for name in self.splat_renderer.cameras:
            if "wrist" in name:
                pos = self.scene[name].data.pos_w[0].detach().cpu().numpy()
                quat = self.scene[name].data.quat_w_world[0]

                rot = math.matrix_from_quat(quat).detach().cpu().numpy()
                cam_extrinsics_dict[name] = {"pos": pos, "rot": rot}

        # perform splat rendering
        # if len(self.splat_renderer.pcds) > 0:
        rgb = self.splat_renderer.render(cam_extrinsics_dict)
        # else:
        #     rgb = {
        #         name: torch.zeros(
        #             (
        #                 self.splat_renderer.cameras[name].image_height,
        #                 self.splat_renderer.cameras[name].image_width,
        #                 3,
        #             )
        #         )
        #         for name in cam_extrinsics_dict
        #     }

        # process output
        for k, v in rgb.items():
            rgb[k] = v.detach().cpu().numpy()
            rgb[k] = np.clip(rgb[k], 0, 1)
            rgb[k] = (rgb[k] * 255).astype(np.uint8)

            # TODO: why is there a resize?
            rgb[k] = cv2.resize(rgb[k], (rgb[k].shape[1] // 2, rgb[k].shape[0] // 2))
            rgb[k] = cv2.resize(rgb[k], (rgb[k].shape[1] * 2, rgb[k].shape[0] * 2))

        return rgb
