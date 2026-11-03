import torch
import numpy as np
import polaris.utils as utils
from polaris.splat_renderer.gaussian_renderer import GaussianModel, render, render_3dgs
from polaris.splat_renderer.scene.cameras import Camera
from pathlib import Path

class DummyPipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    depth_ratio = 0.0
    debug = False


class SplatRenderer:
    def __init__(self, splats, bg_color=[0.5, 0.5, 0.5], device=0):
        # self.bg_color = bg_color
        self.device = device
        self.bg_color = torch.tensor(bg_color).to(self.device).float()
        self.pcds = splats
        self.big_model = GaussianModel(3)
        self.original_big_model = GaussianModel(3)
        self.splat_mapping = {}

        self.init_models()
        print("Finished loading models!")

        self.pipe = DummyPipe()
        # self.cameras = self.init_cams(fovx=fovx, fovy=fovy, res=res)

    def render_raw(self, extrinsics_dict):
        images = {}
        for name in self.cameras:
            if name in extrinsics_dict:
                cam_t = extrinsics_dict[name]["pos"]
                cam_r = extrinsics_dict[name]["rot"]

                self.cameras[name].set_extrinsics(cam_r, cam_t)

                render_pkg = render(
                    self.cameras[name], self.big_model, self.pipe, self.bg_color
                )
                image = render_pkg["render"]
                images[name] = image.permute(1, 2, 0).clone()
        return images

    def render(self, extrinsics_dict):
        """
        extrinsics_dict: dict
        {
            "name": {"pos": torch.Tensor, "rot": torch.Tensor}
            ...
        }
        """

        # permute axis to match coordinate frame
        p_mat = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        images = {}
        # for name, extrinsics in extrinsics_dict.items():
        for name in self.cameras:
            if name in extrinsics_dict:
                cam_t = extrinsics_dict[name]["pos"]
                cam_r = extrinsics_dict[name]["rot"]

                cam_r = cam_r @ p_mat

                self.cameras[name].set_extrinsics(cam_r, cam_t)

            render_pkg = render(
                self.cameras[name], self.big_model, self.pipe, self.bg_color
            )
            image = render_pkg["render"]
            images[name] = image.permute(1, 2, 0).clone()
        return images

    def init_cameras(self, cam_dict):
        """
        cam_dict: dict
        {
            "name": {"fovx": float, "fovy": float, "res": (height, width)}
             ...
        }
        """
        self.cameras = {}
        for name, cam_params in cam_dict.items():
            self.cameras[name] = Camera(
                colmap_id=0,
                R=np.eye(3),
                T=np.array([0.0, 0.0, 0.0]),
                FoVy=cam_params["fovy"],
                FoVx=cam_params["fovx"],
                image=torch.zeros(3, cam_params["res"][0], cam_params["res"][1]),
                gt_alpha_mask=None,
                image_name="test",
                uid=123,
                data_device=self.device,
            )

    def init_models(self):
        self.big_model._xyz = self.big_model._xyz.to(self.device)
        self.big_model._rotation = self.big_model._rotation.to(self.device)
        self.big_model._opacity = self.big_model._opacity.to(self.device)
        self.big_model._features_rest = self.big_model._features_rest.to(self.device)
        self.big_model._features_dc = self.big_model._features_dc.to(self.device)
        self.big_model._scaling = self.big_model._scaling.to(self.device)
        for name, pcd_path in self.pcds.items():
            model = GaussianModel(3)
            model.load_ply(pcd_path, "2dgs")

            # get mappings
            cur_len = self.big_model._xyz.shape
            self.splat_mapping[name] = (cur_len[0], cur_len[0] + model._xyz.shape[0])

            self.big_model._xyz = torch.cat(
                [self.big_model._xyz, model._xyz], dim=0
            ).requires_grad_()
            self.big_model._rotation = torch.cat(
                [self.big_model._rotation, model._rotation], dim=0
            ).requires_grad_()
            self.big_model._opacity = torch.cat(
                [self.big_model._opacity, model._opacity], dim=0
            ).requires_grad_()
            self.big_model._features_rest = torch.cat(
                [self.big_model._features_rest, model._features_rest], dim=0
            ).requires_grad_()
            self.big_model._features_dc = torch.cat(
                [self.big_model._features_dc, model._features_dc], dim=0
            ).requires_grad_()
            self.big_model._scaling = torch.cat(
                [self.big_model._scaling, model._scaling], dim=0
            ).requires_grad_()

        self.original_big_model._xyz = self.big_model._xyz.clone()
        self.original_big_model._rotation = self.big_model._rotation.clone()
        self.original_big_model._opacity = self.big_model._opacity.clone()
        self.original_big_model._features_rest = self.big_model._features_rest.clone()
        self.original_big_model._features_dc = self.big_model._features_dc.clone()
        self.original_big_model._scaling = self.big_model._scaling.clone()

    def add_splats(self, splats):
        for name, pcd_path in splats.items():
            model = GaussianModel(3)
            model.load_ply(pcd_path)

            cur_len = self.big_model._xyz.shape
            self.splat_mapping[name] = (cur_len[0], cur_len[0] + model._xyz.shape[0])

            self.big_model._xyz = torch.cat(
                [self.big_model._xyz, model._xyz], dim=0
            ).requires_grad_()
            self.big_model._rotation = torch.cat(
                [self.big_model._rotation, model._rotation], dim=0
            ).requires_grad_()
            self.big_model._opacity = torch.cat(
                [self.big_model._opacity, model._opacity], dim=0
            ).requires_grad_()
            self.big_model._features_rest = torch.cat(
                [self.big_model._features_rest, model._features_rest], dim=0
            ).requires_grad_()
            self.big_model._features_dc = torch.cat(
                [self.big_model._features_dc, model._features_dc], dim=0
            ).requires_grad_()
            self.big_model._scaling = torch.cat(
                [self.big_model._scaling, model._scaling], dim=0
            ).requires_grad_()

        self.original_big_model._xyz = self.big_model._xyz.clone()
        self.original_big_model._rotation = self.big_model._rotation.clone()
        self.original_big_model._opacity = self.big_model._opacity.clone()
        self.original_big_model._features_rest = self.big_model._features_rest.clone()
        self.original_big_model._features_dc = self.big_model._features_dc.clone()
        self.original_big_model._scaling = self.big_model._scaling.clone()

    def transform_many(self, all_transforms):
        """
        all_transforms: dict
        {
            "name": (pos (torch.Tensor), rot (torch.Tensor))
            ...
        }

        """
        with torch.no_grad():
            indices = []
            properties = []
            for name, transform in all_transforms.items():
                translate = transform[0].to(self.device)
                rotate = transform[1].to(self.device)

                start = self.splat_mapping[name][0]
                end = self.splat_mapping[name][1]

                new_xyz = (
                    utils.rotate_vector_by_quaternion(
                        rotate, self.original_big_model._xyz[start:end]
                    )
                    + translate
                )
                new_rotation = utils.multiply_quaternions(
                    rotate, self.original_big_model._rotation[start:end]
                )

                new_features_rest = self.original_big_model._features_rest[start:end]

                indices.append(torch.arange(start, end))
                properties.append(
                    {
                        "xyz": new_xyz,
                        "rotation": new_rotation,
                        "features_rest": new_features_rest,
                    }
                )

            indices = torch.cat(indices)
            xyzs = torch.cat([prop["xyz"] for prop in properties])
            rotations = torch.cat([prop["rotation"] for prop in properties])
            features_rests = torch.cat([prop["features_rest"] for prop in properties])

            self.big_model._xyz[indices] = xyzs
            self.big_model._rotation[indices] = rotations
            self.big_model._features_rest[indices] = features_rests


class SplatRenderer3DGS:
    def __init__(self, splats, bg_color=[0.5, 0.5, 0.5], device=0):
        self.device = device
        self.bg_color = torch.tensor(bg_color).to(self.device).float()
        self.pcds = splats
        self.pipe = DummyPipe()

        self.model = GaussianModel(3)
        self.model_2dgs = GaussianModel(3)
        self.original_model = GaussianModel(3)
        self.splat_mapping = {}

        self.init_models()
        print("Finished loading models!")

    def _load_into_model(self, pcd_path):
        name = Path(pcd_path).stem
        model = GaussianModel(3)
        model.load_ply(pcd_path, "3dgs")

        cur_len = self.model._xyz.shape[0]
        self.splat_mapping[name] = (cur_len, cur_len + model._xyz.shape[0])

        with torch.no_grad():
            for attr in ['_xyz', '_rotation', '_opacity', '_features_rest', '_features_dc', '_scaling']:
                setattr(self.model, attr, torch.cat(
                    [getattr(self.model, attr), getattr(model, attr).to(self.device)], dim=0
                ))

    def init_models(self):
        with torch.no_grad():
            for attr in ['_xyz', '_rotation', '_opacity', '_features_rest', '_features_dc', '_scaling']:
                setattr(self.model, attr, getattr(self.model, attr).to(self.device))

        for name, pcd_path in self.pcds.items():
            self._load_into_model(pcd_path)

        self._clone_originals()

    def add_splats(self, splats):
        for name, pcd_path in splats.items():
            self._load_into_model(pcd_path)
        self._clone_originals()

    def _clone_originals(self):
        with torch.no_grad():
            for attr in ['_xyz', '_rotation', '_opacity', '_features_rest', '_features_dc', '_scaling']:
                setattr(self.original_model, attr, getattr(self.model, attr).clone())

    def init_cameras(self, cam_dict):
        self.cameras = {}
        for name, cam_params in cam_dict.items():
            self.cameras[name] = Camera(
                colmap_id=0,
                R=np.eye(3),
                T=np.array([0.0, 0.0, 0.0]),
                FoVy=cam_params["fovy"],
                FoVx=cam_params["fovx"],
                image=torch.zeros(3, cam_params["res"][0], cam_params["res"][1]),
                gt_alpha_mask=None,
                image_name="test",
                uid=123,
                data_device=self.device,
            )

    def render(self, extrinsics_dict):
        p_mat = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        images = {}
        for name in self.cameras:
            if name in extrinsics_dict:
                cam_r = extrinsics_dict[name]["rot"] @ p_mat
                self.cameras[name].set_extrinsics(cam_r, extrinsics_dict[name]["pos"])
            with torch.no_grad():
                image = render_3dgs(self.cameras[name], self.model, self.pipe, self.bg_color)["render"]
            images[name] = image.permute(1, 2, 0).clone()
        return images

    def render_raw(self, extrinsics_dict):
        images = {}
        for name in self.cameras:
            if name in extrinsics_dict:
                self.cameras[name].set_extrinsics(
                    extrinsics_dict[name]["rot"], extrinsics_dict[name]["pos"]
                )
            with torch.no_grad():
                image = render_3dgs(self.cameras[name], self.model, self.pipe, self.bg_color)["render"]
            images[name] = image.permute(1, 2, 0).clone()
        return images

    def transform_many(self, all_transforms):
        with torch.no_grad():
            indices, props = [], []
            for name, transform in all_transforms.items():
                if name not in self.splat_mapping:
                    continue
                translate = transform[0].to(self.device)
                rotate = transform[1].to(self.device)
                start, end = self.splat_mapping[name]

                new_xyz = utils.rotate_vector_by_quaternion(
                    rotate, self.original_model._xyz[start:end]
                ) + translate
                new_rotation = utils.multiply_quaternions(
                    rotate, self.original_model._rotation[start:end]
                )
                indices.append(torch.arange(start, end))
                props.append({"xyz": new_xyz, "rotation": new_rotation})

            if indices:
                idx = torch.cat(indices)
                self.model._xyz[idx] = torch.cat([p["xyz"] for p in props])
                self.model._rotation[idx] = torch.cat([p["rotation"] for p in props])