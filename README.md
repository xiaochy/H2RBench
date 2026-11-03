# H2RBench: A Real-to-Sim Benchmark for Evaluating Human-to-Robot Transfer

Accepted at **CoRL 2026**.

**[🌐 Project Website](https://h2rbench.github.io/)** • **[📄 arXiv](https://arxiv.org/abs/2609.24778)** • **[📋 PDF](https://arxiv.org/pdf/2609.24778)**

**Chuyang Xiao\*, Haotian Zhan\*, Sriram Krishna, Peilin Meng, Muhammad Zubair Irshad, Sergey Zakharov, David Held**  
*Robotics Institute, Carnegie Mellon University · University of Michigan · Toyota Research Institute*  
\* Equal contribution; authors are listed in alphabetical order.

This repository provides the simulation benchmark, dataset generation tools, and policy evaluation framework for H2RBench — built on top of [PolaRiS](https://github.com/arhanjain/polaris).

---

## Built on PolaRiS

This project extends [PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies](https://arxiv.org/abs/2512.16881). PolaRiS provides the core environment reconstruction pipeline, Isaac Lab integration, and evaluation infrastructure that we build upon.

If you use this repository, please also cite the original PolaRiS work:

```bibtex
@misc{jain2025polarisscalablerealtosimevaluations,
      title={PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies},
      author={Arhan Jain and Mingtong Zhang and Kanav Arora and William Chen and Marcel Torne
              and Muhammad Zubair Irshad and Sergey Zakharov and Yue Wang and Sergey Levine
              and Chelsea Finn and Wei-Chiu Ma and Dhruv Shah and Abhishek Gupta and Karl Pertsch},
      year={2025},
      eprint={2512.16881},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.16881},
}
```

---

## Environment Setup

### 1. Clone the repository (recursively)

```bash
git clone --recursive git@github.com:DavidZ2851/polaris.git
cd polaris
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies with uv

If you don't have uv installed, see [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

By default CUDA 13 is supported. If you have an older CUDA version, downgrade torch/torchvision in `pyproject.toml` accordingly.

```bash
uv sync
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Download PolaRiS environments

```bash
uvx hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub
```

### 5. Install ffmpeg (for saving videos)

```bash
sudo apt install ffmpeg
```

### 6. Download large files

Several large files are not tracked in git and must be downloaded separately.

**[Download from Google Drive](https://drive.google.com/drive/u/0/folders/17L0uBJ1sspTK_x4m-4raml4SKH1TifHc)**

After downloading, unzip each file and place it in the location described in the [Large File Setup](#large-file-setup) section below.

The human datasets are available on Hugging Face:

- [Kovavavvavava/pick_place_red_mug_human](https://huggingface.co/datasets/Kovavavvavava/pick_place_red_mug_human): 180 episodes
- [Kovavavvavava/pick_toys_human](https://huggingface.co/datasets/Kovavavvavava/pick_toys_human): 315 episodes
- [Kovavavvavava/insert_donut_human](https://huggingface.co/datasets/Kovavavvavava/insert_donut_human): 300 episodes
- [Kovavavvavava/stack_bowls_human](https://huggingface.co/datasets/Kovavavvavava/stack_bowls_human): 300 episodes

The human and robot datasets are also available on Google Drive:

- **[Human data](https://drive.google.com/drive/u/0/folders/1cYsrlzYV29HuG9yh4_F8whCqe9nLCwVM)**
- **[Robot data](https://drive.google.com/drive/u/0/folders/1AnVtV-MxgQs41rIdMai2mfJVDp1N1gH2)**

---

## 1. Generating Datasets

Use `scripts/generate_dataset.py` to collect demonstration data using a motion planner inside the sim.

```bash
python scripts/generate_dataset.py \
    --save_dir /home/$USER/polaris/debug
```

Key options (from `DataArgs` in `src/polaris/config.py`):

| Flag | Default | Description |
|------|---------|-------------|
| `--environment` | `DROID-PutRedCup-no-curtain` | IsaacLab environment to use |
| `--save_dir` | required | Directory to save the dataset |
| `--num_episodes` | 50 | Number of episodes to collect |
| `--max_attempts` | 100 | Max planning attempts per episode |
| `--headless` | True | Run without GUI |

---

## 2. Testing the Environment

Verify the environment loads and renders correctly:

```bash
python scripts/test_env.py
```

---

## 3. Policy Evaluation

Evaluation follows a **client-server** architecture:

- **Server** (`src/polaris/server/`): loads the policy checkpoint and serves actions over ZMQ
- **Client** (`src/polaris/client/`): connects to the server, sends observations, receives actions
- **Eval script** (`scripts/eval_policy.py`): runs the IsaacLab environment and drives the eval loop

### Available Servers and Clients

| Policy | Server | Client Name | Notes |
|--------|--------|-------------|-------|
| Diffusion Policy | `src/polaris/server/diffusion_policy_server.py` | `DiffusionPolicy` | `robodiff` conda env |
| AMPLIFY | `src/polaris/server/amplify_server.py` | `AMPLIFY` | `amplify` conda env |
| LeRobot Diffusion | `src/polaris/server/lerobot_diffusion_server.py` | `LeRobotDiffusion` | `robodiff` conda env |
| DROID JointPos | — | `DroidJointPos` | openpi-based policies |
| GHOST | `src/polaris/server/ghost_server.py` | `ghost` | `lerobot` pixi env; see `server/launch_ghost_server.sh` |
| Point-Policy | `src/polaris/server/point_policy_server.py` | `point_policy` | `point-policy` conda env; see `server/launch_pp_server.sh` |

Client names are case-sensitive and must match `--policy.client` exactly — note that `ghost` and `point_policy` are lowercase.

### Camera Usage per Policy

Camera names in `obs["splat"]` come from the USD scene prim names (e.g. `cam0`, `cam1`, `wrist_cam`).

| Policy | Cameras used | Notes |
|--------|-------------|-------|
| `DiffusionPolicy` | `cam1` | Front camera only (`(H, W, 3)`) |
| `AMPLIFY` | `cam0`, `cam1` | Two views stacked as `(2, H, W, 3)`; `cam0` = front, `cam1` = left |
| `DroidJointPos` | `external_cam`, `wrist_cam` | Uses sim camera names directly |
| `ghost` | `cam0`, `cam1`, `wrist_cam` | RGB **and** depth for `cam0`/`cam1`; depth sent as uint16 millimetres |
| `point_policy` | `cam0`, `cam1` | RGB + depth; mapped to `pixels1`/`pixels2` as Point-Policy expects |

`ghost` and `point_policy` are the only policies that consume depth (`obs["splat"]["cam0_depth"]`, `cam1_depth`).

---

### Example: Diffusion Policy

**Step 1 — Start the server** (separate terminal, `robodiff` env):

```bash
cd ~/vxiao/human2robot/benchmark_new/polaris
conda activate robodiff
python src/polaris/server/diffusion_policy_server.py \
    --ckpt_path policy_ckpt/diffusion_policy/red_mug/new_camera_calib/human40_robot40.ckpt \
    --port 5557
```

**Step 2 — Run evaluation** (polaris `.venv`):

```bash
cd ~/vxiao/human2robot/benchmark_new/polaris
source .venv/bin/activate
python scripts/eval_policy.py \
    --policy.client DiffusionPolicy \
    --policy.host localhost \
    --policy.port 5557 \
    --policy.open_loop_horizon 8 \
    --environment DROID-PutRedCup-no-curtain \
    --run_folder runs/new_camera_calib/human40_robot40 \
    --rollouts 30
```

---

### Example: AMPLIFY

**Step 1 — Bundle checkpoint** (if not already bundled):

```bash
python src/polaris/policy/amplify/amplify/bundle_amplify.py \
    --mt_ckpt  /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/<folder>/motion.pt \
    --fd_ckpt  /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/<folder>/forward.pt \
    --id_ckpt  /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/<folder>/inverse.pt \
    --save-to  /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/<folder>/amplify.pt \
    --overwrite
```

**Step 2 — Start the server** (`amplify` conda env):

```bash
cd ~/vxiao/human2robot/benchmark_new/polaris
conda activate amplify
python src/polaris/server/amplify_server.py \
    --ckpt_path /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/<folder>/amplify.pt \
    --text_emb  /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/pick_mug_text_emb.npy \
    --port 5557 \
    --vis_tracks
```

**Step 3 — Run evaluation** (polaris `.venv`):

```bash
cd ~/vxiao/human2robot/benchmark_new/polaris
source .venv/bin/activate
python scripts/eval_policy.py \
    --policy.client AMPLIFY \
    --policy.host localhost \
    --policy.port 5557 \
    --policy.open_loop_horizon 8 \
    --environment DROID-PutRedCup-no-curtain \
    --run_folder runs/new_camera_calib/amplify/<folder> \
    --rollouts 30
```

---

### Example: GHOST

Code: [r-pad/ghost (`haotian/h2rbenchmark`)](https://github.com/r-pad/ghost/tree/haotian/h2rbenchmark) •
[r-pad/lerobot (`polaris`)](https://github.com/r-pad/lerobot/tree/polaris)

**Step 1 — Start the server** (separate terminal, `lerobot` pixi env):

```bash
cd ~/lerobot
pixi shell
python ~/polaris/src/polaris/server/ghost_server.py \
    --policy_path ~/lerobot/outputs/train/pick_toys_r400h0/checkpoints/last/pretrained_model \
    --open_loop_horizon 8 \
    --port 8768
```

Or use the launcher, which wraps the same command and pins `CUDA_VISIBLE_DEVICES=1`:

```bash
# defaults may be overridden via environment variables
POLICY_PATH=~/lerobot/outputs/train/pick_toys_r400h0/checkpoints/last/pretrained_model \
PORT=8768 \
OPEN_LOOP_HORIZON=8 \
bash src/polaris/server/launch_ghost_server.sh
```

**Step 2 — Run evaluation** (polaris `.venv`):

```bash
source .venv/bin/activate
python scripts/eval_policy.py \
    --policy.client ghost \
    --policy.host localhost \
    --policy.port 8768 \
    --policy.open_loop_horizon 8 \
    --environment DROID-PutRedCup-no-curtain \
    --run_folder runs/ghost/pick_toys_r400h0 \
    --rollouts 30
```

The client converts the policy's end-effector actions to joint commands with CuRobo IK,
and `--open_loop_horizon` must match the value the server was started with.

---

### Example: Point-Policy

Code: [DavidZ2851/Point-Policy (`feat/polaris`)](https://github.com/DavidZ2851/Point-Policy/tree/feat/polaris)

**Step 1 — Start the server** (separate terminal, `point-policy` conda env):

```bash
conda activate point-policy
python src/polaris/server/point_policy_server.py \
    --bc_weight ~/Point-Policy/point_policy/exp_local/2026.05.22/pick_toys_simrobot_100/deterministic/223855_hidden_dim_256/snapshot/100000.pt \
    --port 8765 \
    --overrides "agent=point_policy" "suite=point_policy" "dataloader=point_policy" \
                "suite.use_robot_points=true" "suite.use_object_points=true" \
                "experiment=eval_point_policy" "suite/task/franka_env=pick_place_toys" \
                "data_dir=$HOME/pp_data/pick_place_toys/pick_toys_simrobot_100/processed_data_pkl/expert_demos"
```

`--overrides` are passed straight through to Hydra. Two of them matter most:

| Override | Meaning |
|----------|---------|
| `suite/task/franka_env=<task>` | Task config from `Point-Policy/point_policy/cfgs/suite/task/franka_env/` |
| `data_dir=<path>` | Processed demo directory; the server reads normalization stats and `num_demos_per_task` from here, so it **must** match what the checkpoint was trained on |

Or use the launcher, editing `data_dir_1` / `bc_weight_1` / `TASK_NAME_1` at the top:

```bash
bash src/polaris/server/launch_pp_server.sh
```

**Step 2 — Run evaluation** (polaris `.venv`):

```bash
source .venv/bin/activate
python scripts/eval_policy.py \
    --policy.client point_policy \
    --policy.host localhost \
    --policy.port 8765 \
    --policy.open_loop_horizon 8 \
    --environment DROID-PutRedCup-no-curtain \
    --run_folder runs/point_policy/pick_toys_simrobot_100 \
    --rollouts 30
```

---

## Key `eval_policy.py` Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--policy.client` | required | Client name (e.g. `DiffusionPolicy`, `AMPLIFY`) |
| `--policy.host` | required | Server host (e.g. `localhost`) |
| `--policy.port` | required | Server port |
| `--policy.open_loop_horizon` | required | Steps to execute per action chunk |
| `--environment` | required | IsaacLab environment ID |
| `--run_folder` | required | Where to save videos and `eval_results.csv` |
| `--rollouts` | required | Number of episodes to evaluate |
| `--max-episode-length` | 300 | Max steps per episode |
| `--seed` | 42 | Random seed |
| `--task_config` | `None` | Path to a custom `task_config.yaml` |

Results are saved to `<run_folder>/eval_results.csv` and `<run_folder>/episode_N.mp4`. Evaluation supports **resuming** — if the CSV already exists, completed episodes are skipped automatically.

### Using a Custom Task Config

By default, `eval_policy.py` loads `task_config.yaml` from the environment folder. To evaluate with a different object-randomization / waypoint config, pass `--task_config`:

```
PolaRiS-Hub/put_red_cup_no_curtain/
├── task_config.yaml           # default
├── task_config_y_left.yaml    # custom variant
└── task_config_y_right.yaml   # custom variant
```

```bash
python scripts/eval_policy.py \
    --policy.client DiffusionPolicy \
    --policy.host localhost --policy.port 5557 --policy.open_loop_horizon 8 \
    --environment DROID-PutRedCup-no-curtain \
    --run_folder runs/new_camera_calib/pick_mug/object_placement/mug_left \
    --rollouts 30 \
    --task_config PolaRiS-Hub/put_red_cup_no_curtain/task_config_y_left.yaml
```

---

## Adding a New Policy

1. Create a new client in `src/polaris/client/` inheriting from `InferenceClient` and decorate with `@InferenceClient.register(client_name="YourPolicy")`
2. Create a corresponding server in `src/polaris/server/` that loads your checkpoint and serves actions over ZMQ
3. Import the new client in `src/polaris/client/__init__.py`
4. Run with `--policy.client YourPolicy`

---

## Large File Setup

Several large files are not tracked in git and must be downloaded from Google Drive:

**[Download from Google Drive](https://drive.google.com/drive/u/0/folders/17L0uBJ1sspTK_x4m-4raml4SKH1TifHc)**

### `content.zip` → `src/curobo/src/curobo/content/`

Contains cuRobo asset files (robot meshes, scene assets).

```bash
unzip content.zip -d /
# Files extract to: src/curobo/src/curobo/content/
```

> The zip stores absolute paths, so unzipping to `/` places files at the correct location automatically.

### `put_red_cup_no_curtain.zip` → `PolaRiS-Hub/put_red_cup_no_curtain/`

Contains the put-red-cup task environment assets (meshes, USD files, splat, config).

```bash
unzip put_red_cup_no_curtain.zip -d /tmp/polaris_env
cp -r /tmp/polaris_env/.../put_red_cup_no_curtain PolaRiS-Hub/
```

Expected result:

```
PolaRiS-Hub/put_red_cup_no_curtain/
├── assets/
├── textures/
├── cam_calibration.json
├── initial_conditions.json
├── scene.json
├── scene.usda
├── task_config.yaml
└── task_config_*.yaml   # optional variants
```

### `nvidia_droid.zip` → `PolaRiS-Hub/nvidia_droid/`

Contains the NVIDIA DROID robot USD and segmented Gaussian splat files.

```bash
unzip nvidia_droid.zip -d /
# Files extract to: PolaRiS-Hub/nvidia_droid/
```

Expected result:

```
PolaRiS-Hub/nvidia_droid/
├── franka_robotiq_2f_85_flattened.usd
├── splat.ply
└── SEGMENTED/
    └── *.ply
```

---

## Citation

If you find this repository useful, please cite our work:

```bibtex
@inproceedings{xiao2026h2rbench,
      title={H2RBench: A Real-to-Sim Benchmark for Evaluating Human-to-Robot Transfer},
      author={Chuyang Xiao and Haotian Zhan and Sriram Krishna and Peilin Meng
              and Muhammad Zubair Irshad and Sergey Zakharov and David Held},
      booktitle={Conference on Robot Learning (CoRL)},
      year={2026},
      eprint={2609.24778},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2609.24778},
}
```

And please also cite the underlying PolaRiS framework (see [Built on PolaRiS](#built-on-polaris) above).
