"""
No-op environment runner for training without evaluation rollouts.
Returns a dummy test_mean_score so checkpoint manager is satisfied.
"""
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class NoopImageRunner(BaseImageRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy) -> dict:
        return {'test_mean_score': 0.0}
