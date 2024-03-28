## scripts

### process_adop.py

```python
from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

def process_adop(scene_path: Path, checkpoint_path: Path, scales: List[int]) -> None:
    for scale in scales:
        (scene_path / f'undistorted_images_adop-{scale}').mkdir(exist_ok=True)

    ep_dirs = list(filter(lambda x: 'ep' in x.name, checkpoint_path.iterdir()))
    assert len(ep_dirs) == 1
    pose = list(filter(lambda x: 'poses' in x.name, (checkpoint_path / ep_dirs[0]).iterdir()))
    assert len(pose) == 1
    poses_path = pose[0]

    intrinsic = list(filter(lambda x: 'intrinsics' in x.name, (checkpoint_path / ep_dirs[0]).iterdir()))
    assert len(intrinsic) == 1
    intrinsics_path = intrinsic[0]

    point = list(filter(lambda x: 'points' in x.name, (checkpoint_path / ep_dirs[0]).iterdir()))
    assert len(point) == 1
    points_path = point[0]

    adop_points = torch.load(points_path, map_location='cpu')
    with (scene_path / 'adop-scene-bounds.txt').open('w') as f:
        min_bounds = adop_points.t_position[:, :3].min(dim=0)[0]
        max_bounds = adop_points.t_position[:, :3].max(dim=0)[0]
        f.write('{}\n{}\n'.format(' '.join([str(x.item()) for x in min_bounds]),
                                  ' '.join([str(x.item()) for x in max_bounds])))
        print(f'Scene bounds: {min_bounds} {max_bounds}')

    adop_poses = torch.load(poses_path, map_location='cpu').poses_se3
    min_near = 1e10
    max_far = -1
    with (scene_path / 'adop-poses.txt').open('w') as f:
        for pose in tqdm(adop_poses):
            position = pose[4:7]
            distance = (adop_points.t_position[:, :3] - position).norm(dim=-1)
            near = distance.min().item()
            far = distance.max().item()
            min_near = min(near, min_near)
            max_far = max(far, max_far)
            f.write('{} {} {}\n'.format(' '.join([str(x.item()) for x in pose[:7]]), near, far))
    print(f'Wrote {len(adop_poses)}. Near: {min_near}, far: {far}')

    adop_intrinsics = torch.load(intrinsics_path, map_location='cpu').intrinsics
    assert adop_intrinsics.shape == (1, 13)
    adop_intrinsics = adop_intrinsics.squeeze().detach()
    K = np.eye(3)
    K[0, 0] = adop_intrinsics[0]
    K[0, 1] = adop_intrinsics[4]
    K[1, 1] = adop_intrinsics[1]
    K[0, 2] = adop_intrinsics[2]
    K[1, 2] = adop_intrinsics[3]

    distortion = adop_intrinsics[5:].numpy()

    print('Writing undistorted images')
    new_intrinsics = []
    with (scene_path / 'images.txt').open() as f:
        for line in tqdm(f):
            image_path = scene_path / 'images' / line.strip()
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, distortion, (w, h), 1, (w, h))
            dst = cv2.undistort(img, K, distortion, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            for scale in [1, 2, 4, 8]:
                if scale != 1:
                    cv2.imwrite(str(scene_path / f'undistorted_images_adop-{scale}' / image_path.name),
                                cv2.resize(dst, (dst.shape[1] // scale, dst.shape[0] // scale),
                                           interpolation=cv2.INTER_LANCZOS4))
                else:
                    cv2.imwrite(str(scene_path / f'undistorted_images_adop-{scale}' / image_path.name), dst)
            new_intrinsics.append([dst.shape[1], dst.shape[0]] + newcameramtx.reshape((-1)).tolist())

    with (scene_path / 'undistorted_intrinsics_adop.txt').open('w') as f:
        for i in new_intrinsics:
            f.write('{}\n'.format(' '.join([str(x) for x in i])))


def _get_opts() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--scene_path', type=Path, required=True)
    parser.add_argument('--checkpoint_path', type=Path, required=True)
    parser.add_argument('--scales', type=list, default=[1, 2, 4, 8])

    return parser.parse_args()


def main(hparams: Namespace) -> None:
    process_adop(hparams.scene_path, hparams.checkpoint_path, hparams.scales)


if __name__ == '__main__':
    main(_get_opts())

```

## pynerf

### pynerf_constants.py

```python
from enum import Enum

EXPLICIT_LEVEL = 'explicit_level'
LEVELS = 'levels'
LEVEL_COUNTS = 'level_counts'

RGB = 'image'
DEPTH = 'depth'
RAY_INDEX = 'ray_index'
TRAIN_INDEX = 'train_index'
WEIGHT = 'weight'
POSE_SCALE_FACTOR = 'pose_scale_factor'
RENDER_LEVELS = 'render_levels'
class PyNeRFFieldHeadNames(Enum):
    LEVELS = 'levels'
    LEVEL_COUNTS = 'level_counts'

```

## pynerf

### __init__.py

```python

```

## pynerf

### pynerf_data_config.py

```python
"""
PyNeRF dataparsers configuration file.
"""
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from pynerf.data.dataparsers.adop_dataparser import AdopDataParserConfig
from pynerf.data.dataparsers.mipnerf_dataparser import MipNerf360DataParserConfig
from pynerf.data.dataparsers.multicam_dataparser import MulticamDataParserConfig

multicam_dataparser = DataParserSpecification(config=MulticamDataParserConfig())
mipnerf360_dataparser = DataParserSpecification(config=MipNerf360DataParserConfig())
adop_dataparser = DataParserSpecification(config=AdopDataParserConfig())

```

## pynerf

### pynerf_method_config.py

```python
"""
PyNeRF data configuration file.
"""
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig, ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from pynerf.data.datamanagers.random_subset_datamanager import RandomSubsetDataManagerConfig
from pynerf.data.datamanagers.weighted_datamanager import WeightedDataManagerConfig
from pynerf.data.dataparsers.mipnerf_dataparser import MipNerf360DataParserConfig
from pynerf.data.dataparsers.multicam_dataparser import MulticamDataParserConfig
from pynerf.data.pynerf_pipelines import PyNeRFPipelineConfig, PyNeRFDynamicBatchPipelineConfig
from pynerf.models.pynerf_model import PyNeRFModelConfig
from pynerf.models.pynerf_occupancy_model import PyNeRFOccupancyModelConfig

pynerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name='pynerf',
        steps_per_eval_image=5000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=PyNeRFPipelineConfig(
            datamanager=RandomSubsetDataManagerConfig(
                dataparser=MipNerf360DataParserConfig(),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=4096,
            ),
            model=PyNeRFModelConfig(
                eval_num_rays_per_chunk=4096,
            ),
        ),
        optimizers={
            'mlps': {
                'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                'scheduler': CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            'fields': {
                'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                'scheduler': CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            'proposal_networks': {
                'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                'scheduler': CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis='viewer',
        steps_per_eval_all_images=30000
    ),
    description='PyNeRF with proposal network. The default parameters are suited for outdoor scenes.',
)

pynerf_synthetic_method = MethodSpecification(
    config=TrainerConfig(
        method_name='pynerf-synthetic',
        steps_per_eval_all_images=50000,
        max_num_iterations=50001,
        mixed_precision=True,
        pipeline=PyNeRFPipelineConfig(
            datamanager=WeightedDataManagerConfig(
                dataparser=MulticamDataParserConfig(),
                train_num_rays_per_batch=8192),
            model=PyNeRFModelConfig(
                eval_num_rays_per_chunk=8192,
                appearance_embedding_dim=0,
                disable_scene_contraction=True,
                num_nerf_samples_per_ray=96,
                use_gradient_scaling=True,
                max_resolution=65536
            ),
        ),
        optimizers={
            'mlps': {
                'optimizer': AdamOptimizerConfig(lr=5e-3, eps=1e-15, weight_decay=1e-8),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            'fields': {
                'optimizer': AdamOptimizerConfig(lr=5e-3, eps=1e-15, weight_decay=1e-8),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            'proposal_networks': {
                'optimizer': AdamOptimizerConfig(lr=5e-3, eps=1e-15, weight_decay=1e-8),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis='viewer',
    ),
    description='PyNeRF with proposal network. The default parameters are suited for synthetic scenes.',
)

pynerf_occupancy_method = MethodSpecification(
    config=TrainerConfig(
        method_name='pynerf-occupancy-grid',
        steps_per_eval_all_images=20000,
        max_num_iterations=20001,
        mixed_precision=True,
        pipeline=PyNeRFDynamicBatchPipelineConfig(
            datamanager=WeightedDataManagerConfig(
                dataparser=MulticamDataParserConfig(),
                train_num_rays_per_batch=8192),
            model=PyNeRFOccupancyModelConfig(
                max_resolution=1024,
                eval_num_rays_per_chunk=8192,
                cone_angle=0,
                alpha_thre=0,
                grid_levels=1,
                appearance_embedding_dim=0,
                disable_scene_contraction=True,
                background_color='white',
                output_interpolation='color',
            ),
        ),
        optimizers={
            'mlps': {
                'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-8),
                'scheduler': CosineDecaySchedulerConfig(warm_up_end=512, max_steps=20000),
            },
            'fields': {
                'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-8),
                'scheduler': CosineDecaySchedulerConfig(warm_up_end=512, max_steps=20000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis='viewer',
    ),
    description='PyNeRF with occupancy grid. The default parameters are suited for synthetic scenes.',
)

```

## pynerf/samplers

### __init__.py

```python

```

## pynerf/samplers

### pynerf_vol_sampler.py

```python
from typing import Optional, Callable, Tuple

import torch
from jaxtyping import Float
from nerfacc import OccGridEstimator
from nerfstudio.cameras.rays import RaySamples, RayBundle, Frustums
from nerfstudio.model_components.ray_samplers import Sampler
from torch import Tensor


class PyNeRFVolumetricSampler(Sampler):
    """
    Similar to VolumetricSampler in NerfStudio, but passes additional camera ray information to density_fn
    """

    def __init__(
            self,
            occupancy_grid: OccGridEstimator,
            density_fn: Optional[Callable] = None,
    ):
        super().__init__()
        assert occupancy_grid is not None
        self.density_fn = density_fn
        self.occupancy_grid = occupancy_grid

    def get_sigma_fn(self, origins, directions, pixel_area, times=None) -> Optional[Callable]:
        """Returns a function that returns the density of a point.

        Args:Ø
            origins: Origins of rays
            directions: Directions of rays
            pixel_area: Pixel area of rays
            times: Times at which rays are sampled
        Returns:
            Function that returns the density of a point or None if a density function is not provided.
        """

        if self.density_fn is None or not self.training:
            return None

        density_fn = self.density_fn

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = directions[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

            return density_fn(positions, times=times[ray_indices] if times is not None else None, origins=t_origins,
                              directions=t_dirs, starts=t_starts[:, None], ends=t_ends[:, None],
                              pixel_area=pixel_area[ray_indices]).squeeze(-1)

        return sigma_fn

    def generate_ray_samples(self) -> RaySamples:
        raise RuntimeError(
            "The VolumetricSampler fuses sample generation and density check together. Please call forward() directly."
        )

    def forward(
            self,
            ray_bundle: RayBundle,
            render_step_size: float,
            near_plane: float = 0.0,
            far_plane: Optional[float] = None,
            alpha_thre: float = 0.01,
            cone_angle: float = 0.0,
    ) -> Tuple[RaySamples, Float[Tensor, "total_samples "]]:
        """Generate ray samples in a bounding box.

        Args:
            ray_bundle: Rays to generate samples for
            render_step_size: Minimum step size to use for rendering
            near_plane: Near plane for raymarching
            far_plane: Far plane for raymarching
            alpha_thre: Opacity threshold skipping samples.
            cone_angle: Cone angle for raymarching, set to 0 for uniform marching.

        Returns:
            a tuple of (ray_samples, packed_info, ray_indices)
            The ray_samples are packed, only storing the valid samples.
            The ray_indices contains the indices of the rays that each sample belongs to.
        """

        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        times = ray_bundle.times

        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            t_min = ray_bundle.nears.contiguous().reshape(-1)
            t_max = ray_bundle.fars.contiguous().reshape(-1)

        else:
            t_min = None
            t_max = None

        if far_plane is None:
            far_plane = 1e10

        if ray_bundle.camera_indices is not None:
            camera_indices = ray_bundle.camera_indices.contiguous()
        else:
            camera_indices = None
        ray_indices, starts, ends = self.occupancy_grid.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=t_min,
            t_max=t_max,
            sigma_fn=self.get_sigma_fn(rays_o, rays_d, ray_bundle.pixel_area.contiguous(), times),
            render_step_size=render_step_size,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=self.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        num_samples = starts.shape[0]
        if num_samples == 0:
            # create a single fake sample and update packed_info accordingly
            # this says the last ray in packed_info has 1 sample, which starts and ends at 1
            ray_indices = torch.zeros((1,), dtype=torch.long, device=rays_o.device)
            starts = torch.ones((1,), dtype=starts.dtype, device=rays_o.device)
            ends = torch.ones((1,), dtype=ends.dtype, device=rays_o.device)

        origins = rays_o[ray_indices]
        dirs = rays_d[ray_indices]
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts[..., None],
                ends=ends[..., None],
                pixel_area=ray_bundle[ray_indices].pixel_area,
            ),
            camera_indices=camera_indices,
        )

        if ray_bundle.times is not None:
            ray_samples.times = ray_bundle.times[ray_indices]

        if ray_bundle.metadata is not None:
            ray_samples.metadata = {}
            for k, v in ray_bundle.metadata.items():
                if isinstance(v, torch.Tensor):
                    ray_samples.metadata[k] = v[ray_indices]

        return ray_samples, ray_indices

```

## pynerf/models

### __init__.py

```python

```

## pynerf/models

### pynerf_occupancy_model.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import math
import nerfacc
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils import colormaps
from rich.console import Console

from pynerf.models.pynerf_base_model import PyNeRFBaseModelConfig, PyNeRFBaseModel
from pynerf.pynerf_constants import EXPLICIT_LEVEL, PyNeRFFieldHeadNames, LEVEL_COUNTS, LEVELS
from pynerf.samplers.pynerf_vol_sampler import PyNeRFVolumetricSampler

CONSOLE = Console(width=120)


@dataclass
class PyNeRFOccupancyModelConfig(PyNeRFBaseModelConfig):
    _target: Type = field(
        default_factory=lambda: PyNeRFOccupancyModel
    )

    max_num_samples_per_ray: int = 1024
    """Number of samples in field evaluation."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    grid_levels: int = 4
    """Levels of the grid used for the field."""
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""


class PyNeRFOccupancyModel(PyNeRFBaseModel):
    config: PyNeRFOccupancyModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.render_step_size is None:
            self.render_step_size = (
                    (self.scene_box.aabb[1] - self.scene_box.aabb[0]).max()
                    * math.sqrt(3)
                    / self.config.max_num_samples_per_ray
            ).item()
            CONSOLE.log(f'Setting render step size to {self.render_step_size}')
        else:
            self.render_step_size = self.config.render_step_size

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_box.aabb.flatten(),
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Sampler
        self.sampler = PyNeRFVolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x, step_size=self.render_step_size) * self.render_step_size,
            )

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

        return callbacks

    def get_outputs_inner(self, ray_bundle: RayBundle, explicit_level: Optional[int] = None):
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.near,
                far_plane=self.far,
                render_step_size=self.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        if explicit_level is not None:
            if ray_samples.metadata is None:
                ray_samples.metadata = {}
            ray_samples.metadata[EXPLICIT_LEVEL] = explicit_level

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)

        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0][..., None]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights, ray_indices=ray_indices,
                                num_rays=num_rays, )
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples, ray_indices=ray_indices,
                                    num_rays=num_rays)

        outputs = {'rgb': rgb, 'depth': depth}

        if explicit_level is None:
            accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
            alive_ray_mask = accumulation.squeeze(-1) > 0

            outputs['accumulation'] = accumulation
            outputs['alive_ray_mask'] = alive_ray_mask  # the rays we kept from sampler
            outputs['num_samples_per_ray'] = packed_info[:, 1]

            if self.training:
                outputs[LEVEL_COUNTS] = field_outputs[PyNeRFFieldHeadNames.LEVEL_COUNTS]
            else:
                levels = field_outputs[PyNeRFFieldHeadNames.LEVELS]
                outputs[LEVELS] = self.renderer_level(weights=weights,
                                                      semantics=levels.clamp(0, self.field.num_scales - 1),
                                                      ray_indices=ray_indices, num_rays=num_rays)

        return outputs

    def get_metrics_dict(self, outputs: Dict[str, any], batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        metrics_dict['num_samples_per_batch'] = outputs['num_samples_per_ray'].sum()

        return metrics_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        images_dict['alive_ray_mask'] = colormaps.apply_colormap(outputs['alive_ray_mask'])

        return metrics_dict, images_dict

```

## pynerf/models

### pynerf_base_model.py

```python
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import math
import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer, SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import ColormapOptions
from rich.console import Console
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from pynerf.fields.pynerf_base_field import parse_output_interpolation, parse_level_interpolation
from pynerf.fields.pynerf_field import PyNeRFField
from pynerf.pynerf_constants import LEVEL_COUNTS, LEVELS, WEIGHT, RENDER_LEVELS

CONSOLE = Console(width=120)

def ssim(
        target_rgbs: torch.Tensor,
        rgbs: torch.Tensor,
        max_val: float = 1,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
) -> float:
    """Computes SSIM from two images.
    This function was modeled after tf.image.ssim, and should produce comparable
    output.
    Args:
      rgbs: torch.tensor. An image of size [..., width, height, num_channels].
      target_rgbs: torch.tensor. An image of size [..., width, height, num_channels].
      max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
      filter_size: int >= 1. Window size.
      filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
      k1: float > 0. One of the SSIM dampening parameters.
      k2: float > 0. One of the SSIM dampening parameters.
    Returns:
      Each image's mean SSIM.
    """
    device = rgbs.device
    ori_shape = rgbs.size()
    width, height, num_channels = ori_shape[-3:]
    rgbs = rgbs.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    target_rgbs = target_rgbs.view(-1, width, height, num_channels).permute(0, 3, 1, 2)

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    filt_fn1 = lambda z: F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0], groups=num_channels)
    filt_fn2 = lambda z: F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw], groups=num_channels)

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(rgbs)
    mu1 = filt_fn(target_rgbs)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(rgbs ** 2) - mu00
    sigma11 = filt_fn(target_rgbs ** 2) - mu11
    sigma01 = filt_fn(rgbs * target_rgbs) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom

    return torch.mean(ssim_map.reshape([-1, num_channels * width * height]), dim=-1).item()


@dataclass
class PyNeRFBaseModelConfig(ModelConfig):

    near_plane: float = 0.2
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    geo_feat_dim: int = 15
    """output geo feat dimensions"""
    num_layers: int = 2
    """Number of layers in the base mlp"""
    num_layers_color: int = 3
    """Number of layers in the color mlp"""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 128
    """Dimension of hidden layers for color network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_resolution: int = 16
    """Base resolution of the hashmap for the base mlp."""
    max_resolution: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 20
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 4
    """Number of features per resolution level"""
    appearance_embedding_dim: int = 32
    """Whether to use average appearance embedding or zeros for inference."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    background_color: Literal['random', 'last_sample', 'black', 'white'] = 'last_sample'
    """Whether to randomize the background color."""
    output_interpolation: Literal['color', 'embedding'] = 'embedding'
    """Whether to interpolate between RGB outputs or density network embeddings."""
    level_interpolation: Literal['none', 'linear'] = 'linear'
    """How to interpolate between PyNeRF levels."""
    num_scales: int = 8
    """Number of levels in the PyNeRF hierarchy."""
    scale_factor: float = 2
    """Scale factor between levels in the PyNeRF hierarchy."""
    share_feature_grid: bool = True
    """Whether to share the same feature grid between levels."""

class PyNeRFBaseModel(Model):
    config: PyNeRFBaseModelConfig
    """
    PyNeRF base model.
    """

    def __init__(self, config: PyNeRFBaseModelConfig, metadata: Dict[str, Any], **kwargs) -> None:
        self.near = metadata.get('near', None)
        self.far = metadata.get('far', None)
        self.pose_scale_factor = metadata.get('pose_scale_factor', 1)
        if self.near is not None or self.far is not None:
            CONSOLE.log(
                f'Using near and far bounds {self.near} {self.far} from metadata')

        self.cameras = metadata.get('cameras', None)

        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        near = self.near if self.near is not None else (self.config.near_plane / self.pose_scale_factor)
        far = self.far if self.far is not None else (self.config.far_plane / self.pose_scale_factor)

        if self.config.disable_scene_contraction:
            self.scene_contraction = None
            self.collider = AABBBoxCollider(self.scene_box, near_plane=near)
        else:
            self.scene_contraction = SceneContraction(order=float('inf'))
            # Collider
            self.collider = NearFarCollider(near_plane=near, far_plane=far)

        self.field = PyNeRFField(
            self.scene_box.aabb,
            num_images=self.num_train_data,
            num_layers=self.config.num_layers,
            geo_feat_dim=self.config.geo_feat_dim,
            num_layers_color=self.config.num_layers_color,
            hidden_dim_color=self.config.hidden_dim_color,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            base_resolution=self.config.base_resolution,
            max_resolution=self.config.max_resolution,
            features_per_level=self.config.features_per_level,
            num_levels=self.config.num_levels,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=self.scene_contraction,
            output_interpolation=parse_output_interpolation(self.config.output_interpolation),
            level_interpolation=parse_level_interpolation(self.config.level_interpolation),
            num_scales=self.config.num_scales,
            scale_factor=self.config.scale_factor,
            share_feature_grid=self.config.share_feature_grid,
            cameras=self.cameras,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method='expected')
        self.renderer_level = SemanticRenderer()

        # losses
        self.rgb_loss = MSELoss(reduction='none')

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = ssim
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        mlps = []
        fields = []
        field_children = [self.field.named_children()]

        for children in field_children:
            for name, child in children:
                if 'mlp' in name:
                    mlps += child.parameters()
                else:
                    fields += child.parameters()

        param_groups = {
            'mlps': mlps,
            'fields': fields,
        }

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = self.get_outputs_inner(ray_bundle, None)
        if ray_bundle.metadata is not None and ray_bundle.metadata.get(RENDER_LEVELS, False):
            for i in range(self.field.min_trained_level, self.field.max_trained_level):
                level_outputs = self.get_outputs_inner(ray_bundle, i)
                outputs[f'rgb_level_{i}'] = level_outputs['rgb']
                outputs[f'depth_level_{i}'] = level_outputs['depth']

        outputs['directions_norm'] = ray_bundle.metadata['directions_norm']
        return outputs

    @abstractmethod
    def get_outputs_inner(self, ray_bundle: RayBundle, explicit_level: Optional[int] = None):
        pass

    def get_metrics_dict(self, outputs: Dict[str, any], batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        metrics_dict = {}
        image = batch['image'].to(self.device)
        metrics_dict['psnr'] = self.psnr(outputs['rgb'], image)

        if 'depth_image' in batch:
            metrics_dict['depth'] = F.mse_loss(outputs['depth'], batch['depth_image'] * outputs['directions_norm'])

        if self.training:
            for key, val in outputs[LEVEL_COUNTS].items():
                metrics_dict[f'{LEVEL_COUNTS}_{key}'] = val

        return metrics_dict

    def get_loss_dict(self, outputs: Dict[str, any], batch: Dict[str, any],
                      metrics_dict: Optional[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        loss_dict = self.get_loss_dict_inner(outputs, batch, metrics_dict)

        if self.training:
            for key, val in loss_dict.items():
                assert math.isfinite(val), f'Loss is not finite: {loss_dict}'

        return loss_dict

    def get_loss_dict_inner(self, outputs: Dict[str, any], batch: Dict[str, any],
                            metrics_dict: Optional[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        image = batch['image'].to(self.device)
        rgb_loss = self.rgb_loss(image, outputs['rgb'])

        if WEIGHT in batch:
            weights = batch[WEIGHT].to(self.device).view(-1, 1)
            rgb_loss *= weights

        loss_dict = {'rgb_loss': rgb_loss.mean()}

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch['image'].to(self.device)
        rgb = outputs['rgb']

        combined_rgb = torch.cat([image, rgb], dim=1)

        images_dict = {
            'img': combined_rgb,
        }

        acc = colormaps.apply_colormap(outputs['accumulation'])
        images_dict['accumulation'] = acc

        depth = colormaps.apply_depth_colormap(
            outputs['depth'],
            accumulation=outputs['accumulation'],
        )

        depth_vis = []
        if 'depth_image' in batch:
            depth_vis.append(colormaps.apply_depth_colormap(
                batch['depth_image'] * outputs['directions_norm'],
            ))

        depth_vis.append(depth)
        combined_depth = torch.cat(depth_vis, dim=1)

        images_dict['depth'] = combined_depth

        if not self.training:
            images_dict[LEVELS] = colormaps.apply_colormap(outputs[LEVELS] / self.config.num_levels,
                                                           colormap_options=ColormapOptions(colormap='turbo'))

        for i in range(self.config.num_levels):
            if f'rgb_level_{i}' in outputs:
                images_dict[f'rgb_level_{i}'] = torch.cat([image, outputs[f'rgb_level_{i}']], dim=1)
                images_dict[f'depth_level_{i}'] = colormaps.apply_depth_colormap(
                    outputs[f'depth_level_{i}'],
                    accumulation=outputs['accumulation'],
                )

        if 'mask' in batch:
            mask = batch['mask']
            assert torch.all(mask[:, mask.sum(dim=0) > 0])
            image = image[:, mask.sum(dim=0).squeeze() > 0]
            rgb = rgb[:, mask.sum(dim=0).squeeze() > 0]

        ssim = self.ssim(image, rgb)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        lpips = self.lpips(image, rgb)
        mse = np.exp(-0.1 * np.log(10.) * float(psnr.item()))
        dssim = np.sqrt((1 - float(ssim)) / 2)
        avg_error = np.exp(np.mean(np.log(np.array([mse, dssim, float(lpips)]))))

        # all of these metrics will be logged as scalars
        metrics_dict = {
            'psnr': float(psnr.item()),
            'ssim': float(ssim),
            'lpips': float(lpips),
            'avg_error': avg_error
        }  # type: ignore

        if WEIGHT in batch:
            weight = int(torch.unique(batch[WEIGHT]).item())
            for key, val in set(metrics_dict.items()):
                metrics_dict[f'{key}_{weight}'] = val
            for key, val in set(images_dict.items()):
                if 'level' not in key:
                    images_dict[f'{key}_{weight}'] = val

        return metrics_dict, images_dict

```

## pynerf/models

### pynerf_model.py

```python
from dataclasses import field, dataclass
from typing import Type, Tuple, List, Dict, Optional

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import distortion_loss, interlevel_loss, scale_gradients_by_distance_squared
from nerfstudio.model_components.ray_samplers import UniformSampler, ProposalNetworkSampler
from nerfstudio.utils import colormaps
from torch.nn import Parameter

from pynerf.models.pynerf_base_model import PyNeRFBaseModelConfig, PyNeRFBaseModel
from pynerf.pynerf_constants import EXPLICIT_LEVEL, PyNeRFFieldHeadNames, LEVEL_COUNTS, LEVELS


@dataclass
class PyNeRFModelConfig(PyNeRFBaseModelConfig):
    _target: Type = field(default_factory=lambda: PyNeRFModel)

    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""

class PyNeRFModel(PyNeRFBaseModel):
    config: PyNeRFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()

        proposal_net_args_list = []
        levels = [6, 8]
        max_res = [512, 2048]
        for i in range(num_prop_nets):
            proposal_net_args_list.append({
                'hidden_dim': 16,
                'log2_hashmap_size': 19,
                'num_levels': levels[i],
                'base_res': self.config.base_resolution,
                'max_res': max_res[i],
                'use_linear': False
            })

        if self.config.use_same_proposal_network:
            assert len(proposal_net_args_list) == 1, 'Only one proposal network is allowed.'
            prop_net_args = proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=self.scene_contraction,
                                          **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = proposal_net_args_list[min(i, len(proposal_net_args_list) - 1)]
                network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=self.scene_contraction,
                                              **prop_net_args)
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.scene_contraction is None:
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups['proposal_networks'] = list(self.proposal_networks.parameters())
        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs_inner(self, ray_bundle: RayBundle, explicit_level: Optional[int] = None):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        if explicit_level is not None:
            if ray_samples.metadata is None:
                ray_samples.metadata = {}
            ray_samples.metadata[EXPLICIT_LEVEL] = explicit_level

        field_outputs = self.field(ray_samples)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        outputs = {
            'rgb': self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights),
            'depth': self.renderer_depth(weights=weights, ray_samples=ray_samples)
        }

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        if self.training:
            outputs['weights_list'] = weights_list
            outputs['ray_samples_list'] = ray_samples_list

        if explicit_level is None:
            for i in range(self.config.num_proposal_iterations):
                outputs[f'prop_depth_{i}'] = self.renderer_depth(weights=weights_list[i],
                                                                 ray_samples=ray_samples_list[i])

            outputs['accumulation'] = self.renderer_accumulation(weights=weights)

            if self.training:
                outputs[LEVEL_COUNTS] = field_outputs[PyNeRFFieldHeadNames.LEVEL_COUNTS]
            else:
                levels = field_outputs[PyNeRFFieldHeadNames.LEVELS]
                outputs[LEVELS] = self.renderer_level(weights=weights,
                                                        semantics=levels.clamp(0, self.field.num_scales - 1))

        return outputs

    def get_metrics_dict(self, outputs: Dict[str, any], batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)

        if self.training:
            metrics_dict['distortion'] = distortion_loss(outputs['weights_list'], outputs['ray_samples_list'])
            metrics_dict['interlevel'] = interlevel_loss(outputs['weights_list'], outputs['ray_samples_list'])

        return metrics_dict

    def get_loss_dict_inner(self, outputs: Dict[str, any], batch: Dict[str, any],
                            metrics_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict_inner(outputs, batch, metrics_dict)

        if self.training:
            loss_dict['interlevel_loss'] = self.config.interlevel_loss_mult * metrics_dict['interlevel']
            loss_dict['distortion_loss'] = self.config.distortion_loss_mult * metrics_dict['distortion']

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        for i in range(self.config.num_proposal_iterations):
            key = f'prop_depth_{i}'
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs['accumulation'],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

```

## pynerf/data

### __init__.py

```python

```

## pynerf/data

### weighted_pixel_sampler.py

```python
from typing import Dict

import torch
from nerfstudio.data.pixel_samplers import PixelSampler

from pynerf.pynerf_constants import WEIGHT, TRAIN_INDEX


class WeightedPixelSampler(PixelSampler):

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []
        all_weights = []
        all_train_indices = []

        if "mask" in batch:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape

                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                # TODO(hturki): Need to add unsqueeze(0) so that indices shape is 3 - is this a bug in Nerfstudio?
                indices = self.sample_method(
                    num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i].unsqueeze(0), device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

                if WEIGHT in batch:
                    all_weights.append(batch[WEIGHT][i][indices[:, 1], indices[:, 2]])
                if TRAIN_INDEX in batch:
                    all_train_indices.append(batch[TRAIN_INDEX][i][indices[:, 1], indices[:, 2]])
        else:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
                if WEIGHT in batch:
                    all_weights.append(batch[WEIGHT][i][indices[:, 1], indices[:, 2]])
                if TRAIN_INDEX in batch:
                    all_train_indices.append(batch[TRAIN_INDEX][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key not in {"image_idx", "image", "mask", WEIGHT, TRAIN_INDEX} and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        if WEIGHT in batch:
            collated_batch[WEIGHT] = torch.cat(all_weights, dim=0)
            assert collated_batch[WEIGHT].shape == (num_rays_per_batch,), collated_batch[WEIGHT].shape

        if TRAIN_INDEX in batch:
            collated_batch[TRAIN_INDEX] = torch.cat(all_train_indices, dim=0)
            assert collated_batch[TRAIN_INDEX].shape == (num_rays_per_batch,), collated_batch[TRAIN_INDEX].shape

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

```

## pynerf/data

### weighted_fixed_indices_eval_loader.py

```python
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader

from pynerf.pynerf_constants import TRAIN_INDEX, WEIGHT


class WeightedFixedIndicesEvalDataloader(FixedIndicesEvalDataloader):

    def __next__(self):
        ray_bundle, batch = super().__next__()
        metadata = self.input_dataset._dataparser_outputs.metadata

        camera_indices = ray_bundle.camera_indices
        if TRAIN_INDEX in metadata:
            if ray_bundle.metadata is None:
                ray_bundle.metadata = {}

            ray_bundle.metadata[TRAIN_INDEX] = metadata[TRAIN_INDEX].to(camera_indices.device)[camera_indices]

        if WEIGHT in metadata:
            batch[WEIGHT] = metadata[WEIGHT].to(camera_indices.device)[camera_indices]

        return ray_bundle, batch

```

## pynerf/data

### pynerf_pipelines.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Optional, Type, Dict

import torch
from PIL import Image
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
)
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline, DynamicBatchPipelineConfig
from nerfstudio.utils import profiler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


def get_average_eval_image_metrics(datamanager: DataManager, model: Model, step: Optional[int] = None,
                                   output_path: Optional[Path] = None, get_std: bool = False) -> Dict:
    """Same as in VanillaPipeline but removes the isinstance(self.datamanager, VanillaDataManager) assertion to handle
    RandomSubsetDataManager and can also handle the case where not every metrics_dict has the same keys (which is the
    case for metrics such as psnr_1.0)"""

    metrics_dict_list = []
    num_images = len(datamanager.fixed_indices_eval_dataloader)
    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
    ) as progress:
        task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
        for camera_ray_bundle, batch in datamanager.fixed_indices_eval_dataloader:
            # time this the following line
            inner_start = time()
            height, width = camera_ray_bundle.shape
            num_rays = height * width
            outputs = model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            metrics_dict, images_dict = model.get_image_metrics_and_images(outputs, batch)

            if output_path is not None:
                camera_indices = camera_ray_bundle.camera_indices
                assert camera_indices is not None
                for key, val in images_dict.items():
                    Image.fromarray((val * 255).byte().cpu().numpy()).save(
                        output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                    )
            assert "num_rays_per_sec" not in metrics_dict
            metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
            fps_str = "fps"
            assert fps_str not in metrics_dict
            metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
            metrics_dict_list.append(metrics_dict)
            progress.advance(task)
    # average the metrics list
    metrics_dict = {}
    metric_keys = set()
    for metrics_dict in metrics_dict_list:
        metric_keys.update(metrics_dict.keys())
    for key in metric_keys:
        if get_std:
            key_std, key_mean = torch.std_mean(
                torch.tensor([metrics_dict[key] for metrics_dict in filter(lambda x: key in x, metrics_dict_list)])
            )
            metrics_dict[key] = float(key_mean)
            metrics_dict[f"{key}_std"] = float(key_std)
        else:
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in
                                         filter(lambda x: key in x, metrics_dict_list)]))
            )

    return metrics_dict


@dataclass
class PyNeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: PyNeRFPipeline)
    """target class to instantiate"""


class PyNeRFPipeline(VanillaPipeline):

    @profiler.time_function
    def get_average_eval_image_metrics(
            self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        self.eval()
        metrics_dict = get_average_eval_image_metrics(self.datamanager, self.model, step, output_path, get_std)
        self.train()

        return metrics_dict


@dataclass
class PyNeRFDynamicBatchPipelineConfig(DynamicBatchPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: PyNeRFDynamicBatchPipeline)


class PyNeRFDynamicBatchPipeline(DynamicBatchPipeline):

    @profiler.time_function
    def get_average_eval_image_metrics(
            self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        self.eval()
        metrics_dict = get_average_eval_image_metrics(self.datamanager, self.model, step, output_path, get_std)
        self.train()

        return metrics_dict

```

## pynerf/data/datamanagers

### random_subset_datamanager.py

```python
import random
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union, Literal

import torch
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.comms import get_rank, get_world_size
from rich.console import Console
from torch.nn import Parameter
from torch.utils.data import DistributedSampler, DataLoader

from pynerf.data.dataparsers.multicam_dataparser import MulticamDataParserConfig
from pynerf.data.datasets.image_metadata import ImageMetadata
from pynerf.data.datasets.random_subset_dataset import RandomSubsetDataset
from pynerf.data.weighted_fixed_indices_eval_loader import WeightedFixedIndicesEvalDataloader
from pynerf.pynerf_constants import RGB, WEIGHT, TRAIN_INDEX, DEPTH, POSE_SCALE_FACTOR, RAY_INDEX, RENDER_LEVELS

CONSOLE = Console(width=120)


@dataclass
class RandomSubsetDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: RandomSubsetDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = MulticamDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 4096
    """Number of rays per batch to use per training iteration."""
    eval_num_rays_per_batch: int = 8192
    """Number of rays per batch to use per eval iteration."""
    eval_image_indices: Optional[Tuple[int, ...]] = None
    """Specifies the image indices to use during eval; if None, uses all val images."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(
        optimizer=AdamOptimizerConfig(lr=6e-6, eps=1e-15),
        scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-4, max_steps=125000))
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    items_per_chunk: int = 25600000
    """Number of entries to load into memory at a time"""
    local_cache_path: Optional[str] = "scratch/pynerf-cache"
    """Caches images and metadata in specific path if set."""
    on_demand_threads: int = 16
    """Number of threads to use when reading data"""
    load_all_in_memory: bool = False
    """Load all of the dataset in memory vs sampling from disk"""


class RandomSubsetDataManager(DataManager):
    """Data manager implementation that samples batches of random pixels/rays/metadata in a chunked manner.
    It can handle datasets that are larger than what can be held in memory

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: RandomSubsetDataManagerConfig

    train_dataset: InputDataset
    """Used by the viewer and in various checks in the trainer, but is not actually used to sample batches"""

    def __init__(
            self,
            config: RandomSubsetDataManagerConfig,
            device: Union[torch.device, str] = "cpu",
            test_mode: Literal["test", "val", "inference"] = "test",
            world_size: int = 1,
            local_rank: int = 0,
    ):
        self.test_mode = test_mode # Needed for parent class
        super().__init__()

        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data

        dataparser = self.config.dataparser.setup()
        self.includes_time = dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = dataparser.get_dataparser_outputs(split="train")

        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataparser_outputs.cameras.size, device=self.device)
        self.train_ray_generator = RayGenerator(self.train_dataparser_outputs.cameras.to(self.device),
                                                self.train_camera_optimizer)

        fields_to_load = {RGB}
        for additional_field in {DEPTH, WEIGHT, TRAIN_INDEX}:
            if additional_field in self.train_dataparser_outputs.metadata:
                fields_to_load.add(additional_field)

        self.train_batch_dataset = RandomSubsetDataset(
            items=self._get_image_metadata(self.train_dataparser_outputs),
            fields_to_load=fields_to_load,
            on_demand_threads=self.config.on_demand_threads,
            items_per_chunk=self.config.items_per_chunk,
            load_all_in_memory=self.config.load_all_in_memory,
        )

        self.iter_train_image_dataloader = iter([])
        self.train_dataset = InputDataset(self.train_dataparser_outputs)

        self.eval_dataparser_outputs = dataparser.get_dataparser_outputs(split='test')  # test_mode)

        self.eval_dataset = InputDataset(self.eval_dataparser_outputs)
        self.eval_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.eval_dataparser_outputs.cameras.size, device=self.device)
        self.eval_ray_generator = RayGenerator(self.eval_dataparser_outputs.cameras.to(self.device),
                                               self.eval_camera_optimizer)

        self.eval_image_metadata = self._get_image_metadata(self.eval_dataparser_outputs)
        self.eval_batch_dataset = RandomSubsetDataset(
            items=self.eval_image_metadata,
            fields_to_load=fields_to_load,
            on_demand_threads=self.config.on_demand_threads,
            items_per_chunk=(self.config.eval_num_rays_per_batch * 10),
            load_all_in_memory=self.config.load_all_in_memory
        )

        self.iter_eval_batch_dataloader = iter([])

    @cached_property
    def fixed_indices_eval_dataloader(self):
        image_indices = []
        for item_index in range(get_rank(), len(self.eval_dataparser_outputs.cameras), get_world_size()):
            image_indices.append(item_index)

        return WeightedFixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
            image_indices=image_indices
        )

    def _set_train_loader(self):
        batch_size = self.config.train_num_rays_per_batch // self.world_size
        if self.world_size > 0:
            self.train_sampler = DistributedSampler(self.train_batch_dataset, self.world_size, self.local_rank)
            assert self.config.train_num_rays_per_batch % self.world_size == 0
            self.train_image_dataloader = DataLoader(self.train_batch_dataset, batch_size=batch_size,
                                                     sampler=self.train_sampler, num_workers=0, pin_memory=True)
        else:
            self.train_image_dataloader = DataLoader(self.train_batch_dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=0, pin_memory=True)

        self.iter_train_image_dataloader = iter(self.train_image_dataloader)

    def _set_eval_batch_loader(self):
        batch_size = self.config.eval_num_rays_per_batch // self.world_size
        if self.world_size > 0:
            self.eval_sampler = DistributedSampler(self.eval_batch_dataset, self.world_size, self.local_rank)
            assert self.config.eval_num_rays_per_batch % self.world_size == 0
            self.eval_batch_dataloader = DataLoader(self.eval_batch_dataset, batch_size=batch_size,
                                                    sampler=self.eval_sampler,
                                                    num_workers=0, pin_memory=True)
        else:
            self.eval_batch_dataloader = DataLoader(self.eval_batch_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=0, pin_memory=True)

        self.iter_eval_batch_dataloader = iter(self.eval_batch_dataloader)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        batch = next(self.iter_train_image_dataloader, None)
        if batch is None:
            self.train_batch_dataset.load_chunk()
            self._set_train_loader()
            batch = next(self.iter_train_image_dataloader)

        ray_bundle = self.train_ray_generator(batch[RAY_INDEX])
        self.transfer_train_index(ray_bundle, batch)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        batch = next(self.iter_eval_batch_dataloader, None)
        if batch is None:
            self.eval_batch_dataset.load_chunk()
            self._set_eval_batch_loader()
            batch = next(self.iter_eval_batch_dataloader)

        ray_bundle = self.eval_ray_generator(batch[RAY_INDEX])
        self.transfer_train_index(ray_bundle, batch)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        image_index = random.choice(self.fixed_indices_eval_dataloader.image_indices)
        ray_bundle, batch = self.fixed_indices_eval_dataloader.get_data_from_image_idx(image_index)

        metadata = self.eval_image_metadata[image_index]

        if ray_bundle.metadata is None:
            ray_bundle.metadata = {}
            ray_bundle.metadata[RENDER_LEVELS] = True

        if metadata.train_index is not None:
            ray_bundle.metadata[TRAIN_INDEX] = torch.full_like(ray_bundle.camera_indices, metadata.train_index,
                                                               dtype=torch.int64)
        if metadata.weight is not None:
            batch[WEIGHT] = torch.full_like(ray_bundle.camera_indices, metadata.weight, dtype=torch.float32)

        if metadata.depth_path is not None:
            batch[DEPTH] = metadata.load_depth().to(ray_bundle.camera_indices.device).unsqueeze(-1)

        return image_index, ray_bundle, batch

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups

    def _get_image_metadata(self, outputs: DataparserOutputs) -> List[ImageMetadata]:
        local_cache_path = Path(self.config.local_cache_path) if self.config.local_cache_path is not None else None

        items = []
        for i in range(len(outputs.image_filenames)):
            items.append(
                ImageMetadata(str(outputs.image_filenames[i]),
                              int(outputs.cameras.width[i]),
                              int(outputs.cameras.height[i]),
                              outputs.metadata[DEPTH][i] if DEPTH in outputs.metadata else None,
                              str(outputs.mask_filenames[i]) if outputs.mask_filenames is not None else None,
                              float(outputs.metadata[WEIGHT][i]) if WEIGHT in outputs.metadata else None,
                              int(outputs.metadata[TRAIN_INDEX][i]) if TRAIN_INDEX in outputs.metadata else None,
                              outputs.metadata[POSE_SCALE_FACTOR] if POSE_SCALE_FACTOR in outputs.metadata else 1,
                              local_cache_path))

        return items

    @staticmethod
    def transfer_train_index(ray_bundle: RAY_INDEX, batch: Dict) -> None:
        if TRAIN_INDEX in batch:
            if ray_bundle.metadata is None:
                ray_bundle.metadata = {}
            ray_bundle.metadata[TRAIN_INDEX] = batch[TRAIN_INDEX].unsqueeze(-1).to(ray_bundle.origins.device)
            del batch[TRAIN_INDEX]

```

## pynerf/data/datamanagers

### __init__.py

```python

```

## pynerf/data/datamanagers

### weighted_datamanager.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Type, Tuple, Dict,
)

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager
from nerfstudio.data.pixel_samplers import (
    PixelSamplerConfig,
)

from pynerf.data.datamanagers.random_subset_datamanager import RandomSubsetDataManager
from pynerf.data.datasets.weighted_dataset import WeightedDataset
from pynerf.data.weighted_pixel_sampler import WeightedPixelSampler
from pynerf.pynerf_constants import RENDER_LEVELS


@dataclass
class WeightedPixelSamplerConfig(PixelSamplerConfig):
    _target: Type = field(default_factory=lambda: WeightedPixelSampler)

@dataclass
class WeightedDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: WeightedDataManager)

    pixel_sampler: WeightedPixelSamplerConfig = field(default_factory=lambda: WeightedPixelSamplerConfig())


class WeightedDataManager(VanillaDataManager[WeightedDataset]):

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_train(step)
        RandomSubsetDataManager.transfer_train_index(ray_bundle, batch)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_eval(step)
        RandomSubsetDataManager.transfer_train_index(ray_bundle, batch)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        image_index, ray_bundle, batch = super().next_eval_image(step)
        RandomSubsetDataManager.transfer_train_index(ray_bundle, batch)

        if ray_bundle.metadata is None:
            ray_bundle.metadata = {}
            ray_bundle.metadata[RENDER_LEVELS] = True

        return image_index, ray_bundle, batch

```

## pynerf/data/dataparsers

### adop_dataparser.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List, Optional

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from pyquaternion import Quaternion

from pynerf.data.dataparsers.mipnerf_dataparser import write_mask
from pynerf.pynerf_constants import TRAIN_INDEX, WEIGHT

OPENCV_TO_OPENGL = torch.DoubleTensor([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]])

DOWN_TO_FORWARD = torch.DoubleTensor([[1, 0, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, -1, 0, 0],
                                      [0, 0, 0, 1]])


@dataclass
class AdopDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: Adop)
    """target class to instantiate"""
    data: Path = Path("data/adop/boat")

    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""

    train_split: float = 0.9

    scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])


@dataclass
class Adop(DataParser):
    config: AdopDataParserConfig

    def __init__(self, config: AdopDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor

    def get_dataparser_outputs(self, split="train", scales: Optional[List[int]] = None):
        with (self.config.data / "images.txt").open() as f:
            image_paths = f.readlines()

        with (self.config.data / "adop-poses.txt").open() as f:
            poses = f.readlines()

        with (self.config.data / "undistorted_intrinsics_adop.txt").open() as f:
            intrinsics = f.readlines()

        num_images_base = len(image_paths)
        assert num_images_base == len(poses) == len(intrinsics)

        image_filenames = []
        c2ws = []
        width = []
        height = []
        fx = []
        fy = []
        cx = []
        cy = []
        weights = []
        img_scales = []

        near = 1e10
        far = -1

        if scales is None:
            scales = self.config.scales
        for scale in scales:
            for image_path, c2w_line, K in zip(image_paths, poses, intrinsics):
                image_filenames.append(self.data / f'undistorted_images_adop-{scale}' / image_path.strip())

                pose_line = [float(x) for x in c2w_line.strip().split()]
                w2c = torch.DoubleTensor(
                    Quaternion(w=pose_line[3], x=pose_line[0], y=pose_line[1], z=pose_line[2]).transformation_matrix)
                w2c[:3, 3] = torch.DoubleTensor(pose_line[4:7])

                # Some points seem to be extremely close
                # if near > 0.1:
                near = min(pose_line[7], near)
                far = max(pose_line[8], far)
                c2w = torch.inverse(w2c)

                c2ws.append((DOWN_TO_FORWARD @ (c2w @ OPENCV_TO_OPENGL))[:3].unsqueeze(0))

                K_line = [float(x) for x in K.strip().split()]
                width.append(int(K_line[0]) // scale)
                height.append(int(K_line[1]) // scale)
                fx.append(K_line[2] / scale)
                fy.append(K_line[6] / scale)
                cx.append(K_line[4] / scale)
                cy.append(K_line[7] / scale)
                weights.append(scale ** 2)
                img_scales.append(scale)

        c2ws = torch.cat(c2ws)
        min_bounds = c2ws[:, :, 3].min(dim=0)[0]
        max_bounds = c2ws[:, :, 3].max(dim=0)[0]

        origin = (max_bounds + min_bounds) * 0.5
        print('Calculated origin: {} {} {}'.format(origin, min_bounds, max_bounds))

        pose_scale_factor = ((max_bounds - min_bounds) * 0.5).norm().item()
        print('Calculated pose scale factor: {}'.format(pose_scale_factor))

        for c2w in c2ws:
            c2w[:, 3] = (c2w[:, 3] - origin) / pose_scale_factor
            assert torch.logical_and(c2w >= -1, c2w <= 1).all(), c2w

        # in x,y,z order
        c2ws[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=((torch.stack([min_bounds, max_bounds]) - origin) / pose_scale_factor).float())

        train_indices = set()
        base_train_indices = np.linspace(0, num_images_base, int(num_images_base * self.config.train_split),
                                         endpoint=False, dtype=np.int32)
        for i in range(len(scales)):
            train_indices.update(base_train_indices + num_images_base * i)

        if split.casefold() == 'train':
            mask_filenames = []
            for i in range(len(image_filenames)):
                if i in train_indices:
                    mask_filenames.append(self.data / f'image_full-{img_scales[i]}.png')
                    if not mask_filenames[-1].exists():
                        write_mask(mask_filenames[-1], w=width[i], h=height[i], left_only=False, right_only=False)
                else:
                    mask_filenames.append(self.data / f'image_left-{img_scales[i]}.png')
                    if not mask_filenames[-1].exists():
                        write_mask(mask_filenames[-1], w=width[i], h=height[i], left_only=True, right_only=False)

            indices = torch.arange(len(image_filenames), dtype=torch.long)
        else:
            val_indices = []
            mask_filenames = []
            train_indices = set(train_indices)
            for i in range(len(image_filenames)):
                if i not in train_indices:
                    val_indices.append(i)
                    mask_filenames.append(self.data / f'image_right-{img_scales[i]}.png')
                    if not mask_filenames[-1].exists():
                        write_mask(mask_filenames[-1], w=width[i], h=height[i], left_only=False, right_only=True)

            indices = torch.LongTensor(val_indices)

        cameras = Cameras(
            camera_to_worlds=c2ws[indices].float(),
            fx=torch.FloatTensor(fx)[indices],
            fy=torch.FloatTensor(fy)[indices],
            cx=torch.FloatTensor(cx)[indices],
            cy=torch.FloatTensor(cy)[indices],
            width=torch.IntTensor(width)[indices],
            height=torch.IntTensor(height)[indices],
            camera_type=CameraType.PERSPECTIVE,
        )

        print('Num images in split {}: {}'.format(split, len(indices)))

        embedding_indices = torch.arange(num_images_base).unsqueeze(0).repeat(len(scales), 1).view(-1)

        metadata = {
            TRAIN_INDEX: embedding_indices[indices],
            WEIGHT: torch.FloatTensor(weights)[indices],
            "pose_scale_factor": pose_scale_factor,
            "cameras": cameras,
            'near': near / pose_scale_factor,
            'far': far * 10 / pose_scale_factor
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=[image_filenames[i] for i in indices],
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            mask_filenames=mask_filenames,
            metadata=metadata
        )

        return dataparser_outputs

```

## pynerf/data/dataparsers

### multicam_dataparser.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json

from pynerf.pynerf_constants import WEIGHT, DEPTH, POSE_SCALE_FACTOR


@dataclass
class MulticamDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: Multicam)
    """target class to instantiate"""
    data: Path = Path('data/multicam/lego')
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: Optional[str] = 'white'
    """alpha color of background"""


@dataclass
class Multicam(DataParser):
    config: MulticamDataParserConfig

    def __init__(self, config: MulticamDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split='train'):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        base_meta = load_from_json(self.data / 'metadata.json')
        meta = base_meta[split]
        image_filenames = []
        poses = []
        width = []
        height = []
        focal_length = []
        cx = []
        cy = []
        weights = []
        depth_images = []

        for i in range(len(meta['file_path'])):
            image_filenames.append(self.data / meta['file_path'][i])
            poses.append(np.array(meta['cam2world'][i])[:3])
            width.append(meta['width'][i])
            height.append(meta['height'][i])
            focal_length.append(meta['focal'][i])
            cx.append(meta['width'][i] / 2.0)
            cy.append(meta['height'][i] / 2.0)
            weights.append(meta['lossmult'][i])
            if 'depth_path' in meta:
                depth_images.append(self.data / meta['depth_path'][i])

        poses = np.array(poses).astype(np.float32)
        camera_to_world = torch.from_numpy(poses[:, :3])

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        if 'scene_bounds' in base_meta:
            bounds = torch.FloatTensor(base_meta['scene_bounds'])
        else:
            radius = 1.3 if "ship" not in str(self.data) else 1.5
            bounds = torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]], dtype=torch.float32)

        scene_box = SceneBox(aabb=bounds)

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=torch.FloatTensor(focal_length),
            fy=torch.FloatTensor(focal_length),
            cx=torch.FloatTensor(cx),
            cy=torch.FloatTensor(cy),
            width=torch.IntTensor(width),
            height=torch.IntTensor(height),
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {WEIGHT: weights, 'cameras': cameras, 'near': meta['near'][0], 'far': meta['far'][0]}
        if len(depth_images) > 0:
            metadata[DEPTH] = depth_images
            metadata[POSE_SCALE_FACTOR] = base_meta[POSE_SCALE_FACTOR]

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata=metadata
        )

        return dataparser_outputs

```

## pynerf/data/dataparsers

### __init__.py

```python

```

## pynerf/data/dataparsers

### mipnerf_dataparser.py

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List

import imageio
import numpy as np
import torch
from PIL import Image
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary
from rich.console import Console

from pynerf.pynerf_constants import WEIGHT, TRAIN_INDEX

CONSOLE = Console(width=120)


def write_mask(dest: Path, w: int, h: int, left_only: bool, right_only: bool) -> None:
    mask = torch.ones(int(h), int(w), dtype=torch.bool)
    if left_only:
        assert not right_only
        mask[:, w // 2:] = False
    if right_only:
        assert not left_only
        mask[:, :w // 2] = False

    tmp_path = dest.parent / f'{uuid.uuid4()}{dest.suffix}'
    Image.fromarray(mask.numpy()).save(tmp_path)
    tmp_path.rename(dest)
    CONSOLE.log(f'Wrote new mask file to {dest}')


@dataclass
class MipNerf360DataParserConfig(DataParserConfig):
    """Mipnerf 360 dataset parser config"""

    _target: Type = field(default_factory=lambda: Mipnerf360)
    """target class to instantiate"""
    data: Path = Path("data/mipnerf360/garden")
    """Directory specifying location of data."""
    scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    """How much to downscale images."""
    val_skip: int = 8
    """1/val_skip images to use for validation."""
    auto_scale: bool = True
    """Scale based on pose bounds."""
    aabb_scale: float = 1
    """Scene scale."""
    train_split: float = 7 / 8


@dataclass
class Mipnerf360(DataParser):
    """MipNeRF 360 Dataset"""

    config: MipNerf360DataParserConfig

    @classmethod
    def normalize_orientation(cls, poses: np.ndarray):
        """Set the _up_ direction to be in the positive Y direction.
        Args:
            poses: Numpy array of poses.
        """
        poses_orig = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
        center = poses[:, :3, 3].mean(0)
        vec2 = poses[:, :3, 2].sum(0) / np.linalg.norm(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        vec0 = np.cross(up, vec2) / np.linalg.norm(np.cross(up, vec2))
        vec1 = np.cross(vec2, vec0) / np.linalg.norm(np.cross(vec2, vec0))
        c2w = np.stack([vec0, vec1, vec2, center], -1)  # [3, 4]
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)  # [4, 4]
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])  # [BS, 1, 4]
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # [BS, 4, 4]
        poses = np.linalg.inv(c2w) @ poses
        poses_orig[:, :3, :4] = poses[:, :3, :4]
        return poses_orig

    def _generate_dataparser_outputs(self, split='train'):
        fx = []
        fy = []
        cx = []
        cy = []
        c2ws = []
        width = []
        height = []
        weights = []
        img_scales = []
        image_filenames = []

        camera_params = read_cameras_binary(self.config.data / 'sparse/0/cameras.bin')
        assert camera_params[1].model == 'PINHOLE'
        camera_fx, camera_fy, camera_cx, camera_cy = camera_params[1].params

        for scale in self.config.scales:
            image_dir = "images"
            if scale > 1:
                image_dir += f"_{scale}"
            image_dir = self.config.data / image_dir
            if not image_dir.exists():
                raise ValueError(f"Image directory {image_dir} doesn't exist")

            valid_formats = ['.jpg', '.png']
            num_images = 0
            for f in sorted(image_dir.iterdir()):
                ext = f.suffix
                if ext.lower() not in valid_formats:
                    continue
                image_filenames.append(f)
                num_images += 1

            poses_data = np.load(self.config.data / 'poses_bounds.npy')
            poses = poses_data[:, :-2].reshape([-1, 3, 5]).astype(np.float32)
            bounds = poses_data[:, -2:].transpose([1, 0])

            if num_images != poses.shape[0]:
                raise RuntimeError(f'Different number of images ({num_images}), and poses ({poses.shape[0]})')

            img_0 = imageio.imread(image_filenames[-1])
            image_height, image_width = img_0.shape[:2]

            width.append(torch.full((num_images, 1), image_width, dtype=torch.long))
            height.append(torch.full((num_images, 1), image_height, dtype=torch.long))
            fx.append(torch.full((num_images, 1), camera_fx / scale))
            fy.append(torch.full((num_images, 1), camera_fy / scale))
            cx.append(torch.full((num_images, 1), camera_cx / scale))
            cy.append(torch.full((num_images, 1), camera_cy / scale))
            weights.append(torch.full((num_images,), scale ** 2))
            img_scales.append(torch.full((num_images,), scale, dtype=torch.long))

            # Reorder pose to match nerfstudio convention
            poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], axis=-1)

            # Center poses and rotate. (Compute up from average of all poses)
            poses = self.normalize_orientation(poses)

            # Scale factor used in mipnerf
            if self.config.auto_scale:
                scale_factor = 1 / (np.min(bounds) * 0.75)
                poses[:, :3, 3] *= scale_factor
                bounds *= scale_factor

            # Center poses
            poses[:, :3, 3] = poses[:, :3, 3] - np.mean(poses[:, :3, :], axis=0)[:, 3]
            c2ws.append(torch.from_numpy(poses[:, :3, :4]))

        c2ws = torch.cat(c2ws)
        min_bounds = c2ws[:, :, 3].min(dim=0)[0]
        max_bounds = c2ws[:, :, 3].max(dim=0)[0]

        origin = (max_bounds + min_bounds) * 0.5
        CONSOLE.log('Calculated origin: {} {} {}'.format(origin, min_bounds, max_bounds))

        pose_scale_factor = ((max_bounds - min_bounds) * 0.5).norm().item()
        CONSOLE.log('Calculated pose scale factor: {}'.format(pose_scale_factor))

        for c2w in c2ws:
            c2w[:, 3] = (c2w[:, 3] - origin) / pose_scale_factor
            assert torch.logical_and(c2w >= -1, c2w <= 1).all(), c2w

        # in x,y,z order
        scene_box = SceneBox(aabb=((torch.stack([min_bounds, max_bounds]) - origin) / pose_scale_factor).float())

        train_indices = set()
        base_train_indices = np.linspace(0, num_images, int(num_images * self.config.train_split),
                                         endpoint=False, dtype=np.int32)
        for i in range(len(self.config.scales)):
            train_indices.update(base_train_indices + num_images * i)

        img_scales = torch.cat(img_scales)
        width = torch.cat(width)
        height = torch.cat(height)

        if split.casefold() == 'train':
            mask_filenames = []
            for i in range(len(image_filenames)):
                if i in train_indices:
                    mask_filenames.append(self.config.data / f'image_full-{img_scales[i].item()}.png')
                    if not mask_filenames[-1].exists():
                        write_mask(mask_filenames[-1], w=width[i], h=height[i], left_only=False, right_only=False)

                else:
                    mask_filenames.append(self.config.data / f'image_left-{img_scales[i].item()}.png')
                    if not mask_filenames[-1].exists():
                        write_mask(mask_filenames[-1], w=width[i], h=height[i], left_only=True, right_only=False)

            indices = torch.arange(len(image_filenames), dtype=torch.long)
        else:
            val_indices = []
            mask_filenames = []
            train_indices = set(train_indices)
            for i in range(len(image_filenames)):
                if i not in train_indices:
                    val_indices.append(i)
                    mask_filenames.append(self.config.data / f'image_right-{img_scales[i].item()}.png')
                    if not mask_filenames[-1].exists():
                        write_mask(mask_filenames[-1], w=width[i], h=height[i], left_only=False, right_only=True)

            indices = torch.LongTensor(val_indices)

        cameras = Cameras(
            camera_to_worlds=c2ws[indices].float(),
            fx=torch.cat(fx)[indices],
            fy=torch.cat(fy)[indices],
            cx=torch.cat(cx)[indices],
            cy=torch.cat(cy)[indices],
            width=width[indices],
            height=height[indices],
            camera_type=CameraType.PERSPECTIVE,
        )

        CONSOLE.log('Num images in split {}: {}'.format(split, len(indices)))

        embedding_indices = torch.arange(num_images).unsqueeze(0).repeat(len(self.config.scales), 1).view(-1)

        metadata = {
            TRAIN_INDEX: embedding_indices[indices],
            WEIGHT: torch.cat(weights)[indices],
            'pose_scale_factor': pose_scale_factor,
            'cameras': cameras,
            'near': bounds.min() / pose_scale_factor,
            'far': 10 * bounds.max() / pose_scale_factor
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=[image_filenames[i] for i in indices],
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=1,
            mask_filenames=mask_filenames,
            metadata=metadata
        )

        return dataparser_outputs

```

## pynerf/data/datasets

### random_subset_dataset.py

```python
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set, List

import torch
from rich.console import Console
from torch.utils.data import Dataset

from pynerf.data.datasets.image_metadata import ImageMetadata
from pynerf.pynerf_constants import RGB, RAY_INDEX, WEIGHT, TRAIN_INDEX, DEPTH

CONSOLE = Console(width=120)

PIXEL_INDEX = 'pixel_index'
IMAGE_INDEX = 'image_index'
MASK = 'mask'


class RandomSubsetDataset(Dataset):

    def __init__(self,
                 items: List[ImageMetadata],
                 fields_to_load: Set[str],
                 on_demand_threads: int,
                 items_per_chunk: int,
                 load_all_in_memory: bool):
        super(RandomSubsetDataset, self).__init__()

        self.items = items
        self.fields_to_load = fields_to_load
        self.items_per_chunk = items_per_chunk
        self.load_all_in_memory = load_all_in_memory

        self.chunk_load_executor = ThreadPoolExecutor(max_workers=1)

        if self.load_all_in_memory:
            self.memory_fields = None
        else:
            pixel_indices_to_sample = []
            for item in items:
                pixel_indices_to_sample.append(item.W * item.H)

            self.pixel_indices_to_sample = torch.LongTensor(pixel_indices_to_sample)
            assert len(self.pixel_indices_to_sample) > 0
            self.on_demand_executor = ThreadPoolExecutor(max_workers=on_demand_threads)

        self.loaded_fields = None
        self.loaded_field_offset = 0
        self.chunk_future = None
        self.loaded_chunk = None

    def load_chunk(self) -> None:
        if self.chunk_future is None:
            self.chunk_future = self.chunk_load_executor.submit(self._load_chunk_inner)

        self.loaded_chunk = self.chunk_future.result()
        self.chunk_future = self.chunk_load_executor.submit(self._load_chunk_inner)

    def __len__(self) -> int:
        return self.loaded_chunk[RGB].shape[0] if self.loaded_chunk is not None else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {}
        for key, value in self.loaded_chunk.items():
            if key != PIXEL_INDEX and key != IMAGE_INDEX:
                item[key] = value[idx]

        image_index = self.loaded_chunk[IMAGE_INDEX][idx]
        metadata_item = self.items[image_index]

        if WEIGHT in self.fields_to_load:
            item[WEIGHT] = metadata_item.weight

        if TRAIN_INDEX in self.fields_to_load:
            item[TRAIN_INDEX] = metadata_item.train_index

        width = metadata_item.W
        # image index, row, col
        item[RAY_INDEX] = torch.LongTensor([
            image_index,
            self.loaded_chunk[PIXEL_INDEX][idx] // width,
            self.loaded_chunk[PIXEL_INDEX][idx] % width])

        return item

    def _load_chunk_inner(self) -> Dict[str, torch.Tensor]:
        loaded_chunk = defaultdict(list)
        loaded = 0

        while loaded < self.items_per_chunk:
            if self.loaded_fields is None or self.loaded_field_offset >= len(self.loaded_fields[IMAGE_INDEX]):
                self.loaded_fields = {}
                self.loaded_field_offset = 0

                if self.load_all_in_memory:
                    if self.memory_fields is None:
                        self.memory_fields = self._load_items_into_memory()
                    to_shuffle = self.memory_fields
                else:
                    to_shuffle = self._load_random_subset()

                shuffled_indices = torch.randperm(len(to_shuffle[IMAGE_INDEX]))
                for key, val in to_shuffle.items():
                    self.loaded_fields[key] = val[shuffled_indices]

            to_add = self.items_per_chunk - loaded
            for key, val in self.loaded_fields.items():
                loaded_chunk[key].append(val[self.loaded_field_offset:self.loaded_field_offset + to_add])

            added = len(self.loaded_fields[IMAGE_INDEX][self.loaded_field_offset:self.loaded_field_offset + to_add])
            loaded += added
            self.loaded_field_offset += added

        loaded_chunk = {k: torch.cat(v) for k, v in loaded_chunk.items()}
        if self.load_all_in_memory:
            return loaded_chunk

        fields_to_load = {RGB, DEPTH} if DEPTH in self.fields_to_load else {RGB}
        loaded_fields = self._load_fields(loaded_chunk[IMAGE_INDEX], loaded_chunk[PIXEL_INDEX], fields_to_load, True)
        for key, val in loaded_fields.items():
            loaded_chunk[key] = val

        return loaded_chunk

    def _load_items_into_memory(self) -> Dict[str, torch.Tensor]:
        image_indices = []
        pixel_indices = []

        rgbs = []

        if DEPTH in self.fields_to_load:
            depths = []

        CONSOLE.log('Loading fields into memory')

        for i, metadata_item in enumerate(self.items):
            image_keep_mask = metadata_item.load_mask().view(-1)
            if not torch.any(image_keep_mask > 0):
                continue

            image_indices.append(torch.full_like(image_keep_mask, i, dtype=torch.long)[image_keep_mask > 0])
            pixel_indices.append(torch.arange(metadata_item.W * metadata_item.H, dtype=torch.long)[image_keep_mask > 0])

            image_rgbs = metadata_item.load_image().view(-1, 3)[image_keep_mask > 0].float() / 255.
            rgbs.append(image_rgbs)

            if DEPTH in self.fields_to_load:
                image_depth = metadata_item.load_depth().view(-1)[image_keep_mask > 0]
                depths.append(image_depth)

        CONSOLE.log('Finished loading fields')

        fields = {IMAGE_INDEX: torch.cat(image_indices), PIXEL_INDEX: torch.cat(pixel_indices)}
        fields[RGB] = torch.cat(rgbs)

        if DEPTH in self.fields_to_load:
            fields[DEPTH] = torch.cat(depths)

        return fields

    def _load_random_subset(self) -> Dict[str, torch.Tensor]:
        image_indices = torch.randint(0, len(self.items), (self.items_per_chunk,))
        pixel_indices = (
                torch.rand((self.items_per_chunk,)) * self.pixel_indices_to_sample[image_indices]).floor().long()

        mask = self._load_fields(image_indices, pixel_indices, {MASK})[MASK]

        return {
            IMAGE_INDEX: image_indices[mask > 0],
            PIXEL_INDEX: pixel_indices[mask > 0]
        }

    def _load_fields(self, image_indices: torch.Tensor, pixel_indices: torch.Tensor, fields_to_load: Set[str],
                     verbose: bool = False) -> Dict[str, torch.Tensor]:
        assert image_indices.shape == pixel_indices.shape

        sorted_image_indices, ordering = image_indices.sort()
        unique_image_indices, counts = torch.unique_consecutive(sorted_image_indices, return_counts=True)
        load_futures = {}

        offset = 0
        for image_index, image_count in zip(unique_image_indices, counts):
            load_futures[int(image_index)] = self.on_demand_executor.submit(
                self._load_image_fields, image_index, pixel_indices[ordering[offset:offset + image_count]],
                fields_to_load)
            offset += image_count

        loaded = {}
        offset = 0
        for i, (image_index, image_count) in enumerate(zip(unique_image_indices, counts)):
            if i % 1000 == 0 and verbose:
                CONSOLE.log(f'Loading image {i} of {len(unique_image_indices)}')
            loaded_features = load_futures[int(image_index)].result()
            to_put = ordering[offset:offset + image_count]

            for key, value in loaded_features.items():
                if i == 0:
                    loaded[key] = torch.zeros(image_indices.shape[0:1] + value.shape[1:], dtype=value.dtype)
                loaded[key][to_put] = value

            offset += image_count
            del load_futures[int(image_index)]

        return loaded

    def _load_image_fields(self, image_index: int, pixel_indices: torch.Tensor, fields_to_load: Set[str]) -> \
            Dict[str, torch.Tensor]:
        fields = {}

        item = self.items[image_index]
        for field in fields_to_load:
            if field == RGB:
                fields[RGB] = item.load_image().view(-1, 3)[pixel_indices].float() / 255.
            elif field == DEPTH:
                fields[DEPTH] = item.load_depth().view(-1, 1)[pixel_indices].float() / 255.
            elif field == MASK:
                fields[MASK] = item.load_mask().view(-1)[pixel_indices]
            else:
                raise Exception(f'Unrecognized field: {field}')

        return fields

```

## pynerf/data/datasets

### weighted_dataset.py

```python
"""
Weighted dataset.
"""

from typing import Dict

import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from pynerf.pynerf_constants import WEIGHT, TRAIN_INDEX


class WeightedDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        self.weights = self.metadata.get(WEIGHT, None)
        self.train_indices = self.metadata.get(TRAIN_INDEX, None)
        self.depth_images = self.metadata.get("depth_image", None)

    def get_metadata(self, data: Dict) -> Dict:
        metadata = {}
        if self.weights is not None:
            metadata[WEIGHT] = torch.full(data["image"].shape[:2], float(self.weights[data["image_idx"]]))
        if self.train_indices is not None:
            metadata[TRAIN_INDEX] = torch.full(data["image"].shape[:2], int(self.train_indices[data["image_idx"]]),
                                                dtype=torch.long)

        if self.depth_images is not None:
            filepath = self.depth_images[data["image_idx"]]
            depth_image = torch.FloatTensor(np.load(filepath)).unsqueeze(-1) / self.metadata["pose_scale_factor"]
            metadata["depth_image"] = depth_image

        return metadata

```

## pynerf/data/datasets

### __init__.py

```python

```

## pynerf/data/datasets

### image_metadata.py

```python
import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ImageMetadata:
    def __init__(self, image_path: str, W: int, H: int, depth_path: Optional[str], mask_path: Optional[str],
                 weight: Optional[float], train_index: Optional[int], pose_scale_factor: float,
                 local_cache: Optional[Path]):
        self.image_path = image_path
        self.W = W
        self.H = H
        self.depth_path = depth_path
        self.mask_path = mask_path
        self.weight = weight
        self.train_index = train_index
        self._pose_scale_factor = pose_scale_factor
        self._local_cache = local_cache

    def load_image(self) -> torch.Tensor:
        if self._local_cache is not None and not self.image_path.startswith(str(self._local_cache)):
            self.image_path = self._load_from_cache(self.image_path)

        rgbs = Image.open(self.image_path)
        size = rgbs.size

        if size[0] != self.W or size[1] != self.H:
            rgbs = rgbs.resize((self.W, self.H), Image.LANCZOS)

        rgbs = torch.ByteTensor(np.asarray(rgbs))

        if rgbs.shape[-1] == 4:
            rgbs = rgbs.float()
            alpha = rgbs[:, :, -1:] / 255.0
            rgbs = (rgbs[:, :, :3] * alpha + 255 * (1.0 - alpha)).byte()
            # Image.fromarray((rgbs[:, :, :3] * alpha + 255 * (1.0 - alpha)).byte().numpy()).save('/compute/autobot-0-25/hturki/lol2.png')

        return rgbs

    def load_depth(self) -> torch.Tensor:
        if self._local_cache is not None and not self.depth_path.startswith(str(self._local_cache)):
            self.depth_path = self._load_from_cache(self.depth_path)

        depth = torch.FloatTensor(np.load(self.depth_path))

        if depth.shape[0] != self.H or depth.shape[1] != self.W:
            depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(self.H, self.W)).squeeze()

        return depth / self._pose_scale_factor

    def load_mask(self) -> torch.Tensor:
        if self.mask_path is None:
            return torch.ones(self.H, self.W, dtype=torch.bool)

        if self._local_cache is not None and not self.mask_path.startswith(str(self._local_cache)):
            self.mask_path = self._load_from_cache(self.mask_path)

        mask = Image.open(self.mask_path)
        size = mask.size

        if size[0] != self.W or size[1] != self.H:
            mask = mask.resize((self.W, self.H), Image.NEAREST)

        return torch.BoolTensor(np.asarray(mask))

    def _load_from_cache(self, remote_path: str) -> str:
        sha_hash = hashlib.sha256()
        sha_hash.update(remote_path.encode('utf-8'))
        hashed = sha_hash.hexdigest()
        cache_path = self._local_cache / hashed[:2] / hashed[2:4] / f'{hashed}{Path(remote_path).suffix}'

        if cache_path.exists():
            return str(cache_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f'{cache_path}.{uuid.uuid4()}'
        shutil.copy(remote_path, tmp_path)

        os.rename(tmp_path, cache_path)
        return str(cache_path)

```

## pynerf/fields

### pynerf_field.py

```python
from typing import Optional

import math
import tinycudann as tcnn
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from torch import nn

from pynerf.fields.pynerf_base_field import PyNeRFBaseField, OutputInterpolation, LevelInterpolation


class PyNeRFField(PyNeRFBaseField):

    def __init__(
            self,
            aabb,
            num_images: int,
            num_layers: int = 2,
            hidden_dim: int = 64,
            geo_feat_dim: int = 15,
            num_layers_color: int = 3,
            hidden_dim_color: int = 64,
            appearance_embedding_dim: int = 32,
            max_resolution: int = 4096,
            spatial_distortion: SpatialDistortion = None,
            output_interpolation: OutputInterpolation = OutputInterpolation.EMBEDDING,
            level_interpolation: LevelInterpolation = LevelInterpolation.LINEAR,
            num_scales: int = 8,
            scale_factor: float = 2.0,
            share_feature_grid: bool = False,
            cameras: Cameras = None,
            base_resolution: int = 16,
            features_per_level: int = 2,
            num_levels: int = 16,
            log2_hashmap_size: int = 19,
            trained_level_resolution: Optional[int] = 128,
    ) -> None:
        super().__init__(aabb, num_images, [num_levels * features_per_level for _ in range(num_scales)], num_layers,
                         hidden_dim, geo_feat_dim, num_layers_color, hidden_dim_color, appearance_embedding_dim,
                         max_resolution, spatial_distortion, output_interpolation, level_interpolation, num_scales,
                         scale_factor, share_feature_grid, cameras, trained_level_resolution)

        if not share_feature_grid:
            encodings = []

            for scale in range(num_scales):
                cur_max_res = max_resolution / (scale_factor ** (num_scales - 1 - scale))
                cur_level_scale = math.exp(math.log(cur_max_res / base_resolution) / (num_levels - 1))

                encoding = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        'otype': 'HashGrid',
                        'n_levels': num_levels,
                        'n_features_per_level': features_per_level,
                        'log2_hashmap_size': log2_hashmap_size,
                        'base_resolution': base_resolution,
                        'per_level_scale': cur_level_scale,
                    })

                encodings.append(encoding)

            self.encodings = nn.ModuleList(encodings)
        else:
            per_level_scale = math.exp(math.log(max_resolution / base_resolution) / (num_levels - 1))

            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    'otype': 'HashGrid',
                    'n_levels': num_levels,
                    'n_features_per_level': features_per_level,
                    'log2_hashmap_size': log2_hashmap_size,
                    'base_resolution': base_resolution,
                    'per_level_scale': per_level_scale,
                })

    def get_shared_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        return self.encoding(positions)

    def get_level_encoding(self, level: int, positions: torch.Tensor) -> torch.Tensor:
        return self.encodings[level](positions)

```

## pynerf/fields

### __init__.py

```python

```

## pynerf/fields

### pynerf_base_field.py

```python
"""
PyNeRF field implementation.
"""
from abc import abstractmethod
from collections import defaultdict
from enum import Enum, auto
from typing import Tuple, Optional, Any, Dict, List

import math
import tinycudann as tcnn
import torch
import torch_scatter
from jaxtyping import Shaped
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components import MLP
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from rich.console import Console
from torch import nn, Tensor

from pynerf.pynerf_constants import TRAIN_INDEX, PyNeRFFieldHeadNames, EXPLICIT_LEVEL

CONSOLE = Console(width=120)


class OutputInterpolation(Enum):
    COLOR = auto()
    EMBEDDING = auto()


class LevelInterpolation(Enum):
    NONE = auto()
    LINEAR = auto()


class PyNeRFBaseField(Field):

    def __init__(
            self,
            aabb,
            num_images: int,
            encoding_input_dims: List[int],
            num_layers: int = 2,
            hidden_dim: int = 64,
            geo_feat_dim: int = 15,
            num_layers_color: int = 3,
            hidden_dim_color: int = 64,
            appearance_embedding_dim: int = 32,
            max_resolution: int = 4096,
            spatial_distortion: SpatialDistortion = None,
            output_interpolation: OutputInterpolation = OutputInterpolation.EMBEDDING,
            level_interpolation: LevelInterpolation = LevelInterpolation.LINEAR,
            num_scales: int = 8,
            scale_factor: float = 2.0,
            share_feature_grid: bool = False,
            cameras: Cameras = None,
            trained_level_resolution: Optional[int] = 128,
    ) -> None:
        super().__init__()
        self.register_buffer('aabb', aabb, persistent=False)

        self.geo_feat_dim = geo_feat_dim
        self.appearance_embedding_dim = appearance_embedding_dim
        self.output_interpolation = output_interpolation
        self.level_interpolation = level_interpolation
        self.spatial_distortion = spatial_distortion
        self.num_scales = num_scales
        self.share_feature_grid = share_feature_grid
        self.cameras = cameras

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'SphericalHarmonics',
                'degree': 4,
            },
        )

        if appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(num_images, self.appearance_embedding_dim)

        area_of_interest = (aabb[1] - aabb[0]).max()
        if self.spatial_distortion is not None:
            area_of_interest *= 2  # Scene contraction uses half of the table capacity for contracted space

        self.log_scale_factor = math.log(scale_factor)

        # Get base log of lowest mip level
        self.base_log = math.log(area_of_interest, scale_factor) \
                        - (math.log(max_resolution, scale_factor) - (num_scales - 1))

        self.trained_level_resolution = trained_level_resolution
        if trained_level_resolution is not None:
            self.register_buffer('min_trained_level', torch.full(
                (trained_level_resolution, trained_level_resolution, trained_level_resolution), self.num_scales,
                dtype=torch.float32))
            self.register_buffer('max_trained_level', torch.full(
                (trained_level_resolution, trained_level_resolution, trained_level_resolution), -1,
                dtype=torch.float32))

        mlp_bases = []

        self.encoding_input_dims = encoding_input_dims
        assert len(encoding_input_dims) == num_scales, \
            f'Number of encoding dims {len(encoding_input_dims)} different from number of scales {num_scales}'
        for encoding_input_dim in encoding_input_dims:
            mlp_bases.append(MLP(
                in_dim=encoding_input_dim,
                out_dim=1 + self.geo_feat_dim,
                num_layers=num_layers,
                layer_width=hidden_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation="tcnn"
            ))

        self.mlp_bases = nn.ModuleList(mlp_bases)

        if output_interpolation == OutputInterpolation.COLOR:
            mlp_heads = []
            for i in range(num_scales):
                mlp_heads.append(MLP(
                    in_dim=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
                    out_dim=3,
                    num_layers=num_layers_color,
                    layer_width=hidden_dim_color,
                    activation=nn.ReLU(),
                    out_activation=nn.Sigmoid(),
                    implementation="tcnn"
                ))
            self.mlp_heads = nn.ModuleList(mlp_heads)
        else:
            self.mlp_head = MLP(
                in_dim=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
                out_dim=3,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation="tcnn"
            )

    @abstractmethod
    def get_shared_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_level_encoding(self, level: int, positions: torch.Tensor) -> torch.Tensor:
        pass

    def get_density(self, ray_samples: RaySamples, update_levels: bool = True):
        positions = ray_samples.frustums.get_positions()

        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
            positions_flat = positions.view(-1, 3)
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
            positions_flat = positions.view(-1, 3)

        explicit_level = ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata
        if explicit_level:
            level = ray_samples.metadata[EXPLICIT_LEVEL]
            pixel_levels = torch.full_like(ray_samples.frustums.starts[..., 0], level).view(-1)
        else:
            # Assuming pixels are square
            sample_distances = ((ray_samples.frustums.starts + ray_samples.frustums.ends) / 2)
            pixel_widths = (ray_samples.frustums.pixel_area.sqrt() * sample_distances).view(-1)

            pixel_levels = self.base_log - torch.log(pixel_widths) / self.log_scale_factor
            if self.trained_level_resolution is not None:
                reso_indices = (positions_flat * self.trained_level_resolution).floor().long().clamp(0,
                                                                                                     self.trained_level_resolution - 1)
                if self.training:
                    if update_levels:
                        flat_indices = reso_indices[
                                           ..., 0] * self.trained_level_resolution * self.trained_level_resolution \
                                       + reso_indices[..., 1] * self.trained_level_resolution + reso_indices[..., 2]
                        torch_scatter.scatter_min(pixel_levels, flat_indices, out=self.min_trained_level.view(-1))
                        torch_scatter.scatter_max(pixel_levels, flat_indices, out=self.max_trained_level.view(-1))
                else:
                    min_levels = self.min_trained_level[
                        reso_indices[..., 0], reso_indices[..., 1], reso_indices[..., 2]]
                    max_levels = self.max_trained_level[
                        reso_indices[..., 0], reso_indices[..., 1], reso_indices[..., 2]]
                    pixel_levels = torch.maximum(min_levels, torch.minimum(pixel_levels, max_levels))

        if self.level_interpolation == LevelInterpolation.NONE:
            level_indices = get_levels(pixel_levels, self.num_scales)
            level_weights = {}
            for level, indices in level_indices.items():
                level_weights[level] = torch.ones_like(indices, dtype=pixel_levels.dtype)
        elif self.level_interpolation == LevelInterpolation.LINEAR:
            level_indices, level_weights = get_weights(pixel_levels, self.num_scales)
        else:
            raise Exception(self.level_interpolation)

        if self.share_feature_grid:
            encoding = self.get_shared_encoding(positions_flat)

        if self.output_interpolation == OutputInterpolation.COLOR:
            density = None
            level_embeddings = {}
        else:
            interpolated_h = None

        for level, cur_level_indices in level_indices.items():
            if self.share_feature_grid:
                level_encoding = encoding[cur_level_indices][..., :self.encoding_input_dims[level]]
            else:
                level_encoding = self.get_level_encoding(level, positions_flat[cur_level_indices])

            level_h = self.mlp_bases[level](level_encoding).to(positions)
            cur_level_weights = level_weights[level]
            if self.output_interpolation == OutputInterpolation.COLOR:
                density_before_activation, level_mlp_out = torch.split(level_h, [1, self.geo_feat_dim], dim=-1)
                level_embeddings[level] = level_mlp_out
                level_density = trunc_exp(density_before_activation - 1)
                if density is None:
                    density = torch.zeros(positions_flat.shape[0], *level_density.shape[1:],
                                          dtype=level_density.dtype, device=level_density.device)
                density[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_density
            elif self.output_interpolation == OutputInterpolation.EMBEDDING:
                if interpolated_h is None:
                    interpolated_h = torch.zeros(positions_flat.shape[0], *level_h.shape[1:],
                                                 dtype=level_h.dtype, device=level_h.device)

                interpolated_h[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_h
            else:
                raise Exception(self.output_interpolation)

        if self.output_interpolation == OutputInterpolation.COLOR:
            additional_info = (level_indices, level_weights, level_embeddings)
        elif self.output_interpolation == OutputInterpolation.EMBEDDING:
            density_before_activation, mlp_out = torch.split(interpolated_h, [1, self.geo_feat_dim], dim=-1)
            density = trunc_exp(density_before_activation - 1)
            additional_info = mlp_out
        else:
            raise Exception(self.output_interpolation)

        if self.training:
            level_counts = defaultdict(int)
            for level, indices in level_indices.items():
                level_counts[level] = indices.shape[0] / pixel_levels.shape[0]

            return density.view(ray_samples.frustums.starts.shape), (additional_info, level_counts)
        else:
            return density.view(ray_samples.frustums.starts.shape), (additional_info, pixel_levels.view(density.shape))

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Tuple[Any] = None):
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        if self.appearance_embedding_dim > 0:
            if ray_samples.metadata is not None and TRAIN_INDEX in ray_samples.metadata:
                embedded_appearance = self.embedding_appearance(
                    ray_samples.metadata[TRAIN_INDEX].squeeze().to(d.device))
            elif self.training:
                embedded_appearance = self.embedding_appearance(ray_samples.camera_indices.squeeze())
            else:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)

            embedded_appearance = embedded_appearance.view(-1, self.appearance_embedding_dim)

        outputs = {}

        if self.training:
            density_embedding, level_counts = density_embedding
            outputs[PyNeRFFieldHeadNames.LEVEL_COUNTS] = level_counts
        else:
            density_embedding, levels = density_embedding
            outputs[PyNeRFFieldHeadNames.LEVELS] = levels.view(ray_samples.frustums.starts.shape)

        if ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata:
            level = ray_samples.metadata[EXPLICIT_LEVEL]
            if self.output_interpolation == OutputInterpolation.COLOR:
                _, _, level_embeddings = density_embedding
                density_embedding = level_embeddings[level]

                mlp_head = self.mlp_heads[level]
            else:
                mlp_head = self.mlp_head

            color_inputs = [d, density_embedding]
            if self.appearance_embedding_dim > 0:
                color_inputs.append(embedded_appearance)

            h = torch.cat(color_inputs, dim=-1)

            outputs[FieldHeadNames.RGB] = mlp_head(h).view(directions.shape).to(directions)
            return outputs

        if self.output_interpolation != OutputInterpolation.COLOR:
            color_inputs = [d, density_embedding]
            if self.appearance_embedding_dim > 0:
                color_inputs.append(embedded_appearance)
            h = torch.cat(color_inputs, dim=-1)
            rgbs = self.mlp_head(h).view(directions.shape).to(directions)
            outputs[FieldHeadNames.RGB] = rgbs

            return outputs

        level_indices, level_weights, level_embeddings = density_embedding

        rgbs = None
        for level, cur_level_indices in level_indices.items():
            color_inputs = [d[cur_level_indices], level_embeddings[level]]
            if self.appearance_embedding_dim > 0:
                color_inputs.append(embedded_appearance[cur_level_indices])
            h = torch.cat(color_inputs, dim=-1)

            level_rgbs = self.mlp_heads[level](h).to(directions)
            if rgbs is None:
                rgbs = torch.zeros_like(directions)
            rgbs.view(-1, 3)[cur_level_indices] += level_weights[level].unsqueeze(-1) * level_rgbs

        outputs[FieldHeadNames.RGB] = rgbs
        return outputs

    def density_fn(self, positions: Shaped[Tensor, "*bs 3"], times: Optional[Shaped[Tensor, "*bs 1"]] = None,
                   step_size: int = None, origins: Optional[Shaped[Tensor, "*bs 3"]] = None,
                   directions: Optional[Shaped[Tensor, "*bs 3"]] = None,
                   starts: Optional[Shaped[Tensor, "*bs 1"]] = None,
                   ends: Optional[Shaped[Tensor, "*bs 1"]] = None, pixel_area: Optional[Shaped[Tensor, "*bs 1"]] = None) \
            -> Shaped[Tensor, "*bs 1"]:
        """Returns only the density. Used primarily with the density grid.
        """
        if origins is None:
            camera_ids = torch.randint(0, len(self.cameras), (positions.shape[0],), device=positions.device)
            cameras = self.cameras.to(camera_ids.device)[camera_ids]
            origins = cameras.camera_to_worlds[:, :, 3]
            directions = positions - origins
            directions, _ = camera_utils.normalize_with_norm(directions, -1)
            coords = torch.cat(
                [torch.rand_like(origins[..., :1]) * cameras.height, torch.rand_like(origins[..., :1]) * cameras.width],
                -1).floor().long()

            pixel_area = cameras.generate_rays(torch.arange(len(cameras)).unsqueeze(-1), coords=coords).pixel_area
            starts = (origins - positions).norm(dim=-1, keepdim=True) - step_size / 2
            ends = starts + step_size

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                starts=starts,
                ends=ends,
                pixel_area=pixel_area,
            ),
            times=times
        )

        density, _ = self.get_density(ray_samples, update_levels=False)
        return density


@torch.jit.script
def get_levels(pixel_levels: torch.Tensor, num_levels: int) -> Dict[int, torch.Tensor]:
    sorted_pixel_levels, ordering = pixel_levels.sort(descending=False)
    level_indices: Dict[int, torch.Tensor] = {}

    start = 0
    for level in range(num_levels - 1):
        end = start + (sorted_pixel_levels[start:] <= level).sum()

        if end > start:
            if ordering[start:end].shape[0] > 0:
                level_indices[level] = ordering[start:end]

        start = end

    if ordering[start:].shape[0] > 0:
        level_indices[num_levels - 1] = ordering[start:]

    return level_indices


@torch.jit.script
def get_weights(pixel_levels: torch.Tensor, num_levels: int) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    sorted_pixel_levels, ordering = pixel_levels.sort(descending=False)
    level_indices: Dict[int, torch.Tensor] = {}
    level_weights: Dict[int, torch.Tensor] = {}

    mid = 0
    end = 0
    for level in range(num_levels):
        if level == 0:
            mid = (sorted_pixel_levels < level).sum()
            cur_level_indices = [ordering[:mid]]
            cur_level_weights = [torch.ones_like(cur_level_indices[0], dtype=pixel_levels.dtype)]
        else:
            start = mid
            mid = end
            cur_level_indices = [ordering[start:mid]]
            cur_level_weights = [sorted_pixel_levels[start:mid] - (level - 1)]

        if level < num_levels - 1:
            end = mid + (sorted_pixel_levels[mid:] < level + 1).sum()
            cur_level_indices.append(ordering[mid:end])
            cur_level_weights.append(1 - (sorted_pixel_levels[mid:end] - level))
        else:
            cur_level_indices.append(ordering[mid:])
            cur_level_weights.append(torch.ones_like(cur_level_indices[-1], dtype=pixel_levels.dtype))

        cur_level_indices = torch.cat(cur_level_indices)

        if cur_level_indices.shape[0] > 0:
            level_indices[level] = cur_level_indices
            level_weights[level] = torch.cat(cur_level_weights)

    return level_indices, level_weights


def parse_output_interpolation(model: str) -> OutputInterpolation:
    if model.casefold() == 'color':
        return OutputInterpolation.COLOR
    elif model.casefold() == 'embedding':
        return OutputInterpolation.EMBEDDING
    else:
        raise Exception(model)


def parse_level_interpolation(model: str) -> LevelInterpolation:
    if model.casefold() == 'none':
        return LevelInterpolation.NONE
    elif model.casefold() == 'linear':
        return LevelInterpolation.LINEAR
    else:
        raise Exception(model)

```

