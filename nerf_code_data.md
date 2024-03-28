## 

### pixel_samplers.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for sampling pixels.
"""

import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union

import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.utils.pixel_sampling_utils import erode_mask


@dataclass
class PixelSamplerConfig(InstantiateConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: PixelSampler)
    """Target class to instantiate."""
    num_rays_per_batch: int = 4096
    """Number of rays to sample per batch."""
    keep_full_image: bool = False
    """Whether or not to include a reference to the full image in returned batch."""
    is_equirectangular: bool = False
    """List of whether or not camera i is equirectangular."""
    ignore_mask: bool = False
    """Whether to ignore the masks when sampling."""
    fisheye_crop_radius: Optional[float] = None
    """Set to the radius (in pixels) for fisheye cameras."""
    rejection_sample_mask: bool = True
    """Whether or not to use rejection sampling when sampling images with masks"""
    max_num_iterations: int = 100
    """If rejection sampling masks, the maximum number of times to sample"""


class PixelSampler:
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: PixelSamplerConfig

    def __init__(self, config: PixelSamplerConfig, **kwargs) -> None:
        self.kwargs = kwargs
        self.config = config
        # Possibly override some values if they are present in the kwargs dictionary
        self.config.num_rays_per_batch = self.kwargs.get("num_rays_per_batch", self.config.num_rays_per_batch)
        self.config.keep_full_image = self.kwargs.get("keep_full_image", self.config.keep_full_image)
        self.config.is_equirectangular = self.kwargs.get("is_equirectangular", self.config.is_equirectangular)
        self.config.fisheye_crop_radius = self.kwargs.get("fisheye_crop_radius", self.config.fisheye_crop_radius)
        self.set_num_rays_per_batch(self.config.num_rays_per_batch)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        indices = (
            torch.rand((batch_size, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()

        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            if self.config.rejection_sample_mask:
                num_valid = 0
                for _ in range(self.config.max_num_iterations):
                    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
                    chosen_indices_validity = mask[..., 0][c, y, x].bool()
                    num_valid = int(torch.sum(chosen_indices_validity).item())
                    if num_valid == batch_size:
                        break
                    else:
                        replacement_indices = (
                            torch.rand((batch_size - num_valid, 3), device=device)
                            * torch.tensor([num_images, image_height, image_width], device=device)
                        ).long()
                        indices[~chosen_indices_validity] = replacement_indices

                if num_valid != batch_size:
                    warnings.warn(
                        """
                        Masked sampling failed, mask is either empty or mostly empty.
                        Reverting behavior to non-rejection sampling. Consider setting
                        pipeline.datamanager.pixel-sampler.rejection-sample-mask to False
                        or increasing pipeline.datamanager.pixel-sampler.max-num-iterations
                        """
                    )
                    self.config.rejection_sample_mask = False
                    nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                    chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
                    indices = nonzero_indices[chosen_indices]
            else:
                nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
                indices = nonzero_indices[chosen_indices]

        return indices

    def sample_method_equirectangular(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            # Note: if there is a mask, sampling reduces back to uniform sampling, which gives more
            # sampling weight to the poles of the image than the equators.
            # TODO(kevinddchen): implement the correct mask-sampling method.

            indices = self.sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # We sample theta uniformly in [0, 2*pi]
            # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
            # This is done by inverse transform sampling.
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            num_images_rand = torch.rand(batch_size, device=device)
            phi_rand = torch.acos(1 - 2 * torch.rand(batch_size, device=device)) / torch.pi
            theta_rand = torch.rand(batch_size, device=device)
            indices = torch.floor(
                torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def sample_method_fisheye(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            indices = self.sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # Rejection sampling.
            valid: Optional[torch.Tensor] = None
            indices = None
            while True:
                samples_needed = batch_size if valid is None else int(batch_size - torch.sum(valid).item())

                # Check if done!
                if samples_needed == 0:
                    break

                rand_samples = torch.rand((samples_needed, 2), device=device)
                # Convert random samples to radius and theta.
                radii = self.config.fisheye_crop_radius * torch.sqrt(rand_samples[:, 0])
                theta = 2.0 * torch.pi * rand_samples[:, 1]

                # Convert radius and theta to x and y.
                x = (radii * torch.cos(theta) + image_width // 2).long()
                y = (radii * torch.sin(theta) + image_height // 2).long()
                sampled_indices = torch.stack(
                    [torch.randint(0, num_images, size=(samples_needed,), device=device), y, x], dim=-1
                )

                # Update indices.
                if valid is None:
                    indices = sampled_indices
                    valid = (
                        (sampled_indices[:, 1] >= 0)
                        & (sampled_indices[:, 1] < image_height)
                        & (sampled_indices[:, 2] >= 0)
                        & (sampled_indices[:, 2] < image_width)
                    )
                else:
                    assert indices is not None
                    not_valid = ~valid
                    indices[not_valid, :] = sampled_indices
                    valid[not_valid] = (
                        (sampled_indices[:, 1] >= 0)
                        & (sampled_indices[:, 1] < image_height)
                        & (sampled_indices[:, 2] >= 0)
                        & (sampled_indices[:, 2] < image_width)
                    )
            assert indices is not None

        assert indices.shape == (batch_size, 3)
        return indices

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            elif self.config.fisheye_crop_radius is not None:
                indices = self.sample_method_fisheye(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            else:
                indices = self.sample_method(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
        else:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            elif self.config.fisheye_crop_radius is not None:
                indices = self.sample_method_fisheye(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            else:
                indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }
        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []
        all_depth_images = []

        num_rays_in_batch = num_rays_per_batch // num_images
        if num_rays_in_batch % 2 != 0:
            num_rays_in_batch += 1

        if "mask" in batch:
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape

                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                indices = self.sample_method(
                    num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i].unsqueeze(0), device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
                if "depth_image" in batch:
                    all_depth_images.append(batch["depth_image"][i][indices[:, 1], indices[:, 2]])

        else:
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                if self.config.is_equirectangular:
                    indices = self.sample_method_equirectangular(
                        num_rays_in_batch, 1, image_height, image_width, device=device
                    )
                else:
                    indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
                if "depth_image" in batch:
                    all_depth_images.append(batch["depth_image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key != "image_idx" and key != "image" and key != "mask" and key != "depth_image" and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)
        if "depth_image" in batch:
            collated_batch["depth_image"] = torch.cat(all_depth_images, dim=0)

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch


@dataclass
class PatchPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PatchPixelSampler."""

    _target: Type = field(default_factory=lambda: PatchPixelSampler)
    """Target class to instantiate."""
    patch_size: int = 32
    """Side length of patch. This must be consistent in the method
    config in order for samples to be reshaped into patches correctly."""


class PatchPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        config: the PatchPixelSamplerConfig used to instantiate class
    """

    config: PatchPixelSamplerConfig

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.config.patch_size**2)) * (self.config.patch_size**2)

    # overrides base method
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, Tensor) and not self.config.ignore_mask:
            sub_bs = batch_size // (self.config.patch_size**2)
            half_patch_size = int(self.config.patch_size / 2)
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=half_patch_size)
            nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=sub_bs)
            indices = nonzero_indices[chosen_indices]

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys - half_patch_size
            indices[:, ..., 2] += xxs - half_patch_size

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)
        else:
            sub_bs = batch_size // (self.config.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.config.patch_size, image_width - self.config.patch_size],
                device=device,
            )

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices


@dataclass
class PairPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PairPixelSampler."""

    _target: Type = field(default_factory=lambda: PairPixelSampler)
    """Target class to instantiate."""
    radius: int = 2
    """max distance between pairs of pixels."""


class PairPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples pair of pixels from 'image_batch's. Samples pairs of pixels from
        from the images randomly within a 'radius' distance apart. Useful for pair-based losses.

    Args:
        config: the PairPixelSamplerConfig used to instantiate class
    """

    def __init__(self, config: PairPixelSamplerConfig, **kwargs) -> None:
        self.config = config
        self.radius = self.config.radius
        super().__init__(self.config, **kwargs)
        self.rays_to_sample = self.config.num_rays_per_batch // 2

    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: Optional[int],
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        rays_to_sample = self.rays_to_sample
        if batch_size is not None:
            assert (
                int(batch_size) % 2 == 0
            ), f"PairPixelSampler can only return batch sizes in multiples of two (got {batch_size})"
            rays_to_sample = batch_size // 2

        if isinstance(mask, Tensor) and not self.config.ignore_mask:
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=self.radius)
            nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=rays_to_sample)
            indices = nonzero_indices[chosen_indices]
        else:
            s = (rays_to_sample, 1)
            ns = torch.randint(0, num_images, s, dtype=torch.long, device=device)
            hs = torch.randint(self.radius, image_height - self.radius, s, dtype=torch.long, device=device)
            ws = torch.randint(self.radius, image_width - self.radius, s, dtype=torch.long, device=device)
            indices = torch.concat((ns, hs, ws), dim=1)

        pair_indices = torch.hstack(
            (
                torch.zeros(rays_to_sample, 1, device=device, dtype=torch.long),
                torch.randint(-self.radius, self.radius, (rays_to_sample, 2), device=device, dtype=torch.long),
            )
        )
        pair_indices += indices
        indices = torch.hstack((indices, pair_indices)).view(rays_to_sample * 2, 3)
        return indices

```

## 

### scene_box.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset input structures.
"""

from dataclasses import dataclass
from typing import Tuple, Union

import torch
import viser.transforms as vtf
from jaxtyping import Float
from torch import Tensor


@dataclass
class SceneBox:
    """Data to represent the scene box."""

    aabb: Float[Tensor, "2 3"]
    """aabb: axis-aligned bounding box.
    aabb[0] is the minimum (x,y,z) point.
    aabb[1] is the maximum (x,y,z) point."""

    def within(self, pts: Float[Tensor, "n 3"]):
        """Returns a boolean mask indicating whether each point is within the box."""
        return torch.all(pts > self.aabb[0], dim=-1) & torch.all(pts < self.aabb[1], dim=-1)

    def get_diagonal_length(self):
        """Returns the longest diagonal length."""
        diff = self.aabb[1] - self.aabb[0]
        length = torch.sqrt((diff**2).sum() + 1e-20)
        return length

    def get_center(self):
        """Returns the center of the box."""
        diff = self.aabb[1] - self.aabb[0]
        return self.aabb[0] + diff / 2.0

    def get_centered_and_scaled_scene_box(self, scale_factor: Union[float, torch.Tensor] = 1.0):
        """Returns a new box that has been shifted and rescaled to be centered
        about the origin.

        Args:
            scale_factor: How much to scale the camera origins by.
        """
        return SceneBox(aabb=(self.aabb - self.get_center()) * scale_factor)

    @staticmethod
    def get_normalized_positions(positions: Float[Tensor, "*batch 3"], aabb: Float[Tensor, "2 3"]):
        """Return normalized positions in range [0, 1] based on the aabb axis-aligned bounding box.

        Args:
            positions: the xyz positions
            aabb: the axis-aligned bounding box
        """
        aabb_lengths = aabb[1] - aabb[0]
        normalized_positions = (positions - aabb[0]) / aabb_lengths
        return normalized_positions

    @staticmethod
    def from_camera_poses(poses: Float[Tensor, "*batch 3 4"], scale_factor: float) -> "SceneBox":
        """Returns the instance of SceneBox that fully envelopes a set of poses

        Args:
            poses: tensor of camera pose matrices
            scale_factor: How much to scale the camera origins by.
        """
        xyzs = poses[..., :3, -1]
        aabb = torch.stack([torch.min(xyzs, dim=0)[0], torch.max(xyzs, dim=0)[0]])
        return SceneBox(aabb=aabb * scale_factor)


@dataclass
class OrientedBox:
    R: Float[Tensor, "3 3"]
    """R: rotation matrix."""
    T: Float[Tensor, "3"]
    """T: translation vector."""
    S: Float[Tensor, "3"]
    """S: scale vector."""

    def within(self, pts: Float[Tensor, "n 3"]):
        """Returns a boolean mask indicating whether each point is within the box."""
        R, T, S = self.R, self.T, self.S.to(pts)
        H = torch.eye(4, device=pts.device, dtype=pts.dtype)
        H[:3, :3] = R
        H[:3, 3] = T
        H_world2bbox = torch.inverse(H)
        pts = torch.cat((pts, torch.ones_like(pts[..., :1])), dim=-1)
        pts = torch.matmul(H_world2bbox, pts.T).T[..., :3]

        comp_l = torch.tensor(-S / 2)
        comp_m = torch.tensor(S / 2)
        mask = torch.all(torch.concat([pts > comp_l, pts < comp_m], dim=-1), dim=-1)
        return mask

    @staticmethod
    def from_params(
        pos: Tuple[float, float, float], rpy: Tuple[float, float, float], scale: Tuple[float, float, float]
    ):
        """Construct a box from position, rotation, and scale parameters."""
        R = torch.tensor(vtf.SO3.from_rpy_radians(rpy[0], rpy[1], rpy[2]).as_matrix())
        T = torch.tensor(pos)
        S = torch.tensor(scale)
        return OrientedBox(R=R, T=T, S=S)

```

## 

### __init__.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

## datamanagers

### parallel_datamanager.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parallel data manager that generates training data in multiple python processes.
"""
from __future__ import annotations

import concurrent.futures
import queue
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import torch
from pathos.helpers import mp
from rich.progress import track
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    TDataset,
    VanillaDataManagerConfig,
    variable_res_collate,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig, PixelSampler, PixelSamplerConfig
from nerfstudio.data.utils.dataloaders import CacheDataloader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ParallelDataManagerConfig(VanillaDataManagerConfig):
    """Config for a `ParallelDataManager` which reads data in multiple processes"""

    _target: Type = field(default_factory=lambda: ParallelDataManager)
    """Target class to instantiate."""
    num_processes: int = 1
    """Number of processes to use for train data loading. More than 1 doesn't result in that much better performance"""
    queue_size: int = 2
    """Size of shared data queue containing generated ray bundles and batches.
    If queue_size <= 0, the queue size is infinite."""
    max_thread_workers: Optional[int] = None
    """Maximum number of threads to use in thread pool executor. If None, use ThreadPool default."""


class DataProcessor(mp.Process):  # type: ignore
    """Parallel dataset batch processor.

    This class is responsible for generating ray bundles from an input dataset
    in parallel python processes.

    Args:
        out_queue: the output queue for storing the processed data
        config: configuration object for the parallel data manager
        dataparser_outputs: outputs from the dataparser
        dataset: input dataset
        pixel_sampler: The pixel sampler for sampling rays
    """

    def __init__(
        self,
        out_queue: mp.Queue,  # type: ignore
        config: ParallelDataManagerConfig,
        dataparser_outputs: DataparserOutputs,
        dataset: TDataset,
        pixel_sampler: PixelSampler,
    ):
        super().__init__()
        self.daemon = True
        self.out_queue = out_queue
        self.config = config
        self.dataparser_outputs = dataparser_outputs
        self.dataset = dataset
        self.exclude_batch_keys_from_device = self.dataset.exclude_batch_keys_from_device
        self.pixel_sampler = pixel_sampler
        self.ray_generator = RayGenerator(self.dataset.cameras)
        self.cache_images()

    def run(self):
        """Append out queue in parallel with ray bundles and batches."""
        while True:
            batch = self.pixel_sampler.sample(self.img_data)
            ray_indices = batch["indices"]
            ray_bundle: RayBundle = self.ray_generator(ray_indices)
            # check that GPUs are available
            if torch.cuda.is_available():
                ray_bundle = ray_bundle.pin_memory()
            while True:
                try:
                    self.out_queue.put((ray_bundle, batch))
                    break
                except queue.Full:
                    time.sleep(0.0001)
                except Exception:
                    CONSOLE.print_exception()
                    CONSOLE.print("[bold red]Error: Error occurred in parallel datamanager queue.")

    def cache_images(self):
        """Caches all input images into a NxHxWx3 tensor."""
        indices = range(len(self.dataset))
        batch_list = []
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_thread_workers) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)
            for res in track(results, description="Loading data batch", transient=False):
                batch_list.append(res.result())
        self.img_data = self.config.collate_fn(batch_list)


class ParallelDataManager(DataManager, Generic[TDataset]):
    """Data manager implementation for parallel dataloading.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    def __init__(
        self,
        config: ParallelDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.eval_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split=self.test_split)
        cameras = self.train_dataparser_outputs.cameras
        if len(cameras) > 1:
            for i in range(1, len(cameras)):
                if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                    CONSOLE.print("Variable resolution, using variable_res_collate")
                    self.config.collate_fn = variable_res_collate
                    break
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        # Spawn is critical for not freezing the program (PyTorch compatability issue)
        # check if spawn is already set
        if mp.get_start_method(allow_none=True) is None:  # type: ignore
            mp.set_start_method("spawn")  # type: ignore
        super().__init__()

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[ParallelDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is ParallelDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is ParallelDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is ParallelDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training."""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation."""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1 and type(self.config.pixel_sampler) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")

        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None:
            fisheye_crop_radius = dataset.cameras.metadata.get("fisheye_crop_radius")

        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
        )

    def setup_train(self):
        """Sets up parallel python data processes for training."""
        assert self.train_dataset is not None
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)  # type: ignore
        self.data_queue = mp.Queue(maxsize=self.config.queue_size)  # type: ignore
        self.data_procs = [
            DataProcessor(
                out_queue=self.data_queue,  # type: ignore
                config=self.config,
                dataparser_outputs=self.train_dataparser_outputs,
                dataset=self.train_dataset,
                pixel_sampler=self.train_pixel_sampler,
            )
            for i in range(self.config.num_processes)
        ]
        for proc in self.data_procs:
            proc.start()
        print("Started threads")

    def setup_eval(self):
        """Sets up the data loader for evaluation."""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)  # type: ignore
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the parallel training processes."""
        self.train_count += 1
        bundle, batch = self.data_queue.get()
        ray_bundle = bundle.to(self.device)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Retrieve the next eval image."""
        for camera, batch in self.eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        if self.train_pixel_sampler is not None:
            return self.train_pixel_sampler.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        if self.eval_pixel_sampler is not None:
            return self.eval_pixel_sampler.num_rays_per_batch
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def __del__(self):
        """Clean up the parallel data processes."""
        if hasattr(self, "data_procs"):
            for proc in self.data_procs:
                proc.terminate()
                proc.join()

```

## datamanagers

### __init__.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

## datamanagers

### random_cameras_datamanager.py

```python
# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Data manager without input images, only random camera poses
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union

import torch
from rich.progress import Console
from torch import Tensor
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader

CONSOLE = Console(width=120)


class TrivialDataset(InputDataset):
    """A trivial dataset with blank images for the viewer"""

    # pylint: disable=super-init-not-called
    def __init__(self, cameras: Cameras):
        self.size = cameras.size
        self.cameras = cameras
        self.alpha_color = None
        self.scene_box = SceneBox(torch.Tensor([[-1, -1, -1], [1, 1, 1]]))
        self.mask_filenames = None
        self.metadata = to_immutable_dict({})

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> Dict:
        return {
            "image": torch.cat([torch.ones(128, 256, 3), torch.zeros(128, 256, 3)], dim=0),
            "image_idx": index,
        }


def random_train_pose(
    size: int,
    resolution: int,
    device: Union[torch.device, str],
    radius_mean: float = 1.0,
    radius_std: float = 0.1,
    central_rotation_range: Tuple[float, float] = (0, 360),
    vertical_rotation_range: Tuple[float, float] = (-90, 0),
    focal_range: Tuple[float, float] = (0.75, 1.35),
    jitter_std: float = 0.01,
    center: Tuple[float, float, float] = (0, 0, 0),
) -> Tuple[Cameras, Tensor, Tensor]:
    """generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius_mean: mean radius of the orbit camera.
        radius_std: standard deviation of the radius of the orbit camera.
        central_rotation_range: amount that we rotate around the center of the object
        vertical_rotation_range: amount that we allow the cameras to pan up and down from horizontal
        focal_range: focal length range
        jitter_std: standard deviation of the jitter added to the camera position
        center: center of the object
    Return:
        poses: [size, 4, 4]
    """

    vertical_rotation_range = (
        vertical_rotation_range[0] + 90,
        vertical_rotation_range[1] + 90,
    )
    # This is the uniform sample on the part of the sphere we care about where 0 = 0 degrees and 1 = 360 degrees
    sampled_uniform = (
        torch.rand(size) * (vertical_rotation_range[1] - vertical_rotation_range[0]) + vertical_rotation_range[0]
    ) / 180
    vertical_rotation = torch.arccos(1 - 2 * sampled_uniform)
    central_rotation = torch.deg2rad(
        torch.rand(size) * (central_rotation_range[1] - central_rotation_range[0]) + central_rotation_range[0]
    )

    c_cos = torch.cos(central_rotation)
    c_sin = torch.sin(central_rotation)
    v_cos = torch.cos(vertical_rotation)
    v_sin = torch.sin(vertical_rotation)
    zeros = torch.zeros_like(central_rotation)
    ones = torch.ones_like(central_rotation)

    rot_z = torch.stack(
        [
            torch.stack([c_cos, -c_sin, zeros], dim=-1),
            torch.stack([c_sin, c_cos, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    rot_x = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, v_cos, -v_sin], dim=-1),
            torch.stack([zeros, v_sin, v_cos], dim=-1),
        ],
        dim=-2,
    )

    # Default directions are facing in the -z direction, so origins should face opposite way
    origins = torch.stack([torch.tensor([0, 0, 1])] * size, dim=0)
    origins = (origins * radius_mean) + (origins * (torch.randn((origins.shape)) * radius_std))
    R = torch.bmm(rot_z, rot_x)  # Want to have Rx @ Ry @ origin
    t = (
        torch.bmm(R, origins.unsqueeze(-1))
        + torch.randn((size, 3, 1)) * jitter_std
        + torch.tensor(center)[None, :, None]
    )
    camera_to_worlds = torch.cat([R, t], dim=-1)

    focals = torch.rand(size) * (focal_range[1] - focal_range[0]) + focal_range[0]

    cameras = Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=focals * resolution,
        fy=focals * resolution,
        cx=resolution / 2,
        cy=resolution / 2,
    ).to(device)

    return cameras, torch.rad2deg(vertical_rotation), torch.rad2deg(central_rotation)


@dataclass
class RandomCamerasDataManagerConfig(DataManagerConfig):
    """Configuration for data manager that does not load from a dataset. Instead, it generates random poses."""

    _target: Type = field(default_factory=lambda: RandomCamerasDataManager)
    train_resolution: int = 64
    """Training resolution"""
    eval_resolution: int = 64
    """Evaluation resolution"""
    num_eval_angles: int = 256
    """Number of evaluation angles"""
    train_images_per_batch: int = 1
    """Number of images per batch for training"""
    eval_images_per_batch: int = 1
    """Number of images per batch for evaluation"""
    radius_mean: float = 2.5
    """Mean radius of camera orbit"""
    radius_std: float = 0.1
    """Std of radius of camera orbit"""
    focal_range: Tuple[float, float] = (0.7, 1.35)
    """Range of focal length"""
    vertical_rotation_range: Tuple[float, float] = (-90, 0)
    """Range of vertical rotation"""
    jitter_std: float = 0.05
    """Std of camera direction jitter, so we don't just point the cameras towards the center every time"""
    center: Tuple[float, float, float] = (0, 0, 0)
    """Center coordinate of the camera sphere"""
    horizontal_rotation_warmup: int = 0
    """How many steps until the full horizontal rotation range is used"""


class RandomCamerasDataManager(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: RandomCamerasDataManagerConfig

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        config: RandomCamerasDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"

        if self.config.data is not None:
            CONSOLE.print("[red] --data should not be used with the RandomCamerasDataManager[/red]")
            sys.exit(1)

        cameras, _, _ = random_train_pose(
            self.config.num_eval_angles,
            self.config.eval_resolution,
            device=self.device,
            radius_mean=self.config.radius_mean,
            radius_std=self.config.radius_std,
            focal_range=self.config.focal_range,
            central_rotation_range=(-180, 180),
            vertical_rotation_range=self.config.vertical_rotation_range,
            jitter_std=self.config.jitter_std,
            center=self.config.center,
        )

        self.train_dataset = TrivialDataset(cameras)
        self.eval_dataset = TrivialDataset(cameras)

        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

        # pylint: disable=non-parent-init-called
        DataManager.__init__(self)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""

        self.train_count += 1
        horizontal_range = min((step / max(1, self.config.horizontal_rotation_warmup)), 1) * 180

        cameras, vertical_rotation, central_rotation = random_train_pose(
            self.config.train_images_per_batch,
            self.config.train_resolution,
            device=self.device,
            radius_mean=self.config.radius_mean,
            radius_std=self.config.radius_std,
            focal_range=self.config.focal_range,
            vertical_rotation_range=self.config.vertical_rotation_range,
            jitter_std=self.config.jitter_std,
            center=self.config.center,
            central_rotation_range=(-horizontal_range, horizontal_range),
        )
        ray_bundle = cameras.generate_rays(torch.tensor(list(range(self.config.train_images_per_batch)))).flatten()

        return ray_bundle, {
            "vertical": vertical_rotation,
            "central": central_rotation,
            "initialization": True,
        }

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1

        cameras, vertical_rotation, central_rotation = random_train_pose(
            self.config.eval_images_per_batch,
            self.config.eval_resolution,
            device=self.device,
            radius_mean=self.config.radius_mean,
            radius_std=self.config.radius_std,
            focal_range=self.config.focal_range,
            vertical_rotation_range=self.config.vertical_rotation_range,
            jitter_std=self.config.jitter_std,
            center=self.config.center,
        )
        ray_bundle = cameras.generate_rays(
            torch.tensor([[i] for i in range(self.config.train_images_per_batch)])
        ).flatten()

        return ray_bundle, {"vertical": vertical_rotation, "central": central_rotation}

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        for camera, batch in self.eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_resolution**2

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_resolution**2

    def get_param_groups(
        self,
    ) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        return param_groups

```

## datamanagers

### base_datamanager.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Datamanager.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

import torch
import tyro
from torch import nn
from torch.nn import Parameter
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import TypeVar

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig, PixelSampler, PixelSamplerConfig
from nerfstudio.data.utils.dataloaders import CacheDataloader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper, get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


def variable_res_collate(batch: List[Dict]) -> Dict:
    """Default collate function for the cached dataloader.
    Args:
        batch: Batch of samples from the dataset.
    Returns:
        Collated batch.
    """
    images = []
    imgdata_lists = defaultdict(list)
    for data in batch:
        image = data.pop("image")
        images.append(image)
        topop = []
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                # if the value has same height and width as the image, assume that it should be collated accordingly.
                if len(val.shape) >= 2 and val.shape[:2] == image.shape[:2]:
                    imgdata_lists[key].append(val)
                    topop.append(key)
        # now that iteration is complete, the image data items can be removed from the batch
        for key in topop:
            del data[key]

    new_batch = nerfstudio_collate(batch)
    new_batch["image"] = images
    new_batch.update(imgdata_lists)

    return new_batch


@dataclass
class DataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: DataManager)
    """Target class to instantiate."""
    data: Optional[Path] = None
    """Source of data, may not be used by all models."""
    masks_on_gpu: bool = False
    """Process masks on GPU for speed at the expense of memory, if True."""
    images_on_gpu: bool = False
    """Process images on GPU for speed at the expense of memory, if True."""


class DataManager(nn.Module):
    """Generic data manager's abstract class

    This version of the data manager is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test data managers. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:

    1. 'rays': This will contain the rays or camera we are sampling, with latents and
        conditionals attached (everything needed at inference)
    2. A "batch" of auxiliary information: This will contain the mask, the ground truth
        pixels, etc needed to actually train, score, etc the model

    Rationale:
    Because of this abstraction we've added, we can support more NeRF paradigms beyond the
    vanilla nerf paradigm of single-scene, fixed-images, no-learnt-latents.
    We can now support variable scenes, variable number of images, and arbitrary latents.


    Train Methods:
        setup_train: sets up for being used as train
        iter_train: will be called on __iter__() for the train iterator
        next_train: will be called on __next__() for the training iterator
        get_train_iterable: utility that gets a clean pythonic iterator for your training data

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: will be called on __iter__() for the eval iterator
        next_eval: will be called on __next__() for the eval iterator
        get_eval_iterable: utility that gets a clean pythonic iterator for your eval data


    Attributes:
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually
        train_dataset (Dataset): the dataset for the train dataset
        eval_dataset (Dataset): the dataset for the eval dataset
        includes_time (bool): whether the dataset includes time information

        Additional attributes specific to each subclass are defined in the setup_train and setup_eval
        functions.

    """

    train_dataset: Optional[InputDataset] = None
    eval_dataset: Optional[InputDataset] = None
    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None
    includes_time: bool = False

    def __init__(self):
        """Constructor for the DataManager class.

        Subclassed DataManagers will likely need to override this constructor.

        If you aren't manually calling the setup_train and setup_eval functions from an overriden
        constructor, that you call super().__init__() BEFORE you initialize any
        nn.Modules or nn.Parameters, but AFTER you've already set all the attributes you need
        for the setup functions."""
        super().__init__()
        self.train_count = 0
        self.eval_count = 0
        if self.train_dataset and self.test_mode != "inference":
            self.setup_train()
        if self.eval_dataset and self.test_mode != "inference":
            self.setup_eval()

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def iter_train(self):
        """The __iter__ function for the train iterator.

        This only exists to assist the get_train_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.train_count = 0

    def iter_eval(self):
        """The __iter__ function for the eval iterator.

        This only exists to assist the get_eval_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.eval_count = 0

    def get_train_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_train and next_train functions
        as __iter__ and __next__ methods respectively.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_train_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_eval and next_eval functions
        as __iter__ and __next__ methods respectively.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_eval_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_eval, self.next_eval, length)

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""

    @abstractmethod
    def next_train(self, step: int) -> Tuple[Union[RayBundle, Cameras], Dict]:
        """Returns the next batch of data from the train data manager.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the ray bundle for the image, and a dictionary of additional batch information
            such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple[Union[RayBundle, Cameras], Dict]:
        """Returns the next batch of data from the eval data manager.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the ray/camera for the image, and a dictionary of additional batch information
            such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Retrieve the next eval image.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the step number, the ray/camera for the image, and a dictionary of
            additional batch information such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        raise NotImplementedError

    @abstractmethod
    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        raise NotImplementedError

    @abstractmethod
    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        return []

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}


@dataclass
class VanillaDataManagerConfig(DataManagerConfig):
    """A basic data manager for a ray-based model"""

    _target: Type = field(default_factory=lambda: VanillaDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = field(default_factory=BlenderDataParserConfig)
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = -1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = -1
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    patch_size: int = 1
    """Size of patch to sample from. If > 1, patch-based sampling will be used."""

    # tyro.conf.Suppress prevents us from creating CLI arguments for this field.
    camera_optimizer: tyro.conf.Suppress[Optional[CameraOptimizerConfig]] = field(default=None)
    """Deprecated, has been moved to the model config."""
    pixel_sampler: PixelSamplerConfig = field(default_factory=PixelSamplerConfig)
    """Specifies the pixel sampler used to sample pixels from images."""

    def __post_init__(self):
        """Warn user of camera optimizer change."""
        if self.camera_optimizer is not None:
            import warnings

            CONSOLE.print(
                "\nCameraOptimizerConfig has been moved from the DataManager to the Model.\n", style="bold yellow"
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)


TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)


class VanillaDataManager(DataManager, Generic[TDataset]):
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: VanillaDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True and "mask" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True and "image" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("image")

        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break
        super().__init__()

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[VanillaDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is VanillaDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is VanillaDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is VanillaDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1 and type(self.config.pixel_sampler) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")

        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None:
            fisheye_crop_radius = dataset.cameras.metadata.get("fisheye_crop_radius")

        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
        )

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        for camera, batch in self.eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        if self.train_pixel_sampler is not None:
            return self.train_pixel_sampler.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        if self.eval_pixel_sampler is not None:
            return self.eval_pixel_sampler.num_rays_per_batch
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

```

## datamanagers

### full_images_datamanager.py

```python
# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Data manager that outputs cameras / images instead of raybundles

Good for things like gaussian splatting which require full cameras instead of the standard ray
paradigm
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import cv2
import numpy as np
import torch
from rich.progress import track
from torch.nn import Parameter

from nerfstudio.cameras.camera_utils import fisheye624_project, fisheye624_unproject_helper
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class FullImageDatamanagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: FullImageDatamanager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=NerfstudioDataParserConfig)
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    cache_images: Literal["cpu", "gpu"] = "cpu"
    """Whether to cache images in memory. If "cpu", caches on cpu. If "gpu", caches on device."""
    cache_images_type: Literal["uint8", "float32"] = "float32"
    """The image type returned from manager, caching images in uint8 saves memory"""
    max_thread_workers: Optional[int] = None
    """The maximum number of threads to use for caching images. If None, uses all available threads."""


class FullImageDatamanager(DataManager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    config: FullImageDatamanagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(
        self,
        config: FullImageDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
            CONSOLE.print(
                "Train dataset has over 500 images, overriding cache_images to cpu",
                style="bold yellow",
            )
            self.config.cache_images = "cpu"
        self.cached_train, self.cached_eval = self.cache_images(self.config.cache_images)
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        assert len(self.train_unseen_cameras) > 0, "No data found in dataset"

        super().__init__()

    def cache_images(self, cache_images_option):
        cached_train = []
        cached_eval = []

        def process_train_data(idx):
            # cv2.undistort the images / cameras
            data = self.train_dataset.get_data(idx, image_type=self.config.cache_images_type)
            camera = self.train_dataset.cameras[idx].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is None:
                return data
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
            data["image"] = torch.from_numpy(image)
            if mask is not None:
                data["mask"] = mask

            self.train_dataset.cameras.fx[idx] = float(K[0, 0])
            self.train_dataset.cameras.fy[idx] = float(K[1, 1])
            self.train_dataset.cameras.cx[idx] = float(K[0, 2])
            self.train_dataset.cameras.cy[idx] = float(K[1, 2])
            self.train_dataset.cameras.width[idx] = image.shape[1]
            self.train_dataset.cameras.height[idx] = image.shape[0]
            return data

        def process_eval_data(idx):
            # cv2.undistort the images / cameras
            data = self.eval_dataset.get_data(idx, image_type=self.config.cache_images_type)
            camera = self.eval_dataset.cameras[idx].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is None:
                return data
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
            data["image"] = torch.from_numpy(image)
            if mask is not None:
                data["mask"] = mask

            self.eval_dataset.cameras.fx[idx] = float(K[0, 0])
            self.eval_dataset.cameras.fy[idx] = float(K[1, 1])
            self.eval_dataset.cameras.cx[idx] = float(K[0, 2])
            self.eval_dataset.cameras.cy[idx] = float(K[1, 2])
            self.eval_dataset.cameras.width[idx] = image.shape[1]
            self.eval_dataset.cameras.height[idx] = image.shape[0]
            return data

        CONSOLE.log("Caching / undistorting train images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            cached_train = list(
                track(
                    executor.map(
                        process_train_data,
                        range(len(self.train_dataset)),
                    ),
                    description="Caching / undistorting train images",
                    transient=True,
                    total=len(self.train_dataset),
                )
            )

        CONSOLE.log("Caching / undistorting eval images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            cached_eval = list(
                track(
                    executor.map(
                        process_eval_data,
                        range(len(self.eval_dataset)),
                    ),
                    description="Caching / undistorting eval images",
                    transient=True,
                    total=len(self.eval_dataset),
                )
            )

        if cache_images_option == "gpu":
            for cache in cached_train:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
            for cache in cached_eval:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
        else:
            for cache in cached_train:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
            for cache in cached_eval:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()

        return cached_train, cached_eval

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[FullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is FullImageDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is FullImageDatamanager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is FullImageDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        """Sets up the data loaders for training"""

    def setup_eval(self):
        """Sets up the data loader for evaluation"""

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        image_indices = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval)
        _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            cameras.append(_cameras[i : i + 1])
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return list(zip(cameras, data))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def get_train_rays_per_batch(self):
        # TODO: fix this to be the resolution of the last image rendered
        return 800 * 800

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        data = deepcopy(self.cached_train[image_idx])
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data


def _undistort_image(
    camera: Cameras, distortion_params: np.ndarray, data: dict, image: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    mask = None
    if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
        distortion_params = np.array(
            [
                distortion_params[0],
                distortion_params[1],
                distortion_params[4],
                distortion_params[5],
                distortion_params[2],
                distortion_params[3],
                0,
                0,
            ]
        )
        if np.any(distortion_params):
            newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
            image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
        else:
            newK = K
            roi = 0, 0, image.shape[1], image.shape[0]
        # crop the image and update the intrinsics accordingly
        x, y, w, h = roi
        image = image[y : y + h, x : x + w]
        if "depth_image" in data:
            data["depth_image"] = data["depth_image"][y : y + h, x : x + w]
        if "mask" in data:
            mask = data["mask"].numpy()
            mask = mask.astype(np.uint8) * 255
            if np.any(distortion_params):
                mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
            mask = mask[y : y + h, x : x + w]
            mask = torch.from_numpy(mask).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        K = newK

    elif camera.camera_type.item() == CameraType.FISHEYE.value:
        distortion_params = np.array(
            [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
        )
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
        )
        # and then remap:
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        if "mask" in data:
            mask = data["mask"].numpy()
            mask = mask.astype(np.uint8) * 255
            mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
            mask = torch.from_numpy(mask).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        K = newK
    elif camera.camera_type.item() == CameraType.FISHEYE624.value:
        fisheye624_params = torch.cat(
            [camera.fx, camera.fy, camera.cx, camera.cy, torch.from_numpy(distortion_params)], dim=0
        )
        assert fisheye624_params.shape == (16,)
        assert (
            "mask" not in data
            and camera.metadata is not None
            and "fisheye_crop_radius" in camera.metadata
            and isinstance(camera.metadata["fisheye_crop_radius"], float)
        )
        fisheye_crop_radius = camera.metadata["fisheye_crop_radius"]

        # Approximate the FOV of the unmasked region of the camera.
        upper, lower, left, right = fisheye624_unproject_helper(
            torch.tensor(
                [
                    [camera.cx, camera.cy - fisheye_crop_radius],
                    [camera.cx, camera.cy + fisheye_crop_radius],
                    [camera.cx - fisheye_crop_radius, camera.cy],
                    [camera.cx + fisheye_crop_radius, camera.cy],
                ],
                dtype=torch.float32,
            )[None],
            params=fisheye624_params[None],
        ).squeeze(dim=0)
        fov_radians = torch.max(
            torch.acos(torch.sum(upper * lower / torch.linalg.norm(upper) / torch.linalg.norm(lower))),
            torch.acos(torch.sum(left * right / torch.linalg.norm(left) / torch.linalg.norm(right))),
        )

        # Heuristics to determine parameters of an undistorted image.
        undist_h = int(fisheye_crop_radius * 2)
        undist_w = int(fisheye_crop_radius * 2)
        undistort_focal = undist_h / (2 * torch.tan(fov_radians / 2.0))
        undist_K = torch.eye(3)
        undist_K[0, 0] = undistort_focal  # fx
        undist_K[1, 1] = undistort_focal  # fy
        undist_K[0, 2] = (undist_w - 1) / 2.0  # cx; for a 1x1 image, center should be at (0, 0).
        undist_K[1, 2] = (undist_h - 1) / 2.0  # cy

        # Undistorted 2D coordinates -> rays -> reproject to distorted UV coordinates.
        undist_uv_homog = torch.stack(
            [
                *torch.meshgrid(
                    torch.arange(undist_w, dtype=torch.float32),
                    torch.arange(undist_h, dtype=torch.float32),
                ),
                torch.ones((undist_w, undist_h), dtype=torch.float32),
            ],
            dim=-1,
        )
        assert undist_uv_homog.shape == (undist_w, undist_h, 3)
        dist_uv = (
            fisheye624_project(
                xyz=(
                    torch.einsum(
                        "ij,bj->bi",
                        torch.linalg.inv(undist_K),
                        undist_uv_homog.reshape((undist_w * undist_h, 3)),
                    )[None]
                ),
                params=fisheye624_params[None, :],
            )
            .reshape((undist_w, undist_h, 2))
            .numpy()
        )
        map1 = dist_uv[..., 1]
        map2 = dist_uv[..., 0]

        # Use correspondence to undistort image.
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

        # Compute undistorted mask as well.
        dist_h = camera.height.item()
        dist_w = camera.width.item()
        mask = np.mgrid[:dist_h, :dist_w]
        mask[0, ...] -= dist_h // 2
        mask[1, ...] -= dist_w // 2
        mask = np.linalg.norm(mask, axis=0) < fisheye_crop_radius
        mask = torch.from_numpy(
            cv2.remap(
                mask.astype(np.uint8) * 255,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            / 255.0
        ).bool()[..., None]
        if len(mask.shape) == 2:
            mask = mask[:, :, None]
        assert mask.shape == (undist_h, undist_w, 1)
        K = undist_K.numpy()
    else:
        raise NotImplementedError("Only perspective and fisheye cameras are supported")

    return K, image, mask

```

## dataparsers

### scannetpp_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for ScanNet++ datasets. """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ScanNetppDataParserConfig(DataParserConfig):
    """ScanNet++ dataset config.
    ScanNet++ dataset (https://kaldir.vc.in.tum.de/scannetpp/) is a real-world 3D indoor dataset for semantics understanding and novel view synthesis.
    This dataparser follow the file structure of the dataset.
    Expected structure of the directory:

    .. code-block:: text

        root/
        ├── SCENE_ID0
            ├── dslr
                ├── resized_images
                ├── resized_anon_masks
                ├── nerfstudio/transforms.json
        ├── SCENE_ID1/
        ...
    """

    _target: Type = field(default_factory=lambda: ScanNetpp)
    """target class to instantiate"""
    data: Path = Path("scannetpp/410c470782")
    """Directory to the root of the data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.5
    """How much to scale the region of interest by. Default is 1.5 since the cameras are inside the rooms."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    images_dir: Path = Path("dslr/resized_images")
    """Relative path to the images directory (default: resized_images)"""
    masks_dir: Path = Path("dslr/resized_anon_masks")
    """Relative path to the masks directory (default: resized_anon_masks)"""
    transforms_path: Path = Path("dslr/nerfstudio/transforms.json")
    """Relative path to the transforms.json file"""


@dataclass
class ScanNetpp(DataParser):
    """ScanNet++ DatasetParser"""

    config: ScanNetppDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        meta = load_from_json(self.config.data / self.config.transforms_path)
        data_dir = self.config.data / self.config.images_dir
        mask_dir = self.config.data / self.config.masks_dir

        image_filenames = []
        mask_filenames = []
        poses = []
        i_train = []
        i_eval = []
        # sort the frames by fname
        frames = meta["frames"] + meta["test_frames"]
        test_frames = [f["file_path"] for f in meta["test_frames"]]
        frames.sort(key=lambda x: x["file_path"])

        for idx, frame in enumerate(frames):
            filepath = Path(frame["file_path"])
            fname = data_dir / filepath

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if meta.get("has_mask", True) and "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = mask_dir / mask_filepath
                mask_filenames.append(mask_fname)

            if frame["file_path"] in test_frames:
                i_eval.append(idx)
            else:
                i_train.append(idx)

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        if not self.config.auto_scale_poses:
            # Set aabb_scale to scene_scale * the max of the absolute values of the poses
            aabb_scale = self.config.scene_scale * float(torch.max(torch.abs(poses[:, :3, 3])))
        else:
            aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"])
        fy = float(meta["fl_y"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        height = int(meta["h"])
        width = int(meta["w"])
        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta["k1"]) if "k1" in meta else 0.0,
            k2=float(meta["k2"]) if "k2" in meta else 0.0,
            k3=float(meta["k3"]) if "k3" in meta else 0.0,
            k4=float(meta["k4"]) if "k4" in meta else 0.0,
            p1=float(meta["p1"]) if "p1" in meta else 0.0,
            p2=float(meta["p2"]) if "p2" in meta else 0.0,
        )

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={},
        )
        return dataparser_outputs

```

## dataparsers

### instant_ngp_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for instant ngp data"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple, Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class InstantNGPDataParserConfig(DataParserConfig):
    """Instant-NGP dataset parser config"""

    _target: Type = field(default_factory=lambda: InstantNGP)
    """target class to instantiate"""
    data: Path = Path("data/ours/posterv2")
    """Directory or explicit json file path specifying location of data."""
    scene_scale: float = 0.3333
    """How much to scale the scene."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""


@dataclass
class InstantNGP(DataParser):
    """Instant NGP Dataset"""

    config: InstantNGPDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        poses = []
        num_skipped_image_filenames = 0
        for frame in meta["frames"]:
            fname = data_dir / Path(frame["file_path"])
            # search for png file
            if not fname.exists():
                fname = data_dir / Path(frame["file_path"] + ".png")
            if not fname.exists():
                CONSOLE.log(f"couldn't find {fname} image")
                num_skipped_image_filenames += 1
            else:
                if "w" not in meta:
                    img_0 = imageio.imread(fname)
                    h, w = img_0.shape[:2]
                    meta["w"] = w
                    if "h" in meta:
                        meta_h = meta["h"]
                        assert meta_h == h, f"height of image dont not correspond metadata {h} != {meta_h}"
                    else:
                        meta["h"] = h
                image_filenames.append(fname)
                poses.append(np.array(frame["transform_matrix"]))
                if "mask_path" in frame:
                    mask_fname = data_dir / Path(frame["mask_path"])
                    mask_filenames.append(mask_fname)
        if num_skipped_image_filenames >= 0:
            CONSOLE.print(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert len(image_filenames) != 0, """
        No image files found.
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        poses = np.array(poses).astype(np.float32)
        poses[:, :3, 3] *= self.config.scene_scale

        # find train and eval indices based on the eval_mode specified
        if self.config.eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
        elif self.config.eval_mode == "filename":
            i_train, i_eval = get_train_eval_split_filename(image_filenames)
        elif self.config.eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
        elif self.config.eval_mode == "all":
            CONSOLE.log(
                "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
            )
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        else:
            raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta.get("k1", 0)),
            k2=float(meta.get("k2", 0)),
            k3=float(meta.get("k3", 0)),
            k4=float(meta.get("k4", 0)),
            p1=float(meta.get("p1", 0)),
            p2=float(meta.get("p2", 0)),
        )

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = 0.5 * meta.get("aabb_scale", 1)

        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        fl_x, fl_y = InstantNGP.get_focal_lengths(meta)

        w, h = meta["w"], meta["h"]

        camera_type = CameraType.PERSPECTIVE
        if meta.get("is_fisheye", False):
            camera_type = CameraType.FISHEYE

        cameras = Cameras(
            fx=float(fl_x),
            fy=float(fl_y),
            cx=float(meta.get("cx", 0.5 * w)),
            cy=float(meta.get("cy", 0.5 * h)),
            distortion_params=distortion_params,
            height=int(h),
            width=int(w),
            camera_to_worlds=camera_to_world,
            camera_type=camera_type,
        )

        # TODO(ethan): add alpha background color
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=self.config.scene_scale,
        )

        return dataparser_outputs

    @classmethod
    def get_focal_lengths(cls, meta: Dict) -> Tuple[float, float]:
        """Reads or computes the focal length from transforms dict.
        Args:
            meta: metadata from transforms.json file.
        Returns:
            Focal lengths in the x and y directions. Error is raised if these cannot be calculated.
        """
        fl_x, fl_y = 0, 0

        def fov_to_focal_length(rad, res):
            return 0.5 * res / np.tan(0.5 * rad)

        if "fl_x" in meta:
            fl_x = meta["fl_x"]
        elif "x_fov" in meta:
            fl_x = fov_to_focal_length(np.deg2rad(meta["x_fov"]), meta["w"])
        elif "camera_angle_x" in meta:
            fl_x = fov_to_focal_length(meta["camera_angle_x"], meta["w"])

        if "camera_angle_y" not in meta or "y_fov" not in meta:
            fl_y = fl_x
        else:
            if "fl_y" in meta:
                fl_y = meta["fl_y"]
            elif "y_fov" in meta:
                fl_y = fov_to_focal_length(np.deg2rad(meta["y_fov"]), meta["h"])
            elif "camera_angle_y" in meta:
                fl_y = fov_to_focal_length(meta["camera_angle_y"], meta["h"])

        if fl_x == 0 or fl_y == 0:
            raise AttributeError("Focal length cannot be calculated from transforms.json (missing fields).")

        return (fl_x, fl_y)

```

## dataparsers

### nerfosr_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for NeRF-OSR datasets

    Presented in the paper: https://4dqv.mpi-inf.mpg.de/NeRF-OSR/

"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Tuple, Type

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox


def _find_files(directory: str, exts: List[str]):
    """Find all files in a directory that have a certain file extension.

    Args:
        directory : The directory to search for files.
        exts :  A list of file extensions to search for. Each file extension should be in the form '*.ext'.

    Returns:
        A list of file paths for all the files that were found. The list is sorted alphabetically.
    """
    if os.path.isdir(directory):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(directory, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    return []


def _parse_osm_txt(filename: str):
    """Parse a text file containing numbers and return a 4x4 numpy array of float32 values.

    Args:
        filename : a file containing numbers in a 4x4 matrix.

    Returns:
        A numpy array of shape [4, 4] containing the numbers from the file.
    """
    assert os.path.isfile(filename)
    with open(filename, encoding="UTF-8") as f:
        nums = f.read().split()
    return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)


def get_camera_params(
    scene_dir: str, split: Literal["train", "validation", "test"]
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Load camera intrinsic and extrinsic parameters for a given scene split.

    Args"
      scene_dir : The directory containing the scene data.
      split : The split for which to load the camera parameters.

    Returns
        A tuple containing the intrinsic parameters (as a torch.Tensor of shape [N, 4, 4]),
        the camera-to-world matrices (as a torch.Tensor of shape [N, 4, 4]), and the number of cameras (N).
    """
    split_dir = f"{scene_dir}/{split}"

    # camera parameters files
    intrinsics_files = _find_files(f"{split_dir}/intrinsics", exts=["*.txt"])
    pose_files = _find_files(f"{split_dir}/pose", exts=["*.txt"])

    num_cams = len(pose_files)

    intrinsics = []
    camera_to_worlds = []
    for i in range(num_cams):
        intrinsics.append(_parse_osm_txt(intrinsics_files[i]))

        pose = _parse_osm_txt(pose_files[i])

        # convert from COLMAP/OpenCV to nerfstudio camera (OpenGL/Blender)
        pose[0:3, 1:3] *= -1

        camera_to_worlds.append(pose)

    intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))  # [N, 4, 4]
    camera_to_worlds = torch.from_numpy(np.stack(camera_to_worlds).astype(np.float32))  # [N, 4, 4]

    return intrinsics, camera_to_worlds, num_cams


@dataclass
class NeRFOSRDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: NeRFOSR)
    """target class to instantiate"""
    data: Path = Path("data/NeRF-OSR/Data/")
    """Directory specifying location of data."""
    scene: str = "stjacob"
    """Which scene to load"""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    use_masks: bool = False
    """Whether to use masks."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "vertical"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "focus"
    """The method to use for centering."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""


@dataclass
class NeRFOSR(DataParser):
    """NeRFOSR Dataparser
    Presented in the paper: https://4dqv.mpi-inf.mpg.de/NeRF-OSR/

    Some of this code comes from https://github.com/r00tman/NeRF-OSR/blob/main/data_loader_split.py

    Source data convention is:
      camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
      poses is camera-to-world
      masks are 0 for dynamic content, 255 for static content
    """

    config: NeRFOSRDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        data = self.config.data
        scene = self.config.scene
        split = "validation" if split == "val" else split

        if scene == "trevi":
            scene_dir = f"{data}/{scene}/final_clean"
            split_dir = f"{data}/{scene}/final_clean/{split}"
        else:
            scene_dir = f"{data}/{scene}/final"
            split_dir = f"{data}/{scene}/final/{split}"

        # get all split cam params
        intrinsics_train, camera_to_worlds_train, n_train = get_camera_params(scene_dir, "train")
        intrinsics_val, camera_to_worlds_val, n_val = get_camera_params(scene_dir, "validation")
        intrinsics_test, camera_to_worlds_test, _ = get_camera_params(scene_dir, "test")

        # combine all cam params
        intrinsics = torch.cat([intrinsics_train, intrinsics_val, intrinsics_test], dim=0)
        camera_to_worlds = torch.cat([camera_to_worlds_train, camera_to_worlds_val, camera_to_worlds_test], dim=0)

        camera_to_worlds, _ = camera_utils.auto_orient_and_center_poses(
            camera_to_worlds,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(camera_to_worlds[:, :3, 3]))

        camera_to_worlds[:, :3, 3] *= scale_factor * self.config.scale_factor

        if split == "train":
            camera_to_worlds = camera_to_worlds[:n_train]
            intrinsics = intrinsics[:n_train]
        elif split == "validation":
            camera_to_worlds = camera_to_worlds[n_train : n_train + n_val]
            intrinsics = intrinsics[n_train : n_train + n_val]
        elif split == "test":
            camera_to_worlds = camera_to_worlds[n_train + n_val :]
            intrinsics = intrinsics[n_train + n_val :]

        cameras = Cameras(
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            camera_type=CameraType.PERSPECTIVE,
        )

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        # --- images ---
        image_filenames = _find_files(f"{split_dir}/rgb", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])

        # --- masks ---
        mask_filenames = []
        if self.config.use_masks:
            mask_filenames = _find_files(f"{split_dir}/mask", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=self.config.scale_factor,
        )
        return dataparser_outputs

```

## dataparsers

### minimal_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Data parser for pre-prepared datasets for all cameras, with no additional processing needed
Optional fields - semantics, mask_filenames, cameras.distortion_params, cameras.times
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs, Semantics
from nerfstudio.data.scene_box import SceneBox


@dataclass
class MinimalDataParserConfig(DataParserConfig):
    """Minimal dataset config"""

    _target: Type = field(default_factory=lambda: MinimalDataParser)
    """target class to instantiate"""
    data: Path = Path("/home/nikhil/nerfstudio-main/tests/data/lego_test/minimal_parser")


@dataclass
class MinimalDataParser(DataParser):
    """Minimal DatasetParser"""

    config: MinimalDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        filepath = self.config.data / f"{split}.npz"
        data = np.load(filepath, allow_pickle=True)

        image_filenames = [filepath.parent / path for path in data["image_filenames"].tolist()]
        mask_filenames = None
        if "mask_filenames" in data.keys():
            mask_filenames = [filepath.parent / path for path in data["mask_filenames"].tolist()]

        if "semantics" in data.keys():
            semantics = data["semantics"].item()
            metadata = {
                "semantics": Semantics(
                    filenames=[filepath.parent / path for path in semantics["filenames"].tolist()],
                    classes=semantics["classes"].tolist(),
                    colors=torch.from_numpy(semantics["colors"]),
                    mask_classes=semantics["mask_classes"].tolist(),
                )
            }
        else:
            metadata = {}

        scene_box_aabb = torch.from_numpy(data["scene_box"])
        scene_box = SceneBox(aabb=scene_box_aabb)

        camera_np = data["cameras"].item()
        distortion_params = None
        if "distortion_params" in camera_np.keys():
            distortion_params = torch.from_numpy(camera_np["distortion_params"])
        cameras = Cameras(
            fx=torch.from_numpy(camera_np["fx"]),
            fy=torch.from_numpy(camera_np["fy"]),
            cx=torch.from_numpy(camera_np["cx"]),
            cy=torch.from_numpy(camera_np["cy"]),
            distortion_params=distortion_params,
            height=torch.from_numpy(camera_np["height"]),
            width=torch.from_numpy(camera_np["width"]),
            camera_to_worlds=torch.from_numpy(camera_np["camera_to_worlds"])[:, :3, :4],
            camera_type=torch.from_numpy(camera_np["camera_type"]),
            times=torch.from_numpy(camera_np["times"]) if "times" in camera_np.keys() else None,
        )

        applied_scale = 1.0
        applied_transform = torch.eye(4, dtype=torch.float32)[:3, :]
        if "applied_scale" in data.keys():
            applied_scale = float(data["applied_scale"])
        if "applied_transform" in data.keys():
            applied_transform = data["applied_transform"].astype(np.float32)
            assert applied_transform.shape == (3, 4)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            dataparser_transform=applied_transform,
            dataparser_scale=applied_scale,
            metadata=metadata,
        )
        return dataparser_outputs

```

## dataparsers

### sitcoms3d_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for sitcoms3D dataset.

The dataset is from the paper ["The One Where They Reconstructed 3D Humans and
Environments in TV Shows"](https://ethanweber.me/sitcoms3D/)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs, Semantics
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class Sitcoms3DDataParserConfig(DataParserConfig):
    """sitcoms3D dataset parser config"""

    _target: Type = field(default_factory=lambda: Sitcoms3D)
    """target class to instantiate"""
    data: Path = Path("data/sitcoms3d/TBBT-big_living_room")
    """Directory specifying location of data."""
    include_semantics: bool = True
    """whether or not to include loading of semantics data"""
    downscale_factor: int = 4
    scene_scale: float = 2.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Sitcoms3D axis-aligned bbox will be scaled to this value.
    """


@dataclass
class Sitcoms3D(DataParser):
    """Sitcoms3D Dataset"""

    config: Sitcoms3DDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        cameras_json = load_from_json(self.config.data / "cameras.json")
        frames = cameras_json["frames"]
        bbox = torch.tensor(cameras_json["bbox"])

        downscale_suffix = f"_{self.config.downscale_factor}" if self.config.downscale_factor != 1 else ""
        images_folder = f"images{downscale_suffix}"
        segmentations_folder = f"segmentations{downscale_suffix}"

        image_filenames = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for frame in frames:
            # unpack data
            image_filename = self.config.data / images_folder / frame["image_name"]
            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])[:3]
            # append data
            image_filenames.append(image_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)
        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)

        # rotate the cameras and box 90 degrees about the x axis to put the z axis up
        rotation = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
        camera_to_worlds[:, :3] = rotation @ camera_to_worlds[:, :3]
        bbox = (rotation @ bbox.T).T

        scene_scale = self.config.scene_scale

        # -- set the scene box ---
        scene_box = SceneBox(aabb=bbox)
        # center the box and adjust the cameras too
        center = scene_box.get_center()
        scene_box.aabb -= center
        camera_to_worlds[..., 3] -= center
        # scale the longest dimension to match the cube size
        lengths = scene_box.aabb[1] - scene_box.aabb[0]
        longest_dim = torch.argmax(lengths)
        longest_length = lengths[longest_dim]
        scale = scene_scale / longest_length.item()
        scene_box.aabb = scene_box.aabb * scale  # box
        camera_to_worlds[..., 3] *= scale  # cameras

        # --- semantics ---
        semantics = None
        if self.config.include_semantics:
            empty_path = Path()
            replace_this_path = str(empty_path / images_folder / empty_path)
            with_this_path = str(empty_path / segmentations_folder / "thing" / empty_path)
            filenames = [
                Path(str(image_filename).replace(replace_this_path, with_this_path).replace(".jpg", ".png"))
                for image_filename in image_filenames
            ]
            panoptic_classes = load_from_json(self.config.data / "panoptic_classes.json")
            classes = panoptic_classes["thing"]
            colors = torch.tensor(panoptic_classes["thing_colors"], dtype=torch.float32) / 255.0
            semantics = Semantics(filenames=filenames, classes=classes, colors=colors, mask_classes=["person"])

        assert torch.all(cx[0] == cx), "Not all cameras have the same cx. Our Cameras class does not support this."
        assert torch.all(cy[0] == cy), "Not all cameras have the same cy. Our Cameras class does not support this."

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=float(cx[0]),
            cy=float(cy[0]),
            camera_to_worlds=camera_to_worlds,
            camera_type=CameraType.PERSPECTIVE,
        )
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata={"semantics": semantics} if self.config.include_semantics else {},
            dataparser_scale=scale,
        )
        return dataparser_outputs

```

## dataparsers

### phototourism_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phototourism dataset parser. Datasets and documentation here: http://phototour.cs.washington.edu/datasets/"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox

# TODO(1480) use pycolmap instead of colmap_parsing_utils
# import pycolmap
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary, read_images_binary
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class PhototourismDataParserConfig(DataParserConfig):
    """Phototourism dataset parser config"""

    _target: Type = field(default_factory=lambda: Phototourism)
    """target class to instantiate"""
    data: Path = Path("data/phototourism/brandenburg-gate")
    """Directory specifying location of data."""
    scale_factor: float = 3.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""


@dataclass
class Phototourism(DataParser):
    """Phototourism dataset. This is based on https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/phototourism.py
    and uses colmap's utils file to read the poses.
    """

    config: PhototourismDataParserConfig

    def __init__(self, config: PhototourismDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data

    def _generate_dataparser_outputs(self, split="train"):
        image_filenames = []
        poses = []

        with CONSOLE.status(f"[bold green]Reading phototourism images and poses for {split} split...") as _:
            # TODO(1480) use pycolmap
            # recon = pycolmap.Reconstruction(self.data / "dense" / "sparse")
            # cams = recon.cameras
            # imgs = recon.images
            cams = read_cameras_binary(self.data / "dense/sparse/cameras.bin")
            imgs = read_images_binary(self.data / "dense/sparse/images.bin")

        poses = []
        fxs = []
        fys = []
        cxs = []
        cys = []
        image_filenames = []

        flip = torch.eye(3)
        flip[0, 0] = -1.0
        flip = flip.double()

        for _id, cam in cams.items():
            img = imgs[_id]

            assert cam.model == "PINHOLE", "Only pinhole (perspective) camera model is supported at the moment"

            pose = torch.cat([torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))], dim=1)
            pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
            poses.append(torch.linalg.inv(pose))
            fxs.append(torch.tensor(cam.params[0]))
            fys.append(torch.tensor(cam.params[1]))
            cxs.append(torch.tensor(cam.params[2]))
            cys.append(torch.tensor(cam.params[3]))

            image_filenames.append(self.data / "dense/images" / img.name)

        poses = torch.stack(poses).float()
        poses[..., 1:3] *= -1
        fxs = torch.stack(fxs).float()
        fys = torch.stack(fys).float()
        cxs = torch.stack(cxs).float()
        cys = torch.stack(cys).float()

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_fraction)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        i_all = torch.tensor(i_all)
        i_train = torch.tensor(i_train, dtype=torch.long)
        i_eval = torch.tensor(i_eval, dtype=torch.long)
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses, method=self.config.orientation_method, center_method=self.config.center_method
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            camera_type=CameraType.PERSPECTIVE,
        )

        cameras = cameras[indices]
        image_filenames = [image_filenames[i] for i in indices]

        assert len(cameras) == len(image_filenames)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
        )

        return dataparser_outputs

```

## dataparsers

### __init__.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

## dataparsers

### sdfstudio_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Datapaser for sdfstudio formatted data"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class SDFStudioDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: SDFStudio)
    """target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
    include_mono_prior: bool = False
    """whether or not to load monocular depth and normal """
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    include_foreground_mask: bool = False
    """whether or not to load foreground mask"""
    downscale_factor: int = 1
    scene_scale: float = 2.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the axis-aligned bbox will be scaled to this value.
    """
    skip_every_for_val_split: int = 1
    """sub sampling validation images"""
    auto_orient: bool = True


@dataclass
class SDFStudio(DataParser):
    """SDFStudio Dataset"""

    config: SDFStudioDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        # load meta data
        meta = load_from_json(self.config.data / "meta_data.json")

        indices = list(range(len(meta["frames"])))
        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]

        image_filenames = []
        depth_filenames = []
        normal_filenames = []
        transform = None
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for i, frame in enumerate(meta["frames"]):
            if i not in indices:
                continue

            image_filename = self.config.data / frame["rgb_path"]
            depth_filename = frame.get("mono_depth_path")
            normal_filename = frame.get("mono_normal_path")

            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])

            # append data
            image_filenames.append(image_filename)
            if depth_filename is not None and normal_filename is not None:
                depth_filenames.append(self.config.data / depth_filename)
                normal_filenames.append(self.config.data / normal_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)

        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        c2w_colmap = torch.stack(camera_to_worlds)
        camera_to_worlds = torch.stack(camera_to_worlds)

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        if self.config.auto_orient:
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method="up",
                center_method="none",
            )

        # scene box from meta data
        meta_scene_box = meta["scene_box"]
        aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox(
            aabb=aabb,
        )

        height, width = meta["height"], meta["width"]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO supports downsample
        # cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)
        if self.config.include_mono_prior:
            assert meta["has_mono_prior"], f"no mono prior in {self.config.data}"

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "normal_filenames": normal_filenames if len(normal_filenames) > 0 else None,
                "transform": transform,
                # required for normal maps, these are in colmap format so they require c2w before conversion
                "camera_to_worlds": c2w_colmap if len(c2w_colmap) > 0 else None,
                "include_mono_prior": self.config.include_mono_prior,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs

```

## dataparsers

### scannet_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for ScanNet dataset"""
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import cv2
import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox


@dataclass
class ScanNetDataParserConfig(DataParserConfig):
    """ScanNet dataset config.
    ScanNet dataset (https://www.scan-net.org/) is a large-scale 3D dataset of indoor scenes.
    This dataparser assumes that the dense stream was extracted from .sens files.
    Expected structure of scene directory:

    .. code-block:: text

        root/
        ├── color/
        ├── depth/
        ├── intrinsic/
        ├── pose/
    """

    _target: Type = field(default_factory=lambda: ScanNet)
    """target class to instantiate"""
    data: Path = Path("data/scannet/scene0423_02")
    """Path to ScanNet folder with densely extracted scenes."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""


@dataclass
class ScanNet(DataParser):
    """ScanNet DatasetParser"""

    config: ScanNetDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        image_dir = self.config.data / "color"
        depth_dir = self.config.data / "depth"
        pose_dir = self.config.data / "pose"

        img_dir_sorted = list(sorted(image_dir.iterdir(), key=lambda x: int(x.name.split(".")[0])))
        depth_dir_sorted = list(sorted(depth_dir.iterdir(), key=lambda x: int(x.name.split(".")[0])))
        pose_dir_sorted = list(sorted(pose_dir.iterdir(), key=lambda x: int(x.name.split(".")[0])))

        first_img = cv2.imread(str(img_dir_sorted[0].absolute()))  # type: ignore
        h, w, _ = first_img.shape

        image_filenames, depth_filenames, intrinsics, poses = [], [], [], []

        K = np.loadtxt(self.config.data / "intrinsic" / "intrinsic_color.txt")
        for img, depth, pose in zip(img_dir_sorted, depth_dir_sorted, pose_dir_sorted):
            pose = np.loadtxt(pose)
            pose = np.array(pose).reshape(4, 4)
            pose[:3, 1] *= -1
            pose[:3, 2] *= -1
            pose = torch.from_numpy(pose).float()
            # We cannot accept files directly, as some of the poses are invalid
            if np.isinf(pose).any():
                continue

            poses.append(pose)
            intrinsics.append(K)
            image_filenames.append(img)
            depth_filenames.append(depth)

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_fraction)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = torch.from_numpy(np.stack(poses).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method="none",
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        intrinsics = intrinsics[indices.tolist()]
        poses = poses[indices.tolist()]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=h,
            width=w,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs

```

## dataparsers

### blender_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class BlenderDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: Blender)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""


@dataclass
class Blender(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: BlenderDataParserConfig

    def __init__(self, config: BlenderDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        if self.alpha_color is not None:
            self.alpha_color_tensor = get_color(self.alpha_color)
        else:
            self.alpha_color_tensor = None

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        poses = []
        for frame in meta["frames"]:
            fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs

```

## dataparsers

### colmap_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.prompt import Confirm

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command

MAX_AUTO_RESOLUTION = 1600


@dataclass
class ColmapDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: ColmapDataParser)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    assume_colmap_world_coordinate_convention: bool = True
    """Colmap optimized world often have y direction of the first camera pointing towards down direction,
    while nerfstudio world set z direction to be up direction for viewer. Therefore, we usually need to apply an extra
    transform when orientation_method=none. This parameter has no effects if orientation_method is set other than none.
    When this parameter is set to False, no extra transform is applied when reading data from colmap.
    """
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval (used by most academic papers, e.g. MipNerf360, GSplat).
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""

    images_path: Path = Path("images")
    """Path to images directory relative to the data path."""
    masks_path: Optional[Path] = None
    """Path to masks directory. If not set, masks are not loaded."""
    depths_path: Optional[Path] = None
    """Path to depth maps directory. If not set, depths are not loaded."""
    colmap_path: Path = Path("colmap/sparse/0")
    """Path to the colmap reconstruction directory relative to the data path."""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction. This is helpful for Gaussian splatting and
    generally unused otherwise, but it's typically harmless so we default to True."""
    max_2D_matches_per_3D_point: int = 0
    """Maximum number of 2D matches per 3D point. If set to -1, all 2D matches are loaded. If set to 0, no 2D matches are loaded."""


class ColmapDataParser(DataParser):
    """COLMAP DatasetParser.
    Expects a folder with the following structure:
        images/ # folder containing images used to create the COLMAP model
        sparse/0 # folder containing the COLMAP reconstruction (either TEXT or BINARY format)
        masks/ # (OPTIONAL) folder containing masks for each image
        depths/ # (OPTIONAL) folder containing depth maps for each image
    The paths can be different and can be specified in the config. (e.g., sparse/0 -> sparse)
    Currently, most COLMAP camera models are supported except for the FULL_OPENCV and THIN_PRISM_FISHEYE models.

    The dataparser loads the downscaled images from folders with `_{downscale_factor}` suffix.
    If these folders do not exist, the user can choose to automatically downscale the images and
    create these folders.

    The loader is compatible with the datasets processed using the ns-process-data script and
    can be used as a drop-in replacement. It further supports datasets like Mip-NeRF 360 (although
    in the case of Mip-NeRF 360 the downsampled images may have a different resolution because they
    use different rounding when computing the image resolution).
    """

    config: ColmapDataParserConfig

    def __init__(self, config: ColmapDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None

    def _get_all_images_and_cameras(self, recon_dir: Path):
        if (recon_dir / "cameras.txt").exists():
            cam_id_to_camera = colmap_utils.read_cameras_text(recon_dir / "cameras.txt")
            im_id_to_image = colmap_utils.read_images_text(recon_dir / "images.txt")
        elif (recon_dir / "cameras.bin").exists():
            cam_id_to_camera = colmap_utils.read_cameras_binary(recon_dir / "cameras.bin")
            im_id_to_image = colmap_utils.read_images_binary(recon_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt or cameras.bin in {recon_dir}")

        cameras = {}
        frames = []
        camera_model = None

        # Parse cameras
        for cam_id, cam_data in cam_id_to_camera.items():
            cameras[cam_id] = parse_colmap_camera_params(cam_data)

        # Parse frames
        # we want to sort all images based on im_id
        ordered_im_id = sorted(im_id_to_image.keys())
        for im_id in ordered_im_id:
            im_data = im_id_to_image[im_id]
            # NB: COLMAP uses Eigen / scalar-first quaternions
            # * https://colmap.github.io/format.html
            # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
            # the `rotation_matrix()` handles that format for us.
            rotation = colmap_utils.qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            if self.config.assume_colmap_world_coordinate_convention:
                # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)
                c2w = c2w[np.array([0, 2, 1, 3]), :]
                c2w[2, :] *= -1

            frame = {
                "file_path": (self.config.data / self.config.images_path / im_data.name).as_posix(),
                "transform_matrix": c2w,
                "colmap_im_id": im_id,
            }
            frame.update(cameras[im_data.camera_id])
            if self.config.masks_path is not None:
                frame["mask_path"] = (
                    (self.config.data / self.config.masks_path / im_data.name).with_suffix(".png").as_posix()
                )
            if self.config.depths_path is not None:
                frame["depth_path"] = (
                    (self.config.data / self.config.depths_path / im_data.name).with_suffix(".png").as_posix()
                )
            frames.append(frame)
            if camera_model is not None:
                assert camera_model == frame["camera_model"], "Multiple camera models are not supported"
            else:
                camera_model = frame["camera_model"]

        out = {}
        out["frames"] = frames
        if self.config.assume_colmap_world_coordinate_convention:
            # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)
            applied_transform = np.eye(4)[:3, :]
            applied_transform = applied_transform[np.array([0, 2, 1]), :]
            applied_transform[2, :] *= -1
            out["applied_transform"] = applied_transform.tolist()
        out["camera_model"] = camera_model
        assert len(frames) > 0, "No images found in the colmap model"
        return out

    def _get_image_indices(self, image_filenames, split):
        has_split_files_spec = (
            (self.config.data / "train_list.txt").exists()
            or (self.config.data / "test_list.txt").exists()
            or (self.config.data / "validation_list.txt").exists()
        )
        if (self.config.data / f"{split}_list.txt").exists():
            CONSOLE.log(f"Using {split}_list.txt to get indices for split {split}.")
            with (self.config.data / f"{split}_list.txt").open("r", encoding="utf8") as f:
                filenames = f.read().splitlines()
            # Validate split first
            split_filenames = set(self.config.data / self.config.images_path / x for x in filenames)
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(
                    f"Some filenames for split {split} were not found: {set(map(str, unmatched_filenames))}."
                )

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            elif self.config.eval_mode == "all":
                CONSOLE.log(
                    "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
                )
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")
        return indices

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            image_filenames.append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))
            if "depth_path" in frame:
                depth_filenames.append(Path(frame["depth_path"]))

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        indices = self._get_image_indices(image_filenames, split)
        image_filenames, mask_filenames, depth_filenames, downscale_factor = self._setup_downscale_factor(
            image_filenames, mask_filenames, depth_filenames
        )

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        cameras.rescale_output_resolution(scaling_factor=1.0 / downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        metadata = {}
        if self.config.load_3D_points:
            # Load 3D points
            metadata.update(self._load_3D_points(colmap_path, transform_matrix, scale_factor))

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,
            },
        )
        return dataparser_outputs

    def _load_3D_points(self, colmap_path: Path, transform_matrix: torch.Tensor, scale_factor: float):
        if (colmap_path / "points3D.bin").exists():
            colmap_points = colmap_utils.read_points3D_binary(colmap_path / "points3D.bin")
        elif (colmap_path / "points3D.txt").exists():
            colmap_points = colmap_utils.read_points3D_text(colmap_path / "points3D.txt")
        else:
            raise ValueError(f"Could not find points3D.txt or points3D.bin in {colmap_path}")
        points3D = torch.from_numpy(np.array([p.xyz for p in colmap_points.values()], dtype=np.float32))
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor

        # Load point colours
        points3D_rgb = torch.from_numpy(np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8))
        points3D_num_points = torch.tensor([len(p.image_ids) for p in colmap_points.values()], dtype=torch.int64)
        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
            "points3D_error": torch.from_numpy(np.array([p.error for p in colmap_points.values()], dtype=np.float32)),
            "points3D_num_points2D": points3D_num_points,
        }
        if self.config.max_2D_matches_per_3D_point != 0:
            if (colmap_path / "images.txt").exists():
                im_id_to_image = colmap_utils.read_images_text(colmap_path / "images.txt")
            elif (colmap_path / "images.bin").exists():
                im_id_to_image = colmap_utils.read_images_binary(colmap_path / "images.bin")
            else:
                raise ValueError(f"Could not find images.txt or images.bin in {colmap_path}")
            downscale_factor = self._downscale_factor
            max_num_points = int(torch.max(points3D_num_points).item())
            if self.config.max_2D_matches_per_3D_point > 0:
                max_num_points = min(max_num_points, self.config.max_2D_matches_per_3D_point)
            points3D_image_ids = []
            points3D_image_xy = []
            for p in colmap_points.values():
                nids = np.array(p.image_ids, dtype=np.int64)
                nxy_ids = np.array(p.point2D_idxs, dtype=np.int32)
                if self.config.max_2D_matches_per_3D_point != -1:
                    # Randomly sample 2D matches
                    idxs = np.argsort(p.error)[: self.config.max_2D_matches_per_3D_point]
                    nids = nids[idxs]
                    nxy_ids = nxy_ids[idxs]
                nxy = [im_id_to_image[im_id].xys[pt_idx] for im_id, pt_idx in zip(nids, nxy_ids)]
                nxy = torch.from_numpy(np.stack(nxy).astype(np.float32))
                nids = torch.from_numpy(nids)
                assert len(nids.shape) == 1
                assert len(nxy.shape) == 2
                points3D_image_ids.append(
                    torch.cat((nids, torch.full((max_num_points - len(nids),), -1, dtype=torch.int64)))
                )
                points3D_image_xy.append(
                    torch.cat((nxy, torch.full((max_num_points - len(nxy), nxy.shape[-1]), 0, dtype=torch.float32)))
                    / downscale_factor
                )
            out["points3D_image_ids"] = torch.stack(points3D_image_ids, dim=0)
            out["points3D_points2D_xy"] = torch.stack(points3D_image_xy, dim=0)
        return out

    def _downscale_images(self, paths, get_fname, downscale_factor: int, nearest_neighbor: bool = False):
        with status(msg="[bold yellow]Downscaling images...", spinner="growVertical"):
            assert downscale_factor > 1
            assert isinstance(downscale_factor, int)
            # Using %05d ffmpeg commands appears to be unreliable (skips images).
            for path in paths:
                nn_flag = "" if not nearest_neighbor else ":flags=neighbor"
                path_out = get_fname(path)
                path_out.parent.mkdir(parents=True, exist_ok=True)
                ffmpeg_cmd = [
                    f'ffmpeg -y -noautorotate -i "{path}" ',
                    f"-q:v 2 -vf scale=iw/{downscale_factor}:ih/{downscale_factor}{nn_flag} ",
                    f'"{path_out}"',
                ]
                ffmpeg_cmd = " ".join(ffmpeg_cmd)
                run_command(ffmpeg_cmd)

        CONSOLE.log("[bold green]:tada: Done downscaling images.")

    def _setup_downscale_factor(
        self, image_filenames: List[Path], mask_filenames: List[Path], depth_filenames: List[Path]
    ):
        """
        Setup the downscale factor for the dataset. This is used to downscale the images and cameras.
        """

        def get_fname(parent: Path, filepath: Path) -> Path:
            """Returns transformed file name when downscale factor is applied"""
            rel_part = filepath.relative_to(parent)
            base_part = parent.parent / (str(parent.name) + f"_{self._downscale_factor}")
            return base_part / rel_part

        filepath = next(iter(image_filenames))
        if self._downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                        break
                    df += 1

                self._downscale_factor = 2**df
                CONSOLE.log(f"Using image downscale factor of {self._downscale_factor}")
            else:
                self._downscale_factor = self.config.downscale_factor
            if self._downscale_factor > 1 and not all(
                get_fname(self.config.data / self.config.images_path, fp).parent.exists() for fp in image_filenames
            ):
                # Downscaled images not found
                # Ask if user wants to downscale the images automatically here
                CONSOLE.print(
                    f"[bold red]Downscaled images do not exist for factor of {self._downscale_factor}.[/bold red]"
                )
                if Confirm.ask("\nWould you like to downscale the images now?", default=False, console=CONSOLE):
                    # Install the method
                    self._downscale_images(
                        image_filenames,
                        partial(get_fname, self.config.data / self.config.images_path),
                        self._downscale_factor,
                        nearest_neighbor=False,
                    )
                    if len(mask_filenames) > 0:
                        assert self.config.masks_path is not None
                        self._downscale_images(
                            mask_filenames,
                            partial(get_fname, self.config.data / self.config.masks_path),
                            self._downscale_factor,
                            nearest_neighbor=True,
                        )
                    if len(depth_filenames) > 0:
                        assert self.config.depths_path is not None
                        self._downscale_images(
                            depth_filenames,
                            partial(get_fname, self.config.data / self.config.depths_path),
                            self._downscale_factor,
                            nearest_neighbor=True,
                        )
                else:
                    sys.exit(1)

        # Return transformed filenames
        if self._downscale_factor > 1:
            image_filenames = [get_fname(self.config.data / self.config.images_path, fp) for fp in image_filenames]
            if len(mask_filenames) > 0:
                assert self.config.masks_path is not None
                mask_filenames = [get_fname(self.config.data / self.config.masks_path, fp) for fp in mask_filenames]
            if len(depth_filenames) > 0:
                assert self.config.depths_path is not None
                depth_filenames = [get_fname(self.config.data / self.config.depths_path, fp) for fp in depth_filenames]
        assert isinstance(self._downscale_factor, int)
        return image_filenames, mask_filenames, depth_filenames, self._downscale_factor

```

## dataparsers

### arkitscenes_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for ARKitScenes dataset"""
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import cv2
import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox


# Taken from https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py
def traj_string_to_matrix(traj_string: str):
    """convert traj_string into translation and rotation matrices
    Args:
        traj_string: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)
    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    tokens = traj_string.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))  # type: ignore
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)


@dataclass
class ARKitScenesDataParserConfig(DataParserConfig):
    """ARKitScenes dataset config.
    ARKitScenes dataset (http://github.com/apple/ARKitScenes) is a large-scale 3D dataset of indoor scenes.
    This dataparser uses 3D detection subset of the ARKitScenes dataset.
    """

    _target: Type = field(default_factory=lambda: ARKitScenes)
    """target class to instantiate"""
    data: Path = Path("data/ARKitScenes/3dod/Validation/41069021")
    """Path to ARKitScenes folder with densely extracted scenes."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""


@dataclass
class ARKitScenes(DataParser):
    """ARKitScenes DatasetParser"""

    config: ARKitScenesDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        video_id = self.config.data.name

        image_dir = self.config.data / f"{video_id}_frames" / "lowres_wide"
        depth_dir = self.config.data / f"{video_id}_frames" / "lowres_depth"
        intrinsics_dir = self.config.data / f"{video_id}_frames" / "lowres_wide_intrinsics"
        pose_file = self.config.data / f"{video_id}_frames" / "lowres_wide.traj"

        frame_ids = [x.name for x in sorted(depth_dir.iterdir())]
        frame_ids = [x.split(".png")[0].split("_")[1] for x in frame_ids]
        frame_ids.sort()

        poses_from_traj = {}
        with open(pose_file, "r", encoding="utf-8") as f:
            traj = f.readlines()

        for line in traj:
            poses_from_traj[f"{round(float(line.split(' ')[0]), 3):.3f}"] = np.array(
                traj_string_to_matrix(line)[1].tolist()
            )

        image_filenames, depth_filenames, intrinsics, poses = [], [], [], []
        w, h, _, _, _, _ = np.loadtxt(list(sorted(intrinsics_dir.iterdir()))[0])  # Get image size from first intrinsic

        for frame_id in frame_ids:
            intrinsic = self._get_intrinsic(intrinsics_dir, frame_id, video_id)
            frame_pose = self._get_pose(frame_id, poses_from_traj)

            intrinsics.append(intrinsic)
            image_filenames.append(image_dir / f"{video_id}_{frame_id}.png")
            depth_filenames.append(depth_dir / f"{video_id}_{frame_id}.png")
            poses.append(frame_pose)

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_fraction)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = torch.from_numpy(np.stack(poses).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method="none",
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        intrinsics = intrinsics[indices.tolist()]
        poses = poses[indices.tolist()]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=int(h),
            width=int(w),
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs

    @staticmethod
    def _get_intrinsic(intrinsics_dir: Path, frame_id: str, video_id: str):
        intrinsic_fn = intrinsics_dir / f"{video_id}_{frame_id}.pincam"

        if not intrinsic_fn.exists():
            intrinsic_fn = intrinsics_dir / f"{video_id}_{float(frame_id) - 0.001:.3f}.pincam"

        if not intrinsic_fn.exists():
            intrinsic_fn = intrinsics_dir / f"{video_id}_{float(frame_id) + 0.001:.3f}.pincam"

        _, _, fx, fy, hw, hh = np.loadtxt(intrinsic_fn)
        intrinsic = np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])
        return intrinsic

    @staticmethod
    def _get_pose(frame_id: str, poses_from_traj: dict):
        frame_pose = None
        if str(frame_id) in poses_from_traj:
            frame_pose = np.array(poses_from_traj[str(frame_id)])
        else:
            for my_key in poses_from_traj:
                if abs(float(frame_id) - float(my_key)) < 0.005:
                    frame_pose = np.array(poses_from_traj[str(my_key)])

        assert frame_pose is not None
        frame_pose[0:3, 1:3] *= -1
        frame_pose = frame_pose[np.array([1, 0, 2, 3]), :]
        frame_pose[2, :] *= -1
        return frame_pose

```

## dataparsers

### dnerf_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class DNeRFDataParserConfig(DataParserConfig):
    """D-NeRF dataset parser config"""

    _target: Type = field(default_factory=lambda: DNeRF)
    """target class to instantiate"""
    data: Path = Path("data/dnerf/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""


@dataclass
class DNeRF(DataParser):
    """DNeRF Dataset"""

    config: DNeRFDataParserConfig
    includes_time: bool = True

    def __init__(self, config: DNeRFDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        poses = []
        times = []
        for frame in meta["frames"]:
            fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            times.append(frame["time"])
        poses = np.array(poses).astype(np.float32)
        times = torch.tensor(times, dtype=torch.float32)

        img_0 = imageio.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
            times=times,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs

```

## dataparsers

### neus_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600


@dataclass
class NeusDataParserConfig(DataParserConfig):
    """Neus dataset config"""

    _target: Type = field(default_factory=lambda: Neus)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    mask_color: Optional[Tuple[float, float, float]] = None
    """Replace the unknown pixels with this color. Relevant if you have a mask but still sample everywhere."""
    load_3D_points: bool = False
    """Whether to load the 3D points from the colmap reconstruction."""


@dataclass
class Neus(DataParser):
    """Neus DatasetParser"""

    config: NeusDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2", "distortion_params"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fisheye_crop_radius = meta.get("fisheye_crop_radius", None)
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    torch.tensor(frame["distortion_params"], dtype=torch.float32)
                    if "distortion_params" in frame
                    else camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                depth_filenames.append(depth_fname)

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"])
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            elif self.config.eval_mode == "all":
                CONSOLE.log(
                    "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
                )
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = (
                torch.tensor(meta["distortion_params"], dtype=torch.float32)
                if "distortion_params" in meta
                else camera_utils.get_distortion_params(
                    k1=float(meta["k1"]) if "k1" in meta else 0.0,
                    k2=float(meta["k2"]) if "k2" in meta else 0.0,
                    k3=float(meta["k3"]) if "k3" in meta else 0.0,
                    k4=float(meta["k4"]) if "k4" in meta else 0.0,
                    p1=float(meta["p1"]) if "p1" in meta else 0.0,
                    p2=float(meta["p2"]) if "p2" in meta else 0.0,
                )
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        # Only add fisheye crop radius parameter if the images are actually fisheye, to allow the same config to be used
        # for both fisheye and non-fisheye datasets.
        metadata = {}
        if (camera_type in [CameraType.FISHEYE, CameraType.FISHEYE624]) and (fisheye_crop_radius is not None):
            metadata["fisheye_crop_radius"] = fisheye_crop_radius

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata=metadata,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        # The naming is somewhat confusing, but:
        # - transform_matrix contains the transformation to dataparser output coordinates from saved coordinates.
        # - dataparser_transform_matrix contains the transformation to dataparser output coordinates from original data coordinates.
        # - applied_transform contains the transformation to saved coordinates from original data coordinates.
        applied_transform = None
        colmap_path = self.config.data / "colmap/sparse/0"
        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
        elif colmap_path.exists():
            # For converting from colmap, this was the effective value of applied_transform that was being
            # used before we added the applied_transform field to the output dataformat.
            meta["applied_transform"] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0]]
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)

        if applied_transform is not None:
            dataparser_transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        else:
            dataparser_transform_matrix = transform_matrix

        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        # reinitialize metadata for dataparser_outputs
        metadata = {}

        # _generate_dataparser_outputs might be called more than once so we check if we already loaded the point cloud
        try:
            self.prompted_user
        except AttributeError:
            self.prompted_user = False

        # Load 3D points
        if self.config.load_3D_points:
            if "ply_file_path" in meta:
                ply_file_path = data_dir / meta["ply_file_path"]

            elif colmap_path.exists():
                from rich.prompt import Confirm

                # check if user wants to make a point cloud from colmap points
                if not self.prompted_user:
                    self.create_pc = Confirm.ask(
                        "load_3D_points is true, but the dataset was processed with an outdated ns-process-data that didn't convert colmap points to .ply! Update the colmap dataset automatically?"
                    )

                if self.create_pc:
                    import json

                    from nerfstudio.process_data.colmap_utils import create_ply_from_colmap

                    with open(self.config.data / "transforms.json") as f:
                        transforms = json.load(f)

                    # Update dataset if missing the applied_transform field.
                    if "applied_transform" not in transforms:
                        transforms["applied_transform"] = meta["applied_transform"]

                    ply_filename = "sparse_pc.ply"
                    create_ply_from_colmap(
                        filename=ply_filename,
                        recon_dir=colmap_path,
                        output_dir=self.config.data,
                        applied_transform=applied_transform,
                    )
                    ply_file_path = data_dir / ply_filename
                    transforms["ply_file_path"] = ply_filename

                    # This was the applied_transform value

                    with open(self.config.data / "transforms.json", "w", encoding="utf-8") as f:
                        json.dump(transforms, f, indent=4)
                else:
                    ply_file_path = None
            else:
                if not self.prompted_user:
                    CONSOLE.print(
                        "[bold yellow]Warning: load_3D_points set to true but no point cloud found. splatfacto will use random point cloud initialization."
                    )
                ply_file_path = None

            if ply_file_path:
                sparse_points = self._load_3D_points(ply_file_path, transform_matrix, scale_factor)
                if sparse_points is not None:
                    metadata.update(sparse_points)
            self.prompted_user = True

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=dataparser_transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "normal_filenames": None,
                "camera_to_worlds": None,
                "transform": None,
                "include_mono_prior": False,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "mask_color": self.config.mask_color,
                **metadata,
            },
        )
        return dataparser_outputs

    def _load_3D_points(self, ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float):
        """Loads point clouds positions and colors from .ply

        Args:
            ply_file_path: Path to .ply file
            transform_matrix: Matrix to transform world coordinates
            scale_factor: How much to scale the camera origins by.

        Returns:
            A dictionary of points: points3D_xyz and colors: points3D_rgb
        """
        import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        pcd = o3d.io.read_point_cloud(str(ply_file_path))

        # if no points found don't read in an initial point cloud
        if len(pcd.points) == 0:
            return None

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }
        return out

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath

```

## dataparsers

### dycheck_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for DyCheck (https://arxiv.org/abs/2210.13445) dataset of `iphone` subset"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Type

import cv2
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


def downscale(img, scale: int) -> np.ndarray:
    """Function from DyCheck's repo. Downscale an image.

    Args:
        img: Input image
        scale: Factor of the scale

    Returns:
        New image
    """
    if scale == 1:
        return img
    height, width = img.shape[:2]
    if height % scale > 0 or width % scale > 0:
        raise ValueError(f"Image shape ({height},{width}) must be divisible by the" f" scale ({scale}).")
    out_height, out_width = height // scale, width // scale
    resized = cv2.resize(img, (out_width, out_height), cv2.INTER_AREA)  # type: ignore
    return resized


def upscale(img, scale: int) -> np.ndarray:
    """Function from DyCheck's repo. Upscale an image.

    Args:
        img: Input image
        scale: Factor of the scale

    Returns:
        New image
    """
    if scale == 1:
        return img
    height, width = img.shape[:2]
    out_height, out_width = height * scale, width * scale
    resized = cv2.resize(img, (out_width, out_height), cv2.INTER_AREA)  # type: ignore
    return resized


def rescale(img, scale_factor: float, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Function from DyCheck's repo. Rescale an image.

    Args:
        img: Input image
        scale: Factor of the scale
        interpolation: Interpolation method in opencv

    Returns:
        New image
    """
    scale_factor = float(scale_factor)
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be a non-negative number.")
    if scale_factor == 1.0:
        return img

    height, width = img.shape[:2]
    if scale_factor.is_integer():
        return upscale(img, int(scale_factor))

    inv_scale = 1.0 / scale_factor
    if inv_scale.is_integer() and (scale_factor * height).is_integer() and (scale_factor * width).is_integer():
        return downscale(img, int(inv_scale))

    print(f"Resizing image by non-integer factor {scale_factor}, this may lead to artifacts.")
    height, width = img.shape[:2]
    out_height = math.ceil(height * scale_factor)
    out_height -= out_height % 2
    out_width = math.ceil(width * scale_factor)
    out_width -= out_width % 2

    return cv2.resize(img, (out_width, out_height), interpolation)  # type: ignore


def _load_scene_info(data_dir: Path) -> Tuple[np.ndarray, float, float, float]:
    """Function from DyCheck's repo. Load scene info from json.

    Args:
        data_dir: data path

    Returns:
        A tuple of scene info: center, scale, near, far
    """
    scene_dict = load_from_json(data_dir / "scene.json")
    center = np.array(scene_dict["center"], dtype=np.float32)
    scale = scene_dict["scale"]
    near = scene_dict["near"]
    far = scene_dict["far"]
    return center, scale, near, far


def _load_metadata_info(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Function from DyCheck's repo. Load scene metadata from json.

    Args:
        data_dir: data path

    Returns:
        A tuple of scene info: frame_names_map, time_ids, camera_ids
    """
    dataset_dict = load_from_json(data_dir / "dataset.json")
    _frame_names = np.array(dataset_dict["ids"])

    metadata_dict = load_from_json(data_dir / "metadata.json")
    time_ids = np.array([metadata_dict[k]["warp_id"] for k in _frame_names], dtype=np.uint32)
    camera_ids = np.array([metadata_dict[k]["camera_id"] for k in _frame_names], dtype=np.uint32)

    frame_names_map = np.zeros((time_ids.max() + 1, camera_ids.max() + 1), _frame_names.dtype)
    for i, (t, c) in enumerate(zip(time_ids, camera_ids)):
        frame_names_map[t, c] = _frame_names[i]

    return frame_names_map, time_ids, camera_ids


def _rescale_depth(depth_raw: np.ndarray, cam: Dict) -> np.ndarray:
    """Depth rescale function from DyCheck.

    Args:
        depth: A numpy ndarray of the raw depth
        cam: Dict of the camera

    Returns:
        A numpy ndarray of the processed depth
    """
    xx, yy = np.meshgrid(np.arange(cam["width"], dtype=np.float32), np.arange(cam["height"], dtype=np.float32))
    pixels = np.stack([xx, yy], axis=-1)
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    y = (pixels[..., 1] - cam["cy"]) / cam["fy"]
    x = (pixels[..., 0] - cam["cx"]) / cam["fx"]
    # x = (pixels[..., 0] - self.principal_point_x - y * self.skew) / self.scale_factor_x
    # assume skew = 0
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    local_viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)
    viewdirs = (cam["camera_to_worlds"][:3, :3] @ local_viewdirs[..., None])[..., 0]
    viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)
    viewdirs = viewdirs.reshape((*batch_shape, 3))
    cosa = viewdirs @ (cam["camera_to_worlds"][:, 2])
    if depth_raw.ndim == cosa.ndim:
        depth = depth_raw[..., None] / cosa[..., None]
    else:
        depth = depth_raw / cosa[..., None]
    return depth


@dataclass
class DycheckDataParserConfig(DataParserConfig):
    """Dycheck (https://arxiv.org/abs/2210.13445) dataset parser config"""

    _target: Type = field(default_factory=lambda: Dycheck)
    """target class to instantiate"""
    data: Path = Path("data/iphone/mochi-high-five")
    """Directory specifying location of data."""
    scale_factor: float = 5.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    downscale_factor: int = 1
    """How much to downscale images."""
    scene_box_bound: float = 1.5
    """Boundary of scene box."""


@dataclass
class Dycheck(DataParser):
    """Dycheck (https://arxiv.org/abs/2210.13445) Dataset `iphone` subset"""

    config: DycheckDataParserConfig
    includes_time: bool = True

    def __init__(self, config: DycheckDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        # load extra info from "extra.json"
        extra_path = self.data / "extra.json"
        extra_dict = load_from_json(extra_path)
        self._factor = extra_dict["factor"]
        self._fps = extra_dict["fps"]
        self._bbox = np.array(extra_dict["bbox"], dtype=np.float32)
        self._lookat = np.array(extra_dict["lookat"], dtype=np.float32)
        self._up = np.array(extra_dict["up"], dtype=np.float32)
        self._center, self._scale, self._near, self._far = _load_scene_info(self.data)
        self._frame_names_map, self._time_ids, self._camera_ids = _load_metadata_info(self.data)

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None
        splits_dir = self.data / "splits"

        # scale the scene to fill the aabb bbox
        sf = self.config.scene_box_bound / 4 / (self._scale * self._far)
        # CONSOLE.print(f"scale factor changed from {self.config.scale_factor} to {sf}")
        self.config.scale_factor = sf

        if not (splits_dir / f"{split}.json").exists():
            CONSOLE.print(f"split {split} not found, using split train")
            split = "train"
        split_dict = load_from_json(splits_dir / f"{split}.json")
        frame_names = np.array(split_dict["frame_names"])
        time_ids = np.array(split_dict["time_ids"])
        if split != "train":
            CONSOLE.print(f"split {split} is empty, using the 1st training image")
            split_dict = load_from_json(splits_dir / "train.json")
            frame_names = np.array(split_dict["frame_names"])[[0]]
            time_ids = np.array(split_dict["time_ids"])[[0]]

        image_filenames, depth_filenames, cams = self.process_frames(frame_names.tolist(), time_ids)

        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-self.config.scene_box_bound] * 3, [self.config.scene_box_bound] * 3], dtype=torch.float32
            )
        )
        cam_dict = {}
        for k in cams[0].keys():
            cam_dict[k] = torch.stack([torch.as_tensor(c[k]) for c in cams], dim=0)
        cameras = Cameras(camera_type=CameraType.PERSPECTIVE, **cam_dict)

        scale = self._scale * self.config.scale_factor
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            metadata={
                "depth_filenames": depth_filenames,
                "depth_unit_scale_factor": scale,
                "scale": scale,
                "near": self._near * scale,
                "far": self._far * scale,
            },
        )

        return dataparser_outputs

    def process_frames(self, frame_names: List[str], time_ids: np.ndarray) -> Tuple[List, List, List]:
        """Read cameras and filenames from the name list.

        Args:
            frame_names: list of file names.
            time_ids: time id of each frame.

        Returns:
            A list of camera, each entry is a dict of the camera.
        """
        image_filenames, depth_filenames = [], []
        cams = []
        for idx, frame in enumerate(frame_names):
            image_filenames.append(self.data / f"rgb/{self.config.downscale_factor}x/{frame}.png")
            depth_filenames.append(self.data / f"processed_depth/{self.config.downscale_factor}x/{frame}.npy")
            cam_json = load_from_json(self.data / f"camera/{frame}.json")
            c2w = torch.as_tensor(cam_json["orientation"]).T
            position = torch.as_tensor(cam_json["position"])
            position -= self._center  # some scenes look weird (wheel)
            position *= self._scale * self.config.scale_factor
            pose = torch.zeros([3, 4])
            pose[:3, :3] = c2w
            pose[:3, 3] = position
            # from opencv coord to opengl coord (used by nerfstudio)
            pose[0:3, 1:3] *= -1  # switch cam coord x,y
            pose = pose[[1, 0, 2], :]  # switch world x,y
            pose[2, :] *= -1  # invert world z
            # for aabb bbox usage
            pose = pose[[1, 2, 0], :]  # switch world xyz to zxy
            cams.append(
                {
                    "camera_to_worlds": pose,
                    "fx": cam_json["focal_length"] / self.config.downscale_factor,
                    "fy": cam_json["focal_length"] * cam_json["pixel_aspect_ratio"] / self.config.downscale_factor,
                    "cx": cam_json["principal_point"][0] / self.config.downscale_factor,
                    "cy": cam_json["principal_point"][1] / self.config.downscale_factor,
                    "height": cam_json["image_size"][1] // self.config.downscale_factor,
                    "width": cam_json["image_size"][0] // self.config.downscale_factor,
                    "times": torch.as_tensor(time_ids[idx] / self._time_ids.max()).float(),
                }
            )

        d = self.config.downscale_factor
        if not image_filenames[0].exists():
            CONSOLE.print(f"downscale factor {d}x not exist, converting")
            ori_h, ori_w = cv2.imread(str(self.data / f"rgb/1x/{frame_names[0]}.png")).shape[:2]
            (self.data / f"rgb/{d}x").mkdir(exist_ok=True)
            h, w = ori_h // d, ori_w // d
            for frame in frame_names:
                cv2.imwrite(
                    str(self.data / f"rgb/{d}x/{frame}.png"),
                    cv2.resize(cv2.imread(str(self.data / f"rgb/1x/{frame}.png")), (w, h)),
                )
            CONSOLE.print("finished")

        if not depth_filenames[0].exists():
            CONSOLE.print(f"processed depth downscale factor {d}x not exist, converting")
            (self.data / f"processed_depth/{d}x").mkdir(exist_ok=True, parents=True)
            for idx, frame in enumerate(frame_names):
                depth = np.load(self.data / f"depth/1x/{frame}.npy")
                mask = rescale((depth != 0).astype(np.uint8) * 255, 1 / d, cv2.INTER_AREA)
                depth = rescale(depth, 1 / d, cv2.INTER_AREA)
                depth[mask != 255] = 0
                depth = _rescale_depth(depth, cams[idx])
                np.save(str(self.data / f"processed_depth/{d}x/{frame}.npy"), depth)
            CONSOLE.print("finished")

        return image_filenames, depth_filenames, cams

```

## dataparsers

### nerfstudio_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600


@dataclass
class NerfstudioDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nerfstudio)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    mask_color: Optional[Tuple[float, float, float]] = None
    """Replace the unknown pixels with this color. Relevant if you have a mask but still sample everywhere."""
    load_3D_points: bool = False
    """Whether to load the 3D points from the colmap reconstruction."""


@dataclass
class Nerfstudio(DataParser):
    """Nerfstudio DatasetParser"""

    config: NerfstudioDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2", "distortion_params"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fisheye_crop_radius = meta.get("fisheye_crop_radius", None)
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    torch.tensor(frame["distortion_params"], dtype=torch.float32)
                    if "distortion_params" in frame
                    else camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                depth_filenames.append(depth_fname)

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"])
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            elif self.config.eval_mode == "all":
                CONSOLE.log(
                    "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
                )
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = (
                torch.tensor(meta["distortion_params"], dtype=torch.float32)
                if "distortion_params" in meta
                else camera_utils.get_distortion_params(
                    k1=float(meta["k1"]) if "k1" in meta else 0.0,
                    k2=float(meta["k2"]) if "k2" in meta else 0.0,
                    k3=float(meta["k3"]) if "k3" in meta else 0.0,
                    k4=float(meta["k4"]) if "k4" in meta else 0.0,
                    p1=float(meta["p1"]) if "p1" in meta else 0.0,
                    p2=float(meta["p2"]) if "p2" in meta else 0.0,
                )
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        # Only add fisheye crop radius parameter if the images are actually fisheye, to allow the same config to be used
        # for both fisheye and non-fisheye datasets.
        metadata = {}
        if (camera_type in [CameraType.FISHEYE, CameraType.FISHEYE624]) and (fisheye_crop_radius is not None):
            metadata["fisheye_crop_radius"] = fisheye_crop_radius

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata=metadata,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        # The naming is somewhat confusing, but:
        # - transform_matrix contains the transformation to dataparser output coordinates from saved coordinates.
        # - dataparser_transform_matrix contains the transformation to dataparser output coordinates from original data coordinates.
        # - applied_transform contains the transformation to saved coordinates from original data coordinates.
        applied_transform = None
        colmap_path = self.config.data / "colmap/sparse/0"
        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
        elif colmap_path.exists():
            # For converting from colmap, this was the effective value of applied_transform that was being
            # used before we added the applied_transform field to the output dataformat.
            meta["applied_transform"] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0]]
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)

        if applied_transform is not None:
            dataparser_transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        else:
            dataparser_transform_matrix = transform_matrix

        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        # reinitialize metadata for dataparser_outputs
        metadata = {}

        # _generate_dataparser_outputs might be called more than once so we check if we already loaded the point cloud
        try:
            self.prompted_user
        except AttributeError:
            self.prompted_user = False

        # Load 3D points
        if self.config.load_3D_points:
            if "ply_file_path" in meta:
                ply_file_path = data_dir / meta["ply_file_path"]

            elif colmap_path.exists():
                from rich.prompt import Confirm

                # check if user wants to make a point cloud from colmap points
                if not self.prompted_user:
                    self.create_pc = Confirm.ask(
                        "load_3D_points is true, but the dataset was processed with an outdated ns-process-data that didn't convert colmap points to .ply! Update the colmap dataset automatically?"
                    )

                if self.create_pc:
                    import json

                    from nerfstudio.process_data.colmap_utils import create_ply_from_colmap

                    with open(self.config.data / "transforms.json") as f:
                        transforms = json.load(f)

                    # Update dataset if missing the applied_transform field.
                    if "applied_transform" not in transforms:
                        transforms["applied_transform"] = meta["applied_transform"]

                    ply_filename = "sparse_pc.ply"
                    create_ply_from_colmap(
                        filename=ply_filename,
                        recon_dir=colmap_path,
                        output_dir=self.config.data,
                        applied_transform=applied_transform,
                    )
                    ply_file_path = data_dir / ply_filename
                    transforms["ply_file_path"] = ply_filename

                    # This was the applied_transform value

                    with open(self.config.data / "transforms.json", "w", encoding="utf-8") as f:
                        json.dump(transforms, f, indent=4)
                else:
                    ply_file_path = None
            else:
                if not self.prompted_user:
                    CONSOLE.print(
                        "[bold yellow]Warning: load_3D_points set to true but no point cloud found. splatfacto will use random point cloud initialization."
                    )
                ply_file_path = None

            if ply_file_path:
                sparse_points = self._load_3D_points(ply_file_path, transform_matrix, scale_factor)
                if sparse_points is not None:
                    metadata.update(sparse_points)
            self.prompted_user = True

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=dataparser_transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "mask_color": self.config.mask_color,
                **metadata,
            },
        )
        return dataparser_outputs

    def _load_3D_points(self, ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float):
        """Loads point clouds positions and colors from .ply

        Args:
            ply_file_path: Path to .ply file
            transform_matrix: Matrix to transform world coordinates
            scale_factor: How much to scale the camera origins by.

        Returns:
            A dictionary of points: points3D_xyz and colors: points3D_rgb
        """
        import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        pcd = o3d.io.read_point_cloud(str(ply_file_path))

        # if no points found don't read in an initial point cloud
        if len(pcd.points) == 0:
            return None

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }
        return out

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath

```

## dataparsers

### nuscenes_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for NuScenes dataset"""
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Type

import numpy as np
import pyquaternion
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox


def rotation_translation_to_pose(r_quat, t_vec):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""

    pose = np.eye(4)

    # NB: Nuscenes recommends pyquaternion, which uses scalar-first format (w x y z)
    # https://github.com/nutonomy/nuscenes-devkit/issues/545#issuecomment-766509242
    # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L299
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    pose[:3, :3] = pyquaternion.Quaternion(r_quat).rotation_matrix

    pose[:3, 3] = t_vec
    return pose


@dataclass
class NuScenesDataParserConfig(DataParserConfig):
    """NuScenes dataset config.
    NuScenes (https://www.nuscenes.org/nuscenes) is an autonomous driving dataset containing 1000 20s clips.
    Each clip was recorded with a suite of sensors including 6 surround cameras.
    It also includes 3D cuboid annotations around objects.
    We optionally use these cuboids to mask dynamic objects by specifying the mask_dir flag.
    To create these masks use nerfstudio/scripts/datasets/process_nuscenes_masks.py.
    """

    _target: Type = field(default_factory=lambda: NuScenes)
    """target class to instantiate"""
    data: Path = Path("scene-0103")  # TODO: rename to scene but keep checkpoint saving name?
    """Name of the scene."""
    data_dir: Path = Path("/mnt/local/NuScenes")
    """Path to NuScenes dataset."""
    version: Literal["v1.0-mini", "v1.0-trainval"] = "v1.0-mini"
    """Dataset version."""
    cameras: Tuple[Literal["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"], ...] = ("FRONT",)
    """Which cameras to use."""
    mask_dir: Optional[Path] = None
    """Path to masks of dynamic objects."""

    train_split_fraction: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""

    verbose: bool = False
    """Load dataset with verbose messaging"""


@dataclass
class NuScenes(DataParser):
    """NuScenes DatasetParser"""

    config: NuScenesDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        # nuscenes is slow to import, so we only do it if we need it.
        from nuscenes.nuscenes import NuScenes as NuScenesDatabase

        nusc = NuScenesDatabase(
            version=self.config.version,
            dataroot=str(self.config.data_dir.absolute()),
            verbose=self.config.verbose,
        )
        cameras = ["CAM_" + camera for camera in self.config.cameras]

        assert (
            len(cameras) == 1
        ), "waiting on multiple camera support"  # TODO: remove once multiple cameras are supported

        # get samples for scene
        samples = [
            samp for samp in nusc.sample if nusc.get("scene", samp["scene_token"])["name"] == str(self.config.data)
        ]

        # sort by timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

        transform1 = np.array(
            [
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        transform2 = np.array(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )

        # get image filenames and camera data
        image_filenames = []
        mask_filenames = []
        mask_dir = self.config.mask_dir if self.config.mask_dir is not None else Path("")
        intrinsics = []
        poses = []
        for sample in samples:
            for camera in cameras:
                camera_data = nusc.get("sample_data", sample["data"][camera])
                calibrated_sensor_data = nusc.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
                ego_pose_data = nusc.get("ego_pose", camera_data["ego_pose_token"])

                ego_pose = rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
                cam_pose = rotation_translation_to_pose(
                    calibrated_sensor_data["rotation"], calibrated_sensor_data["translation"]
                )
                pose = ego_pose @ cam_pose

                # rotate to opencv frame
                pose = transform1 @ pose

                # convert from opencv camera to nerfstudio camera
                pose[0:3, 1:3] *= -1
                pose = pose[np.array([1, 0, 2, 3]), :]
                pose[2, :] *= -1

                # rotate to z-up in viewer
                pose = transform2 @ pose

                image_filenames.append(self.config.data_dir / camera_data["filename"])
                mask_filenames.append(
                    mask_dir / "masks" / camera / os.path.split(camera_data["filename"])[1].replace("jpg", "png")
                )
                intrinsics.append(calibrated_sensor_data["camera_intrinsic"])
                poses.append(pose)
        poses = torch.from_numpy(np.stack(poses).astype(np.float32))
        intrinsics = torch.from_numpy(np.array(intrinsics).astype(np.float32))

        # center poses
        poses[:, :3, 3] -= poses[:, :3, 3].mean(dim=0)

        # scale poses
        poses[:, :3, 3] /= poses[:, :3, 3].abs().max()

        # filter image_filenames and poses based on train/eval split percentage
        num_snapshots = len(samples)
        num_train_snapshots = math.ceil(num_snapshots * self.config.train_split_fraction)
        num_eval_snapshots = num_snapshots - num_train_snapshots
        i_all = np.arange(num_snapshots)
        i_train = np.linspace(
            0, num_snapshots - 1, num_train_snapshots, dtype=int
        )  # equally spaced training snapshots starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_snapshots
        i_train = (i_train[None, :] * len(cameras) + np.arange(len(cameras))[:, None]).ravel()
        i_eval = (i_eval[None, :] * len(cameras) + np.arange(len(cameras))[:, None]).ravel()
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices]
        intrinsics = intrinsics[indices]
        poses = poses[indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = 1.0
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            fx=intrinsics[:, 0, 0].detach().clone(),
            fy=intrinsics[:, 1, 1].detach().clone(),
            cx=intrinsics[:, 0, 2].detach().clone(),
            cy=intrinsics[:, 1, 2].detach().clone(),
            height=900,
            width=1600,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if self.config.mask_dir is not None else None,
        )
        return dataparser_outputs

```

## dataparsers

### base_dataparser.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A set of standard datasets."""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

import torch
from jaxtyping import Float
from torch import Tensor

import nerfstudio.configs.base_config as cfg
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox


@dataclass
class Semantics:
    """Dataclass for semantic labels."""

    filenames: List[Path]
    """filenames to load semantic data"""
    classes: List[str]
    """class labels for data"""
    colors: Tensor
    """color mapping for classes"""
    mask_classes: List[str] = field(default_factory=lambda: [])
    """classes to mask out from training for all modalities"""


@dataclass
class DataparserOutputs:
    """Dataparser outputs for the which will be used by the DataManager
    for creating RayBundle and RayGT objects."""

    image_filenames: List[Path]
    """Filenames for the images."""
    cameras: Cameras
    """Camera object storing collection of camera information in dataset."""
    alpha_color: Optional[Float[Tensor, "3"]] = None
    """Color of dataset background."""
    scene_box: SceneBox = field(default_factory=lambda: SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]])))
    """Scene box of dataset. Used to bound the scene or provide the scene scale depending on model."""
    mask_filenames: Optional[List[Path]] = None
    """Filenames for any masks that are required"""
    metadata: Dict[str, Any] = to_immutable_dict({})
    """Dictionary of any metadata that be required for the given experiment.
    Will be processed by the InputDataset to create any additional tensors that may be required.
    """
    dataparser_transform: Float[Tensor, "3 4"] = torch.eye(4)[:3, :]
    """Transform applied by the dataparser."""
    dataparser_scale: float = 1.0
    """Scale applied by the dataparser."""

    def as_dict(self) -> dict:
        """Returns the dataclass as a dictionary."""
        return vars(self)

    def save_dataparser_transform(self, path: Path):
        """Save dataparser transform to json file. Some dataparsers will apply a transform to the poses,
        this method allows the transform to be saved so that it can be used in other applications.

        Args:
            path: path to save transform to
        """
        data = {
            "transform": self.dataparser_transform.tolist(),
            "scale": float(self.dataparser_scale),
        }
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, "w", encoding="UTF-8") as file:
            json.dump(data, file, indent=4)

    def transform_poses_to_original_space(
        self,
        poses: Float[Tensor, "num_poses 3 4"],
        camera_convention: Literal["opengl", "opencv"] = "opencv",
    ) -> Float[Tensor, "num_poses 3 4"]:
        """
        Transforms the poses in the transformed space back to the original world coordinate system.
        Args:
            poses: Poses in the transformed space
            camera_convention: Camera system convention used for the transformed poses
        Returns:
            Original poses
        """
        return transform_poses_to_original_space(
            poses,
            self.dataparser_transform,
            self.dataparser_scale,
            camera_convention=camera_convention,
        )


@dataclass
class DataParserConfig(cfg.InstantiateConfig):
    """Basic dataset config"""

    _target: Type = field(default_factory=lambda: DataParser)
    """_target: target class to instantiate"""
    data: Path = Path()
    """Directory specifying location of data."""


@dataclass
class DataParser:
    """A dataset.

    Args:
        config: datasetparser config containing all information needed to instantiate dataset

    Attributes:
        config: datasetparser config containing all information needed to instantiate dataset
        includes_time: Does the dataset include time information in the camera poses.
    """

    config: DataParserConfig
    includes_time: bool = False

    def __init__(self, config: DataParserConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def _generate_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> DataparserOutputs:
        """Abstract method that returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).
            kwargs: kwargs for generating dataparser outputs.

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """

    def get_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> DataparserOutputs:
        """Returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).
            kwargs: kwargs for generating dataparser outputs.

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """
        dataparser_outputs = self._generate_dataparser_outputs(split, **kwargs)
        return dataparser_outputs


def transform_poses_to_original_space(
    poses: Float[Tensor, "num_poses 3 4"],
    applied_transform: Float[Tensor, "3 4"],
    applied_scale: float,
    camera_convention: Literal["opengl", "opencv"] = "opencv",
) -> Float[Tensor, "num_poses 3 4"]:
    """
    Transforms the poses in the transformed space back to the original world coordinate system.
    Args:
        poses: Poses in the transformed space
        applied_transform: Transform matrix applied in the data processing step
        applied_scale: Scale used in the data processing step
        camera_convention: Camera system convention used for the transformed poses
    Returns:
        Original poses
    """
    output_poses = torch.cat(
        (
            poses,
            torch.tensor([[[0, 0, 0, 1]]], dtype=poses.dtype, device=poses.device).repeat_interleave(len(poses), 0),
        ),
        1,
    )
    output_poses[..., :3, 3] /= applied_scale
    inv_transform = torch.linalg.inv(
        torch.cat(
            (
                applied_transform,
                torch.tensor([[0, 0, 0, 1]], dtype=applied_transform.dtype, device=applied_transform.device),
            ),
            0,
        )
    )
    output_poses = torch.einsum("ij,bjk->bik", inv_transform, output_poses)
    if camera_convention == "opencv":
        output_poses[..., 0:3, 1:3] *= -1
    elif camera_convention == "opengl":
        pass
    else:
        raise ValueError(f"Camera convention {camera_convention} is not supported.")
    return output_poses[:, :3]

```

## utils

### data_utils.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions to allow easy re-use of common operations across dataloaders"""
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image


def get_image_mask_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    if len(mask_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype=torch.int64).view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.NEAREST)
    semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float64) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :, np.newaxis])

```

## utils

### __init__.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

## utils

### pixel_sampling_utils.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Pixel sampling utils such as eroding of valid masks that we sample from. """

import torch
from jaxtyping import Float
from torch import Tensor


def dilate(tensor: Float[Tensor, "bs 1 H W"], kernel_size=3) -> Float[Tensor, "bs 1 H W"]:
    """Dilate a tensor with 0s and 1s. 0s will be be expanded based on the kernel size.

    Args:
        kernel_size: Size of the pooling region. Dilates/contracts 1 pixel if kernel_size is 3.
    """

    unique_vals = torch.unique(tensor)
    if any(val not in (0, 1) for val in unique_vals) or tensor.dtype != torch.float32:
        raise ValueError("Input tensor should contain only values 0 and 1, and should have dtype torch.float32.")

    return torch.nn.functional.max_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)


def erode(tensor: Float[Tensor, "bs 1 H W"], kernel_size=3) -> Float[Tensor, "bs 1 H W"]:
    """Erode a tensor with 0s and 1s. 1s will be expanded based on the kernel size.

    Args:
        kernel_size: Size of the pooling region. Erodes/expands 1 pixel if kernel_size is 3.
    """

    unique_vals = torch.unique(tensor)
    if any(val not in (0, 1) for val in unique_vals) or tensor.dtype != torch.float32:
        raise ValueError("Input tensor should contain only values 0 and 1, and should have dtype torch.float32.")

    x = 1 - dilate(1 - tensor, kernel_size=kernel_size)
    # set edge pixels to 0
    p = (kernel_size - 1) // 2
    x[:, :, :p, :] *= 0
    x[:, :, :, :p] *= 0
    x[:, :, -p:, :] *= 0
    x[:, :, :, -p:] *= 0
    return x


def erode_mask(tensor: Float[Tensor, "bs 1 H W"], pixel_radius: int = 1):
    """Erode a mask. Expands 1 values to nearby pixels with a max pooling operation.
    A pixel radius of 1 will also extend the 1s along the diagonal.

    Args:
        pixel_radius: The number of pixels away from valid pixels (1s) that we may sample.
    """
    kernel_size = 1 + 2 * pixel_radius
    return erode(tensor, kernel_size=kernel_size)

```

## utils

### dataparsers_utils.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser utils for nerfstudio datasets. """

import math
import os
from typing import List, Tuple

import numpy as np


def get_train_eval_split_fraction(image_filenames: List, train_split_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split fraction based on the number of images and the train split fraction.

    Args:
        image_filenames: list of image filenames
        train_split_fraction: fraction of images to use for training
    """

    # filter image_filenames and poses based on train/eval split percentage
    num_images = len(image_filenames)
    num_train_images = math.ceil(num_images * train_split_fraction)
    num_eval_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images

    return i_train, i_eval


def get_train_eval_split_filename(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the filename of the images.

    Args:
        image_filenames: list of image filenames
    """

    num_images = len(image_filenames)
    basenames = [os.path.basename(image_filename) for image_filename in image_filenames]
    i_all = np.arange(num_images)
    i_train = []
    i_eval = []
    for idx, basename in zip(i_all, basenames):
        # check the frame index
        if "train" in basename:
            i_train.append(idx)
        elif "eval" in basename:
            i_eval.append(idx)
        else:
            raise ValueError("frame should contain train/eval in its name to use this eval-frame-index eval mode")

    return np.array(i_train), np.array(i_eval)


def get_train_eval_split_interval(image_filenames: List, eval_interval: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the interval of the images.

    Args:
        image_filenames: list of image filenames
        eval_interval: interval of images to use for eval
    """

    num_images = len(image_filenames)
    all_indices = np.arange(num_images)
    train_indices = all_indices[all_indices % eval_interval != 0]
    eval_indices = all_indices[all_indices % eval_interval == 0]
    i_train = train_indices
    i_eval = eval_indices

    return i_train, i_eval


def get_train_eval_split_all(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split where all indices are used for both train and eval.

    Args:
        image_filenames: list of image filenames
    """
    num_images = len(image_filenames)
    i_all = np.arange(num_images)
    i_train = i_all
    i_eval = i_all
    return i_train, i_eval

```

## utils

### nerfstudio_collate.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom collate function that includes cases for nerfstudio types.
"""

import collections
import collections.abc
import re
from typing import Any, Callable, Dict, Union

import torch
import torch.utils.data

from nerfstudio.cameras.cameras import Cameras

NERFSTUDIO_COLLATE_ERR_MSG_FORMAT = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, " "dicts, lists or anything in {}; found {}"
)
np_str_obj_array_pattern = re.compile(r"[SaUO]")


def nerfstudio_collate(batch: Any, extra_mappings: Union[Dict[type, Callable], None] = None) -> Any:
    r"""
    This is the default pytorch collate function, but with support for nerfstudio types. All documentation
    below is copied straight over from pytorch's default_collate function, python version 3.8.13,
    pytorch version '1.12.1+cu113'. Custom nerfstudio types are accounted for at the end, and extra
    mappings can be passed in to handle custom types. These mappings are from types: callable (types
    being like int or float or the return value of type(3.), etc). The only code before we parse for custom types that
    was changed from default pytorch was the addition of the extra_mappings argument, a find and replace operation
    from default_collate to nerfstudio_collate, and the addition of the nerfstudio_collate_err_msg_format variable.


    Function that takes in a batch of data and puts the elements within the batch
    into a tensor with an additional outer dimension - batch size. The exact output type can be
    a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, nerfstudio_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[nerfstudio_collate([V1_1, V1_2, ...]),
          nerfstudio_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[nerfstudio_collate([V1_1, V1_2, ...]),
          nerfstudio_collate([V2_1, V2_2, ...]), ...]`

    Args:
        batch: a single batch to be collated

    Examples:
        >>> # Example with a batch of `int`s:
        >>> nerfstudio_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> nerfstudio_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> nerfstudio_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> nerfstudio_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> nerfstudio_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Example with `List` inside the batch:
        >>> nerfstudio_collate([[0, 1], [2, 3]])
        [tensor([0, 2]), tensor([1, 3])]
    """
    if extra_mappings is None:
        extra_mappings = {}
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ in ("ndarray", "memmap"):
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(NERFSTUDIO_COLLATE_ERR_MSG_FORMAT.format(elem.dtype))

            return nerfstudio_collate([torch.as_tensor(b) for b in batch], extra_mappings=extra_mappings)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type(
                {key: nerfstudio_collate([d[key] for d in batch], extra_mappings=extra_mappings) for key in elem}
            )
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: nerfstudio_collate([d[key] for d in batch], extra_mappings=extra_mappings) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                return elem_type([nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed]

    # NerfStudio types supported below

    elif isinstance(elem, Cameras):
        # If a camera, just concatenate along the batch dimension. In the future, this may change to stacking
        assert all((isinstance(cam, Cameras) for cam in batch))
        assert all((cam.distortion_params is None for cam in batch)) or all(
            (cam.distortion_params is not None for cam in batch)
        ), "All cameras must have distortion parameters or none of them should have distortion parameters.\
            Generalized batching will be supported in the future."

        # If no batch dimension exists, then we need to stack everything and create a batch dimension on 0th dim
        if elem.shape == ():
            op = torch.stack
        # If batch dimension exists, then we need to concatenate along the 0th dimension
        else:
            op = torch.cat

        # Create metadata dictionary
        metadata_keys = batch[0].metadata.keys()
        assert all(
            (cam.metadata.keys() == metadata_keys for cam in batch)
        ), "All cameras must have the same metadata keys."
        metadata = {key: op([cam.metadata[key] for cam in batch], dim=0) for key in metadata_keys}

        return Cameras(
            op([cameras.camera_to_worlds for cameras in batch], dim=0),
            op([cameras.fx for cameras in batch], dim=0),
            op([cameras.fy for cameras in batch], dim=0),
            op([cameras.cx for cameras in batch], dim=0),
            op([cameras.cy for cameras in batch], dim=0),
            height=op([cameras.height for cameras in batch], dim=0),
            width=op([cameras.width for cameras in batch], dim=0),
            distortion_params=op(
                [
                    cameras.distortion_params
                    if cameras.distortion_params is not None
                    else torch.zeros_like(cameras.distortion_params)
                    for cameras in batch
                ],
                dim=0,
            ),
            camera_type=op([cameras.camera_type for cameras in batch], dim=0),
            times=torch.stack(
                [cameras.times if cameras.times is not None else -torch.ones_like(cameras.times) for cameras in batch],
                dim=0,
            ),
            metadata=metadata,
        )

    for type_key in extra_mappings:
        if isinstance(elem, type_key):
            return extra_mappings[type_key](batch)

    raise TypeError(NERFSTUDIO_COLLATE_ERR_MSG_FORMAT.format(elem_type))

```

## utils

### dataloaders.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for sampling images from a dataset of images.
"""

# for multithreading
import concurrent.futures
import multiprocessing
import random
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sized, Tuple, Union

import torch
from rich.progress import track
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.utils.rich_utils import CONSOLE


class CacheDataloader(DataLoader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 for all images.
        num_times_to_repeat_images: How often to collate new images. -1 to never pick new images.
        device: Device to perform computation.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(
        self,
        dataset: Dataset,
        num_images_to_sample_from: int = -1,
        num_times_to_repeat_images: int = -1,
        device: Union[torch.device, str] = "cpu",
        collate_fn: Callable[[Any], Any] = nerfstudio_collate,
        exclude_batch_keys_from_device: Optional[List[str]] = None,
        **kwargs,
    ):
        if exclude_batch_keys_from_device is None:
            exclude_batch_keys_from_device = ["image"]
        self.dataset = dataset
        assert isinstance(self.dataset, Sized)

        super().__init__(dataset=dataset, **kwargs)  # This will set self.dataset
        self.num_times_to_repeat_images = num_times_to_repeat_images
        self.cache_all_images = (num_images_to_sample_from == -1) or (num_images_to_sample_from >= len(self.dataset))
        self.num_images_to_sample_from = len(self.dataset) if self.cache_all_images else num_images_to_sample_from
        self.device = device
        self.collate_fn = collate_fn
        self.num_workers = kwargs.get("num_workers", 0)
        self.exclude_batch_keys_from_device = exclude_batch_keys_from_device

        self.num_repeated = self.num_times_to_repeat_images  # starting value
        self.first_time = True

        self.cached_collated_batch = None
        if self.cache_all_images:
            CONSOLE.print(f"Caching all {len(self.dataset)} images.")
            if len(self.dataset) > 500:
                CONSOLE.print(
                    "[bold yellow]Warning: If you run out of memory, try reducing the number of images to sample from."
                )
            self.cached_collated_batch = self._get_collated_batch()
        elif self.num_times_to_repeat_images == -1:
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, without resampling."
            )
        else:
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, "
                f"resampling every {self.num_times_to_repeat_images} iters."
            )

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""

        assert isinstance(self.dataset, Sized)
        indices = random.sample(range(len(self.dataset)), k=self.num_images_to_sample_from)
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())

        return batch_list

    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self._get_batch_list()
        collated_batch = self.collate_fn(batch_list)
        collated_batch = get_dict_to_torch(
            collated_batch, device=self.device, exclude=self.exclude_batch_keys_from_device
        )
        return collated_batch

    def __iter__(self):
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or (
                self.num_times_to_repeat_images != -1 and self.num_repeated >= self.num_times_to_repeat_images
            ):
                # trigger a reset
                self.num_repeated = 0
                collated_batch = self._get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch


class EvalDataloader(DataLoader):
    """Evaluation dataloader base class

    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        self.input_dataset = input_dataset
        self.cameras = input_dataset.cameras.to(device)
        self.device = device
        self.kwargs = kwargs
        super().__init__(dataset=input_dataset)

    @abstractmethod
    def __iter__(self):
        """Iterates over the dataset"""
        return self

    @abstractmethod
    def __next__(self) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data"""

    def get_camera(self, image_idx: int = 0) -> Tuple[Cameras, Dict]:
        """Get camera for the given image index

        Args:
            image_idx: Camera image index
        """
        camera = self.cameras[image_idx : image_idx + 1]
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        assert isinstance(batch, dict)
        return camera, batch

    def get_data_from_image_idx(self, image_idx: int) -> Tuple[RayBundle, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        ray_bundle = self.cameras.generate_rays(camera_indices=image_idx, keep_shape=True)
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        assert isinstance(batch, dict)
        return ray_bundle, batch


class FixedIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns a fixed set of indices.

    Args:
        input_dataset: InputDataset to load data from
        image_indices: List of image indices to load data from. If None, then use all images.
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        if image_indices is None:
            self.image_indices = list(range(len(input_dataset)))
        else:
            self.image_indices = image_indices
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            camera, batch = self.get_camera(image_idx)
            self.count += 1
            return camera, batch
        raise StopIteration


class RandIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns random images.
    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __iter__(self):
        return self

    def __next__(self):
        # choose a random image index
        image_idx = random.randint(0, len(self.cameras) - 1)
        camera, batch = self.get_camera(image_idx)
        return camera, batch

```

## utils

### colmap_parsing_utils.py

```python
"""
This file copied with small modifications from:
 * https://github.com/colmap/colmap/blob/1a4d0bad2e90aa65ce997c9d1779518eaed998d5/scripts/python/read_write_model.py

TODO(1480) Delete this file when moving to pycolmap.


"""

# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

import collections
import os
import struct

import numpy as np

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
            )
        assert len(cameras) == num_cameras
    return cameras


def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum((len(img.point3D_ids) for _, img in images.items())) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(len(images), mean_observations)
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs
                )
    return points3D


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs
            )
    return points3D


def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items())) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def detect_model_format(path, ext):
    if (
        os.path.isfile(os.path.join(path, "cameras" + ext))
        and os.path.isfile(os.path.join(path, "images" + ext))
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def write_model(cameras, images, points3D, path, ext=".bin"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        write_images_binary(images, os.path.join(path, "images" + ext))
        write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

```

## datasets

### depth_dataset.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Depth dataset.
"""

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image
from rich.progress import track

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.model_components import losses
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE


class DepthDataset(InputDataset):
    """Dataset that returns images and depths. If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        # if there are no depth images than we want to generate them all with zoe depth

        if len(dataparser_outputs.image_filenames) > 0 and (
            "depth_filenames" not in dataparser_outputs.metadata.keys()
            or dataparser_outputs.metadata["depth_filenames"] is None
        ):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            CONSOLE.print("[bold yellow] No depth data found! Generating pseudodepth...")
            losses.FORCE_PSEUDODEPTH_LOSS = True
            CONSOLE.print("[bold red] Using psueodepth: forcing depth loss to be ranking loss.")
            cache = dataparser_outputs.image_filenames[0].parent / "depths.npy"
            # Note: this should probably be saved to disk as images, and then loaded with the dataparser.
            #  That will allow multi-gpu training.
            if cache.exists():
                CONSOLE.print("[bold yellow] Loading pseudodata depth from cache!")
                # load all the depths
                self.depths = np.load(cache)
                self.depths = torch.from_numpy(self.depths).to(device)
            else:
                depth_tensors = []
                transforms = self._find_transform(dataparser_outputs.image_filenames[0])
                data = dataparser_outputs.image_filenames[0].parent
                if transforms is not None:
                    meta = json.load(open(transforms, "r"))
                    frames = meta["frames"]
                    filenames = [data / frames[j]["file_path"].split("/")[-1] for j in range(len(frames))]
                else:
                    meta = None
                    frames = None
                    filenames = dataparser_outputs.image_filenames

                repo = "isl-org/ZoeDepth"
                self.zoe = torch_compile(torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(device))

                for i in track(range(len(filenames)), description="Generating depth images"):
                    image_filename = filenames[i]
                    pil_image = Image.open(image_filename)
                    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
                    if len(image.shape) == 2:
                        image = image[:, :, None].repeat(3, axis=2)
                    image = torch.from_numpy(image.astype("float32") / 255.0)

                    with torch.no_grad():
                        image = torch.permute(image, (2, 0, 1)).unsqueeze(0).to(device)
                        if image.shape[1] == 4:
                            image = image[:, :3, :, :]
                        depth_tensor = self.zoe.infer(image).squeeze().unsqueeze(-1)

                    depth_tensors.append(depth_tensor)

                self.depths = torch.stack(depth_tensors)
                np.save(cache, self.depths.cpu().numpy())
            dataparser_outputs.metadata["depth_filenames"] = None
            dataparser_outputs.metadata["depth_unit_scale_factor"] = 1.0
            self.metadata["depth_filenames"] = None
            self.metadata["depth_unit_scale_factor"] = 1.0

        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]

    def get_metadata(self, data: Dict) -> Dict:
        if self.depth_filenames is None:
            return {"depth_image": self.depths[data["image_idx"]]}

        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )

        return {"depth_image": depth_image}

    def _find_transform(self, image_path: Path) -> Union[Path, None]:
        while image_path.parent != image_path:
            transform_path = image_path.parent / "transforms.json"
            if transform_path.exists():
                return transform_path
            image_path = image_path.parent
        return None

```

## datasets

### sdf_dataset.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SDFStudio dataset.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import Tensor

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class SDFDataset(InputDataset):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["depth", "normal"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        # can be none if monoprior not included
        self.depth_filenames = self.metadata["depth_filenames"]
        self.normal_filenames = self.metadata["normal_filenames"]
        self.camera_to_worlds = self.metadata["camera_to_worlds"]
        # can be none if auto orient not enabled in dataparser
        self.transform = self.metadata["transform"]
        self.include_mono_prior = self.metadata["include_mono_prior"]

    def get_metadata(self, data: Dict) -> Dict:
        # TODO supports foreground_masks
        metadata = {}
        if self.include_mono_prior:
            depth_filepath = self.depth_filenames[data["image_idx"]]
            normal_filepath = self.normal_filenames[data["image_idx"]]
            camtoworld = self.camera_to_worlds[data["image_idx"]]

            # Scale depth images to meter units and also by scaling applied to cameras
            depth_image, normal_image = self.get_depths_and_normals(
                depth_filepath=depth_filepath, normal_filename=normal_filepath, camtoworld=camtoworld
            )
            metadata["depth"] = depth_image
            metadata["normal"] = normal_image

        return metadata

    def get_depths_and_normals(self, depth_filepath: Path, normal_filename: Path, camtoworld: Tensor):
        """function to process additional depths and normal information
        Args:
            depth_filepath: path to depth file
            normal_filename: path to normal file
            camtoworld: camera to world transformation matrix
        """

        # load mono depth
        depth = np.load(depth_filepath)
        depth = torch.from_numpy(depth).float()

        # load mono normal
        normal = np.load(normal_filename)

        # transform normal to world coordinate system
        normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
        normal = torch.from_numpy(normal).float()

        rot = camtoworld[:3, :3]

        normal_map = normal.reshape(3, -1)
        normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

        normal_map = rot @ normal_map
        normal = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)

        if self.transform is not None:
            h, w, _ = normal.shape
            normal = self.transform[:3, :3] @ normal.reshape(-1, 3).permute(1, 0)
            normal = normal.permute(1, 0).reshape(h, w, 3)

        return depth, normal

```

## datasets

### __init__.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

## datasets

### semantic_dataset.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Semantic dataset.
"""

from typing import Dict

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path


class SemanticDataset(InputDataset):
    """Dataset that returns images and semantics and masks.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["mask", "semantics"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Semantics)
        self.semantics = self.metadata["semantics"]
        self.mask_indices = torch.tensor(
            [self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]
        ).view(1, 1, -1)

    def get_metadata(self, data: Dict) -> Dict:
        # handle mask
        filepath = self.semantics.filenames[data["image_idx"]]
        semantic_label, mask = get_semantics_and_mask_tensors_from_path(
            filepath=filepath, mask_indices=self.mask_indices, scale_factor=self.scale_factor
        )
        if "mask" in data.keys():
            mask = mask & data["mask"]
        return {"mask": mask, "semantics": semantic_label}

```

## datasets

### base_dataset.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path


class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask"]
    cameras: Cameras

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.mask_color = dataparser_outputs.metadata.get("mask_color", None)

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image_float32(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in float32 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_image_uint8(self, image_idx: int) -> UInt8[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in uint8 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx))
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * (image[:, :, -1:] / 255.0) + 255.0 * self._dataparser_outputs.alpha_color * (
                1.0 - image[:, :, -1:] / 255.0
            )
            image = torch.clamp(image, min=0, max=255).to(torch.uint8)
        return image

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
            )
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        del data
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self._dataparser_outputs.image_filenames

```

## datasets

### neus_dataset.py

```python
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SDFStudio dataset.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import Tensor

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class NeusDataset(InputDataset):
    """Dataset that returns images and depths. And Handels Numpy files as images

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["depth", "normal"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        # can be none if monoprior not included
        self.depth_filenames = self.metadata["depth_filenames"]
        self.normal_filenames = self.metadata["normal_filenames"]
        self.camera_to_worlds = self.metadata["camera_to_worlds"]
        # can be none if auto orient not enabled in dataparser
        self.transform = self.metadata["transform"]
        self.include_mono_prior = self.metadata["include_mono_prior"]

    def get_metadata(self, data: Dict) -> Dict:
        # TODO supports foreground_masks
        metadata = {}
        if self.include_mono_prior:
            depth_filepath = self.depth_filenames[data["image_idx"]]
            normal_filepath = self.normal_filenames[data["image_idx"]]
            camtoworld = self.camera_to_worlds[data["image_idx"]]

            # Scale depth images to meter units and also by scaling applied to cameras
            depth_image, normal_image = self.get_depths_and_normals(
                depth_filepath=depth_filepath, normal_filename=normal_filepath, camtoworld=camtoworld
            )
            metadata["depth"] = depth_image
            metadata["normal"] = normal_image

        return metadata

    def get_depths_and_normals(self, depth_filepath: Path, normal_filename: Path, camtoworld: Tensor):
        """function to process additional depths and normal information
        Args:
            depth_filepath: path to depth file
            normal_filename: path to normal file
            camtoworld: camera to world transformation matrix
        """

        # load mono depth
        depth = np.load(depth_filepath)
        depth = torch.from_numpy(depth).float()

        # load mono normal
        normal = np.load(normal_filename)

        # transform normal to world coordinate system
        normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
        normal = torch.from_numpy(normal).float()

        rot = camtoworld[:3, :3]

        normal_map = normal.reshape(3, -1)
        normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

        normal_map = rot @ normal_map
        normal = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)

        if self.transform is not None:
            h, w, _ = normal.shape
            normal = self.transform[:3, :3] @ normal.reshape(-1, 3).permute(1, 0)
            normal = normal.permute(1, 0).reshape(h, w, 3)

        return depth, normal

def get_data(self, image_idx: int, image_type: Literal["uint8", "float32", "numpy"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        elif image_type == "numpy":
            image = self.get_numpy_image(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
            )
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data
```

