from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

import os
import json
import math
import shutil
import torch


class BaseShape(ABC):
    def __init__(self, num_points=100):
        if num_points is not None:
            self.points = self.compute_points(num_points)
            self.edges = self.compute_edges()

    @abstractmethod
    def compute_points(self, num_points=100):
        pass

    def compute_edges(self):
        n = len(self.points)
        return np.column_stack((np.arange(n), (np.arange(n) + 1) % n)).astype(int)

    def get_points(self):
        return self.points

    def get_edges(self):
        return self.edges

    def to_arrays(self):
        return self.points, self.edges


class Circle(BaseShape):
    def compute_points(self, num_points=100):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.stack((x, y), axis=-1)


class Hexagon(BaseShape):
    def compute_points(self, num_points=100):
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        xx = np.cos(angles)
        yy = np.sin(angles)

        points_per_edge = max(1, num_points // 6)
        total_points = 6 * points_per_edge
        x = np.empty(total_points, dtype=float)
        y = np.empty(total_points, dtype=float)

        for i in range(6):
            x_i, y_i = xx[i], yy[i]
            x1, y1 = xx[(i + 1) % 6], yy[(i + 1) % 6]

            t = np.linspace(0, 1, points_per_edge, endpoint=False)
            x_edge = x_i + t * (x1 - x_i)
            y_edge = y_i + t * (y1 - y_i)

            start = i * points_per_edge
            end = start + points_per_edge
            x[start:end] = x_edge
            y[start:end] = y_edge

        return np.stack((x, y), axis=-1)


class Triangle(BaseShape):
    def compute_points(self, num_points=100):
        angles = np.linspace(0, 2 * np.pi, 4)[:-1]
        xx = np.cos(angles)
        yy = np.sin(angles)

        points_per_edge = max(1, num_points // 3)
        total_points = 3 * points_per_edge
        x = np.empty(total_points, dtype=float)
        y = np.empty(total_points, dtype=float)

        for i in range(3):
            x_i, y_i = xx[i], yy[i]
            x1, y1 = xx[(i + 1) % 3], yy[(i + 1) % 3]

            t = np.linspace(0, 1, points_per_edge, endpoint=False)
            x_edge = x_i + t * (x1 - x_i)
            y_edge = y_i + t * (y1 - y_i)

            start = i * points_per_edge
            end = start + points_per_edge
            x[start:end] = x_edge
            y[start:end] = y_edge

        return np.stack((x, y), axis=-1)


class Star(BaseShape):
    def __init__(self, num_points=100, n_tips=5, inner_radius=0.45):
        if n_tips < 3:
            raise ValueError("n_tips must be >= 3")
        self.n_tips = n_tips
        self.inner_radius = inner_radius
        super().__init__(num_points=num_points)

    def compute_points(self, num_points=100):
        n_vertices = 2 * self.n_tips
        outer_radius = 1.0

        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n_vertices, endpoint=False)
        radii = np.where(np.arange(n_vertices) % 2 == 0, outer_radius, self.inner_radius)
        xx = radii * np.cos(angles)
        yy = radii * np.sin(angles)

        points_per_edge = max(1, num_points // n_vertices)
        total_points = n_vertices * points_per_edge
        x = np.empty(total_points, dtype=float)
        y = np.empty(total_points, dtype=float)

        for i in range(n_vertices):
            x_i, y_i = xx[i], yy[i]
            x1, y1 = xx[(i + 1) % n_vertices], yy[(i + 1) % n_vertices]

            t = np.linspace(0, 1, points_per_edge, endpoint=False)
            x_edge = x_i + t * (x1 - x_i)
            y_edge = y_i + t * (y1 - y_i)

            start = i * points_per_edge
            end = start + points_per_edge
            x[start:end] = x_edge
            y[start:end] = y_edge

        return np.stack((x, y), axis=-1)

class RegularPolygon(BaseShape):
    def __init__(self, num_points=100, n_tips=5):
        if n_tips < 3:
            raise ValueError("n_tips must be >= 3")
        self.n_tips = n_tips
        super().__init__(num_points=num_points)

    def compute_points(self, num_points=100):
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, self.n_tips, endpoint=False)
        xx = np.cos(angles)
        yy = np.sin(angles)

        points_per_edge = max(1, num_points // self.n_tips)
        total_points = self.n_tips * points_per_edge

        x = np.empty(total_points, dtype=float)
        y = np.empty(total_points, dtype=float)

        for i in range(self.n_tips):
            x_i, y_i = xx[i], yy[i]
            x1, y1 = xx[(i + 1) % self.n_tips], yy[(i + 1) % self.n_tips]

            t = np.linspace(0, 1, points_per_edge, endpoint=False)
            x_edge = x_i + t * (x1 - x_i)
            y_edge = y_i + t * (y1 - y_i)

            start = i * points_per_edge
            end = start + points_per_edge
            x[start:end] = x_edge
            y[start:end] = y_edge

        return np.stack((x, y), axis=-1)


class Cog(BaseShape):
    def __init__(self, num_points=100, n_tips=8, inner_radius=0.72, outer_radius=1.0, tip_width=0.45):
        if n_tips < 3:
            raise ValueError("n_tips must be >= 3")
        if not (0 < inner_radius < outer_radius):
            raise ValueError("Require 0 < inner_radius < outer_radius")
        if not (0 < tip_width < 1):
            raise ValueError("tip_width must be in (0, 1)")

        self.n_tips = n_tips
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.tip_width = tip_width
        super().__init__(num_points=num_points)

    def compute_points(self, num_points=100):
        verts = []
        tip_angle = 2 * np.pi / self.n_tips
        flat_half_width = 0.5 * self.tip_width * tip_angle

        for i in range(self.n_tips):
            center = np.pi / 2 + i * tip_angle

            a0 = center - tip_angle / 2
            a1 = center - flat_half_width
            a2 = center + flat_half_width
            a3 = center + tip_angle / 2

            verts.extend([
                [self.inner_radius * np.cos(a0), self.inner_radius * np.sin(a0)],
                [self.outer_radius * np.cos(a1), self.outer_radius * np.sin(a1)],
                [self.outer_radius * np.cos(a2), self.outer_radius * np.sin(a2)],
                [self.inner_radius * np.cos(a3), self.inner_radius * np.sin(a3)],
            ])

        verts = np.asarray(verts, dtype=float)
        n_vertices = len(verts)

        points_per_edge = max(1, num_points // n_vertices)
        total_points = n_vertices * points_per_edge

        x = np.empty(total_points, dtype=float)
        y = np.empty(total_points, dtype=float)

        for i in range(n_vertices):
            x_i, y_i = verts[i]
            x1, y1 = verts[(i + 1) % n_vertices]

            t = np.linspace(0, 1, points_per_edge, endpoint=False)
            x_edge = x_i + t * (x1 - x_i)
            y_edge = y_i + t * (y1 - y_i)

            start = i * points_per_edge
            end = start + points_per_edge
            x[start:end] = x_edge
            y[start:end] = y_edge

        return np.stack((x, y), axis=-1)


class Flower(BaseShape):
    def __init__(self, num_points=100, n_tips=6, inner_radius=0.75):
        if n_tips < 2:
            raise ValueError("n_tips must be >= 2")
        if not (0 < inner_radius < 1):
            raise ValueError("inner_radius must be in (0, 1)")

        self.n_tips = n_tips
        self.inner_radius = inner_radius
        super().__init__(num_points=num_points)

    def compute_points(self, num_points=100):
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        amplitude = 1.0 - self.inner_radius
        r = 1.0 - 0.5 * amplitude + 0.5 * amplitude * np.cos(self.n_tips * theta)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return np.stack((x, y), axis=-1)


class Heart(BaseShape):
    def compute_points(self, num_points=100):
        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        x = 16 * np.sin(t) ** 3
        y = (
            13 * np.cos(t)
            - 5 * np.cos(2 * t)
            - 2 * np.cos(3 * t)
            - np.cos(4 * t)
        )

        points = np.stack((x, y), axis=-1)

        points = points - points.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / scale

        return points

class Moon(BaseShape):
    def __init__(self, num_points=100, inner_radius=0.75, offset=0.35):
        if not (0 < inner_radius < 1):
            raise ValueError("inner_radius must be in (0, 1)")
        if offset <= 0:
            raise ValueError("offset must be > 0")

        self.outer_radius = 1.0
        self.inner_radius = inner_radius
        self.offset = offset
        super().__init__(num_points=num_points)

    def compute_points(self, num_points=100):
        R = self.outer_radius
        r = self.inner_radius
        d = self.offset

        # Need intersecting circles
        if not (abs(R - r) < d < R + r):
            raise ValueError("Moon parameters must satisfy |R-r| < offset < R+r")

        # Intersection geometry
        a = (R**2 - r**2 + d**2) / (2 * d)
        h = np.sqrt(max(R**2 - a**2, 0.0))

        x_int = a
        y_int = h

        # Angles on outer circle
        theta_top = np.arctan2(y_int, x_int)
        theta_bottom = np.arctan2(-y_int, x_int)

        # Angles on inner circle (centered at (d,0))
        phi_top = np.arctan2(y_int, x_int - d)
        phi_bottom = np.arctan2(-y_int, x_int - d)

        n_outer = num_points // 2
        n_inner = num_points - n_outer

        # Outer boundary: take the LEFT major arc from top -> bottom
        outer_angles = np.linspace(theta_top, theta_bottom + 2 * np.pi, n_outer, endpoint=False)
        x_outer = R * np.cos(outer_angles)
        y_outer = R * np.sin(outer_angles)

        # Inner boundary: take the LEFT major arc from bottom -> top
        # This must go the long way around, so we go clockwise.
        inner_angles = np.linspace(phi_bottom, phi_top - 2 * np.pi, n_inner, endpoint=False)
        x_inner = d + r * np.cos(inner_angles)
        y_inner = r * np.sin(inner_angles)

        x = np.concatenate([x_outer, x_inner])
        y = np.concatenate([y_outer, y_inner])

        points = np.stack((x, y), axis=-1)

        # Normalize
        points = points - points.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / scale

        return points

class Blob(BaseShape):
    def __init__(self, num_points=100, n_tips=5, inner_radius=0.75, seed=0):
        if n_tips < 1:
            raise ValueError("n_tips must be >= 1")
        if not (0 < inner_radius < 1):
            raise ValueError("inner_radius must be in (0, 1)")

        self.n_tips = n_tips
        self.inner_radius = inner_radius
        self.seed = seed
        super().__init__(num_points=num_points)

    def compute_points(self, num_points=100):
        rng = np.random.default_rng(self.seed)

        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        r = np.ones_like(theta)

        amplitude = 1.0 - self.inner_radius

        for k in range(1, self.n_tips + 1):
            a = rng.uniform(-amplitude, amplitude) / k
            b = rng.uniform(-amplitude, amplitude) / k
            r += a * np.cos(k * theta) + b * np.sin(k * theta)

        r = np.clip(r, self.inner_radius, None)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        points = np.stack((x, y), axis=-1)

        points = points - points.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / scale

        return points



class TransformedShapeBatch:
    """
    Represents many transformed versions of one base shape.

    The affine transform is:

        new_points = base_points @ linear_matrix.T + translation

    where:
        base_points:     [P, 2]
        linear_matrix:   [B, 2, 2]
        translation:     [B, 2]
        new_points:      [B, P, 2]
    """

    def __init__(
        self,
        name,
        base_shape: BaseShape,
        translation,
        linear_matrix,
        device="cpu",
        dtype=torch.float32,
    ):
        self.name = name
        self.base_shape = base_shape
        self.device = torch.device(device)
        self.dtype = dtype

        self.translation = torch.as_tensor(
            translation,
            device=self.device,
            dtype=self.dtype,
        )

        self.linear_matrix = torch.as_tensor(
            linear_matrix,
            device=self.device,
            dtype=self.dtype,
        )

        self.base_points = torch.as_tensor(
            self.base_shape.get_points(),
            device=self.device,
            dtype=self.dtype,
        )

        self.edges = torch.as_tensor(
            self.base_shape.get_edges(),
            device=self.device,
            dtype=torch.long,
        )

        if self.translation.ndim != 2 or self.translation.shape[1] != 2:
            raise ValueError("translation must have shape [B, 2]")

        if self.linear_matrix.ndim != 3 or self.linear_matrix.shape[1:] != (2, 2):
            raise ValueError("linear_matrix must have shape [B, 2, 2]")

        if self.translation.shape[0] != self.linear_matrix.shape[0]:
            raise ValueError("translation and linear_matrix must have the same batch size")

    def __len__(self):
        return self.translation.shape[0]

    @property
    def num_points(self):
        return self.base_points.shape[0]

    def points(self, max_samples=None):
        if max_samples is None:
            translation = self.translation
            linear_matrix = self.linear_matrix
        else:
            translation = self.translation[:max_samples]
            linear_matrix = self.linear_matrix[:max_samples]

        points = torch.matmul(
            self.base_points.unsqueeze(0),
            linear_matrix.transpose(1, 2),
        )

        points = points + translation.unsqueeze(1)

        return points

    def to_arrays(self, max_samples=None):
        return (
            self.points(max_samples=max_samples).detach().cpu().numpy(),
            self.edges.detach().cpu().numpy(),
        )

    def to_payload(self, save_points=False):
        payload = {
            "translation": self.translation.detach().cpu(),
            "linear_matrix": self.linear_matrix.detach().cpu(),
        }

        if save_points:
            payload["points"] = self.points().detach().cpu()
            payload["edges"] = self.edges.detach().cpu()

        return payload

    def get_shape_array(self, index=0):
        points = self.points(max_samples=index + 1)[index].detach().cpu().numpy()
        edges = self.edges.detach().cpu().numpy()
        return points, edges

    def get_base_shape_array(self):
        return (
            self.base_points.detach().cpu().numpy(),
            self.edges.detach().cpu().numpy(),
        )


class ShapeMixtureBatch:
    """
    Holds several TransformedShapeBatch objects.
    Usually one batch per shape type.
    """

    def __init__(self):
        self.batches = {}

    def add(self, name, batch: TransformedShapeBatch):
        self.batches[name] = batch

    def __getitem__(self, name):
        return self.batches[name]

    def __len__(self):
        return sum(len(batch) for batch in self.batches.values())

    def keys(self):
        return self.batches.keys()

    def items(self):
        return self.batches.items()

    def plot(
        self,
        max_per_shape=10,
        colors=None,
        figsize=(8, 8),
        point_size=2,
        point_alpha=0.2,
        line_alpha=0.45,
        linewidth=0.7,
        title="Shape mixture batch",
        output_file=None,
        show=True,
    ):
        if colors is None:
            colors = {
                "circle": "tab:blue",
                "hexagon": "tab:orange",
                "triangle": "tab:green",
                "star_5": "tab:red",
                "star_12": "tab:purple",
            }

        fig, ax = plt.subplots(figsize=figsize)

        for shape_name, batch in self.batches.items():
            color = colors.get(shape_name, None)

            n = len(batch) if max_per_shape is None else min(max_per_shape, len(batch))
            points, edges = batch.to_arrays(max_samples=n)

            first_label = True

            for i in range(n):
                p = points[i]
                label = shape_name if first_label else None

                for edge in edges:
                    ax.plot(
                        p[edge, 0],
                        p[edge, 1],
                        color=color,
                        alpha=line_alpha,
                        linewidth=linewidth,
                        label=label,
                    )
                    label = None

                ax.scatter(
                    p[:, 0],
                    p[:, 1],
                    color=color,
                    s=point_size,
                    alpha=point_alpha,
                )

                first_label = False

        ax.axis("equal")
        ax.legend()
        ax.set_title(title)

        if output_file is not None:
            fig.savefig(output_file, dpi=200, bbox_inches="tight", facecolor="white")

        if show:
            plt.show()

        return fig, ax


# the reason of this class is to cache base shapes and not recompute them each time
# also makes large augmentation generation fast in torch.
class ShapeGenerator:
    def __init__(self):
        self.shape_classes = {
            "circle": Circle,
            "hexagon": Hexagon,
            "triangle": Triangle,
            "star": Star,
            "polygon": RegularPolygon,
            "cog": Cog,
            "flower": Flower,
            "heart": Heart,
            "moon": Moon,
            "blob": Blob,
        }

        # NumPy BaseShape cache
        self.base_shape_cache = {}

        # Torch tensor cache
        self.base_tensor_cache = {}

    # Returns a cached BaseShape object, or creates it if it does not exist yet.
    def get_base_shape(self, shape_name, num_points=100, **shape_kwargs):
        key = (shape_name, num_points, tuple(sorted(shape_kwargs.items())))

        if key not in self.base_shape_cache:
            if shape_name not in self.shape_classes:
                raise ValueError(f"Unsupported shape: {shape_name}")

            self.base_shape_cache[key] = self.shape_classes[shape_name](
                num_points=num_points,
                **shape_kwargs,
            )

        return self.base_shape_cache[key]

    # Converts a cached NumPy BaseShape into Torch tensors.
    def get_base_tensors(
        self,
        shape_name,
        num_points=100,
        device="cpu",
        dtype=torch.float32,
        **shape_kwargs,
    ):

        device = torch.device(device)

        key = (
            shape_name,
            num_points,
            tuple(sorted(shape_kwargs.items())),
            str(device),
            str(dtype),
        )

        if key not in self.base_tensor_cache:
            base_shape = self.get_base_shape(
                shape_name,
                num_points=num_points,
                **shape_kwargs,
            )

            points_np, edges_np = base_shape.to_arrays()

            points = torch.as_tensor(points_np, device=device, dtype=dtype)
            edges = torch.as_tensor(edges_np, device=device, dtype=torch.long)

            self.base_tensor_cache[key] = (points, edges)

        return self.base_tensor_cache[key]

    # Creates a seeded Torch random generator.
    # This makes the Torch dataset generation reproducible.
    @staticmethod
    def _make_generator(seed, device):
        device = torch.device(device)
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        return g

    # Samples uniform random values in [low, high] with Torch.
    @staticmethod
    def _uniform(low, high, size, generator, device, dtype):
        return low + (high - low) * torch.rand(
            size,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    # Reads a range from either a normal dict or a Hydra DictConfig.
    @staticmethod
    def _get_range(transform_cfg, name):
        if isinstance(transform_cfg, dict):
            value = transform_cfg[name]
        else:
            value = getattr(transform_cfg, name)

        return float(value[0]), float(value[1])

    # Reads optional values from either a normal dict or a Hydra DictConfig.
    @staticmethod
    def _cfg_get(cfg, name, default):
        if isinstance(cfg, dict):
            return cfg.get(name, default)

        if hasattr(cfg, "get"):
            return cfg.get(name, default)

        return getattr(cfg, name, default)

    # Samples a whole batch of translations and 2x2 linear transformation matrices.
    def sample_transforms_torch(
        self,
        shape_name,
        batch_size,
        transform_cfg,
        generator,
        device,
        dtype,
    ):
        t_lo, t_hi = self._get_range(transform_cfg, "translation_range")
        s_lo, s_hi = self._get_range(transform_cfg, "scale_range")
        r_lo, r_hi = self._get_range(transform_cfg, "rotation_range")

        translation = self._uniform(
            t_lo,
            t_hi,
            (batch_size, 2),
            generator,
            device,
            dtype,
        )

        isotropic_scale = bool(
            self._cfg_get(transform_cfg, "isotropic_scale", False)
        )

        scale_x = self._uniform(
            s_lo,
            s_hi,
            (batch_size,),
            generator,
            device,
            dtype,
        )

        if isotropic_scale:
            scale_y = scale_x
        else:
            scale_y = self._uniform(
                s_lo,
                s_hi,
                (batch_size,),
                generator,
                device,
                dtype,
            )

        angle_deg = self._uniform(
            r_lo,
            r_hi,
            (batch_size,),
            generator,
            device,
            dtype,
        )

        angle = angle_deg * math.pi / 180.0

        c = torch.cos(angle)
        s = torch.sin(angle)

        linear_matrix = torch.empty(
            batch_size,
            2,
            2,
            device=device,
            dtype=dtype,
        )

        # A = R @ diag(scale_x, scale_y)
        linear_matrix[:, 0, 0] = c * scale_x
        linear_matrix[:, 0, 1] = -s * scale_y
        linear_matrix[:, 1, 0] = s * scale_x
        linear_matrix[:, 1, 1] = c * scale_y

        return translation, linear_matrix

    # Applies one 2x2 linear transform and one translation to each copy of the base shape.
    @staticmethod
    def transform_points_torch(base_points, translation, linear_matrix):
        """
        base_points: [P, 2]
        translation: [B, 2]
        linear_matrix: [B, 2, 2]

        returns:
            points: [B, P, 2]
        """

        points = torch.matmul(
            base_points.unsqueeze(0),
            linear_matrix.transpose(1, 2),
        )

        points = points + translation.unsqueeze(1)

        return points

    # Generates one transformed batch for a single shape type.
    def generate_batch_torch(
        self,
        instance_name,
        shape_spec,
        transform_cfg,
        batch_size,
        seed=0,
        device="cpu",
        dtype=torch.float32,
    ):
        """
        Generates many transformed copies of one base shape in parallel.
        """

        device = torch.device(device)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but CUDA is not available.")

        shape_name = shape_spec.get("shape", instance_name)
        num_points = shape_spec.get("num_points", 100)

        shape_kwargs = {
            k: v for k, v in shape_spec.items()
            if k not in ("shape", "num_points", "percentage", "transform", "name")
        }

        base_shape = self.get_base_shape(
            shape_name=shape_name,
            num_points=num_points,
            **shape_kwargs,
        )

        generator = self._make_generator(seed, device)

        translation, linear_matrix = self.sample_transforms_torch(
            shape_name=shape_name,
            batch_size=batch_size,
            transform_cfg=transform_cfg,
            generator=generator,
            device=device,
            dtype=dtype,
        )

        return TransformedShapeBatch(
            name=instance_name,
            base_shape=base_shape,
            translation=translation,
            linear_matrix=linear_matrix,
            device=device,
            dtype=dtype,
        )

    # Generates a mixture batch containing multiple shape types.
    def generate_mixture_batch_torch(
        self,
        shapes,
        transform_cfg,
        N=None,
        samples_per_shape=None,
        seed=0,
        device="cpu",
        dtype=torch.float32,
    ):
        if samples_per_shape is None and N is None:
            raise ValueError("Either N or samples_per_shape must be provided.")

        if samples_per_shape is not None:
            counts = {name: int(samples_per_shape) for name in shapes.keys()}
        else:
            counts = self.counts_from_percentages(shapes, N)

        mixture = ShapeMixtureBatch()

        for spec_idx, (instance_name, shape_spec) in enumerate(shapes.items()):
            count = counts[instance_name]
            batch_seed = seed + spec_idx * 1_000_003

            batch = self.generate_batch_torch(
                instance_name=instance_name,
                shape_spec=shape_spec,
                transform_cfg=transform_cfg,
                batch_size=count,
                seed=batch_seed,
                device=device,
                dtype=dtype,
            )

            mixture.add(instance_name, batch)

        return mixture

    # Computes how many samples each shape should get from its percentage.
    @staticmethod
    def counts_from_percentages(shapes, N):
        names = list(shapes.keys())

        percentages = []
        for name in names:
            percentages.append(float(shapes[name].get("percentage", 1.0 / len(names))))

        total = sum(percentages)

        if abs(total - 1.0) > 1e-8:
            percentages = [p / total for p in percentages]

        raw = [p * N for p in percentages]
        counts = [int(math.floor(x)) for x in raw]

        remainder = N - sum(counts)

        fractional_parts = sorted(
            enumerate([x - c for x, c in zip(raw, counts)]),
            key=lambda x: (-x[1], x[0]),
        )

        for i in range(remainder):
            counts[fractional_parts[i][0]] += 1

        return dict(zip(names, counts))

    # Generates the full dataset in shards and saves it to disk.
    # This avoids keeping the whole dataset in RAM.
    def generate_to_disk_torch(
        self,
        shapes,
        transform_cfg,
        out_dir,
        N,
        batch_size=8192,
        seed=0,
        device="cpu",
        dtype=torch.float32,
        save_mode="params",
        overwrite=False,
    ):
        """
        Efficient dataset generation.

        save_mode="params":
            saves only translation and linear_matrix.
            Best for huge datasets.

        save_mode="full":
            also saves transformed points.
            Much larger on disk.
        """

        if save_mode not in ("params", "full"):
            raise ValueError("save_mode must be either 'params' or 'full'")

        device = torch.device(device)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but CUDA is not available.")

        if overwrite and os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        os.makedirs(out_dir, exist_ok=True)

        counts = self.counts_from_percentages(shapes, N)

        manifest = {
            "N": int(N),
            "seed": int(seed),
            "batch_size": int(batch_size),
            "device": str(device),
            "dtype": str(dtype),
            "save_mode": save_mode,
            "specs": [],
            "shards": [],
        }

        for spec_idx, (instance_name, shape_spec) in enumerate(shapes.items()):
            count = counts[instance_name]

            shape_name = shape_spec.get("shape", instance_name)
            num_points = shape_spec.get("num_points", 100)
            save_name = shape_spec.get("name", instance_name)

            shape_kwargs = {
                k: v for k, v in shape_spec.items()
                if k not in ("shape", "num_points", "percentage", "transform", "name")
            }

            spec_dir = os.path.join(out_dir, f"{spec_idx:02d}_{save_name}")
            os.makedirs(spec_dir, exist_ok=True)

            base_points, edges = self.get_base_tensors(
                shape_name=shape_name,
                num_points=num_points,
                device=device,
                dtype=dtype,
                **shape_kwargs,
            )

            base_path = os.path.join(spec_dir, "base_shape.pt")

            torch.save(
                {
                    "base_points": base_points.cpu(),
                    "edges": edges.cpu(),
                    "shape": shape_name,
                    "num_points": int(base_points.shape[0]),
                    "params": shape_kwargs,
                    "name": save_name,
                },
                base_path,
            )

            manifest["specs"].append(
                {
                    "spec_idx": int(spec_idx),
                    "instance_name": instance_name,
                    "shape": shape_name,
                    "num_samples": int(count),
                    "num_points": int(base_points.shape[0]),
                    "params": shape_kwargs,
                    "base_shape_file": os.path.relpath(base_path, out_dir),
                }
            )

            num_shards = math.ceil(count / batch_size)

            for shard_idx in range(num_shards):
                current_batch = min(batch_size, count - shard_idx * batch_size)
                shard_seed = seed + spec_idx * 1_000_003 + shard_idx

                batch = self.generate_batch_torch(
                    instance_name=instance_name,
                    shape_spec=shape_spec,
                    transform_cfg=transform_cfg,
                    batch_size=current_batch,
                    seed=shard_seed,
                    device=device,
                    dtype=dtype,
                )

                payload = batch.to_payload(save_points=(save_mode == "full"))

                payload["spec_idx"] = torch.full(
                    (current_batch,),
                    spec_idx,
                    dtype=torch.long,
                )

                shard_path = os.path.join(spec_dir, f"shard_{shard_idx:06d}.pt")

                torch.save(payload, shard_path)

                manifest["shards"].append(
                    {
                        "path": os.path.relpath(shard_path, out_dir),
                        "spec_idx": int(spec_idx),
                        "num_samples": int(current_batch),
                        "seed": int(shard_seed),
                    }
                )

        manifest_path = os.path.join(out_dir, "manifest.json")

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest
