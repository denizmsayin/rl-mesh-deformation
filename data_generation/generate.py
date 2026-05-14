import random 
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra  

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







# the reason of this class is to cache base shapes and not recompute them each time
# also makes large augmentation generation fast in torch.
class ShapeGenerator:
    def __init__(self):
        self.shape_classes = {
            "circle": Circle,
            "hexagon": Hexagon,
            "triangle": Triangle,
            "star": Star,
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

    # Builds one transformed shape using the NumPy path.
    # This is mainly for visualization/debugging, not for large dataset generation.
    def build_shape(
        self,
        shape_name,
        num_points=100,
        center=(0.0, 0.0),
        linear_matrix=None,
        scale=1.0,
        angle=0.0,
        **shape_kwargs,
    ):
        base_shape = self.get_base_shape(shape_name, num_points, **shape_kwargs)

        return TransformedShape(
            base_shape,
            translation=center,
            linear_matrix=linear_matrix,
            scale=scale,
            angle=angle,
        )

    # Samples one random translation and one random 2x2 linear transform.
    # This is for visualization and not for large dataset generation.
    def build_random_shape(self, instance_name, shape_spec, transform_cfg):
        shape_name = shape_spec.get("shape", instance_name)
        num_points = shape_spec.get("num_points", 100)

        shape_kwargs = {
            k: v for k, v in shape_spec.items()
            if k not in ("shape", "num_points", "percentage", "transform", "name")
        }

        center = (
            random.uniform(*transform_cfg.translation_range),
            random.uniform(*transform_cfg.translation_range),
        )

        scale_x = random.uniform(*transform_cfg.scale_range)
        scale_y = random.uniform(*transform_cfg.scale_range)

        angle = np.deg2rad(random.uniform(*transform_cfg.rotation_range))

        c, s = np.cos(angle), np.sin(angle)

        # A = R @ diag(scale_x, scale_y)
        linear_matrix = np.array(
            [
                [c * scale_x, -s * scale_y],
                [s * scale_x,  c * scale_y],
            ],
            dtype=float,
        )

        return self.build_shape(
            shape_name=shape_name,
            num_points=num_points,
            center=center,
            linear_matrix=linear_matrix,
            **shape_kwargs,
        )

    def generate_shapes(self, shapes, transform_cfg):
        V = []
        L = []

        for instance_name, shape_spec in shapes.items():
            shape = self.build_random_shape(instance_name, shape_spec, transform_cfg)
            vertices, edges = shape.to_arrays()

            V.append(vertices)
            L.append(edges)

        return V, L

    # ----------------------------
    # Torch for large generation
    # ----------------------------

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

        scale_x = self._uniform(
            s_lo,
            s_hi,
            (batch_size,),
            generator,
            device,
            dtype,
        )

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
    # This is the main vectorized Torch operation.
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

    # Generates one Torch batch for a single shape type.
    # If return_points=False, it saves memory by only returning transform parameters.
    def generate_batch_torch(
        self,
        instance_name,
        shape_spec,
        transform_cfg,
        batch_size,
        seed=0,
        device="cpu",
        dtype=torch.float32,
        return_points=True,
    ):
        """
        Generates many transformed copies of one base shape in parallel.
        """

        device = torch.device(device)

        shape_name = shape_spec.get("shape", instance_name)
        num_points = shape_spec.get("num_points", 100)

        shape_kwargs = {
            k: v for k, v in shape_spec.items()
            if k not in ("shape", "num_points", "percentage", "transform", "name")
        }

        base_points, edges = self.get_base_tensors(
            shape_name=shape_name,
            num_points=num_points,
            device=device,
            dtype=dtype,
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

        output = {
            "translation": translation,
            "linear_matrix": linear_matrix,
            "edges": edges,
        }

        if return_points:
            output["points"] = self.transform_points_torch(
                base_points=base_points,
                translation=translation,
                linear_matrix=linear_matrix,
            )

        return output

    # Computes how many samples each shape should get from its percentage.
    # Example: percentage 0.30 with N=100000 gives about 30000 samples.
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

        if device == "cuda" and not torch.cuda.is_available():
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
                    return_points=(save_mode == "full"),
                )

                payload = {
                    "translation": batch["translation"].cpu(),
                    "linear_matrix": batch["linear_matrix"].cpu(),
                    "spec_idx": torch.full(
                        (current_batch,),
                        spec_idx,
                        dtype=torch.long,
                    ),
                }

                if save_mode == "full":
                    payload["points"] = batch["points"].cpu()
                    payload["edges"] = batch["edges"].cpu()

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


class TransformedShape:
    def __init__(
        self,
        baseshape: BaseShape,
        translation: tuple = (0.0, 0.0),
        linear_matrix=None,
        scale: float = 1.0,
        angle: float = 0.0,
    ):
        self.baseshape = baseshape
        self.translation = np.asarray(translation, dtype=float)
        self.linear_matrix = None if linear_matrix is None else np.asarray(linear_matrix, dtype=float)
        self.scale = scale
        self.angle = angle
        self.points = self.compute_points()

    def compute_points(self):
        points = np.asarray(self.baseshape.get_points(), dtype=float)

        if self.linear_matrix is not None:
            points = points @ self.linear_matrix.T
        else:
            if self.angle != 0.0:
                c, s = np.cos(self.angle), np.sin(self.angle)
                rotation = np.array([[c, -s], [s, c]])
                points = points @ rotation.T

            points = points * self.scale

        points = points + self.translation

        return points

    def get_points(self):
        return self.points

    def get_edges(self):
        return self.baseshape.get_edges()

    def to_arrays(self):
        return self.points, self.get_edges()


           
@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def main(cfg: DictConfig):
    print("Generating shapes with the following configuration:")
    print(cfg)

    shapes = {
        "circle": {"shape": "circle", "num_points": 60, "percentage": 0.30},
        "hexagon": {"shape": "hexagon", "num_points": 60, "percentage": 0.25},
        "triangle": {"shape": "triangle", "num_points": 60, "percentage": 0.20},
        "star_5": {"shape": "star", "num_points": 60, "n_tips": 5, "inner_radius": 0.45, "percentage": 0.15},
        "star_12": {"shape": "star", "num_points": 2500, "n_tips": 12, "inner_radius": 0.6, "percentage": 0.10},
    }

    shape_generator = ShapeGenerator()

    V, L = shape_generator.generate_shapes(shapes, cfg)

    print("Shapes generated successfully!")

    fig, ax = plt.subplots()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple','tab:cyan']

    for i, (shape_name, v, l) in enumerate(zip(shapes.keys(), V, L)):
        color = colors[i % len(colors)]
        ax.scatter(v[:, 0], v[:, 1], color=color, label=shape_name)

        for edge in l:
            ax.plot(v[edge, 0], v[edge, 1], color=color, linewidth=1.2)

    ax.axis('equal')

    output_file = "data_generation/generated_shapes.png"
    fig.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved {output_file}")
    #plt.show()

    manifest = shape_generator.generate_to_disk_torch(
        shapes=shapes,
        transform_cfg=cfg,
        out_dir="data_generation/generated_dataset",
        N=100_000,
        batch_size=4096,
        seed=123,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_mode="params",
        overwrite=True,
    )

    print("Dataset generated successfully!")
    print(f"Number of shards: {len(manifest['shards'])}")


if __name__ == "__main__":
    main()