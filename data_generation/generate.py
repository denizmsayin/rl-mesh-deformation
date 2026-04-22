import random 
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra  




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

#the reason of this class is to cache base shapes and not recompute them each time
class ShapeGenerator:
    def __init__(self):
        self.shape_classes = {
            'circle': Circle,
            'hexagon': Hexagon,
            'triangle': Triangle,
            'star': Star,
        }
        self.base_shape_cache = {}

    def get_base_shape(self, shape_name, num_points=100, **shape_kwargs):
        key = (shape_name, num_points, tuple(sorted(shape_kwargs.items())))
        if key not in self.base_shape_cache:
            if shape_name not in self.shape_classes:
                raise ValueError(f"Unsupported shape: {shape_name}")
            self.base_shape_cache[key] = self.shape_classes[shape_name](
                num_points=num_points, **shape_kwargs
            )
        return self.base_shape_cache[key]

    def build_shape(self, shape_name, num_points=100, center=(0.0, 0.0), scale=1.0, angle=0.0, **shape_kwargs):
        base_shape = self.get_base_shape(shape_name, num_points, **shape_kwargs)
        return TransformedShape(base_shape,translation=center,scale=scale,angle=angle,)

    def build_random_shape(self, instance_name, shape_spec, transform_cfg):
        shape_name = shape_spec.get('shape', instance_name)
        num_points = shape_spec.get('num_points', 100)
        shape_kwargs = {
            k: v for k, v in shape_spec.items()
            if k not in ('shape', 'num_points')
        }

        center = (
            random.uniform(*transform_cfg.translation_range),
            random.uniform(*transform_cfg.translation_range),
        )
        scale = random.uniform(*transform_cfg.scale_range)
        angle = 0.0 if shape_name == 'circle' else np.deg2rad(
            random.uniform(*transform_cfg.rotation_range)
        )

        return self.build_shape(shape_name=shape_name,num_points=num_points,center=center,scale=scale,angle=angle,**shape_kwargs)

    def generate_shapes(self, shapes, transform_cfg):
        V = []
        L = []

        for instance_name, shape_spec in shapes.items():
            shape = self.build_random_shape(instance_name, shape_spec, transform_cfg)
            vertices, edges = shape.to_arrays()
            V.append(vertices)
            L.append(edges)

        return V, L


class TransformedShape:
    def __init__(self, baseshape: BaseShape, translation: tuple = (0.0, 0.0), scale: float = 1.0, angle: float = 0.0):
        self.baseshape = baseshape # if the same across different instances, it should not get copied.
        self.translation = np.asarray(translation)
        self.scale = scale
        self.angle = angle
        self.points = self.compute_points()
        #self.edges = np.asarray(self.baseshape.get_edges(), dtype=int)

    def compute_points(self):
        points = np.asarray(self.baseshape.get_points(), dtype=float)

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
    shapes ={
        'circle': {'shape': 'circle', 'num_points': 60},
        'hexagon': {'shape': 'hexagon', 'num_points': 60},
        'triangle': {'shape': 'triangle', 'num_points': 60},
        'star_5': {'shape': 'star', 'num_points': 60, 'n_tips': 5, 'inner_radius': 0.45},
        'star_12': {'shape': 'star', 'num_points': 2500, 'n_tips': 12, 'inner_radius': 0.6},
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


if __name__ == "__main__":    main()


   

   