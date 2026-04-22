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

    @abstractmethod
    def compute_points(self, num_points=100):
        pass

    def get_points(self):
        return self.points


class Circle(BaseShape):
    def compute_points(self, num_points=100):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        return x, y
    

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

        return x, y


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

        return x, y


class TransformedShape:
    def __init__(self, shape: BaseShape, translation: tuple = (0.0, 0.0), scale: float = 1.0, angle: float = 0.0):
        self.shape = shape
        self.translation = np.asarray(translation)
        self.scale = scale
        self.angle = angle
        self.points = self.compute_points()

    def compute_points(self):
        x, y = self.shape.get_points()
        points = np.stack((x, y), axis=-1)

        if self.angle != 0.0:
            c, s = np.cos(self.angle), np.sin(self.angle)
            rotation = np.array([[c, -s], [s, c]])
            points = points @ rotation.T

        points = points * self.scale
        points = points + self.translation
        return points[:, 0], points[:, 1]

    def get_points(self):
        return self.points


class GenerateShape:
    def __init__(self, shapes: dict, transform_cgf: DictConfig):
        self.shapes = shapes
        self.shape_classes = {
            'circle': Circle,
            'hexagon': Hexagon,
            'triangle': Triangle,
        }
        self.base_shape_cache = {}
        translation_range = transform_cgf.translation_range
        scale_range = transform_cgf.scale_range
        rotation_range = transform_cgf.rotation_range
        self.transformations = {
            'translation': lambda: (random.uniform(*translation_range), random.uniform(*translation_range)),
            'scale': lambda: random.uniform(*scale_range),
            'rotation': lambda: np.deg2rad(random.uniform(*rotation_range))
        }

    def get_base_shape(self, shape_name: str, num_points: int):
        key = (shape_name, num_points)
        if key not in self.base_shape_cache:
            if shape_name not in self.shape_classes:
                raise ValueError(f"Unsupported shape: {shape_name}")
            self.base_shape_cache[key] = self.shape_classes[shape_name](num_points=num_points)
        return self.base_shape_cache[key]
       
    def generate_shapes(self):
        V=[]
        L=[]

        for shape_name, num_points in self.shapes.items():
            translation = self.transformations['translation']()
            scale = self.transformations['scale']()
            rotation = self.transformations['rotation']()

            base_shape = self.get_base_shape(shape_name, num_points)
            angle = 0.0 if shape_name == 'circle' else rotation
            shape = TransformedShape(base_shape, translation=translation, scale=scale, angle=angle)

            x, y = shape.get_points()
            coords = np.stack((x, y), axis=-1)
            V.append(coords)  # vertices of current shape
            L.append(self.connectivity(coords))  # connectivity of current shape
        return V, L

    def connectivity(self, vertices):
        edges = []
        for i in range(len(vertices)):
            edges.append((i, (i + 1) % len(vertices)))  # close the polygon
        return edges
           
@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def main(cfg: DictConfig):
    print("Generating shapes with the following configuration:")
    print(cfg)
    shapes ={
        'circle': 60,
        'hexagon': 60,
        'triangle': 60
    }
    shape_generator = GenerateShape(shapes, cfg)
    V, L = shape_generator.generate_shapes()
    print("Shapes generated successfully!")
    fig, ax = plt.subplots()
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (shape_name, v, l) in enumerate(zip(shapes.keys(), V, L)):
        color = colors[i % len(colors)]
        ax.scatter(v[:, 0], v[:, 1], color=color, label=shape_name)
        for edge in l:
            ax.plot(v[edge, 0], v[edge, 1], color=color, linewidth=1.2)

    ax.axis('equal')
    output_file = "data_generation/generated_shapes.png"
    fig.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved {output_file}")
    plt.show()


if __name__ == "__main__":    main()


   

   