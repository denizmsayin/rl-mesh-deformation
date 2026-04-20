import random 
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra  




class Circle():
    def __init__(self, translation: tuple, scale: float, num_points=100):
        self.radius = 1.0
        self.center = (0.0, 0.0)
        self.translate(translation)
        self.scale(scale)
        if num_points is not None:
            self.points = self.compute_points(num_points)

    def get_radius(self):
        return self.radius
    
    def get_center(self):
        return self.center
    
    def translate(self, dx, dy):
        self.center = (self.center[0] + dx, self.center[1] + dy)

    def scale(self, alpha):
        self.radius *= alpha

    def compute_points(self, num_points=100):
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = self.center[0] + self.radius * np.cos(angles)
        y = self.center[1] + self.radius * np.sin(angles)
        return x, y
    
    def get_points(self):
        return self.points
    
class Hexagon(Circle):
    def __init__(self, angle: float, translation: tuple, scale: float, num_points=100):
        super().__init__(translation, scale, num_points = None)
        self.angle = angle
        self.points = self.compute_points(num_points, angle)

    def compute_points(self, num_points=100, angle=0.0):
        angles = np.linspace(0, 2 * np.pi, 7)[:-1] + angle
        xx = self.center[0] + self.radius * np.cos(angles)
        yy = self.center[1] + self.radius * np.sin(angles)

        points_per_edge = num_points // 6
        x = np.array([])
        y = np.array([])
        for x_i, y_i in zip(xx, yy):
            for j in np.linspace(x_i, xx[(np.where(xx == x_i)[0] + 1) % 6], points_per_edge, endpoint=False):
                x = np.append(x, j)
                y = np.append(y, np.interp(j, [x_i, xx[(np.where(xx == x_i)[0] + 1) % 6]], [y_i, yy[(np.where(yy == y_i)[0] + 1) % 6]]))

        return x, y

class Triangle(Circle):
    def __init__(self, angle: float, translation: tuple, scale: float, num_points=100):
        super().__init__(translation, scale, num_points = None)
        self.angle = angle
        self.points = self.compute_points(num_points, angle)

    def compute_points(self, num_points=100, angle=0.0):
        angles = np.linspace(0, 2 * np.pi, 4)[:-1] + angle
       
        xx = self.center[0] + self.radius * np.cos(angles)
        yy = self.center[1] + self.radius * np.sin(angles)

        points_per_edge = num_points // 3
        x = np.array([])
        y = np.array([])
        for x_i, y_i in zip(xx, yy):
            for j in np.linspace(x_i, xx[(np.where(xx == x_i)[0] + 1) % 3], points_per_edge, endpoint=False):
                x = np.append(x, j)
                y = np.append(y, np.interp(j, [x_i, xx[(np.where(xx == x_i)[0] + 1) % 3]], [y_i, yy[(np.where(yy == y_i)[0] + 1) % 3]]))

        return x, y
        

class GenerateShape():
    def __init__(self, shapes: dict, transform_cgf: DictConfig):
        self.shapes = shapes
        translation_range = transform_cgf.translation_range
        scale_range = transform_cgf.scale_range
        rotation_range = transform_cgf.rotation_range
        self.transformations = {
            'translation': lambda: (random.uniform(*translation_range), random.uniform(*translation_range)),
            'scale': lambda: random.uniform(*scale_range),
            'rotation': lambda: random.uniform(*rotation_range)
        }
       
    def generate_shapes(self):
        V=[]
        L=[]

        for shape_name, num_points in self.shapes.items():
            translation = self.transformations['translation']()
            scale = self.transformations['scale']()
            rotation = self.transformations['rotation']()

            if shape_name == 'circle':
                shape = Circle(translation, scale, num_points)
            elif shape_name == 'hexagon':
                shape = Hexagon(rotation, translation, scale, num_points)
            elif shape_name == 'triangle':
                shape = Triangle(rotation, translation, scale, num_points)
            else:
                raise ValueError(f"Unsupported shape: {shape_name}")

            x, y = shape.get_points()
            coords = np.stack((x, y), axis=-1)
            V.append(coords) # vertices of shapes
            L.append(self.connectivity(V)) # connectivity of the shapes

    
    def connectivity(self, V):
        L = []
        for i in range(len(V)):
            L.append((i, (i + 1) % len(V)))  # connect each vertex to the next and the last to the first
        return L
           
@hydra.main(config_path="/home/sofiasannino/projects/rl-mesh-deformation/configs", config_name="generate")
def main(cfg: DictConfig):
    shapes ={
        'name': 'circle', 'num_points': 100,
        'name': 'hexagon', 'num_points': 100,
        'name': 'triangle', 'num_points': 100
    }
    shape_generator = GenerateShape(shapes, cfg)
    shape_generator.generate_shapes()

    for V, L in zip(shape_generator.V, shape_generator.L):
        plt.figure()
        plt.scatter(V[:, 0], V[:, 1])
        for edge in L:
            plt.plot(V[edge, 0], V[edge, 1], 'k-')
        plt.title('Generated Shape')
        plt.axis('equal')
        plt.show()
        plt.savefig(f"{cfg.output_dir}/generated_shape.png")


   

   