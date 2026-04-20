import random 
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra  




class Circle():
    def __init__(self, translation: tuple, scale: float, num_points=100):
        self.radius = 1.0
        self.center = (0.0, 0.0)
        dx = translation[0]
        dy = translation[1]
        self.translate(dx, dy)
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
        for i in range(6):
            x_i, y_i = xx[i], yy[i]
            x1, y1 = xx[(i + 1) % 6], yy[(i + 1) % 6]

            t = np.linspace(0, 1, points_per_edge, endpoint=False)
            x_edge = x_i + t * (x1 - x_i)
            y_edge = y_i + t * (y1 - y_i)

            x = np.append(x, x_edge)
            y = np.append(y, y_edge)

        return np.array(x), np.array(y)

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
        for i in range(3):
            x_i, y_i = xx[i], yy[i]
            x1, y1 = xx[(i + 1) % 3], yy[(i + 1) % 3]

            t = np.linspace(0, 1, points_per_edge, endpoint=False)
            x_edge = x_i + t * (x1 - x_i)
            y_edge = y_i + t * (y1 - y_i)

            x = np.append(x, x_edge)
            y = np.append(y, y_edge)

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
        return V, L

    
    def connectivity(self, V):
        L = []
        l_list = []
        for v in V:
            for i in range(len(v)):
                l_list.append((i, (i + 1) % len(v)))  # connect each vertex to the next and the last to the first
            L.append(l_list)
        return L
           
@hydra.main(version_base=None, config_path="/home/sofiasannino/projects/rl-mesh-deformation/configs", config_name="generate")
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
    i = 0 
    for v, l in zip(V, L):
        plt.figure()
        plt.scatter(v[:, 0], v[:, 1])
        for edge in l:
            plt.plot(v[edge, 0], v[edge, 1], 'k-')
        plt.title('Generated Shape')
        plt.axis('equal')
        plt.show()
        plt.savefig(f"generated_shape_{i}.png")
        i += 1


if __name__ == "__main__":    main()


   

   