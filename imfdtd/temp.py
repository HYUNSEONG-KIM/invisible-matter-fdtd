@@ -0,0 +1,378 @@
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle, Polygon, Patch, PathPatch
from matplotlib.path import Path

from typing import Literal, Tuple, Union, Callable
from _collections_abc import Iterable

import sympy as sp
#------------------------------------------------------------------
def get_func_gij(
        variables:Tuple[sp.S,...], 
        G:sp.Matrix, 
        modules:Literal["numpy"]='numpy')->Callable:
    """This function returns a fuction which calculates g_ij tensor of 2 dim matrix form.
        Usually, 'lambdify` routine is enough to use for general Numpy data type, however, 
        for matrices returning functions, there is an issue of ragged nested element result.
        This function detecting constant equations in `G` unfurl to general `float` type ndarray.
        with some transformation of index.

    Args:
        variables (Tuple[sp.S,...]): Sympy symbols array. [q1 , q2, ..., qn]
        G (sp.Matrix): Metric tensor definition in 2 dim matrix form. The elements are equations of the variables in first argument symbols.
        modules (Literal[&quot;numpy&quot;], optional): Sympy `lambdify` function argument. Defaults to 'numpy'.

    Returns:
        Callable: A function of calculating g tensor for each X, Y, Z meshgrid points.
    """

    # Detecting constant part. 
    nx, ny = sp.shape(G)
    constant_index = []
    for i in range(0, nx):
        for j in range(0, ny):
            if G[i,j].is_constant():
                constant_index.append([i, j])
    
    _g_ij = sp.lambdify(variables, G, modules=modules) # Numpy based calculation module 

    def func_g_ij(*args):
        data_shape = args[0].shape
        dim = len(data_shape)
        ones= np.ones(data_shape)
        result_arr = _g_ij(*args)

        # Fixing ragged nested elements
        for (i, j) in constant_index:
            result_arr[i, j] = result_arr[i, j]*ones # constant * ndarray

        # dtype=objet -> dtype=float
        g_r_arr =[]
        for g in result_arr:
            g_r_arr.append(np.stack(g))
        g_xyz = np.stack(g_r_arr)

        # The returned result is (M.dim, Data.dim) transpose it to (Data.dim, M.dim)
        return np.transpose(g_xyz , (*(list(range(2, dim+2))), 0, 1))

    return func_g_ij

#
def yee_lattices(X,Y, n=[20, 20]):
    xi,xf = X
    yi,yf = Y
    
    nx, ny = n
    # Electric field grid
    xline = np.linspace(xi, xf, nx, endpoint=True)
    yline = np.linspace(yi, yf, ny, endpoint=True)

    dx = xline[1]-xline[0]
    dy = yline[1]-yline[0]
    # Magnetic field grid
    xline_m = np.linspace(xi-dx/2, xf+dx/2, nx+1, endpoint=True)
    yline_m = np.linspace(yi-dy/2, yf+dy/2, ny+1, endpoint=True)

    return (np.meshgrid(xline, yline), np.meshgrid(xline_m, yline_m))



class Geometry: 
    # Internal geometry setting
    # All shape must have closed form
    # See SVG d-path tag explanation.
    # Restriction - the center must be in the polygon

    # Code
    # Circle: 0
    # Ellipse: 1
    # Rectangle: 2
    # Polygons: 3
    # Arbitary Path: 4
    shape_code= {
        0: "circle",
        1: "ellipse",
        2: "rectangle",
        3: "polygon",
        4: "general curve"
    }
    def __init__(self, shape_type:int, shape_params:Union[str, Tuple[str, ...]], naive_region:Tuple[float],angle=0, ):
        if int(shape_type) not in self.shape_code.keys():
            raise ValueError("Invaild shape_type value:{}".format(shape_type))
        self.shape_type = int(shape_type)
        self.shape_params = shape_params # Using interal coordinate
        self.path = self._generate_path()
        self.angle = angle
        self.naive_region = self.naive_region_update() # Rough occupy dimension
    def naive_region_update(self):
        x, y = self.path.interpolated(200).T
        return [[x.min(), x.max()],[y.min(), y.max()]]
    def _generate_path(self):
        pass
    def show_shape(self,gap=[1.2,1.2]):
        gx, gy = gap
        (xmin, xmax), (ymin, ymax) = self.naive_region 

        fig, ax = plt.subplots()
        patch = self._patch_generator()
        
        ax.add_patch(patch)
        ax.set_title(f"{self.shape_code[self.shape_type]}")
        ax.set_ylim([gy*ymin, gy*ymax])
        ax.set_xlim([gx*xmin, gx*xmax])
        ax.grid(alpha=0.6,)
        # square plot
        ax.set_aspect('equal', adjustable='box')
        
        return fig, ax
    def _patch_generator(self):
        if self.shape_type==0:
            a= b = 2*self.shape_params
            patch = Ellipse(xy=[0,0], width = a, height=b, angle=self.angle, facecolor='none', edgecolor='k')
        elif self.shape_type== 1:
            a, b= self.shape_params
            patch = Ellipse(xy=[0,0], width = 2*a, height=2*b, angle=self.angle, facecolor='none', edgecolor='k')
        elif self.shape_type== 2:
            w, h = self.shape_params
            x0 = -w/2
            x1 = -x0
            y0 = -h/2
            y1 = -y0

            vertices = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [0,0]]
            codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
            path = Path(codes=codes, vertices=vertices)
            patch = PathPatch(path, facecolor='none', edgecolor='k')
        elif self.shape_type== 3:
            N = len(self.shape_params)
            codes = [Path.MOVETO] + [Path.LINETO]*(N-2)  + [Path.CLOSEPOLY]
            vertices = self.shape_params
            path = Path(codes=codes, vertices=vertices)
            patch = PathPatch(path, facecolor='none', edgecolor='k')
        elif self.shape_type == 4:
            pass
        return patch
    @classmethod
    def circle(cls, radi):
        radi = math.fabs(float(radi))
        return cls(0, radi, [[-radi, radi], [-radi, radi]])
    @classmethod
    def ellipse(cls, a, b):
        a = math.fabs(float(a))
        b = math.fabs(float(b))
        return cls(1, [a,b], [[-a, a],[-b, b]])
    @classmethod
    def rectangle(cls, width, height):
        w = math.fabs(float(width))
        h = math.fabs(float(height))
        return cls(2, [w, h], [[-w/2, w/2],[-h/2, h/2]])
    @classmethod
    def from_polygons(cls, points):
        p_list = []
        xmin = xmax = ymin = ymax =0
        if isinstance(points, Iterable) and not isinstance(points, str):
            for p in points:
                if isinstance(p, str):
                    x, y = p.strip().split(" ")
                else:
                    x, y = p
                x = float(x)
                y = float(y)
                xmin = x if x < xmin else xmin
                xmax = x if x > xmax else xmax
                ymin = y if y < ymin else ymin
                ymax = y if y > ymax else ymax
                p_list.append([x,y])
        else: # string
            ps = points.split(",")
            for p in ps:
                x, y = p.strip().split(" ")
                x = float(x)
                y = float(y)
                xmin = x if x < xmin else xmin
                xmax = x if x > xmax else xmax
                ymin = y if y < ymin else ymin
                ymax = y if y > ymax else ymax
                p_list.append([x,y])
        return cls(3, points, [[xmin, xmax], [ymin, ymax]])
    @classmethod
    def arb_path(cls, svg_path, ):
        pass
    #----------------------------
    def determining_points(self, ex, ey, dx, dy):
        # ex, ey: location vector of graphic object center from its nearest gird.
        # dx, dy: units of grid in x and y direction.
        pass

class Object:
    def __init__(self, position, geometry, **kwargs):
        self.position = position
        self.geomerty = geometry 
        self.kwargs = kwargs
    # Flood fill algorithm
    def _bsp_tree(self): 
        # BSP tree algorithm
        # to determining the lattice points
        pass
    def set_position(self, position):
        self.position = position
    def set_geometry(self, geometry):
        self.geometry =geometry

    def get_internal_indices(self, grid):
        # convert global lattice location to internal lattice
        indices = None
        return indices 
    def obj_constraint(self, center, *args):
        def constraint(*args):
            pass
        return constraint 


class ObjectGroup:
    def __init__(self,):
        pass
    def add(self, objs):
        pass
    def _determine_overlap(self):
        pass
    def move_to(self, ):
        pass
    def set_to_origin(self,):
        pass

    def obj_group_constraint(self, center, *args):
        pass

class Source:
    def __init__(self):
        pass

# Line source
class LineSource(Source, Object):
    def __init__(self):
        pass
# point source
class PointSource(Source):
    def __init__(self):
        pass
# Gaussian beam source
class BeamSource(Source):
    def __init__(self):
        pass
class GaussianBeamSource(BeamSource):
    pass

# Material Object

class Matter(Object):
    def __init__(self):
        
        self.geometry = Geometry()
        self.internal_holes = []

        self.rel_electirc_permitity = np.eye(3,3)
        self.rel_magnetic_permeability = np.eye(3,3)

        self.func_gij = lambda *args: np.eye(3,3)
        self.func_gij_default = True
    @classmethod
    def from_shape(cls, shape, *optics):
        return cls(shape, )
    
    def set_origin(self, x, y):
        self.origin = [x, y]
    def set_internal_hole(self, geometry):
        pass
    def set_transform_coordinate(self, variables, G):
        self.func_gij = get_func_gij(variables, G)
        self.func_gij_default = False
        return True
    def reset_transform_coordinate(self):
        self.func_gji =  lambda *args: np.eye(3,3)
        self.func_gij_default = True
        return True
    def get_lattice_indices(self, grid):
        # Mesh to points conversion
        #points = 
        p_index = self.geomerty.contains_points(points)
        index = np.where(p_index == True)[0]
        inside = points[index]

        points_xy = inside.T

        # Boundaries, Interanl, 
        pass


class FDTDData:
    def __init__(self, mode = "animate", *args, **kwargs):
        pass
    @classmethod
    def load_from_file(cls, *args):
        pass
    def to_video(self, ):
        pass
    def to_record(self, *args):
        pass

class FDTDSolver:
    """This class is a FDTD simulator.
        Cosisting the geometric and electric proeprties of simulation.

    """
    def __init__(self, dimension:Literal["2d", "3d"]="2d"):
        pass
    
    @classmethod
    def get_solver_with_intial(cls, *args, **kwargs):
        tem_solver = cls()
        return tem_solver
    
    # Setting source and objects
    def add_objects(self, objects:Tuple[Object, ...]):
        pass
    def add_sources(self, sources:Tuple[Source, ...]):
        pass

    def update_object(self, obj_index, *args, **kwargs):
        pass

    # Graphical indication
    def plot_parameters_on_space(self,
                      plot_profile,
                      plot_list=[],
                      graphical_setting=[],
                      *args, **kwargs
                      ):
        #return fig and axes
        pass

    # Boundary condition setting
    def boundary_condition(self,):
        pass

    # Run
    def run(self, 
            mode:Literal["animation", "data"]="animatate", 
            format="gif", **file_args):
        
        # Region segmentation and split the grid points to regions
        # inside of object 1, ... et cetera
        #   This method using ray intersection method

        # Mapping region to each properties tensors, mu, rho ...
        # inside objects return the tensor (x, y, z) ->M_3x3(c)

        # Iteratation
        # - There are two update patterns
        #   1. Generated from internal source and external field
        #   2. Time-evolution of generated field value in space.
        # 2. Each point have its own physical constants for update next iter step value.
        
        
        return FDTDData()
