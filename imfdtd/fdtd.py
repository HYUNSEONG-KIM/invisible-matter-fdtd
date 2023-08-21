from typing import Union, Iterable

import numpy as np

from .utils import mv_mul, curl_E, curl_H
from .object import Matter, Source
from .consts import c0, mu0, ep0

class FDTD:
    def __init__(self, dim, lattice, mode="2d"):
        assert len(dim) == len(lattice), "The given dimension and lattice number must be same."
        
        self.dim = dim # [[xi, xf], [yi, yf]]
        self.lattice_dim = lattice
        self.mode = "2d"
        
        self.matter_objs = []
        self.source_objs = []
        self.boundary = None
        
        self.simulation_params =None
        self.run_state = False
        
    def add_object(self, objs:Union[list, Matter]):
        if isinstance(objs, Matter):
            self.matter_objs.append(objs)
        elif isinstance(objs, Iterable):
            for obj in objs:
                if isinstance(obj, Matter):
                    self.matter_objs.append(obj)
        self.run_state=False
    def add_source(self, srcs:Union[list, Source]):
        if isinstance(srcs, Matter):
            self.matter_objs.append(srcs)
        elif isinstance(srcs, Iterable):
            for src in srcs:
                if isinstance(src, Matter):
                    self.matter_srcs.append(src)
        self.run_state=False
    def add_boundary(self, bd):
        self.boundary = bd
        self.run_state=False
    @staticmethod
    def generate_yee_grid(dim, lattice, mode="2d"):
        xdim = dim[0]
        ydim = dim[1]
        
        nx, ny = lattice[:2]
        
        xi, xf = xdim
        yi, yf = ydim
        # Electric field grid
        xline = np.linspace(xi, xf, nx, endpoint=True)
        yline = np.linspace(yi, yf, ny, endpoint=True)

        dx = xline[1]-xline[0]
        dy = yline[1]-yline[0]
        # Magnetic field grid
        xline_m1 = np.linspace(xi-dx/2, xf+dx/2, nx+1, endpoint=True)
        yline_m1 = np.linspace(yi-dy/2, yf+dy/2, ny+1, endpoint=True)

        Epoints = (np.meshgrid(xline, yline))
        Hxpoints = (np.meshgrid(xline, yline_m1))
        Hypoints = (np.meshgrid(xline_m1, yline))

        return Epoints, Hxpoints, Hypoints

    
    def prepare_system(self, save_path:str=None):
        # Grid preparation
        print("Grid perparation:", end=None)
        P_e, P_h = self.generate_yee_grid(self.dim, self.lattice_dim)
        print("Done")
        
        nx, ny = self.lattice_dim[:2]
        
        dx = P_e[0][0][1] - P_e[0][0][0]
        dy = P_e[0][1][1] - P_e[0][1][0]
        
        dt = min(dx, dy)/(2*c0)
        print(f"Spatial difference: {dx:.4E}, {dy:.4E}")
        print(f"Dimension: {nx}, {ny}")
        print(f"Time difference(dt):{dt:.4E} sec")

        m_shape_e = P_e.shape
        m_shape_h = P_h.shape
        
        # Fields 
        E_field = np.zeros((*m_shape_e, 3))
        D_field = np.zeros(E_field.shape)
        
        H_field = np.zeros((*m_shape_h, 3))
        
        # Relative tensor 
        I3 = np.eye(3)
        r_tensor_e = np.zeros((*m_shape_e, 3, 3))
        r_tensor_h = np.zeros((*m_shape_h, 3, 3))
        
        r_tensor_e[..., :, :] = I3
        r_tensor_h[..., :, :] = I3
        
        # Mapping Relative tensor by the objects
        print("Appyling object properties:", end=None)
        for obj in self.matter_objs:
            (row_e, col_e), g_ij_e = obj.cal_rel_tensor(P_e)
            (row_h, col_h), g_ij_h = obj.cal_rel_tensor(P_h)
            
            r_tensor_e[row_e, col_e] = g_ij_e
            r_tensor_h[row_h, col_h] = g_ij_h
        
        # Register Source update function
        
        #-----
        
        # PML 
        
        # Register initial state
        self.run_state = True
        self.simulation_params={
            "Yee": (P_e, P_h),
            "Field": (E_field, D_field, H_field),
            "Rel_T": (r_tensor_e, r_tensor_h),
            "SrcUpdate": [],
            "PML": []
        }
    def run_simulation(self, T):
        if self.run_state is False:
            return False
        
        n = int(T/self.dt)
        
        for i in range(0, n):
            pass
            
        
        